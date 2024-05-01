import logging
import torch
from torch import nn

# https://github.com/arogozhnikov/einops
from einops.layers.torch import Rearrange


logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """Vanilla multi-layer perceptron (MLP) for images.

    Consists of an input dimension, a variable number of hidden dimensions, followed by an output dimension.
    """

    def __init__(
        self,
        im_size: tuple[int, int],
        in_channels: int,
        hidden_dims: list[int],
        num_classes: int,
        activation: str = "relu",
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]
        activation = get_activation(activation)
        input_dim = im_size[0] * im_size[1] * in_channels
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation]

        for ind, hidden_dim_out in enumerate(hidden_dims[1:]):
            hidden_dim_in = hidden_dims[ind]
            layers += [
                nn.Linear(hidden_dim_in, hidden_dim_out),
                activation,
            ]
        layers += [nn.Linear(hidden_dims[-1], num_classes)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neural network forward pass.

        Args:
            x: Batch tensor of images.
            Tensor [B, C, H, W]
        Returns:
            Batch tensor of outputs.
            Tensor [B, D_out]
        """
        x = x.flatten(start_dim=1)
        return self.model(x)


class FCNN(nn.Module):
    """Vanilla fully convolutional network for images.

    Applies a series of convolution layers consisting of:
    Convolution -> Normalization -> Activation.

    Each layer should progressively downsample the input.
    The final layer performs a temporal average so the
    output dimension is fixed regardless of input resolution.
    """

    def __init__(
        self,
        in_channels: int,
        channel_params: list[tuple[int, int, int]],
        num_classes: int,
        activation: str = "relu",
    ):
        super().__init__()
        activation = get_activation(activation)
        layers = []
        for kernel_size, out_channels, stride in channel_params:
            if not kernel_size % 2:
                raise ValueError(
                    f"Only odd-size kernels are supported, got {kernel_size=}"
                )
            padding = 1 + kernel_size // 2  # Pad to retain input shape.
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,  # bias=True since batch norm makes it redundant.
            )
            norm = nn.BatchNorm2d(out_channels)
            layers += [
                conv,
                norm,
                activation,
            ]
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.avg = nn.AdaptiveAvgPool2d(1)  # Avg. over spatial dims.
        self.linear = nn.Linear(out_channels, num_classes)

    def forward(self, x_BHW: torch.Tensor) -> torch.Tensor:
        """Neural network forward pass.

        Args:
            x: Batch tensor of images.
            Tensor [B, C, H, W]
        Returns:
            Batch tensor of outputs.
            Tensor [B, D_out]
        """
        x_BCXY = self.cnn(x_BHW)
        x_BC = self.avg(x_BCXY).squeeze()
        return self.linear(x_BC)


class ViT(nn.Module):
    """Vanilla vision transformer for images.

    Inspired by: https://github.com/lucidrains/vit-pytorch

    Splits an image into patches and alternates between independent,
    non-linear transformations of patches and patch-wise attention.
    The final set of patch features are averaged to get the overall
    feature vector for the image.
    """

    def __init__(
        self,
        im_size: tuple[int, int],
        in_channels: int,
        patch_size: tuple[int, int],
        feature_dim: int,
        num_layers: int,
        dim_ff: int,
        num_classes: int,
        nhead: int = 8,
        dropout: float = 0.1,
        norm_first: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        activation = get_activation(activation)
        patch_dim = in_channels * patch_size[0] * patch_size[1]
        num_patches = (im_size[0] // patch_size[0]) * (im_size[1] // patch_size[1])
        # Operation that reshapes the input images into patches.
        # i.e. from B x C x H x W --> B x (N_p) X (P_d)
        # where N_p is the number of patches and P_d is the patch feature dimension.
        # This re-shapes the input images into patches, then projects the patches
        # to the desired feature dimension with a linear layer.
        # Normalization layers are added in between for good measure.
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size[0],
                p2=patch_size[1],
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        # Learnable positional embedding for each of the patches.
        # Without this, transformer 'sees' a set of unordered patches,
        # though prior work has shown a transformer can still
        # learn positional information anyway (see NoPE paper)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, feature_dim))
        # The CLS token is an additional, learnable 'patch' that the transformer layers
        # can use to store information. Giving ViT models additional tokens i.e. 'registers'
        # has been shown to improve interpretability of the resulting attention maps.
        # See "Vision Transformers Need Registers": https://arxiv.org/pdf/2309.16588.pdf
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            dim_feedforward=dim_ff,
            activation=activation,
            nhead=nhead,
            batch_first=True,
            dropout=dropout,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, x_BCHW: torch.Tensor) -> torch.Tensor:
        """Neural network forward pass.

        Args:
            x: Batch tensor of images.
            Tensor [B, C, H, W]
        Returns:
            Batch tensor of outputs.
            Tensor [B, D_out]
        """
        B = x_BCHW.size(0)
        # Transform images into patches.
        x_BPE = self.to_patch_embedding(x_BCHW)
        # Add CLS token for each element in the batch: F = E + 1
        x_BPF = torch.cat((x_BPE, self.cls_token.repeat(B, 1, 1)), dim=1)
        # Add positional information, so the transformer can learn the ordering of input elements.
        x_BPF += self.pos_embedding
        # Apply the transformer encoder model to the patch features.
        x_BPF = self.transformer_encoder(x_BPF)
        # Average over patch features to get overall feature vector for the image.
        # Using the CLS token is an alternative method.
        x_BF = x_BPF.mean(dim=1)
        # Transform to logits for classes.
        return self.linear(x_BF)


def get_activation(name: str, **kwargs):
    """Return an activation function from its name."""
    if name == "relu":
        return nn.ReLU(**kwargs)
    elif name == "leaky":
        return nn.LeakyReLU(**kwargs)
    elif name == "gelu":
        return nn.GELU(**kwargs)
    else:
        raise ValueError(f"{name=} is unsupported.")


def get_device():
    """Auto-discover accelerators."""
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    elif (xm := get_xm()) is not None:
        return xm.xla_device()
    else:
        device = torch.device("cpu")
    logger.debug(f"Using {device=}.")
    return device


def get_xm():
    """Get PyTorch XLA module if available."""
    try:
        import torch_xla.core.xla_model as xm

        return xm
    except ImportError:
        return None
