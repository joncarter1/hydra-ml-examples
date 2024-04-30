import hydra
import logging
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision as tv

from models import get_device

logger = logging.getLogger(__name__)


def train(model, device, train_loader, optimizer, epoch, log_interval: int = 10):
    """Training epoch function."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    """Test epoch function."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    pct = 100.0 * correct / len(test_loader.dataset)
    logger.info(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({pct:.0f}%)"
    )


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    # Set accelerator.
    device = get_device() if cfg.device == "auto" else cfg.device
    # Instantiate the model. Type hints on instantiations can improve readability.
    model: nn.Module = hydra.utils.instantiate(cfg.model).to(device)
    logger.info(f"Running with model:\n{model}")
    optimizer: optim.Optimizer = hydra.utils.instantiate(
        cfg.optimizer, params=model.parameters()
    )
    # Apply dataset-specific normalisation to images.
    transform = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                cfg.dataset.channel_means, cfg.dataset.channel_stds
            ),
        ]
    )
    train_data: tv.datasets.VisionDataset = hydra.utils.instantiate(
        cfg.dataset.train, transform=transform
    )
    train_loader = DataLoader(
        train_data,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
    )
    logger.info("Starting training...")
    for epoch in range(cfg.train.epochs):
        train(model, device, train_loader, optimizer, epoch)
    logger.info("Evaluating...")
    test_data: tv.datasets.VisionDataset = hydra.utils.instantiate(
        cfg.dataset.test, transform=transform
    )
    test_loader = DataLoader(
        test_data,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
    )
    test(model, device, test_loader)


if __name__ == "__main__":
    main()
