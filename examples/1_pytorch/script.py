import hydra
import logging
from omegaconf import DictConfig

import torchvision

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model)
    print("HERE")
    print(model)
    return
    # Instantiate the model. Type hints on instantiations can improve readability.
    mnist_train = torchvision.datasets.MNIST(
        root="/tmp/mnist/train", train=True, download=True
    )
    mnist_test = torchvision.datasets.MNIST(
        root="/tmp/mnist/test", train=True, download=True
    )
    return


if __name__ == "__main__":
    main()
