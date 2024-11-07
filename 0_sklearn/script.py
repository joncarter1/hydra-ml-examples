import hydra
import logging
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from typing import Union

SKLearnClassifier = Union[RandomForestClassifier, MLPClassifier, SVC]

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    logger.info("Starting experiment...")
    # Demonstrate pretty color logging with hydra_colorlog
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    # Instantiate the model. Type hints on instantiations can improve readability.
    model: SKLearnClassifier = hydra.utils.instantiate(cfg.model)
    logger.info("Instantiated model: %s", model.__class__.__name__)
    # Instantiate the dataset.
    X, y = hydra.utils.call(cfg.dataset)
    # Split the dataset into train/test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed
    )
    # Train the model.
    logger.info("Training the model...")
    model.fit(X_train, y_train)
    # Evaluate the model.
    logger.info(f"Test score: {model.score(X_test, y_test)}")


if __name__ == "__main__":
    main()
