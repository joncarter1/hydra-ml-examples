# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "hydra-core~=1.3",
#     "hydra-colorlog~=1.2",
#     "hydra-optuna-sweeper~=1.2",
#     "pyyaml",
#     "scikit-learn~=1.2",
# ]
# ///

import warnings
from pathlib import Path

import hydra
import logging
import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Apple Accelerate BLAS emits spurious overflow/divide-by-zero warnings during matmul.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main", version_base=None)
def objective(cfg: DictConfig) -> float:
    if cfg.evaluate and HydraConfig.get().mode == RunMode.MULTIRUN:
        raise RuntimeError("Cannot use evaluate=true with --multirun (would test on held-out set repeatedly)")

    if cfg.best_params is not None:
        results = yaml.safe_load(Path(cfg.best_params).read_text())
        overrides = OmegaConf.from_dotlist(
            [f"{k}={v}" for k, v in results["best_params"].items()]
        )
        cfg = OmegaConf.merge(cfg, overrides)
        logger.info(f"Loaded best params from {cfg.best_params}: {results['best_params']}")

    model = make_pipeline(StandardScaler(), hydra.utils.instantiate(cfg.model))
    X, y = hydra.utils.call(cfg.dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed
    )

    if cfg.evaluate:
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        logger.info(f"Test accuracy: {accuracy:.4f}")
    else:
        scores = cross_val_score(
            model, X_train, y_train, cv=cfg.cv_folds, scoring="accuracy"
        )
        accuracy = scores.mean()
        logger.info(f"CV accuracy: {accuracy:.4f} (+/- {scores.std():.4f})")

    return accuracy


if __name__ == "__main__":
    objective()
