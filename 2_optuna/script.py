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


def load_best_params(cfg: DictConfig) -> DictConfig:
    """Load best params from an optimization_results.yaml into the config.

    Config group overrides (undotted keys like "model") control Hydra's config
    composition, which has already run by this point. They must be passed on
    the CLI so Hydra loads the right config group — raises if missing.
    """
    results = yaml.safe_load(Path(cfg.best_params).read_text())
    best = results["best_params"]
    group_overrides = {k: v for k, v in best.items() if "." not in k}
    param_overrides = {k: v for k, v in best.items() if "." in k}
    if group_overrides:
        cli_overrides = {
            o.split("=")[0] for o in HydraConfig.get().overrides.task
        }
        missing = {k: v for k, v in group_overrides.items() if k not in cli_overrides}
        if missing:
            flags = " ".join(f"{k}={v}" for k, v in missing.items())
            raise RuntimeError(
                f"best_params contains config group overrides that must be "
                f"passed on the CLI: {flags}\n"
                f"Re-run with: {flags} "
                f"best_params={cfg.best_params}"
            )
    overrides = OmegaConf.from_dotlist(
        [f"{k}={v}" for k, v in param_overrides.items()]
    )
    cfg = OmegaConf.merge(cfg, overrides)
    logger.info("Loaded params from %s: %s", cfg.best_params, param_overrides)
    return cfg


@hydra.main(config_path="config", config_name="main", version_base=None)
def objective(cfg: DictConfig) -> float:
    if cfg.evaluate and HydraConfig.get().mode == RunMode.MULTIRUN:
        raise RuntimeError("Cannot use evaluate=true with --multirun (would test on held-out set repeatedly)")

    hc = HydraConfig.get()
    if hc.mode == RunMode.MULTIRUN and hc.job.num == 0:
        logger.info(f"Results will be saved to {hc.sweep.dir}/optimization_results.yaml")

    if cfg.best_params is not None:
        cfg = load_best_params(cfg)

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
