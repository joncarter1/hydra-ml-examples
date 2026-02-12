from omegaconf import DictConfig

# Map config group names to their model-specific search spaces.
_MODEL_PARAMS = {
    "randomforest": lambda trial: {
        "model.n_estimators": trial.suggest_int("model.n_estimators", 50, 500),
        "model.max_depth": trial.suggest_int("model.max_depth", 2, 20),
        "model.min_samples_split": trial.suggest_int("model.min_samples_split", 2, 10),
    },
    "mlp": lambda trial: {
        "model.learning_rate_init": trial.suggest_float(
            "model.learning_rate_init", 1e-4, 1e-1, log=True
        ),
        "model.alpha": trial.suggest_float("model.alpha", 1e-5, 1e-1, log=True),
        "model.activation": trial.suggest_categorical(
            "model.activation", ["relu", "tanh", "logistic"]
        ),
    },
    "svm": lambda trial: {
        "model.C": trial.suggest_float("model.C", 0.01, 100, log=True),
        "model.kernel": trial.suggest_categorical(
            "model.kernel", ["rbf", "linear", "poly"]
        ),
    },
}


def configure(cfg: DictConfig, trial) -> None:
    """Suggest model-specific hyperparameters based on the selected model.

    Uses trial.params["model"] (set by the declarative ``choice()`` param)
    to determine which model was chosen for this trial, then suggests
    the appropriate hyperparameters.
    """
    model_name = trial.params.get("model")
    if model_name and model_name in _MODEL_PARAMS:
        _MODEL_PARAMS[model_name](trial)
