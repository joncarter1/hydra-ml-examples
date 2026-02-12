# Hydra + Optuna: Hyperparameter Optimization

This example demonstrates how to use the [Optuna sweeper plugin](https://hydra.cc/docs/plugins/optuna_sweeper/) for Hydra to replace grid search with intelligent hyperparameter optimization (Bayesian optimization via TPE).

It uses scikit-learn classifiers on toy datasets to keep things fast and focused on the Optuna/Hydra integration.

## Usage

### Step 1: HPO — find best hyperparameters

Runs 20 Optuna trials tuning `learning_rate_init`, `alpha`, and `activation` for an MLP on the moons dataset. Each trial is scored by **k-fold cross-validation on the train set only** — the test set is never seen during HPO.

```bash
uv run 2_optuna/script.py --multirun
```

Override number of trials or CV folds:

```bash
uv run 2_optuna/script.py --multirun hydra.sweeper.n_trials=50 cv_folds=10
```

Change the dataset:

```bash
uv run 2_optuna/script.py --multirun dataset=circles
```

### Step 2: Evaluate — score best params on the held-out test set

After HPO, point `best_params` at the `optimization_results.yaml` from the sweep to automatically load the best hyperparameters and evaluate on the held-out test set:

```bash
uv run 2_optuna/script.py evaluate=true best_params=/tmp/hydra/logs/script/2025-01-01/12-00-00/optimization_results.yaml
```

For cross-model HPO results, the best model type is a config group override that must be passed on the CLI (the script will warn you):

```bash
uv run 2_optuna/script.py evaluate=true model=randomforest best_params=/tmp/hydra/logs/script/2025-01-01/12-00-00/optimization_results.yaml
```

You can also pass all params manually:

```bash
uv run 2_optuna/script.py evaluate=true model.learning_rate_init=0.005 model.alpha=0.0006 model.activation=tanh
```

### Advanced: Cross-model HPO with custom search space

Switch to the `cross_model` HPO config group to sweep over model types with conditional per-model hyperparameters. This uses a `custom_search_space` callback (`search_space.py`) that suggests different hyperparams depending on which model Optuna selects:

```bash
uv run 2_optuna/script.py --multirun hpo=cross_model
```

This tells Optuna to:
1. Choose a model type each trial (`randomforest`, `mlp`, or `svm`) via `choice()` in the sweeper params
2. Call `search_space.configure()` to conditionally suggest model-specific hyperparams (e.g. `n_estimators` for RF, `learning_rate_init` for MLP, `C` for SVM)

The `hpo/` config group (under `config/hpo/`) uses `@package hydra.sweeper` to inject sweeper params — `mlp.yaml` defines the declarative search space, `cross_model.yaml` wires up the custom search space function.

## Search space types

Defined in `config/hpo/mlp.yaml` (the default `hpo` config group option):

| Syntax | Example | Description |
|--------|---------|-------------|
| `interval(low, high)` | `interval(1e-4, 1e-1)` | Uniform float range |
| `tag(log, interval(...))` | `tag(log, interval(1e-4, 1e-1))` | Log-uniform sampling |
| `choice(a, b, c)` | `choice(relu, tanh, logistic)` | Categorical choices |

## Output

Logs and `optimization_results.yaml` are written to `/tmp/hydra/logs/`. The best trial's parameters and score are printed at the end of the sweep.
