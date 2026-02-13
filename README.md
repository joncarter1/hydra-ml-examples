# Configurability with Hydra <img src="https://hydra.cc/img/logo.svg" alt="Hydra logo" style="height: 100px; width:100px; padding: 0px 0px 0px 10px; margin:-10px 0px -15px 0px;"/>

Minimal working examples of ML experiment configuration using [Hydra](https://hydra.cc/).

## Contents

| Directory | Description | Quick start |
|-----------|-------------|-------------|
| [`0_sklearn/`](0_sklearn/) | Scikit-learn classifiers (RandomForest, MLP, SVM) on toy datasets | `uv run 0_sklearn/script.py` |
| [`1_pytorch/`](1_pytorch/) | PyTorch models (MLP, CNN, ViT) on image datasets (MNIST, SVHN) | `uv run 1_pytorch/script.py` |
| [`2_optuna/`](2_optuna/) | Bayesian HPO with Optuna sweeper plugin | `uv run 2_optuna/script.py --multirun` |

## Key Hydra concepts

### Hierarchical composition

The `defaults` list in `main.yaml` composes a final config from modular YAML files in subdirectories:

```yaml
# 0_sklearn/config/main.yaml
defaults:
  - model: randomforest
  - dataset: blobs
  - _self_

seed: 42
test_size: 0.3
```

Selecting `model=mlp` at the command line swaps in `config/model/mlp.yaml` instead of `randomforest.yaml` — no code changes needed.

### Variable interpolation

[OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/) resolvers let configs reference each other, ensuring a single source of truth:

```yaml
# 0_sklearn/config/model/randomforest.yaml
random_state: ${seed}           # resolves 'seed' from main.yaml

# 1_pytorch/config/model/mlp.yaml
im_size: ${dataset.im_size}     # cross-config reference to dataset properties
num_classes: ${dataset.num_classes}
```

### Object instantiation

Configs can specify a `_target_` to directly instantiate classes and functions from config, removing the need for boilerplate code to propagate configuration to backing classes/functions:
```yaml
# 0_sklearn/config/model/randomforest.yaml
_target_: sklearn.ensemble.RandomForestClassifier
n_estimators: 100
random_state: ${seed}
```

```python
# 0_sklearn/script.py
model = hydra.utils.instantiate(cfg.model)   # returns a RandomForestClassifier
X, y = hydra.utils.call(cfg.dataset)         # calls sklearn.datasets.make_blobs(...)
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/). Run from the repo root:

```bash
uv run 0_sklearn/script.py
```

Logs go to `/tmp/hydra/logs/`.

## Scikit-learn example

Override config groups or individual parameters from the command line:

```bash
# Select a different model and dataset
uv run 0_sklearn/script.py model=mlp dataset=moons

# Override a specific parameter
uv run 0_sklearn/script.py model=randomforest model.n_estimators=400

# Sweep over all combinations
uv run 0_sklearn/script.py --multirun model=randomforest,mlp,svm dataset=blobs,circles,moons
```

**Any parameter supported by the backing class can be modified from the command line.** For parameters not explicitly listed in the YAML config, use append syntax[^2]:

```bash
uv run 0_sklearn/script.py --multirun model=mlp +model.momentum=0.5,0.7,0.9
```

## PyTorch example

The PyTorch example adds optimizer config groups and cross-config interpolation (model configs reference dataset properties like `im_size` and `num_classes`):

```bash
# Default: MLP on MNIST with Adam
uv run 1_pytorch/script.py

# Override model, optimizer, and learning rate
uv run 1_pytorch/script.py model=cnn optimizer=sgd optimizer.lr=1e-4

# Sweep over models, optimizers, and learning rates
uv run 1_pytorch/script.py --multirun model=mlp,cnn,vit optimizer=adam,sgd optimizer.lr=1e-5,1e-4,1e-3
```

See [`1_pytorch/README.md`](1_pytorch/README.md) for model details.

## Optuna HPO

The `2_optuna/` example replaces Hydra's default grid search with intelligent Bayesian optimization via the [Optuna sweeper plugin](https://hydra.cc/docs/plugins/optuna_sweeper/):

```bash
# Default: tune MLP hyperparameters
uv run 2_optuna/script.py --multirun

# Cross-model HPO with conditional per-model search spaces
uv run 2_optuna/script.py --multirun hpo=cross_model

# Evaluate: reload best params from a previous sweep
uv run 2_optuna/script.py evaluate=true best_params=/tmp/hydra/logs/script/2025-01-01/12-00-00/optimization_results.yaml
```

See [`2_optuna/README.md`](2_optuna/README.md) for more details.

[^2]: [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/#modifying-the-config-object).
