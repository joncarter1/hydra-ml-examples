# Configurability with Hydra <img src="https://hydra.cc/img/logo.svg" alt="Hydra logo" style="height: 100px; width:100px; padding: 0px 0px 0px 10px; margin:-10px 0px -15px 0px;"/>

This repository contains minimal working examples of machine learning experiment configuration using [Hydra](https://hydra.cc/).

These examples are intended to highlight some of the properties of Hydra which make it incredibly useful for machine learning research, including:
1. **Hierarchical composition**, using the defaults list.
https://github.com/joncarter1/hydra-ml-examples/blob/bc89f4c1e4777d3cbf98b11ffdd8592b74f0ea30/0_sklearn/config/main.yaml#L1-L4
4. **Variable interpolation**[^1], which ensures a single source of truth for inter-linked configuration options.
https://github.com/joncarter1/hydra-ml-examples/blob/bc89f4c1e4777d3cbf98b11ffdd8592b74f0ea30/0_sklearn/config/model/mlp.yaml#L12
https://github.com/joncarter1/hydra-ml-examples/blob/bc89f4c1e4777d3cbf98b11ffdd8592b74f0ea30/0_sklearn/config/dataset/blobs.yaml#L6
5. **Object instantiation**, which removes the need for boilerplate code to propagate configuration to backing classes/functions.
https://github.com/joncarter1/hydra-ml-examples/blob/bc89f4c1e4777d3cbf98b11ffdd8592b74f0ea30/0_sklearn/script.py#L23-L24

## Getting started

Requires [uv](https://docs.astral.sh/uv/). The following commands run the basic example: a sweep over all combinations of model and dataset for a toy problem using scikit-learn.

```
uv run 0_sklearn/script.py --multirun dataset=blobs,circles,moons model=randomforest,mlp,svm
```

## Advanced usage
Overriding parameters of the underlying model or dataset:
```
uv run 0_sklearn/script.py model=mlp model.activation=tanh
uv run 0_sklearn/script.py model=randomforest model.n_estimators=400
uv run 0_sklearn/script.py --multirun dataset=blobs,circles,moons dataset.n_samples=100,500,1000
```
**Any parameter supported by the backing class can be modified from the command line.**

For parameters which aren't explicitly specified in the configuration file, this can be achieved using append syntax[^2]:
```
uv run 0_sklearn/script.py --multirun model=mlp +model.momentum=0.5,0.7,0.9
```
In this example the backing class is an MLP from scikit-learn ([docs](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)). 

This mechanism is even more convenient with complex neural network definitions e.g. using Pytorch.


## Hyperparameter optimization with Optuna

The `2_optuna/` example replaces Hydra's default grid search with intelligent Bayesian optimization via the [Optuna sweeper plugin](https://hydra.cc/docs/plugins/optuna_sweeper/):

```
uv run 2_optuna/script.py --multirun
```

See [`2_optuna/README.md`](2_optuna/README.md) for more details, including cross-model hyper-patameter optimisation with conditional per-model hyperparameters.

[^1]: Using [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/)
[^2]: [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/#modifying-the-config-object).
