# Configurability with Hydra <img src="https://hydra.cc/img/logo.svg" alt="Hydra logo" style="height: 100px; width:100px; padding: 0px 0px 0px 10px; margin:-10px 0px -15px 0px;"/>

This repository contains a minimal working example of machine learning experiment configuration using [Hydra](https://hydra.cc/).

The example highlights three properties of Hydra which make it incredibly useful for machine learning research:
1. Hierarchical composition, using the defaults list.
https://github.com/joncarter1/hydra-examples/blob/c8cf053e68a84f4b4cfe0318b93087a77094f0a0/basic-example/config/main.yaml#L1-L4
2. Variable interpolation[^1], which ensures a single source of truth for inter-linked configuration options.
https://github.com/joncarter1/hydra-examples/blob/c8cf053e68a84f4b4cfe0318b93087a77094f0a0/basic-example/config/model/mlp.yaml#L12
https://github.com/joncarter1/hydra-examples/blob/c8cf053e68a84f4b4cfe0318b93087a77094f0a0/basic-example/config/dataset/blobs.yaml#L6
3. Object instantiation, which removes the need for boilerplate code to propagate configuration to backing classes/functions.
https://github.com/joncarter1/hydra-examples/blob/c8cf053e68a84f4b4cfe0318b93087a77094f0a0/basic-example/script.py#L16-L19

## Getting started
The following commands can be used to perform a sweep over all combinations of model and dataset.

### With Conda

```
conda env create --file envs/environment.yaml
conda activate hydra-example
python basic-example/script.py --multirun dataset=blobs,circles,moons model=randomforest,mlp,svm
```

### With Docker

```
docker build --file env/Dockerfile --tag hydra-example .
docker run hydra-example --multirun dataset=blobs,circles,moons model=randomforest,mlp,svm
```

## Advanced usage
Overriding parameters of the underlying model or dataset:
```
python basic-example/script.py model=mlp model.activation=tanh
python basic-example/script.py model=randomforest model.n_estimators=400
python basic-example/script.py --multirun dataset=blobs,circles,moons dataset.n_samples=100,500,1000
```
Any parameter supported by the backing class can be modified from the command line, even if they aren't explicitly specified in the configuration file, using append syntax[^2]:
```
python basic-example/script.py --multirun model=mlp +model.momentum=0.5,0.7,0.9
```
In this example the backing class is an MLP from scikit-learn ([docs](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)). 

This mechanism is even more convenient with complex neural network definitions e.g. using Pytorch.


[^1]: Using [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/)
[^2]: [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/#modifying-the-config-object).
