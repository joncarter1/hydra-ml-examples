# Configurability with Hydra <img src="https://hydra.cc/img/logo.svg" alt="Hydra logo" style="height: 100px; width:100px; padding: 0px 0px 0px 10px; margin:-10px 0px -15px 0px;"/>

This repository contains minimal working examples of machine learning experiment configuration using [Hydra](https://hydra.cc/).

These examples are intended to highlight some of the properties of Hydra which make it incredibly useful for machine learning research, including:
1. **Hierarchical composition**, using the defaults list.
https://github.com/joncarter1/hydra-ml-examples/blob/ff84ff5f72e6f18facb666e179eb524f5ba2f626/examples/0_basic/config/main.yaml#L1-L4
4. **Variable interpolation**[^1], which ensures a single source of truth for inter-linked configuration options.
https://github.com/joncarter1/hydra-ml-examples/blob/ff84ff5f72e6f18facb666e179eb524f5ba2f626/examples/0_basic/config/model/mlp.yaml#L12
https://github.com/joncarter1/hydra-ml-examples/blob/ff84ff5f72e6f18facb666e179eb524f5ba2f626/examples/0_basic/config/dataset/blobs.yaml#L6
5. **Object instantiation**, which removes the need for boilerplate code to propagate configuration to backing classes/functions.
https://github.com/joncarter1/hydra-ml-examples/blob/ff84ff5f72e6f18facb666e179eb524f5ba2f626/examples/0_basic/script.py#L16-L19

## Getting started
The following commands can be used to perform run the basic example: a sweep over all combinations of model and dataset for a toy problem using scikit-learn.

### With Conda

```
conda env create --file 0_sklearn/env/environment.yaml
conda activate hydra-example-0
python 0_sklearn/script.py --multirun dataset=blobs,circles,moons model=randomforest,mlp,svm
```

### With Docker

```
docker build --build-arg EXAMPLE="0_sklearn" --tag hydra-example-0 .
docker run hydra-example-0 --multirun dataset=blobs,circles,moons model=randomforest,mlp,svm
```

## Advanced usage
Overriding parameters of the underlying model or dataset:
```
python 0_sklearn/script.py model=mlp model.activation=tanh
python 0_sklearn/script.py model=randomforest model.n_estimators=400
python 0_sklearn/script.py --multirun dataset=blobs,circles,moons dataset.n_samples=100,500,1000
```
**Any parameter supported by the backing class can be modified from the command line.**

For parameters which aren't explicitly specified in the configuration file, this can be achieved using append syntax[^2]:
```
python 0_sklearn/script.py --multirun model=mlp +model.momentum=0.5,0.7,0.9
```
In this example the backing class is an MLP from scikit-learn ([docs](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)). 

This mechanism is even more convenient with complex neural network definitions e.g. using Pytorch.


[^1]: Using [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/)
[^2]: [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/#modifying-the-config-object).
