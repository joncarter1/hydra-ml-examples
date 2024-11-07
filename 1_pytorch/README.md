# 1. Pytorch + Hydra 
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/PyTorch_logo_icon.svg/640px-PyTorch_logo_icon.svg.png" alt="Pytorch logo" style="height: 40px; width:auto; padding: 0px 0px 0px 0px; margin:-0px 10px -10px 10px;"/><img src="https://hydra.cc/img/logo.svg" alt="Hydra logo" style="height: 40px; width:auto; padding: 0px 10px 0px 10px; margin:-0px 0px -10px 0px;"/>

This example shows how Hydra can be used to simplify the configuration of ML research using PyTorch.

Here Hydra is particularly helpful in reducing 'boilerplate factory code' i.e. the logic that turns your runtime configuration options into the Python objects within your application.

This code often looks something like:
```python
if args.model == "mlp":
    model = MLP(hidden_dims=args.hidden_dims, ...)
elif args.model == "cnn":
    model = ...
```
As the number of configuration options you may wish to change grows (model, optimizer, dataset, accelerator etc.), this type of boilerplate code can often end up swamping your codebase, and can become increasingly difficult to maintain as you increase complexity.

## Example Commands
From the command line, we can override high-level options such as:
1. The dataset being used e.g. MNIST vs SVHN.
2. The optimizer e.g. SGD or Adam.
3. The model class e.g. MLP, CNN or ViT.

We can also override low-level options, such as specific hyper-parameters of the underlying model.

Default: MLP training run on MNIST using the Adam optimizer:.
```python
python script.py
```
Overriding the MLP hidden dims:
```python
python script.py --multirun model.hidden_dims=[32,64,32]
```

Sweeping over all models with default hyper-parameters:
```python
python script.py --multirun model=mlp,cnn,vit
```

Sweeping over a grid of combinations of optimizer and learning rates:
```python
python script.py --multirun optimizer=adam,sgd optimizer.lr=1e-5,1e-4,1e-3
```

## Installation
To run these examples, you should install the dependencies from the adjacent `env` folder.

n.b. to take advantage of any local accelerator hardware e.g. Apple M1, you may need to follow alternative instructions from the official PyTorch documentation:
https://pytorch.org/get-started/locally/

## Acknowledgements
Adapted from the official PyTorch MNIST example.

You can find the original example, plus a number of other valuable Pytorch examples at: https://github.com/pytorch/examples