# gnn-reinforcement-learning

## Setup Guide

### Local Development and Setup

First we need to install all requirements. Preferably in a new conda environment:
```
conda create -n "adlr"
conda activate adlr
```
Don't forget to activate your new conda environment.

<br/>

Now we can get to requirements:

1. Install [PyBullet](https://github.com/bulletphysics/bullet3):
```
pip install pybullet --upgrade --user
```

2. Install [Pytorch](https://pytorch.org/) as described [here](https://pytorch.org/get-started/locally/), for example using:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
```

3. Install the [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) libraries as described [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html):

Get your PyTorch and Cuda version using:
```
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
```
Then install the correct package versions. For example, for PyTorch 1.7.0 and CUDA 11.0, type:

```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
pip install torch-geometric
```

4. Install [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) directly from the source.
At the time of writing this, the latest version you can get using `pip install stable-baselines3` is not recent enough.
Instead use one of the following options to install the latest version from github:
```
pip install -e git+https://github.com/DLR-RM/stable-baselines3#egg=stable-baselines3[docs,tests,extra]
```
or alternatively for development purposes:
```
git clone https://github.com/DLR-RM/stable-baselines3 && cd stable-baselines3
pip install -e .[docs,tests,extra]
```


5. Install our NerveNet implementation with the rest of the dependencies. Change the directory to the repositories root folder and run:
```
pip install -e .
```

## Docker
https://github.com/GoogleCloudPlatform/ai-platform-samples/tree/master/training/pytorch/structured/custom_containers/gpu
