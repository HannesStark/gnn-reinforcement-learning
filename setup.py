import logging
import subprocess

from setuptools import setup

# import torch

# cuda_v = f"cu{torch.version.cuda.replace('.', '')}"
# torch_v = torch.__version__.split('.')
# torch_v = '.'.join(torch_v[:-1] + ['0'])


# def system(command: str):
#     output = subprocess.check_output(command, shell=True)
#     logging.info(output)


# system(f'pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
# system(f'pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
# system(f'pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')
# system(f'pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{torch_v}+{cuda_v}.html')

install_requires = [
    'gym',
    'num2words',
    'beautifulsoup4',
    'networkx',
    'lxml',
    'google-cloud-storage',
    'pandas',
    # 'torch-geometric'
]

setup(
    name='NerveNet',
    version='1.0.0',
    description='Implementation of NerveNet for use with PyTorch and StableBaselines3',
    author='Tobias Schmidt, Hannes Stärk',
    author_email='tsbaunatal@gmail.com',
    packages=['NerveNet'],
    install_requires=install_requires,
    zip_safe=False,
)
