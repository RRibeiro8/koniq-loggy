# Blind Image Quality Assessment with DeepLearning: a Replicability Study and its Reproducibility in Lifelogging

This work is a replicability study of best model of the paper ["KonIQ-10k: An Ecologically Valid Database for Deep Learning of Blind Image Quality Assessment"](https://arxiv.org/pdf/1910.06180.pdf) and its reproducibility in a lifelogging dataset.

The [koniq](koniq/) folder is the [KonIQ-10k](https://github.com/subpic/koniq) original code repository and the [ku](ku/) folder is the [Keras Utilities](https://github.com/subpic/ku) library provided by the author of the original repository. (Note that some changes have been made to this library files to update some libraries to newer versions.)

## Requirements

- Python 3.8
- Tensorflow 2.6
- PyTorch 1.9 
- CUDA 11.X

### Install Requirements

You can install the requirements using the [requirements.txt](requirements.txt) file by running the command:

```
pip install -r requirements.txt

```

Note that the PyTorch library is not in the [requirements.txt](requirements.txt), so you have to install it by running:

```
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

```

The CUDA 11.X library can be installed by using the original documentation ["NVIDIA CUDA Installation Guide"](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).


## Datasets

