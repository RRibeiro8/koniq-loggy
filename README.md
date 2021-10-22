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

In order to train and test models, you have to download the required datasets:

### KonIQ-10k Image Database (512x384)

The original KonIQ-10k dataset can be download from the [KonIQ-10k website](http://database.mmsp-kn.de/koniq-10k-database.html).
We used the dataset with the image of resolution 512x384.

After downloading the 'koniq10k_512x384.zip' into the folder of this project, you can follow the next command:


```
unzip koniq10k_512x384.zip -d koniq/images/
```


### LIVE In the Wild Image Quality Challenge Database

The LIVE-itW dataset can be download from the [LIVE In the Wild Image Quality Challenge Database](https://live.ece.utexas.edu/research/ChallengeDB/index.html) website.

You only have to copy the test images of the LIVE-itW dataset to the folder ['koniq/images/live_500x500/'](koniq/images/live_500x500/).



### LSC'21 Dataset

The lifelogging dataset can be downloaded from the [LSC'21 Challenge](http://lsc.dcu.ie/lsc_data/) webpage.

After downloading put the date folders with the images of the dataset into the folder ['datasets/LSC21/lsc21-image/'](datasets/LSC21/lsc21-image/) of this project. 


## Experimental Setup and Results

In this section, we provide our procedure to train models with KonIQ-10k dataset and the test results in the KonIQ-10k and LIVE-itW. We also provide some files with the MOS (Mean Oinion Score) of the lifelogging images using two pre-trained models.

The original KonCept512 pre-trained model can be found and downloaded from the original repository of [KonIQ-10k](https://github.com/subpic/koniq).


### Training Models


#### With Tensorflow

In a first stage, we trained the KonCept model three times with different batch sizes of 2, 4 and 8.

In order to run the training of the models, we provide the [koncept512_train.py](koncept512_train.py) python script, that can be used as follows:


```
python koncept512_train.py
```

This script is prepared to train the KonCept512 model with batch size of 8 and MSE (Mean Square Error) loss function.
However, you can change in the code the batch size to 4 or 2, in order to train the other models.

To train the model with the PLCC-induced loss, the loss function has to be changed to 'loss=ops.plcc_loss' in the code of the [koncept512_train.py](koncept512_train.py) python script. And run this script again.

The trained models are save in the folder [koniq/models/koniq/](koniq/models/koniq/) folder.


#### With Keras utilities (Kuti) library installed with pip


Install the [Keras utilities (Kuti)](https://pypi.org/project/kuti/) with:

```
pip install kuti
```

And run the [koncept512_train_with_kuti.py](koncept512_train_with_kuti.py) python script to train the model:

```
python koncept512_train_with_kuti.py
```

#### With PyTorch


Run the [pytorch_train_koncept512.py](pytorch_train_koncept512.py) python script:

```
python pytorch_train_koncept512.py
```


### Testing Models

We tested the models in the KonIQ-10k and LIVE-itW datasets. 
The pre-trained models to be tested must be in the [trained_models/](trained_models/) folder. 

#### KonIQ-10k 

Change the name of the pre-trained model to be tested in the code of the python script and run:


```
python test_koncept512.py
```

or with kuti library installed with pip:

```
python test_koncept512_with_kuti.py
```


#### LIVE-itW

Change the name of the pre-trained model to be tested in the code of the python script and run:


```
python test_LIVEitw.py
```


#### With PyTorch

To test the pytorch models in KonIQ-10k and LIVE-itW datasets run the python script:

```
python pytorch_test_koncept512.py
```

Note that to change the testing from KonIQ-10k to LIVE-itW, you have to change in the code the 'ids' and 'data_dir' variables to 'ids = read_mat_to_DataFrame()' and 'data_dir="koniq/images/live_500x500"'.



### Results and Pre-trained models
