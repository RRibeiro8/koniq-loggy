from ku import model_helper as mh
from ku import applications as apps
from ku import tensor_ops as ops
from ku import generic as gen
from ku import image_utils as iu

import pandas as pd, numpy as np, os
from matplotlib import pyplot as plt
from munch import Munch

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      #tf.config.set_logical_device_configuration(
      #  gpu,
      #  [tf.config.LogicalDeviceConfiguration(memory_limit=6144)])
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



data_root = 'koniq/'

ids = pd.read_csv(data_root + 'metadata/koniq10k_distributions_sets.csv')

from keras.models import Model

# Build scoring model
base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
head = apps.fc_layers(base_model.output, name='fc', 
                      fc_sizes      = [2048, 1024, 256, 1], 
                      dropout_rates = [0.25, 0.25, 0.5, 0], 
                      batch_norm    = 2)    

model = Model(inputs = base_model.input, outputs = head)

# Parameters of the generator
pre = lambda im: preprocess_fn(
         iu.ImageAugmenter(im, remap=False).fliplr().result)
gen_params = dict(batch_size  = 8,# 4 or 2
                  data_path   = data_root+'images/512x384/',
                  process_fn  = pre, 
                  input_shape = (384,512,3),
                  inputs      = ['image_name'],
                  outputs     = ['MOS'])

# Wrapper for the model, helps with training and testing
helper = mh.ModelHelper(model, 'KonCept512', ids, 
                     loss='MSE', metrics=['MAE', ops.plcc,],#loss= 'MSE' or ops.plcc_loss
                     monitor_metric = 'val_loss', 
                     monitor_mode   = 'min', 
                     multiproc   = True, workers = 1,
                     logs_root   = data_root + 'logs/koniq',
                     models_root = data_root + 'models/koniq',
                     gen_params  = gen_params)


helper.train(lr=1e-4, epochs=40)
helper.load_model()
helper.train(lr=1e-4/5, epochs=20)
helper.load_model()
helper.train(lr=1e-4/10, epochs=10)

