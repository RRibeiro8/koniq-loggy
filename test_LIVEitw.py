from kuti import model_helper as mh
from kuti import applications as apps
from kuti import tensor_ops as ops
from kuti import generic as gen
from kuti import image_utils as iu

import pandas as pd, numpy as np, os
from scipy.io import loadmat
from matplotlib import pyplot as plt
from munch import Munch

'''import tensorflow as tf

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
    print(e)'''

def read_mat_to_DataFrame():

  live_images_mat = loadmat(data_root + 'metadata/AllImages_release.mat')
  live_MOS_mat = loadmat(data_root + 'metadata/AllMOS_release.mat')

  live_images_mdata = live_images_mat['AllImages_release']
  live_MOS_mdata = live_MOS_mat['AllMOS_release']


  live_images_path = [item.flat[0][0] for item in live_images_mdata]
  live_MOS_values = [item.flat[0] for item in live_MOS_mdata[0]]
  
  
  live_test = {'image_name': [], 'MOS': [], 'set': []}
  for img_path, mos in zip(live_images_path, live_MOS_values):

    if not img_path.startswith('t'):
      live_test['image_name'].append(img_path)
      live_test['MOS'].append(mos)
      live_test['set'].append('test') 

  return pd.DataFrame(live_test, columns=['image_name','MOS', 'set'])


data_root = 'koniq/'
model_root = 'trained_models/'

#ids = pd.read_csv(data_root + 'metadata/koniq10k_distributions_sets.csv')

#print(ids)

ids = read_mat_to_DataFrame()

from tensorflow.keras.models import Model

base_model, preprocess_fn = apps.get_model_imagenet(apps.InceptionResNetV2)
head = apps.fc_layers(base_model.output, name='fc', 
                      fc_sizes      = [2048, 1024, 256, 1], 
                      dropout_rates = [0.25, 0.25, 0.5, 0], 
                      batch_norm    = 2)    

model = Model(inputs = base_model.input, outputs = head)

# Parameters of the generator
pre = lambda im: preprocess_fn(
         iu.ImageAugmenter(im, remap=False).fliplr().resize((384,512)).result)
gen_params = dict(batch_size  = 8,
                  data_path   = data_root+'images/live_500x500/',
                  process_fn  = pre, 
                  input_shape = (384,512,3),
                  inputs      = ['image_name'],
                  outputs     = ['MOS'])

# Wrapper for the model, helps with training and testing
helper = mh.ModelHelper(model, 'KonCept512', ids, 
                     loss='MSE', metrics=["MAE", ops.plcc_tf],
                     monitor_metric = 'val_loss', 
                     monitor_mode   = 'min', 
                     multiproc   = True, workers = 5,
                     logs_root   = data_root + 'logs/koniq',
                     models_root = data_root + 'models/koniq',
                     gen_params  = gen_params)

helper.model.load_weights(model_root + 'original_koncep512-trained-model.h5')

y_pred = helper.predict()
y_true = ids[ids.set=='test'].MOS.values
apps.rating_metrics(y_true, y_pred);