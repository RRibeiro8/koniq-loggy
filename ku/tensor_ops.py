from __future__ import print_function
from __future__ import division
from builtins import map
from past.utils import old_div
import os, sys
from keras import backend as K
import tensorflow as tf
from scipy import stats
import numpy as np
import tensorflow_probability as tfp

#if sys.version_info.major==3:
    #tf = K.tensorflow_backend.tf
#else:
    #tf = K.tf

# Keras configuration directives
def SetActiveGPU(number=0):
    """
    Set visibility of GPUs to the Tensorflow engine.

    * number: scalar or list of GPU indices
              e.g. 0 for the 1st GPU, or [0,2] for the 1st and 3rd GPU
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not isinstance(number,list): number=[number]
    os.environ["CUDA_VISIBLE_DEVICES"] = ', '.join(map(str,number))
    print('Visible GPU(s):', os.environ["CUDA_VISIBLE_DEVICES"])

def GPUMemoryCap(fraction=0):
    """
    Limit the amount of GPU memory that can be used by an active kernel.

    * fraction: in [0, 1], 1 = the entire available GPU memory.
    """
    config = tf.compat.v1.ConfigProto()#tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    config.gpu_options.allow_growth = True
    K.set_session(tf.compat.v1.Session(config=config)) #tf.Session(config=config)


# Metrics and losses

def correlationMetric(x, y, axis=-2):
    """Metric returning the Pearson correlation coefficient of two tensors over some axis, default -2."""
    x = tf.convert_to_tensor(x)
    y = tf.cast(y, x.dtype)
    n = tf.cast(tf.shape(x)[axis], x.dtype)
    xsum = tf.reduce_sum(x, axis=axis)
    ysum = tf.reduce_sum(y, axis=axis)
    xmean = xsum / n
    ymean = ysum / n
    xvar = tf.reduce_sum( tf.math.squared_difference(x, xmean), axis=axis)
    yvar = tf.reduce_sum( tf.math.squared_difference(y, ymean), axis=axis)
    cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
    corr = cov / tf.sqrt(xvar * yvar)
    return corr#tf.constant(1.0, dtype=x.dtype) - corr
    
def plcc(x, y):
    """PLCC metric"""
    return tfp.stats.correlation(x,y)


def plcc_tf(x, y):

    xc = x - K.mean(x)
    yc = y - K.mean(y) 

    return K.mean(xc*yc)/(K.std(x)*K.std(y) + K.epsilon())

def plcc_loss(x, y):
    """Loss version of `plcc_tf`"""
    return (1. - plcc(x, y)) / 2.

def earth_mover_loss(y_true, y_pred):
    """
    Earth Mover's Distance loss.

    Reproduced from https://github.com/titu1994/neural-image-assessment/blob/master/train_inception_resnet.py
    """
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

def make_loss(loss, **params_defa):
    def custom_loss(*args, **kwargs):
        kwargs.update(params_defa)
        return loss(*args, **kwargs)
    return custom_loss


def get_plcc_dist_tf(scores_array=[1., 2, 3, 4, 5]):
    """
    Function generator for `plcc_dist_tf`
    Computes the PLCC between the MOS values computed
    from pairs of distributions of scores. Used as a metric.

    * scores_array: scale values
    """
    def plcc_dist_tf(x, y):
        scores = K.constant(scores_array)
        xm = K.sum((x / K.reshape(K.sum(x, 1), [-1, 1])) * scores, 1)
        ym = K.sum((y / K.reshape(K.sum(y, 1), [-1, 1])) * scores, 1)
        x_sd = K.std(xm)
        y_sd = K.std(ym)
        xm_center = xm - K.mean(xm)
        ym_center = ym - K.mean(ym)
        return K.mean(xm_center*ym_center)/(x_sd*y_sd + 1e-3)
    return plcc_dist_tf

def get_plcc_dist_loss(scores_array=[1., 2, 3, 4, 5]):
    """Loss version of `plcc_dist_tf`"""
    plcc_dist_tf = get_plcc_dist_tf(scores_array)
    def plcc_dist_loss(x, y):
        return (1-plcc_dist_tf(x, y))/2
    return plcc_dist_loss