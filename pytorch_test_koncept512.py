from __future__ import print_function 
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import copy
import pandas as pd
from tqdm import tqdm
from scipy import stats
from koncept_model import model_qa
from scipy.io import loadmat


print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device  = torch.device("cuda:0")

data_root = 'koniq/'

def plcc(x, y):
    """Pearson Linear Correlation Coefficient"""
    x, y = np.float32(x), np.float32(y)
    return stats.pearsonr(x, y)[0]

def srocc(xs, ys):
    """Spearman Rank Order Correlation Coefficient"""
    xranks = pd.Series(xs).rank()    
    yranks = pd.Series(ys).rank()    
    return plcc(xranks, yranks)

def rating_metrics(y_true, y_pred, show_plot=True):    
    """
    Print out performance measures given ground-truth (`y_true`) and predicted (`y_pred`) scalar arrays.
    """
    y_true, y_pred = np.array(y_true).squeeze(), np.array(y_pred).squeeze()
    p_plcc = np.round(plcc(y_true, y_pred),3)
    p_srocc = np.round(srocc(y_true, y_pred),3)
    p_mae  = np.round(np.mean(np.abs(y_true - y_pred)),3)
    p_rmse  = np.round(np.sqrt(np.mean((y_true - y_pred)**2)),3)
    
    if show_plot:
        print('SRCC: {} | PLCC: {} | MAE: {} | RMSE: {}'.\
              format(p_srocc, p_plcc, p_mae, p_rmse))    
        plt.plot(y_true, y_pred,'.',markersize=1)
        plt.xlabel('ground-truth')
        plt.ylabel('predicted')
        plt.show()
    return (p_srocc, p_plcc, p_mae, p_rmse)

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

def main():

    KonCept512 = model_qa(num_classes=1) 
    KonCept512.load_state_dict(torch.load('trained_models/KonCept512.pth'))
    KonCept512.eval().to(device)

    data_transforms = {
        'train': transforms.Compose([
    #         transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ]),
        'val': transforms.Compose([
            transforms.Resize((384,512)),
    #         transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }

    #ids = pd.read_csv('koniq/metadata/koniq10k_distributions_sets.csv')
    ids = read_mat_to_DataFrame()
    #data_dir='koniq/images/512x384' images/live_500x500/
    data_dir='koniq/images/live_500x500'
    #ids_train = ids[ids.set=='training'].reset_index()
    #ids_val = ids[ids.set=='validation'].reset_index()
    ids_test = ids[ids.set=='test'].reset_index()

    batch_size=8
    num_batches = np.int(np.ceil(len(ids_test)/batch_size))


    # Iterate over data.
    outputs=np.zeros((len(ids_test),1))
    for k in tqdm(range(0,num_batches)):
        batch_size_cur=min(batch_size,len(ids_test)-k*batch_size)
        img_batch=torch.zeros(batch_size_cur,3,384,512).to(device)   
        for i in range(batch_size_cur):  
            img_batch[i]=data_transforms['val'](Image.open(os.path.join(data_dir,ids_test['image_name'][k*batch_size+i])))  
        label_batch=torch.tensor(list(ids_test['MOS'][k*batch_size:k*batch_size+batch_size_cur]))
        outputs[k*batch_size:k*batch_size+batch_size_cur] = KonCept512(img_batch).detach().cpu().numpy()
     
    y_true = ids[ids.set=='test'].MOS.values
    rating_metrics(y_true, outputs)

    return 0

if __name__ == "__main__":
    main()