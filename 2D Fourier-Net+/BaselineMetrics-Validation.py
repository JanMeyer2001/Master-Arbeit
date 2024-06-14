import os
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from Models import *
from Functions import *
import torch.utils.data as Data
import matplotlib.pyplot as plt
import csv
import time
from skimage.metrics import structural_similarity, mean_squared_error

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

parser = ArgumentParser()
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon',
                    #default='/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/FullSample', #AccFactor04
                    help="data path for training images")
parser.add_argument("--mode", type=int,
                    dest="mode",
                    default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
opt = parser.parse_args()

bs = opt.bs
datapath = opt.datapath
mode = opt.mode

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

# load CMR validation data
validation_set = ValidationDatasetCMR(datapath, mode) 
validation_generator = Data.DataLoader(dataset=validation_set, batch_size=bs, shuffle=True, num_workers=4)

csv_name = './BaselineMetrics-Validation-Mode' + str(mode) + '.csv'
f = open(csv_name, 'w')
with f:
    fnames = ['Index','MSE','SSIM','Mean MSE','Mean SSIM']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

image_num = 1 
MSE_Validation = []
SSIM_Validation = []
for mov_img, fix_img in validation_generator: 
    #MSE = torch.mean((fix_img.cuda() - mov_img.cuda()) ** 2).cpu().numpy()
    # convert to numpy array
    mov_img = mov_img[0,0,:,:].cpu().numpy()
    fix_img = fix_img[0,0,:,:].cpu().numpy()
    MSE = mean_squared_error(mov_img, fix_img)
    MSE_Validation.append(MSE)
    #print('size moving image: ', mov_img.cpu().numpy().shape, 'size fixed image: ', fix_img.cpu().numpy().shape)
    SSIM = structural_similarity(mov_img, fix_img, data_range=1) #, win_size=[10,10]
    SSIM_Validation.append(SSIM)
    #print('  MSE for validation image pair', image_num, ' is: ', MSE)
    f = open(csv_name, 'a')
    with f:
        writer = csv.writer(f)
        writer.writerow([image_num, MSE, SSIM, '-', '-']) 
    image_num += 1

csv_MSE = np.mean(MSE_Validation)
csv_SSIM = np.mean(SSIM_Validation)
f = open(csv_name, 'a')
with f:
    writer = csv.writer(f)
    writer.writerow(['-', '-', '-', csv_MSE, csv_SSIM])    

print('\n mean MSE: ', csv_MSE,'\n mean SSIM: ', csv_SSIM)