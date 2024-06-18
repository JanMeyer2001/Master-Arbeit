import os
from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
import csv
import time
from skimage.metrics import structural_similarity, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import nibabel

#Dices=[]
MSE_test = []
SSIM_test = []
#NegJ=[]
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

csv_name = './TestMetrics-NiftyReg-FullySampled.csv'
f = open(csv_name, 'w')
with f:
    fnames = ['Image Pair','MSE','SSIM','Mean MSE','Mean SSIM'] #,'Mean NegJ'
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

path = '/home/jmeyer/storage/students/janmeyer_711878/data/Nifti'
image_pairs = [os.path.basename(f.path) for f in os.scandir(path) if f.is_dir() and f.name.startswith('ImagePair')]

for image_pair in image_pairs:
    # read in images from folder 
    warped_img  = nibabel.load(join(path, image_pair, 'WarpedImage.nii'))
    fix_img     = nibabel.load(join(path, image_pair, 'FixedImage.nii'))

    warped_img  = np.array(warped_img.get_fdata(), dtype='float32')
    fix_img     = np.array(fix_img.get_fdata(), dtype='float32')

    # calculate metrics on fully sampled images
    MSE = mean_squared_error(warped_img, fix_img)
    SSIM = structural_similarity(warped_img, fix_img, data_range=1)

    MSE_test.append(MSE)
    SSIM_test.append(SSIM)
    """
    hh, ww = V_xy.shape[-2:]
    V_xy = V_xy.detach().cpu().numpy()
    V_xy[:,0,:,:] = V_xy[:,0,:,:] * hh / 2
    V_xy[:,1,:,:] = V_xy[:,1,:,:] * ww / 2

    jac_det = jacobian_determinant_vxm(V_xy[0, :, :, :])
    negJ = np.sum(jac_det <= 0) / 160 / 192 * 100
    NegJ.append(negJ)
    """
    # save to csv file
    f = open(csv_name, 'a')
    with f:
        writer = csv.writer(f)
        writer.writerow([image_pair[-1], MSE, SSIM, '-', '-']) 

    """
    for bs_index in range(bs):
        dice_bs = dice(warped_mov_lab[bs_index,...].data.cpu().numpy().copy(),fix_lab[bs_index,...].data.cpu().numpy().copy())
        Dices.append(dice_bs)
    """ 

mean_MSE = np.mean(MSE_test)
std_MSE = np.std(MSE_test)

mean_SSIM = np.mean(SSIM_test)
std_SSIM = np.std(SSIM_test)

#mean_NegJ = np.mean(NegJ)
#std_NegJ = np.std(NegJ)

f = open(csv_name, 'a')
with f:
    writer = csv.writer(f)
    writer.writerow(['-', '-', '-', mean_MSE, mean_SSIM])  #, mean_NegJ

print('Mean MSE: ', mean_MSE, 'Std MSE: ', std_MSE)
print('Mean SSIM: ', mean_SSIM, 'Std SSIM: ', std_SSIM)
#print('Mean DetJ<0 %:', mean_NegJ, 'Std DetJ<0 %:', std_NegJ)
#print('Mean Dice Score: ', np.mean(Dices), 'Std Dice Score: ', np.std(Dices))
