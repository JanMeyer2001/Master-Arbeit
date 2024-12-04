import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
from os import listdir, scandir, remove
from os.path import join, isfile, basename
import matplotlib.pyplot as plt
import itertools
from natsort import natsorted
import glob
import h5py
import fastmri
from fastmri.data import transforms as T
import torch.nn.functional as F
from skimage.io import imread
import time
from skimage.metrics import structural_similarity, mean_squared_error
import csv
from os import mkdir
from os.path import isdir
from matplotlib.pyplot import cm
from torch.fft import fftn, ifftn, fftshift, ifftshift
from typing import Optional, Tuple, Union
from skimage.util import view_as_windows
from scipy.sparse import csr_matrix
import math
import random
from Functions import *

# get path for fully sampled k-space
path_origin = '/home/jmeyer/storage/staff/ziadalhajhemid/CMRxRecon23/MultiCoil/Cine/TestSet/FullSample/P001'
fullmulti   = readfile2numpy(join(path_origin, 'cine_sax.mat'),real=False)
[num_frames, num_slices, num_coils, H, W] = fullmulti.shape

# get subsampling mask für Acc4
path_subsampled = '/home/jmeyer/storage/staff/ziadalhajhemid/CMRxRecon23/MultiCoil/Cine/TestSet/AccFactor04/P001'
mask = readfile2numpy(join(path_subsampled, 'cine_sax_mask.mat'),real=True)

# load k-space data and coil maps (size of [C,H,W] with C being coil channels)
k_space = fullmulti[:,0,:,:,:]
image = normalize(fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(T.to_tensor(k_space[0,:,:,:]))), dim=0))
plt.subplot(1,1,1)
plt.imshow(image.cpu().detach().numpy(), cmap='gray')
plt.axis('off')
plt.savefig('./Thesis/Images/LineSwappingOriginal.png')
plt.close

# add motion artifacts to the k-space data via line swapping
k_spaces_motion_16 = k_space[0,:,:,:]
# get random numbers for the original line, the new line to replace it and the frame to use
original_lines  = random.sample(range(H), 16)
new_lines       = random.sample(range(H), 16)
for i, line in enumerate(original_lines):
    new_frame = random.sample(range(num_frames), 1)
    k_spaces_motion_16[:,line,:] = k_space[new_frame,:,new_lines[i],:] 

# subsampling motion-corrupted k-spaces
k_spaces_motion_16_subsampled = k_spaces_motion_16 * mask

# get subsampled images from the motion-corrupted k-space data
image_16 = normalize(fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(T.to_tensor(k_spaces_motion_16_subsampled))), dim=0))
plt.subplot(1,1,1)
plt.imshow(image_16.cpu().detach().numpy(), cmap='gray')
plt.axis('off')
plt.savefig('./Thesis/Images/LineSwapping16.png')
plt.close

# add motion artifacts to the k-space data via line swapping
k_spaces_motion_32 = k_space[0,:,:,:]
# get random numbers for the original line, the new line to replace it and the frame to use
original_lines  = random.sample(range(H), 32)
new_lines       = random.sample(range(H), 32)
for i, line in enumerate(original_lines):
    new_frame = random.sample(range(num_frames), 1)
    k_spaces_motion_32[:,line,:] = k_space[new_frame,:,new_lines[i],:] 

# subsampling motion-corrupted k-spaces
k_spaces_motion_32_subsampled = k_spaces_motion_32 * mask

# get subsampled images from the motion-corrupted k-space data
image_32 = normalize(fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(T.to_tensor(k_spaces_motion_32_subsampled))), dim=0))
plt.subplot(1,1,1)
plt.imshow(image_32.cpu().detach().numpy(), cmap='gray')
plt.axis('off')
plt.savefig('./Thesis/Images/LineSwapping32.png')
plt.close