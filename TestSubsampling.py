import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import os
from os import listdir, scandir
from os.path import join, isfile
import matplotlib.pyplot as plt
import itertools
import scipy
import h5py
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc

def centerCrop(k_space, offsetx, offsety):
    [ncoil, ny, nx, _] = k_space.shape
    k_space[:, 0:(round(ny/2)-offsety), 0:(round(nx/2)-offsetx),:] = 0
    k_space[:, (round(ny/2)-offsety):ny, (round(nx/2)+offsetx):nx,:] = 0

    """
    for i in range(ncoil):
        crop = torch.zeros([ny, nx])
        crop[(round(ny/2)-offsety):(round(ny/2)+offsety), (round(nx/2)-offsetx):(round(nx/2)+offsetx)] = 1
        
        plt.subplot(1, 1, 1)
        plt.imshow(crop, cmap='gray', vmax = 1)
        plt.axis('off')
        plt.savefig('CenterCrop.png') 
        plt.close
        
        k_space[i,:,:,0] = k_space[i,:,:,0] * crop
        k_space[i,:,:,1] = k_space[i,:,:,1] * crop
    """
    return k_space

def readfile2numpy(file_name):
    '''
    read the data from mat and convert to numpy array
    '''
    hf = h5py.File(file_name)
    keys = list(hf.keys())
    assert len(keys) == 1, f"Expected only one key in file, got {len(keys)} instead"
    new_value = hf[keys[0]][()]
    data = new_value["real"] + 1j*new_value["imag"]
    return data

# Data from CMRxRecon
data_path = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/FullSample'
names = [f.path for f in os.scandir(data_path) if f.is_dir()]

# read files from mat to numpy
fullmulti = readfile2numpy(os.path.join(data_path, names[0], 'cine_sax.mat'))
[nframe, nslice, ncoil, ny, nx] = fullmulti.shape

slice_kspace = fullmulti[0,0] 

# Convert from numpy array to pytorch tensor
slice_kspace2 = T.to_tensor(slice_kspace) 

# create mask for subsampling
mask_func = RandomMaskFunc(center_fractions=[0.04], accelerations=[8])

# subsample the k-space
slice_kspace_subsampled, mask, _ = T.apply_mask(slice_kspace2, mask_func)  #centerCrop(slice_kspace2, offsetx=40, offsety=40)

# Apply Inverse Fourier Transform to get the complex image      
image = fastmri.ifft2c(slice_kspace2)        
image_subsampled = fastmri.ifft2c(slice_kspace_subsampled)  

# Compute absolute value to get a real image      
image_abs = fastmri.complex_abs(image)        
image_abs_subsampled = fastmri.complex_abs(image_subsampled)   

# combine the coil images to a coil-combined one
slice_image_rss = fastmri.rss(image_abs, dim=0)
slice_image_rss_subsampled = fastmri.rss(image_abs_subsampled, dim=0)

# plot the final image
plt.subplot(1, 2, 1)
plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray', vmax = 0.0015)
plt.axis('off')
plt.title('Original')
plt.subplot(1, 2, 2)
plt.imshow(np.abs(slice_image_rss_subsampled.numpy()), cmap='gray', vmax = 0.0015)
plt.axis('off')
plt.title('Subsampling')
plt.tight_layout()
plt.savefig('./Thesis/Images/Subsampling.png') 
plt.close


