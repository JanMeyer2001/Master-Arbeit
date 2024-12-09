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
import torch.nn.functional as F

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


def show_coils(data, slice_nums, cmap=None, vmax = 0.0005):
    '''
    plot the figures along the first dims.
    '''
    fig = plt.figure(figsize=(6,10))
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0.0, top=1)
        plt.imshow(data[num], cmap="gray",vmax=vmax)
        plt.axis('off')
        
    plt.show()
    plt.tight_layout(h_pad=0.1, w_pad=0.3, pad=0)
    plt.savefig('./Images/Coils.png', dpi=600, pad_inches=0.05, bbox_inches="tight") 
    plt.close


# fully sampled data from CMRxRecon
data_path = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/FullSample'
names = [f.path for f in os.scandir(data_path) if f.is_dir() and f.name.endswith('P120')]
#print('name of fully sampled data: ', names[0])

# read files from mat to numpy
fullmulti = readfile2numpy(os.path.join(data_path, names[0], 'cine_sax.mat'))
[nframe, nslice, ncoil, nx, ny] = fullmulti.shape

# choose frame and slice
slice_kspace = fullmulti[0,0] 

slice_kspace = T.to_tensor(slice_kspace) 
# Apply Inverse Fourier Transform to get the complex image  
image = fastmri.ifft2c(slice_kspace)
# Compute absolute value to get a real image      
image = fastmri.complex_abs(image) 
crop = [int(nx / 4), int(ny / 4)]
image_crop = image[:,crop[0] : -crop[0], crop[1] : -crop[1]]
show_coils(image_crop,[0,3,6])
# combine the coil images to a coil-combined one
image = fastmri.rss(image, dim=0, cmap='gray', vmax=1)
# normalize images have data range [0,1]
image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
# interpolate images to be the same size
image = F.interpolate(image.unsqueeze(0).unsqueeze(0), (246,512), mode='bilinear').squeeze(0).squeeze(0)

# plot the final image
plt.subplot(1, 1, 1)
#plt.imshow(np.log(np.abs(slice_kspace) + 1e-9)[0,:,:], cmap='gray') # for plotting k-space
plt.imshow(image, cmap='gray') # for plotting coresponding reconstructed image
#plt.title('Fully Sampled')
plt.axis('off')
plt.tight_layout()
#plt.savefig('./Images/k-space_fullysampled.png') # for saving the k-space plot
plt.savefig('./Images/image_fullysampled.png') # for saving the image
plt.close

# Data from CMRxRecon
data_path = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/AccFactor04'
names = [f.path for f in os.scandir(data_path) if f.is_dir() and f.name.endswith('P120')]
#print('name of subsampled data: ', names[0])

# read files from mat to numpy
fullmulti = readfile2numpy(os.path.join(data_path, names[0], 'cine_sax.mat'))
[nframe, nslice, ncoil, ny, nx] = fullmulti.shape

# choose frame and slice
slice_kspace = fullmulti[0,0] 

slice_kspace = T.to_tensor(slice_kspace) 
# Apply Inverse Fourier Transform to get the complex image  
image = fastmri.ifft2c(slice_kspace)
# Compute absolute value to get a real image      
image = fastmri.complex_abs(image) 
# combine the coil images to a coil-combined one
image = fastmri.rss(image, dim=0)
# normalize images have data range [0,1]
image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
# interpolate images to be the same size
image = F.interpolate(image.unsqueeze(0).unsqueeze(0), (246,512), mode='bilinear').squeeze(0).squeeze(0)

# plot the final image
plt.subplot(1, 1, 1)
#plt.imshow(np.log(np.abs(slice_kspace) + 1e-9)[0,:,:], cmap='gray') # for plotting k-space
plt.imshow(image, cmap='gray') # for plotting coresponding reconstructed image
#plt.title('Subsampled')
plt.axis('off')
plt.tight_layout()
#plt.savefig('./Images/k-space_subsampled.png') # for saving the k-space plot
plt.savefig('./Images/image_subsampled.png') # for saving the image
plt.close
