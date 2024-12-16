import numpy as np
import torch
from os import scandir, mkdir
from os.path import join, basename, isdir
from fastmri.data import transforms as T
import time
from Functions import *
import nibabel as nib
from mri.operators import FFT
from fastmri.data.subsample import RandomMaskFunc, EquispacedMaskFractionFunc
import warnings
warnings.filterwarnings("ignore")


# path for image data
path_origin = '/home/jmeyer/storage/datasets/ACDC'

# modes
modes = [1,2,3]

# shape of the frames
input_shape = (1,1,216,256)

for mode in modes:
    if mode == 1:
        # get Acc4 mask
        mask_func = EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[4]) 
        # create and repeat mask
        mask, _ = mask_func(input_shape)
        mask = mask.repeat(1,1,1,input_shape[-1]) 
    elif mode == 2:
        # get Acc8 mask
        mask_func = EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[8])
        # create and repeat mask
        mask, _ = mask_func(input_shape)
        mask = mask.repeat(1,1,1,input_shape[-1])  
    elif mode == 3:
        # get Acc10 mask
        mask_func = EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[10]) 
        # create and repeat mask
        mask, _ = mask_func(input_shape)
        mask = mask.repeat(1,1,1,input_shape[-1]) 
    
    # Load 4D image data
    file_path = join(path_origin,'patient001','patient001_4d.nii.gz')
    nim1 = nib.load(file_path) 
    volume = nim1.get_fdata() 

    image = volume[:,:,0,0]
    image = np.array(255*image/image.max(), dtype='uint8')
    
    # turn image into a float tensor
    image = torch.from_numpy(image/255).float()
    # interpolate all images to the same size
    image = F.interpolate(image.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').squeeze(0).squeeze(0)
    # apply fft to get k-space
    k_space_fullysampled = torch.fft.fftn(image)                             
    # apply mask for subsampling
    k_space_subsampled = torch.view_as_real(k_space_fullysampled.unsqueeze(0).unsqueeze(0)[mask.bool()])
    # use pySAP for reconstruction
    fourier_op = FFT(mask=mask, shape=image.shape, n_coils=1)
    kspace_obs = fourier_op.op(image)
    # Zero order solution
    image = np.linalg.norm(fourier_op.adj_op(kspace_obs), axis=0)[0,:,:] 
    #image = np.array(image*255, dtype='uint8') 

    plt.subplot(1, 1, 1)
    plt.imshow(image, cmap='gray') 
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/jmeyer/storage/students/janmeyer_711878/Master-Arbeit/Images/ManualSubsamplingACDC_Mode{}.png'.format(mode)) 
    plt.close