import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import os
from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt
import h5py
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc, EquispacedMaskFractionFunc
import torch.nn.functional as F
from torchvision.utils import save_image
from argparse import ArgumentParser
import time

parser = ArgumentParser()

parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose mode: fully sampled (0) or 4x accelerated (1) and 8x accelerated (2)")
opt = parser.parse_args()
mode = opt.mode

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

def save_slice(fullmulti, frame, slice, savepath):
    '''
    generate slice, reconstruct image from k-space, normalize to [0,1], interpolate to [246,512] and save image
    '''
    slice_kspace = fullmulti[frame,slice] 
    # convert to tensor
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
    # save image to target dir as png
    save_image(image, savepath + '/Frame' + str(frame) + '_Slice' + str(slice) + '.png')


if mode == 0:
    # path for fully sampled CMR k-space data
    data_path = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/FullSample'
    # target path for reconstructed images
    image_path = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/TrainingSet/FullySampled/'
elif mode == 1:
    # path for accelerated CMR k-space data
    data_path = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/AccFactor04'
    # target path for reconstructed images
    image_path = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/TrainingSet/AccFactor04/'
elif mode == 2:
    # path for accelerated CMR k-space data
    data_path = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/AccFactor08'
    # target path for reconstructed images
    image_path = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/TrainingSet/AccFactor08/'
else:
    print('Wrong input for mode!! Choose either fully sampled (0), 4x accelerated (1) or 8x accelerated (2)')    

# get all patient folders
folderpaths = [f.path for f in os.scandir(data_path) if f.is_dir() and not (f.name.find('P') == -1)]

# create dir if not already there 
if not os.path.isdir(image_path):
    os.mkdir(image_path)

i = 0
print('Started generating image data on ', time.ctime())

for path in folderpaths:
    if i == 0:
        start = time.time()
    elif i == 1:
        end = time.time()
        print('Expected time remaining: ', ((end-start)*(len(folderpaths)-1))/60, ' minutes.') 
    
    folder = os.path.basename(path)
    #print('working on folder: ', folder)
    
    # ensure that all target folders exist
    if not os.path.isdir(os.path.join(image_path, folder)):
        os.mkdir(os.path.join(image_path, folder)+'/')
    
    # Load k-space
    fullmulti = readfile2numpy(os.path.join(path, 'cine_sax.mat'))  
    [nframe, nslice, ncoil, ny, nx] = fullmulti.shape
    
    # get image for every frame/slice and save it to the corresponding folder
    for slice in range(nslice):
        # ensure that subfolder for each slice exists
        subfolder = 'Slice' + str(slice)
        if not os.path.isdir(os.path.join(image_path, folder, subfolder)):
            os.mkdir(os.path.join(image_path, folder, subfolder)+'/')

        for frame in range(nframe):
            #save all frames into the subfolder of the slice
            save_slice(fullmulti, frame, slice, os.path.join(image_path, folder, subfolder))
    i = i+1

print('Finished generating image data on ', time.ctime())        
