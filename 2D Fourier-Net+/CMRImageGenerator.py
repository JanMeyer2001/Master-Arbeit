import numpy as np
import torch
import os
from os import listdir
from os.path import join, isfile, isdir
import matplotlib.pyplot as plt
import h5py
import fastmri
from fastmri.data import transforms as T
import torch.nn.functional as F
#from torchvision.utils import save_image # stopped working for some reason...
from skimage.io import imsave
from argparse import ArgumentParser
import time
from skimage.morphology import area_opening
from skimage.util import img_as_ubyte
from Functions import *

parser = ArgumentParser()

parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose mode: fully sampled (0) or 4x accelerated (1), 8x accelerated (2) and 10x accelerated (3)")
parser.add_argument("--subset", type=int, dest="subset", default='0',
                    help="choose subset: training data (0), validation data (1) or test data (2)")
opt = parser.parse_args()
mode = opt.mode
subset = opt.subset


# path for k-space data
path_origin = '/home/jmeyer/storage/staff/ziadalhajhemid/CMRxRecon23/MultiCoil/Cine/'
#path_origin = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/'

# path for image data
path_target = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/'

if subset == 0:
    subset = 'TrainingSet'
elif subset == 1:
    subset = 'ValidationSet'
elif subset == 2:
    subset = 'TestSet'
else:
    print('Wrong input for set!! Choose either training set (0), validation set (1) or test set (2)')    

if mode == 0:
    # path for fully sampled CMR k-space data
    data_path = path_origin + subset + '/FullSample'
    # target path for reconstructed images
    image_path = path_target + subset + '/Croped/FullySampled/'
elif mode == 1:
    # path for accelerated CMR k-space data
    data_path = path_origin + subset + '/AccFactor04'
    # target path for reconstructed images
    image_path = path_target + subset + '/Croped/AccFactor04/'
elif mode == 2:
    # path for accelerated CMR k-space data
    data_path = path_origin + subset + '/AccFactor08'
    # target path for reconstructed images
    image_path = path_target + subset + '/Croped/AccFactor08/'
elif mode == 3:
    # path for accelerated CMR k-space data
    data_path = path_origin + subset + '/AccFactor10'
    # target path for reconstructed images
    image_path = path_target + subset + '/Croped/AccFactor10/'
else:
    print('Wrong input for mode!! Choose either fully sampled (0), 4x accelerated (1) or 8x accelerated (2)')    

# get all patient folders
patients_folders = [f.path for f in os.scandir(data_path) if f.is_dir() and not (f.name.find('P') == -1)]
#patients_folders = [join(data_path,'P116')]

# create dir if not already there 
if not isdir(image_path):
    os.mkdir(image_path)

print('Started generating image data on ', time.ctime())
print('  working on folder: ')

for i, patient_path in enumerate(patients_folders):
    """
    if i == 0:
        start = time.time()
    elif i == 1:
        end = time.time()
        print('Expected time remaining: ', ((end-start)*(len(patients_folders)-1))/60, ' minutes.') 
    """
    # get the name of the patient folder with the full path
    folder = os.path.basename(patient_path)
    # create list for already processed patients
    if subset == 'TrainingSet':
        already_processed = ['P080'] # no .sax data for this folder in the training set
    elif subset == 'ValidationSet':
        already_processed = ['P004'] # this folder causes problems for the validation set
    elif subset == 'TestSet':
        already_processed = ['P012', 'P018', 'P042', 'P074', 'P080', 'P113', 'P117'] # no .sax data for these folders in the test set
    else:     
        already_processed = []
    
    if folder not in already_processed: # block already processed folders
        print('   ', folder)
        
        # ensure that all target folders exist
        if not isdir(join(image_path, folder)):
            os.mkdir(join(image_path, folder)+'/')
        
        # Load k-space
        fullmulti = readfile2numpy(join(patient_path, 'cine_sax.mat'))  
        [frames, slices, ncoil, ny, nx] = fullmulti.shape
        [H, W] = [246, 512]
        images = torch.zeros([slices, frames, H, W]) 
        
        # get image for every frame/slice
        for slice in range(slices):
            for frame in range(frames):
                # store all frames for all slices
                images[slice, frame,:,:] = extract_slice(fullmulti, slice, frame, H, W)
        
        # get sum of differences between frames for every slice
        sum_diffs_slices = normalize(extract_differences(images, frames, slices, H, W))

        # create mask
        mask = sum_diffs_slices>0.3                                    # threshold to get the rough mask
        mask = area_opening(mask.int().numpy(), area_threshold=125)     # perform opening to get rid of any small artifacts
        
        # get all non-zero positions in the mask to find the center coordinates for cardiac region
        positions = np.nonzero(mask)    
        center_x = positions[0].min() + int((positions[0].max() - positions[0].min())/2)
        center_y = positions[1].min() + int((positions[1].max() - positions[1].min())/2)
        
        # crop all images for the patient accordingly
        crop_x = int(H/6)
        crop_y = int(W/6)
        images = images[:,:,(center_x-crop_x):(center_x+crop_x),(center_y-crop_y):(center_y+crop_y)]
        
        # save images to the corresponding folders
        for slice in range(slices):
            # ensure that subfolder for each slice exists
            subfolder = 'Slice' + str(slice)
            if not os.path.isdir(os.path.join(image_path, folder, subfolder)):
                os.mkdir(os.path.join(image_path, folder, subfolder)+'/')
            for frame in range(frames):
                # convert image to uint8
                image = img_as_ubyte(images[slice,frame,:,:].numpy())
                # save image to target dir as png
                imsave(join(image_path, folder, subfolder) + '/Frame' + str(frame) + '.png', image, check_contrast=False) # + '_Slice' + str(slice)
                #save_image(images[slice,frame,:,:], join(image_path, folder, subfolder) + '/Frame' + str(frame) + '.png') # + '_Slice' + str(slice)


print('Finished generating image data on ', time.ctime())        
