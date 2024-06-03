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

parser = ArgumentParser()

parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose mode: fully sampled (0) or 4x accelerated (1) and 8x accelerated (2)")
parser.add_argument("--set", type=int, dest="set", default='0',
                    help="choose subset: training data (0), validation data (1) or test data (2)")
opt = parser.parse_args()
mode = opt.mode
set = opt.set

def normalize(image):
    """ expects an image as tensor and normalizes the values between [0,1] """
    return (image - torch.min(image)) / (torch.max(image) - torch.min(image))

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

def extract_slice(fullmulti, H, W):
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
    image = F.interpolate(image.unsqueeze(0).unsqueeze(0), (H, W), mode='bilinear').squeeze(0).squeeze(0)

    return image

def extract_differences(images, frames, slices, H, W):
    differences = torch.zeros([slices,frames-1, H, W]) 
    for slice in range(slices):
        for i in range(frames):
            if i>1:
                differences[slice,i-2,:,:] = abs(images[slice,i-1,:,:]-images[slice,i-2,:,:])

    mean_diff_frames = torch.sum(differences, dim=1)/differences.shape[1] # mean over all differences
    diffs_slices = torch.sum(mean_diff_frames, dim=0) # sum over all differences between slices

    return diffs_slices

if set == 0:
    set = 'TrainingSet'
elif set == 1:
    set = 'ValidationSet'
elif set == 2:
    set = 'TestSet'
else:
    print('Wrong input for set!! Choose either training set (0), validation set (1) or test set (2)')    

if mode == 0:
    # path for fully sampled CMR k-space data
    data_path = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/' + set + '/FullSample'
    # target path for reconstructed images
    image_path = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/' + set + '/Croped/FullySampled/'
elif mode == 1:
    # path for accelerated CMR k-space data
    data_path = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/' + set + '/AccFactor04'
    # target path for reconstructed images
    image_path = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/' + set + '/Croped/AccFactor04/'
elif mode == 2:
    # path for accelerated CMR k-space data
    data_path = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/' + set + '/AccFactor08'
    # target path for reconstructed images
    image_path = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/' + set + '/Croped/AccFactor08/'
else:
    print('Wrong input for mode!! Choose either fully sampled (0), 4x accelerated (1) or 8x accelerated (2)')    

# get all patient folders
patients_folders = [f.path for f in os.scandir(data_path) if f.is_dir() and not (f.name.find('P') == -1)]

# create dir if not already there 
if not isdir(image_path):
    os.mkdir(image_path)

print('Started generating image data on ', time.ctime())

for i, patient_path in enumerate(patients_folders):
    if i == 0:
        start = time.time()
    elif i == 1:
        end = time.time()
        print('Expected time remaining: ', ((end-start)*(len(patients_folders)-1))/60, ' minutes.') 
    
    folder = os.path.basename(patient_path)
    if folder != 'P004': # this folder in the ValidationSet-FullSample causes issues...
        print('working on folder: ', folder)
        
        # ensure that all target folders exist
        if not isdir(join(image_path, folder)):
            os.mkdir(join(image_path, folder)+'/')
        
        # Load k-space
        fullmulti = readfile2numpy(join(patient_path, 'cine_sax.mat'))  
        [frames, slices, ncoil, ny, nx] = fullmulti.shape
        [H, W] = [246, 512]
        images = torch.zeros([slices, frames, H, W]) 
        diffs = torch.zeros([slices, frames-1, H, W]) 
        
        # get image for every frame/slice
        for slice in range(slices):
            for frame in range(frames):
                # store all frames for all slices
                images[slice, frame,:,:] = extract_slice(fullmulti, H, W)
        
        # get sum of differences between frames for every slice
        sum_diffs_slices = normalize(extract_differences(images, frames, slices, H, W))

        # create mask
        mask = sum_diffs_slices>0.5                                    # threshold to get the rough mask
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
