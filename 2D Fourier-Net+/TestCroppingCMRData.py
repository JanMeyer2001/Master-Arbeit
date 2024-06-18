import os
from os import listdir, scandir
from os.path import join, isfile
import matplotlib.pyplot as plt
from skimage.io import imread
import torch
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.feature import canny
import numpy as np
from skimage.morphology import area_opening
import cv2
from Functions import *

def extract_mean_differences(images, frames, slices, H, W):
    ''' take mean of frames and sum them over all slices of a patient  '''
    differences = torch.zeros([slices,frames-1, H, W]) 
    for slice in range(slices):
        for i in range(frames):
            if i>1:
                differences[slice,i-2,:,:] = abs(images[slice,i-1,:,:]-images[slice,i-2,:,:])

    mean_diff_frames = torch.sum(differences, dim=1)#/differences.shape[1] # mean over all differences
    
    return mean_diff_frames

data_path = '/home/jmeyer/storage/staff/ziadalhajhemid/CMRxRecon23/MultiCoil/Cine/TrainingSet/FullSample/P007' 
# Load k-space
fullmulti = readfile2numpy(join(data_path, 'cine_sax.mat'))  
[frames, slices, ncoil, ny, nx] = fullmulti.shape
[H, W] = [246, 512]
images = torch.zeros([slices, frames, H, W]) 

# get image for every frame/slice
for slice in range(slices):
    for frame in range(frames):
        # store all frames for all slices
        images[slice, frame,:,:] = extract_slice(fullmulti, slice, frame, H, W)

## new

diffs = extract_mean_differences(images, frames, slices, H, W)
plt.subplots(figsize=(7, 4))
plt.axis('off')
for i, diff in enumerate(diffs):
    plt.subplot(3,4,i+1) 
    plt.imshow(diff, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.savefig('AllDiffs.png') #./Thesis/Images/
plt.close

masks = diffs>0.25
plt.subplots(figsize=(7, 4))
plt.axis('off')
for i, mask in enumerate(masks):
    plt.subplot(3,4,i+1) 
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.savefig('AllMasks.png') #./Thesis/Images/
plt.close

masks = area_opening(masks.int().numpy(), area_threshold=125) 
plt.subplots(figsize=(7, 4))
plt.axis('off')
for i, mask in enumerate(masks):
    plt.subplot(3,4,i+1) 
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.savefig('AllMasksAfterOpening.png') #./Thesis/Images/
plt.close

sum_masks = np.sum(masks, axis=0)
mask = sum_masks>0.5                                    # threshold to get the rough mask
mask = area_opening(mask, area_threshold=125)     # perform opening to get rid of any small artifacts

# get all non-zero positions in the mask to find the center coordinates for cardiac region
positions = np.nonzero(mask)  
#"""
center_x = positions[0].min() + int((positions[0].max() - positions[0].min())/2)
center_y = positions[1].min() + int((positions[1].max() - positions[1].min())/2)

# crop all images for the patient accordingly
crop_x = int(H/6)
crop_y = int(W/6)
image = images[0,0,:,:]
images = images[:,:,(center_x-crop_x):(center_x+crop_x),(center_y-crop_y):(center_y+crop_y)]
image_crop = images[0,0,:,:]
#"""

plt.subplots(figsize=(7, 4))
plt.axis('off')

plt.subplot(1,4,1) 
plt.imshow(image, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout()

plt.subplot(1,4,2) 
plt.imshow(sum_masks, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout() 

plt.subplot(1,4,3) 
plt.imshow(mask, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout()

plt.subplot(1,4,4) 
plt.imshow(image_crop, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout()

plt.savefig('TestCrop.png') 
plt.close


## old


images_old = torch.zeros([slices, frames, H, W]) 
# get image for every frame/slice
for slice in range(slices):
    for frame in range(frames):
        # store all frames for all slices
        images_old[slice, frame,:,:] = extract_slice(fullmulti, slice, frame, H, W)

# get sum of differences between frames for every slice
sum_diffs_slices = normalize(extract_differences(images_old, frames, slices, H, W))

# create mask
mask = sum_diffs_slices>0.3                                    # threshold to get the rough mask
mask = area_opening(mask.int().numpy(), area_threshold=125)     # perform opening to get rid of any small artifacts

# get all non-zero positions in the mask to find the center coordinates for cardiac region
positions = np.nonzero(mask)  
#"""
center_x = positions[0].min() + int((positions[0].max() - positions[0].min())/2)
center_y = positions[1].min() + int((positions[1].max() - positions[1].min())/2)

# crop all images for the patient accordingly
crop_x = int(H/6)
crop_y = int(W/6)
image = images_old[0,0,:,:]
images_old = images_old[:,:,(center_x-crop_x):(center_x+crop_x),(center_y-crop_y):(center_y+crop_y)]
image_crop = images_old[0,0,:,:]
#"""

plt.subplots(figsize=(7, 4))
plt.axis('off')

plt.subplot(1,4,1) 
plt.imshow(image, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout()

plt.subplot(1,4,2) 
plt.imshow(sum_diffs_slices, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout() 

plt.subplot(1,4,3) 
plt.imshow(mask, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout()

plt.subplot(1,4,4) 
plt.imshow(image_crop, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout()

plt.savefig('TestCropOld.png') 
plt.close