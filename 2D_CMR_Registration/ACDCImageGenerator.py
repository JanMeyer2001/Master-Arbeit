import numpy as np
import torch
import os
from os import scandir, mkdir
from os.path import join, basename, isdir
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
import nibabel as nib
from mri.operators import FFT

parser = ArgumentParser()

parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose mode: fully sampled (0) or 4x accelerated (1), 8x accelerated (2) and 10x accelerated (3)")
opt = parser.parse_args()
mode = opt.mode


# path for image data
path_origin = '/home/jmeyer/storage/datasets/ACDC'
# path to save data to
path_target = '/home/jmeyer/storage/students/janmeyer_711878/data/ACDC/'

assert mode >= 0 and mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
if mode == 0:
    # target path for images
    path_target = path_target + 'FullySampled/'
elif mode == 1:
    # target path for images
    path_target = path_target + 'AccFactor04/'
    # get Acc4 mask
    mask_func = EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[4]) 
elif mode == 2:
    # target path for images
    path_target = path_target + 'AccFactor08/'
    # get Acc8 mask
    mask_func = EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[8]) 
elif mode == 3:
    # target path for images
    path_target = path_target + 'AccFactor10/'
    # get Acc10 mask
    mask_func = EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[10]) 
else:
    print('Wrong input for mode!! Choose either fully sampled (0), 4x accelerated (1) or 8x accelerated (2)')    

# get all patient folders
patients_folders = [f.path for f in scandir(path_origin) if f.is_dir() and not (f.name.find('patient') == -1)]

print('Started generating image data on ', time.ctime())

for i, patient_path in enumerate(patients_folders):
    # get the name of the patient folder with the full path
    folder = basename(patient_path)   
    already_processed = []

    # take 70 patients as training data
    if i == 0:
        print(' Generating training data\n  working on folder: ')
        image_path = path_target + 'Training/'
    # take 10 patients as validation data
    elif i == 70:
        print(' Generating validation data\n  working on folder: ')
        image_path = path_target + 'Validation/'
    # take 20 patients as test data
    elif i == 80:
        print(' Generating test data\n  working on folder: ')
        image_path = path_target + 'Test/'
    # create dir if not already there 
    if not isdir(image_path):
        mkdir(image_path)
       
    if folder not in already_processed: # block already processed folders
        print('   ', folder)
        
        # ensure that all target folders exist
        if not isdir(join(image_path, folder)):
            mkdir(join(image_path, folder)+'/')
        
        if 0 <= i < 70:
            file_name = folder+'_4d.nii.gz'
            file_path = join(patient_path, file_name)
            # Load 4D image data
            nim1 = nib.load(file_path) 
            volume = nim1.get_fdata() 

            for frame in range(volume.shape[3]):
                # save images to the corresponding folders
                for slice in range(volume.shape[2]):
                    # ensure that subfolder for each slice exists
                    subfolder = 'Slice' + str(slice)
                    if not isdir(join(image_path, folder, subfolder)):
                        mkdir(join(image_path, folder, subfolder)+'/')
                    
                    image = volume[:,:,slice,frame]
                    image = np.array(255*image/image.max(), dtype='uint8')
                    
                    # manually subsample image for accelerated data
                    if mode != 0:
                        # apply fft to get k-space
                        k_space_fullysampled = torch.fft.fftn(torch.from_numpy(image))                             
                        # apply mask for subsampling
                        k_space_subsampled, mask, _ = T.apply_mask(k_space_fullysampled.unsqueeze(0).unsqueeze(0), mask_func)     
                        # use pySAP for reconstruction
                        fourier_op = FFT(mask=mask, shape=image.shape, n_coils=1)
                        kspace_obs = fourier_op.op(image)
                        # Zero order solution
                        image = np.linalg.norm(fourier_op.adj_op(kspace_obs), axis=0)[0,:,:]
                        image = np.array(image*255, dtype='uint8')   

                    # save image to target dir as png and gt as nifti file
                    imsave(join(image_path, folder, subfolder) + '/Image_Frame{0:02d}.png'.format(frame), image, check_contrast=False) 
                
        else:
            # get all frame files
            all_files = [f.path for f in scandir(patient_path) if f.is_file() and not (f.name.find('frame') == -1)]
            # get ground truth frames
            gt_files = [f.path for f in scandir(patient_path) if f.is_file() and not (f.name.find('frame') == -1) and not (f.name.find('gt') == -1)]
            # get image frames
            frame_files = [f for f in all_files if f not in gt_files]

            for frame_path in frame_files:
                frame = str(basename(frame_path)[-9:-7])
                
                # Load image data
                nim1 = nib.load(frame_path) 
                volume = nim1.get_fdata()
                
                # Load ground truth segmentation
                seg_path = [f for f in gt_files if not (f.find(frame) == -1) ]
                nim2 = nib.load(seg_path[0]) 
                segmentation = nim2.get_fdata()
                
                # save images to the corresponding folders
                for slice in range(volume.shape[2]):
                    # ensure that subfolder for each slice exists
                    subfolder = 'Slice' + str(slice)
                    if not isdir(join(image_path, folder, subfolder)):
                        mkdir(join(image_path, folder, subfolder)+'/')
                    
                    image = volume[:,:,slice]
                    image = np.array(255*image/image.max(), dtype='uint8')
                    gt = np.array(segmentation[:,:,slice], dtype='uint8')

                # manually subsample image for accelerated data
                if mode != 0:
                    # apply fft to get k-space
                    k_space_fullysampled = torch.fft.fftn(torch.from_numpy(image))                             
                    # apply mask for subsampling
                    k_space_subsampled, mask, _ = T.apply_mask(k_space_fullysampled.unsqueeze(0).unsqueeze(0), mask_func)     
                    # use pySAP for reconstruction
                    fourier_op = FFT(mask=mask, shape=image.shape, n_coils=1)
                    kspace_obs = fourier_op.op(image)
                    # Zero order solution
                    image = np.linalg.norm(fourier_op.adj_op(kspace_obs), axis=0)[0,:,:] 
                    image = np.array(image*255, dtype='uint8')   

                # save image to target dir as png and gt as nifti file
                imsave(join(image_path, folder, subfolder) + '/Image_Frame' + frame + '.png', image, check_contrast=False) # + '_Slice' + str(slice)
                imsave(join(image_path, folder, subfolder) + '/Segmentation_Frame' + frame + '.png', gt, check_contrast=False) # + '_Slice' + str(slice)
                #nibabel.save(gt, join(image_path, folder, subfolder) + '/Segmentation_Frame' + str(basename(frame_path)[-2]))
                
print('Finished generating image data on ', time.ctime())        
