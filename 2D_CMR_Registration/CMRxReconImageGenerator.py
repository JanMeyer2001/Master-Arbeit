import numpy as np
import torch
from os import scandir, mkdir, makedirs
from os.path import join, basename, isdir
import matplotlib.pyplot as plt
from skimage.io import imsave
from argparse import ArgumentParser
import time
from skimage.morphology import area_opening
from skimage.util import img_as_ubyte
from Functions import *

parser = ArgumentParser()

parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose mode: fully sampled (0) or 4x accelerated (1), 8x accelerated (2) and 10x accelerated (3)")
parser.add_argument("--subset", type=int, dest="subset", default='3',
                    help="choose subset: training data (0), validation data (1), test data (2) or all three (3)")
parser.add_argument("--crop", type=int, dest="crop", default='0',
                    help="crop cardiac region (1) or generate full images (0)")
opt = parser.parse_args()

mode = opt.mode
subset = opt.subset
crop = opt.crop

# path for k-space data
path_origin = '/home/jmeyer/storage/staff/ziadalhajhemid/CMRxRecon23/MultiCoil/Cine/'
#path_origin = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/'

# path for image data
path_target = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/'

if subset == 0:
    subsets = ['TrainingSet']
elif subset == 1:
    subsets = ['ValidationSet']
elif subset == 2:
    subsets = ['TestSet']
elif subset == 3:
    subsets = ['TrainingSet', 'ValidationSet', 'TestSet']
else:
    print('Wrong input for set!! Choose either training set (0), validation set (1) or test set (2)')    

print('Started generating image data on ', time.ctime())
for subset in subsets:
    print('Working on subset ', subset)

    assert crop == 0 or crop == 1, f"Expected crop to be one of no cropping (0) or cropping (1), but got: {crop}"
    assert mode >= 0 and mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
    if mode == 0:
        # path for fully sampled CMR k-space data
        data_path = path_origin + subset + '/FullSample'
        # target path for reconstructed images
        if crop == 0:
            image_path = path_target + subset + '/Full/FullySampled/'
        else:    
            image_path = path_target + subset + '/Croped/FullySampled/'
    elif mode == 1:
        # path for accelerated CMR k-space data
        data_path = path_origin + subset + '/AccFactor04'
        # target path for reconstructed images
        if crop == 0:
            image_path = path_target + subset + '/Full/AccFactor04/'
        else:    
            image_path = path_target + subset + '/Croped/AccFactor04/'
    elif mode == 2:
        # path for accelerated CMR k-space data
        data_path = path_origin + subset + '/AccFactor08' 
        # target path for reconstructed images
        if crop == 0:
            image_path = path_target + subset + '/Full/AccFactor08/'
        else:   
            image_path = path_target + subset + '/Croped/AccFactor08/'
    elif mode == 3:
        # path for accelerated CMR k-space data
        data_path = path_origin + subset + '/AccFactor10'
        # target path for reconstructed images
        if crop == 0:
            image_path = path_target + subset + '/Full/AccFactor10/'
        else:    
            image_path = path_target + subset + '/Croped/AccFactor10/'
    else:
        print('Wrong input for mode!! Choose either fully sampled (0), 4x accelerated (1) or 8x accelerated (2)')    

    # get all patient folders
    patients_folders = [f.path for f in scandir(data_path) if f.is_dir() and not (f.name.find('P') == -1)]
    #patients_folders = [join(data_path,'P116')]

    # create dir if not already there 
    if not isdir(image_path):
        makedirs(image_path)

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
        folder = basename(patient_path)
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
                mkdir(join(image_path, folder)+'/')
            
            # Load k-space
            fullmulti = readfile2numpy(join(patient_path, 'cine_sax.mat'),real=False)  
            [frames, slices, ncoil, ny, nx] = fullmulti.shape
            [H, W] = [256, 512]
            images = torch.zeros([slices, frames, H, W]) 
            # get image for every frame/slice
            for slice in range(slices):
                for frame in range(frames):
                    # store all frames for all slices
                    images[slice, frame,:,:] = extract_slice(fullmulti, slice, frame, H, W)
            if mode != 0:
                # and corresponding masks
                mask = readfile2numpy(join(patient_path, 'cine_sax_mask.mat'),real=True)
                mask  = img_as_ubyte(mask)
                imsave(join(image_path, folder) + '/Mask.png', mask, check_contrast=False) 

            if crop == 1:    
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
                if not isdir(join(image_path, folder, subfolder)):
                    mkdir(join(image_path, folder, subfolder)+'/')
                for frame in range(frames):
                    # convert image and mask to uint8
                    image = img_as_ubyte(images[slice,frame,:,:].numpy())
                    # save image and mask to target dir as png
                    imsave(join(image_path, folder, subfolder) + '/Image_Frame' + str(frame) + '.png', image, check_contrast=False) 
            
print('Finished generating image data on ', time.ctime())        
