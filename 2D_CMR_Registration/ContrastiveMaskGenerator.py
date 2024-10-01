import numpy as np
import torch
from os import scandir, mkdir
from os.path import join, basename, isdir
from fastmri.data import transforms as T
from skimage.io import imsave
from argparse import ArgumentParser
import time
from Functions import *

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
elif mode == 2:
    # target path for images
    path_target = path_target + 'AccFactor08/' 
elif mode == 3:
    # target path for images
    path_target = path_target + 'AccFactor10/'
else:
    print('Wrong input for mode!! Choose either fully sampled (0), 4x accelerated (1) or 8x accelerated (2)')    

# get all patient folders
patients_folders = [f.path for f in scandir(path_origin) if f.is_dir() and not (f.name.find('patient') == -1)][0:70]

for i, patient_path in enumerate(patients_folders):
    # get the name of the patient folder with the full path
    folder = basename(patient_path)   
    already_processed = []
    image_path = path_target + 'Training/'
       
    if folder not in already_processed: # block already processed folders
        print('   ', folder)
        
        # get ground truth frames
        gt_files = [f.path for f in scandir(patient_path) if f.is_file() and not (f.name.find('frame') == -1) and not (f.name.find('gt') == -1)]
        masks_frames = []

        for frame_path in gt_files:
            frame = str(basename(frame_path)[-12:-10])
            #if frame == '01':
            # Load ground truth segmentation
            seg_path = [f for f in gt_files if not (f.find(frame) == -1) ]
            nim2 = nib.load(seg_path[0]) 
            segmentation = nim2.get_fdata()
            masks_slices = np.zeros((segmentation.shape[2],216,256))
            
            # get masks for each slice
            for slice in range(segmentation.shape[2]):
                gt   = np.array(segmentation[:,:,slice], dtype='uint8')
            # turn image into a float tensor
                gt = torch.from_numpy(gt).float()
                # interpolate all images to the same size
                gt = F.interpolate(gt.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').squeeze(0).squeeze(0)
                masks_slices[slice,:,:] = np.array(gt, dtype='uint8') > 0

                """
                plt.subplot(1,2,1)
                plt.imshow(gt)
                plt.axis('off')
                plt.subplot(1,2,2)
                plt.imshow(masks[0])
                plt.axis('off')
                plt.savefig('mask_frame{}'.format(frame),bbox_inches='tight')
                plt.close()
                """
            # save masks of slices for the frame
            masks_frames.append(masks_slices)

        # get masks for each slice
        for slice in range(segmentation.shape[2]):
            # ensure that subfolder for each slice exists
            subfolder = 'Slice' + str(slice)
            # get combined mask
            mask = masks_frames[0][slice,:,:].astype(bool) | masks_frames[1][slice,:,:].astype(bool)
            # convert back to binary ints 
            mask = np.array(mask, dtype='uint8')
            """
            plt.subplot(1,3,1)
            plt.imshow(masks_frames[0][slice,:,:])
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(masks_frames[1][slice,:,:])
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(mask)
            plt.axis('off')
            plt.savefig('mask_slice{}'.format(slice),bbox_inches='tight')
            plt.close()
            """
            # save image to target dir as png and gt as nifti file
            imsave(join(image_path, folder, subfolder) + '/Mask.png', mask, check_contrast=False)
      