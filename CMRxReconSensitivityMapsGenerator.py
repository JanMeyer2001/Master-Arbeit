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
import sigpy.mri as mr
from Functions import *

parser = ArgumentParser()

parser.add_argument("--mode", type=int, dest="mode", default='1',
                    help="choose mode: fully sampled (0) or 4x accelerated (1), 8x accelerated (2) and 10x accelerated (3)")
parser.add_argument("--subset", type=int, dest="subset", default='3',
                    help="choose subset: training data (0), validation data (1), test data (2) or all three (3)")
opt = parser.parse_args()

mode = opt.mode
subset = opt.subset

# path for k-space data
path_origin = '/home/jmeyer/storage/staff/ziadalhajhemid/CMRxRecon23/MultiCoil/Cine/'

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

    assert mode >= 0 and mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
    if mode == 0:
        data_path = path_origin + subset + '/FullSample'
        image_path = path_target + subset + '/Full/FullySampled/'
    elif mode == 1:
        data_path = path_origin + subset + '/AccFactor04'
        image_path = path_target + subset + '/Full/AccFactor04/'
    elif mode == 2:
        data_path = path_origin + subset + '/AccFactor08' 
        image_path = path_target + subset + '/Full/AccFactor08/'
    elif mode == 3:
        data_path = path_origin + subset + '/AccFactor10'
        image_path = path_target + subset + '/Full/AccFactor10/'
    else:
        print('Wrong input for mode!! Choose either fully sampled (0), 4x accelerated (1) or 8x accelerated (2)')    

    # get all patient folders
    patients_folders = [f.path for f in scandir(data_path) if f.is_dir() and not (f.name.find('P') == -1)]
    
    # create dir if not already there 
    if not isdir(image_path):
        makedirs(image_path)

    print('  working on folder: ')

    for i, patient_path in enumerate(patients_folders):
        if i == 0:
            start = time.time()
        elif i == 1:
            end = time.time()
            print('Expected time remaining: ', ((end-start)*(len(patients_folders)-1))/60, ' minutes.') 
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
                
            # save images to the corresponding folders
            for slice in range(slices):
                # ensure that subfolder for each slice exists
                subfolder = 'Slice' + str(slice)
                if not isdir(join(image_path, folder, subfolder)):
                    mkdir(join(image_path, folder, subfolder)+'/')
                for frame in range(frames):
                    k_space = fullmulti[frame, slice]
                    k_space = T.to_tensor(k_space) # [n,x,y,c] --> n channels, size [x,y], c for real and complex values
                    torch.save(k_space, join(image_path, folder, subfolder) + '/k-space_Frame' + str(frame) + '.pt') 
                    
                    # Derive ESPIRiT sensitivity maps
                    k_space = torch.view_as_complex(k_space).squeeze().numpy()
                    maps    = np.fft.fftshift(mr.app.EspiritCalib(k_space, calib_width=24, thresh=0.02, kernel_width=6, crop=0.01, max_iter=100, show_pbar=False).run(),axes=(1, 2))
                    """
                    for idx in range(maps.shape[2]):
                        plt.subplot(1, maps.shape[2], idx + 1)
                        plt.imshow(np.abs(maps[:,:,idx]), cmap='gray')
                        plt.axis('off')
                    plt.show()
                    plt.savefig('ESPIRiT-Maps.png') # for saving the image
                    plt.close
                    """
                    torch.save(maps, join(image_path, folder, subfolder) + '/SensitivityMaps_Frame' + str(frame) + '.pt') 
            
print('Finished generating data on ', time.ctime())        
