from os import mkdir, makedirs
from os.path import isdir, join
from argparse import ArgumentParser
import numpy as np
import torch
from Functions import *
import torch.utils.data as Data
from skimage.io import imsave
import warnings
warnings.filterwarnings("ignore")

"""
Script for generating k-space crops for LAPNet and k-space Fourier-Net
"""

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, 
                    dest="dataset", default="ACDC",
                    help="dataset for training images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--mode", type=int,
                    dest="mode",
                    default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
opt = parser.parse_args()

dataset = opt.dataset
mode = opt.mode

bs = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dataset == 'ACDC':
    # load ACDC test data
    train_set = TrainDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    train_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=4)
    base_path = '/home/jmeyer/storage/students/janmeyer_711878/data/kSpace/ACDC'
elif dataset == 'CMRxRecon':
    # load CMRxRecon test data
    train_set = TrainDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode) 
    train_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=4)
    base_path = '/home/jmeyer/storage/students/janmeyer_711878/data/kSpace/CMRxRecon'
elif dataset == 'OASIS':
    # path for OASIS test dataset
    train_set = TrainDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS') 
    train_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=4)
    base_path = '/home/jmeyer/storage/students/janmeyer_711878/data/kSpace/OASIS'
else:
    raise ValueError('Dataset should be "ACDC", "CMRxRecon" or "OASIS", but found "%s"!' % dataset)

assert mode >= 0 and mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
if mode == 0:
    save_path = join(base_path, 'FullySampled')
elif mode == 1:
    save_path = join(base_path, 'Acc4') 
elif mode == 2:
    save_path = join(base_path, 'Acc8') 
elif mode == 3:
    save_path = join(base_path, 'Acc10')   

print('Begin generating cropped training data...')
save_path = join(save_path, 'Training')

if not isdir(save_path):
    makedirs(save_path)

folders_ignore = [] #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

for i, image_pair in enumerate(train_generator): 
    if i not in folders_ignore:
        # get images
        mov_img = image_pair[0].to(device)
        fix_img = image_pair[1].to(device)
        x = mov_img.shape[2]
        y = mov_img.shape[3]

        # pad M and F for cropping function
        padding = int((33-1)/2)
        padding_array = (padding,padding,padding,padding)
        mov_img = F.pad(mov_img, padding_array, "constant", 0).to(device)
        fix_img = F.pad(fix_img, padding_array, "constant", 0).to(device)

        # FFT to get to k-space domain
        M_temp_fourier_all = torch.view_as_real(torch.fft.fftn(mov_img)).to(device)
        F_temp_fourier_all = torch.view_as_real(torch.fft.fftn(fix_img)).to(device)
        
        # concatenate real and imaginary parts of the k-space into two channels
        M_temp_fourier_concat = torch.cat([M_temp_fourier_all[:,:,:,:,0],M_temp_fourier_all[:,:,:,:,1]],1).to(device)
        F_temp_fourier_concat = torch.cat([F_temp_fourier_all[:,:,:,:,0],F_temp_fourier_all[:,:,:,:,1]],1).to(device)

        # create folder if it does not exist 
        path = join(save_path,'ImagePair{:06d}'.format(i+1))
        if not isdir(path):
            mkdir(path)

        for i in range(x):
            for j in range(y):
                # crop 33x33 patches from the k-spaces
                mov_img, fix_img = crop2D(M_temp_fourier_concat, F_temp_fourier_concat, None, [i,j], 33)

                # save k-space tensors
                torch.save(mov_img, join(path,'MovingImageCrop_{}_{}.pt'.format(i,j)))
                torch.save(fix_img, join(path,'FixedImageCrop_{}_{}.pt'.format(i,j)))

print('...finished!')