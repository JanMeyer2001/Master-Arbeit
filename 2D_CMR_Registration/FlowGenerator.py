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
Script for generating image pair to be generate a ground truth flow field for LAPNet
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
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
if dataset == 'ACDC':
    # load ACDC test data
    train_set = TrainDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    train_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=4)
    base_path = '/home/jmeyer/storage/students/janmeyer_711878/data/Flow/ACDC'
elif dataset == 'CMRxRecon':
    # load CMRxRecon test data
    train_set = TrainDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode) 
    train_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=4)
    base_path = '/home/jmeyer/storage/students/janmeyer_711878/data/Flow/CMRxRecon'
elif dataset == 'OASIS':
    # path for OASIS test dataset
    train_set = TrainDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS') 
    train_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=False, num_workers=4)
    base_path = '/home/jmeyer/storage/students/janmeyer_711878/data/Flow/OASIS'
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

if not isdir(save_path):
    makedirs(save_path)

print('Begin generating training data...')

for i, image_pair in enumerate(train_generator): 
    # get images
    mov_img = image_pair[0][0,0,:,:].cpu().numpy()
    mov_img = np.array(mov_img*255, dtype='uint8')  
    fix_img = image_pair[1][0,0,:,:].cpu().numpy()  
    fix_img = np.array(fix_img*255, dtype='uint8')  

    # create folder if it does not exist 
    path = join(save_path,'ImagePair{:06d}'.format(i+1))
    if not isdir(path):
        mkdir(path)

    # save images for flow field generation
    imsave(join(path,'MovingImage.png'), mov_img, check_contrast=False)
    imsave(join(path,'FixedImage.png'), fix_img, check_contrast=False)

print('...finished!')