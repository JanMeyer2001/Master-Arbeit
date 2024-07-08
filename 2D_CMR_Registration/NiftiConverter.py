from os import mkdir
from os.path import isdir, join
from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
import nibabel
import warnings
warnings.filterwarnings("ignore")

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
    test_set = TestDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
    base_path = '/home/jmeyer/storage/students/janmeyer_711878/data/Nifti/ACDC'
elif dataset == 'CMRxRecon':
    # load CMRxRecon test data
    test_set = TestDatasetCMRxReconBenchmark('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode) 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
    base_path = '/home/jmeyer/storage/students/janmeyer_711878/data/Nifti/CMRxRecon'
elif dataset == 'OASIS':
    # path for OASIS test dataset
    test_set = TestDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS') 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
    base_path = '/home/jmeyer/storage/students/janmeyer_711878/data/Nifti/OASIS'
else:
    raise ValueError('Dataset should be "ACDC", "CMRxRecon" or "OASIS", but found "%s"!' % dataset)

assert mode >= 0 and mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
if mode == 0:
    save_path = join(base_path, 'Nifti_FullySampled')
elif mode == 1:
    save_path = join(base_path, 'Nifti_Acc4') 
elif mode == 2:
    save_path = join(base_path, 'Nifti_Acc8') 
elif mode == 3:
    save_path = join(base_path, 'Nifti_Acc10')   

if not isdir(save_path):
    mkdir(save_path)

for i, image_pair in enumerate(test_generator): 
    if dataset == 'CMRxRecon':
        mov_img = image_pair[2][0,0,:,:].cpu().numpy()
        fix_img = image_pair[3][0,0,:,:].cpu().numpy()
    else:
        mov_img = image_pair[0][0,0,:,:].cpu().numpy()
        fix_img = image_pair[1][0,0,:,:].cpu().numpy()    
    
    # convert to Nifit format
    mov_img_nifti = nibabel.Nifti1Image(mov_img, np.eye(4))
    fix_img_nifti = nibabel.Nifti1Image(fix_img, np.eye(4))
    
    # create folder if it does not exist 
    path = join(save_path,'ImagePair{:04d}'.format(i+1))
    if not isdir(path):
        mkdir(path)

    # save Nifti images
    nibabel.save(mov_img_nifti, join(path,'MovingImage'))
    nibabel.save(fix_img_nifti, join(path,'FixedImage'))
