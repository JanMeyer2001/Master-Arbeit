import os
from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
import csv
import time
from skimage.metrics import structural_similarity, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import nibabel

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, 
                    dest="dataset", default="ACDC",
                    help="dataset for training images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--mode", type=int, dest="mode", default='0',
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
    path = join(base_path, 'Nifti_FullySampled')
    print('Start TestBenchmark for fully sampled data')
elif mode == 1:
    path = join(base_path, 'Nifti_Acc4') 
    print('Start TestBenchmark for Acc4 data')
elif mode == 2:
    path = join(base_path, 'Nifti_Acc8')
    print('Start TestBenchmark for Acc8 data') 
elif mode == 3:
    path = join(base_path, 'Nifti_Acc10') 
    print('Start TestBenchmark for Acc10 data')

transform = SpatialTransform().cuda()

Dices=[]
MSE_test = []
SSIM_test = []
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

csv_name = './TestResults-{}/TestMetrics-NiftyReg_Mode{}.csv'.format(dataset,mode)
f = open(csv_name, 'w')
with f:
    if dataset == 'CMRxRecon':
        fnames = ['Image Pair','SSIM','MSE','Mean SSIM','Mean MSE']
    else:
        fnames = ['Image Pair','Dice','SSIM','MSE','Mean_Dice','Mean SSIM','Mean MSE']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

image_pairs = [basename(f.path) for f in scandir(path) if f.is_dir() and f.name.startswith('ImagePair')]

for image_pair in image_pairs:
    if mode == 0:
        # read in images from folder 
        warped_img  = nibabel.load(join(path, image_pair, 'WarpedImage.nii'))
        fix_img     = nibabel.load(join(path, image_pair, 'FixedImage.nii'))
        # convert to float array
        warped_img  = np.array(warped_img.get_fdata(), dtype='float32')
        fix_img     = np.array(fix_img.get_fdata(), dtype='float32')
    else:   
        # read in resampled warped image from the subsampled folder
        warped_fully_sampled = nibabel.load(join(path, image_pair, 'WarpedImage_FullySampled.nii'))
        warped_img = np.array(warped_fully_sampled.get_fdata(), dtype='float32')
        # get fully sampled image pairs from test dataset
        mov_img, fix_img,_ ,_ = test_set.__getitem__(int(image_pair.replace('ImagePair', ''))-1)
        # convert to numpy
        fix_img = fix_img.squeeze().cpu().numpy()
    
    if dataset != 'CMRxRecon':
        # read in segmentations from folder 
        warped_seg  = nibabel.load(join(path, image_pair, 'WarpedSegmentation.nii'))
        fix_seg     = nibabel.load(join(path, image_pair, 'FixedSegmentation.nii'))
        # convert to float array
        warped_seg  = np.array(warped_seg.get_fdata(), dtype='float32')
        fix_seg     = np.array(fix_seg.get_fdata(), dtype='float32')

    # calculate metrics 
    csv_MSE = mean_squared_error(warped_img, fix_img)
    csv_SSIM = structural_similarity(warped_img, fix_img, data_range=1)
    if dataset == 'ACDC':
        csv_DICE = dice_ACDC(warped_seg,fix_seg)
    elif dataset == 'OASIS':
        csv_DICE = dice(warped_seg,fix_seg)    

    MSE_test.append(csv_MSE)
    SSIM_test.append(csv_SSIM)
    if dataset != 'CMRxRecon':
        Dices.append(csv_DICE)
    
    # save to csv file
    f = open(csv_name, 'a')
    with f:
        writer = csv.writer(f)
        if dataset == 'CMRxRecon':
            writer.writerow([image_pair[-1], csv_SSIM, csv_MSE, '-', '-'])  
        else:
            writer.writerow([image_pair[-1], csv_DICE, csv_SSIM, csv_MSE, '-', '-', '-']) 

mean_MSE = np.mean(MSE_test)
std_MSE = np.std(MSE_test)

mean_SSIM = np.mean(SSIM_test)
std_SSIM = np.std(SSIM_test)

if dataset != 'CMRxRecon':
    mean_DICE = np.mean(Dices)
    std_DICE = np.std(Dices)

f = open(csv_name, 'a')
with f:
    writer = csv.writer(f)
    if dataset == 'CMRxRecon':
        writer.writerow(['-', '-', '-', mean_SSIM, mean_MSE])  
    else:
        writer.writerow(['-', '-', '-', '-', mean_DICE, mean_SSIM, mean_MSE])  

if dataset == 'CMRxRecon':
    print('MSE: {:.6f} +- {:.6f}\nSSIM: {:.5f} +- {:.5f}'.format(mean_SSIM,std_SSIM,mean_MSE,std_MSE))
else:
    print('DICE: {:.5f} +- {:.5f}\nSSIM: {:.5f} +- {:.5f}\nMSE: {:.6f} +- {:.6f}'.format(mean_DICE,std_DICE,mean_SSIM,std_SSIM,mean_MSE,std_MSE))
    