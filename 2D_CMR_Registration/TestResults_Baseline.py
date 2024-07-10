from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
import csv
from skimage.metrics import structural_similarity, mean_squared_error

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, dest="dataset", default="ACDC",
                    help="dataset for training images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
opt = parser.parse_args()

dataset = opt.dataset
mode = opt.mode

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

bs = 1
if dataset == 'ACDC':
    # load ACDC test data
    test_set = TestDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
elif dataset == 'CMRxRecon':
    # load CMRxRecon test data
    test_set = TestDatasetCMRxReconBenchmark('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode) 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
elif dataset == 'OASIS':
    # path for OASIS test dataset
    test_set = TestDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS') 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
else:
    raise ValueError('Dataset should be "ACDC", "CMRxRecon" or "OASIS", but found "%s"!' % dataset)


csv_name = './TestResults-{}/TestMetrics-Baseline_Mode{}.csv'.format(dataset,mode)
f = open(csv_name, 'w')
with f:
    if dataset == 'CMRxRecon':
        fnames = ['Image Pair','SSIM','MSE','Mean SSIM','Mean MSE']
    elif dataset == 'OASIS':
        fnames = ['Image Pair','Dice','SSIM','MSE','Mean Dice','Mean SSIM','Mean MSE']
    elif dataset == 'ACDC':
        fnames = ['Image Pair','Dice full','Dice no background','SSIM','MSE','Mean Dice full',' Mean Dice no background','Mean SSIM','Mean MSE']    
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

MSE_Test = []
SSIM_Test = []
if dataset != 'CMRxRecon':
    Dice_test_full = []
    Dice_test_noBackground = []

for i, image_pairs in enumerate(test_generator): 
    with torch.no_grad():
        mov_img_fullySampled = image_pairs[0]
        fix_img_fullySampled = image_pairs[1]
        if dataset == 'CMRxRecon':
            mov_img_subSampled = image_pairs[2]
            fix_img_subSampled = image_pairs[3]
        else:
            mov_seg = image_pairs[2]
            fix_seg = image_pairs[3]

        # calculate MSE, SSIM and Dice 
        if dataset == 'OASIS':
            csv_Dice_full = dice(mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy())
        elif dataset == 'ACDC':
            dices_temp = dice_ACDC(mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy())
            csv_Dice_full = np.mean(dices_temp)
            csv_Dice_noBackground = np.mean(dices_temp[1:3])
        if dataset == 'CMRxRecon':
            csv_SSIM = structural_similarity(mov_img_subSampled[0,0,:,:].cpu().numpy(), fix_img_subSampled[0,0,:,:].cpu().numpy(), data_range=1)
            csv_MSE = mean_squared_error(mov_img_subSampled[0,0,:,:].cpu().numpy(), fix_img_subSampled[0,0,:,:].cpu().numpy()) 
        else:        
            csv_MSE = mean_squared_error(mov_img_fullySampled[0,0,:,:].cpu().numpy(), fix_img_fullySampled[0,0,:,:].cpu().numpy())
            csv_SSIM = structural_similarity(mov_img_fullySampled[0,0,:,:].cpu().numpy(), fix_img_fullySampled[0,0,:,:].cpu().numpy(), data_range=1)
                  
        SSIM_Test.append(csv_SSIM)
        MSE_Test.append(csv_MSE)
        if dataset == 'OASIS':
            Dice_test_full.append(csv_Dice_full)
        elif dataset == 'ACDC':
            Dice_test_full.append(csv_Dice_full)
            Dice_test_noBackground.append(csv_Dice_noBackground)

        # save to csv file
        f = open(csv_name, 'a')
        with f:
            writer = csv.writer(f)
            if dataset == 'CMRxRecon':
                writer.writerow([i, csv_SSIM, csv_MSE, '-', '-']) 
            elif dataset == 'OASIS':
                writer.writerow([i, csv_Dice_full,csv_MSE, csv_SSIM, '-', '-', '-']) 
            elif dataset == 'ACDC':    
                writer.writerow([i, csv_Dice_full, csv_Dice_noBackground, csv_MSE, csv_SSIM, '-', '-', '-', '-'])  

mean_MSE = np.mean(MSE_Test)
std_MSE = np.std(MSE_Test)

mean_SSIM = np.mean(SSIM_Test)
std_SSIM = np.std(SSIM_Test)

if dataset == 'OASIS':
    mean_Dice_full = np.mean(Dice_test_full)
    std_Dice_full = np.std(Dice_test_full)
elif dataset == 'ACDC':
    mean_Dice_full = np.mean(Dice_test_full)
    std_Dice_full = np.std(Dice_test_full)
    mean_Dice_noBackground = np.mean(Dice_test_noBackground)
    std_Dice_noBackground = np.std(Dice_test_noBackground)

f = open(csv_name, 'a')
with f:
    writer = csv.writer(f)
    if dataset == 'CMRxRecon':
        writer.writerow(['-', '-', '-', mean_SSIM, mean_MSE])
    elif dataset == 'OASIS':
        writer.writerow(['-', '-', '-', '-', mean_Dice_full, mean_SSIM, mean_MSE])
    elif dataset == 'ACDC':
        writer.writerow(['-', '-', '-', '-', mean_Dice_full, mean_Dice_noBackground, mean_SSIM, mean_MSE])  

if dataset == 'CMRxRecon':
    print('SSIM: {:.5f} +- {:.5f}\nMSE: {:.6f} +- {:.6f}'.format(mean_SSIM,std_SSIM,mean_MSE,std_MSE))
elif dataset == 'OASIS':
    print('DICE: {:.5f} +- {:.5f}\nSSIM: {:.5f} +- {:.5f}\nMSE: {:.6f} +- {:.6f}\n'.format(mean_Dice_full, std_Dice_full, mean_SSIM, std_SSIM, mean_MSE, std_MSE))
elif dataset == 'ACDC':
    print('DICE full: {:.5f} +- {:.5f}\nDICE no background: {:.5f} +- {:.5f}\nSSIM: {:.5f} +- {:.5f}\nMSE: {:.6f} +- {:.6f}'.format(mean_Dice_full, std_Dice_full, mean_Dice_noBackground, std_Dice_noBackground, mean_SSIM, std_SSIM, mean_MSE, std_MSE))
