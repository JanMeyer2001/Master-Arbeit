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
import argparse

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float,
                    dest="learning_rate", default=1e-4, help="learning rate")
parser.add_argument("--lambda", type=float,
                    dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--dataset", type=str, 
                    dest="dataset", default="ACDC",
                    help="dataset for training images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--choose_loss", type=int, dest="choose_loss", default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2), SSIM (3)")
parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
parser.add_argument("--F_Net_plus", type=bool,
                    dest="F_Net_plus", default=True, action=argparse.BooleanOptionalAction, 
                    help="choose whether to use Fourier-Net (False) or Fourier-Net+ (True) as the model")
parser.add_argument("--diffeo", type=bool,
                    dest="diffeo", default=True, action=argparse.BooleanOptionalAction, 
                    help="choose whether to use a diffeomorphic transform (True) or not (False)")
parser.add_argument("--FT_size_x", type=int,
                    dest="FT_size_x", default=24,
                    help="choose size x of FT crop: Should be smaller than 40.")
parser.add_argument("--FT_size_y", type=int,
                    dest="FT_size_y", default=24,
                    help="choose size y of FT crop: Should be smaller than 84.")
opt = parser.parse_args()

learning_rate = opt.learning_rate
start_channel = opt.start_channel
smooth = opt.smth_lambda
dataset = opt.dataset
choose_loss = opt.choose_loss
mode = opt.mode
F_Net_plus = opt.F_Net_plus
diffeo = opt.diffeo
FT_size = [opt.FT_size_x,opt.FT_size_y]

# choose the model
model_name = 0
if F_Net_plus:
    assert  FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade(2, 2, start_channel, FT_size).cuda() 
    model_name = 1
else:     
    model = Fourier_Net(2, 2, start_channel).cuda()  

# choose whether to use a diffeomorphic transform or not
diffeo_name = 0
if diffeo:
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    diffeo_name = 1

transform = SpatialTransform().cuda()

path = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_name,diffeo_name,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
model_idx = -1
from natsort import natsorted
print('Best model: {}'.format(natsorted(os.listdir(path))[model_idx]))
modelpath = path + natsorted(os.listdir(path))[model_idx]
bs = 1

torch.backends.cudnn.benchmark = True
model.load_state_dict(torch.load(modelpath))
model.eval()
transform.eval()
MSE_test = []
SSIM_test = []
NegJ = []
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
times = []
if dataset != 'CMRxRecon':
    Dice_test_full = []
    Dice_test_noBackground = []

if dataset == 'ACDC':
    # load ACDC test data
    test_set = TestDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
elif dataset == 'CMRxRecon':
    # load CMRxRecon test data
    test_set = TestDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode) 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
elif dataset == 'OASIS':
    # path for OASIS test dataset
    test_set = TestDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS') 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
else:
    raise ValueError('Dataset should be "ACDC", "CMRxRecon" or "OASIS", but found "%s"!' % dataset)

csv_name = './TestResults-{}/TestMetrics-Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}.csv'.format(dataset,model_name,diffeo_name,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
f = open(csv_name, 'w')
with f:
    if dataset == 'CMRxRecon':
        fnames = ['Image Pair','SSIM','MSE','Mean SSIM','Mean MSE','Mean Time','Mean NegJ']
    elif dataset == 'OASIS':
        fnames = ['Image Pair','Dice','SSIM','MSE','Mean Dice','Mean SSIM','Mean MSE','Mean Time','Mean NegJ']
    elif dataset == 'ACDC':
        fnames = ['Image Pair','Dice full','Dice no background','SSIM','MSE','Mean Dice full',' Mean Dice no background','Mean SSIM','Mean MSE','Mean Time','Mean NegJ']    
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

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

        start = time.time()
        # calculate displacement on subsampled data
        if dataset == 'CMRxRecon':
            V_xy = model(mov_img_subSampled.float().to(device), fix_img_subSampled.float().to(device))
        else:
            V_xy = model(mov_img_fullySampled.float().to(device), fix_img_fullySampled.float().to(device))
        
        # get inference time
        inference_time = time.time()-start
        times.append(inference_time)
        
        # but warp fully sampled data
        __, warped_mov_img_fullySampled = transform(mov_img_fullySampled.float().to(device), V_xy.permute(0, 2, 3, 1), mod = 'nearest')
        if dataset != 'CMRxRecon':
            __, warped_mov_seg = transform(mov_seg.float().to(device), V_xy.permute(0, 2, 3, 1), mod = 'nearest')
        
        # calculate MSE, SSIM and Dice 
        if dataset == 'OASIS':
            csv_Dice_full = dice(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy())
        elif dataset == 'ACDC':
            dices_temp = dice_ACDC(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy())
            csv_Dice_full = np.mean(dices_temp)
            csv_Dice_noBackground = np.mean(dices_temp[1:3])
        csv_MSE = mean_squared_error(warped_mov_img_fullySampled[0,0,:,:].cpu().numpy(), fix_img_fullySampled[0,0,:,:].cpu().numpy())
        csv_SSIM = structural_similarity(warped_mov_img_fullySampled[0,0,:,:].cpu().numpy(), fix_img_fullySampled[0,0,:,:].cpu().numpy(), data_range=1)
                  
        MSE_test.append(csv_MSE)
        SSIM_test.append(csv_SSIM)
        if dataset == 'OASIS':
            Dice_test_full.append(csv_Dice_full)
        elif dataset == 'ACDC':
            Dice_test_full.append(csv_Dice_full)
            Dice_test_noBackground.append(csv_Dice_noBackground)
    
        hh, ww = V_xy.shape[-2:]
        V_xy = V_xy.detach().cpu().numpy()
        V_xy[:,0,:,:] = V_xy[:,0,:,:] * hh / 2
        V_xy[:,1,:,:] = V_xy[:,1,:,:] * ww / 2

        jac_det = jacobian_determinant_vxm(V_xy[0, :, :, :])
        negJ = np.sum(jac_det <= 0) / 160 / 192 * 100
        NegJ.append(negJ)

        # save test results to csv file
        f = open(csv_name, 'a')
        with f:
            writer = csv.writer(f)
            if dataset == 'CMRxRecon':
                writer.writerow([i, csv_SSIM, csv_MSE, '-', '-', '-', '-']) 
            elif dataset == 'OASIS':
                writer.writerow([i, csv_Dice_full,csv_MSE, csv_SSIM, '-', '-', '-', '-', '-']) 
            elif dataset == 'ACDC':    
                writer.writerow([i, csv_Dice_full, csv_Dice_noBackground, csv_MSE, csv_SSIM, '-', '-', '-', '-', '-', '-'])

mean_MSE = np.mean(MSE_test)
std_MSE = np.std(MSE_test)

mean_SSIM = np.mean(SSIM_test)
std_SSIM = np.std(SSIM_test)

mean_NegJ = np.mean(NegJ)
std_NegJ = np.std(NegJ)

mean_time = np.mean(times)

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
        writer.writerow(['-', '-', '-', mean_SSIM, mean_MSE, mean_time, mean_NegJ])
    elif dataset == 'OASIS':
        writer.writerow(['-', '-', '-', '-', mean_Dice_full, mean_SSIM, mean_MSE, mean_time, mean_NegJ])
    elif dataset == 'ACDC':
        writer.writerow(['-', '-', '-', '-', mean_Dice_full, mean_Dice_noBackground, mean_SSIM, mean_MSE, mean_time, mean_NegJ])

if dataset == 'CMRxRecon':
    print('Mean inference time: {:.4f} seconds\n     SSIM: {:.5f} +- {:.5f}\n     MSE: {:.6f} +- {:.6f}\n     DetJ<0 %: {:.4f} +- {:.4f}'.format(mean_time, mean_SSIM, std_SSIM, mean_MSE, std_MSE, mean_NegJ, std_NegJ))
elif dataset == 'OASIS':
    print('Mean inference time: {:.4f} seconds\n     DICE: {:.5f} +- {:.5f}\n     SSIM: {:.5f} +- {:.5f}\n     MSE: {:.6f} +- {:.6f}\n     DetJ<0 %: {:.4f} +- {:.4f}'.format(mean_time, mean_Dice_full, std_Dice_full, mean_SSIM, std_SSIM, mean_MSE, std_MSE, mean_NegJ, std_NegJ))
elif dataset == 'ACDC':
    print('Mean inference time: {:.4f} seconds\n     DICE full: {:.5f} +- {:.5f}\n     DICE no background: {:.5f} +- {:.5f}\n     SSIM: {:.5f} +- {:.5f}\n     MSE: {:.6f} +- {:.6f}\n     DetJ<0 %: {:.4f} +- {:.4f}'.format(mean_time, mean_Dice_full, std_Dice_full, mean_Dice_noBackground, std_Dice_noBackground, mean_SSIM, std_SSIM, mean_MSE, std_MSE, mean_NegJ, std_NegJ))
