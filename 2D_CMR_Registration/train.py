import os
import sys
import argparse
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from Models import *
from Functions import *
import torch.utils.data as Data
import matplotlib.pyplot as plt
import csv
import time
from skimage.metrics import structural_similarity, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=320001,
                    help="number of total iterations")
parser.add_argument("--smth_lambda", type=float,
                    dest="smth_lambda", default=0.02,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=1, 
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon',
                    #default='/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/FullSample', #AccFactor04
                    help="data path for training images")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss",
                    default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2), SSIM (3)")
parser.add_argument("--mode", type=int,
                    dest="mode",
                    default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
parser.add_argument("--F_Net_plus", type=bool,
                    dest="F_Net_plus", default=True, action=argparse.BooleanOptionalAction, 
                    help="choose whether to use Fourier-Net (False) or Fourier-Net+ (True) as the model")
parser.add_argument("--diffeo", type=bool,
                    dest="diffeo", default=True, action=argparse.BooleanOptionalAction, 
                    help="choose whether to use a diffeomorphic transform (True) or not (False)")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
n_checkpoint = opt.checkpoint
smooth = opt.smth_lambda
datapath = opt.datapath
choose_loss = opt.choose_loss
mode = opt.mode
F_Net_plus = opt.F_Net_plus
diffeo = opt.diffeo

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

# choose the model
model_name = 0
if F_Net_plus:
    model = Cascade(2, 2, start_channel).cuda()
    model_name = 1
else:
    model = Fourier_Net(2, 2, start_channel).cuda()  

# choose the loss function for similarity
if choose_loss == 1:
    loss_similarity = MSE().loss
elif choose_loss == 0:
    loss_similarity = SAD().loss
elif choose_loss == 2:
    loss_similarity = NCC(win=9)
elif choose_loss == 3:
    ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1, win_size=9)
    loss_similarity = SAD().loss
loss_smooth = smoothloss

# choose whether to use a diffeomorphic transform or not
diffeo_name = 0
if diffeo:
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    diffeo_name = 1

transform = SpatialTransform().cuda()

for param in transform.parameters():
    param.requires_grad = False
    param.volatile = True

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

lossall = np.zeros((3, iteration))

# load CMR data
train_set = TrainDatasetCMR(datapath, mode) 
training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
validation_set = ValidationDatasetCMR(datapath, mode) 
validation_generator = Data.DataLoader(dataset=validation_set, batch_size=bs, shuffle=True, num_workers=4)

"""
# path for OASIS dataset
datapath = '/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS'
train_set = TrainDataset(datapath,trainingset = 4) 
training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
#valid_set = ValidationDataset(datapath)
#valid_generator = Data.DataLoader(dataset=valid_set, batch_size=bs, shuffle=False, num_workers=2)
"""

model_dir = './ModelParameters/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(model_name,diffeo_name,choose_loss,start_channel,smooth, lr, mode)
model_dir_png = './ModelParameters/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_Smth_{}_LR_{}_Mode_{}_Png/'.format(model_name,diffeo_name,choose_loss,start_channel,smooth, lr, mode)

if not os.path.isdir(model_dir_png):
    os.mkdir(model_dir_png)

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

csv_name = model_dir_png + 'Model_{}_Diffeo_{}_Loss_{}_Chan_{}_Smth_{}_LR_{}_Mode_{}.csv'.format(model_name,diffeo_name,choose_loss,start_channel,smooth, lr, mode)
f = open(csv_name, 'w')
with f:
    fnames = ['Index','MSE','SSIM']
    #fnames = ['Index','Dice']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

step = 1
print('\nStarted training on ', time.ctime())

while step <= iteration:
    if step == 1:
        start = time.time()
    elif step == 2:
        end = time.time()
        print('Expected time for training: ', ((end-start)*(iteration-1))/60, ' minutes.')    
    
    for mov_img, fix_img in training_generator:

        fix_img = fix_img.cuda().float()
        mov_img = mov_img.cuda().float()
        
        f_xy = model(mov_img, fix_img)
        if diffeo:
            Df_xy = diff_transform(f_xy)
        else:
            Df_xy = f_xy
        grid, warped_mov = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
        
        loss1 = loss_similarity(fix_img, warped_mov) 
        loss5 = loss_smooth(Df_xy)
        
        loss = loss1 + smooth * loss5
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lossall[:,step-2] = np.array([loss.item(),loss1.item(),loss5.item()])
        sys.stdout.write("\r" + 'step {0}/'.format(step) + str(iteration) + ' -> training loss "{0:.4f}" - sim "{1:.4f}" -smo "{2:.4f}" '.format(loss.item(),loss1.item(),loss5.item()))
        sys.stdout.flush()
        
        #"""
        if (step % n_checkpoint == 0) or (step==1):
            with torch.no_grad():
                model.eval()
                #Dices_Validation = []
                MSE_Validation = []
                SSIM_Validation = []
                for mov_img, fix_img in validation_generator: #vmov_lab, vfix_lab
                    fix_img = fix_img.cuda().float()
                    mov_img = mov_img.cuda().float()
                    f_xy = model(mov_img, fix_img)
                    if diffeo:
                        Df_xy = diff_transform(f_xy)
                    else:
                        Df_xy = f_xy
                    grid, warped_mov = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
                    # calculate MSE and SSIM
                    MSE_Validation.append(mean_squared_error(warped_mov[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy()))
                    SSIM_Validation.append(structural_similarity(warped_mov[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy(), data_range=1))
                    # change later when Dice-Score is available
                    """
                    grid, warped_vmov_lab = transform(vmov_lab.float().to(device), DV_xy.permute(0, 2, 3, 1), mod = 'nearest')
                    dice_bs = dice(warped_vmov_lab[0,...].data.cpu().numpy().copy(),vfix_lab[0,...].data.cpu().numpy().copy())
                    Dices_Validation.append(dice_bs)
                    """
                """
                modelname = 'DiceVal_{:.4f}_Step_{:06d}.pth'.format(np.mean(Dices_Validation), step)
                csv_dice = np.mean(Dices_Validation)
                f = open(csv_name, 'a')
                with f:
                    writer = csv.writer(f)
                    writer.writerow([step, csv_dice])
                """
                csv_MSE = np.mean(MSE_Validation)
                csv_SSIM = np.mean(SSIM_Validation)
                print('\nmean MSE: ', csv_MSE)
                print('mean SSIM: ', csv_SSIM)
                modelname = 'SSIM_{:.6f}_MSE_{:.6f}_Step_{:06d}.pth'.format(csv_SSIM, csv_MSE, step)
                f = open(csv_name, 'a')
                with f:
                    writer = csv.writer(f)
                    writer.writerow([step, csv_MSE, csv_SSIM])     
                save_checkpoint(model.state_dict(), model_dir, modelname)
                np.save(model_dir_png + 'Loss.npy', lossall)
        #"""
        if (step % n_checkpoint == 0):
            sample_path = os.path.join(model_dir_png, 'Step_{:06d}-images.jpg'.format(step))
            save_flow(mov_img, fix_img, warped_mov, grid.permute(0, 3, 1, 2), sample_path)
            print("one epoch pass")
        step += 1

        if step > iteration:
            break
np.save(model_dir + '/Loss.npy', lossall)
print('\nTraining ended on ', time.ctime())
