import os
from argparse import ArgumentParser, BooleanOptionalAction
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
parser.add_argument("--learning_rate", type=float,
                    dest="learning_rate", default=1e-4, help="learning rate")
parser.add_argument("--iterations", type=int,
                    dest="epochs", default=100,
                    help="number of epochs")
parser.add_argument("--lambda", type=float,
                    dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/home/jmeyer/storage/students/janmeyer_711878/data/ACDC',
                    #default='/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon',
                    help="data path for training images")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss", default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2) or SSIM (3)")
parser.add_argument("--mode", type=int,
                    dest="mode", default=0,
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
parser.add_argument("--F_Net_plus", type=bool,
                    dest="F_Net_plus", default=True, action=BooleanOptionalAction, 
                    help="choose whether to use Fourier-Net (False) or Fourier-Net+ (True) as the model")
parser.add_argument("--diffeo", type=bool,
                    dest="diffeo", default=True, action=BooleanOptionalAction, 
                    help="choose whether to use a diffeomorphic transform (True) or not (False)")
parser.add_argument("--FT_size", type=tuple,
                    dest="FT_size", default=[24,24],
                    help="choose size of FT crop: Should be smaller than [40,84].")
parser.add_argument("--earlyStop", type=bool,
                    dest="earlyStop", default=True, action=BooleanOptionalAction, 
                    help="choose whether to use early stopping to prevent overfitting (True) or not (False)")
opt = parser.parse_args()

learning_rate = opt.learning_rate
epochs = opt.epochs
start_channel = opt.start_channel
smth_lambda = opt.smth_lambda
datapath = opt.datapath
choose_loss = opt.choose_loss
mode = opt.mode
F_Net_plus = opt.F_Net_plus
diffeo = opt.diffeo
FT_size = opt.FT_size
earlyStop = opt.earlyStop

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

# choose the model
model_name = 0
if F_Net_plus:
    assert  FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade(2, 2, start_channel, FT_size).cuda() 
    model_name = 1
else:
    model = Fourier_Net(2, 2, start_channel).cuda()  

# choose the loss function for similarity
assert choose_loss >= 0 and choose_loss <= 3, f"Expected choose_loss to be one of SAD (0), MSE (1), NCC (2) or SSIM (3), but got: {choose_loss}"
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

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

assert mode >= 0 and mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"

# load ACDC data
train_set = TrainDatasetACDC(datapath, mode) 
training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
validation_set = ValidationDatasetACDC(datapath, mode) 
validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=True, num_workers=4)

"""
# load CMRxRecon data
train_set = TrainDatasetCMRxRecon(datapath, mode) 
training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
validation_set = ValidationDatasetCMRxRecon(datapath, mode) 
validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=True, num_workers=4)
"""

"""
# path for OASIS dataset
datapath = '/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS'
train_set = TrainDataset(datapath,trainingset = 4) 
training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
#valid_set = ValidationDataset(datapath)
#valid_generator = Data.DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=2)
"""

model_dir = './ModelParameters/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(model_name,diffeo_name,choose_loss,start_channel,FT_size[0],FT_size[1],smth_lambda, learning_rate, mode)
model_dir_png = './ModelParameters/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Png/'.format(model_name,diffeo_name,choose_loss,start_channel,FT_size[0],FT_size[1],smth_lambda, learning_rate, mode)

if not os.path.isdir(model_dir_png):
    os.mkdir(model_dir_png)

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

csv_name = model_dir_png + 'Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}.csv'.format(model_name,diffeo_name,choose_loss,start_channel,FT_size[0],FT_size[1],smth_lambda,learning_rate,mode)
f = open(csv_name, 'w')
with f:
    fnames = ['Epoch','Dice','MSE','SSIM'] #
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

##############
## Training ##
##############

if earlyStop:
    # counter and best SSIM for early stopping
    counter_earlyStopping = 0
    best_Dice = 0


print('\nStarted training on ', time.ctime())

for epoch in range(epochs):
    losses = np.zeros(training_generator.__len__())
    for i, image_pair in enumerate(training_generator):
        mov_img = image_pair[0].cuda().float()
        fix_img = image_pair[1].cuda().float()
        
        f_xy = model(mov_img, fix_img)
        if diffeo:
            Df_xy = diff_transform(f_xy)
        else:
            Df_xy = f_xy
        grid, warped_mov = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
        
        loss1 = loss_similarity(fix_img, warped_mov) 
        loss5 = loss_smooth(Df_xy)
        
        loss = loss1 + smth_lambda * loss5
        losses[i] = loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ################
    ## Validation ##
    ################

    with torch.no_grad():
        model.eval()
        MSE_Validation = []
        SSIM_Validation = []
        Dice_Validation = []
        
        for mov_img, fix_img, mov_seg, fix_seg in validation_generator: 
            fix_img = fix_img.cuda().float()
            mov_img = mov_img.cuda().float()
            mov_seg = mov_seg.cuda().float()
            fix_seg = fix_seg.cuda().float()
            
            f_xy = model(mov_img, fix_img)
            if diffeo:
                Df_xy = diff_transform(f_xy)
            else:
                Df_xy = f_xy
            # get warped image and segmentation
            grid, warped_mov_img = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
            grid, warped_mov_seg = transform(mov_seg, Df_xy.permute(0, 2, 3, 1))
            
            # calculate MSE, SSIM and Dice 
            MSE_Validation.append(mean_squared_error(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy()))
            SSIM_Validation.append(structural_similarity(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy(), data_range=1))
            Dice_Validation.append(dice(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy()))
    
        # calculate mean of validation metrics
        Mean_MSE = np.mean(MSE_Validation)
        Mean_SSIM = np.mean(SSIM_Validation)
        Mean_Dice = np.mean(Dice_Validation)

        f = open(csv_name, 'a')
        with f:
            writer = csv.writer(f)
            writer.writerow([epoch, Mean_Dice, Mean_MSE, Mean_SSIM]) 

        if earlyStop:
            # save best metrics and reset counter if Dice got better, else increase counter for early stopping
            if Mean_Dice>best_Dice:
                best_Dice = Mean_Dice
                counter_earlyStopping = 0
            else:
                counter_earlyStopping += 1    
        
        # save and log model     
        modelname = 'DICE_{:.5f}_SSIM_{:.5f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_Dice,Mean_SSIM, Mean_MSE, epoch)
        save_checkpoint(model.state_dict(), model_dir, modelname)
        
        # save image
        sample_path = join(model_dir_png, 'Epoch_{:04d}-images.jpg'.format(epoch))
        save_flow(mov_img, fix_img, warped_mov, grid.permute(0, 3, 1, 2), sample_path)
        print("epoch {:d}/{:d} - DICE_val: {:.5f} MSE_val: {:.6f}, SSIM_val: {:.5f}".format(epoch, epochs, Mean_Dice, Mean_MSE, Mean_SSIM))

        # stop training if metrics stop improving for three epochs (only on the first run)
        if counter_earlyStopping == 3:
            break
            
    print('Training ended on ', time.ctime())
