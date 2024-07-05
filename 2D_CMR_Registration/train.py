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
parser.add_argument("--epochs", type=int,
                    dest="epochs", default=50,
                    help="number of epochs")
parser.add_argument("--lambda", type=float,
                    dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--dataset", type=str,
                    dest="dataset",
                    default="ACDC",
                    help="dataset for training images: Select either ACDC, CMRxRecon or OASIS")
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
parser.add_argument("--earlyStop", type=int,
                    dest="earlyStop", default=3,
                    help="choose after how many epochs early stopping is applied")
opt = parser.parse_args()

learning_rate = opt.learning_rate
epochs = opt.epochs
start_channel = opt.start_channel
smth_lambda = opt.smth_lambda
dataset = opt.dataset
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

if dataset == 'ACDC':
    # load ACDC data
    train_set = TrainDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=True, num_workers=4)
elif dataset == 'CMRxRecon':
    # load CMRxRecon data
    train_set = TrainDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=True, num_workers=4)
elif dataset == 'OASIS':
    # path for OASIS dataset
    train_set = TrainDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS',trainingset = 4) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS')
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=False, num_workers=2)
else:
    raise ValueError('Dataset should be "ACDC", "CMRxRecon" or "OASIS", but found "%s"!' % dataset)

model_dir = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_name,diffeo_name,choose_loss,start_channel,FT_size[0],FT_size[1],smth_lambda, learning_rate, mode)
model_dir_png = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Png/'.format(dataset,model_name,diffeo_name,choose_loss,start_channel,FT_size[0],FT_size[1],smth_lambda, learning_rate, mode)

if not isdir(model_dir_png):
    mkdir(model_dir_png)

if not isdir(model_dir):
    mkdir(model_dir)

csv_name = model_dir_png + 'Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}.csv'.format(model_name,diffeo_name,choose_loss,start_channel,FT_size[0],FT_size[1],smth_lambda,learning_rate,mode)
f = open(csv_name, 'w')
with f:
    if dataset == 'CMRxRecon':
        fnames = ['Epoch','MSE','SSIM'] #'Dice',
    else:
        fnames = ['Epoch','Dice','MSE','SSIM']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

##############
## Training ##
##############

# counter and best SSIM for early stopping
counter_earlyStopping = 0
if dataset == 'CMRxRecon':
    best_SSIM = 0
else:
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
        if dataset != 'CMRxRecon':
            Dice_Validation = []
        
        for i, image_pair in enumerate(validation_generator): 
            fix_img = image_pair[0].cuda().float()
            mov_img = image_pair[1].cuda().float()
            
            if dataset != 'CMRxRecon':
                mov_seg = image_pair[2].cuda().float()
                fix_seg = image_pair[3].cuda().float()
            
            f_xy = model(mov_img, fix_img)
            if diffeo:
                Df_xy = diff_transform(f_xy)
            else:
                Df_xy = f_xy

            # get warped image and segmentation
            grid, warped_mov_img = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
            if dataset != 'CMRxRecon':
                grid, warped_mov_seg = transform(mov_seg, Df_xy.permute(0, 2, 3, 1))
            
            # calculate MSE, SSIM and Dice 
            MSE_Validation.append(mean_squared_error(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy()))
            SSIM_Validation.append(structural_similarity(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy(), data_range=1))
            if dataset == 'OASIS':
                Dice_Validation.append(dice(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy()))
            elif dataset == 'ACDC':
                Dice_Validation.append(dice_ACDC(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy()))
    
        # calculate mean of validation metrics
        Mean_MSE = np.mean(MSE_Validation)
        Mean_SSIM = np.mean(SSIM_Validation)
        if dataset != 'CMRxRecon':
            Mean_Dice = np.mean(Dice_Validation)

        f = open(csv_name, 'a')
        with f:
            writer = csv.writer(f)
            if dataset == 'CMRxRecon':
                writer.writerow([epoch, Mean_MSE, Mean_SSIM]) 
            else:
                writer.writerow([epoch, Mean_Dice, Mean_MSE, Mean_SSIM]) 

        # save best metrics and reset counter if Dice/SSIM got better, else increase counter for early stopping
        if dataset == 'CMRxRecon':
            if Mean_SSIM>best_SSIM:
                best_SSIM = Mean_SSIM
                counter_earlyStopping = 0
            else:
                counter_earlyStopping += 1   
        else:
            if Mean_Dice>best_Dice:
                best_Dice = Mean_Dice
                counter_earlyStopping = 0
            else:
                counter_earlyStopping += 1    
        
        # save model     
        if dataset == 'CMRxRecon':
            modelname = 'SSIM_{:.5f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_SSIM, Mean_MSE, epoch)
        else:
            modelname = 'DICE_{:.5f}_SSIM_{:.5f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_Dice, Mean_SSIM, Mean_MSE, epoch)
        save_checkpoint(model.state_dict(), model_dir, modelname)
        
        # save image
        sample_path = join(model_dir_png, 'Epoch_{:04d}-images.jpg'.format(epoch))
        save_flow(mov_img, fix_img, warped_mov_img, grid.permute(0, 3, 1, 2), sample_path)
        if dataset == 'CMRxRecon':
            print("epoch {:d}/{:d} - SSIM_val: {:.5f}, MSE_val: {:.6f}".format(epoch+1, epochs, Mean_SSIM, Mean_MSE))
        else:
            print("epoch {:d}/{:d} - DICE_val: {:.5f}, SSIM_val: {:.5f}, MSE_val: {:.6f}".format(epoch+1, epochs, Mean_Dice, Mean_SSIM, Mean_MSE))

        # stop training if metrics stop improving for three epochs (only on the first run)
        if counter_earlyStopping >= earlyStop:  
            break
            
print('Training ended on ', time.ctime())
