import numpy as np
import torch
import wandb.sync
from Models import *
from Functions import *
import torch.utils.data as Data
import time
from skimage.metrics import structural_similarity, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
from os import mkdir
from os.path import isdir, join
import os

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float,
                    dest="learning_rate", default=1e-4, help="learning rate")
parser.add_argument("--epochs", type=int, dest="epochs", default=50,
                    help="number of total epochs")
parser.add_argument("--lambda", type=float, dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--dataset", type=str,
                    dest="dataset", default='ACDC',
                    help="dataset for training images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--mode", type=int, dest="mode", default=0,
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss", default=1,
                    help="choose similarity loss: MSE (0) or NCC (1)")
parser.add_argument("--earlyStop", type=int,
                    dest="earlyStop", default=3,
                    help="choose after how many epochs early stopping is applied")
opt = parser.parse_args()

learning_rate = opt.learning_rate
bs = 1
epochs = opt.epochs
smth_lambda = opt.smth_lambda
dataset = opt.dataset
mode = opt.mode
choose_loss = opt.choose_loss
earlyStop = opt.earlyStop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

# load CMR training, validation and test data
assert mode >= 0 and mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
if dataset == 'ACDC':
    # load ACDC data
    train_set = TrainDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=False, num_workers=4)
    #input_shape = [216,256]
elif dataset == 'CMRxRecon':
    # load CMRxRecon data
    train_set = TrainDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', False, mode) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', False, mode) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=False, num_workers=4)
    #input_shape = [82,170] # for crops
elif dataset == 'OASIS':
    # path for OASIS dataset
    train_set = TrainDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS',trainingset = 4) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS')
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=False, num_workers=2)
    #input_shape = [160,224]
else:
    print('Incorrect dataset selected!! Must be either ACDC, CMRxRecon or OASIS...')

input_shape = train_set.__getitem__(0)[0].shape[1:3]

# path to save model parameters to
model_dir = './ModelParameters-{}/Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}/'.format(dataset,choose_loss,smth_lambda,learning_rate,mode)
model_dir_png = './ModelParameters-{}/Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}_Png/'.format(dataset,choose_loss,smth_lambda,learning_rate,mode)

if not isdir(model_dir):
    mkdir(model_dir)
    
if not isdir(model_dir_png):
    mkdir(model_dir_png)

csv_name = model_dir_png + 'Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}.csv'.format(choose_loss,smth_lambda,learning_rate,mode)
f = open(csv_name, 'w')
with f:
    if dataset == 'CMRxRecon':
        fnames = ['Epoch','MSE','SSIM'] #'Dice',
    else:
        fnames = ['Epoch','Dice','MSE','SSIM']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = True

# use dense voxelmorph
model = VxmDense(inshape=input_shape, nb_unet_features=32, bidir=False, nb_unet_levels=4).to(device)  #, int_steps=7, int_downsize=2
model.train()  

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
transform = SpatialTransformer(input_shape).to(device) #SpatialTransform().to(device)

# prepare image loss
if choose_loss == 1:
    loss_similarity = vxm_NCC().loss #NCC(win=9)
elif choose_loss == 0:
    loss_similarity = vxm_MSE().loss #MSE().loss
else:
    raise ValueError('Image loss should be 0 for MSE or 1 for NCC, but found %d' % choose_loss)
loss_smooth = vxm_Grad('l2').loss #smoothloss

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
        mov_img = image_pair[0].to(device).float()
        fix_img = image_pair[1].to(device).float()
        
        warped_mov, Df_xy = model(mov_img, fix_img)
        
        loss1 = loss_similarity(fix_img, warped_mov) 
        loss5 = loss_smooth(fix_img, warped_mov) #Df_xy
        
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
            fix_img = image_pair[0].to(device).float()
            mov_img = image_pair[1].to(device).float()
            
            if dataset != 'CMRxRecon':
                mov_seg = image_pair[2].to(device).float()
                fix_seg = image_pair[3].to(device).float()
            
            warped_mov_img, Df_xy = model(mov_img, fix_img)
            
            if dataset != 'CMRxRecon':
                warped_mov_seg = transform(mov_seg, Df_xy) #.permute(0, 2, 3, 1)
            
            # calculate MSE, SSIM and Dice 
            MSE_Validation.append(mean_squared_error(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy()))
            SSIM_Validation.append(structural_similarity(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy(), data_range=1))
            if dataset == 'OASIS':
                Dice_Validation.append(dice(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy()))
            elif dataset == 'ACDC':
                Dice_Validation.append(np.mean(dice_ACDC(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy())))
    
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

        # save best metrics and reset counter if Dice got better, else increase counter for early stopping
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
        
        # save and log model     
        if dataset == 'CMRxRecon':
            modelname = 'SSIM_{:.5f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_SSIM, Mean_MSE, epoch+1)
        else:
            modelname = 'DICE_{:.5f}_SSIM_{:.5f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_Dice, Mean_SSIM, Mean_MSE, epoch+1)
        save_checkpoint(model.state_dict(), model_dir, modelname)
        
        # save image
        sample_path = join(model_dir_png, 'Epoch_{:04d}-images.jpg'.format(epoch+1))
        save_flow(mov_img, fix_img, warped_mov_img, None, sample_path)
        if dataset == 'CMRxRecon':
            print("epoch {:d}/{:d} - SSIM_val: {:.5f}, MSE_val: {:.6f}".format(epoch+1, epochs, Mean_SSIM, Mean_MSE))
        else:
            print("epoch {:d}/{:d} - DICE_val: {:.5f}, SSIM_val: {:.5f}, MSE_val: {:.6f}".format(epoch+1, epochs, Mean_Dice, Mean_SSIM, Mean_MSE))

        # stop training if metrics stop improving for three epochs 
        if counter_earlyStopping >= earlyStop:
            break
            
print('Training ended on ', time.ctime())
