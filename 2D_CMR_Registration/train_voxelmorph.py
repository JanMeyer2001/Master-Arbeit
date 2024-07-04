import wandb
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
from argparse import ArgumentParser, BooleanOptionalAction
from os import mkdir
from os.path import isdir, join

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float,
                    dest="learning_rate", default=1e-4, help="learning rate")
parser.add_argument("--epochs", type=int, dest="epochs", default=100,
                    help="number of total epochs")
parser.add_argument("--lambda", type=float, dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--dataset", type=str,
                    dest="dataset",
                    default='ACDC',
                    help="dataset for training images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--mode", type=int, dest="mode", default=0,
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
parser.add_argument('--image_loss', type=str, dest="image_loss", default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument("--earlyStop", type=bool,
                    dest="earlyStop", default=True, action=BooleanOptionalAction, 
                    help="choose whether to use early stopping to prevent overfitting (True) or not (False)")
opt = parser.parse_args()

learning_rate = opt.learning_rate
bs = 1
epochs = opt.epochs
smth_lambda = opt.smth_lambda
dataset = opt.dataset
mode = opt.mode
image_loss = opt.image_loss
earlyStop = opt.earlyStop

# load CMR training, validation and test data
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
    valid_set = ValidationDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS')
    valid_generator = Data.DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=2)
else:
    print('Incorrect dataset selected!! Must be either ACDC, CMRxRecon or OASIS...')

# path to save model parameters to
model_dir = './ModelParameters-{}/Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}/'.format(dataset,image_loss,smth_lambda,learning_rate,mode)
model_dir_png = './ModelParameters-{}/Voxelmorph_Png/'.format(dataset)

if not isdir(model_dir):
    mkdir(model_dir)
    
if not isdir(model_dir_png):
    mkdir(model_dir_png)

csv_name = model_dir_png + 'Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}.csv'.format(image_loss,smth_lambda,learning_rate,mode)
f = open(csv_name, 'w')
with f:
    if dataset == 'CMRxRecon':
        fnames = ['Epoch','MSE','SSIM'] #'Dice',
    else:
        fnames = ['Epoch','Dice','MSE','SSIM']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

# define model
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

# use dense voxelmorph
model = VxmDense(inshape=[82,170], nb_unet_features=[enc_nf, dec_nf], bidir=False).cuda()     #, int_steps=7, int_downsize=2
model.train()  

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
transform = SpatialTransform().cuda()

# prepare image loss
if image_loss == 'ncc':
    loss_similarity = NCC().loss
elif image_loss == 'mse':
    loss_similarity = MSE().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % image_loss)
loss_smooth = smoothloss

##############
## Training ##
##############

if earlyStop:
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
        
        warped_mov, Df_xy = model(mov_img, fix_img)
        
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
            
            warped_mov_img, Df_xy = model(mov_img, fix_img)
            
            if dataset != 'CMRxRecon':
                grid, warped_mov_seg = transform(mov_seg, Df_xy.permute(0, 2, 3, 1))
            
            # calculate MSE, SSIM and Dice 
            MSE_Validation.append(mean_squared_error(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy()))
            SSIM_Validation.append(structural_similarity(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy(), data_range=1))
            if dataset != 'CMRxRecon':
                Dice_Validation.append(dice(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy()))
    
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

        if earlyStop:
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
            modelname = 'SSIM_{:.5f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_SSIM, Mean_MSE, epoch)
        else:
            modelname = 'DICE_{:.5f}_SSIM_{:.5f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_Dice, Mean_SSIM, Mean_MSE, epoch)
        save_checkpoint(model.state_dict(), model_dir, modelname)
        
        # save image
        sample_path = join(model_dir_png, 'Epoch_{:04d}-images.jpg'.format(epoch))
        save_flow(mov_img, fix_img, warped_mov, grid.permute(0, 3, 1, 2), sample_path)
        if dataset == 'CMRxRecon':
            print("epoch {:d}/{:d} - SSIM_val: {:.5f}, MSE_val: {:.6f}".format(epoch, epochs, Mean_SSIM, Mean_MSE))
        else:
            print("epoch {:d}/{:d} - DICE_val: {:.5f}, SSIM_val: {:.5f}, MSE_val: {:.6f}".format(epoch, epochs, Mean_Dice, Mean_SSIM, Mean_MSE))

        # stop training if metrics stop improving for three epochs (only on the first run)
        if counter_earlyStopping == 3:
            break
            
    print('Training ended on ', time.ctime())
