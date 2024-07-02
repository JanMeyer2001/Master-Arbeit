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
parser.add_argument("--datapath", type=str, dest="datapath",
                    default='/home/jmeyer/storage/students/janmeyer_711878/data/ACDC',
                    #default='/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon',
                    help="data path for training images")
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
datapath = opt.datapath
mode = opt.mode
image_loss = opt.image_loss
earlyStop = opt.earlyStop

# load CMR training, validation and test data
assert mode >= 0 and mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
train_set = TrainDatasetACDC(datapath, mode) 
training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
validation_set = ValidationDatasetACDC(datapath, mode) 
validation_generator = Data.DataLoader(dataset=validation_set, batch_size=bs, shuffle=True, num_workers=4)
test_set = TestDatasetACDC(data_path=datapath, mode=mode)
test_generator = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

# path to save model parameters to
model_dir = './ModelParameters/Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}/'.format(image_loss,smth_lambda,learning_rate,mode)
model_dir_png = './ModelParameters/Voxelmorph_Png/'

if not isdir(model_dir):
    mkdir(model_dir)

csv_name = model_dir_png + 'Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}.csv'.format(image_loss,smth_lambda,learning_rate,mode)
f = open(csv_name, 'w')
with f:
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
        Dice_Validation = []
        
        for mov_img, fix_img, mov_seg, fix_seg in validation_generator: 
            fix_img = fix_img.cuda().float()
            mov_img = mov_img.cuda().float()
            mov_seg = mov_seg.cuda().float()
            fix_seg = fix_seg.cuda().float()
            
            warped_mov_img, Df_xy = model(mov_img, fix_img)
            warped_mov_seg, Df_xy = model(mov_seg, fix_img)
            
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
        save_flow(mov_img, fix_img, warped_mov, None, sample_path)
        print("epoch {:d}/{:d} - DICE_val: {:.5f} MSE_val: {:.6f}, SSIM_val: {:.5f}".format(epoch, epochs, Mean_Dice, Mean_MSE, Mean_SSIM))

        # stop training if metrics stop improving for three epochs (only on the first run)
        if counter_earlyStopping == 3:
            break
            
    print('Training ended on ', time.ctime())
