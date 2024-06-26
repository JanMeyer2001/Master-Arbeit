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
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--epochs", type=int, dest="epochs", default=100,
                    help="number of total epochs")
parser.add_argument("--lambda", type=float, dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--datapath", type=str, dest="datapath",
                    default='/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon',
                    help="data path for training images")
parser.add_argument("--mode", type=int, dest="mode", default=0,
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
parser.add_argument('--image_loss', type=str, dest="image_loss", default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
opt = parser.parse_args()

lr = opt.lr
bs = 1
epochs = opt.epochs
smth_lambda = opt.smth_lambda
datapath = opt.datapath
mode = opt.mode
image_loss = opt.image_loss

# counter and best SSIM for early stopping
counter_earlyStopping = 0
best_SSIM = 0

# load CMR training, validation and test data
assert mode >= 0 and mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
train_set = TrainDatasetCMR(datapath, mode) 
training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
validation_set = ValidationDatasetCMR(datapath, mode) 
validation_generator = Data.DataLoader(dataset=validation_set, batch_size=bs, shuffle=True, num_workers=4)
test_set = TestDatasetCMRBenchmark(data_path=datapath, mode=mode)
test_generator = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

# path to save model parameters to
model_dir = './ModelParameters/Voxelmorph/'
model_dir_png = './ModelParameters/Voxelmorph_Png/'

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

# define model
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

# use dense voxelmorph
model = VxmDense(inshape=[82,170], nb_unet_features=[enc_nf, dec_nf], bidir=False).cuda()     #, int_steps=7, int_downsize=2
model.train()  

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
        
        for mov_img, fix_img in validation_generator: 
            fix_img = fix_img.cuda().float()
            mov_img = mov_img.cuda().float()
            
            f_xy = model(mov_img, fix_img)
            
            # calculate MSE and SSIM
            MSE_Validation.append(mean_squared_error(warped_mov[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy()))
            SSIM_Validation.append(structural_similarity(warped_mov[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy(), data_range=1))
    
        # calculate mean of validation metrics
        Mean_MSE = np.mean(MSE_Validation)
        Mean_SSIM = np.mean(SSIM_Validation)
        
        # save best metrics and reset counter if SSIM got better, else increase counter for early stopping
        if Mean_SSIM>best_SSIM:
            best_SSIM = Mean_SSIM
            counter_earlyStopping = 0
        else:
            counter_earlyStopping += 1    
        
        # log loss and validation metrics to wandb
        wandb.log({"Loss": np.mean(losses), "MSE": Mean_MSE, "SSIM": Mean_SSIM})
        
        # save and log model     
        modelname = 'SSIM_{:.6f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_SSIM, Mean_MSE, epoch)
        save_checkpoint(model.state_dict(), model_dir, modelname)
        wandb.log_model(path=model_dir, name=modelname)

        # save image
        save_path = os.path.join(model_dir_png, 'Epoch_{:04d}-images.jpg'.format(epoch))
        save_flow(mov=mov_img, fix=fix_img, warp=warped_mov, grid=None, save_path=save_path)
        print("epoch {0:d}/{1:d} - MSE_val: {2:.6f}, SSIM_val: {3:.5f}".format(epoch, epochs, Mean_MSE, Mean_SSIM))
        
        # stop training if metrics stop improving for three epochs
        if counter_earlyStopping == 3:
            break
        
print('Training ended on ', time.ctime())