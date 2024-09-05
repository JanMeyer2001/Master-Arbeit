import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
import csv
import time
from skimage.metrics import structural_similarity, mean_squared_error
from fastmri.data.subsample import RandomMaskFunc, EquispacedMaskFractionFunc
import sys

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float,
                    dest="learning_rate", default=1e-3, help="learning rate")
parser.add_argument("--epochs", type=int,
                    dest="epochs", default=15,
                    help="number of epochs")
parser.add_argument("--lambda", type=float,
                    dest="smth_lambda", default=10,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=16,
                    help="number of start channels")
parser.add_argument("--dataset", type=str, 
                    dest="dataset", default="ACDC",
                    help="dataset for training images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss", default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2), L1 (3) or L2 (4)")
parser.add_argument("--mode", type=int,
                    dest="mode", default=1,
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
parser.add_argument("--model", type=int,
                    dest="model_num", default=0, 
                    help="choose whether to use Fourier-Net (0), Fourier-Net+ (1), cascaded Fourier-Net+ (2), dense Fourier-Net (3), dense Fourier-Net+ (4), dense cascaded Fourier-Net+ (5), k-space Fourier-Net (6), k-space Fourier-Net+ (7) or cascaded k-space Fourier-Net+ (8) as the model")
parser.add_argument("--diffeo", type=int,
                    dest="diffeo", default=0, 
                    help="choose whether to use a diffeomorphic transform (1) or not (0)")
parser.add_argument("--FT_size_x", type=int,
                    dest="FT_size_x", default=24,
                    help="choose size x of FT crop: Should be smaller than 40.")
parser.add_argument("--FT_size_y", type=int,
                    dest="FT_size_y", default=24,
                    help="choose size y of FT crop: Should be smaller than 84.")
parser.add_argument("--earlyStop", type=int,
                    dest="earlyStop", default=3,
                    help="choose after how many epochs early stopping is applied")
opt = parser.parse_args()

learning_rate = opt.learning_rate
epochs = opt.epochs
start_channel = opt.start_channel
smooth = opt.smth_lambda
dataset = opt.dataset
choose_loss = opt.choose_loss
mode = opt.mode
model_num = opt.model_num
diffeo = opt.diffeo
FT_size = [opt.FT_size_x,opt.FT_size_y]
earlyStop = opt.earlyStop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

# choose the loss function for similarity
assert choose_loss >= 0 and choose_loss <= 4, f"Expected choose_loss to be one of SAD (0), MSE (1), NCC (2), L1 (3) or L2 (4), but got: {choose_loss}"
if choose_loss == 1:
    loss_similarity = MSE().loss
elif choose_loss == 0:
    loss_similarity = SAD().loss
elif choose_loss == 2:
    loss_similarity = NCC(win=9)
elif choose_loss == 3:
    loss_similarity = torch.nn.L1Loss()    
elif choose_loss == 4:
    loss_similarity = RelativeL2Loss()
loss_smooth = smoothloss

transform = SpatialTransform().to(device)

for param in transform.parameters():
    param.requires_grad = False
    param.volatile = True

assert mode >= 0 and mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
if dataset == 'ACDC' and model_num < 6:
    # load ACDC data
    train_set = TrainDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=False, num_workers=4)
    input_shape = train_set.__getitem__(0)[0].unsqueeze(0).shape
elif dataset == 'ACDC' and model_num >= 6:
    # load k-space ACDC data
    train_set = TrainDatasetACDC_kSpace('/home/jmeyer/storage/students/janmeyer_711878/data/kSpace/ACDC', mode) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=216*256, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetACDC_kSpace('/home/jmeyer/storage/students/janmeyer_711878/data/kSpace/ACDC', mode) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=216*256, shuffle=False, num_workers=4)
    input_shape = [216,256]    
elif dataset == 'CMRxRecon':
    # load CMRxRecon data
    train_set = TrainDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=False, num_workers=4)
    input_shape = train_set.__getitem__(0)[0].unsqueeze(0).shape
elif dataset == 'OASIS':
    # path for OASIS dataset
    train_set = TrainDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS',trainingset = 4) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS')
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=False, num_workers=2)
    input_shape = train_set.__getitem__(0)[0].unsqueeze(0).shape
else:
    raise ValueError('Dataset should be "ACDC", "CMRxRecon" or "OASIS", but found "%s"!' % dataset)

# choose the model
assert model_num >= 0 or model_num <= 8, f"Expected F_Net_plus to be between 0 and 8, but got: {model_num}"
assert diffeo == 0 or diffeo == 1, f"Expected diffeo to be either 0 or 1, but got: {diffeo}"
if model_num == 0:
    model = Fourier_Net(2, 2, start_channel, diffeo).to(device) 
elif model_num == 1:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Fourier_Net_plus(2, 2, start_channel, diffeo, FT_size).to(device) 
elif model_num == 2:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade(2, 2, start_channel, diffeo, FT_size).to(device) 
elif model_num == 3:
    model = Fourier_Net_dense(2, 2, start_channel, diffeo).to(device) 
elif model_num == 4:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Fourier_Net_plus_dense(2, 2, start_channel, diffeo, FT_size).to(device) 
elif model_num == 5:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade_dense(2, 2, start_channel, diffeo, FT_size).to(device)  
elif model_num == 6:
    model = Fourier_Net_kSpace(in_shape=input_shape, diffeo=diffeo).to(device) 
elif model_num == 7:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Fourier_Net_plus_kSpace(4, 2, start_channel, diffeo, FT_size).to(device) 
elif model_num == 8:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade_kSpace(4, 2, start_channel, diffeo, FT_size).to(device) 

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if mode == 1:
    # get Acc4 masking function
    mask_func = EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[4])
elif mode == 2:
    # get Acc8 masking function
    mask_func = EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[8])  
elif mode == 3:
    # get Acc10 masking function
    mask_func = EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[10])  
elif mode == 0:
    # get fully sampled masking function
    mask_func = EquispacedMaskFractionFunc(center_fractions=[0.08], accelerations=[0]) 

if mode != 0:
    # create and repeat mask
    mask, _ = mask_func(input_shape)
    mask = mask.repeat(1,1,1,input_shape[-1])        

model_dir = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
model_dir_png = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Png/'.format(dataset,model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)

if not isdir(model_dir_png):
    mkdir(model_dir_png)

if not isdir(model_dir):
    mkdir(model_dir)

csv_name = model_dir_png + 'Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}.csv'.format(model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
f = open(csv_name, 'w')
with f:
    if dataset == 'CMRxRecon':
        fnames = ['Epoch','MSE','SSIM'] 
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
        mov_img = image_pair[0].to(device).float()
        fix_img = image_pair[1].to(device).float()

        if model_num == 3:
            # ensure that all images have the same size for dense F-Net
            mov_img = F.interpolate(mov_img, [224,256], mode='nearest') 
            fix_img = F.interpolate(fix_img, [224,256], mode='nearest')
                
        Df_xy = model(mov_img, fix_img)
        grid, warped_mov = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
        
        # compute similarity loss in the image space
        loss1 = loss_similarity(fix_img, warped_mov) 
        loss2 = loss_smooth(Df_xy)
        
        loss = loss1 + smooth * loss2
        losses[i] = loss

        #"""
        sys.stdout.write("\r" + 'epoch {}/{} -> training loss {:.4f} - sim {:.4f} -smo {:.4f}'.format(epoch+1,epochs,loss,loss1,loss2))
        sys.stdout.flush()
        #"""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot losses 
    plt.subplots(figsize=(7, 4))
    plt.subplot(1,1,1)
    plt.plot(losses)
    plt.title('Loss Epoch {}'.format(epoch+1))
    sample_path = join(model_dir_png, 'Losses_Epoch_{:04d}.jpg'.format(epoch+1))
    plt.savefig(sample_path)  

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
            
            if model_num == 3:
                # ensure that all images and segmentations have the same size for dense F-Net
                mov_img = F.interpolate(mov_img, [224,256], mode='nearest') 
                fix_img = F.interpolate(fix_img, [224,256], mode='nearest')
                mov_seg = F.interpolate(mov_seg, [224,256], mode='nearest') 
                fix_seg = F.interpolate(fix_seg, [224,256], mode='nearest')     
            
            Df_xy = model(mov_img, fix_img)

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
                writer.writerow([epoch+1, Mean_MSE, Mean_SSIM]) 
            else:
                writer.writerow([epoch+1, Mean_Dice, Mean_MSE, Mean_SSIM]) 

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
            modelname = 'SSIM_{:.5f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_SSIM, Mean_MSE, epoch+1)
        else:
            modelname = 'DICE_{:.5f}_SSIM_{:.5f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_Dice, Mean_SSIM, Mean_MSE, epoch+1)
        save_checkpoint(model.state_dict(), model_dir, modelname)
        
        # save image
        sample_path = join(model_dir_png, 'Epoch_{:04d}-images.jpg'.format(epoch+1))
        save_flow(mov_img, fix_img, warped_mov_img, grid.permute(0, 3, 1, 2), sample_path)
        if dataset == 'CMRxRecon':
            print("epoch {:d}/{:d} - SSIM_val: {:.5f}, MSE_val: {:.6f}".format(epoch+1, epochs, Mean_SSIM, Mean_MSE))
        else:
            print("epoch {:d}/{:d} - DICE_val: {:.5f}, SSIM_val: {:.5f}, MSE_val: {:.6f}".format(epoch+1, epochs, Mean_Dice, Mean_SSIM, Mean_MSE))

        # stop training if metrics stop improving for three epochs (only on the first run)
        if counter_earlyStopping >= earlyStop:  
            break
            
print('Training ended on ', time.ctime())
