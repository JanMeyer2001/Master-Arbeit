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
parser.add_argument("--lambda", type=float,
                    dest="smth_lambda", default=0.1,
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
                    dest="model_num", default=6, 
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
opt = parser.parse_args()

learning_rate = opt.learning_rate
start_channel = opt.start_channel
smooth = opt.smth_lambda
dataset = opt.dataset
choose_loss = opt.choose_loss
mode = opt.mode
model_num = opt.model_num
diffeo = opt.diffeo
FT_size = [opt.FT_size_x,opt.FT_size_y]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

# choose the loss function for similarity
assert choose_loss >= 0 and choose_loss <= 4, f"Expected choose_loss to be one of SAD (0), Huber (1), NCC (2), L1 (3) or L2 (4), but got: {choose_loss}"
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
if dataset == 'ACDC':
    # load ACDC data
    train_set = TrainDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=False, num_workers=4)
    input_shape = train_set.__getitem__(0)[0].unsqueeze(0).shape
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
    model = Fourier_Net_dense(2, 2, start_channel, diffeo, FT_size).to(device) 
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

##############
## Training ##
##############

# counter and best SSIM for early stopping
counter_earlyStopping = 0
if dataset == 'CMRxRecon':
    best_SSIM = 0
else:
    best_Dice = 0

# test for one epoch
for i, image_pair in enumerate(training_generator):
    mov_img = image_pair[0].to(device).float()
    fix_img = image_pair[1].to(device).float()

    if model_num == 3:
        # ensure that all images have the same size for dense F-Net
        mov_img = F.interpolate(mov_img, [224,256], mode='nearest') 
        fix_img = F.interpolate(fix_img, [224,256], mode='nearest')
            
    Df_xy = model(mov_img, fix_img)
    grid, warped_mov = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
    
    loss1 = loss_similarity(fix_img, warped_mov) 
    loss2 = loss_smooth(Df_xy)
    
    loss = loss1 + smooth * loss2

    #"""
    sys.stdout.write("\r" + ' training loss {:.4f} - sim {:.4f} -smo {:.4f}'.format(loss,loss1,loss2))
    sys.stdout.flush()
    #"""
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
