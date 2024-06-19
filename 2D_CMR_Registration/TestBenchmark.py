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
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--smth_lambda", type=float,
                    dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon',
                    help="data path for training images")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss", default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2), SSIM (3)")
parser.add_argument("--mode", type=int,
                    dest="mode", default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
parser.add_argument("--F_Net_plus", type=bool,
                    dest="F_Net_plus", default=True, action=argparse.BooleanOptionalAction, 
                    help="choose whether to use Fourier-Net (False) or Fourier-Net+ (True) as the model")
parser.add_argument("--diffeo", type=bool,
                    dest="diffeo", default=True, action=argparse.BooleanOptionalAction, 
                    help="choose whether to use a diffeomorphic transform (True) or not (False)")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
smooth = opt.smth_lambda
datapath = opt.datapath
choose_loss = opt.choose_loss
mode = opt.mode
F_Net_plus = opt.F_Net_plus
diffeo = opt.diffeo

# choose the model
model_name = 0
if F_Net_plus:
    model = Cascade(2, 2, start_channel).cuda()
    model_name = 1
else:     
    model = Fourier_Net(2, 2, start_channel).cuda()  

# choose whether to use a diffeomorphic transform or not
diffeo_name = 0
if diffeo:
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    diffeo_name = 1

transform = SpatialTransform().cuda()

path ='./ModelParameters/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(model_name,diffeo_name,choose_loss,start_channel,smooth, lr, mode)
model_idx = -1
from natsort import natsorted
print('Best model: {}'.format(natsorted(os.listdir(path))[model_idx]))
modelpath = path + natsorted(os.listdir(path))[model_idx]
bs = 1

torch.backends.cudnn.benchmark = True
model.load_state_dict(torch.load(modelpath))
model.eval()
transform.eval()
#Dices=[]
MSE_test = []
SSIM_test = []
NegJ=[]
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
test_set = TestDatasetCMRBenchmark(data_path=datapath, mode=mode)
test_generator = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)
times = []

csv_name = './TestResults-Metrics/TestMetrics-Model_{}_Diffeo_{}_Loss_{}_Chan_{}_Smth_{}_LR_{}_Mode_{}.csv'.format(model_name,diffeo_name,choose_loss,start_channel,smooth, lr, mode)
f = open(csv_name, 'w')
with f:
    fnames = ['Image Pair','MSE','SSIM','Time','Mean MSE','Mean SSIM','Mean Time','Mean NegJ']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

image_num = 1 
for mov_img_fullySampled, fix_img_fullySampled, mov_img_subSampled, fix_img_subSampled in test_generator: #, mov_lab, fix_lab
    with torch.no_grad():
        start = time.time()
        # calculate displacement on subsampled data
        V_xy = model(mov_img_subSampled.float().to(device), fix_img_subSampled.float().to(device))
        # but warp fully sampled data
        __, warped_mov_img_fullySampled = transform(mov_img_fullySampled.float().to(device), V_xy.permute(0, 2, 3, 1), mod = 'nearest')
        
        #__, warped_mov_lab = transform(mov_lab.float().to(device), V_xy.permute(0, 2, 3, 1), mod = 'nearest') # old code for dice score
    
        # get inference time
        inference_time = time.time()-start
        times.append(inference_time)

        # convert to numpy
        warped_mov_img_fullySampled = warped_mov_img_fullySampled[0,0,:,:].cpu().numpy()
        fix_img_fullySampled = fix_img_fullySampled[0,0,:,:].cpu().numpy()
        
        # calculate metrics on fully sampled images
        MSE = mean_squared_error(warped_mov_img_fullySampled, fix_img_fullySampled)
        SSIM = structural_similarity(warped_mov_img_fullySampled, fix_img_fullySampled, data_range=1)

        MSE_test.append(MSE)
        SSIM_test.append(SSIM)

        hh, ww = V_xy.shape[-2:]
        V_xy = V_xy.detach().cpu().numpy()
        V_xy[:,0,:,:] = V_xy[:,0,:,:] * hh / 2
        V_xy[:,1,:,:] = V_xy[:,1,:,:] * ww / 2

        jac_det = jacobian_determinant_vxm(V_xy[0, :, :, :])
        negJ = np.sum(jac_det <= 0) / 160 / 192 * 100
        NegJ.append(negJ)

        # save to csv file
        f = open(csv_name, 'a')
        with f:
            writer = csv.writer(f)
            writer.writerow([image_num, MSE, SSIM, inference_time, '-', '-', '-', '-']) 
        image_num += 1

        """
        for bs_index in range(bs):
            dice_bs = dice(warped_mov_lab[bs_index,...].data.cpu().numpy().copy(),fix_lab[bs_index,...].data.cpu().numpy().copy())
            Dices.append(dice_bs)
        """ 

mean_MSE = np.mean(MSE_test)
std_MSE = np.std(MSE_test)

mean_SSIM = np.mean(SSIM_test)
std_SSIM = np.std(SSIM_test)

mean_NegJ = np.mean(NegJ)
std_NegJ = np.std(NegJ)

mean_time = np.mean(times)

f = open(csv_name, 'a')
with f:
    writer = csv.writer(f)
    writer.writerow(['-', '-', '-', '-', mean_MSE, mean_SSIM, mean_time, mean_NegJ])  

print('Mean inference time: ', mean_time, ' seconds')
print('Mean MSE: ', mean_MSE, 'Std MSE: ', std_MSE)
print('Mean SSIM: ', mean_SSIM, 'Std SSIM: ', std_SSIM)
print('Mean DetJ<0 %:', mean_NegJ, 'Std DetJ<0 %:', std_NegJ)
#print('Mean Dice Score: ', np.mean(Dices), 'Std Dice Score: ', np.std(Dices))

