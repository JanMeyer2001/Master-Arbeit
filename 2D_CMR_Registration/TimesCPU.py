import csv 
from argparse import ArgumentParser
import numpy as np
from Functions import *
import torch
from Models import *
from argparse import ArgumentParser
from natsort import natsorted
from os import listdir
import warnings
warnings.filterwarnings("ignore")
import time

parser = ArgumentParser()
parser.add_argument("--start_channel", type=int, dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--FT_size_x", type=int,
                    dest="FT_size_x", default=24,
                    help="choose size x of FT crop: Should be smaller than 40.")
parser.add_argument("--FT_size_y", type=int,
                    dest="FT_size_y", default=24,
                    help="choose size y of FT crop: Should be smaller than 84.")
parser.add_argument("--diffeo", type=int,
                    dest="diffeo", default=0, 
                    help="choose whether to use a diffeomorphic transform (1) or not (0)")
parser.add_argument("--domain_sim", type=int,
                    dest="domain_sim", default=0,
                    help="choose which domain the similarity loss should be applied: image space (0) or k-space (1)")
parser.add_argument("--mode", type=int, dest="mode", default=0,
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
opt = parser.parse_args()

start_channel = opt.start_channel
FT_size = [opt.FT_size_x,opt.FT_size_y]
mode = opt.mode
diffeo = opt.diffeo
domain_sim = opt.domain_sim

dataset = 'ACDC'
# load ACDC test data
test_set = TestDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
input_shape = [216,256]

device = torch.device("cpu") 
# init different models
model_voxelmorph              = VxmDense(inshape=input_shape, nb_unet_features=32, bidir=False, nb_unet_levels=4).to(device)
model_f_net                   = Fourier_Net(2, 2, start_channel, 0).to(device)
model_f_net_plus              = Fourier_Net_plus(2, 2, start_channel, 0, FT_size).to(device) 
model_f_net_plus_cascade      = Cascade(2, 2, start_channel, 0, FT_size).to(device)
if diffeo == 1:
    model_f_net_diff              = Fourier_Net(2, 2, start_channel, 1).to(device)
    model_f_net_plus_diff         = Fourier_Net_plus(2, 2, start_channel, 1, FT_size).to(device) 
    model_f_net_plus_cascade_diff = Cascade(2, 2, start_channel, 1, FT_size).to(device)
    # times for 7 models and 224 test images
    timesCPU = np.zeros((7,224))
else:
    # times for 4 models (without diffeo-versions) and 224 test images
    timesCPU = np.zeros((4,224))    

# load different models
path_voxelmorph      = './ModelParameters-{}/Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}/'.format(dataset,0,0.01,0.0001,mode) # for voxelmorph 0 is MSE loss
modelpath_voxelmorph = path_voxelmorph + natsorted(listdir(path_voxelmorph))[-1]
model_voxelmorph.load_state_dict(torch.load(modelpath_voxelmorph,map_location='cpu'))
model_voxelmorph.eval()

path_f_net      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Sim{}_Pth/'.format(dataset,0,0,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode,domain_sim) #_Sim_0
modelpath_f_net = path_f_net + natsorted(listdir(path_f_net))[-1]
model_f_net.load_state_dict(torch.load(modelpath_f_net,map_location='cpu'))
model_f_net.eval()

path_f_net_plus      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Sim{}_Pth/'.format(dataset,1,0,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode,domain_sim)
modelpath_f_net_plus = path_f_net_plus + natsorted(listdir(path_f_net_plus))[-1]
model_f_net_plus.load_state_dict(torch.load(modelpath_f_net_plus,map_location='cpu'))
model_f_net_plus.eval()

path_f_net_plus_cascade      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Sim{}_Pth/'.format(dataset,2,0,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode,domain_sim)
modelpath_f_net_plus_cascade = path_f_net_plus_cascade + natsorted(listdir(path_f_net_plus_cascade))[-1]
model_f_net_plus_cascade.load_state_dict(torch.load(modelpath_f_net_plus_cascade,map_location='cpu'))
model_f_net_plus_cascade.eval()

if diffeo == 1:
    path_f_net_diff      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Sim{}_Pth/'.format(dataset,0,1,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode,domain_sim)
    modelpath_f_net_diff = path_f_net_diff + natsorted(listdir(path_f_net_diff))[-1]
    model_f_net_diff.load_state_dict(torch.load(modelpath_f_net_diff,map_location='cpu'))
    model_f_net_diff.eval()

    path_f_net_plus_diff      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Sim{}_Pth/'.format(dataset,1,1,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode,domain_sim)
    modelpath_f_net_plus_diff = path_f_net_plus_diff + natsorted(listdir(path_f_net_plus_diff))[-1]
    model_f_net_plus_diff.load_state_dict(torch.load(modelpath_f_net_plus_diff,map_location='cpu'))
    model_f_net_plus_diff.eval()

    path_f_net_plus_cascade_diff      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Sim{}_Pth/'.format(dataset,2,1,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode,domain_sim)
    modelpath_f_net_plus_cascade_diff = path_f_net_plus_cascade_diff + natsorted(listdir(path_f_net_plus_cascade_diff))[-1]
    model_f_net_plus_cascade_diff.load_state_dict(torch.load(modelpath_f_net_plus_cascade_diff,map_location='cpu'))
    model_f_net_plus_cascade_diff.eval()

csv_name = './TestResults-{}/TimesCPU_Mode_{}.csv'.format(dataset,mode)

for i, image_pairs in enumerate(test_generator): 
    with torch.no_grad():
        moving_image = image_pairs[0].float().to(device)
        fixed_image  = image_pairs[1].float().to(device)
        moving_seg   = image_pairs[2].float().to(device)
        fixed_seg    = image_pairs[3].float().to(device)
        
        start = time.time()
        warped_image_voxelmorph, V_voxelmorph = model_voxelmorph(moving_image,fixed_image)
        timesCPU[0,i] = time.time() - start
        start = time.time()
        V_f_net                               = model_f_net(moving_image,fixed_image)
        timesCPU[1,i] = time.time() - start
        start = time.time()
        V_f_net_plus                          = model_f_net_plus(moving_image,fixed_image)
        timesCPU[2,i] = time.time() - start
        start = time.time()
        V_f_net_plus_cascade                  = model_f_net_plus_cascade(moving_image,fixed_image)
        timesCPU[3,i] = time.time() - start
        start = time.time()
        if diffeo == 1:
            start = time.time()
            V_f_net_diff                          = model_f_net(moving_image,fixed_image)
            timesCPU[4,i] = time.time() - start
            start = time.time()
            V_f_net_plus_diff                     = model_f_net_plus(moving_image,fixed_image)
            timesCPU[5,i] = time.time() - start
            start = time.time()
            V_f_net_plus_cascade_diff             = model_f_net_plus_cascade(moving_image,fixed_image)
            timesCPU[6,i] = time.time() - start 
        
# take mean value and std over all 224 test images 
timesCPU_mean = np.mean(timesCPU, axis=1)
timesCPU_std  = np.std(timesCPU, axis=1)

print('Mean times on CPU (in seconds):\n    VoxelMorph        {:.4f}\n    Fourier-Net       {:.4f}\n    Fourier-Net+      {:.4f}\n    4xFourier-Net+    {:.4f}'.format(timesCPU_mean[0],timesCPU_mean[1],timesCPU_mean[2],timesCPU_mean[3]))

# write times to the csv file
f = open(csv_name, 'w')
with f:
    fnames = ['Model','Mean Time on CPU','Std Time on CPU']  
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()
    writer = csv.writer(f)
    writer.writerow(['VoxelMorph', timesCPU_mean[0], timesCPU_std[0]])
    writer.writerow(['Fourier-Net', timesCPU_mean[1], timesCPU_std[1]])
    writer.writerow(['Fourier-Net+', timesCPU_mean[2], timesCPU_std[2]])
    writer.writerow(['4xFourier-Net+', timesCPU_mean[3], timesCPU_std[3]])
    if diffeo == 1: 
        writer.writerow(['Diff-Fourier-Net', timesCPU_mean[4], timesCPU_std[4]])
        writer.writerow(['Diff-Fourier-Net+', timesCPU_mean[5], timesCPU_std[5]])
        writer.writerow(['Diff-4xFourier-Net+', timesCPU_mean[6], timesCPU_std[6]])