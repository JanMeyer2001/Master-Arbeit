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
import nibabel

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
parser.add_argument("--mode", type=int, dest="mode", default=0,
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
opt = parser.parse_args()

start_channel = opt.start_channel
FT_size = [opt.FT_size_x,opt.FT_size_y]
mode = opt.mode
diffeo = opt.diffeo

dataset = 'ACDC'
# load ACDC test data
test_set = TestDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
input_shape = [216,256]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
transform = SpatialTransform().to(device)
transform_voxelmorph = SpatialTransformer(input_shape, mode = 'nearest').to(device)

# init different models
model_voxelmorph              = VxmDense(inshape=input_shape, nb_unet_features=32, bidir=False, nb_unet_levels=4).to(device)
model_f_net                   = Fourier_Net(2, 2, start_channel, 0).to(device)
model_f_net_plus              = Fourier_Net_plus(2, 2, start_channel, 0, FT_size).to(device) 
model_f_net_plus_cascade      = Cascade(2, 2, start_channel, 0, FT_size).to(device)
if diffeo == 1:
    model_f_net_diff              = Fourier_Net(2, 2, start_channel, 1).to(device)
    model_f_net_plus_diff         = Fourier_Net_plus(2, 2, start_channel, 1, FT_size).to(device) 
    model_f_net_plus_cascade_diff = Cascade(2, 2, start_channel, 1, FT_size).to(device)

# load different models
path_voxelmorph      = './ModelParameters-{}/Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}/'.format(dataset,0,0.01,0.0001,mode) # for voxelmorph 0 is MSE loss
modelpath_voxelmorph = path_voxelmorph + natsorted(listdir(path_voxelmorph))[-1]
model_voxelmorph.load_state_dict(torch.load(modelpath_voxelmorph))
model_voxelmorph.eval()

path_f_net      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,0,0,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode)
#modelpath_f_net = path_f_net + natsorted(listdir(path_f_net))[-1]
modelpath_f_net = [f.path for f in scandir(path_f_net) if f.is_file() and not (f.name.find('Epoch_0006') == -1)][0]
model_f_net.load_state_dict(torch.load(modelpath_f_net))
model_f_net.eval()

path_f_net_plus      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,1,0,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode)
#modelpath_f_net_plus = path_f_net_plus + natsorted(listdir(path_f_net_plus))[-1]
modelpath_f_net_plus = [f.path for f in scandir(path_f_net_plus) if f.is_file() and not (f.name.find('Epoch_0006') == -1)][0]
model_f_net_plus.load_state_dict(torch.load(modelpath_f_net_plus))
model_f_net_plus.eval()

path_f_net_plus_cascade      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,2,0,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode)
#modelpath_f_net_plus_cascade = path_f_net_plus_cascade + natsorted(listdir(path_f_net_plus_cascade))[-1]
modelpath_f_net_plus_cascade = [f.path for f in scandir(path_f_net_plus_cascade) if f.is_file() and not (f.name.find('Epoch_0004') == -1)][0]
model_f_net_plus_cascade.load_state_dict(torch.load(modelpath_f_net_plus_cascade))
model_f_net_plus_cascade.eval()

if diffeo == 1:
    path_f_net_diff      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,0,1,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode)
    modelpath_f_net_diff = path_f_net_diff + natsorted(listdir(path_f_net_diff))[-1]
    model_f_net_diff.load_state_dict(torch.load(modelpath_f_net_diff))
    model_f_net_diff.eval()

    path_f_net_plus_diff      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,1,1,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode)
    modelpath_f_net_plus_diff = path_f_net_plus_diff + natsorted(listdir(path_f_net_plus_diff))[-1]
    model_f_net_plus_diff.load_state_dict(torch.load(modelpath_f_net_plus_diff))
    model_f_net_plus_diff.eval()

    path_f_net_plus_cascade_diff      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,2,1,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode)
    modelpath_f_net_plus_cascade_diff = path_f_net_plus_cascade_diff + natsorted(listdir(path_f_net_plus_cascade_diff))[-1]
    model_f_net_plus_cascade_diff.load_state_dict(torch.load(modelpath_f_net_plus_cascade_diff))
    model_f_net_plus_cascade_diff.eval()

csv_name = './TestResults-{}/DiceScores_Mode_{}.csv'.format(dataset,mode)

# init arrays for 4 labels and 224 test images
dices_baseline                = np.zeros((4,224))
dices_NiftyReg                = np.zeros((4,224))
dices_voxelmorph              = np.zeros((4,224))
dices_f_net                   = np.zeros((4,224))
dices_f_net_plus              = np.zeros((4,224))
dices_f_net_plus_cascade      = np.zeros((4,224))
if diffeo == 1:
    dices_f_net_diff              = np.zeros((4,224))
    dices_f_net_plus_diff         = np.zeros((4,224))
    dices_f_net_plus_cascade_diff = np.zeros((4,224))
        

for i, image_pairs in enumerate(test_generator): 
    with torch.no_grad():
        moving_image = image_pairs[0].float().to(device)
        fixed_image  = image_pairs[1].float().to(device)
        moving_seg   = image_pairs[2].float().to(device)
        fixed_seg    = image_pairs[3].float().to(device)
        
        warped_image_voxelmorph, V_voxelmorph = model_voxelmorph(moving_image,fixed_image)
        V_f_net, __                           = model_f_net(moving_image,fixed_image)
        V_f_net_plus, __                      = model_f_net_plus(moving_image,fixed_image)
        V_f_net_plus_cascade, __              = model_f_net_plus_cascade(moving_image,fixed_image)
        if diffeo == 1:
            V_f_net_diff, __                  = model_f_net_diff(moving_image,fixed_image)
            V_f_net_plus_diff, __             = model_f_net_plus_diff(moving_image,fixed_image)
            V_f_net_plus_cascade_diff, __     = model_f_net_plus_cascade_diff(moving_image,fixed_image)
        
        warped_seg_voxelmorph                  = transform_voxelmorph(moving_seg, V_voxelmorph) #.permute(0, 2, 3, 1), mod = 'nearest'
        __, warped_seg_f_net                   = transform(moving_seg, V_f_net.permute(0, 2, 3, 1), mod = 'nearest')
        __, warped_seg_f_net_plus              = transform(moving_seg, V_f_net_plus.permute(0, 2, 3, 1), mod = 'nearest')
        __, warped_seg_f_net_plus_cascade      = transform(moving_seg, V_f_net_plus_cascade.permute(0, 2, 3, 1), mod = 'nearest')
        if diffeo == 1:
            __, warped_seg_f_net_diff              = transform(moving_seg, V_f_net_diff.permute(0, 2, 3, 1), mod = 'nearest')
            __, warped_seg_f_net_plus_diff         = transform(moving_seg, V_f_net_plus_diff.permute(0, 2, 3, 1), mod = 'nearest')
            __, warped_seg_f_net_plus_cascade_diff = transform(moving_seg, V_f_net_plus_cascade_diff.permute(0, 2, 3, 1), mod = 'nearest')

        # calculate Dices
        dices_baseline[:,i]                = dice_ACDC(moving_seg[0,0,:,:].cpu().numpy(),fixed_seg[0,0,:,:].cpu().numpy())
        dices_voxelmorph[:,i]              = dice_ACDC(warped_seg_voxelmorph[0,0,:,:].cpu().numpy(),fixed_seg[0,0,:,:].cpu().numpy())
        dices_f_net[:,i]                   = dice_ACDC(warped_seg_f_net[0,0,:,:].cpu().numpy(),fixed_seg[0,0,:,:].cpu().numpy())
        dices_f_net_plus[:,i]              = dice_ACDC(warped_seg_f_net_plus[0,0,:,:].cpu().numpy(),fixed_seg[0,0,:,:].cpu().numpy())
        dices_f_net_plus_cascade[:,i]      = dice_ACDC(warped_seg_f_net_plus_cascade[0,0,:,:].cpu().numpy(),fixed_seg[0,0,:,:].cpu().numpy())
        if diffeo == 1:
            dices_f_net_diff[:,i]              = dice_ACDC(warped_seg_f_net_diff[0,0,:,:].cpu().numpy(),fixed_seg[0,0,:,:].cpu().numpy())
            dices_f_net_plus_diff[:,i]         = dice_ACDC(warped_seg_f_net_plus_diff[0,0,:,:].cpu().numpy(),fixed_seg[0,0,:,:].cpu().numpy())
            dices_f_net_plus_cascade_diff[:,i] = dice_ACDC(warped_seg_f_net_plus_cascade_diff[0,0,:,:].cpu().numpy(),fixed_seg[0,0,:,:].cpu().numpy())

# path to NiftyReg images        
path = '/home/jmeyer/storage/students/janmeyer_711878/data/Nifti/ACDC/Nifti_FullySampled'
image_pairs = [basename(f.path) for f in scandir(path) if f.is_dir() and f.name.startswith('ImagePair')]

# read in corresponding images from NiftyReg folder
for i, image_pair in enumerate(image_pairs):
    # read in segmentations from folder 
    warped_seg  = nibabel.load(join(path, image_pair, 'WarpedSegmentation.nii'))
    fixed_seg   = nibabel.load(join(path, image_pair, 'FixedSegmentation.nii'))
    # convert to float array
    warped_seg_NiftyReg = np.array(warped_seg.get_fdata(), dtype='float32')
    fixed_seg_NiftyReg  = np.array(fixed_seg.get_fdata(), dtype='float32')
    # calculate Dice scores
    dices_NiftyReg[:,i] = dice_ACDC(warped_seg_NiftyReg,fixed_seg_NiftyReg)

# for testing to see whether the csv-file is correctly read-in
#print('Mean Dice scores:\n    Baseline          {}\n    NiftyReg          {}\n    VoxelMorph        {}\n    Fourier-Net       {}\n    Fourier-Net+      {}\n    4xFourier-Net+   {}'.format(np.mean(dices_baseline), np.mean(dices_NiftyReg), np.mean(dices_voxelmorph), np.mean(dices_f_net), np.mean(dices_f_net_plus), np.mean(dices_f_net_plus_cascade)))

# save Dice scores to csv file
f = open(csv_name, 'a')
with f:
    writer = csv.writer(f)   
    # write Dice scores for baseline first
    writer.writerow(dices_baseline[0,:]) 
    writer.writerow(dices_baseline[1,:])
    writer.writerow(dices_baseline[2,:])
    writer.writerow(dices_baseline[3,:]) 
    # then NiftyReg
    writer.writerow(dices_NiftyReg[0,:]) 
    writer.writerow(dices_NiftyReg[1,:])
    writer.writerow(dices_NiftyReg[2,:])
    writer.writerow(dices_NiftyReg[3,:])   
    # then VoxelMorph
    writer.writerow(dices_voxelmorph[0,:]) 
    writer.writerow(dices_voxelmorph[1,:])
    writer.writerow(dices_voxelmorph[2,:])
    writer.writerow(dices_voxelmorph[3,:])      
    # then Fourier-Net
    writer.writerow(dices_f_net[0,:]) 
    writer.writerow(dices_f_net[1,:])
    writer.writerow(dices_f_net[2,:])
    writer.writerow(dices_f_net[3,:])          
    # then Fourier-Net+
    writer.writerow(dices_f_net_plus[0,:]) 
    writer.writerow(dices_f_net_plus[1,:])
    writer.writerow(dices_f_net_plus[2,:])
    writer.writerow(dices_f_net_plus[3,:])             
    # then 4xCascaded Fourier-Net+
    writer.writerow(dices_f_net_plus_cascade[0,:]) 
    writer.writerow(dices_f_net_plus_cascade[1,:])
    writer.writerow(dices_f_net_plus_cascade[2,:])
    writer.writerow(dices_f_net_plus_cascade[3,:])       
    if diffeo == 1:
        # then Diff-Fourier-Net
        writer.writerow(dices_f_net_diff[0,:]) 
        writer.writerow(dices_f_net_diff[1,:])
        writer.writerow(dices_f_net_diff[2,:])
        writer.writerow(dices_f_net_diff[3,:])          
        # then Diff-Fourier-Net+Diff
        writer.writerow(dices_f_net_plus_diff[0,:]) 
        writer.writerow(dices_f_net_plus_diff[1,:])
        writer.writerow(dices_f_net_plus_diff[2,:])
        writer.writerow(dices_f_net_plus_diff[3,:])             
        # then Diff-4xCascaded Fourier-Net+
        writer.writerow(dices_f_net_plus_cascade_diff[0,:]) 
        writer.writerow(dices_f_net_plus_cascade_diff[1,:])
        writer.writerow(dices_f_net_plus_cascade_diff[2,:])
        writer.writerow(dices_f_net_plus_cascade_diff[3,:])       
    