import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
import matplotlib.pyplot as plt 
import torch
import torch.nn.functional as F
from skimage.io import imread
import neurite as ne
from Models import *
from argparse import ArgumentParser
from natsort import natsorted
from os import listdir
import warnings
warnings.filterwarnings("ignore")
import pystrum
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
parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
parser.add_argument("--domain_sim", type=int,
                    dest="domain_sim", default=0,
                    help="choose which domain the similarity loss should be applied: image space (0) or k-space (1)")
opt = parser.parse_args()

start_channel = opt.start_channel
FT_size = [opt.FT_size_x,opt.FT_size_y]
mode = opt.mode
domain_sim = opt.domain_sim

dataset = 'ACDC'
input_shape = [216,256]
diffeo = 0
device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else 
transform = SpatialTransform().to(device)
transform_voxelmorph = SpatialTransformer(input_shape, mode = 'nearest').to(device)

# define paths to images and segmentations
if mode == 0:
    folder = 'FullySampled'
    folder_nifti = 'Nifti_FullySampled'
elif mode == 1:
    folder = 'AccFactor04'
    folder_nifti = 'Nifti_Acc4'
elif mode == 2:
    folder = 'AccFactor08'
    folder_nifti = 'Nifti_Acc8'
elif mode == 3:
    folder = 'AccFactor10'
    folder_nifti = 'Nifti_Acc10'

pathname1 = '/home/jmeyer/storage/students/janmeyer_711878/data/ACDC/' + folder + '/Test/patient081/Slice2/Image_Frame01.png'
pathname2 = '/home/jmeyer/storage/students/janmeyer_711878/data/ACDC/' + folder + '/Test/patient081/Slice2/Image_Frame07.png' #13

pathname1_nifti = '/home/jmeyer/storage/students/janmeyer_711878/data/Nifti/ACDC/' + folder_nifti + '/ImagePair0003/WarpedImage.nii'
pathname2_nifti = '/home/jmeyer/storage/students/janmeyer_711878/data/Nifti/ACDC/' + folder_nifti + '/ImagePair0003/WarpedSegmentation.nii'
pathname3_nifti = '/home/jmeyer/storage/students/janmeyer_711878/data/Nifti/ACDC/' + folder_nifti + '/ImagePair0003/ControlPointGrid.nii'

# path to save the plots to
save_path = '/home/jmeyer/storage/students/janmeyer_711878/Master-Arbeit/Thesis/Images/'

# read in images 
moving_image = imread(pathname1, as_gray=True)/255
fixed_image = imread(pathname2, as_gray=True)/255

# read in nifti images from folder 
warped_image_nifti  = nibabel.load(pathname1_nifti)
warped_image_nifti  = np.array(warped_image_nifti.get_fdata(), dtype='float32')

# read in segmentations (only have value 0 for background and 1,2,3 for structures)
moving_seg = imread(pathname1.replace('Image','Segmentation'), as_gray=True)/3
fixed_seg = imread(pathname2.replace('Image','Segmentation'), as_gray=True)/3

# read in nifti segmentations from folder 
warped_seg_nifti  = nibabel.load(pathname2_nifti)
warped_seg_nifti  = np.array(warped_seg_nifti.get_fdata(), dtype='float32')

# read in nifti images from folder 
grid_nifti  = nibabel.load(pathname3_nifti)
grid_nifti  = np.array(grid_nifti.get_fdata(), dtype='float32')

# convert to tensors and add singleton dimension for the correct size
moving_image        = torch.from_numpy(moving_image)
fixed_image         = torch.from_numpy(fixed_image)
moving_seg          = torch.from_numpy(moving_seg)
fixed_seg           = torch.from_numpy(fixed_seg)
warped_image_nifti  = torch.from_numpy(warped_image_nifti)
warped_seg_nifti    = torch.from_numpy(warped_seg_nifti)
grid_nifti          = torch.from_numpy(grid_nifti)

# interpolate all images and segmentations to the same size
moving_image        = F.interpolate(moving_image.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').float().to(device)
fixed_image         = F.interpolate(fixed_image.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').float().to(device)
warped_image_nifti  = F.interpolate(warped_image_nifti.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').float().to(device)

# use nearest interpolation for the segmentations to preserve labels
moving_seg          = F.interpolate(moving_seg.unsqueeze(0).unsqueeze(0), (216,256), mode='nearest').float().to(device)
fixed_seg           = F.interpolate(fixed_seg.unsqueeze(0).unsqueeze(0), (216,256), mode='nearest').float().to(device)
warped_seg_nifti    = F.interpolate(warped_seg_nifti.unsqueeze(0).unsqueeze(0), (216,256), mode='nearest').float().to(device)
#grid_nifti          = F.interpolate(grid_nifti.unsqueeze(0).unsqueeze(0), (216,256), mode='nearest').float().to(device)

# init different models
model_voxelmorph         = VxmDense(inshape=input_shape, nb_unet_features=32, bidir=False, nb_unet_levels=4).to(device)
model_f_net              = Fourier_Net(2, 2, start_channel, diffeo).to(device)
model_f_net_plus         = Fourier_Net_plus(2, 2, start_channel, diffeo, FT_size).to(device) 
model_f_net_plus_cascade = Cascade(2, 2, start_channel, diffeo, FT_size).to(device)

# load different models
path_voxelmorph = './ModelParameters-{}/Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}/'.format(dataset,0,0.01,0.0001,0) # for voxelmorph 0 is MSE loss
modelpath_voxelmorph = path_voxelmorph + natsorted(listdir(path_voxelmorph))[-1]
model_voxelmorph.load_state_dict(torch.load(modelpath_voxelmorph))
model_voxelmorph.eval()

path_f_net      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Sim_{}_Pth/'.format(dataset,0,diffeo,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,0,domain_sim)
modelpath_f_net = path_f_net + natsorted(listdir(path_f_net))[-1]
model_f_net.load_state_dict(torch.load(modelpath_f_net))
model_f_net.eval()

path_f_net_plus      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Sim_{}_Pth/'.format(dataset,1,diffeo,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,0,domain_sim)
modelpath_f_net_plus = path_f_net_plus + natsorted(listdir(path_f_net_plus))[-1]
model_f_net_plus.load_state_dict(torch.load(modelpath_f_net_plus))
model_f_net_plus.eval()

path_f_net_plus_cascade      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Sim_{}_Pth/'.format(dataset,2,diffeo,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,0,domain_sim)
modelpath_f_net_plus_cascade = path_f_net_plus_cascade + natsorted(listdir(path_f_net_plus_cascade))[-1]
model_f_net_plus_cascade.load_state_dict(torch.load(modelpath_f_net_plus_cascade))
model_f_net_plus_cascade.eval()

with torch.no_grad():
    # get displacements for each model
    warped_image_voxelmorph, V_voxelmorph   = model_voxelmorph(moving_image,fixed_image)
    V_f_net                                 = model_f_net(moving_image,fixed_image)
    V_f_net_plus                            = model_f_net_plus(moving_image,fixed_image)
    V_f_net_plus_cascade                    = model_f_net_plus_cascade(moving_image,fixed_image)

# warp moving images
__, warped_image_f_net              = transform(moving_image, V_f_net.permute(0, 2, 3, 1), mod = 'nearest')
__, warped_image_f_net_plus         = transform(moving_image, V_f_net_plus.permute(0, 2, 3, 1), mod = 'nearest')
__, warped_image_f_net_plus_cascade = transform(moving_image, V_f_net_plus_cascade.permute(0, 2, 3, 1), mod = 'nearest')

# sort images and titles
images = [moving_image,fixed_image,warped_image_nifti,warped_image_voxelmorph,warped_image_f_net,warped_image_f_net_plus,warped_image_f_net_plus_cascade]
titles = ['Moving','Fixed','NiftyReg','VoxelMorph','Fourier-Net','Fourier-Net+','4xFourier-Net+'] # titles for each grid

# warp segmentations
warped_seg_voxelmorph               = transform_voxelmorph(moving_seg, V_voxelmorph) #.permute(0, 2, 3, 1), mod = 'nearest'
__, warped_seg_f_net                = transform(moving_seg, V_f_net.permute(0, 2, 3, 1), mod = 'nearest')
__, warped_seg_f_net_plus           = transform(moving_seg, V_f_net_plus.permute(0, 2, 3, 1), mod = 'nearest')
__, warped_seg_f_net_plus_cascade   = transform(moving_seg, V_f_net_plus_cascade.permute(0, 2, 3, 1), mod = 'nearest')

# sort segmentations
segmentations = [moving_seg,fixed_seg,warped_seg_nifti,warped_seg_voxelmorph,warped_seg_f_net,warped_seg_f_net_plus,warped_seg_f_net_plus_cascade]
dices = []
for seg in segmentations:
    dices.append(np.mean(dice_ACDC(fixed_seg.data.cpu().numpy()[0, 0, ...],seg.data.cpu().numpy()[0, 0, ...])))

# plot images and segmentations
assert len(titles) == len(images), f"Every images needs a title!"
"""
fig, axs = plt.subplots(nrows=1, ncols=len(titles), figsize=(6, 1)) #nrows=2
# plot images
for i in range(len(titles)):
    ax = axs[i] #[0][i]
    # turn off axis
    ax.axis('off')
    # add titles
    #ax.title.set_text(titles[i])
    # show image
    im_ax = ax.imshow(images[i].data.cpu().numpy()[0, 0, ...], cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.savefig('Images.png') #save_path+
plt.close()

fig, axs = plt.subplots(nrows=1, ncols=len(titles), figsize=(6, 1)) #figsize=(5, 1)
# plot segmentations
for i in range(len(titles)):
    ax = axs[i] #[1][i]
    # turn off axis
    ax.axis('off')
    # show image
    im_ax = ax.imshow(seg[i].data.cpu().numpy()[0, 0, ...])
plt.axis('off')
plt.savefig('Segmentations.png') #save_path+
plt.close()
"""

fig, axs = plt.subplots(nrows=2, ncols=len(titles), figsize=(48, 8))  #, gridspec_kw = {'wspace':0, 'hspace':0}
# plot segmentations
for i in range(len(titles)):
    ax = axs[0][i]
    ax.axis('off')
    ax.set_title(titles[i], fontsize=22) # title only for the images
    im_ax = ax.imshow(images[i].data.cpu().numpy()[0, 0, ...], cmap='gray', vmin=0, vmax = 1)
    ax = axs[1][i]
    ax.axis('off')
    ax.set_title('Dice: {:.5f}'.format(dices[i]), fontsize=22, y=0.125, color='white') # display dice scores
    im_ax = ax.imshow(segmentations[i].data.cpu().numpy()[0, 0, ...])
plt.axis('off')
#plt.tight_layout(pad=-3)
plt.subplots_adjust(wspace=-0.75, hspace=0) # make subplots tighter
plt.savefig(save_path+'TestExamples.png', bbox_inches="tight") 
plt.close()

"""
# generate grid for visualization
grid = torch.from_numpy(pystrum.pynd.ndutils.bw_grid(vol_shape=input_shape, spacing=2)).float().to(device).unsqueeze(0).unsqueeze(0)

# warp grid with models
grid_voxelmorph             = transform_voxelmorph(grid, V_voxelmorph) # .permute(0, 2, 3, 1), mod = 'bilinear'
__, grid_f_net              = transform(grid, V_f_net.permute(0, 2, 3, 1), mod = 'bilinear')
__, grid_f_net_plus         = transform(grid, V_f_net_plus.permute(0, 2, 3, 1), mod = 'bilinear')
__, grid_f_net_plus_cascade = transform(grid, V_f_net_plus_cascade.permute(0, 2, 3, 1), mod = 'bilinear')

# visualize grids for each model
grids = [grid.data.cpu().numpy()[0, 0, ...],grid.data.cpu().numpy()[0, 0, ...],grid_nifti,grid_voxelmorph.data.cpu().numpy()[0, 0, ...],grid_f_net.data.cpu().numpy()[0, 0, ...],grid_f_net_plus.data.cpu().numpy()[0, 0, ...],grid_f_net_plus_cascade.data.cpu().numpy()[0, 0, ...]] 
(fig, axs) = ne.plot.slices(grids, width=6, show=False) #titles=titles,
plt.axis('off')
plt.savefig('Grids.png') #save_path+
plt.close()

# plot all flows
flows = [V_voxelmorph,V_f_net,V_f_net_plus,V_f_net_plus_cascade] 
(fig, axs) = ne.plot.flow(flows, width=5,show=False)
plt.axis('off')
plt.savefig('Flows.png') #save_path+
plt.close()
"""