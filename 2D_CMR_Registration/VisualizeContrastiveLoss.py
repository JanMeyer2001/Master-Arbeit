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
import flow_vis

parser = ArgumentParser()
parser.add_argument("--mode", type=int, dest="mode", default='1',
                    help="choose dataset mode: 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
opt = parser.parse_args()
mode = opt.mode

lambdas = [0,0.0005,0.001,0.005,0.01,0.05] # weightings for the contrastive loss

start_channel = 16
FT_size = [24,24]
dataset = 'ACDC'
input_shape = [216,256]
diffeo = 0
device = torch.device("cpu") #"cuda" if torch.cuda.is_available() else 
transform = SpatialTransform().to(device)
transform_voxelmorph = SpatialTransformer(input_shape, mode = 'nearest').to(device)

# define paths to images and segmentations
if mode == 1:
    folder = 'AccFactor04'
elif mode == 2:
    folder = 'AccFactor08'
elif mode == 3:
    folder = 'AccFactor10'

pathname1 = '/home/jmeyer/storage/students/janmeyer_711878/data/ACDC/' + folder + '/Test/patient081/Slice2/Image_Frame01.png'
pathname2 = '/home/jmeyer/storage/students/janmeyer_711878/data/ACDC/' + folder + '/Test/patient081/Slice2/Image_Frame07.png' #13

# path to save the plots to
save_path = '/home/jmeyer/storage/students/janmeyer_711878/Master-Arbeit/Thesis/Images/'

# read in images 
moving_image = imread(pathname1, as_gray=True)/255
fixed_image = imread(pathname2, as_gray=True)/255

# read in segmentations (only have value 0 for background and 1,2,3 for structures)
moving_seg = imread(pathname1.replace('Image','Segmentation'), as_gray=True)/3
fixed_seg = imread(pathname2.replace('Image','Segmentation'), as_gray=True)/3

# convert to tensors and add singleton dimension for the correct size
moving_image        = torch.from_numpy(moving_image)
fixed_image         = torch.from_numpy(fixed_image)
moving_seg          = torch.from_numpy(moving_seg)
fixed_seg           = torch.from_numpy(fixed_seg)

# interpolate all images and segmentations to the same size
moving_image        = F.interpolate(moving_image.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').float().to(device)
fixed_image         = F.interpolate(fixed_image.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').float().to(device)

# use nearest interpolation for the segmentations to preserve labels
moving_seg          = F.interpolate(moving_seg.unsqueeze(0).unsqueeze(0), (216,256), mode='nearest').float().to(device)
fixed_seg           = F.interpolate(fixed_seg.unsqueeze(0).unsqueeze(0), (216,256), mode='nearest').float().to(device)

# init different models
model_f_net              = Fourier_Net(2, 2, start_channel, diffeo).to(device)
model_f_net_plus         = Fourier_Net_plus(2, 2, start_channel, diffeo, FT_size).to(device) 
model_f_net_plus_cascade = Cascade(2, 2, start_channel, diffeo, FT_size).to(device)

for l in lambdas:
    # load different models
    path_f_net      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,0,diffeo,5,start_channel,FT_size[0],FT_size[1],l,0.0001,mode)
    modelpath_f_net = path_f_net + natsorted(listdir(path_f_net))[-1]
    model_f_net.load_state_dict(torch.load(modelpath_f_net))
    model_f_net.eval()
    """
    path_f_net_plus      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,1,diffeo,5,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode)
    modelpath_f_net_plus = path_f_net_plus + natsorted(listdir(path_f_net_plus))[-1]
    model_f_net_plus.load_state_dict(torch.load(modelpath_f_net_plus))
    model_f_net_plus.eval()

    path_f_net_plus_cascade      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,2,diffeo,5,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode)
    modelpath_f_net_plus_cascade = path_f_net_plus_cascade + natsorted(listdir(path_f_net_plus_cascade))[-1]
    model_f_net_plus_cascade.load_state_dict(torch.load(modelpath_f_net_plus_cascade))
    model_f_net_plus_cascade.eval()
    """
    with torch.no_grad():
        # get displacements for each model
        V_f_net, __                             = model_f_net(moving_image,fixed_image)
        #V_f_net_plus, __                        = model_f_net_plus(moving_image,fixed_image)
        #V_f_net_plus_cascade, __                = model_f_net_plus_cascade(moving_image,fixed_image)

    # warp moving images
    __, warped_image_f_net              = transform(moving_image, V_f_net.permute(0, 2, 3, 1), mod = 'nearest')
    #__, warped_image_f_net_plus         = transform(moving_image, V_f_net_plus.permute(0, 2, 3, 1), mod = 'nearest')
    #__, warped_image_f_net_plus_cascade = transform(moving_image, V_f_net_plus_cascade.permute(0, 2, 3, 1), mod = 'nearest')

    # warp segmentations
    __, warped_seg_f_net                = transform(moving_seg, V_f_net.permute(0, 2, 3, 1), mod = 'nearest')
    #__, warped_seg_f_net_plus           = transform(moving_seg, V_f_net_plus.permute(0, 2, 3, 1), mod = 'nearest')
    #__, warped_seg_f_net_plus_cascade   = transform(moving_seg, V_f_net_plus_cascade.permute(0, 2, 3, 1), mod = 'nearest')

    # save results
    if l == 0:
        # init lists with moving and fixed images
        images = [moving_image,fixed_image]
        titles = ['Moving','Fixed']
        segmentations = [moving_seg,fixed_seg] 
        # create empty flow fields for moving and fixed image
        empty_flow = torch.zeros_like(V_f_net)
        flows = [empty_flow,empty_flow]
    # add images and segmentations for contrastive loss
    images.append(warped_image_f_net)
    segmentations.append(warped_seg_f_net)
    flows.append(V_f_net) 
    titles.append('\u03BB = {}'.format(l))
   
# init dices
dices = []
 
for seg in segmentations:
    dices.append(np.mean(dice_ACDC(fixed_seg.data.cpu().numpy()[0, 0, ...],seg.data.cpu().numpy()[0, 0, ...])))

# convert to flow images
for i, flow in enumerate(flows):
    flows[i] = flow_vis.flow_to_color(flow.data.cpu().squeeze().permute(1,2,0).numpy(), convert_to_bgr=False)

# plot images and segmentations
assert len(titles) == len(images), f"Every images needs a title!"

fig, axs = plt.subplots(nrows=3, ncols=len(titles), figsize=(48, 12))  #, gridspec_kw = {'wspace':0, 'hspace':0}
# plot segmentations
for i in range(len(titles)):
    # first column with images
    ax = axs[0][i]
    ax.axis('off')
    ax.set_title(titles[i], fontsize=28) # title only for the images
    ax.imshow(images[i].data.cpu().numpy()[0, 0, ...], cmap='gray', vmin=0, vmax = 1)
    # second column with segmentations
    ax = axs[1][i]
    ax.axis('off')
    ax.set_title('Dice: {:.5f}'.format(dices[i]), fontsize=22, y=0.125, color='white') # display dice scores
    ax.imshow(segmentations[i].data.cpu().numpy()[0, 0, ...])
    # third column with flows
    ax = axs[2][i]
    ax.axis('off')
    ax.imshow(flows[i])   
plt.axis('off')
plt.subplots_adjust(wspace=-0.65, hspace=0.05) # make subplots tighter
plt.savefig(save_path+'ContrastiveLoss_Mode{}.png'.format(mode), bbox_inches="tight") 
plt.close()
