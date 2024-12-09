import torch
from Models import *
from Functions import *
from skimage.metrics import mean_squared_error
from piq import psnr, ssim, haarpsi
from natsort import natsorted
import warnings
warnings.filterwarnings("ignore")

# define parameters
learning_rate = 1e-4
start_channel = 16
smooth = 0.01
FT_size = [24,24]
epoch = 14
L = 4
dataset = 'CMRxRecon'
device  = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")
diffeo = 0
H = 246
W = 512
input_shape = [H,W] 
modes = [1,2,3]         # Acc factors 4, 8, 10
boldpositions = [1,2,2] # positions for best model

transform = SpatialTransform().to(device)
transform_voxelmorph = SpatialTransformer(input_shape, mode = 'nearest').to(device)

for mode in modes:
    save_name = '/home/jmeyer/storage/students/janmeyer_711878/Master-Arbeit/Images/ResultsReconstruction_mode{}.png'.format(mode)
    # print mode
    if mode == 1:
        print('R=4')
        folder = 'AccFactor04'
    elif mode == 2:
        print('R=8')
        folder = 'AccFactor08'
    elif mode == 3:
        print('R=10')
        folder = 'AccFactor10'

    # import image/k-space data and coil sensitivity maps for all slices and frames
    data = getExampleMotionCorruptedData(folder,'P001','Slice0',H,W,L,transform)

    # get data
    images_fullysampled = data[0]               # tensor with size (F/2,H,W)
    images_subsampled   = data[1]               # tensor with size (L+1,F/2,H,W) --> L is the number of simulated lung movement
    masks               = data[2].unsqueeze(0)  # tensor with size (1,1,C,L+1*F/2,H,W)
    k_spaces            = data[3].unsqueeze(0)  # tensor with size (1,1,C,L+1*F/2,H,W)
    coil_maps           = data[4].unsqueeze(0)  # tensor with size (1,1,C,L+1*F/2,H,W)

    num_frames = images_fullysampled.shape[0]   # number of frames F/2
    num_coils  = k_spaces.shape[2]              # number of coils C
    max_iter   = 10                             # number of iterations
    tol        = 1e-12                          # error tolerance

    # init different models
    model_voxelmorph         = VxmDense(inshape=input_shape, nb_unet_features=32, bidir=False, nb_unet_levels=4).to(device)
    model_f_net              = Fourier_Net(2, 2, start_channel, diffeo, False).to(device)
    model_f_net_plus         = Fourier_Net_plus(2, 2, start_channel, diffeo, FT_size).to(device) 
    model_f_net_plus_cascade = Cascade(2, 2, start_channel, diffeo, FT_size).to(device)

    # load different models
    path_voxelmorph = './ModelParameters-{}/Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}/'.format(dataset,0,0.01,0.0001,mode) # for voxelmorph 0 is MSE loss
    modelpath_voxelmorph = path_voxelmorph + natsorted(listdir(path_voxelmorph))[-1]
    model_voxelmorph.load_state_dict(torch.load(modelpath_voxelmorph))
    model_voxelmorph.eval()

    path_f_net      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,0,diffeo,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode)
    modelpath_f_net = path_f_net + natsorted(listdir(path_f_net))[-1]
    model_f_net.load_state_dict(torch.load(modelpath_f_net))
    model_f_net.eval()

    path_f_net_plus      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,1,diffeo,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode)
    modelpath_f_net_plus = path_f_net_plus + natsorted(listdir(path_f_net_plus))[-1]
    model_f_net_plus.load_state_dict(torch.load(modelpath_f_net_plus))
    model_f_net_plus.eval()

    path_f_net_plus_cascade      = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,2,diffeo,1,start_channel,FT_size[0],FT_size[1],0.01,0.0001,mode)
    modelpath_f_net_plus_cascade = path_f_net_plus_cascade + natsorted(listdir(path_f_net_plus_cascade))[-1]
    model_f_net_plus_cascade.load_state_dict(torch.load(modelpath_f_net_plus_cascade))
    model_f_net_plus_cascade.eval()

    titles = ["VoxelMorph", "Fourier-Net", "Fourier-Net+", "4xFourier-Net+"]
    method_images = []
    metrics = np.zeros((4,4))

    # init torch tensor for flow fields
    flows_voxelmorph            = torch.zeros(L+1,num_frames,H,W,2)
    flows_f_net                 = torch.zeros(L+1,num_frames,H,W,2)
    flows_f_net_plus            = torch.zeros(L+1,num_frames,H,W,2)
    flows_f_net_plus_cascade    = torch.zeros(L+1,num_frames,H,W,2)

    # init torch tensor for subsampled aligned images
    images_subsampled_voxelmorph         = images_subsampled
    images_subsampled_f_net              = images_subsampled
    images_subsampled_f_net_plus         = images_subsampled
    images_subsampled_f_net_plus_cascade = images_subsampled

    for frame_num in range(num_frames):
        for i in range(L):   
            # get flows for VoxelMorph
            images_subsampled_voxelmorph[i+1,frame_num,:,:], flow  = model_voxelmorph(images_subsampled[i+1,frame_num,:,:].unsqueeze(0).unsqueeze(0).float(), images_subsampled[0,frame_num,:,:].unsqueeze(0).unsqueeze(0).float())
            flows_voxelmorph[i+1,frame_num,:,:,:]       = flow.squeeze().permute(1,2,0)
            # get flows for Fourier-Net
            flow, features_disp                         = model_f_net(images_subsampled[i+1,frame_num,:,:].unsqueeze(0).unsqueeze(0).float(), images_subsampled[0,frame_num,:,:].unsqueeze(0).unsqueeze(0).float())
            grid, images_subsampled_f_net[i+1,frame_num,:,:]  = transform(images_subsampled[i+1,frame_num,:,:].unsqueeze(0).unsqueeze(0).float(), flows_f_net[i+1,frame_num,:,:,:].unsqueeze(0).float())
            flows_f_net[i+1,frame_num,:,:,:]            = flow.squeeze().permute(1,2,0)
            # get flows for Fourier-Net+
            flow, features_disp                         = model_f_net_plus(images_subsampled[i+1,frame_num,:,:].unsqueeze(0).unsqueeze(0).float(), images_subsampled[0,frame_num,:,:].unsqueeze(0).unsqueeze(0).float())
            grid, images_subsampled_f_net_plus[i+1,frame_num,:,:]  = transform(images_subsampled[i+1,frame_num,:,:].unsqueeze(0).unsqueeze(0).float(), flows_f_net_plus[i+1,frame_num,:,:,:].unsqueeze(0).float())
            flows_f_net_plus[i+1,frame_num,:,:,:]       = flow.squeeze().permute(1,2,0)
            # get flows for 4xFourier-Net+
            flow, features_disp                           = model_f_net_plus_cascade(images_subsampled[i+1,frame_num,:,:].unsqueeze(0).unsqueeze(0).float(), images_subsampled[0,frame_num,:,:].unsqueeze(0).unsqueeze(0).float())
            grid, images_subsampled_f_net_plus_cascade[i+1,frame_num,:,:]    = transform(images_subsampled[i+1,frame_num,:,:].unsqueeze(0).unsqueeze(0).float(), flows_f_net_plus_cascade[i+1,frame_num,:,:,:].unsqueeze(0).float())
            flows_f_net_plus_cascade[i+1,frame_num,:,:,:] = flow.squeeze().permute(1,2,0)

    # reshape tensors into correct size for the reconstruction pipeline
    images_subsampled_voxelmorph         = torch.reshape(images_subsampled_voxelmorph, ((L+1)*num_frames,H,W)).unsqueeze(0)
    images_subsampled_f_net              = torch.reshape(images_subsampled_f_net, ((L+1)*num_frames,H,W)).unsqueeze(0)
    images_subsampled_f_net_plus         = torch.reshape(images_subsampled_f_net_plus, ((L+1)*num_frames,H,W)).unsqueeze(0)
    images_subsampled_f_net_plus_cascade = torch.reshape(images_subsampled_f_net_plus_cascade, ((L+1)*num_frames,H,W)).unsqueeze(0)
    flows_voxelmorph         = torch.reshape(flows_voxelmorph, ((L+1)*num_frames,H,W,2)).unsqueeze(0).unsqueeze(0)
    flows_f_net              = torch.reshape(flows_f_net, ((L+1)*num_frames,H,W,2)).unsqueeze(0).unsqueeze(0)
    flows_f_net_plus         = torch.reshape(flows_f_net_plus, ((L+1)*num_frames,H,W,2)).unsqueeze(0).unsqueeze(0)
    flows_f_net_plus_cascade = torch.reshape(flows_f_net_plus_cascade, ((L+1)*num_frames,H,W,2)).unsqueeze(0).unsqueeze(0)

    # init pipeline and reconstruct images
    recon                               = ReconDCPMMotion(max_iter=max_iter, coil_axis=2)
    img_recon_motion_voxelmorph         = torch.reshape(normalize(torch.abs(recon(images_subsampled_voxelmorph, k_spaces, masks, coil_maps, flows_voxelmorph, transform, num_frames)).squeeze()), (L+1,num_frames,H,W))
    img_recon_motion_f_net              = torch.reshape(normalize(torch.abs(recon(images_subsampled_f_net, k_spaces, masks, coil_maps, flows_f_net, transform, num_frames)).squeeze()), (L+1,num_frames,H,W))
    img_recon_motion_f_net_plus         = torch.reshape(normalize(torch.abs(recon(images_subsampled_f_net_plus, k_spaces, masks, coil_maps, flows_f_net_plus, transform, num_frames)).squeeze()), (L+1,num_frames,H,W))
    img_recon_motion_f_net_plus_cascade = torch.reshape(normalize(torch.abs(recon(images_subsampled_f_net_plus_cascade, k_spaces, masks, coil_maps, flows_f_net_plus_cascade, transform, num_frames)).squeeze()), (L+1,num_frames,H,W))
        
    # ground truth image
    images_fullysampled = images_fullysampled[0,:,:].unsqueeze(0).unsqueeze(0)

    # get metrics for VoxelMorph
    img_recon_motion    = img_recon_motion_voxelmorph[3,0,:,:].unsqueeze(0).unsqueeze(0)
    metrics[0,0]        = haarpsi(img_recon_motion, images_fullysampled, data_range=1).item()*100
    metrics[0,1]        = psnr(img_recon_motion, images_fullysampled, data_range=1).item()
    metrics[0,2]        = ssim(img_recon_motion, images_fullysampled, data_range=1).item()*100
    metrics[0,3]        = mean_squared_error(images_fullysampled.cpu().detach().numpy(), img_recon_motion.cpu().detach().numpy())*100
    method_images.append(img_recon_motion.squeeze().squeeze().detach().cpu().numpy())
    # get metrics for Fourier-Net
    img_recon_motion    = img_recon_motion_f_net[2,0,:,:].unsqueeze(0).unsqueeze(0)
    metrics[1,0]        = haarpsi(img_recon_motion, images_fullysampled, data_range=1).item()*100
    metrics[1,1]        = psnr(img_recon_motion, images_fullysampled, data_range=1).item()
    metrics[1,2]        = ssim(img_recon_motion, images_fullysampled, data_range=1).item()*100
    metrics[1,3]        = mean_squared_error(images_fullysampled.cpu().detach().numpy(), img_recon_motion.cpu().detach().numpy())*100
    method_images.append(img_recon_motion.squeeze().squeeze().detach().cpu().numpy())
    # get metrics for Fourier-Net+
    img_recon_motion    = img_recon_motion_f_net_plus[0,0,:,:].unsqueeze(0).unsqueeze(0)
    metrics[2,0]        = haarpsi(img_recon_motion, images_fullysampled, data_range=1).item()*100
    metrics[2,1]        = psnr(img_recon_motion, images_fullysampled, data_range=1).item()
    metrics[2,2]        = ssim(img_recon_motion, images_fullysampled, data_range=1).item()*100
    metrics[2,3]        = mean_squared_error(images_fullysampled.cpu().detach().numpy(), img_recon_motion.cpu().detach().numpy())*100
    method_images.append(img_recon_motion.squeeze().squeeze().detach().cpu().numpy())
    # get metrics for 4xFourier-Net+
    img_recon_motion    = img_recon_motion_f_net_plus_cascade[1,0,:,:].unsqueeze(0).unsqueeze(0)
    metrics[3,0]        = haarpsi(img_recon_motion, images_fullysampled, data_range=1).item()*100
    metrics[3,1]        = psnr(img_recon_motion, images_fullysampled, data_range=1).item()
    metrics[3,2]        = ssim(img_recon_motion, images_fullysampled, data_range=1).item()*100
    metrics[3,3]        = mean_squared_error(images_fullysampled.cpu().detach().numpy(), img_recon_motion.cpu().detach().numpy())*100
    method_images.append(img_recon_motion.squeeze().squeeze().detach().cpu().numpy())

    display_and_compare_images(images_fullysampled.squeeze().squeeze(), method_images, titles, metrics, mode, boldpositions[mode-1], save_name)  