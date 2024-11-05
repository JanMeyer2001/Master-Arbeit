from mrirecon import sense
from argparse import ArgumentParser
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
from skimage.metrics import structural_similarity, mean_squared_error
from natsort import natsorted
import sigpy.mri as mr
import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float,
                    dest="learning_rate", default=1e-4, help="learning rate")
parser.add_argument("--lambda", type=float,
                    dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=16,
                    help="number of start channels")
parser.add_argument("--choose_loss", type=int, dest="choose_loss", default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2), SSIM (3)")
parser.add_argument("--mode", type=int, dest="mode", default='1',
                    help="choose dataset mode: 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
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
parser.add_argument("--epoch", type=int,
                    dest="epoch", default=14, 
                    help="choose which epoch is used in the evaluation (for input 0 the best version will be chosen)")
opt = parser.parse_args()

learning_rate = opt.learning_rate
start_channel = opt.start_channel
smooth = opt.smth_lambda
choose_loss = opt.choose_loss
mode = opt.mode
model_num = opt.model_num
diffeo = opt.diffeo
FT_size = [opt.FT_size_x,opt.FT_size_y]
epoch = opt.epoch

dataset = 'CMRxRecon'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import image/k-space data and coil sensitivity maps for all slices and frames
assert mode > 0 or mode <= 3, f"Expected mode to be between 1 and 3, but got: {mode}"
data_set = DatasetCMRxReconstruction('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', False, mode) 
data_generator = Data.DataLoader(dataset=data_set, batch_size=1, shuffle=False, num_workers=4)

H = 246
W = 512
input_shape = [1,1,H,W] #data_set.__getitem__(0)[0].unsqueeze(0).shape

# select and import models for motion correction
assert model_num >= 0 or model_num <= 3, f"Expected model_num to be between 0 and 3, but got: {model_num}"
assert diffeo == 0 or diffeo == 1, f"Expected diffeo to be either 0 or 1, but got: {diffeo}"
if model_num == 0:
    model = Fourier_Net(2, 2, start_channel, diffeo).to(device) 
    # TODO: remove '2D_CMR_Registration/' after debugging
    path = './2D_CMR_Registration/ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
    transform = SpatialTransform().to(device)
elif model_num == 1:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Fourier_Net_plus(2, 2, start_channel, diffeo, FT_size).to(device) 
    path = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
    transform = SpatialTransform().to(device)
elif model_num == 2:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade(2, 2, start_channel, diffeo, FT_size).to(device) 
    path = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
    transform = SpatialTransform().to(device)
elif model_num == 3:
    model = VxmDense(inshape=input_shape, nb_unet_features=32, bidir=False, nb_unet_levels=4).to(device)  #, int_steps=7, int_downsize=2
    transform = SpatialTransformer(input_shape, mode = 'nearest').to(device)    

if epoch == 0:
    # choose best model
    print('Using model: {}'.format(natsorted(os.listdir(path))[-1]))
    modelpath = path + natsorted(os.listdir(path))[-1]
else:
    # choose model after certain epoch of training
    modelpath = [f.path for f in scandir(path) if f.is_file() and not (f.name.find('Epoch_{:04d}'.format(epoch)) == -1)][0]
    print('Using model: {}'.format(basename(modelpath)))

model.load_state_dict(torch.load(modelpath))
model.eval()
transform.eval()

for data in data_generator:
    # get data
    images_fullysampled = data[0].squeeze().numpy()         # array with size (H,W,F)
    images_subsampled   = data[1].permute(1,2,0,3).numpy()  # array with size (H,W,1,F)
    masks               = data[2].squeeze().numpy()         # array with size (H,W,C,F)
    k_spaces            = data[3].squeeze().numpy()         # array with size (H,W,C,F)
    coil_maps           = data[4].squeeze().numpy()         # array with size (H,W,C,F)

    num_frames = k_spaces.shape[3]    # number of frames F
    num_coils  = k_spaces.shape[2]    # number of coils C
    max_iter   = 10                      # number of iterations
    tol        = 1e-12                   # error tolerance

    """
    # perform iterative SENSE reconstruction (no motion)
    img_recon = SENSE_iter(k_space[0].squeeze().numpy(),max_iter,mask[0].squeeze().squeeze().numpy(),coil_map[0].squeeze().squeeze().numpy(),img_fullySampled[0].squeeze().squeeze().numpy())
    # plot the reconstructed image
    plt.subplot(1, 3, 1)
    plt.imshow(img_subSampled[0].squeeze().squeeze(), cmap='gray')
    plt.title('SubSampled')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img_fullySampled[0].squeeze().squeeze(), cmap='gray')
    plt.title('Fully Sampled')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img_recon, cmap='gray')
    plt.title('Reconstruction')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('reconstructedImage.png') 
    plt.close
    """
    # init numpy array for flow fields
    flows = np.zeros((H,W,2,num_frames))
    
    for frame_num in range(num_frames-1):
        # get displacements relative to the first image (first entry is deliberately left empty)
        if model_num == 3:
            warped_mov, flow = model(torch.from_numpy(images_subsampled[:,:,0,0]).unsqueeze(0).unsqueeze(0).float().to(device), torch.from_numpy(images_subsampled[:,:,0,frame_num+1]).unsqueeze(0).unsqueeze(0).float().to(device))
        else:    
            flow, features_disp = model(torch.from_numpy(images_subsampled[:,:,0,0]).unsqueeze(0).unsqueeze(0).float().to(device), torch.from_numpy(images_subsampled[:,:,0,frame_num+1]).unsqueeze(0).unsqueeze(0).float().to(device))
        flows[:,:,:,frame_num+1] = flow.squeeze().permute(1,2,0).cpu().detach().numpy()

    # old iterative SENSE code extended for motion-compensation
    #img_recon_motion = SENSE_motion_compensated(subsampled_k_space=k_spaces,num_iter=10,mask=mask,est_sensemap=coil_maps,flow=flows,transform=transform)   

    # perform motion-compensated SENSE reconstruction (takes some time...)
    A  = ForwardOperator                                        # image space to k-space
    AH = AdjointOperator                                        # k-space to image space
    noisy = AH(k_spaces, masks, coil_maps, flows, transform)     # get naive starting image
    # conjugate gradient optimization for image reconstruction
    img_recon_motion = conjugate_gradient([noisy, k_spaces, masks, coil_maps, flows, transform], A, AH, max_iter, tol)
    # coil combine the images
    img_recon_motion = np.sqrt(np.sum(np.abs(img_recon_motion)**2,2))
      
    # plot the reconstructed motion-compensated frames
    for frame in range(num_frames):
        plt.subplot(num_frames, 3, 3*frame+1)
        plt.imshow(images_subsampled[:,:,0,frame], cmap='gray')
        if frame == 0:
            plt.title('Subsampled') # Frames
        plt.axis('off')
        plt.subplot(num_frames, 3, 3*frame+2)
        plt.imshow(images_fullysampled[:,:,frame], cmap='gray')
        if frame == 0:
            plt.title('Fully Sampled') # Frames
        plt.axis('off')
        plt.subplot(num_frames, 3, 3*frame+3)
        plt.imshow(img_recon_motion[:,:,frame], cmap='gray')
        if frame == 0:
            plt.title('Motion-Reconstructed') # Frames
        plt.axis('off')
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.00005,hspace=0.00005) 
    plt.savefig('MotionReconstructedImage.png') 
    plt.close

    # TODO: evaluate reconstructed frames