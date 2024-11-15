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
print('Loading in data...')
data_set = DatasetCMRxReconstruction('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', False, mode) 
data_generator = Data.DataLoader(dataset=data_set, batch_size=1, shuffle=False, num_workers=4)
print('   Load-in complete!')

H = 246
W = 512
input_shape = [1,1,H,W] #data_set.__getitem__(0)[0].unsqueeze(0).shape

# select and import models for motion correction
assert model_num >= 0 or model_num <= 3, f"Expected model_num to be between 0 and 3, but got: {model_num}"
assert diffeo == 0 or diffeo == 1, f"Expected diffeo to be either 0 or 1, but got: {diffeo}"
print('Selecting model...')
if model_num == 0:
    model = Fourier_Net(2, 2, start_channel, diffeo).to(device) 
    path  = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
    transform = SpatialTransform().to(device)
    print('   Fourier-Net!')
elif model_num == 1:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Fourier_Net_plus(2, 2, start_channel, diffeo, FT_size).to(device) 
    path  = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
    transform = SpatialTransform().to(device)
    print('   Fourier-Net+!')
elif model_num == 2:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade(2, 2, start_channel, diffeo, FT_size).to(device) 
    path  = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
    transform = SpatialTransform().to(device)
    print('   4xFourier-Net+!')
elif model_num == 3:
    model = VxmDense(inshape=input_shape, nb_unet_features=32, bidir=False, nb_unet_levels=4).to(device)  #, int_steps=7, int_downsize=2
    path  = './ModelParameters-{}/Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}/'.format(dataset,choose_loss,smooth,learning_rate,mode)
    transform = SpatialTransformer(input_shape, mode = 'nearest').to(device)
    print('   VoxelMorph!')    

print('Load pre-trained model parameters...')
if epoch == 0:
    # choose best model
    print('   Using parameters: {}'.format(natsorted(os.listdir(path))[-1]))
    modelpath = path + natsorted(os.listdir(path))[-1]
else:
    # choose model after certain epoch of training
    modelpath = [f.path for f in scandir(path) if f.is_file() and not (f.name.find('Epoch_{:04d}'.format(epoch)) == -1)][0]
    print('   Using parameters: {}'.format(basename(modelpath)))

model.load_state_dict(torch.load(modelpath))
model.eval()
transform.eval()

# save test results in a csv file
if model_num == 3:
    csv_name = './TestResults-Reconstruction/TestMetrics-Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}_Epoch{}.csv'.format(dataset,choose_loss,smooth,learning_rate,mode,epoch)
else:
    csv_name = './TestResults-Reconstruction/TestMetrics-Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Epoch{}.csv'.format(model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode,epoch)
f = open(csv_name, 'w')
with f:
    fnames = ['SSIM','MSE','Mean SSIM','Std SSIM','Mean MSE','Std MSE']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()
MSE_test   = []
SSIM_test  = []

print('Begin Reconstruction on {}.'.format(time.ctime()))
for data in data_generator:
    # get data
    images_fullysampled = data[0].squeeze().cpu().detach().numpy()     # array with size (F,H,W)
    images_subsampled   = data[1]               # tensor with size (1,F,H,W)
    masks               = data[2]               # tensor with size (1,1,C,F,H,W)
    k_spaces            = data[3]               # tensor with size (1,1,C,F,H,W)
    coil_maps           = data[4]               # tensor with size (1,1,C,F,H,W)

    num_frames = k_spaces.shape[3]    # number of frames F
    num_coils  = k_spaces.shape[2]    # number of coils C
    max_iter   = 10                   # number of iterations
    tol        = 1e-12                # error tolerance
    
    # init torch tensor for flow fields
    flows = torch.zeros(1,1,num_frames,H,W,2)
    
    for frame_num in range(num_frames-1):
        # get displacements relative to the first image (first entry is deliberately left empty)
        if model_num == 3:
            warped_mov, flow = model(images_subsampled[0,frame_num+1,:,:].unsqueeze(0).unsqueeze(0).float().to(device), images_subsampled[0,0,:,:].unsqueeze(0).unsqueeze(0).float().to(device))
        else:    
            flow, features_disp = model(images_subsampled[0,frame_num+1,:,:].unsqueeze(0).unsqueeze(0).float().to(device), images_subsampled[0,0,:,:].unsqueeze(0).unsqueeze(0).float().to(device))
        flows[:,:,frame_num+1,:,:,:] = flow.squeeze().permute(1,2,0)
    
    # init pipeline and reconstruct images
    recon = ReconDCPMMotion(max_iter=max_iter, coil_axis=2)
    img_recon_motion = torch.abs(recon(images_subsampled, k_spaces, masks, coil_maps, flows, transform, num_frames)).squeeze()
    # normalize (just to be sure) and turn into numpy array
    img_recon_motion = normalize(img_recon_motion).cpu().detach().numpy()
    # evaluate reconstructed frames
    for frame in range(num_frames):
        # get MSE and SSIM between first fully sampled frame and all motion-corrected reconstructed frames
        csv_MSE  = mean_squared_error(images_fullysampled[0,:,:], img_recon_motion[frame,:,:])
        csv_SSIM = structural_similarity(images_fullysampled[0,:,:], img_recon_motion[frame,:,:], data_range=1)    
        MSE_test.append(csv_MSE)
        SSIM_test.append(csv_SSIM)
        # save test results to csv file
        f = open(csv_name, 'a')
        with f:
            writer = csv.writer(f)
            writer.writerow([csv_SSIM, csv_MSE, '-', '-', '-', '-']) 

print('Finished reconstruction on {}.\nPlot test examples...'.format(time.ctime()))
# plot the reconstructed motion-compensated frames
plt.figure(layout='compressed', figsize=(16, 16))
plt.subplots_adjust(wspace=0,hspace=0) 
plt.subplot(3, 2, 1)
plt.imshow(images_subsampled[0,0,:,:].cpu().detach().numpy(), cmap='gray')
if mode == 1:
    plt.title('R=4')
elif mode == 2:
    plt.title('R=8')   
elif mode == 3:
    plt.title('R=10')    
plt.axis('off')
plt.subplot(3, 2, 2)
plt.imshow(images_fullysampled[0,:,:], cmap='gray')
plt.title('R=0')
plt.axis('off')
plt.subplot(3, 2, 3)
plt.imshow(img_recon_motion[0,:,:], cmap='gray')
plt.title('t=0')
plt.axis('off')
plt.subplot(3, 2, 4)
plt.imshow(img_recon_motion[2,:,:], cmap='gray')
plt.title('t=2')
plt.axis('off')
plt.subplot(3, 2, 5)
plt.imshow(img_recon_motion[4,:,:], cmap='gray')
plt.title('t=4')
plt.axis('off')
plt.subplot(3, 2, 6)
plt.imshow(img_recon_motion[10,:,:], cmap='gray')
plt.title('t=10')
plt.axis('off')
if model_num == 3:
    plt.savefig('./TestResults-Reconstruction/Images-Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}_Epoch{}.png'.format(dataset,choose_loss,smooth,learning_rate,mode,epoch))
else:
    plt.savefig('./TestResults-Reconstruction/Images-Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Epoch{}.png'.format(model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode,epoch))
plt.close

"""
# plot the reconstructed motion-compensated frames
plt.figure(layout='compressed', figsize=(16, 16))
plt.subplots_adjust(wspace=0,hspace=0) 
plt.subplot(num_frames+2, 2, 1)
plt.imshow(images_subsampled[:,:,0,0], cmap='gray')
if mode == 1:
    plt.title('R=4')
elif mode == 2:
    plt.title('R=8')   
elif mode == 3:
    plt.title('R=10')    
plt.axis('off')
plt.subplot(num_frames+2, 2, 2)
plt.imshow(images_fullysampled[:,:,0], cmap='gray')
plt.title('R=0')
plt.axis('off')
for frame in range(num_frames):
    plt.subplot(num_frames+2, 2, frame+3)
    plt.imshow(img_recon_motion[:,:,frame], cmap='gray')
    plt.title('t={}'.format(frame))
    plt.axis('off')
plt.savefig('MotionReconstructedImages.png') #
plt.close
"""   
print('    Plot saved.\nEvaluation results:')
# get mean and std  
mean_SSIM   = np.mean(SSIM_test)
std_SSIM    = np.std(SSIM_test)
mean_MSE    = np.mean(MSE_test)
std_MSE     = np.std(MSE_test)

# write results to csv file
f = open(csv_name, 'a')
with f:
    writer = csv.writer(f)
    writer.writerow(['-', '-', '-', mean_SSIM, std_SSIM, mean_MSE, std_MSE])

print('   % SSIM: {:.4f} \\pm {:.4f}\n   MSE (e-3): {:.4f} \\pm {:.4f}'.format(mean_SSIM*100, std_SSIM*100, mean_MSE*100, std_MSE*100))
