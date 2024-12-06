from argparse import ArgumentParser
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
from skimage.metrics import mean_squared_error
from piq import psnr, ssim, haarpsi
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
parser.add_argument("--mode", type=int, dest="mode", default='1',
                    help="choose dataset mode: 4x accelerated (1), 8x accelerated (2), 10x accelerated (3) or (0) for all acceleration factors.")
parser.add_argument("--model", type=int,
                    dest="model_num", default=0, 
                    help="choose whether to use Fourier-Net (0), Fourier-Net+ (1), cascaded Fourier-Net+ (2), dense Fourier-Net (3), dense Fourier-Net+ (4), dense cascaded Fourier-Net+ (5), k-space Fourier-Net (6), k-space Fourier-Net+ (7) or cascaded k-space Fourier-Net+ (8) as the model")
parser.add_argument("--FT_size_x", type=int,
                    dest="FT_size_x", default=24,
                    help="choose size x of FT crop: Should be smaller than 40.")
parser.add_argument("--FT_size_y", type=int,
                    dest="FT_size_y", default=24,
                    help="choose size y of FT crop: Should be smaller than 84.")
parser.add_argument("--epoch", type=int,
                    dest="epoch", default=14, 
                    help="choose which epoch is used in the evaluation (for input 0 the best version will be chosen)")
parser.add_argument("--L", type=int,
                    dest="L", default=4, 
                    help="number of simulated lung movement frames")
opt = parser.parse_args()

learning_rate = opt.learning_rate
start_channel = opt.start_channel
smooth = opt.smth_lambda
mode = opt.mode
model_num = opt.model_num
FT_size = [opt.FT_size_x,opt.FT_size_y]
epoch = opt.epoch
L = opt.L

dataset = 'CMRxRecon'
device  = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")

diffeo = 0
H = 246
W = 512
input_shape = [H,W] 
  

assert mode >= 0 or mode <= 3, f"Expected mode to be between 0 and 3, but got: {mode}"
if mode == 0:
    modes = [1,2,3]     # Acc factors 4, 8, 10
else:
    modes = [mode]

assert model_num >= 0 or model_num <= 3, f"Expected model_num to be between 0 and 3, but got: {model_num}"
assert diffeo == 0 or diffeo == 1, f"Expected diffeo to be either 0 or 1, but got: {diffeo}"
transform = SpatialTransform().to(device)

for mode in modes:
    # print mode
    if mode == 1:
        print('R=4')
    elif mode == 2:
        print('R=8')
    elif mode == 3:
        print('R=10')

    # import image/k-space data and coil sensitivity maps for all slices and frames
    data_set = DatasetMotionReconstruction_LungMovement('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', False, mode, transform, L) 
    data_generator = Data.DataLoader(dataset=data_set, batch_size=1, shuffle=False, num_workers=4)

    # select and import models for motion correction
    if model_num == 0:
        model = Fourier_Net(2, 2, start_channel, diffeo).to(device) 
        path  = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_num,diffeo,1,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
    elif model_num == 1:
        assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
        model = Fourier_Net_plus(2, 2, start_channel, diffeo, FT_size).to(device) 
        path  = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_num,diffeo,1,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
    elif model_num == 2:
        assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
        model = Cascade(2, 2, start_channel, diffeo, FT_size).to(device) 
        path  = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_num,diffeo,1,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
    elif model_num == 3:
        model = VxmDense(inshape=input_shape, nb_unet_features=32, bidir=False, nb_unet_levels=4).to(device)  #, int_steps=7, int_downsize=2
        path  = './ModelParameters-{}/Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}/'.format(dataset,0,smooth,learning_rate,mode)

    if epoch == 0:
        # choose best model
        modelpath = path + natsorted(os.listdir(path))[-1]
    else:
        # choose model after certain epoch of training
        modelpath = [f.path for f in scandir(path) if f.is_file() and not (f.name.find('Epoch_{:04d}'.format(epoch)) == -1)][0]
        
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    transform.eval()

    # save test results in a csv file
    if model_num == 3:
        csv_name = './TestResults-Reconstruction/LungMovement/TestMetrics-Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}_Epoch{}.csv'.format(0,smooth,learning_rate,mode,epoch)
    else:
        csv_name = './TestResults-Reconstruction/LungMovement/TestMetrics-Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Epoch{}.csv'.format(model_num,diffeo,1,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode,epoch)
    f = open(csv_name, 'w')
    with f:
        fnames = ['HaarPSI','PSNR','SSIM','MSE','Mean HaarPSI','Std HaarPSI','Mean PSNR','Std PSNR','Mean SSIM','Std SSIM','Mean MSE','Std MSE']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    HaarPSI_test = []
    PSNR_test    = []
    SSIM_test    = []
    MSE_test     = []

    print('Begin Reconstruction on {}.'.format(time.ctime()))
    for data_num, data in enumerate(data_generator):
        # get data
        images_fullysampled = data[0].squeeze()     # tensor with size (F/2,H,W)
        images_subsampled   = data[1].squeeze()     # tensor with size (L+1,F/2,H,W) --> L is the number of simulated lung movement
        masks               = data[2]               # tensor with size (1,1,C,L+1*F/2,H,W)
        k_spaces            = data[3]               # tensor with size (1,1,C,L+1*F/2,H,W)
        coil_maps           = data[4]               # tensor with size (1,1,C,L+1*F/2,H,W)

        num_frames = images_fullysampled.shape[0]   # number of frames F/2
        num_coils  = k_spaces.shape[2]              # number of coils C
        max_iter   = 10                             # number of iterations
        tol        = 1e-12                          # error tolerance
        
        # init torch tensor for flow fields
        flows = torch.zeros(5,num_frames,H,W,2)
    
        for frame_num in range(num_frames):
            for i in range(L):
                if model_num == 3:
                    images_subsampled[i+1,frame_num,:,:], flow = model(images_subsampled[i+1,frame_num,:,:].unsqueeze(0).unsqueeze(0).float(), images_subsampled[0,frame_num,:,:].unsqueeze(0).unsqueeze(0).float())
                    flows[i+1,frame_num,:,:,:] = flow.squeeze().permute(1,2,0)
                else:    
                    flow, features_disp = model(images_subsampled[i+1,frame_num,:,:].unsqueeze(0).unsqueeze(0).float(), images_subsampled[0,frame_num,:,:].unsqueeze(0).unsqueeze(0).float())
                    grid, images_subsampled[i+1,frame_num,:,:] = transform(images_subsampled[i+1,frame_num,:,:].unsqueeze(0).unsqueeze(0).float(), flows[i+1,frame_num,:,:,:].unsqueeze(0).float())
                    flows[i+1,frame_num,:,:,:] = flow.squeeze().permute(1,2,0)

        # reshape tensor into correct size for the reconstruction pipeline
        images_subsampled = torch.reshape(images_subsampled, ((L+1)*num_frames,H,W)).unsqueeze(0)
        flows = torch.reshape(flows, ((L+1)*num_frames,H,W,2)).unsqueeze(0).unsqueeze(0)

        # init pipeline and reconstruct images
        recon = ReconDCPMMotion(max_iter=max_iter, coil_axis=2)
        img_recon_motion = torch.abs(recon(images_subsampled, k_spaces, masks, coil_maps, flows, transform, num_frames)).squeeze()
        # normalize (just to be sure)
        img_recon_motion = normalize(img_recon_motion)

        # reshape reconstructed images back 
        img_recon_motion  = torch.reshape(img_recon_motion, (L+1,num_frames,H,W))
        
        # evaluate reconstructed frames
        for frame in range(num_frames):
            # get MSE and SSIM between first fully sampled frame and all motion-corrected reconstructed frames      
            csv_HaarPSI = haarpsi(img_recon_motion[0,frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
            csv_PSNR    = psnr(img_recon_motion[0,frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
            csv_SSIM    = ssim(img_recon_motion[0,frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
            csv_MSE     = mean_squared_error(images_fullysampled[frame,:,:].cpu().detach().numpy(), img_recon_motion[0,frame,:,:].cpu().detach().numpy())
            HaarPSI_test.append(csv_HaarPSI)
            PSNR_test.append(csv_PSNR)
            SSIM_test.append(csv_SSIM)
            MSE_test.append(csv_MSE)
            # save test results to csv file
            f = open(csv_name, 'a')
            with f:
                writer = csv.writer(f)
                writer.writerow([csv_HaarPSI,csv_PSNR, csv_SSIM, csv_MSE, '-', '-', '-', '-', '-', '-', '-', '-']) 
        """
        if data_num == 0:
            # plot the reconstructed motion-compensated frames
            plt.figure(layout='compressed', figsize=(16, 16))
            plt.subplots_adjust(wspace=0,hspace=0) 
            plt.subplot(3, 2, 1)
            plt.imshow(images_fullysampled[0,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('Original')
            plt.axis('off')
            plt.subplot(3, 2, 2)
            plt.imshow(images_fullysampled[0,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('Original')
            plt.axis('off')
            plt.subplot(3, 2, 3)
            plt.imshow(img_recon_motion[1,0,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('L=1 Reconstruction')
            plt.axis('off')
            plt.subplot(3, 2, 4)
            plt.imshow(img_recon_motion[2,0,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('L=2 Reconstruction')
            plt.axis('off')
            plt.subplot(3, 2, 5)
            plt.imshow(np.abs(img_recon_motion[1,0,:,:].cpu().detach().numpy()-images_fullysampled[0,:,:].cpu().detach().numpy()), cmap='gray')
            plt.title('L=1 Difference')
            plt.axis('off')
            plt.subplot(3, 2, 6)
            plt.imshow(np.abs(img_recon_motion[2,0,:,:].cpu().detach().numpy()-images_fullysampled[0,:,:].cpu().detach().numpy()), cmap='gray')
            plt.title('L=2 Difference')
            plt.axis('off')
            if model_num == 3:
                plt.savefig(join('./TestResults-Reconstruction/Images-Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}_Epoch{}.png'.format(dataset,0,smooth,learning_rate,mode,epoch)))
            else:
                plt.savefig(join('./TestResults-Reconstruction/Images-Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Epoch{}.png'.format(model_num,diffeo,1,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode,epoch)))
            plt.close
        """
    print('Finished reconstruction on {}.\nEvaluation results:'.format(time.ctime()))

    # get mean and std  
    mean_HaarPSI = np.mean(HaarPSI_test)*100
    std_HaarPSI  = np.std(HaarPSI_test)*100
    mean_PSNR    = np.mean(PSNR_test)
    std_PSNR     = np.std(PSNR_test)
    mean_SSIM    = np.mean(SSIM_test)*100
    std_SSIM     = np.std(SSIM_test)*100
    mean_MSE     = np.mean(MSE_test)*100
    std_MSE      = np.std(MSE_test)*100

    # write results to csv file
    f = open(csv_name, 'a')
    with f:
        writer = csv.writer(f)
        writer.writerow(['-', '-', '-', '-', mean_HaarPSI, std_HaarPSI, mean_PSNR, std_PSNR, mean_SSIM, std_SSIM, mean_MSE, std_MSE])

    print('   % HaarPSI: {:.3f} \\pm {:.3f}\n   PSNR (dB): {:.3f} \\pm {:.3f}\n   % SSIM: {:.3f} \\pm {:.3f}\n   MSE (e-3): {:.3f} \\pm {:.3f}'.format(mean_HaarPSI, std_HaarPSI, mean_PSNR, std_PSNR, mean_SSIM, std_SSIM, mean_MSE, std_MSE))
