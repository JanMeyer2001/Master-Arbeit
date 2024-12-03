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
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--L", type=int,
                    dest="L", default=4, 
                    help="number of simulated lung movement frames")
opt = parser.parse_args()
L = opt.L

dataset = 'CMRxRecon'
device  = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")

H = 246
W = 512

transform = SpatialTransform().to(device)
transform.eval()

modes = [1,2,3]     # Acc factors 4, 8, 10

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

    # save test results in a csv file
    csv_name = './TestResults-Reconstruction/LungMovement/TestBaseline_Mode_{}.csv'.format(mode)
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
        
        # evaluate reconstructed frames
        for frame in range(num_frames):
            for i in range(L):    
                # get MSE and SSIM between first fully sampled frame and all motion-corrected reconstructed frames      
                csv_HaarPSI = haarpsi(images_subsampled[i+1,frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
                csv_PSNR    = psnr(images_subsampled[i+1,frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
                csv_SSIM    = ssim(images_subsampled[i+1,frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
                csv_MSE     = mean_squared_error(images_fullysampled[frame,:,:].cpu().detach().numpy(), images_subsampled[i+1,frame,:,:].cpu().detach().numpy())
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
            plt.subplot(4, 2, 1)
            plt.imshow(images_subsampled[1,0,:,:].cpu().detach().numpy(), cmap='gray') 
            plt.title('L=1 Corrupted')  
            plt.axis('off')
            plt.subplot(4, 2, 2)
            plt.imshow(images_subsampled[2,0,:,:].cpu().detach().numpy(), cmap='gray') 
            plt.title('L=2 Corrupted')  
            plt.axis('off')
            plt.subplot(4, 2, 3)
            plt.imshow(images_fullysampled[0,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('Original')
            plt.axis('off')
            plt.subplot(4, 2, 4)
            plt.imshow(images_fullysampled[0,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('Original')
            plt.axis('off')
            plt.subplot(4, 2, 5)
            plt.imshow(img_recon_motion[1,0,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('L=1 Reconstruction')
            plt.axis('off')
            plt.subplot(4, 2, 6)
            plt.imshow(img_recon_motion[2,0,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('L=2 Reconstruction')
            plt.axis('off')
            plt.subplot(4, 2, 7)
            plt.imshow(np.abs(img_recon_motion[1,0,:,:].cpu().detach().numpy()-images_fullysampled[0,:,:].cpu().detach().numpy()), cmap='gray')
            plt.title('L=1 Difference')
            plt.axis('off')
            plt.subplot(4, 2, 8)
            plt.imshow(np.abs(img_recon_motion[2,0,:,:].cpu().detach().numpy()-images_fullysampled[0,:,:].cpu().detach().numpy()), cmap='gray')
            plt.title('L=2 Difference')
            plt.axis('off')
            plt.savefig(join('./TestResults-Reconstruction/Images-Baseline_Mode_{}.png'.format(mode)))
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
