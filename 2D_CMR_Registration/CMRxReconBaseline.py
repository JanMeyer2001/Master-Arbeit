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
    data_set = DatasetCMRxReconstruction('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', False, mode) 
    data_generator = Data.DataLoader(dataset=data_set, batch_size=1, shuffle=False, num_workers=4)

    # save test results in a csv file
    csv_name = './TestResults-Reconstruction/TestBaseline_Mode_{}.csv'.format(mode)
    f = open(csv_name, 'w')
    with f:
        fnames = ['HaarPSI','PSNR','SSIM','MSE','Mean HaarPSI','Std HaarPSI','Mean PSNR','Std PSNR','Mean SSIM','Std SSIM','Mean MSE','Std MSE']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    HaarPSI_test = []
    PSNR_test    = []
    SSIM_test    = []
    MSE_test     = []

    for data, i in enumerate(data_generator):
        # get data
        images_fullysampled = data[0].squeeze()     # tensor with size (F,H,W)
        images_subsampled   = data[1]               # tensor with size (1,F,H,W)
        masks               = data[2]               # tensor with size (1,1,C,F,H,W)
        k_spaces            = data[3]               # tensor with size (1,1,C,F,H,W)
        coil_maps           = data[4]               # tensor with size (1,1,C,F,H,W)

        num_frames = k_spaces.shape[3]    # number of frames F
        num_coils  = k_spaces.shape[2]    # number of coils C
        max_iter   = 10                   # number of iterations
        tol        = 1e-12                # error tolerance
        
        # set flow fields to zero for no motion-correction
        flows = torch.zeros(1,1,num_frames,H,W,2)
        
        # init pipeline and reconstruct images
        recon = ReconDCPMMotion(max_iter=max_iter, coil_axis=2)
        img_recon_motion = torch.abs(recon(images_subsampled, k_spaces, masks, coil_maps, flows, transform, num_frames)).squeeze()
        # normalize (just to be sure)
        img_recon_motion = normalize(img_recon_motion)
        
        # evaluate reconstructed frames
        for frame in range(num_frames):
            # get MSE and SSIM between first fully sampled frame and all motion-corrected reconstructed frames      
            csv_HaarPSI = haarpsi(img_recon_motion[frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[0,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
            csv_PSNR    = psnr(img_recon_motion[frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[0,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
            csv_SSIM    = ssim(img_recon_motion[frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[0,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
            csv_MSE     = mean_squared_error(images_fullysampled[0,:,:].cpu().detach().numpy(), img_recon_motion[frame,:,:].cpu().detach().numpy())
            HaarPSI_test.append(csv_HaarPSI)
            PSNR_test.append(csv_PSNR)
            SSIM_test.append(csv_SSIM)
            MSE_test.append(csv_MSE)
            # save test results to csv file
            f = open(csv_name, 'a')
            with f:
                writer = csv.writer(f)
                writer.writerow([csv_HaarPSI,csv_PSNR, csv_SSIM, csv_MSE, '-', '-', '-', '-', '-', '-', '-', '-']) 

        if i == 0:
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
            plt.imshow(images_fullysampled[0,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('R=0')
            plt.axis('off')
            plt.subplot(3, 2, 3)
            plt.imshow(img_recon_motion[0,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('t=0')
            plt.axis('off')
            plt.subplot(3, 2, 4)
            plt.imshow(img_recon_motion[2,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('t=2')
            plt.axis('off')
            plt.subplot(3, 2, 5)
            plt.imshow(img_recon_motion[4,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('t=4')
            plt.axis('off')
            plt.subplot(3, 2, 6)
            plt.imshow(img_recon_motion[10,:,:].cpu().detach().numpy(), cmap='gray')
            plt.title('t=10')
            plt.axis('off')
            plt.savefig('./TestResults-Reconstruction/Images-Baseline_Mode_{}.png'.format(mode))
            plt.close

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

    print('   % HaarPSI: {:.4f} \\pm {:.4f}\n   PSNR (dB): {:.4f} \\pm {:.4f}\n   % SSIM: {:.4f} \\pm {:.4f}\n   MSE (e-3): {:.4f} \\pm {:.4f}'.format(mean_HaarPSI, std_HaarPSI, mean_PSNR, std_PSNR, mean_SSIM, std_SSIM, mean_MSE, std_MSE))
