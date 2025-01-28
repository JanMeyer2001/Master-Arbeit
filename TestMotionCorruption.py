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

transform = SpatialTransform()
H = 246
W = 512

# import image/k-space data and coil sensitivity maps for all slices and frames
data_set = DatasetMotionReconstruction_LungMovement('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', False, 3, transform, 4) 
data_generator = Data.DataLoader(dataset=data_set, batch_size=1, shuffle=False, num_workers=0)

for i, data in enumerate(data_generator):
    # get data
    images_fullysampled = data[0].squeeze()     # tensor with size (F/2,H,W)
    images_subsampled   = data[1].squeeze()     # tensor with size (L,F/2,H,W) --> L is the number of simulated lung movement
    masks               = data[2]               # tensor with size (1,1,C,L*F/2,H,W)
    k_spaces            = data[3]               # tensor with size (1,1,C,L*F/2,H,W)
    coil_maps           = data[4]               # tensor with size (1,1,C,L*F/2,H,W)
    #flows               = data[5]               # tensor with size (1,1,L*F/2,H,W,2) 

    L = 5
    num_frames  = images_fullysampled.shape[0]
    num_coils  = k_spaces.shape[2]              # number of coils C
    max_iter    = 10
    
    model = Fourier_Net(2, 2, 16, 0)
    path  = './2D_CMR_Registration/ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format('CMRxRecon',0,0,1,16,24,24,0.01,0.0001,1)
    
    modelpath = [f.path for f in scandir(path) if f.is_file() and not (f.name.find('Epoch_{:04d}'.format(14)) == -1)][0]
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    transform.eval()
    # init torch tensor for flow fields
    flows = torch.zeros(L,num_frames,H,W,2)

    images_subsampled = images_subsampled[:,:,10:202,106:440]
    
    plt.figure(layout='compressed', figsize=(6,10))
    plt.subplots_adjust(wspace=0,hspace=0) 
    plt.subplot(1, 1, 1)
    plt.imshow(images_subsampled[1,0,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.tight_layout(h_pad=0.1, w_pad=0.3, pad=0)
    plt.savefig('./Images/LungMotion1.png', dpi=600, pad_inches=0.05, bbox_inches="tight")
    plt.close
    plt.subplot(1, 1, 1)
    plt.imshow(images_subsampled[2,0,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.tight_layout(h_pad=0.1, w_pad=0.3, pad=0)
    plt.savefig('./Images/LungMotion2.png', dpi=600, pad_inches=0.05, bbox_inches="tight")
    plt.close
    plt.subplot(1, 1, 1)
    plt.imshow(images_subsampled[3,0,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.tight_layout(h_pad=0.1, w_pad=0.3, pad=0)
    plt.savefig('./Images/LungMotion3.png', dpi=600, pad_inches=0.05, bbox_inches="tight")
    plt.close
    plt.subplot(1, 1, 1)
    plt.imshow(images_subsampled[4,0,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.tight_layout(h_pad=0.1, w_pad=0.3, pad=0)
    plt.savefig('./Images/LungMotion4.png', dpi=600, pad_inches=0.05, bbox_inches="tight")
    plt.close
    """
    for frame_num in range(num_frames):
        for i in range(L):
            # get displacements relative to the first image (first entry is deliberately left empty)
            flow, features_disp = model(images_subsampled[i,frame_num,:,:].unsqueeze(0).unsqueeze(0).float(), images_subsampled[0,frame_num,:,:].unsqueeze(0).unsqueeze(0).float())
            flows[i,frame_num,:,:,:] = flow.squeeze().permute(1,2,0)
    
    # for testing image registration
    warped_mov = torch.zeros(L,H,W)
    for i in range(L):
        grid, warped_mov[i,:,:] = transform(images_subsampled[i,0,:,:].unsqueeze(0).unsqueeze(0).float(), flows[i,0,:,:,:].unsqueeze(0).float())
    
    plt.subplot(5, 2, 2)
    plt.imshow(warped_mov[0,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.title('Warped')
    plt.subplot(5, 2, 4)
    plt.imshow(warped_mov[1,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.subplot(5, 2, 6)
    plt.imshow(warped_mov[2,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.subplot(5, 2, 8)
    plt.imshow(warped_mov[3,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.subplot(5, 2, 10)
    plt.imshow(warped_mov[4,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.savefig('TestMotionWarping.png')
    plt.close
    """

    # reshape tensor into correct size for the reconstruction pipeline
    images_subsampled = torch.reshape(images_subsampled, ((L)*num_frames,H,W)).unsqueeze(0)
    flows = torch.reshape(flows, ((L)*num_frames,H,W,2)).unsqueeze(0).unsqueeze(0)
    """
    recon = ReconDCPMMotion(max_iter=max_iter, coil_axis=2)
    img_recon_motion = torch.abs(recon(images_subsampled, k_spaces, masks, coil_maps, flows, transform, num_frames)).squeeze()
    """
    
    # change sizes to fit new reconstruction pipeline
    k_spaces    = k_spaces.squeeze().squeeze().permute(1,0,2,3)     # tensor with size (L*F/2,H,W,2)
    flows       = flows.squeeze().squeeze()                         # tensor with size (C,L*F/2,H,W)
    masks       = masks[0,0,0,0,:,:]                                # tensor with size (H,W)        
    coil_maps   = coil_maps[0,0,:,0,:,:]                            # tensor with size (C,H,W)
    img_recon_motion = reconstruct_denoised_image(images_subsampled.squeeze(), k_spaces, flows, coil_maps, masks, num_frames, L, num_coils, H, W, 1e-3, 1)
    
    # normalize (just to be sure)
    img_recon_motion = normalize(img_recon_motion)

    # reshape reconstructed images back 
    images_subsampled = torch.reshape(images_subsampled, (L,num_frames,H,W))
    
    HaarPSI_test = []
    PSNR_test    = []
    SSIM_test    = []
    MSE_test     = []

    # evaluate reconstructed frames
    for frame in range(num_frames):
        # get MSE and SSIM between first fully sampled frame and all motion-corrected reconstructed frames      
        csv_HaarPSI = haarpsi(img_recon_motion[frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
        csv_PSNR    = psnr(img_recon_motion[frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
        csv_SSIM    = ssim(img_recon_motion[frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
        csv_MSE     = mean_squared_error(images_fullysampled[frame,:,:].cpu().detach().numpy(), img_recon_motion[frame,:,:].cpu().detach().numpy())
        HaarPSI_test.append(csv_HaarPSI)
        PSNR_test.append(csv_PSNR)
        SSIM_test.append(csv_SSIM)
        MSE_test.append(csv_MSE)
    
    # get mean and std  
    mean_HaarPSI = np.mean(HaarPSI_test)*100
    std_HaarPSI  = np.std(HaarPSI_test)*100
    mean_PSNR    = np.mean(PSNR_test)
    std_PSNR     = np.std(PSNR_test)
    mean_SSIM    = np.mean(SSIM_test)*100
    std_SSIM     = np.std(SSIM_test)*100
    mean_MSE     = np.mean(MSE_test)*100
    std_MSE      = np.std(MSE_test)*100

    print('   % HaarPSI: {:.3f} \\pm {:.3f}\n   PSNR (dB): {:.3f} \\pm {:.3f}\n   % SSIM: {:.3f} \\pm {:.3f}\n   MSE (e-3): {:.3f} \\pm {:.3f}'.format(mean_HaarPSI, std_HaarPSI, mean_PSNR, std_PSNR, mean_SSIM, std_SSIM, mean_MSE, std_MSE))
    
    img_recon_motion  = torch.reshape(img_recon_motion, (L,num_frames,H,W))
    
    HaarPSI_test = []
    PSNR_test    = []
    SSIM_test    = []
    MSE_test     = []

    # evaluate reconstructed frames
    for frame in range(num_frames):
        for i in range(L):    
            # get MSE and SSIM between first fully sampled frame and all motion-corrected reconstructed frames      
            csv_HaarPSI = haarpsi(img_recon_motion[i,frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
            csv_PSNR    = psnr(img_recon_motion[i,frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
            csv_SSIM    = ssim(img_recon_motion[i,frame,:,:].unsqueeze(0).unsqueeze(0), images_fullysampled[frame,:,:].unsqueeze(0).unsqueeze(0), data_range=1).item()
            csv_MSE     = mean_squared_error(images_fullysampled[frame,:,:].cpu().detach().numpy(), img_recon_motion[i,frame,:,:].cpu().detach().numpy())
            HaarPSI_test.append(csv_HaarPSI)
            PSNR_test.append(csv_PSNR)
            SSIM_test.append(csv_SSIM)
            MSE_test.append(csv_MSE)
    
    # get mean and std  
    mean_HaarPSI = np.mean(HaarPSI_test)*100
    std_HaarPSI  = np.std(HaarPSI_test)*100
    mean_PSNR    = np.mean(PSNR_test)
    std_PSNR     = np.std(PSNR_test)
    mean_SSIM    = np.mean(SSIM_test)*100
    std_SSIM     = np.std(SSIM_test)*100
    mean_MSE     = np.mean(MSE_test)*100
    std_MSE      = np.std(MSE_test)*100

    print('   % HaarPSI: {:.3f} \\pm {:.3f}\n   PSNR (dB): {:.3f} \\pm {:.3f}\n   % SSIM: {:.3f} \\pm {:.3f}\n   MSE (e-3): {:.3f} \\pm {:.3f}'.format(mean_HaarPSI, std_HaarPSI, mean_PSNR, std_PSNR, mean_SSIM, std_SSIM, mean_MSE, std_MSE))
    
    plt.figure(layout='compressed', figsize=(16, 16))
    plt.subplots_adjust(wspace=0,hspace=0) 
    plt.subplot(3, 2, 1)
    plt.imshow(images_fullysampled[0,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.title('R=0')
    plt.subplot(3, 2, 2)
    plt.imshow(images_subsampled[0,0,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.title('R=4')
    plt.subplot(3, 2, 3)
    plt.imshow(images_subsampled[0,1,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.title('t=1')
    plt.subplot(3, 2, 4)
    plt.imshow(images_subsampled[0,2,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.title('t=2')
    plt.subplot(3, 2, 5)
    plt.imshow(img_recon_motion[1,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.title('Recon t=1')
    plt.subplot(3, 2, 6)
    plt.imshow(img_recon_motion[2,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.title('Recon t=2')
    plt.savefig('TestMotionCorruptedReconstructionNetwork.png')
    plt.close

    """
    plt.figure(layout='compressed', figsize=(16, 16))
    plt.subplots_adjust(wspace=0,hspace=0) 
    plt.subplot(3, 2, 1)
    plt.imshow(images_fullysampled[1,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.subplot(3, 2, 2)
    plt.imshow(images_subsampled[0,1,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.subplot(3, 2, 3)
    plt.imshow(images_subsampled[1,1,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.subplot(3, 2, 4)
    plt.imshow(images_subsampled[2,1,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.subplot(3, 2, 5)
    plt.imshow(images_subsampled[3,1,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.subplot(3, 2, 6)
    plt.imshow(images_subsampled[4,1,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.savefig('TestMotionCorruption.png')
    plt.close

    plt.figure(layout='compressed', figsize=(16, 16))
    plt.subplots_adjust(wspace=0,hspace=0) 
    plt.subplot(2, 2, 1)
    plt.imshow(images_subsampled[0,0,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(masks[0,0,0,0,:,:].cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(torch.abs(k_spaces[0,0,0,0,:,:]).cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.imshow(torch.abs(coil_maps[0,0,0,0,:,:]).cpu().detach().numpy(), cmap='gray')  
    plt.axis('off')
    plt.savefig(join('./2D_CMR_Registration/TestResults-Reconstruction/TestMotionCorruption.png'))
    plt.close
    """