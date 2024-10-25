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
                    dest="model_num", default=1, 
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
                    dest="epoch", default=6, 
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
"""
# select and import models for motion correction
assert model_num >= 0 or model_num <= 3, f"Expected model_num to be between 0 and 3, but got: {model_num}"
assert diffeo == 0 or diffeo == 1, f"Expected diffeo to be either 0 or 1, but got: {diffeo}"
if model_num == 0:
    model = Fourier_Net(2, 2, start_channel, diffeo).to(device) 
    path = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)
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
    print('Best model: {}'.format(natsorted(os.listdir(path))[-1]))
    modelpath = path + natsorted(os.listdir(path))[-1]
else:
    # choose model after certain epoch of training
    modelpath = [f.path for f in scandir(path) if f.is_file() and not (f.name.find('Epoch_{:04d}'.format(epoch)) == -1)][0]
    print('Best model: {}'.format(basename(modelpath)))

model.load_state_dict(torch.load(modelpath))
model.eval()
transform.eval()
"""

for data in data_generator:
    img_fullySampled = data[0]
    img_subSampled   = data[1]
    mask             = data[2]
    k_space          = data[3]
    coil_map         = data[4]
    
    ksp_sli = k_space.squeeze().numpy()
    num_coils, rows, cols = ksp_sli.shape 
    
    # Create and apply US pattern
    us_pat, num_low_freqs = generate_US_pattern(ksp_sli.shape, R=4) 
    #us_ksp = us_pat * ksp_sli # not needed because k-space is already subsampled
    us_ksp = ksp_sli
    coil_map_new = np.fft.fftshift(mr.app.EspiritCalib(us_ksp, calib_width=num_low_freqs, thresh=0.02, kernel_width=6, crop=0.01, max_iter=100, show_pbar=False).run(),axes=(1, 2))

    # perform iterative SENSE reconstruction 
    img_recon = SENSE_iter(k_space.squeeze().numpy(),50,mask.squeeze().squeeze().numpy(),coil_map_new,img_fullySampled.squeeze().squeeze().numpy())
    #img_recon = SENSE_iter(k_space.squeeze().numpy(),10,mask.squeeze().squeeze().numpy(),coil_map.squeeze().numpy(),img_fullySampled.squeeze().squeeze().numpy())

    # plot the reconstructed image
    plt.subplot(1, 3, 1)
    plt.imshow(img_subSampled.squeeze().squeeze(), cmap='gray')
    plt.title('SubSampled')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(img_fullySampled.squeeze().squeeze(), cmap='gray')
    plt.title('Fully Sampled')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(img_recon), cmap='gray')
    plt.title('Reconstruction')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('reconstructedImage.png') 
    plt.close

#### Old Dataloader...
"""
coil_num = 10 # number of coil channels
for patients in data_generator:    
    k_spaces  = torch.zeros(len(patients),coil_num,len(patients[0][0]),H,W) 
    coil_maps = torch.zeros(len(patients),coil_num,len(patients[0][0]),H,W) 
    for slice_num, slices in enumerate(patients):
        # unpack image paths and get Displacements
        image_paths = slices[0]
        images = torch.zeros(len(image_paths),H,W)
        for image_num, image_path in enumerate(image_paths):
            image = imread(image_path[0], as_gray=True)/255
            images[image_num,:,:] = torch.from_numpy(image).unsqueeze(0)
        
        # correct for motion
        if model_num == 3:
            warped_img, Df_xy = model(mov_img, fix_img)
        else:
            V_xy, __ = model(mov_img, fix_img)
            __, warped_img = transform(mov_img, V_xy.permute(0, 2, 3, 1), mod='nearest') 
        
        # load k-space data
        k_space_paths = slices[1]
        for num_k_space, k_space_path in enumerate(k_space_paths): 
            k_space = torch.load(k_space_path[0])
            #k_spaces[slice_num,:,num_k_space,:,:] = torch.view_as_complex(k_space)
            image = fastmri.ifft2c(k_space)
            image = fastmri.complex_abs(image) 
            k_spaces[slice_num,:,num_k_space,:,:] = image #torch.view_as_complex(image)

        # load coil maps
        coil_map_paths = slices[2]
        for num_coil_maps, coil_map_path in enumerate(coil_map_paths): 
            coil_map = torch.load(coil_map_path[0])
            coil_maps[slice_num,:,num_coil_maps,:,:] = torch.from_numpy(coil_map).permute(2,0,1)

    # TODO: estimate noise matrix --> for now identity matrix with coil channels as dimensions
    noise_cov = torch.eye(coil_num,coil_num)

    # perform SENSE reconstruction --> gives back images with [Slices,Frames,H,W]
    reconstructed_images = sense(data=k_spaces, csm=coil_maps, noise_cov=noise_cov, acceleration_rate=mode).squeeze()
    
    # Display reconstructed images
    for slices in range(reconstructed_images.shape[0]):
        for frames in range(reconstructed_images.shape[1]):
            plt.subplot(reconstructed_images.shape[0], reconstructed_images.shape[1], slices * reconstructed_images.shape[0] + frames + 1)
            plt.imshow(reconstructed_images[slices,frames,:,:], cmap='gray') #np.abs()
            plt.axis('off')
    #plt.title('Frames')
    #plt.supylabel('Slices',fontsize=40,x=0.22)
    plt.show()
    plt.savefig('ReconstructedImages.png') # for saving the image
    plt.close
    """
