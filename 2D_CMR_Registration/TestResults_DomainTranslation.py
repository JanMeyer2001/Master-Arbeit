from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
import csv
import time
from skimage.metrics import structural_similarity, mean_squared_error
from natsort import natsorted
import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float,
                    dest="learning_rate", default=1e-4, help="learning rate")
parser.add_argument("--lambda", type=float,
                    dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--indomain", type=str, 
                    dest="indomain", default="ACDC",
                    help="indomain for training images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--outdomain", type=str, 
                    dest="outdomain", default="CMRxRecon",
                    help="outdomain for test images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--choose_loss", type=int, dest="choose_loss", default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2), SSIM (3)")
parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
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
parser.add_argument("--gpu", type=int,
                    dest="gpu", default=1, 
                    help="choose whether to use the gpu (1) or not (0)")
parser.add_argument("--epoch", type=int,
                    dest="epoch", default=0, 
                    help="choose which epoch is used in the evaluation (for input 0 the best version will be chosen)")
opt = parser.parse_args()

learning_rate = opt.learning_rate
start_channel = opt.start_channel
smooth = opt.smth_lambda
indomain = opt.indomain
outdomain = opt.outdomain
choose_loss = opt.choose_loss
mode = opt.mode
model_num = opt.model_num
diffeo = opt.diffeo
FT_size = [opt.FT_size_x,opt.FT_size_y]
gpu = opt.gpu
epoch = opt.epoch

assert indomain != outdomain, f"In- and Out-Domain need to be different!! Else use the normal TestResults-script."

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if gpu==1 else "cpu")

# choose the model
assert model_num >= 0 or model_num <= 8, f"Expected F_Net_plus to be between 0 and 8, but got: {model_num}"
assert diffeo == 0 or diffeo == 1, f"Expected diffeo to be either 0 or 1, but got: {diffeo}"
if model_num == 0:
    model = Fourier_Net(2, 2, start_channel, diffeo).to(device) 
elif model_num == 1:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Fourier_Net_plus(2, 2, start_channel, diffeo, FT_size).to(device) 
elif model_num == 2:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade(2, 2, start_channel, diffeo, FT_size).to(device) 
elif model_num == 3:
    model = Fourier_Net_dense(2, 2, start_channel, diffeo).to(device) 
elif model_num == 4:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Fourier_Net_plus_dense(2, 2, start_channel, diffeo, FT_size).to(device) 
elif model_num == 5:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade_dense(2, 2, start_channel, diffeo, FT_size).to(device)  
elif model_num == 6:
    model = Fourier_Net_kSpace(4, 2, start_channel, diffeo).to(device) 
elif model_num == 7:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Fourier_Net_plus_kSpace(4, 2, start_channel, diffeo, FT_size).to(device) 
elif model_num == 8:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade_kSpace(4, 2, start_channel, diffeo, FT_size).to(device) 

transform = SpatialTransform().to(device)

path = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(indomain,model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode)

if epoch == 0:
    # choose best model
    print('Best model: {}'.format(natsorted(os.listdir(path))[-1]))
    modelpath = path + natsorted(os.listdir(path))[-1]
else:
    # choose model after certain epoch of training
    modelpath = [f.path for f in scandir(path) if f.is_file() and not (f.name.find('Epoch_{:04d}'.format(epoch)) == -1)][0]
    print('Best model: {}'.format(basename(modelpath)))

bs = 1

torch.backends.cudnn.benchmark = True
model.load_state_dict(torch.load(modelpath))
model.eval()
transform.eval()
MSE_test = []
SSIM_test = []
NegJ = []
times = []
if outdomain != 'CMRxRecon':
    Dice_test_full = []
    Dice_test_noBackground = []

if outdomain == 'ACDC':
    # load ACDC test data
    test_set = TestDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
    input_shape = test_set.__getitem__(0)[0].unsqueeze(0).shape
elif outdomain == 'CMRxRecon':
    # load CMRxRecon test data
    test_set = TestDatasetCMRxReconBenchmark('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', False, mode) 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
    input_shape = test_set.__getitem__(0)[0].unsqueeze(0).shape
elif outdomain == 'OASIS':
    # path for OASIS test dataset
    test_set = TestDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS') 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
    input_shape = test_set.__getitem__(0)[0].unsqueeze(0).shape
else:
    raise ValueError('Dataset should be "ACDC", "CMRxRecon" or "OASIS", but found "%s"!' % outdomain)

csv_name = './TestResults-DomainTranslation/TestMetrics-Domain_{}-{}_Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Epoch{}.csv'.format(indomain,outdomain,model_num,diffeo,choose_loss,start_channel,FT_size[0],FT_size[1],smooth,learning_rate,mode,epoch)
f = open(csv_name, 'w')
with f:
    if outdomain == 'CMRxRecon':
        fnames = ['Image Pair','SSIM','MSE','Mean SSIM','Mean MSE','Mean Time','Mean NegJ']
    elif outdomain == 'OASIS':
        fnames = ['Image Pair','Dice','SSIM','MSE','Mean Dice','Mean SSIM','Mean MSE','Mean Time','Mean NegJ']
    elif outdomain == 'ACDC':
        fnames = ['Image Pair','Dice full','Dice no background','SSIM','MSE','Mean Dice full',' Mean Dice no background','Mean SSIM','Mean MSE','Mean Time','Mean NegJ']    
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

for i, image_pairs in enumerate(test_generator): 
    with torch.no_grad():
        mov_img_fullySampled = image_pairs[0].float().to(device)
        fix_img_fullySampled = image_pairs[1].float().to(device)
        if outdomain == 'CMRxRecon':
            mov_img_subSampled = image_pairs[2].float().to(device)
            fix_img_subSampled = image_pairs[3].float().to(device)
        else:
            mov_seg = image_pairs[2].float().to(device)
            fix_seg = image_pairs[3].float().to(device)

        if model_num == 3:
            # ensure that all images and segmentations have the same size for dense F-Net
            mov_img_fullySampled = F.interpolate(mov_img_fullySampled, [224,256], mode='nearest') 
            fix_img_fullySampled = F.interpolate(fix_img_fullySampled, [224,256], mode='nearest')
            mov_seg = F.interpolate(mov_seg, [224,256], mode='nearest') 
            fix_seg = F.interpolate(fix_seg, [224,256], mode='nearest')    

        start = time.time()
        # calculate displacement on subsampled data
        if outdomain == 'CMRxRecon':
            V_xy, __ = model(mov_img_subSampled, fix_img_subSampled)
        else:
            V_xy, __ = model(mov_img_fullySampled, fix_img_fullySampled)
        
        # get inference time
        inference_time = time.time()-start
        times.append(inference_time)
        
        # but warp fully sampled data
        __, warped_mov_img_fullySampled = transform(mov_img_fullySampled, V_xy.permute(0, 2, 3, 1), mod = 'nearest')
        if outdomain != 'CMRxRecon':
            __, warped_mov_seg = transform(mov_seg, V_xy.permute(0, 2, 3, 1), mod = 'nearest')
        
        # calculate MSE, SSIM and Dice 
        if outdomain == 'OASIS':
            csv_Dice_full = dice(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy())
        elif outdomain == 'ACDC':
            dices_temp = dice_ACDC(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy())
            csv_Dice_full = np.mean(dices_temp)
            csv_Dice_noBackground = np.mean(dices_temp[1:3])
        csv_MSE = mean_squared_error(warped_mov_img_fullySampled[0,0,:,:].cpu().numpy(), fix_img_fullySampled[0,0,:,:].cpu().numpy())
        csv_SSIM = structural_similarity(warped_mov_img_fullySampled[0,0,:,:].cpu().numpy(), fix_img_fullySampled[0,0,:,:].cpu().numpy(), data_range=1)
                  
        MSE_test.append(csv_MSE)
        SSIM_test.append(csv_SSIM)
        if outdomain == 'OASIS':
            Dice_test_full.append(csv_Dice_full)
        elif outdomain == 'ACDC':
            Dice_test_full.append(csv_Dice_full)
            Dice_test_noBackground.append(csv_Dice_noBackground)
    
        hh, ww = V_xy.shape[-2:]
        V_xy = V_xy.detach().cpu().numpy()
        V_xy[:,0,:,:] = V_xy[:,0,:,:] * hh / 2
        V_xy[:,1,:,:] = V_xy[:,1,:,:] * ww / 2

        jac_det = jacobian_determinant_vxm(V_xy[0, :, :, :])    # get jacobian determinant
        negJ = np.sum(jac_det <= 0)                             # get number of non positive values
        negJ = negJ * 100 / (input_shape[2] * input_shape[3])   # get percentage over the whole image
        NegJ.append(negJ)

        # save test results to csv file
        f = open(csv_name, 'a')
        with f:
            writer = csv.writer(f)
            if outdomain == 'CMRxRecon':
                writer.writerow([i, csv_SSIM, csv_MSE, '-', '-', '-', '-']) 
            elif outdomain == 'OASIS':
                writer.writerow([i, csv_Dice_full,csv_MSE, csv_SSIM, '-', '-', '-', '-', '-']) 
            elif outdomain == 'ACDC':    
                writer.writerow([i, csv_Dice_full, csv_Dice_noBackground, csv_MSE, csv_SSIM, '-', '-', '-', '-', '-', '-'])

mean_MSE = np.mean(MSE_test)
std_MSE = np.std(MSE_test)

mean_SSIM = np.mean(SSIM_test)
std_SSIM = np.std(SSIM_test)

mean_NegJ = np.mean(NegJ)
std_NegJ = np.std(NegJ)

mean_time = np.mean(times)

if outdomain == 'OASIS':
    mean_Dice_full = np.mean(Dice_test_full)
    std_Dice_full = np.std(Dice_test_full)
elif outdomain == 'ACDC':
    mean_Dice_full = np.mean(Dice_test_full)
    std_Dice_full = np.std(Dice_test_full)
    mean_Dice_noBackground = np.mean(Dice_test_noBackground)
    std_Dice_noBackground = np.std(Dice_test_noBackground)

f = open(csv_name, 'a')
with f:
    writer = csv.writer(f)
    if outdomain == 'CMRxRecon':
        writer.writerow(['-', '-', '-', mean_SSIM, mean_MSE, mean_time, mean_NegJ])
    elif outdomain == 'OASIS':
        writer.writerow(['-', '-', '-', '-', mean_Dice_full, mean_SSIM, mean_MSE, mean_time, mean_NegJ])
    elif outdomain == 'ACDC':
        writer.writerow(['-', '-', '-', '-', '-', mean_Dice_full, mean_Dice_noBackground, mean_SSIM, mean_MSE, mean_time, mean_NegJ])

if outdomain == 'CMRxRecon':
    print('Mean inference time: {:.4f} seconds\n     % SSIM: {:.2f} \\pm {:.2f}\n     MSE (e-3): {:.2f} \\pm {:.2f}\n     % DetJ<0: {:.2f} \\pm {:.2f}'.format(mean_time, mean_SSIM*100, std_SSIM*100, mean_MSE*100, std_MSE*100, mean_NegJ, std_NegJ))
elif outdomain == 'OASIS':
    print('Mean inference time: {:.4f} seconds\n     % DICE: {:.2f} \\pm {:.2f}\n     % SSIM: {:.2f} \\pm {:.2f}\n     MSE (e-3): {:.2f} \\pm {:.2f}\n     % DetJ<0: {:.2f} \\pm {:.2f}'.format(mean_time, mean_Dice_full*100, std_Dice_full*100, mean_SSIM*100, std_SSIM*100, mean_MSE*100, std_MSE*100, mean_NegJ, std_NegJ))
elif outdomain == 'ACDC':
    print('Mean inference time: {:.4f} seconds\n     % DICE full: {:.2f} \\pm {:.2f}\n     % DICE no background: {:.2f} \\pm {:.2f}\n     % SSIM: {:.2f} \\pm {:.2f}\n     MSE (e-3): {:.2f} \\pm {:.2f}\n     % DetJ<0: {:.2f} \\pm {:.2f}'.format(mean_time, mean_Dice_full*100, std_Dice_full*100, mean_Dice_noBackground*100, std_Dice_noBackground*100, mean_SSIM*100, std_SSIM*100, mean_MSE*100, std_MSE*100, mean_NegJ, std_NegJ))
