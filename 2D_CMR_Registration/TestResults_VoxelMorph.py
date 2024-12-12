import os
from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
import csv
import time
from skimage.metrics import structural_similarity, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from natsort import natsorted

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float,
                    dest="learning_rate", default=1e-4, help="learning rate")
parser.add_argument("--lambda", type=float,
                    dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--dataset", type=str, 
                    dest="dataset", default="ACDC",
                    help="dataset for training images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--choose_loss", type=int, dest="choose_loss", default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2), SSIM (3)")
parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
parser.add_argument("--gpu", type=int,
                    dest="gpu", default=0, 
                    help="choose whether to use the gpu (1) or not (0)")
parser.add_argument("--epoch", type=int,
                    dest="epoch", default=0, 
                    help="choose which epoch is used in the evaluation (for input 0 the best version will be chosen)")
opt = parser.parse_args()

learning_rate = opt.learning_rate
smooth = opt.smth_lambda
dataset = opt.dataset
choose_loss = opt.choose_loss
mode = opt.mode
gpu = opt.gpu
epoch = opt.epoch

device = torch.device("cuda" if gpu==1 else "cpu")

if dataset == 'ACDC':
    # load ACDC test data
    test_set = TestDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
elif dataset == 'CMRxRecon':
    # load CMRxRecon test data
    test_set = TestDatasetCMRxReconBenchmark('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', False, mode) 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
elif dataset == 'OASIS':
    # path for OASIS test dataset
    test_set = TestDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS') 
    test_generator = Data.DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)
else:
    raise ValueError('Dataset should be "ACDC", "CMRxRecon" or "OASIS", but found "%s"!' % dataset)

input_shape = test_set.__getitem__(0)[0].shape[1:3]

# use dense voxelmorph
model = VxmDense(inshape=input_shape, nb_unet_features=32, bidir=False, nb_unet_levels=4).to(device)  #, int_steps=7, int_downsize=2
transform = SpatialTransformer(input_shape, mode = 'nearest').to(device)

path = './ModelParameters-{}/Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}/'.format(dataset,choose_loss,smooth,learning_rate,mode)

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
MSE_test = []
SSIM_test = []
NegJ = []
times = []
if dataset != 'CMRxRecon':
    Dice_test_full = []
    Dice_test_noBackground = []

csv_name = './TestResults-{}/TestMetrics-Voxelmorph_Loss_{}_Smth_{}_LR_{}_Mode_{}_Epoch{}.csv'.format(dataset,choose_loss,smooth,learning_rate,mode,epoch)
f = open(csv_name, 'w')
with f:
    if dataset == 'CMRxRecon':
        fnames = ['Image Pair','SSIM','MSE','Mean SSIM','Mean MSE','Mean Time','Mean NegJ']
    elif dataset == 'OASIS':
        fnames = ['Image Pair','Dice','SSIM','MSE','Mean Dice','Mean SSIM','Mean MSE','Mean Time','Mean NegJ']
    elif dataset == 'ACDC':
        fnames = ['Image Pair','Dice full','Dice no background','SSIM','MSE','Mean Dice full',' Mean Dice no background','Mean SSIM','Mean MSE','Mean Time','Mean NegJ']    
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

for i, image_pairs in enumerate(test_generator): 
    with torch.no_grad():
        mov_img_fullySampled = image_pairs[0].float().to(device)
        fix_img_fullySampled = image_pairs[1].float().to(device)
        if dataset == 'CMRxRecon':
            mov_img_subSampled = image_pairs[2].float().to(device)
            fix_img_subSampled = image_pairs[3].float().to(device)
        else:
            mov_seg = image_pairs[2].float().to(device)
            fix_seg = image_pairs[3].float().to(device)

        start = time.time()
        # calculate displacement on subsampled data
        if dataset == 'CMRxRecon':
            warped_mov_img_fullySampled, Df_xy = model(mov_img_subSampled, fix_img_subSampled)
        else:
            warped_mov_img_fullySampled, Df_xy = model(mov_img_fullySampled, fix_img_fullySampled)
        
        # get inference time
        inference_time = time.time()-start
        times.append(inference_time)
            
        if dataset != 'CMRxRecon':
            warped_mov_seg = transform(mov_seg, Df_xy) #.permute(0, 2, 3, 1)
            
        # calculate MSE, SSIM and Dice 
        if dataset == 'OASIS':
            csv_Dice_full = dice(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy())
        elif dataset == 'ACDC':
            dices_temp = dice_ACDC(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy())
            csv_Dice_full = np.mean(dices_temp)
            csv_Dice_noBackground = np.mean(dices_temp[1:3])
        csv_MSE = mean_squared_error(warped_mov_img_fullySampled[0,0,:,:].cpu().numpy(), fix_img_fullySampled[0,0,:,:].cpu().numpy())
        csv_SSIM = structural_similarity(warped_mov_img_fullySampled[0,0,:,:].cpu().numpy(), fix_img_fullySampled[0,0,:,:].cpu().numpy(), data_range=1)
                  
        MSE_test.append(csv_MSE)
        SSIM_test.append(csv_SSIM)
        if dataset == 'OASIS':
            Dice_test_full.append(csv_Dice_full)
        elif dataset == 'ACDC':
            Dice_test_full.append(csv_Dice_full)
            Dice_test_noBackground.append(csv_Dice_noBackground)
        
        # get jacobian determinant
        NegJ.append(jacobian_determinant_vxm(Df_xy.squeeze().numpy()))

        # save test results to csv file
        f = open(csv_name, 'a')
        with f:
            writer = csv.writer(f)
            if dataset == 'CMRxRecon':
                writer.writerow([i, csv_SSIM, csv_MSE, '-', '-', '-', '-']) 
            elif dataset == 'OASIS':
                writer.writerow([i, csv_Dice_full,csv_MSE, csv_SSIM, '-', '-', '-', '-', '-']) 
            elif dataset == 'ACDC':    
                writer.writerow([i, csv_Dice_full, csv_Dice_noBackground, csv_MSE, csv_SSIM, '-', '-', '-', '-', '-', '-']) 

mean_MSE = np.mean(MSE_test)
std_MSE = np.std(MSE_test)

mean_SSIM = np.mean(SSIM_test)
std_SSIM = np.std(SSIM_test)

mean_NegJ = np.mean(NegJ)
std_NegJ = np.std(NegJ)

mean_time = np.mean(times)

if dataset == 'OASIS':
    mean_Dice_full = np.mean(Dice_test_full)
    std_Dice_full = np.std(Dice_test_full)
elif dataset == 'ACDC':
    mean_Dice_full = np.mean(Dice_test_full)
    std_Dice_full = np.std(Dice_test_full)
    mean_Dice_noBackground = np.mean(Dice_test_noBackground)
    std_Dice_noBackground = np.std(Dice_test_noBackground)

f = open(csv_name, 'a')
with f:
    writer = csv.writer(f)
    if dataset == 'CMRxRecon':
        writer.writerow(['-', '-', '-', mean_SSIM, mean_MSE, mean_time, mean_NegJ])
    elif dataset == 'OASIS':
        writer.writerow(['-', '-', '-', '-', mean_Dice_full, mean_SSIM, mean_MSE, mean_time, mean_NegJ])
    elif dataset == 'ACDC':
        writer.writerow(['-', '-', '-', '-', mean_Dice_full, mean_Dice_noBackground, mean_SSIM, mean_MSE, mean_time, mean_NegJ])

if dataset == 'CMRxRecon':
    print('Mean inference time: {:.4f} seconds\n     % SSIM: {:.2f} \\pm {:.2f}\n     MSE (e-3): {:.2f} \\pm {:.2f}\n     % DetJ<0: {:.2f} \\pm {:.2f}'.format(mean_time, mean_SSIM*100, std_SSIM*100, mean_MSE*100, std_MSE*100, mean_NegJ, std_NegJ))
elif dataset == 'OASIS':
    print('Mean inference time: {:.4f} seconds\n     % DICE: {:.2f} \\pm {:.2f}\n     % SSIM: {:.2f} \\pm {:.2f}\n     MSE (e-3): {:.2f} \\pm {:.2f}\n     % DetJ<0: {:.2f} \\pm {:.2f}'.format(mean_time, mean_Dice_full*100, std_Dice_full*100, mean_SSIM*100, std_SSIM*100, mean_MSE*100, std_MSE*100, mean_NegJ, std_NegJ))
elif dataset == 'ACDC':
    print('Mean inference time: {:.4f} seconds\n     % DICE full: {:.2f} \\pm {:.2f}\n     % DICE no background: {:.2f} \\pm {:.2f}\n     % SSIM: {:.2f} \\pm {:.2f}\n     MSE (e-3): {:.2f} \\pm {:.2f}\n     % DetJ<0: {:.2f} \\pm {:.2f}'.format(mean_time, mean_Dice_full*100, std_Dice_full*100, mean_Dice_noBackground*100, std_Dice_noBackground*100, mean_SSIM*100, std_SSIM*100, mean_MSE*100, std_MSE*100, mean_NegJ, std_NegJ))
