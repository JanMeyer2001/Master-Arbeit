from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
import csv
from skimage.metrics import structural_similarity, mean_squared_error

parser = ArgumentParser()
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon',
                    #default='/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/FullSample', #AccFactor04
                    help="data path for training images")
parser.add_argument("--mode", type=int,
                    dest="mode",
                    default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
opt = parser.parse_args()

datapath = opt.datapath
mode = opt.mode

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

# load CMR validation data
bs = 1
test_set = TestDatasetCMRBenchmark(datapath, mode) 
test_generator = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=True, num_workers=4)

csv_name = './Test-Results-Metrics/TestMetrics-Baseline_Mode' + str(mode) + '.csv'
f = open(csv_name, 'w')
with f:
    fnames = ['Index','MSE','SSIM','Mean MSE','Mean SSIM']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

image_num = 1 
MSE_Test = []
SSIM_Test = []
for mov_img, fix_img,_ ,_ in test_generator: 
    # convert to numpy array
    mov_img = mov_img[0,0,:,:].cpu().numpy()
    fix_img = fix_img[0,0,:,:].cpu().numpy()
    
    MSE = mean_squared_error(mov_img, fix_img)
    MSE_Test.append(MSE)
    
    SSIM = structural_similarity(mov_img, fix_img, data_range=1)
    SSIM_Test.append(SSIM)
    
    f = open(csv_name, 'a')
    with f:
        writer = csv.writer(f)
        writer.writerow([image_num, MSE, SSIM, '-', '-']) 
    image_num += 1

csv_MSE = np.mean(MSE_Test)
csv_SSIM = np.mean(SSIM_Test)
f = open(csv_name, 'a')
with f:
    writer = csv.writer(f)
    writer.writerow(['-', '-', '-', csv_MSE, csv_SSIM])    

print('\n mean MSE: ', csv_MSE,'\n mean SSIM: ', csv_SSIM)