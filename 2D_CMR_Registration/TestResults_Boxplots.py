import matplotlib.pyplot as plt 
import csv 
from argparse import ArgumentParser
import argparse
import numpy as np
from Functions import *
  
parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--smth_lambda", type=float,
                    dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss",
                    default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2), SSIM (3)")
parser.add_argument("--F_Net_plus", type=bool,
                    dest="F_Net_plus", default=True, action=argparse.BooleanOptionalAction, 
                    help="choose whether to use Fourier-Net (False) or Fourier-Net+ (True) as the model")
parser.add_argument("--diffeo", type=bool,
                    dest="diffeo", default=True, action=argparse.BooleanOptionalAction, 
                    help="choose whether to use a diffeomorphic transform (True) or not (False)")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
smth_lambda = opt.smth_lambda
choose_loss = opt.choose_loss
F_Net_plus = opt.F_Net_plus
diffeo = opt.diffeo


model_name = 0
if F_Net_plus:
    model_name = 1
diffeo_name = 0
if diffeo:
    diffeo_name = 1

modes = [0,1,2,3]  # modes for box plot
# init MSE and SSIM for 4 modes, 2 methods and 9669 data points
MSE = np.zeros([4,2,9669])
SSIM = np.zeros([4,2,9669])
path = './TestResults-Metrics/TestMetrics-'

for mode in modes:
    model_path = 'Model_{}_Diffeo_{}_Loss_{}_Chan_{}_Smth_{}_LR_{}_Mode_{}.csv'.format(model_name,diffeo_name,choose_loss,start_channel,smth_lambda, lr,mode)
    nifty_path = 'NiftyReg_Mode{}.csv'.format(mode)

    # get data for the model
    with open(path+model_path,'r') as csvfile: 
        lines = csv.reader(csvfile, delimiter=',') 
        for i, row in enumerate(lines): 
            if mode == 3:
                if i>0 and i<9559:              # Acc10 has slightly less data
                    MSE[mode,0,i] = float(row[1])
                    SSIM[mode,0,i] = float(row[2])
            else:    
                if i>0 and i<MSE.shape[2]:      # rest should have 9669
                    MSE[mode,0,i] = float(row[1])
                    SSIM[mode,0,i] = float(row[2])

    # get data for NiftyReg            
    with open(path+nifty_path,'r') as csvfile: 
        lines = csv.reader(csvfile, delimiter=',') 
        for i, row in enumerate(lines): 
            if mode == 3:
                if i>0 and i<9559:              # Acc10 has slightly less data
                    MSE[mode,1,i] = float(row[1])
                    SSIM[mode,1,i] = float(row[2])
            else:    
                if i>0 and i<MSE.shape[2]:      # rest should have 9669 
                    MSE[mode,1,i] = float(row[1])
                    SSIM[mode,1,i] = float(row[2])

# sort MSE data for box plot
data_model_MSE = [MSE[0,0,:], MSE[1,0,:], MSE[2,0,:], MSE[3,0,:][0:9559]]
data_NiftyReg_MSE = [MSE[0,1,:], MSE[1,1,:], MSE[2,1,:], MSE[3,1,:][0:9559]]

# sort SSIM data for box plot
data_model_SSIM = [SSIM[0,0,:], SSIM[1,0,:], SSIM[2,0,:], SSIM[3,0,:][0:9559]]
data_NiftyReg_SSIM = [SSIM[0,1,:], SSIM[1,1,:], SSIM[2,1,:], SSIM[3,1,:][0:9559]]

# create labels for the x axis
labels = ['Fully Sampled', 'Acc4', 'Acc8', 'Acc10']
# create legend 
legend = ['Fourier-Net+','NiftyReg']
# create path to save the boxplot to
save_path = '/home/jmeyer/storage/students/janmeyer_711878/Master-Arbeit/Thesis/Images/TestBenchmark_MSE_Boxplots.png'
# create title
#title='Test Benchmark - MSE'
# create boxplot for MSE
create_AB_boxplot(savename=save_path, data_A=data_model_MSE, data_B=data_NiftyReg_MSE, labels=labels, legend=legend, figure_size=(10, 6))

# create path to save the boxplot to
save_path = '/home/jmeyer/storage/students/janmeyer_711878/Master-Arbeit/Thesis/Images/TestBenchmark_SSIM_Boxplots.png'
# create title
# title = 'Test Benchmark - SSIM'
# create boxplot for SSIM
create_AB_boxplot(savename=save_path, data_A=data_model_SSIM, data_B=data_NiftyReg_SSIM, labels=labels, legend=legend, figure_size=(10, 6))
