import csv 
from argparse import ArgumentParser
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
parser.add_argument("--F_Net_plus", type=int,
                    dest="F_Net_plus", default=1, 
                    help="choose whether to use Fourier-Net (0), Fourier-Net+ (1) or cascaded Fourier-Net (2) as the model")
parser.add_argument("--diffeo", type=int,
                    dest="diffeo", default=0, 
                    help="choose whether to use a diffeomorphic transform (1) or not (0)")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
smth_lambda = opt.smth_lambda
choose_loss = opt.choose_loss
F_Net_plus = opt.F_Net_plus
diffeo = opt.diffeo


modes = [0,1,2,3]  # modes for box plot
# init MSE and SSIM for 4 modes, 2 methods and 9669 data points
MSE_test = np.zeros([4,2,9669])
SSIM_test = np.zeros([4,2,9669])
path = './TestResults-Metrics/TestMetrics-'

for mode in modes:
    model_path = 'Model_{}_Diffeo_{}_Loss_{}_Chan_{}_Smth_{}_LR_{}_Mode_{}.csv'.format(F_Net_plus,diffeo,choose_loss,start_channel,smth_lambda, lr,mode)
    nifty_path = 'NiftyReg_Mode{}.csv'.format(mode)

    # get data for the model
    with open(path+model_path,'r') as csvfile: 
        lines = csv.reader(csvfile, delimiter=',') 
        for i, row in enumerate(lines): 
            if mode == 3:
                if i>0 and i<9559:              # Acc10 has slightly less data
                    MSE_test[mode,0,i] = float(row[1])
                    SSIM_test[mode,0,i] = float(row[2])
            else:    
                if i>0 and i<MSE_test.shape[2]:      # rest should have 9669
                    MSE_test[mode,0,i] = float(row[1])
                    SSIM_test[mode,0,i] = float(row[2])

    # get data for NiftyReg            
    with open(path+nifty_path,'r') as csvfile: 
        lines = csv.reader(csvfile, delimiter=',') 
        for i, row in enumerate(lines): 
            if mode == 3:
                if i>0 and i<9559:              # Acc10 has slightly less data
                    MSE_test[mode,1,i] = float(row[1])
                    SSIM_test[mode,1,i] = float(row[2])
            else:    
                if i>0 and i<MSE_test.shape[2]:      # rest should have 9669 
                    MSE_test[mode,1,i] = float(row[1])
                    SSIM_test[mode,1,i] = float(row[2])

# sort MSE data for box plot
data_model_MSE = [MSE_test[0,0,:], MSE_test[1,0,:], MSE_test[2,0,:], MSE_test[3,0,:][0:9559]]
data_NiftyReg_MSE = [MSE_test[0,1,:], MSE_test[1,1,:], MSE_test[2,1,:], MSE_test[3,1,:][0:9559]]

# sort SSIM data for box plot
data_model_SSIM = [SSIM_test[0,0,:], SSIM_test[1,0,:], SSIM_test[2,0,:], SSIM_test[3,0,:][0:9559]]
data_NiftyReg_SSIM = [SSIM_test[0,1,:], SSIM_test[1,1,:], SSIM_test[2,1,:], SSIM_test[3,1,:][0:9559]]

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
