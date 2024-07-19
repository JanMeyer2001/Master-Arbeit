import csv 
from argparse import ArgumentParser
import numpy as np
from Functions import *
import torch
from Models import *
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--start_channel", type=int, dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--FT_size_x", type=int,
                    dest="FT_size_x", default=24,
                    help="choose size x of FT crop: Should be smaller than 40.")
parser.add_argument("--FT_size_y", type=int,
                    dest="FT_size_y", default=24,
                    help="choose size y of FT crop: Should be smaller than 84.")
parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
opt = parser.parse_args()

start_channel = opt.start_channel
FT_size = [opt.FT_size_x,opt.FT_size_y]
mode = opt.mode

dataset = 'ACDC'
input_shape = [216,256]
diffeo = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
transform = SpatialTransform().to(device)
transform_voxelmorph = SpatialTransformer(input_shape, mode = 'nearest').to(device)
csv_name = './TestResults-{}/DiceScores_Mode_{}.csv'.format(dataset,mode)

# init Dice score 6 methods, 4 labels and 224 data points
DICE_test = np.zeros([6,4,224])# init different models

# get data for the models
with open(csv_name,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for i, row in enumerate(lines):   
        DICE_test[int(i/4), i%4, :] = np.array(row, dtype=float)

# for testing to see whether the csv-file is correctly read-in
#print('Mean Dice scores:\n    Baseline          {}\n    NiftyReg          {}\n    VoxelMorph        {}\n    Fourier-Net       {}\n    Fourier-Net+      {}\n    4xFourier-Net+   {}'.format(np.mean(DICE_test[0,:,:]), np.mean(DICE_test[1,:,:]), np.mean(DICE_test[2,:,:]), np.mean(DICE_test[3,:,:]), np.mean(DICE_test[4,:,:]), np.mean(DICE_test[5,:,:])))

# create labels for the x axis
labels = ['Background', 'RV Cavity', 'Myocardium', 'LV Cavity']
# create legend 
legend = ['Baseline','NiftyReg','VoxelMorph','Fourier-Net','Fourier-Net+','4xFourier-Net+']
# create path to save the boxplot to
save_path = '/home/jmeyer/storage/students/janmeyer_711878/Master-Arbeit/Thesis/Images/Boxplot_DiceScores.png'
# init offsets so that plots do not overlap
offsets = np.arange(start=-0.6,stop=0.6,step=1.2/DICE_test.shape[0])
#print('offsets: ',offsets)
# create boxplot for MSE
create_boxplot(savename=save_path, data=DICE_test, labels=labels, legend=legend, figure_size=(10, 6), offsets=offsets, width=0.175)
