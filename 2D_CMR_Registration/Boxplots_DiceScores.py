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
parser.add_argument("--diffeo", type=int,
                    dest="diffeo", default=0, 
                    help="choose whether to use a diffeomorphic transform (1) or not (0)")
parser.add_argument("--mode", type=int, dest="mode", default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
opt = parser.parse_args()

mode = opt.mode
diffeo = opt.diffeo

dataset = 'ACDC'
csv_name = './TestResults-{}/DiceScores_Mode_{}.csv'.format(dataset,mode)

# init Dice score 6 methods (9 with Diff-versions), 4 labels and 224 data points
if diffeo == 1:
    DICE_test = np.zeros([9,4,224])
else:
    DICE_test = np.zeros([6,4,224])    

# get data for the models
with open(csv_name,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for i, row in enumerate(lines):   
        DICE_test[int(i/4), i%4, :] = np.array(row, dtype=float)

# for testing to see whether the csv-file is correctly read-in
#print('Mean Dice scores:\n    Baseline          {}\n    NiftyReg          {}\n    VoxelMorph        {}\n    Fourier-Net       {}\n    Fourier-Net+      {}\n    4xFourier-Net+   {}'.format(np.mean(DICE_test[0,:,:]), np.mean(DICE_test[1,:,:]), np.mean(DICE_test[2,:,:]), np.mean(DICE_test[3,:,:]), np.mean(DICE_test[4,:,:]), np.mean(DICE_test[5,:,:])))

# create labels for the x axis
labels = ['RV Cavity', 'Myocardium', 'LV Cavity'] #'Background', 

# create legend and init offsets so that plots do not overlap
if diffeo == 1:
    legend = ['Baseline','NiftyReg','VoxelMorph','Fourier-Net','Fourier-Net+','4xFourier-Net+','Diff-Fourier-Net','Diff-Fourier-Net+','Diff-4xFourier-Net+']
    offsets = np.arange(start=-0.9,stop=0.9,step=1.8/DICE_test.shape[0])
    width = 0.175
else:
    legend = ['Baseline','NiftyReg','VoxelMorph','Fourier-Net','Fourier-Net+','4xFourier-Net+']    
    offsets = np.arange(start=-0.75,stop=0.75,step=1.5/DICE_test.shape[0])
    width = 0.19

# create path to save the boxplot to
save_path = '/home/jmeyer/storage/students/janmeyer_711878/Master-Arbeit/Thesis/Images/Boxplot_DiceScores_FullySampled.png'

# create boxplot for Dice scores (without background)
create_boxplot(savename=save_path, data=DICE_test[:,1:4,:], labels=labels, legend=legend, figure_size=(10, 6), offsets=offsets, width=width)
