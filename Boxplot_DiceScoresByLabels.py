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
opt = parser.parse_args()

diffeo = opt.diffeo

# define dataset and modes
dataset = 'ACDC'
modes = [0,1,2,3]

# init Dice score 6 methods (9 with Diff-versions), 4 modes, 4 labels and 224 data points
if diffeo == 1:
    DICE_test = np.zeros([9,4,4,224])
else:
    DICE_test = np.zeros([6,4,4,224])    

for mode in modes:
    # get name of csv file
    csv_name = './TestResults-{}/DiceScores_Mode_{}.csv'.format(dataset,mode)

    # get data for the models
    with open(csv_name,'r') as csvfile: 
        lines = csv.reader(csvfile, delimiter=',') 
        for i, row in enumerate(lines):   
            DICE_test[int(i/4), mode, i%4, :] = np.array(row, dtype=float)

# create labels for the x axis
labels = ['RV-Cavity', 'Myocardium', 'LV-Cavity'] #'Background', 

# create legend and init offsets so that plots do not overlap
if diffeo == 1:
    legend = ['Baseline','NiftyReg','VoxelMorph','Fourier-Net','Fourier-Net+','4xFourier-Net+','Diff-Fourier-Net','Diff-Fourier-Net+','Diff-4xFourier-Net+']
    offsets = np.arange(start=-0.9,stop=0.9,step=1.8/DICE_test.shape[0])
    width = 0.175
else:
    legend = ['Baseline','NiftyReg','VoxelMorph','Fourier-Net','Fourier-Net+','4xFourier-Net+']    
    offsets = np.arange(start=-0.65,stop=0.65,step=1.3/DICE_test.shape[0])
    width = 0.18
    anchor_loc = [0.22, 0.125]      # coords for the legend to apprear

# create path to save the boxplot to
path = './Images/Boxplot_DiceScores_'

for label in labels:
    save_path = path + label + '.png'

    if label == 'RV-Cavity':
        data = DICE_test[:,:,1,:]
    elif label == 'Myocardium':
        data = DICE_test[:,:,2,:]
    elif label == 'LV-Cavity':
        data = DICE_test[:,:,3,:]

    # create boxplot for Dice scores (without background)
    create_boxplot(savename=save_path, data=data, labels=['R=0', 'R=4', 'R=8', 'R=10'], legend=legend, figure_size=(12, 6), offsets=offsets, width=width, anchor_loc=anchor_loc)
