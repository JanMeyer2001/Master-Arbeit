import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from Models import *
from Functions import *
import torch.utils.data as Data
from natsort import natsorted
import csv

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--smth_lambda", type=float,
                    dest="smth_lambda", default=0.02,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS',
                    help="data path for training images")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=4,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss",
                    default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2)")
opt = parser.parse_args()


def test(model_dir):
    bs = 1
    use_cuda = True
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Cascade(2, 2, opt.start_channel).to(device)
    
    
    model_idx = -1
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])#['state_dict']
    model.load_state_dict(best_model)
    
    transform = SpatialTransform().to(device)
    diff_transform = DiffeomorphicTransform(time_step=7).to(device)
    model.eval()
    transform.eval()
    diff_transform.eval()
    Dices_35=[]
    NegJ_35=[]
    GradJ_35=[]
    test_set = TestDataset(opt.datapath)
    test_generator = Data.DataLoader(dataset=test_set, batch_size=bs,
                                         shuffle=False, num_workers=2)
    for __, __, mov_img, fix_img, mov_lab, fix_lab in test_generator:
        with torch.no_grad():
            V_xy = model(mov_img.float().to(device), fix_img.float().to(device))
            
if __name__ == '__main__':
    model_dir = './Loss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_BZ_{}_Pth1/'.format(opt.choose_loss, opt.start_channel, opt.smth_lambda, opt.trainingset, opt.lr, opt.bs)
    print(model_dir)
    import time
    start = time.time()
    dice35_temp= test(model_dir)
    print((time.time()-start)/400.0)
   