import os
import glob
import sys
from argparse import ArgumentParser
import numpy as np
import torch.nn.functional as F
import torch
from torchvision import transforms
from Models import *
from Functions import *
import torch.utils.data as Data
from natsort import natsorted
import csv
import time

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=120000,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=1000.0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--magnitude", type=float,
                    dest="magnitude", default=0.001,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--smth_lambda", type=float,
                    dest="smth_lambda", default=0.02,
                    help="labmda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=401,
                    help="frequency of saving models")
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

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
local_ori = opt.local_ori
magnitude = opt.magnitude
n_checkpoint = opt.checkpoint
smooth = opt.smth_lambda
datapath = opt.datapath
trainingset = opt.trainingset
choose_loss = opt.choose_loss

def train():
    print('Started training on ', time.ctime())
    
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = Cascade(2, 3, start_channel).cuda()
    if choose_loss == 1:
        loss_similarity = MSE().loss
    elif choose_loss == 0:
        loss_similarity = SAD().loss
    elif choose_loss == 2:
        loss_similarity = NCC()
    loss_smooth = smoothloss

    transform = SpatialTransform().cuda()
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = './Loss_{}_Chan_{}_Smth_{}_LR_{}/'.format(choose_loss,start_channel,smooth,lr)
    if not os.path.isdir(model_dir+'model/'):
        os.mkdir(model_dir)
        os.mkdir(model_dir+'model/')
    csv_name = model_dir + 'Loss_{}_Chan_{}_Smth_{}_LR_{}.csv'.format(choose_loss,start_channel,smooth,lr)
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    lossall = np.zeros((3, iteration))
    
    train_set = TrainDataset(datapath,trainingset = trainingset) #
    train_loader = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
    val_set = ValidationDataset(opt.datapath)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=bs, shuffle=False, num_workers=2)
    f = open(csv_name, 'w')

    step = 1
    epoch = 0
    while step <= iteration:
        for X, Y in train_loader:

            X = X.cuda().float()
            Y = Y.cuda().float()
            
            f_xy = model(X, Y)
            X_Y = transform(X, f_xy.permute(0, 2, 3, 4, 1))
            
            loss1 = loss_similarity(Y, X_Y)
            loss5 = loss_smooth(f_xy)
            loss = loss1 + smooth * loss5
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step-2] = np.array([loss.item(),loss1.item(),loss5.item()])
            sys.stdout.write("\r" + 'step {0}/'.format(step) + str(iteration) + ' -> training loss "{0:.4f}" - sim "{1:.4f}" -smo "{2:.4f}" '.format(loss.item(),loss1.item(),loss5.item()))
            sys.stdout.flush()

            if (step % n_checkpoint == 0):
                with torch.no_grad():
                    Dices_Validation = []
                    for data in val_loader:
                        model.eval()
                        
                        xv = data[0]
                        yv = data[1]
                        
                        xv_seg = data[2]
                        yv_seg = data[3]
                        
                        vf_xy = model(xv.float().to(device), yv.float().to(device))
                        warped_xv_seg = transform(xv_seg.float().to(device), vf_xy.permute(0, 2, 3, 4, 1), mod = 'nearest')
                        
                        for bs_index in range(bs):
                            dice_bs=dice(warped_xv_seg[bs_index,...].data.cpu().numpy().copy(),yv_seg[bs_index,...].data.cpu().numpy().copy())
                            Dices_Validation.append(dice_bs)
                    modelname = 'DiceVal_{:.4f}_Step_{:06d}_Epoch_{:03d}.pth'.format(np.mean(Dices_Validation), step, epoch)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, np.mean(Dices_Validation)])
                    save_checkpoint(model.state_dict(), model_dir+'/model/', modelname)
                    np.save(model_dir + 'Loss.npy', lossall)
                    print("one epoch pass")
                    epoch = epoch + 1
            step += 1

            if step > iteration:
                break
    np.save(model_dir + 'Loss.npy', lossall)
    print('Training ended on ', time.ctime())
    
train()
