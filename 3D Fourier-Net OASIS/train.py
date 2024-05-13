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
parser.add_argument("--mask_labda", type=float,
                    dest="mask_labda", default=0.25,
                    help="mask_labda loss: suggested range 0.1 to 10")
parser.add_argument("--data_labda", type=float,
                    dest="data_labda", default=0.02,
                    help="data_labda loss: suggested range 0.1 to 10")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.02,
                    help="labda loss: suggested range 0.1 to 10")
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
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
local_ori = opt.local_ori
magnitude = opt.magnitude
n_checkpoint = opt.checkpoint
smooth = opt.smth_labda
datapath = opt.datapath
mask_labda = opt.mask_labda
data_labda = opt.data_labda
trainingset = opt.trainingset
using_l2 = opt.using_l2

def dice(pred1, truth1):
    mask4_value1 = np.unique(pred1)
    mask4_value2 = np.unique(truth1)
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    dice_list=[]
    for k in mask_value4[1:]:
        #print(k)
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        # print(intersection)
        dice_list.append(intersection / (np.sum(pred) + np.sum(truth)))
    return np.mean(dice_list)

def dice_old(pred1, truth1):
    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
    dice_35=np.zeros(len(VOI_lbls))
    index = 0
    for k in VOI_lbls:
        #print(k)
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        # print(intersection)
        dice_35[index]=intersection / (np.sum(pred) + np.sum(truth))
        index = index + 1
    return np.mean(dice_35)

def save_checkpoint(state, save_dir, save_filename, max_model_num=10):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    # print(model_lists)
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def train():
    print('Started training on ', time.ctime())
    
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    
    model = SYMNet(2, 3, start_channel).cuda()
    if using_l2 == 1:
        loss_similarity = MSE().loss
    elif using_l2 == 0:
        loss_similarity = SAD().loss
    elif using_l2 == 2:
        loss_similarity = NCC()
    loss_smooth = smoothloss

    transform = SpatialTransform().cuda()
    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = './L2ss_{}_Chan_{}_Smth_{}_LR_{}/'.format(using_l2,start_channel,smooth,lr)
    if not os.path.isdir(model_dir+'model/'):
        os.mkdir(model_dir)
        os.mkdir(model_dir+'model/')
    csv_name = model_dir + 'L2ss_{}_Chan_{}_Smth_{}_LR_{}.csv'.format(using_l2,start_channel,smooth,lr)
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
            
            out_1, out_2, out_3 = model(X, Y)
            out_1 = out_1.squeeze().squeeze()
            out_2 = out_2.squeeze().squeeze()
            out_3 = out_3.squeeze().squeeze()
            out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
            out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
            out_ifft3 = torch.fft.fftshift(torch.fft.fftn(out_3))
            # print(out_ifft1.shape)
            p3d = (72, 72, 84, 84, 60, 60)
            out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
            out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)
            out_ifft3 = F.pad(out_ifft3, p3d, "constant", 0)
            disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))# * (img_x * img_y * img_z / 8))))
            disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))# * (img_x * img_y * img_z / 8))))
            disp_mf_3 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft3)))# * (img_x * img_y * img_z / 8))))
            f_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0), disp_mf_3.unsqueeze(0).unsqueeze(0)], dim = 1)
            """
            print('size of X: ', X.shape)
            print('size of f_xy: ', f_xy.permute(0, 2, 3, 4, 1).shape)
            """
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
                        vout_1, vout_2, vout_3  = model(xv.float().to(device), yv.float().to(device))
                        vout_1 = vout_1.squeeze().squeeze()
                        vout_2 = vout_2.squeeze().squeeze()
                        vout_3 = vout_3.squeeze().squeeze()
                        vout_ifft1 = torch.fft.fftshift(torch.fft.fftn(vout_1))
                        vout_ifft2 = torch.fft.fftshift(torch.fft.fftn(vout_2))
                        vout_ifft3 = torch.fft.fftshift(torch.fft.fftn(vout_3))
                        p3d = (72, 72, 84, 84, 60, 60)
                        vout_ifft1 = F.pad(vout_ifft1, p3d, "constant", 0)
                        vout_ifft2 = F.pad(vout_ifft2, p3d, "constant", 0)
                        vout_ifft3 = F.pad(vout_ifft3, p3d, "constant", 0)
                        vdisp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft1)))
                        vdisp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft2)))
                        vdisp_mf_3 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft3)))
                        vf_xy = torch.cat([vdisp_mf_1.unsqueeze(0).unsqueeze(0), vdisp_mf_2.unsqueeze(0).unsqueeze(0), vdisp_mf_3.unsqueeze(0).unsqueeze(0)], dim = 1)
                        warped_xv_seg= transform(xv_seg.float().to(device), vf_xy.permute(0, 2, 3, 4, 1), mod = 'nearest')
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
