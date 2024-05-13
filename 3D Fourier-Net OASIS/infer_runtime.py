import glob
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from Functions import *
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from Models import *
import time

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=320001,
                    help="number of total iterations")
parser.add_argument("--local_ori", type=float,
                    dest="local_ori", default=1000.0,
                    help="Local Orientation Consistency loss: suggested range 1 to 1000")
parser.add_argument("--magnitude", type=float,
                    dest="magnitude", default=1000.0,
                    help="magnitude loss: suggested range 0.001 to 1.0")
parser.add_argument("--smth_labda", type=float,
                    dest="smth_labda", default=0.25,
                    help="smth_labda loss: suggested range 0.1 to 10")
parser.add_argument("--data_labda", type=float,
                    dest="data_labda", default=0.02,
                    help="data_labda loss: suggested range 0.1 to 10")
parser.add_argument("--fft_labda", type=float,
                    dest="fft_labda", default=0.02,
                    help="fft_labda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=403,
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
data_labda = opt.data_labda
using_l2 = opt.using_l2

def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model_idx = -1
    model_dir = './L2ss_{}_Chan_{}_Smth_{}_LR_{}/'.format(using_l2, start_channel, smooth, lr)    
    model = SYMNet(2, 3, start_channel).to(device)
    
    best_model = torch.load(model_dir+'model/' + natsorted(os.listdir(model_dir+'model/'))[model_idx])#['state_dict']
    model.load_state_dict(best_model)

    test_set = TestDataset(opt.datapath)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

    with torch.no_grad():
        for data in test_loader:
            model.eval()
            data = [t.to(device) for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            vout_1, vout_2, vout_3  = model(x.float().to(device), y.float().to(device))
            vout_1 = vout_1.squeeze().squeeze()
            vout_2 = vout_2.squeeze().squeeze()
            vout_3 = vout_3.squeeze().squeeze()
            vout_ifft1 = torch.fft.fftshift(torch.fft.fftn(vout_1))
            vout_ifft2 = torch.fft.fftshift(torch.fft.fftn(vout_2))
            vout_ifft3 = torch.fft.fftshift(torch.fft.fftn(vout_3))
            p3d = (84, 84, 72, 72, 60, 60)
            vout_ifft1 = F.pad(vout_ifft1, p3d, "constant", 0)
            vout_ifft2 = F.pad(vout_ifft2, p3d, "constant", 0)
            vout_ifft3 = F.pad(vout_ifft3, p3d, "constant", 0)
            vdisp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft1)))# * (img_x * img_y * img_z / 8))))
            vdisp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft2)))# * (img_x * img_y * img_z / 8))))
            vdisp_mf_3 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft3)))# * (img_x * img_y * img_z / 8))))
            D_f_xy = torch.cat([vdisp_mf_1.unsqueeze(0).unsqueeze(0), vdisp_mf_2.unsqueeze(0).unsqueeze(0), vdisp_mf_3.unsqueeze(0).unsqueeze(0)], dim = 1)
            
def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    start = time.time()
    main()
    print('runtime: ', (time.time() -start) /115, ' seconds')