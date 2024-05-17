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
parser.add_argument("--smth_lambda", type=float,
                    dest="smth_lambda", default=0.25,
                    help="smth_lambda loss: suggested range 0.1 to 10")
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
start_channel = opt.start_channel
n_checkpoint = opt.checkpoint
smooth = opt.smth_lambda
datapath = opt.datapath
choose_loss = opt.choose_loss

def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model_idx = -1
    model_dir = './Loss_{}_Chan_{}_Smth_{}_LR_{}/'.format(choose_loss, start_channel, smooth, lr)    
    model = SYMNet(2, 3, start_channel).to(device)
    
    best_model = torch.load(model_dir+'model/' + natsorted(os.listdir(model_dir+'model/'))[model_idx])#['state_dict']
    model.load_state_dict(best_model)

    test_set = TestDataset(datapath)
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
            vdisp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft1)))
            vdisp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft2)))
            vdisp_mf_3 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(vout_ifft3)))
            D_f_xy = torch.cat([vdisp_mf_1.unsqueeze(0).unsqueeze(0), vdisp_mf_2.unsqueeze(0).unsqueeze(0), vdisp_mf_3.unsqueeze(0).unsqueeze(0)], dim = 1)
            
def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    start = time.time()
    main()
    print('runtime: ', (time.time() -start) /115, ' seconds')