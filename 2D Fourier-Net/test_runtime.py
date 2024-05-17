import os
from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
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
                    dest="trainingset", default=3,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss",
                    default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2)")
opt = parser.parse_args()


def test(modelpath):
    use_cuda = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if use_cuda else "cpu")
    bs = 1
    model = SYMNet(2, 2, opt.start_channel).to(device)
    transform = SpatialTransform().to(device)
    model.load_state_dict(torch.load(modelpath, map_location = device))
    model.eval()
    transform.eval()
    test_set = TestDataset(opt.datapath)
    test_generator = Data.DataLoader(dataset=test_set, batch_size=bs,
                                         shuffle=False, num_workers=2)
    for __, __, mov_img, fix_img, mov_lab, fix_lab in test_generator:
        with torch.no_grad():
            out_1, out_2 = model(mov_img.float().to(device), fix_img.float().to(device))
            out_1 = out_1.squeeze().squeeze()
            out_2 = out_2.squeeze().squeeze()
            out_ifft1 = torch.fft.fftshift(torch.fft.fft2(out_1))
            out_ifft2 = torch.fft.fftshift(torch.fft.fft2(out_2))
            p3d = (72, 72, 60, 60)
            out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
            out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)
            disp_mf_1 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft1)))
            disp_mf_2 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft2)))
            V_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
            
if __name__ == '__main__':
    model_path='./Loss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_Pth/'.format(opt.choose_loss, opt.start_channel, opt.smth_lambda, opt.trainingset, opt.lr)
    model_idx = -1
    from natsort import natsorted
    print('Best model: {}'.format(natsorted(os.listdir(model_path))[model_idx]))
    import time
    start = time.time()
    test(model_path + natsorted(os.listdir(model_path))[model_idx])
    print('runtime: ', (time.time() - start)/400)
    