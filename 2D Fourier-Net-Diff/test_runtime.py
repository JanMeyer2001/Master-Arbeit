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
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=320001,
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
                    dest="checkpoint", default=800,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    #default='/export/local/xxj946/AOSBraiCN2',
                    default='/bask/projects/d/duanj-ai-imaging/Accreg/brain/OASIS_AffineData/',
                    help="data path for training images")
parser.add_argument("--trainingset", type=int,
                    dest="trainingset", default=3,
                    help="1 Half : 200 Images, 2 The other Half 200 Images 3 All 400 Images")
parser.add_argument("--using_l2", type=int,
                    dest="using_l2",
                    default=1,
                    help="using l2 or not")
opt = parser.parse_args()


def test(modelpath):
    use_cuda = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if use_cuda else "cpu")
    bs = 1
    model = SYMNet(2, 2, opt.start_channel).to(device)
    transform = SpatialTransform().to(device)
    diff_transform = DiffeomorphicTransform(time_step=7).to(device)
    model.load_state_dict(torch.load(modelpath, map_location = device))
    model.eval()
    transform.eval()
    diff_transform.eval()
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
            disp_mf_1 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft1)))# * (img_x * img_y * img_z / 8))))
            disp_mf_2 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft2)))# * (img_x * img_y * img_z / 8))))
            V_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
            D_vf_xy = diff_transform(V_xy)
            

if __name__ == '__main__':
    model_path='./L2ss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_Pth/'.format(opt.using_l2, opt.start_channel, opt.smth_labda, opt.trainingset, opt.lr)
    model_idx = -1
    from natsort import natsorted
    print('Best model: {}'.format(natsorted(os.listdir(model_path))[model_idx]))
    import time
    start = time.time()
    test(model_path + natsorted(os.listdir(model_path))[model_idx])
    print((time.time() - start)/400)
    