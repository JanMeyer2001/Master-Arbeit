import os
import sys
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from Models import *
from Functions import *
import torch.utils.data as Data
import matplotlib.pyplot as plt
import csv
import time
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--iteration", type=int,
                    dest="iteration", default=320001,
                    help="number of total iterations")
parser.add_argument("--smth_lambda", type=float,
                    dest="smth_lambda", default=0.02,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=800,
                    help="frequency of saving models")
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

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
n_checkpoint = opt.checkpoint
smooth = opt.smth_lambda
datapath = opt.datapath
trainingset = opt.trainingset
choose_loss = opt.choose_loss



def train():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = SYMNet(2, 2, start_channel).cuda()
    
    if choose_loss == 1:
        loss_similarity = MSE().loss
    elif choose_loss == 0:
        loss_similarity = SAD().loss
    elif choose_loss == 2:
        loss_similarity = NCC(win=9)
    elif choose_loss == 3:
        ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1, win_size=9)
        ssim_module = SSIM(data_range=1, size_average=True, channel=1) # channel=1 for grayscale images
        loss_similarity = SAD().loss
    
    loss_smooth = smoothloss
    transform = SpatialTransform().cuda()
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
                                                                                                                                         
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lossall = np.zeros((3, iteration))
    train_set = TrainDataset(datapath,trainingset = trainingset) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
    valid_set = ValidationDataset(datapath)
    valid_generator = Data.DataLoader(dataset=valid_set, batch_size=bs, shuffle=False, num_workers=2)
    model_dir = './Loss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_Pth/'.format(choose_loss,start_channel,smooth, trainingset, lr)
    model_dir_png = './Loss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}_Png/'.format(choose_loss,start_channel,smooth, trainingset, lr)
    
    if not os.path.isdir(model_dir_png):
        os.mkdir(model_dir_png)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    csv_name = model_dir_png + 'Loss_{}_Chan_{}_Smth_{}_Set_{}_LR_{}.csv'.format(choose_loss,start_channel,smooth, trainingset, lr)
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    
    step = 1

    while step <= iteration:
        if step == 1:
            start = time.time()
        elif step == 2:
            end = time.time()
            print('Expected time for training: ', ((end-start)*(iteration-1))/60, ' minutes.')    
        
        for X, Y in training_generator:

            X = X.cuda().float()
            Y = Y.cuda().float()
            
            out_1, out_2 = model(X, Y)
            out_1 = out_1.squeeze().squeeze()
            out_2 = out_2.squeeze().squeeze()
            out_ifft1 = torch.fft.fftshift(torch.fft.fft2(out_1))
            out_ifft2 = torch.fft.fftshift(torch.fft.fft2(out_2))
            p3d = (72, 72, 60, 60)
            out_ifft1 = F.pad(out_ifft1, p3d, "constant", 0)
            out_ifft2 = F.pad(out_ifft2, p3d, "constant", 0)
            disp_mf_1 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft1)))
            disp_mf_2 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(out_ifft2)))
            f_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
            
            D_f_xy = diff_transform(f_xy)
            grid, X_Y = transform(X, D_f_xy.permute(0, 2, 3, 1))
                                                           
            if choose_loss == 3:
                loss1 = (1 - ms_ssim_module(X_Y , Y)) + (1 - ssim_module(X_Y , Y))
            else:
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
                    for mov_img, fix_img, mov_lab, fix_lab in valid_generator:
                        model.eval()
                        vout_1, vout_2 = model(mov_img.float().to(device), fix_img.float().to(device))
                        vout_1 = vout_1.squeeze().squeeze()
                        vout_2 = vout_2.squeeze().squeeze()
                        vout_ifft1 = torch.fft.fftshift(torch.fft.fft2(vout_1))
                        vout_ifft2 = torch.fft.fftshift(torch.fft.fft2(vout_2))
                        p3d = (72, 72, 60, 60)
                        vout_ifft1 = F.pad(vout_ifft1, p3d, "constant", 0)
                        vout_ifft2 = F.pad(vout_ifft2, p3d, "constant", 0)
                        vdisp_mf_1 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(vout_ifft1)))# * (img_x * img_y * img_z / 8))))
                        vdisp_mf_2 = torch.real(torch.fft.ifft2(torch.fft.ifftshift(vout_ifft2)))# * (img_x * img_y * img_z / 8))))
                        vf_xy = torch.cat([vdisp_mf_1.unsqueeze(0).unsqueeze(0), vdisp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
                        D_vf_xy = diff_transform(vf_xy)
                        __, warped_xv_seg= transform(mov_lab.float().to(device), D_vf_xy.permute(0, 2, 3, 1), mod = 'nearest')
                        for bs_index in range(bs):
                            dice_bs=dice(warped_xv_seg[bs_index,...].data.cpu().numpy().copy(),fix_lab[bs_index,...].data.cpu().numpy().copy())
                            Dices_Validation.append(dice_bs)
                    modelname = 'DiceVal_{:.5f}_Epoch_{:09d}.pth'.format(np.mean(Dices_Validation),step)
                    csv_dice = np.mean(Dices_Validation)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([step, csv_dice])
                    save_checkpoint(model.state_dict(), model_dir, modelname)
                    np.save(model_dir_png + 'Loss.npy', lossall)
            
            if (step % n_checkpoint == 0):
                sample_path = os.path.join(model_dir_png, '{:09d}-images.jpg'.format(step))
                save_flow(X, Y, X_Y, grid.permute(0, 3, 1, 2), sample_path)
                     
            if step > iteration:
                break
            step += 1
        print("one epoch pass")
    np.save(model_dir_png + '/loss_SYMNet.npy', lossall)

def save_flow(X, Y, X_Y, f_xy, sample_path):
    x = X.data.cpu().numpy()
    y = Y.data.cpu().numpy()
    x_pred = X_Y.data.cpu().numpy()
    x_pred = x_pred[0,...]
    x = x[0,...]
    y = y[0,...]
    
    flow = f_xy.data.cpu().numpy()
    op_flow =flow[0,:,:,:]
    
    plt.subplots(figsize=(7, 4))
    plt.axis('off')

    moving_image = rotate_image(x[0, :, :])
    plt.subplot(2,3,1)
    plt.imshow(moving_image, cmap='gray', vmin=0, vmax=1)
    plt.title('Moving Image')
    plt.axis('off')

    fixed_image = rotate_image(y[0, :, :])
    plt.subplot(2,3,2)
    plt.imshow(fixed_image, cmap='gray', vmin=0, vmax=1)
    plt.title('Fixed Image')
    plt.axis('off')

    warped_image = rotate_image(x_pred[0, :, :])
    plt.subplot(2,3,3)
    plt.imshow(warped_image, cmap='gray', vmin=0, vmax=1)
    plt.title('Warped Image')
    plt.axis('off')

    plt.subplot(2,3,4)
    interval = 5
    [w,h,j] = op_flow.shape
    op_flow_new = np.zeros([w,j,h])
    op_flow_new[0,:,:] = rotate_image(op_flow[0,:,:])
    op_flow_new[1,:,:] = rotate_image(op_flow[1,:,:])

    for i in range(0,op_flow_new.shape[1]-1,interval):
        plt.plot(op_flow_new[0,i,:], op_flow_new[1,i,:],c='g',lw=1)
    #plot the vertical lines
    for i in range(0,op_flow_new.shape[2]-1,interval):
        plt.plot(op_flow_new[0,:,i], op_flow_new[1,:,i],c='g',lw=1)

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.title('Displacement Field')
    plt.axis('off')

    diff_before = rotate_image(abs(x[0, :, :]-y[0, :, :]))
    plt.subplot(2,3,5)
    plt.imshow(diff_before, cmap='gray', vmin=0, vmax=1)
    plt.title('Difference before')
    plt.axis('off')
    
    diff_after = rotate_image(abs(x_pred[0, :, :]-y[0, :, :]))
    plt.subplot(2,3,6)
    plt.imshow(diff_after, cmap='gray', vmin=0, vmax=1)
    plt.title('Difference after')
    plt.axis('off')
    plt.savefig(sample_path,bbox_inches='tight')
    plt.close()

train()
