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
                    dest="checkpoint", default=400,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/AccFactor04', #FullSample
                    help="data path for training images")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss",
                    default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2), SSIM (3)")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
iteration = opt.iteration
start_channel = opt.start_channel
n_checkpoint = opt.checkpoint
smooth = opt.smth_lambda
datapath = opt.datapath
choose_loss = opt.choose_loss

def train():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Cascade(2, 2, start_channel).cuda()
    if choose_loss == 1:
        loss_similarity = MSE().loss
    elif choose_loss == 0:
        loss_similarity = SAD().loss
    elif choose_loss == 2:
        loss_similarity = NCC(win=9)
    elif choose_loss == 3:
        ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1, win_size=9)
        loss_similarity = SAD().loss
    loss_smooth = smoothloss

    transform = SpatialTransform().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lossall = np.zeros((3, iteration))
    # load CMR data
    train_set = TrainDatasetCMR(datapath) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
    """
    # path for OASIS dataset
    datapath = /imagedata/Learn2Reg_Dataset_release_v1.1/OASIS
    train_set = TrainDataset(datapath,trainingset = trainingset) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=bs, shuffle=True, num_workers=4)
    valid_set = ValidationDataset(datapath)
    valid_generator = Data.DataLoader(dataset=valid_set, batch_size=bs, shuffle=False, num_workers=2)
    """
    model_dir = './Loss_{}_Chan_{}_Smth_{}_LR_{}_Pth/'.format(choose_loss,start_channel,smooth, lr)
    model_dir_png = './Loss_{}_Chan_{}_Smth_{}_LR_{}_Png/'.format(choose_loss,start_channel,smooth, lr)
    
    if not os.path.isdir(model_dir_png):
        os.mkdir(model_dir_png)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    csv_name = model_dir_png + 'Loss_{}_Chan_{}_Smth_{}_LR_{}.csv'.format(choose_loss,start_channel,smooth, lr)
    f = open(csv_name, 'w')
    with f:
        fnames = ['Index','Dice']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()
    
    step = 1
    print('Started training on ', time.ctime())
    while step <= iteration:
        if step == 1:
            start = time.time()
        elif step == 2:
            end = time.time()
            print('Expected time for training: ', ((end-start)*(iteration-1))/60, ' minutes.')    
        
        for mov_img, fix_img in training_generator:

            fix_img = fix_img.cuda().float()
            mov_img = mov_img.cuda().float()
            
            Df_xy = model(mov_img, fix_img)
            grid, warped_mov = transform(mov_img.unsqueeze(0), Df_xy.permute(0, 2, 3, 1)) #
           
            loss1 = loss_similarity(fix_img, warped_mov) 
            loss5 = loss_smooth(Df_xy)
            
            loss = loss1 + smooth * loss5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossall[:,step-2] = np.array([loss.item(),loss1.item(),loss5.item()])
            sys.stdout.write("\r" + 'step {0}/'.format(step) + str(iteration) + ' -> training loss "{0:.4f}" - sim "{1:.4f}" -smo "{2:.4f}" '.format(loss.item(),loss1.item(),loss5.item()))
            sys.stdout.flush()
            
            """
            if (step % n_checkpoint == 0) or (step==1):
                with torch.no_grad():
                    
                    Dices_Validation = []
                    for vmov_img, vfix_img, vmov_lab, vfix_lab in valid_generator:
                        model.eval()
                        V_xy = model(vmov_img.float().to(device), vfix_img.float().to(device))
                        DV_xy = V_xy
                        grid, warped_vmov_lab = transform(vmov_lab.float().to(device), DV_xy.permute(0, 2, 3, 1), mod = 'nearest')
                        dice_bs = dice(warped_vmov_lab[0,...].data.cpu().numpy().copy(),vfix_lab[0,...].data.cpu().numpy().copy())
                        Dices_Validation.append(dice_bs)
                    modelname = 'DiceVal_{:.4f}_Step_{:06d}.pth'.format(np.mean(Dices_Validation), step)
                    csv_dice = np.mean(Dices_Validation)
                    save_checkpoint(model.state_dict(), model_dir, modelname)
                    np.save(model_dir + 'Loss.npy', lossall)
                    f = open(csv_name, 'a')
                    with f:
                        writer = csv.writer(f)
                        writer.writerow([step, csv_dice])
            """
            if (step % n_checkpoint == 0):
                sample_path = os.path.join(model_dir, '{:06d}-images.jpg'.format(step))
                save_flow(mov_img, fix_img, warped_mov, grid.permute(0, 3, 1, 2), sample_path)
            step += 1

            if step > iteration:
                break
        print("one epoch pass")
    np.save(model_dir + '/Loss.npy', lossall)
    print('Training ended on ', time.ctime())

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
