import glob
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from Functions import *
from Models import *
import time

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--bs", type=int,
                    dest="bs", default=1, help="batch_size")
parser.add_argument("--smth_lambda", type=float,
                    dest="smth_lambda", default=0.02,
                    help="smth_lambda loss: suggested range 0.1 to 10")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS',
                    help="data path for training images")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss",
                    default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2)")
opt = parser.parse_args()

lr = opt.lr
bs = opt.bs
start_channel = opt.start_channel
smooth = opt.smth_lambda
datapath = opt.datapath
choose_loss = opt.choose_loss

def main():
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    transform = SpatialTransform().cuda()

    model_idx = -1
    model_dir = './Loss_{}_Chan_{}_Smth_{}_LR_{}/'.format(choose_loss, start_channel, smooth, lr)
   
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+model_dir[:-1]+'_Test.csv'):
        os.remove('Quantitative_Results/'+model_dir[:-1]+'_Test.csv')
    csv_writter(model_dir[:-1], 'Quantitative_Results/' + model_dir[:-1]+'_Test')
    line = ''
    csv_writter(line +','+'non_jec', 'Quantitative_Results/' + model_dir[:-1]+'_Test')

    model = Cascade(2, 3, start_channel).cuda()
    
    print('Best model: {}'.format(natsorted(os.listdir(model_dir+'model/'))[model_idx]))
    best_model = torch.load(model_dir+'model/' + natsorted(os.listdir(model_dir+'model/'))[model_idx])#['state_dict']
    model.load_state_dict(best_model)
    model.cuda()
    
    test_set = TestDataset(datapath)
    test_loader = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)

    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            if stdy_idx == 0:
                start = time.time()
            elif stdy_idx == 1:
                end = time.time()
                expected_time = len(test_loader)*(end-start)/60   
                print('Test Evaluation will take about ', expected_time, ' minutes...') 
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
           
            D_f_xy = model(x.float().to(device), y.float().to(device))
            def_out= transform(x_seg.float().to(device), D_f_xy.permute(0, 2, 3, 4, 1), mod = 'nearest')
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            
            dd, hh, ww = D_f_xy.shape[-3:]
            D_f_xy = D_f_xy.detach().cpu().numpy()
            D_f_xy[:,0,:,:,:] = D_f_xy[:,0,:,:,:] * dd / 2
            D_f_xy[:,1,:,:,:] = D_f_xy[:,1,:,:,:] * hh / 2
            D_f_xy[:,2,:,:,:] = D_f_xy[:,2,:,:,:] * ww / 2
            
            jac_det = jacobian_determinant_vxm(D_f_xy[0, :, :, :, :])
            line = dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line +','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + model_dir[:-1]+'_Test')
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans = dice(def_out.long(), y_seg.long(), 46)
            dsc_raw = dice(x_seg.long(), y_seg.long(), 46)
            print('Aligned Dice: {:.4f}, Raw Dice: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Aligned Dice: {:.3f} +- {:.3f}, Raw Dice: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('Aligned det: {}, std: {}'.format(eval_det.avg, eval_det.std))


def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    main()