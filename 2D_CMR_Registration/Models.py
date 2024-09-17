import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # to shut tensorflow the fuck up
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
import inspect
import functools
import math
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.initializers import VarianceScaling
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, LeakyReLU, Lambda, Input, AveragePooling2D, Layer
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.regularizers import l2
from  Functions import *

class SYMNet_dense(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(SYMNet_dense, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=5, stride=1, padding=2, bias=False)
        self.dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=5, stride=1, padding=2, bias=False)

        self.up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))#,
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        d3 = torch.cat((self.up4(d2), e0), 1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)

        f_xy = self.dc9(d3)
        f_yx = self.dc10(d3)

        return f_xy[:,0:1,:,:], f_xy[:,1:2,:,:]

class Fourier_Net_dense(nn.Module):
    """ Version of Fourier-Net that gives back a dense displacement field """
    def __init__(self, in_channel, n_classes, start_channel, diffeo):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.diffeo = diffeo

        super(Fourier_Net_dense, self).__init__()
        self.model = SYMNet_dense(self.in_channel, self.n_classes, self.start_channel)
        if self.diffeo == 1:
            self.diff_transform = DiffeomorphicTransform(time_step=7).cuda()
        
    def forward(self, Moving, Fixed):
        out_1, out_2 = self.model(Moving, Fixed)

        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
                
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        f_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)

        if self.diffeo == 1:
            Df_xy = self.diff_transform(f_xy)
        else:
            Df_xy = f_xy
         
        return Df_xy

class Fourier_Net_plus_dense(nn.Module):
    """ Version of Fourier-Net+ that gives back a dense displacement field """
    def __init__(self, in_channel, n_classes, start_channel, diffeo, offset):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.offset = offset
        self.diffeo = diffeo

        super(Fourier_Net_plus_dense, self).__init__()
        self.model = SYMNet_dense(self.in_channel, self.n_classes, self.start_channel)
        if self.diffeo == 1:
            self.diff_transform = DiffeomorphicTransform(time_step=7).cuda()
        
    def forward(self, Moving, Fixed):
        M_temp = Moving.squeeze().squeeze()
        F_temp = Fixed.squeeze().squeeze()
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        F_temp_fourier_all = torch.fft.fftn(F_temp)
        
        # compressing the images
        centerx = int((M_temp_fourier_all.shape[0])/2)
        centery = int((M_temp_fourier_all.shape[1])/2)
        [offsetx, offsety] = self.offset
        #offsetx = 24 #80
        #offsety = 24 #96
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        F_temp_fourier_low = torch.fft.fftshift(F_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        
        M_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(M_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        F_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(F_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
            
        # normalize images have data range [0,1]
        M_temp_low_spatial_low = normalize(M_temp_low_spatial_low)
        F_temp_low_spatial_low = normalize(F_temp_low_spatial_low)

        out_1, out_2 = self.model(M_temp_low_spatial_low, F_temp_low_spatial_low)

        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))

        padx = int((M_temp.shape[0]-out_ifft1.shape[0])/2) #calculate padding for x axis
        pady = int((M_temp.shape[1]-out_ifft1.shape[1])/2) #calculate padding for x axis
        padxy = (pady, pady, padx, padx) # adaptive padding
        out_ifft1 = F.pad(out_ifft1, padxy, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, padxy, "constant", 0)
                
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        f_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)

        if self.diffeo == 1:
            Df_xy = self.diff_transform(f_xy)
        else:
            Df_xy = f_xy
         
        return Df_xy

class Cascade_dense(nn.Module):
    """ Version of 4xFourier-Net+ that gives back a dense displacement field """
    def __init__(self, in_channel, n_classes, start_channel, diffeo, offset):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.offset = offset
        self.diffeo = diffeo

        super(Cascade_dense, self).__init__()
        self.net1 = SYMNet_dense(self.in_channel, self.n_classes, self.start_channel)
        self.net2 = SYMNet_dense(self.in_channel, self.n_classes, self.start_channel)
        self.net3 = SYMNet_dense(self.in_channel, self.n_classes, self.start_channel)
        self.net4 = SYMNet_dense(self.in_channel, self.n_classes, self.start_channel)
        self.warp = SpatialTransform()
        if self.diffeo == 1:
            self.diff_transform = DiffeomorphicTransform(time_step=7).cuda()
        
    def forward(self, Moving, Fixed):
        M_temp = Moving.squeeze().squeeze()
        F_temp = Fixed.squeeze().squeeze()
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        F_temp_fourier_all = torch.fft.fftn(F_temp)
        
        # compressing the images
        centerx = int((M_temp_fourier_all.shape[0])/2)
        centery = int((M_temp_fourier_all.shape[1])/2)
        [offsetx, offsety] = self.offset
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        F_temp_fourier_low = torch.fft.fftshift(F_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        
        M_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(M_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        F_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(F_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
            
        # normalize images have data range [0,1]
        M_temp_low_spatial_low = normalize(M_temp_low_spatial_low)
        F_temp_low_spatial_low = normalize(F_temp_low_spatial_low)

        # input into the network
        out_1, out_2 = self.net1(M_temp_low_spatial_low, F_temp_low_spatial_low)

        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        padx = int((M_temp.shape[0]-out_ifft1.shape[0])/2) #calculate padding for x axis
        pady = int((M_temp.shape[1]-out_ifft1.shape[1])/2) #calculate padding for x axis
        padxy = (pady, pady, padx, padx) # adaptive padding
        out_ifft1 = F.pad(out_ifft1, padxy, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, padxy, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        fxy_1 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
         
        __, Moving = self.warp(Moving, fxy_1.permute(0, 2, 3, 1))
                
        M_temp = Moving.squeeze().squeeze()
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        M_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(M_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        
        M_temp_low_spatial_low = normalize(M_temp_low_spatial_low)
        
        out_1, out_2 = self.net2(M_temp_low_spatial_low, F_temp_low_spatial_low)
        
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        padx = int((M_temp.shape[0]-out_ifft1.shape[0])/2) #calculate padding for x axis
        pady = int((M_temp.shape[1]-out_ifft1.shape[1])/2) #calculate padding for x axis
        padxy = (pady, pady, padx, padx) # adaptive padding
        out_ifft1 = F.pad(out_ifft1, padxy, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, padxy, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        fxy_2 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
                  
        __, fxy_2_ = self.warp(fxy_1, fxy_2.permute(0, 2, 3, 1))
        
        fxy_2_ = fxy_2_ + fxy_2
                
        __, Moving = self.warp(Moving, fxy_2_.permute(0, 2, 3, 1))
                
        M_temp = Moving.squeeze().squeeze()
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        M_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(M_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        
        M_temp_low_spatial_low = normalize(M_temp_low_spatial_low)
        
        out_1, out_2 = self.net3(M_temp_low_spatial_low, F_temp_low_spatial_low)
        
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        padx = int((M_temp.shape[0]-out_ifft1.shape[0])/2) #calculate padding for x axis
        pady = int((M_temp.shape[1]-out_ifft1.shape[1])/2) #calculate padding for x axis
        padxy = (pady, pady, padx, padx) # adaptive padding
        out_ifft1 = F.pad(out_ifft1, padxy, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, padxy, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        fxy_3 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
           
        __, fxy_3_ = self.warp(fxy_2_, fxy_3.permute(0, 2, 3, 1))
        fxy_3_ = fxy_3_ + fxy_3
        
        __, Moving = self.warp(Moving, fxy_3_.permute(0, 2, 3, 1))
        
        M_temp = Moving.squeeze().squeeze()
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        M_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(M_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        
        M_temp_low_spatial_low = normalize(M_temp_low_spatial_low)
        
        out_1, out_2 = self.net4(M_temp_low_spatial_low, F_temp_low_spatial_low)
                
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        padx = int((M_temp.shape[0]-out_ifft1.shape[0])/2) #calculate padding for x axis
        pady = int((M_temp.shape[1]-out_ifft1.shape[1])/2) #calculate padding for x axis
        padxy = (pady, pady, padx, padx) # adaptive padding
        out_ifft1 = F.pad(out_ifft1, padxy, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, padxy, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        fxy_4 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
           
        __, fxy_4_ = self.warp(fxy_3_, fxy_4.permute(0, 2, 3, 1))
        fxy_4_ = fxy_4_ + fxy_4

        if self.diffeo == 1:
            Df_xy = self.diff_transform(fxy_4_)
        else:
            Df_xy = fxy_4_
        
        return Df_xy

class Fourier_Net_kSpace(nn.Module):
    def __init__(self, in_shape, diffeo):
        self.in_shape = in_shape
        self.diffeo = diffeo

        super(Fourier_Net_kSpace, self).__init__()
        self.model = LAPNet_PyTorch_2D(self.in_shape)
        #self.model = SYMNet(self.in_channel, self.n_classes, self.start_channel) #Net_1_4
        if self.diffeo == 1:
            self.diff_transform = DiffeomorphicTransform(time_step=7).cuda()
        
    def forward(self, Moving, Fixed):
        # get sizes of the image
        x = self.in_shape[2]
        y = self.in_shape[3]

        # create placeholders for the displacements
        disp_1 = torch.zeros([x,y])
        disp_2 = torch.zeros([x,y])

        for i in range(216*256):
            # select correct crops
            M_temp_fourier_crop = Moving[i,:,:,:]
            F_temp_fourier_crop = Fixed[i,:,:,:]

            # get image space displacement of center pixel from LAPNet
            out_1, out_2 = self.model(M_temp_fourier_crop.cuda(), F_temp_fourier_crop.cuda())
            out_1 = out_1.squeeze().squeeze()
            out_2 = out_2.squeeze().squeeze()

            # add to the displacements
            disp_1[int(i/256),i%256] = out_1
            disp_2[int(i/256),i%256] = out_2

        """
        # calculate padding for x and y axis
        padx = int((Moving.shape[2]-out_1.shape[0])/2) 
        pady = int((Moving.shape[3]-out_1.shape[1])/2) 
        padxy = (pady, pady, padx, padx) 

        # pad band-limited displacement
        out_1 = F.pad(out_1, padxy, "constant", 0)
        out_2 = F.pad(out_2, padxy, "constant", 0)
        
        # iDFT to get the displacement in the image domain 
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_2)))

        # concatenate displacements
        f_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
        """
        # concatenate displacements
        f_xy = torch.cat([disp_1.unsqueeze(0).unsqueeze(0), disp_2.unsqueeze(0).unsqueeze(0)], dim = 1)

        if self.diffeo == 1:
            Df_xy = self.diff_transform(f_xy)
        else:
            Df_xy = f_xy
         
        return Df_xy

class Fourier_Net_plus_kSpace(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, diffeo, offset):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.offset = offset
        self.diffeo = diffeo

        super(Fourier_Net_plus_kSpace, self).__init__()
        self.model = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        if self.diffeo == 1:
            self.diff_transform = DiffeomorphicTransform(time_step=7).cuda()
        
    def forward(self, Moving, Fixed):
        M_temp = Moving.squeeze().squeeze()
        F_temp = Fixed.squeeze().squeeze()
        
        # FFT to get to k-space domain
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        F_temp_fourier_all = torch.fft.fftn(F_temp)
        
        # center-crop the k-space
        centerx = int((M_temp_fourier_all.shape[0])/2)
        centery = int((M_temp_fourier_all.shape[1])/2)
        [offsetx, offsety] = self.offset
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]
        F_temp_fourier_low = torch.fft.fftshift(F_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]
        
        # concatenate real and imaginary parts of the k-space into two channels
        M_temp_fourier_low_concat = torch.cat([torch.view_as_real(M_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,0],torch.view_as_real(M_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,1]],1)
        F_temp_fourier_low_concat = torch.cat([torch.view_as_real(F_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,0],torch.view_as_real(F_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,1]],1)
        
        # get band-limited displacement
        out_1, out_2 = self.model(M_temp_fourier_low_concat, F_temp_fourier_low_concat)
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        
        # pad band-limited displacement
        padx = int((Moving.shape[2]-out_1.shape[0])/2) 
        pady = int((Moving.shape[3]-out_1.shape[1])/2) 
        padxy = (pady, pady, padx, padx) 
        out_1 = F.pad(out_1, padxy, "constant", 0)
        out_2 = F.pad(out_2, padxy, "constant", 0)
        
        # iDFT to get the displacement in the image domain 
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_2)))
        f_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)

        if self.diffeo == 1:
            Df_xy = self.diff_transform(f_xy)
        else:
            Df_xy = f_xy
         
        return Df_xy

class Cascade_kSpace(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, diffeo, offset):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.offset = offset
        self.diffeo = diffeo

        super(Cascade_kSpace, self).__init__()
        self.net1 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.net2 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.net3 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.net4 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.warp = SpatialTransform()
        if self.diffeo == 1:
            self.diff_transform = DiffeomorphicTransform(time_step=7).cuda()
        
    def forward(self, Moving, Fixed):
        M_temp = Moving.squeeze().squeeze()
        F_temp = Fixed.squeeze().squeeze()
        
        # FFT to get to k-space domain
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        F_temp_fourier_all = torch.fft.fftn(F_temp)
        
        # center-crop the k-space
        centerx = int((M_temp_fourier_all.shape[0])/2)
        centery = int((M_temp_fourier_all.shape[1])/2)
        [offsetx, offsety] = self.offset
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]
        F_temp_fourier_low = torch.fft.fftshift(F_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]
        
        # concatenate real and imaginary parts of the k-space into two channels
        M_temp_fourier_low_concat = torch.cat([torch.view_as_real(M_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,0],torch.view_as_real(M_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,1]],1)
        F_temp_fourier_low_concat = torch.cat([torch.view_as_real(F_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,0],torch.view_as_real(F_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,1]],1)
        
        # get band-limited displacement
        out_1, out_2 = self.net1(M_temp_fourier_low_concat, F_temp_fourier_low_concat)
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        
        # pad band-limited displacement
        padx = int((Moving.shape[2]-out_1.shape[0])/2) 
        pady = int((Moving.shape[3]-out_1.shape[1])/2) 
        padxy = (pady, pady, padx, padx) 
        out_1 = F.pad(out_1, padxy, "constant", 0)
        out_2 = F.pad(out_2, padxy, "constant", 0)
        
        # iDFT to get the displacement in the image domain 
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_2)))
        fxy_1 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
         
        __, Moving = self.warp(Moving, fxy_1.permute(0, 2, 3, 1))
                
        M_temp = Moving.squeeze().squeeze()        
        # FFT to get to k-space domain
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        # center-crop the k-space
        centerx = int((M_temp_fourier_all.shape[0])/2)
        centery = int((M_temp_fourier_all.shape[1])/2)
        [offsetx, offsety] = self.offset
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]
        
        # concatenate real and imaginary parts of the k-space into two channels
        M_temp_fourier_low_concat = torch.cat([torch.view_as_real(M_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,0],torch.view_as_real(M_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,1]],1)
        
        # get band-limited displacement
        out_1, out_2 = self.net2(M_temp_fourier_low_concat, F_temp_fourier_low_concat)
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        
        # pad band-limited displacement
        padx = int((Moving.shape[2]-out_1.shape[0])/2) 
        pady = int((Moving.shape[3]-out_1.shape[1])/2) 
        padxy = (pady, pady, padx, padx) 
        out_1 = F.pad(out_1, padxy, "constant", 0)
        out_2 = F.pad(out_2, padxy, "constant", 0)
        
        # iDFT to get the displacement in the image domain 
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_2)))
        fxy_2 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
                  
        __, fxy_2_ = self.warp(fxy_1, fxy_2.permute(0, 2, 3, 1))
        
        fxy_2_ = fxy_2_ + fxy_2
                
        __, Moving = self.warp(Moving, fxy_2_.permute(0, 2, 3, 1))
                
        M_temp = Moving.squeeze().squeeze()        
        # FFT to get to k-space domain
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        # center-crop the k-space
        centerx = int((M_temp_fourier_all.shape[0])/2)
        centery = int((M_temp_fourier_all.shape[1])/2)
        [offsetx, offsety] = self.offset
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]
        
        # concatenate real and imaginary parts of the k-space into two channels
        M_temp_fourier_low_concat = torch.cat([torch.view_as_real(M_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,0],torch.view_as_real(M_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,1]],1)
        
        # get band-limited displacement
        out_1, out_2 = self.net3(M_temp_fourier_low_concat, F_temp_fourier_low_concat)
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        
        # pad band-limited displacement
        padx = int((Moving.shape[2]-out_1.shape[0])/2) 
        pady = int((Moving.shape[3]-out_1.shape[1])/2) 
        padxy = (pady, pady, padx, padx) 
        out_1 = F.pad(out_1, padxy, "constant", 0)
        out_2 = F.pad(out_2, padxy, "constant", 0)
        
        # iDFT to get the displacement in the image domain 
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_2)))
        fxy_3 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
           
        __, fxy_3_ = self.warp(fxy_2_, fxy_3.permute(0, 2, 3, 1))
        fxy_3_ = fxy_3_ + fxy_3
        
        __, Moving = self.warp(Moving, fxy_3_.permute(0, 2, 3, 1))
        
        M_temp = Moving.squeeze().squeeze()        
        # FFT to get to k-space domain
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        # center-crop the k-space
        centerx = int((M_temp_fourier_all.shape[0])/2)
        centery = int((M_temp_fourier_all.shape[1])/2)
        [offsetx, offsety] = self.offset
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]
        
        # concatenate real and imaginary parts of the k-space into two channels
        M_temp_fourier_low_concat = torch.cat([torch.view_as_real(M_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,0],torch.view_as_real(M_temp_fourier_low).unsqueeze(0).unsqueeze(0)[:,:,:,:,1]],1)
        
        # get band-limited displacement
        out_1, out_2 = self.net4(M_temp_fourier_low_concat, F_temp_fourier_low_concat)
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        
        # pad band-limited displacement
        padx = int((Moving.shape[2]-out_1.shape[0])/2) 
        pady = int((Moving.shape[3]-out_1.shape[1])/2) 
        padxy = (pady, pady, padx, padx) 
        out_1 = F.pad(out_1, padxy, "constant", 0)
        out_2 = F.pad(out_2, padxy, "constant", 0)
        
        # iDFT to get the displacement in the image domain 
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_2)))
        fxy_4 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
           
        __, fxy_4_ = self.warp(fxy_3_, fxy_4.permute(0, 2, 3, 1))
        fxy_4_ = fxy_4_ + fxy_4

        if self.diffeo == 1:
            Df_xy = self.diff_transform(fxy_4_)
        else:
            Df_xy = fxy_4_
        
        return Df_xy

class Fourier_Net(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, diffeo):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.diffeo = diffeo

        super(Fourier_Net, self).__init__()
        self.model = SYMNet(self.in_channel, self.n_classes, self.start_channel) #Net_1_4
        if self.diffeo == 1:
            self.diff_transform = DiffeomorphicTransform(time_step=7).cuda()
        
    def forward(self, Moving, Fixed):
        out_1, out_2, features_disp = self.model(Moving, Fixed)#.squeeze().squeeze()

        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        
        padx = int((Moving.shape[2]-out_ifft1.shape[0])/2) #calculate padding for x axis
        pady = int((Moving.shape[3]-out_ifft1.shape[1])/2) #calculate padding for x axis
        padxy = (pady, pady, padx, padx) # adaptive padding
        out_ifft1 = F.pad(out_ifft1, padxy, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, padxy, "constant", 0)
        
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        f_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)

        if self.diffeo == 1:
            Df_xy = self.diff_transform(f_xy)
        else:
            Df_xy = f_xy
         
        return Df_xy, features_disp

class Fourier_Net_plus(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, diffeo, offset):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.offset = offset
        self.diffeo = diffeo

        super(Fourier_Net_plus, self).__init__()
        self.model = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        if self.diffeo == 1:
            self.diff_transform = DiffeomorphicTransform(time_step=7).cuda()
        
    def forward(self, Moving, Fixed):
        M_temp = Moving.squeeze().squeeze()
        F_temp = Fixed.squeeze().squeeze()
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        F_temp_fourier_all = torch.fft.fftn(F_temp)
        
        # compressing the images
        centerx = int((M_temp_fourier_all.shape[0])/2)
        centery = int((M_temp_fourier_all.shape[1])/2)
        [offsetx, offsety] = self.offset
        #offsetx = 24 #80
        #offsety = 24 #96
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        F_temp_fourier_low = torch.fft.fftshift(F_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        
        M_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(M_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        F_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(F_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
            
        # normalize images have data range [0,1]
        M_temp_low_spatial_low = normalize(M_temp_low_spatial_low)
        F_temp_low_spatial_low = normalize(F_temp_low_spatial_low)

        out_1, out_2 = self.model(M_temp_low_spatial_low, F_temp_low_spatial_low)

        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        
        padx = int((Moving.shape[2]-out_ifft1.shape[0])/2) #calculate padding for x axis
        pady = int((Moving.shape[3]-out_ifft1.shape[1])/2) #calculate padding for x axis
        padxy = (pady, pady, padx, padx) # adaptive padding
        out_ifft1 = F.pad(out_ifft1, padxy, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, padxy, "constant", 0)
        
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        f_xy = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)

        if self.diffeo == 1:
            Df_xy = self.diff_transform(f_xy)
        else:
            Df_xy = f_xy
         
        return Df_xy

class Cascade(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, diffeo, offset):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel
        self.offset = offset
        self.diffeo = diffeo

        super(Cascade, self).__init__()
        self.net1 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.net2 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.net3 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.net4 = Net_1_4(self.in_channel, self.n_classes, self.start_channel)
        self.warp = SpatialTransform()
        if self.diffeo == 1:
            self.diff_transform = DiffeomorphicTransform(time_step=7).cuda()
        
    def forward(self, Moving, Fixed):
        M_temp = Moving.squeeze().squeeze()
        F_temp = Fixed.squeeze().squeeze()
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        F_temp_fourier_all = torch.fft.fftn(F_temp)
        
        # compressing the images
        centerx = int((M_temp_fourier_all.shape[0])/2)
        centery = int((M_temp_fourier_all.shape[1])/2)
        [offsetx, offsety] = self.offset
        #offsetx = 24 #80
        #offsety = 24 #96
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        F_temp_fourier_low = torch.fft.fftshift(F_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        
        M_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(M_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        F_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(F_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
            
        # normalize images have data range [0,1]
        M_temp_low_spatial_low = normalize(M_temp_low_spatial_low)
        F_temp_low_spatial_low = normalize(F_temp_low_spatial_low)

        """
        plt.subplots(figsize=(4, 2))
        plt.axis('off')

        plt.subplot(4,2,1)
        plt.imshow(M_temp_low_spatial_low.data.cpu().numpy()[0, 0, ...], cmap='gray', vmin=0, vmax = 1)
        plt.title('M')
        plt.axis('off')

        plt.subplot(4,2,2)
        plt.imshow(F_temp_low_spatial_low.data.cpu().numpy()[0, 0, ...], cmap='gray', vmin=0, vmax = 1)
        plt.title('F')
        plt.axis('off')        
        """

        # input into the network
        out_1, out_2 = self.net1(M_temp_low_spatial_low, F_temp_low_spatial_low)

        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        #padxy = (72, 72, 60, 60) # old padding for image size (224,192)
        #padxy = (232, 232, 103, 103) # new padding for image size (246,512)
        padx = int((M_temp.shape[0]-out_ifft1.shape[0])/2) #calculate padding for x axis
        pady = int((M_temp.shape[1]-out_ifft1.shape[1])/2) #calculate padding for x axis
        padxy = (pady, pady, padx, padx) # adaptive padding
        out_ifft1 = F.pad(out_ifft1, padxy, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, padxy, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        fxy_1 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
         
        __, Moving = self.warp(Moving, fxy_1.permute(0, 2, 3, 1))
                
        M_temp = Moving.squeeze().squeeze()
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        M_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(M_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        
        M_temp_low_spatial_low = normalize(M_temp_low_spatial_low)
        """
        plt.subplot(4,2,3)
        plt.imshow(M_temp_low_spatial_low.data.cpu().numpy()[0, 0, ...], cmap='gray', vmin=0, vmax = 1)
        plt.axis('off')

        plt.subplot(4,2,4)
        plt.imshow(F_temp_low_spatial_low.data.cpu().numpy()[0, 0, ...], cmap='gray', vmin=0, vmax = 1)
        plt.axis('off')
        """
        out_1, out_2 = self.net2(M_temp_low_spatial_low, F_temp_low_spatial_low)
        
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        padx = int((M_temp.shape[0]-out_ifft1.shape[0])/2) #calculate padding for x axis
        pady = int((M_temp.shape[1]-out_ifft1.shape[1])/2) #calculate padding for x axis
        padxy = (pady, pady, padx, padx) # adaptive padding
        out_ifft1 = F.pad(out_ifft1, padxy, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, padxy, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        fxy_2 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
                  
        __, fxy_2_ = self.warp(fxy_1, fxy_2.permute(0, 2, 3, 1))
        
        fxy_2_ = fxy_2_ + fxy_2
                
        __, Moving = self.warp(Moving, fxy_2_.permute(0, 2, 3, 1))
                
        M_temp = Moving.squeeze().squeeze()
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        M_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(M_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        
        M_temp_low_spatial_low = normalize(M_temp_low_spatial_low)
        """
        plt.subplot(4,2,5)
        plt.imshow(M_temp_low_spatial_low.data.cpu().numpy()[0, 0, ...], cmap='gray', vmin=0, vmax = 1)
        plt.axis('off')

        plt.subplot(4,2,6)
        plt.imshow(F_temp_low_spatial_low.data.cpu().numpy()[0, 0, ...], cmap='gray', vmin=0, vmax = 1)
        plt.axis('off')
        """
        out_1, out_2 = self.net3(M_temp_low_spatial_low, F_temp_low_spatial_low)
        
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        padx = int((M_temp.shape[0]-out_ifft1.shape[0])/2) #calculate padding for x axis
        pady = int((M_temp.shape[1]-out_ifft1.shape[1])/2) #calculate padding for x axis
        padxy = (pady, pady, padx, padx) # adaptive padding
        out_ifft1 = F.pad(out_ifft1, padxy, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, padxy, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        fxy_3 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
           
        __, fxy_3_ = self.warp(fxy_2_, fxy_3.permute(0, 2, 3, 1))
        fxy_3_ = fxy_3_ + fxy_3
        
        __, Moving = self.warp(Moving, fxy_3_.permute(0, 2, 3, 1))
        
        M_temp = Moving.squeeze().squeeze()
        M_temp_fourier_all = torch.fft.fftn(M_temp)
        
        M_temp_fourier_low = torch.fft.fftshift(M_temp_fourier_all)[(centerx-offsetx):(centerx+offsetx),(centery-offsety):(centery+offsety)]#[40:120,48:144]
        M_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(M_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        
        M_temp_low_spatial_low = normalize(M_temp_low_spatial_low)
        """
        plt.subplot(4,2,7)
        plt.imshow(M_temp_low_spatial_low.data.cpu().numpy()[0, 0, ...], cmap='gray', vmin=0, vmax = 1)
        plt.axis('off')

        plt.subplot(4,2,8)
        plt.imshow(F_temp_low_spatial_low.data.cpu().numpy()[0, 0, ...], cmap='gray', vmin=0, vmax = 1)
        plt.axis('off')
        
        plt.savefig('CompressedImages',bbox_inches='tight')
        plt.close()
        """
        out_1, out_2 = self.net4(M_temp_low_spatial_low, F_temp_low_spatial_low)
                
        out_1 = out_1.squeeze().squeeze()
        out_2 = out_2.squeeze().squeeze()
        out_ifft1 = torch.fft.fftshift(torch.fft.fftn(out_1))
        out_ifft2 = torch.fft.fftshift(torch.fft.fftn(out_2))
        padx = int((M_temp.shape[0]-out_ifft1.shape[0])/2) #calculate padding for x axis
        pady = int((M_temp.shape[1]-out_ifft1.shape[1])/2) #calculate padding for x axis
        padxy = (pady, pady, padx, padx) # adaptive padding
        out_ifft1 = F.pad(out_ifft1, padxy, "constant", 0)
        out_ifft2 = F.pad(out_ifft2, padxy, "constant", 0)
        disp_mf_1 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft1)))
        disp_mf_2 = torch.real(torch.fft.ifftn(torch.fft.ifftshift(out_ifft2)))
        fxy_4 = torch.cat([disp_mf_1.unsqueeze(0).unsqueeze(0), disp_mf_2.unsqueeze(0).unsqueeze(0)], dim = 1)
           
        __, fxy_4_ = self.warp(fxy_3_, fxy_4.permute(0, 2, 3, 1))
        fxy_4_ = fxy_4_ + fxy_4

        if self.diffeo == 1:
            Df_xy = self.diff_transform(fxy_4_)
        else:
            Df_xy = fxy_4_
        
        return Df_xy

class UNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(UNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc10 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        self.up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        e4 = self.ec8(e3)
        e4 = self.ec9(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)

        d3 = torch.cat((self.up4(d2), e0), 1)
        d3 = self.dc7(d3)
        d3 = self.dc8(d3)

        f_xy = self.dc9(d3)

        return f_xy


class Net_1_4(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(Net_1_4, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 4, bias=bias_opt)
        
        self.dc1 = self.encoder(self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc2 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.dc3 = self.encoder(self.start_channel * 2, self.start_channel * 1, kernel_size=3,
                                stride=1, bias=bias_opt)
        self.dc9 = self.outputs(self.start_channel * 1, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.up1 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up2 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))#,
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)

        d0 = self.up1(e3)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = self.up2(d0)

        d1 = self.dc3(d1)
        f_xy = self.dc9(d1)

        return f_xy[:,0:1,:,:], f_xy[:,1:2,:,:]

class SYMNet(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel):
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        bias_opt = True

        super(SYMNet, self).__init__()
        self.eninput = self.encoder(self.in_channel, self.start_channel, bias=bias_opt)
        self.ec1 = self.encoder(self.start_channel, self.start_channel, bias=bias_opt)
        self.ec2 = self.encoder(self.start_channel, self.start_channel * 2, stride=2, bias=bias_opt)
        self.ec3 = self.encoder(self.start_channel * 2, self.start_channel * 2, bias=bias_opt)
        self.ec4 = self.encoder(self.start_channel * 2, self.start_channel * 4, stride=2, bias=bias_opt)
        self.ec5 = self.encoder(self.start_channel * 4, self.start_channel * 4, bias=bias_opt)
        self.ec6 = self.encoder(self.start_channel * 4, self.start_channel * 8, stride=2, bias=bias_opt)
        self.ec7 = self.encoder(self.start_channel * 8, self.start_channel * 8, bias=bias_opt)
        self.ec8 = self.encoder(self.start_channel * 8, self.start_channel * 16, stride=2, bias=bias_opt)
        self.ec9 = self.encoder(self.start_channel * 16, self.start_channel * 8, bias=bias_opt)

        self.r_dc1 = self.encoder(self.start_channel * 8 + self.start_channel * 8, self.start_channel * 8, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc2 = self.encoder(self.start_channel * 8, self.start_channel * 4, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc3 = self.encoder(self.start_channel * 4 + self.start_channel * 4, self.start_channel * 4, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc4 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc5 = self.encoder(self.start_channel * 2 + self.start_channel * 2, self.start_channel * 4, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc6 = self.encoder(self.start_channel * 4, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.r_dc7 = self.encoder(self.start_channel * 2 + self.start_channel * 1, self.start_channel * 2, kernel_size=3,
                                  stride=1, bias=bias_opt)
        self.r_dc8 = self.encoder(self.start_channel * 2, self.start_channel * 2, kernel_size=3, stride=1, bias=bias_opt)
        self.rr_dc9 = self.outputs(self.start_channel * 2, self.n_classes, kernel_size=3, stride=1, padding=1, bias=False)

        # needed for removed layer for ACDC dataset    
        self.r_new = self.encoder(self.start_channel * 12, self.start_channel * 8, kernel_size=3,
                                  stride=1, bias=bias_opt)

        self.r_up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.r_up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.r_up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.r_up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        

    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))#,
        return layer

    def forward(self, x, y):
        x_in = torch.cat((x, y), 1)
        e0 = self.eninput(x_in)
        e0 = self.ec1(e0)

        e1 = self.ec2(e0)
        e1 = self.ec3(e1)

        e2 = self.ec4(e1)
        e2 = self.ec5(e2)

        e3 = self.ec6(e2)
        e3 = self.ec7(e3)
        """
        # remove layer as it leads to problems
        e4 = self.ec8(e3)
        e4 = self.ec9(e4)
        r_d0 = torch.cat((self.r_up1(e4), e3), 1)
        """
        r_d0 = torch.cat((self.r_up1(e3), e2), 1)

        #r_d0 = self.r_dc1(r_d0)
        #r_d0 = self.r_dc2(r_d0)
        
        r_d0 = self.r_new(r_d0)
        r_d0 = self.r_dc2(r_d0)
        r_d0 = self.r_dc4(r_d0)

        #r_d1 = torch.cat((self.r_up2(r_d0), e2), 1)
        r_d1 = torch.cat((self.r_up3(r_d0), e1), 1)

        #r_d1 = self.r_dc3(r_d1)
        r_d1 = self.r_dc4(r_d1)

        f_r = self.rr_dc9(r_d1) * 64   
        
        return f_r[:,0:1,:,:], f_r[:,1:2,:,:], r_d1

class LAPNet_PyTorch_2D(nn.Module): 
    def __init__(self, inshape):
        self.input_shape = (inshape[2], inshape[3], 4)
        
        super(LAPNet_PyTorch_2D, self).__init__()
        # down-scaling layers
        self.down1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.down2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.down3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.down4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.down5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1, bias=False)
        self.down6 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        
        # output layers convert displacement to image space 
        self.output = nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=(1, 1))
        self.ReLU    = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.MaxPool = nn.MaxPool2d(kernel_size=5)

        # up-scaling layers convert displacements back to original size
        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,  output_padding=1, bias=False)
        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1,  output_padding=1, bias=False)
        self.up5 = nn.ConvTranspose2d(in_channels=64, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x, y):
        x = self.ReLU(self.down1(torch.cat((x, y), 1)))
        x = self.ReLU(self.down2(x))
        x = self.ReLU(self.down3(x))
        x = self.ReLU(self.down4(x))
        x = self.ReLU(self.down5(x))
        x = self.ReLU(self.down6(x))
        x = self.MaxPool(x)
        x = self.output(x)

        return x[:,0:1,:,:], x[:,1:2,:,:]

class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        h2, w2 = mov_image.shape[-2:]
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_h = flow[:,:,:,0]
        flow_w = flow[:,:,:,1]
        
        # Remove Channel Dimension
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h), 3)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True,padding_mode="border")
        
        return sample_grid, warped

class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
        h2, w2, = flow.shape[-2:]
        grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(flow.device).float()
        grid_w = grid_w.to(flow.device).float()
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)

        for i in range(self.time_step):
            flow_h = flow[:,0,:,:]
            flow_w = flow[:,1,:,:]
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)
            
            deformation = torch.stack((disp_w,disp_h), dim=3)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear',padding_mode="border", align_corners=True)
        
        return flow


class CompositionTransform(nn.Module):
    def __init__(self):
        super(CompositionTransform, self).__init__()

    def forward(self, flow_1, flow_2, sample_grid, range_flow):
        size_tensor = sample_grid.size()
        grid = sample_grid + (flow_1.permute(0,2,3,4,1) * range_flow)
        grid[:, :, :, :, 0] = (grid[:, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
        grid[:, :, :, :, 1] = (grid[:, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
        grid[:, :, :, :, 2] = (grid[:, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
        compos_flow = F.grid_sample(flow_2, grid, mode='bilinear', align_corners = True,padding_mode="border") + flow_1
        return compos_flow


def smoothloss(y_pred):
    #print('smoothloss y_pred.shape    ',y_pred.shape)
    #[N,3,D,H,W]
    h2, w2 = y_pred.shape[-2:]
    # dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :]) / 2 * d2
    dx = torch.abs(y_pred[:,:, 1:, :] - y_pred[:, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:,:, :, 1:] - y_pred[:, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx) + torch.mean(dz*dz))/2.0


def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


def magnitude_loss(flow_1, flow_2):
    num_ele = torch.numel(flow_1)
    flow_1_mag = torch.sum(torch.abs(flow_1))
    flow_2_mag = torch.sum(torch.abs(flow_2))

    diff = (torch.abs(flow_1_mag - flow_2_mag))/num_ele

    return diff

"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""
class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=5, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv2d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class SAD:
    """
    Mean absolute error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

class RelativeL2Loss(nn.Module):
    def __init__(self, sigma=1.0, reg_weight=0):
        super(RelativeL2Loss, self).__init__()
        self.epsilon = 1e-5
        self.sigma = sigma
        self.reg_weight = reg_weight
    
    def forward(self, input, target):
        if input.dtype == torch.float:
            input = torch.view_as_complex(input) 
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        target = target / target.abs().max()
        input = input / input.abs().max()
        loss = 0
        for x, y in zip([input.real, input.imag], [target.real, target.imag]):
            magnitude = x.clone().detach()**2
            scaler = magnitude+self.epsilon
            squared_loss = (x - y)**2
            loss_add = (squared_loss / scaler).mean() 
            if not math.isnan(loss_add):
                loss = loss + loss_add
    
        return loss

################
## Voxelmorph ##
################


def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'. This is used to assist
    model loading - see LoadableModel.
    """

    argspec = inspect.getfullargspec(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}

        # first save the default values
        if argspec.defaults:
            for attr, val in zip(reversed(argspec.args), reversed(argspec.defaults)):
                self.config[attr] = val

        # next handle positional args
        for attr, val in zip(argspec.args[1:], args):
            self.config[attr] = val

        # lastly handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val

        return func(self, *args, **kwargs)
    return wrapper

class LoadableModel(nn.Module):
    """
    Base class for easy pytorch model loading without having to manually
    specify the architecture configuration at load time.

    We can cache the arguments used to the construct the initial network, so that
    we can construct the exact same network when loading from file. The arguments
    provided to __init__ are automatically saved into the object (in self.config)
    if the __init__ method is decorated with the @store_config_args utility.
    """

    # this constructor just functions as a check to make sure that every
    # LoadableModel subclass has provided an internal config parameter
    # either manually or via store_config_args
    def __init__(self, *args, **kwargs):
        if not hasattr(self, 'config'):
            raise RuntimeError('models that inherit from LoadableModel must decorate the '
                               'constructor with @store_config_args')
        super().__init__(*args, **kwargs)

    def save(self, path):
        """
        Saves the model configuration and weights to a pytorch file.
        """
        # don't save the transformer_grid buffers - see SpatialTransformer doc for more info
        sd = self.state_dict().copy()
        grid_buffers = [key for key in sd.keys() if key.endswith('.grid')]
        for key in grid_buffers:
            sd.pop(key)
        torch.save({'config': self.config, 'model_state': sd}, path)

    @classmethod
    def load(cls, path, device):
        """
        Load a python model configuration and weights.
        """
        checkpoint = torch.load(path, map_location=torch.device(device))
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state'], strict=False)
        return model


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. 
                Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if not unet_half_res and int_steps > 0 and int_downsize > 1:
            self.resize = ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, source, target, registration=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)
        y_target = self.transformer(target, neg_flow) if self.bidir else None

        """
        # return non-integrated flow field if training
        if not registration:
            return (y_source, y_target, preint_flow) if self.bidir else (y_source, preint_flow)
        else:
            return y_source, pos_flow
        """
        return y_source, pos_flow #flow_field


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out
    
class vxm_NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class vxm_MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class vxm_Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class vxm_Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
        

################
#### LAPNet ####
################

# Create the 2D model
def buildLAPNet_model_2D(inshape): #crop_size=33
    #input_shape = (crop_size, crop_size, 4)
    input_shape = (inshape[2], inshape[3], 4)
    inputs = Input(shape=input_shape)
    model = keras.Sequential()
    initializer = VarianceScaling(scale=2.0)
    model.add(inputs)
    model.add(Conv2D(filters=64,
                     kernel_size=3,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv1"))
    # model.add(Activation(leaky_relu, name='act1'))
    model.add(LeakyReLU(alpha=0.1, name='act1'))
    model.add(Conv2D(filters=128,
                     kernel_size=3,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv2"))
    # model.add(Activation(leaky_relu, name='act2'))
    model.add(LeakyReLU(alpha=0.1, name='act2'))
    model.add(Conv2D(filters=256,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv2_1"))
    # model.add(Activation(leaky_relu, name='act2_1'))
    model.add(LeakyReLU(alpha=0.1, name='act2_1'))
    model.add(Conv2D(filters=512,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv3_1"))
    # model.add(Activation(leaky_relu, name='act3_1'))
    model.add(LeakyReLU(alpha=0.1, name='act3_1'))
    model.add(Conv2D(filters=1024,
                     kernel_size=3,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv4"))
    # model.add(Activation(leaky_relu, name='act4'))
    model.add(LeakyReLU(alpha=0.1, name='act4'))
    model.add(Conv2D(filters=1024,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv4_1"))
    # model.add(Activation(leaky_relu, name='act4_1'))
    model.add(LeakyReLU(alpha=0.1, name='act4_1'))
    model.add(MaxPooling2D(pool_size=5, name='pool'))
    model.add(Conv2D(2, [1, 1], name="fc2"))
    model.add(Lambda(squeeze_func, name="fc8/squeezed"))
    return model


# Model with descendent kernel sizes
def buildLAPNet_model_2D_old(crop_size=33):
    input_shape = (crop_size, crop_size, 4)
    inputs = Input(shape=input_shape, )
    model = keras.Sequential()
    initializer = VarianceScaling(scale=2.0)
    model.add(inputs, name="input")
    model.add(Conv2D(filters=64,
                     kernel_size=7,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv1"))
    # model.add(Activation(leaky_relu, name='act1'))
    model.add(LeakyReLU(alpha=0.1, name='act1'))
    model.add(Conv2D(filters=128,
                     kernel_size=5,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv2"))
    # model.add(Activation(leaky_relu, name='act2'))
    model.add(LeakyReLU(alpha=0.1, name='act2'))
    model.add(Conv2D(filters=256,
                     kernel_size=5,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv2_1"))
    # model.add(Activation(leaky_relu, name='act2_1'))
    model.add(LeakyReLU(alpha=0.1, name='act2_1'))
    model.add(Conv2D(filters=512,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv3_1"))
    # model.add(Activation(leaky_relu, name='act3_1'))
    model.add(LeakyReLU(alpha=0.1, name='act3_1'))
    model.add(Conv2D(filters=1024,
                     kernel_size=3,
                     strides=2,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv4"))
    # model.add(Activation(leaky_relu, name='act4'))
    model.add(LeakyReLU(alpha=0.1, name='act4'))
    model.add(Conv2D(filters=1024,
                     kernel_size=3,
                     strides=1,
                     padding='same',
                     kernel_regularizer=l2(0.0004),
                     kernel_initializer=initializer,
                     name="conv4_1"))
    # model.add(Activation(leaky_relu, name='act4_1'))
    model.add(LeakyReLU(alpha=0.1, name='act4_1'))
    model.add(MaxPooling2D(pool_size=5, name='pool'))
    model.add(Conv2D(2, [1, 1], name="fc2"))
    model.add(Lambda(squeeze_func, name="fc8/squeezed"))
    return model


# design the 3D model
def buildLAPNet_model_3D():
    initializer = VarianceScaling(scale=2.0)

    input_shape = (33, 33, 4)
    coronal_input = Input(shape=input_shape, name="coronal")
    sagital_input = Input(shape=input_shape, name="sagital")

    coronal_features = Conv2D(filters=64,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv1")(coronal_input)
    coronal_features = LeakyReLU(alpha=0.1, name='c_act1')(coronal_features)
    coronal_features = Conv2D(filters=128,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv2")(coronal_features)
    coronal_features = LeakyReLU(alpha=0.1, name='c_act2')(coronal_features)
    coronal_features = Conv2D(filters=256,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv2_1")(coronal_features)
    coronal_features = Conv2D(filters=512,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv3_1")(coronal_features)
    coronal_features = LeakyReLU(alpha=0.1, name='c_act3_1')(coronal_features)
    coronal_features = Conv2D(filters=1024,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv4")(coronal_features)
    coronal_features = LeakyReLU(alpha=0.1, name='c_act4')(coronal_features)
    coronal_features = Conv2D(filters=1024,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="c_conv4_1")(coronal_features)
    coronal_features = LeakyReLU(alpha=0.1, name='c_act4_1')(coronal_features)
    coronal_features = AveragePooling2D(pool_size=5, name='pool_c')(coronal_features)
    coronal_features = Conv2D(2, [1, 1], name="c_fc3")(coronal_features)

    sagital_features = Conv2D(filters=64,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv1")(sagital_input)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act1')(sagital_features)
    sagital_features = Conv2D(filters=128,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv2")(sagital_features)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act2')(sagital_features)
    sagital_features = Conv2D(filters=256,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv2_1")(sagital_features)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act2_1')(sagital_features)
    sagital_features = Conv2D(filters=512,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv3_1")(sagital_features)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act3_1')(sagital_features)
    sagital_features = Conv2D(filters=1024,
                              kernel_size=3,
                              strides=2,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv4")(sagital_features)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act4')(sagital_features)
    sagital_features = Conv2D(filters=1024,
                              kernel_size=3,
                              strides=1,
                              padding='same',
                              kernel_regularizer=l2(0.0004),
                              kernel_initializer=initializer,
                              name="s_conv4_1")(sagital_features)
    sagital_features = LeakyReLU(alpha=0.1, name='s_act4_1')(sagital_features)
    sagital_features = AveragePooling2D(pool_size=5, name='pool_s')(sagital_features)
    sagital_features = Conv2D(2, [1, 1], name="s_fc3")(sagital_features)

    final_flow_c = Lambda(squeeze_func, name="fc8/squeezed_c")(coronal_features)
    final_flow_s = Lambda(squeeze_func, name="fc8/squeezed_s")(sagital_features)

    final_flow = WeightedSum(name='final_flow_weighted')([final_flow_c, final_flow_s])
    final_flow = Lambda(squeeze_func, name="squeezed_flow")(final_flow)

    model = keras.Model(
        inputs=[coronal_input, sagital_input],
        outputs=[final_flow],
    )

    return model


# Removes dimensions of size 1 from the shape of the input tensor
def squeeze_func(x):
    try:
        return tf.squeeze(x, axis=[1, 2])
    except:
        try:
            return tf.squeeze(x, axis=[1])
        except Exception as e:
            print(e)


# load checkpoints of trained models in tensorflow 1
def load_tf1_LAPNet_cropping_ckpt(
        checkpoint_path_old='/mnt/data/projects/MoCo/LAPNet/UnFlow/log/ex/resp/srx424_drUS_1603/model.ckpt'):
    model = buildLAPNet_model_2D_old()
    model.compile(optimizer=adam_v2(beta_1=0.9, beta_2=0.999, lr=0.0),
                  loss=[modified_EPE],
                  metrics=['accuracy'])
    reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path_old)
    layers_name = ['conv1', 'conv2', 'conv2_1', 'conv3_1', 'conv4', 'conv4_1', 'fc2']
    for i in range(len(layers_name)):
        weights_key = 'flownet_s/' + layers_name[i] + '/weights'
        bias_key = 'flownet_s/' + layers_name[i] + '/biases'
        weights = reader.get_tensor(weights_key)
        biases = reader.get_tensor(bias_key)
        model.get_layer(layers_name[i]).set_weights([weights, biases])  # name the layers
    return model


class WeightedSum(Layer):
    """A custom keras layer to learn a weighted sum of tensors"""

    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape=1):
        self.a = self.add_weight(name='weighted',
                                 shape=(1),
                                 initializer=tf.keras.initializers.Constant(0.5),
                                 dtype='float32',
                                 trainable=True,
                                 constraint=tf.keras.constraints.min_max_norm(max_value=1, min_value=0))
        super(WeightedSum, self).build(input_shape)

    def call(self, model_outputs):
        return tf.stack((self.a * tf.gather(model_outputs[0], [0], axis=1) + (1 - self.a) * tf.gather(model_outputs[1],
                                                                                                      [0], axis=1),
                         tf.gather(model_outputs[0], [1], axis=1),
                         tf.gather(model_outputs[1], [1], axis=1)), axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

def step_decay(epoch):
    initial_lr = 2.5e-04
    drop = 0.5
    epochs_drop = 2
    lr = initial_lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lr


# EPE function modified
def modified_EPE(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    final_loss = tf.reduce_mean(squared_difference)
    return final_loss


def EPE(flows_gt, flows):
    # Given ground truth and estimated flow must be unscaled
    return tf.reduce_mean(tf.norm(flows_gt - flows, ord=2, axis=0))


def EAE(y_true, y_pred):
    final_loss = tf.math.real(tf.math.acos((1 + tf.reduce_sum(tf.math.multiply(y_true, y_pred))) /
                                           (tf.math.sqrt(1 + tf.reduce_sum(tf.math.pow(y_pred, 2))) *
                                            tf.math.sqrt(1 + tf.reduce_sum(tf.math.pow(y_true, 2))))))
    return final_loss


def LAP_loss_function(y_true, y_pred):
    w_1 = 0.8
    w_2 = 0.2
    return tf.add(w_1 * modified_EPE(y_true, y_pred), w_2 * EAE(y_true, y_pred))


def loss1(y_true, y_pred):
    w = 0.5
    y1 = LAP_loss_function(y_true[0], y_pred[0])
    y2 = w * LAP_loss_function(y_true[1], y_pred[1])
    squared_difference = tf.stack([y1, y2], axis=-1)
    return squared_difference


def loss2(y_true, y_pred):
    w = 0.5
    y1 = w * LAP_loss_function(y_true[0], y_pred[0])
    y2 = LAP_loss_function(y_true[1], y_pred[1])
    squared_difference = tf.stack([y1, y2], axis=-1)
    return squared_difference
