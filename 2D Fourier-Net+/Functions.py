"""
Helper functions from https://github.com/zhangjun001/ICNet.

Some functions has been modified.
"""

import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import os
from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt
import itertools
from natsort import natsorted
import glob
import h5py
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc, EquispacedMaskFractionFunc
import torch.nn.functional as F
from skimage.io import imread

def rotate_image(image):
    [w, h] = image.shape
    image_new = np.zeros((h,w))
    for y in range(h):
        for x in range(w):
            image_new[h-y-1, x] = image[x, y]

    return image_new    

def crop_and_pad(img,sizex,sizey,sizez):
    img_new = np.zeros((sizex,sizey,sizez))
    h = np.amin([sizex,img.shape[0]])
    w = np.amin([sizey,img.shape[1]])
    d = np.amin([sizez,img.shape[2]])

    img_new[sizex//2-h//2:sizex//2+h//2,sizey//2-w//2:sizey//2+w//2,sizez//2-d//2:sizez//2+d//2]=img[img.shape[0]//2-h//2:img.shape[0]//2+h//2,img.shape[1]//2-w//2:img.shape[1]//2+w//2,img.shape[2]//2-d//2:img.shape[2]//2+d//2]
    return img_new

def rescale_intensity(image, thres=(0.0, 100.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    image = image.astype(np.float32)
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2

def normalize(image):
    """ expects an image as tensor and normalizes the values between [0,1] """
    return (image - torch.min(image)) / (torch.max(image) - torch.min(image))

def get_image_pairs(set, data_path, subfolder):
    """ get the corresponding paths of image pairs for the dataset """
    image_pairs = []  
    # get folder names for patients
    patients = [os.path.basename(f.path) for f in os.scandir(join(data_path, set, subfolder)) if f.is_dir() and not (f.name.find('P') == -1)]
    for patient in patients:
        # get subfolder names for image slices
        slices = [os.path.basename(f.path) for f in os.scandir(join(data_path, set, subfolder, patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
        for slice in slices:
            # get all frames for each slice
            frames = [f.path for f in os.scandir(join(data_path, set, subfolder, patient, slice)) if isfile(join(data_path, set, subfolder, patient, slice, f))]
            # add all combinations of the frames to a list 
            image_pairs = image_pairs + list(zip(frames[:-1], frames[1:]))#list(itertools.permutations(frames, 2))
    return image_pairs

def load_image_pair_CMR(pathname1, pathname2):
    """ expects paths for files and loads the corresponding images """
    # read in images 
    image1 = imread(pathname1, as_gray=True)/255
    image2 = imread(pathname2, as_gray=True)/255
    
    # convert to tensor
    image1 = torch.from_numpy(image1).unsqueeze(0)
    image2 = torch.from_numpy(image2).unsqueeze(0)
    """
    # crop images
    image1 = crop_image(image1)
    image2 = crop_image(image2)
    # convert to tensor
    image1 = torch.from_numpy(image1)
    image2 = torch.from_numpy(image2)
    # interpolate to size [246, 512]
    image1 = F.interpolate(image1.unsqueeze(0).unsqueeze(0), (246,512), mode='bilinear').squeeze(0)
    image2 = F.interpolate(image2.unsqueeze(0).unsqueeze(0), (246,512), mode='bilinear').squeeze(0)
    """
    return image1, image2

class TrainDatasetCMR(Data.Dataset):
  'Training dataset for CMR data'
  def __init__(self, data_path, mode):
        'Initialization'
        super(TrainDatasetCMR, self).__init__()
        # choose subfolder according to mode
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08' 
        else:
            print('Invalid mode for CMR training dataset!!')
        # get paths of training data
        self.data_path = data_path
        self.paths = get_image_pairs(set='TrainingSet/Croped', data_path=data_path, subfolder=subfolder)
            
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img = load_image_pair_CMR(self.paths[index][0], self.paths[index][1])
        return  mov_img, fix_img

class ValidationDatasetCMR(Data.Dataset):
  'Validation dataset for CMR data'
  def __init__(self, data_path, mode):
        'Initialization'
        # choose subfolder according to mode
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08' 
        else:
            print('Invalid mode for CMR training dataset!!')
        # get names and paths of training data
        self.data_path = data_path
        self.paths = get_image_pairs(set='ValidationSet/Croped', data_path=data_path, subfolder=subfolder)
            
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img = load_image_pair_CMR(self.paths[index][0], self.paths[index][1])
        return  mov_img, fix_img
  
class TestDatasetCMR(Data.Dataset):
  'Test dataset for CMR data'
  def __init__(self, data_path, mode):
        'Initialization'
        self.data_path = data_path
        # choose subfolder according to mode
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08' 
        else:
            print('Invalid mode for CMR training dataset!!')
        # get names and paths of training data
        self.data_path = data_path
        self.paths = get_image_pairs(set='TestSet', data_path=data_path, subfolder=subfolder)

        """
        self.names = []  
        self.foldernames = [os.path.basename(f.path) for f in os.scandir(join(data_path, 'ValidationSet', subfolder)) if f.is_dir() and not (f.name.find('P') == -1)]
        for folder in self.foldernames:
            names = [f.path for f in os.scandir(join(data_path, 'ValidationSet', subfolder, folder)) if isfile(join(data_path, 'ValdiationSet', subfolder, folder, f))]
            self.names = self.names+names
        self.zip_pathname_1 = list(zip(self.names[:-1], self.names[1:]))
        self.zip_pathname_2 = list(zip(self.names[1:], self.names[:-1]))
        self.paths = self.zip_pathname_1 + self.zip_pathname_2
        """  
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img = load_image_pair_CMR(self.paths[index][0], self.paths[index][1])
        return  mov_img, fix_img


class TrainDataset(Data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data_path, trainingset = 1):
        'Initialization'
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.names = [f for f in listdir(join(data_path, 'imagesTr')) if isfile(join(data_path, 'imagesTr', f))][0:201]
        if trainingset == 1:
            self.filename = list(zip(self.names[:-1], self.names[1:]))
            assert len(self.filename) == 200, "Oh no! # of images != 200."
        elif trainingset == 2:
            self.filename = list(zip(self.names[1:], self.names[:-1]))
            assert len(self.filename) == 200, "Oh no! # of images != 200."
        elif trainingset == 3:
            self.zip_filename_1 = list(zip(self.names[:-1], self.names[1:]))
            self.zip_filename_2 = list(zip(self.names[1:], self.names[:-1]))
            self.filename = self.zip_filename_1 + self.zip_filename_2
            assert len(self.filename) == 400, "Oh no! # of images != 400."
        elif trainingset == 4:
            self.filename = list(itertools.permutations(self.names, 2))
            assert len(self.filename) == 40200, "Oh no! # of images != 40200."   
        else:
             assert 0==1, print('TrainDataset Invalid!')
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img = load_train_pair_OASIS(self.data_path, self.filename[index][0], self.filename[index][1])
        return  mov_img, fix_img

def load_train_pair_OASIS(data_path, filename1, filename2):
    # Load images and labels
    nim1 = nib.load(os.path.join(data_path, 'imagesTr', filename1)) 
    image1 = nim1.get_fdata()[:,96,:]
    image1 = np.array(image1, dtype='float32')

    nim2 = nib.load(os.path.join(data_path, 'imagesTr', filename2)) 
    image2 = nim2.get_fdata()[:,96,:]
    image2 = np.array(image2, dtype='float32')
    
    #"""
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    #"""
    return image1, image2

class ValidationDataset(Data.Dataset):
  'Validation Dataset'
  def __init__(self, data_path):
        'Initialization'
        super(ValidationDataset, self).__init__()
        self.data_path = data_path
        self.names = [f for f in listdir(join(data_path, 'imagesTr')) if isfile(join(data_path, 'imagesTr', f))][202:213]
        self.zip_filename_1 = list(zip(self.names[:-1], self.names[1:]))
        self.zip_filename_2 = list(zip(self.names[1:], self.names[:-1]))
        self.filename = self.zip_filename_1 + self.zip_filename_2
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_A, img_B, label_A, label_B = load_validation_pair(self.data_path, self.filename[index][0], self.filename[index][1])
        
        #return self.filename[index][0],self.filename[index][1], img_A, img_B, label_A, label_B
        return img_A, img_B, label_A, label_B
  
def load_validation_pair(data_path, filename1, filename2):
    # Load images and labels
    nim1 = nib.load(os.path.join(data_path, 'imagesTr', filename1)) 
    image1 = nim1.get_fdata()[:,96,:]
    image1 = np.array(image1, dtype='float32')

    nim2 = nib.load(os.path.join(data_path, 'imagesTr', filename2)) 
    image2 = nim2.get_fdata()[:,96,:] 
    image2 = np.array(image2, dtype='float32')
    
    nim5 = nib.load(os.path.join(data_path, 'labelsTr', filename1)) 
    image5 = nim5.get_fdata()[:,96,:]
    image5 = np.array(image5, dtype='float32')
    # image5 = image5 / 35.0
    nim6 = nib.load(os.path.join(data_path, 'labelsTr', filename2)) 
    image6 = nim6.get_fdata()[:,96,:]
    image6 = np.array(image6, dtype='float32') 
    
    #"""
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    image5 = np.reshape(image5, (1,) + image5.shape)
    image6 = np.reshape(image6, (1,) + image6.shape)
    #"""
    return image1, image2, image5, image6

class TestDataset(Data.Dataset):
  'Test Dataset'
  def __init__(self, data_path):
        super(TestDataset, self).__init__()
        self.data_path = data_path
        self.names = [f for f in listdir(join(data_path, 'imagesTr')) if isfile(join(data_path, 'imagesTr', f))][214:414]
        self.zip_filename_1 = list(zip(self.names[:-1], self.names[1:]))
        self.zip_filename_2 = list(zip(self.names[1:], self.names[:-1]))
        self.filename = self.zip_filename_1 + self.zip_filename_2
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_A, img_B, label_A, label_B = load_validation_pair(self.data_path, self.filename[index][0], self.filename[index][1])
        return img_A, img_B, label_A, label_B
  
def load_test_pair(data_path, filename1, filename2):
    # Load images and labels
    nim1 = nib.load(os.path.join(data_path, 'imagesTr', filename1))
    image1 = nim1.get_fdata()[:,96,:]
    image1 = np.array(image1, dtype='float32')

    nim2 = nib.load(os.path.join(data_path, 'imagesTr', filename2)) 
    image2 = nim2.get_fdata()[:,96,:] 
    image2 = np.array(image2, dtype='float32')
    
    nim5 = nib.load(os.path.join(data_path, 'labelsTr', filename1)) 
    image5 = nim5.get_fdata()[:,96,:]
    image5 = np.array(image5, dtype='float32')
    # image5 = image5 / 35.0
    nim6 = nib.load(os.path.join(data_path, 'labelsTr', filename2)) 
    image6 = nim6.get_fdata()[:,96,:]
    image6 = np.array(image6, dtype='float32') 
    
    #"""
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    image5 = np.reshape(image5, (1,) + image5.shape)
    image6 = np.reshape(image6, (1,) + image6.shape)
    #"""
    return image1, image2, image5, image6

    
def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    # disp = disp.transpose(1, 2, 3, 0)
    disp = disp.transpose(1, 2, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    
    import pystrum.pynd.ndutils as nd
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def dice(pred1, truth1):
    mask4_value1 = np.unique(pred1)
    mask4_value2 = np.unique(truth1)
    mask_value4 = list(set(mask4_value1) & set(mask4_value2))
    dice_list=[]
    for k in mask_value4[1:]:
        truth = truth1 == k
        pred = pred1 == k
        intersection = np.sum(pred * truth) * 2.0
        dice_list.append(intersection / (np.sum(pred) + np.sum(truth)))
    return np.mean(dice_list)

def save_checkpoint(state, save_dir, save_filename, max_model_num=10):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))
