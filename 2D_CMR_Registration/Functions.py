"""
Helper functions from https://github.com/zhangjun001/ICNet.

Some functions has been modified.
"""

import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
from os import listdir, scandir, remove
from os.path import join, isfile, basename
import matplotlib.pyplot as plt
import itertools
from natsort import natsorted
import glob
import h5py
import fastmri
from fastmri.data import transforms as T
import torch.nn.functional as F
from skimage.io import imread
import time
from skimage.metrics import structural_similarity, mean_squared_error
import csv
from os import mkdir
from os.path import isdir
from matplotlib.pyplot import cm
from torch.fft import fftn, ifftn, fftshift, ifftshift
from typing import Optional, Tuple, Union
from skimage.util import view_as_windows

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

def get_image_pairs_ACDC(subset, data_path, subfolder):
    """ get the corresponding paths of image pairs for the ACDC dataset """
    image_pairs = []  
    # get folder names for patients
    patients = [basename(f.path) for f in scandir(join(data_path, subfolder, subset)) if f.is_dir() and not (f.name.find('patient') == -1)]
    for patient in patients:
        # get subfolder names for image slices
        slices = [basename(f.path) for f in scandir(join(data_path, subfolder, subset, patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
        for slice in slices:
            # get all frames for each slice
            frames = [f.path for f in scandir(join(data_path, subfolder, subset, patient, slice)) if isfile(join(data_path, subfolder, subset, patient, slice, f)) and f.name.startswith('Image_Frame')]
            if subset == 'Training':
                # add all combinations of the frames to a list 
                image_pairs = image_pairs + list(itertools.combinations(frames, 2)) 
            else:  
                # add pairs of the frames to a list 
                image_pairs = image_pairs + list(zip(frames[:-1], frames[1:]))
    return image_pairs

def get_image_pairs_ACDC_ContrastiveLearning(subset, data_path, subfolder):
    """ get the corresponding paths of image pairs for contrastive learning on the ACDC dataset """
    image_pairs_fullySampled = []  
    image_pairs_subSampled = []  
    # get folder names for patients
    patients_fullySampled = [basename(f.path) for f in scandir(join(data_path, 'FullySampled', subset)) if f.is_dir() and not (f.name.find('patient') == -1)]
    patients_subSampled = [basename(f.path) for f in scandir(join(data_path, subfolder, subset)) if f.is_dir() and not (f.name.find('patient') == -1)]
    
    for patient in patients_fullySampled:
        if patient in patients_subSampled:
            # get subfolder names for image slices
            slices_fullySampled = [basename(f.path) for f in scandir(join(data_path, 'FullySampled', subset, patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
            slices_subSampled = [basename(f.path) for f in scandir(join(data_path, subfolder, subset, patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
            for slice in slices_fullySampled:
                if slice in slices_subSampled:
                    # get all frames for each slice
                    frames_fullySampled = [f.path for f in scandir(join(data_path, 'FullySampled', subset, patient, slice)) if isfile(join(data_path, 'FullySampled', subset, patient, slice, f)) and f.name.startswith('Image_Frame')]#[0:len(frames_subSampled)]
                    frames_subSampled = [f.path for f in scandir(join(data_path, subfolder, subset, patient, slice)) if isfile(join(data_path, subfolder, subset, patient, slice, f)) and f.name.startswith('Image_Frame')]
                    # add pairs of the frames to a list 
                    image_pairs_fullySampled = image_pairs_fullySampled + list(itertools.combinations(frames_fullySampled, 2))
                    image_pairs_subSampled = image_pairs_subSampled + list(itertools.combinations(frames_subSampled, 2))
    return image_pairs_fullySampled, image_pairs_subSampled

def get_image_pairs_ACDC_kSpace(subset, data_path, subfolder):
    """ get the corresponding paths of k-space pairs for the ACDC dataset """
    image_pairs = []  
    # get folder names
    pairs = [basename(f.path) for f in scandir(join(data_path, subfolder, subset)) if f.is_dir() and not (f.name.find('ImagePair') == -1)]
    for pair in pairs:
        # get moving image crops
        mov_crops = [basename(f.path) for f in scandir(join(data_path, subfolder, subset, pair)) if f.is_dir() and f.name.startswith('MovingImage')]
        # get fixed image crops
        fix_crops = [basename(f.path) for f in scandir(join(data_path, subfolder, subset, pair)) if f.is_dir() and f.name.startswith('FixedImage')]
        # sort all crops?
        image_pairs = image_pairs + zip(mov_crops,fix_crops)
    return image_pairs

def load_image_pair_ACDC_train(pathname1, pathname2):
    """ expects paths for files and loads the corresponding image pair"""
    # read in images 
    image1 = imread(pathname1, as_gray=True)/255
    image2 = imread(pathname2, as_gray=True)/255
    # convert to tensors and add singleton dimension for the correct size
    image1 = torch.from_numpy(image1)#.unsqueeze(0).unsqueeze(0)
    image2 = torch.from_numpy(image2)#.unsqueeze(0).unsqueeze(0)
    # interpolate all images to the same size
    image1 = F.interpolate(image1.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').squeeze(0)
    image2 = F.interpolate(image2.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').squeeze(0)

    return image1, image2

def load_image_pair_ACDC(pathname1, pathname2):
    """ expects paths for files and loads the corresponding images and segmentations """
    # read in images 
    image1 = imread(pathname1, as_gray=True)/255
    image2 = imread(pathname2, as_gray=True)/255

    # read in segmentations (only have value 0 for background and 1,2,3 for structures)
    seg1 = imread(pathname1.replace('Image','Segmentation'), as_gray=True)/3
    seg2 = imread(pathname2.replace('Image','Segmentation'), as_gray=True)/3
    
    # convert to tensors and add singleton dimension for the correct size
    image1 = torch.from_numpy(image1)#.unsqueeze(0).unsqueeze(0)
    image2 = torch.from_numpy(image2)#.unsqueeze(0).unsqueeze(0)
    seg1 = torch.from_numpy(seg1)#.unsqueeze(0).unsqueeze(0)
    seg2 = torch.from_numpy(seg2)#.unsqueeze(0).unsqueeze(0)

    # interpolate all images and segmentations to the same size
    image1 = F.interpolate(image1.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').squeeze(0)
    image2 = F.interpolate(image2.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').squeeze(0)
    # use nearest interpolation for the segmentations to preserve labels
    seg1 = F.interpolate(seg1.unsqueeze(0).unsqueeze(0), (216,256), mode='nearest').squeeze(0)
    seg2 = F.interpolate(seg2.unsqueeze(0).unsqueeze(0), (216,256), mode='nearest').squeeze(0)
    
    return image1, image2, seg1, seg2

def load_image_pair_ACDC_train_LAPNet(pathname1, pathname2):
    """ expects paths for files and loads the corresponding image pair"""
    # read in images 
    image1 = imread(pathname1, as_gray=True)/255
    image2 = imread(pathname2, as_gray=True)/255
    # convert to tensors and add singleton dimension for the correct size
    image1 = torch.from_numpy(image1)
    image2 = torch.from_numpy(image2)
    # interpolate all images to the same size
    image1 = F.interpolate(image1.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').squeeze(0)
    image2 = F.interpolate(image2.unsqueeze(0).unsqueeze(0), (216,256), mode='bilinear').squeeze(0)
    # convert to k-space domain
    k_space1 = FFT(image1)
    k_space2 = FFT(image2)
    # convert to real tensors with 2 extra dimensions for real and imaginary parts
    k_space1 = torch.view_as_real(k_space1)
    k_space2 = torch.view_as_real(k_space2)
    # get ground truth flow between images
    flow = ''

    return k_space1, k_space2, flow

class TrainDatasetACDC_ContrastiveLearning(Data.Dataset):
  'Training dataset for ACDC data'
  def __init__(self, data_path, mode):
        'Initialization'
        super(TrainDatasetACDC_ContrastiveLearning, self).__init__()
        # choose subfolder according to mode
        assert mode >= 1 and mode <= 3, f"Expected mode for ACDC training dataset to be one of 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08'
        elif mode == 3:
            subfolder = 'AccFactor10'  
        # get paths of training data
        self.data_path = data_path
        self.paths_fullySampled, self.paths_subSampled = get_image_pairs_ACDC_ContrastiveLearning(subset='Training', data_path=data_path, subfolder=subfolder)
            
  def __len__(self):
        'total number of training samples'
        return len(self.paths_fullySampled)

  def __getitem__(self, index):
        'Loads fully sampled and subsampled image pairs'
        mov_img_fullySampled, fix_img_fullySampled  = load_image_pair_ACDC_train(self.paths_fullySampled[index][0], self.paths_fullySampled[index][1])
        mov_img_subSampled, fix_img_subSampled      = load_image_pair_ACDC_train(self.paths_subSampled[index][0], self.paths_subSampled[index][1])
        return  mov_img_fullySampled, fix_img_fullySampled, mov_img_subSampled, fix_img_subSampled

class TrainDatasetACDC_kSpace(Data.Dataset):
  'Training dataset for k-space ACDC data'
  def __init__(self, data_path, mode):
        'Initialization'
        super(TrainDatasetACDC_kSpace, self).__init__()
        # choose subfolder according to mode
        assert mode >= 0 and mode <= 3, f"Expected mode for ACDC training dataset to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08'
        elif mode == 3:
            subfolder = 'AccFactor10'  
        # get paths of training data
        self.data_path = data_path
        self.paths = get_image_pairs_ACDC_kSpace(subset='Training', data_path=data_path, subfolder=subfolder)
            
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        k_space1 = torch.load(self.paths[index][0])
        k_space2 = torch.load(self.paths[index][1])
        return k_space1, k_space2

class TrainDatasetACDC(Data.Dataset):
  'Training dataset for ACDC data'
  def __init__(self, data_path, mode):
        'Initialization'
        super(TrainDatasetACDC, self).__init__()
        # choose subfolder according to mode
        assert mode >= 0 and mode <= 3, f"Expected mode for ACDC training dataset to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08'
        elif mode == 3:
            subfolder = 'AccFactor10'  
        # get paths of training data
        self.data_path = data_path
        self.paths = get_image_pairs_ACDC(subset='Training', data_path=data_path, subfolder=subfolder)
            
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img = load_image_pair_ACDC_train(self.paths[index][0], self.paths[index][1])
        return  mov_img, fix_img

class ValidationDatasetACDC_kSpace(Data.Dataset):
  'Validation dataset for k-space ACDC data'
  def __init__(self, data_path, mode):
        'Initialization'
        # choose subfolder according to mode
        assert mode >= 0 and mode <= 3, f"Expected mode for ACDC validation dataset to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08' 
        elif mode == 3:
            subfolder = 'AccFactor10' 
        # get names and paths of training data
        self.data_path = data_path
        self.paths = get_image_pairs_ACDC(subset='Validation', data_path=data_path, subfolder=subfolder)
            
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img, mov_seg, fix_seg = load_image_pair_ACDC(self.paths[index][0], self.paths[index][1])
        return  mov_img, fix_img, mov_seg, fix_seg

class ValidationDatasetACDC(Data.Dataset):
  'Validation dataset for ACDC data'
  def __init__(self, data_path, mode):
        'Initialization'
        # choose subfolder according to mode
        assert mode >= 0 and mode <= 3, f"Expected mode for ACDC validation dataset to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08' 
        elif mode == 3:
            subfolder = 'AccFactor10' 
        # get names and paths of training data
        self.data_path = data_path
        self.paths = get_image_pairs_ACDC(subset='Validation', data_path=data_path, subfolder=subfolder)
            
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img, mov_seg, fix_seg = load_image_pair_ACDC(self.paths[index][0], self.paths[index][1])
        return  mov_img, fix_img, mov_seg, fix_seg
  
class TestDatasetACDC(Data.Dataset):
  'Test dataset for ACDC data'
  def __init__(self, data_path, mode):
        'Initialization'
        self.data_path = data_path
        # choose subfolder according to mode
        assert mode >= 0 and mode <= 3, f"Expected mode for ACDC test dataset to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08' 
        elif mode == 3:
            subfolder = 'AccFactor10' 
        # get names and paths of training data
        self.data_path = data_path
        self.paths = get_image_pairs_ACDC(subset='Test', data_path=data_path, subfolder=subfolder)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img, mov_seg, fix_seg = load_image_pair_ACDC(self.paths[index][0], self.paths[index][1])
        return  mov_img, fix_img, mov_seg, fix_seg

class TestDatasetACDCBenchmark(Data.Dataset):
  'Test dataset for benchmark of subsampled ACDC data'
  def __init__(self, data_path, mode):
        'Initialization'
        # choose subfolder according to mode
        assert mode >= 0 and mode <= 3, f"Expected mode for ACDC test benchmark to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08' 
        elif mode == 3:
            subfolder = 'AccFactor10' 
        
        # get names and paths of training data
        subset = 'Test'
        self.image_pairs_fullySampled = []  
        self.image_pairs_subSampled = []  
        # get folder names for patients
        patients_fullySampled = [basename(f.path) for f in scandir(join(data_path, subset, 'FullySampled')) if f.is_dir() and not (f.name.find('P') == -1)]
        patients_subSampled = [basename(f.path) for f in scandir(join(data_path, subset, subfolder)) if f.is_dir() and not (f.name.find('P') == -1)]
        
        for patient in patients_fullySampled:
            if patient in patients_subSampled:
                # get subfolder names for image slices
                slices_fullySampled = [basename(f.path) for f in scandir(join(data_path, subset, 'FullySampled', patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
                slices_subSampled = [basename(f.path) for f in scandir(join(data_path, subset, subfolder, patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
                for slice in slices_fullySampled:
                    if slice in slices_subSampled:
                        # get all frames for each slice
                        frames_subSampled = [f.path for f in scandir(join(data_path, subset, subfolder, patient, slice)) if isfile(join(data_path, subset, subfolder, patient, slice, f))]
                        frames_fullySampled = [f.path for f in scandir(join(data_path, subset, 'FullySampled', patient, slice)) if isfile(join(data_path, subset, 'FullySampled', patient, slice, f))][0:len(frames_subSampled)]
                        # add pairs of the frames to a list 
                        self.image_pairs_fullySampled = self.image_pairs_fullySampled + list(zip(frames_fullySampled[:-1], frames_fullySampled[1:]))
                        self.image_pairs_subSampled = self.image_pairs_subSampled + list(zip(frames_subSampled[:-1], frames_subSampled[1:]))
        
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_pairs_fullySampled)

  def __getitem__(self, index):
        'Generates two image pairs (1x fully sampled, 1x subsampled)'
        # read in image data
        mov_img_fullySampled = imread(self.image_pairs_fullySampled[index][0], as_gray=True)/255
        fix_img_fullySampled = imread(self.image_pairs_fullySampled[index][1], as_gray=True)/255
        mov_img_subSampled = imread(self.image_pairs_subSampled[index][0], as_gray=True)/255
        fix_img_subSampled = imread(self.image_pairs_subSampled[index][1], as_gray=True)/255
        
        # convert to tensor
        mov_img_fullySampled = torch.from_numpy(mov_img_fullySampled).unsqueeze(0)
        fix_img_fullySampled = torch.from_numpy(fix_img_fullySampled).unsqueeze(0)
        mov_img_subSampled = torch.from_numpy(mov_img_subSampled).unsqueeze(0)
        fix_img_subSampled = torch.from_numpy(fix_img_subSampled).unsqueeze(0)

        return  mov_img_fullySampled, fix_img_fullySampled, mov_img_subSampled, fix_img_subSampled


def get_image_pairs_CMRxRecon(subset, data_path, subfolder):
    """ get the corresponding paths of image pairs for the CMRxRecon dataset """
    image_pairs = []  
    # get folder names for patients
    patients = [basename(f.path) for f in scandir(join(data_path, subset, subfolder)) if f.is_dir() and not (f.name.find('P') == -1)]
    for patient in patients:
        # get subfolder names for image slices
        slices = [basename(f.path) for f in scandir(join(data_path, subset, subfolder, patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
        for slice in slices:
            # get all frames for each slice
            frames = [f.path for f in scandir(join(data_path, subset, subfolder, patient, slice)) if isfile(join(data_path, subset, subfolder, patient, slice, f))]
            if subset == 'TestSet':
                # add pairs of the frames to a list 
                image_pairs = image_pairs + list(zip(frames[:-1], frames[1:]))
            else:    
                # add all combinations of the frames to a list 
                image_pairs = image_pairs + list(itertools.combinations(frames, 2)) #list(zip(frames[:-1], frames[1:]))#list(itertools.permutations(frames, 2))
    return image_pairs

def load_image_pair_CMRxRecon(pathname1, pathname2):
    """ expects paths for files and loads the corresponding images """
    # read in images 
    image1 = imread(pathname1, as_gray=True)/255
    image2 = imread(pathname2, as_gray=True)/255
    
    # convert to tensor
    image1 = torch.from_numpy(image1).unsqueeze(0)
    image2 = torch.from_numpy(image2).unsqueeze(0)
    
    return image1, image2

class TrainDatasetCMRxRecon(Data.Dataset):
  'Training dataset for CMR data'
  def __init__(self, data_path, mode):
        'Initialization'
        super(TrainDatasetCMRxRecon, self).__init__()
        # choose subfolder according to mode
        assert mode >= 0 and mode <= 3, f"Expected mode for CMRxRecon training dataset to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08'
        elif mode == 3:
            subfolder = 'AccFactor10'  
        # get paths of training data
        self.data_path = data_path
        self.paths = get_image_pairs_CMRxRecon(subset='TrainingSet/Croped', data_path=data_path, subfolder=subfolder)
            
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img = load_image_pair_CMRxRecon(self.paths[index][0], self.paths[index][1])
        return  mov_img, fix_img

class ValidationDatasetCMRxRecon(Data.Dataset):
  'Validation dataset for CMRxRecon data'
  def __init__(self, data_path, mode):
        'Initialization'
        # choose subfolder according to mode
        assert mode >= 0 and mode <= 3, f"Expected mode for CMRxRecon validation dataset to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08' 
        elif mode == 3:
            subfolder = 'AccFactor10' 
        # get names and paths of training data
        self.data_path = data_path
        self.paths = get_image_pairs_CMRxRecon(subset='ValidationSet/Croped', data_path=data_path, subfolder=subfolder)
            
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img = load_image_pair_CMRxRecon(self.paths[index][0], self.paths[index][1])
        return  mov_img, fix_img
  
class TestDatasetCMRxRecon(Data.Dataset):
  'Test dataset for CMRxRecon data'
  def __init__(self, data_path, mode):
        'Initialization'
        self.data_path = data_path
        # choose subfolder according to mode
        assert mode >= 0 and mode <= 3, f"Expected mode for CMRxRecon test dataset to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08' 
        elif mode == 3:
            subfolder = 'AccFactor10' 
        # get names and paths of training data
        self.data_path = data_path
        self.paths = get_image_pairs_CMRxRecon(subset='TestSet/Croped', data_path=data_path, subfolder=subfolder)

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        mov_img, fix_img = load_image_pair_CMRxRecon(self.paths[index][0], self.paths[index][1])
        return  mov_img, fix_img

class TestDatasetCMRxReconBenchmark(Data.Dataset):
  'Test dataset for benchmark of subsampled CMRxRecon data'
  def __init__(self, data_path, mode):
        'Initialization'
        # choose subfolder according to mode
        assert mode >= 0 and mode <= 3, f"Expected mode for CMRxRecon test benchmark to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 0:
            subfolder = 'FullySampled'
        elif mode == 1:
            subfolder = 'AccFactor04'
        elif mode == 2:
            subfolder = 'AccFactor08' 
        elif mode == 3:
            subfolder = 'AccFactor10' 
        # get names and paths of training data
        subset = 'TestSet/Croped'
        self.image_pairs_fullySampled = []  
        self.image_pairs_subSampled = []  
        # get folder names for patients
        patients_fullySampled = [basename(f.path) for f in scandir(join(data_path, subset, 'FullySampled')) if f.is_dir() and not (f.name.find('P') == -1)]
        patients_subSampled = [basename(f.path) for f in scandir(join(data_path, subset, subfolder)) if f.is_dir() and not (f.name.find('P') == -1)]
        
        for patient in patients_fullySampled:
            if patient in patients_subSampled:
                # get subfolder names for image slices
                slices_fullySampled = [basename(f.path) for f in scandir(join(data_path, subset, 'FullySampled', patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
                slices_subSampled = [basename(f.path) for f in scandir(join(data_path, subset, subfolder, patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
                for slice in slices_fullySampled:
                    if slice in slices_subSampled:
                        # get all frames for each slice
                        frames_subSampled = [f.path for f in scandir(join(data_path, subset, subfolder, patient, slice)) if isfile(join(data_path, subset, subfolder, patient, slice, f))]
                        frames_fullySampled = [f.path for f in scandir(join(data_path, subset, 'FullySampled', patient, slice)) if isfile(join(data_path, subset, 'FullySampled', patient, slice, f))][0:len(frames_subSampled)]
                        # add pairs of the frames to a list 
                        self.image_pairs_fullySampled = self.image_pairs_fullySampled + list(zip(frames_fullySampled[:-1], frames_fullySampled[1:]))
                        self.image_pairs_subSampled = self.image_pairs_subSampled + list(zip(frames_subSampled[:-1], frames_subSampled[1:]))
        
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.image_pairs_fullySampled)

  def __getitem__(self, index):
        'Generates two image pairs (1x fully sampled, 1x subsampled)'
        # read in image data
        mov_img_fullySampled = imread(self.image_pairs_fullySampled[index][0], as_gray=True)/255
        fix_img_fullySampled = imread(self.image_pairs_fullySampled[index][1], as_gray=True)/255
        mov_img_subSampled = imread(self.image_pairs_subSampled[index][0], as_gray=True)/255
        fix_img_subSampled = imread(self.image_pairs_subSampled[index][1], as_gray=True)/255
        
        # convert to tensor
        mov_img_fullySampled = torch.from_numpy(mov_img_fullySampled).unsqueeze(0)
        fix_img_fullySampled = torch.from_numpy(fix_img_fullySampled).unsqueeze(0)
        mov_img_subSampled = torch.from_numpy(mov_img_subSampled).unsqueeze(0)
        fix_img_subSampled = torch.from_numpy(fix_img_subSampled).unsqueeze(0)

        return  mov_img_fullySampled, fix_img_fullySampled, mov_img_subSampled, fix_img_subSampled


class TrainDatasetOASIS(Data.Dataset):
  'OASIS dataset'
  def __init__(self, data_path, trainingset = 1):
        'Initialization'
        super(TrainDatasetOASIS, self).__init__()
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
    nim1 = nib.load(join(data_path, 'imagesTr', filename1)) 
    image1 = nim1.get_fdata()[:,96,:]
    image1 = np.array(image1, dtype='float32')

    nim2 = nib.load(join(data_path, 'imagesTr', filename2)) 
    image2 = nim2.get_fdata()[:,96,:]
    image2 = np.array(image2, dtype='float32')
    
    #"""
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    #"""
    return image1, image2

class ValidationDatasetOASIS(Data.Dataset):
  'Validation Dataset'
  def __init__(self, data_path):
        'Initialization'
        super(ValidationDatasetOASIS, self).__init__()
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
        img_A, img_B, label_A, label_B = load_validation_pair_OASIS(self.data_path, self.filename[index][0], self.filename[index][1])
        
        #return self.filename[index][0],self.filename[index][1], img_A, img_B, label_A, label_B
        return img_A, img_B, label_A, label_B
  
def load_validation_pair_OASIS(data_path, filename1, filename2):
    # Load images and labels
    nim1 = nib.load(join(data_path, 'imagesTr', filename1)) 
    image1 = nim1.get_fdata()[:,96,:]
    image1 = np.array(image1, dtype='float32')

    nim2 = nib.load(join(data_path, 'imagesTr', filename2)) 
    image2 = nim2.get_fdata()[:,96,:] 
    image2 = np.array(image2, dtype='float32')
    
    nim5 = nib.load(join(data_path, 'labelsTr', filename1)) 
    image5 = nim5.get_fdata()[:,96,:]
    image5 = np.array(image5, dtype='float32')
    # image5 = image5 / 35.0
    nim6 = nib.load(join(data_path, 'labelsTr', filename2)) 
    image6 = nim6.get_fdata()[:,96,:]
    image6 = np.array(image6, dtype='float32') 
    
    #"""
    image1 = np.reshape(image1, (1,) + image1.shape)
    image2 = np.reshape(image2, (1,) + image2.shape)
    image5 = np.reshape(image5, (1,) + image5.shape)
    image6 = np.reshape(image6, (1,) + image6.shape)
    #"""
    return image1, image2, image5, image6

class TestDatasetOASIS(Data.Dataset):
  'Test Dataset'
  def __init__(self, data_path):
        super(TestDatasetOASIS, self).__init__()
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
        img_A, img_B, label_A, label_B = load_validation_pair_OASIS(self.data_path, self.filename[index][0], self.filename[index][1])
        return img_A, img_B, label_A, label_B

def save_flow(mov, fix, warp, grid, save_path):
    mov = mov.data.cpu().numpy()[0, 0, ...]
    fix = fix.data.cpu().numpy()[0, 0, ...]
    warp = warp.data.cpu().numpy()[0, 0, ...] 
    
    plt.subplots(figsize=(7, 4))
    plt.axis('off')

    plt.subplot(2,3,1)
    plt.imshow(mov, cmap='gray', vmin=0, vmax = 1)
    plt.title('Moving Image')
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(fix, cmap='gray', vmin=0, vmax = 1)
    plt.title('Fixed Image')
    plt.axis('off')

    plt.subplot(2,3,3)
    plt.imshow(warp, cmap='gray', vmin=0, vmax = 1)
    plt.title('Warped Image')
    plt.axis('off')

    if type(grid) != type(None):
        grid = grid.data.cpu().numpy()[0,:,:,:]
        plt.subplot(2,3,4)
        interval = 5
        for i in range(0,grid.shape[1]-1,interval):
            plt.plot(grid[0,i,:], grid[1,i,:],c='g',lw=1)
        #plot the vertical lines
        for i in range(0,grid.shape[2]-1,interval):
            plt.plot(grid[0,:,i], grid[1,:,i],c='g',lw=1)

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.title('Displacement Field')
        plt.axis('off')

    plt.subplot(2,3,5)
    plt.imshow(abs(mov-fix), cmap='gray', vmin=0, vmax = 1)
    plt.title('Difference before')
    plt.axis('off')
    
    plt.subplot(2,3,6)
    plt.imshow(abs(warp-fix), cmap='gray', vmin=0, vmax = 1)
    plt.title('Difference after')
    plt.axis('off')
    plt.savefig(save_path,bbox_inches='tight')
    plt.close()

    
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

def dice_ACDC(pred, truth):
    '''
    read in segmentations (numpy float arrays) and calculate the dice score for all four labels of the ACDC dataset
    '''
    #labels_pred = np.unique(pred)
    #labels_truth = np.unique(truth)
    pred = pred*3
    pred = pred.astype(np.int64)
    truth = truth*3
    truth = truth.astype(np.int64)
    label_values = [0,1,2,3] #[0.0, 0.33333334, 0.6666667, 1.0]
    #labels_pred_int = np.unique(pred)
    #labels_truth_int = np.unique(truth)
    dice_list=np.ones(4)
    for i,k in enumerate(label_values):
        truth_bool = truth == k
        pred_bool = pred == k
        sum_pred = np.sum(pred_bool)
        sum_truth = np.sum(truth_bool)
        sum_total = np.sum(pred_bool * truth_bool)
        # only calculate dice if label is in one of the segmentations (get NaN values otherwise)
        if (sum_pred != 0) or (sum_truth != 0):
            intersection = sum_total * 2.0
            dice_list[i] = intersection / (sum_pred + sum_truth)
    return dice_list

def save_checkpoint(state, save_dir, save_filename, max_model_num=10):
    torch.save(state, save_dir + save_filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))

def log_TrainTest(wandb, model, model_name, diffeo_name, dataset, FT_size, learning_rate, start_channel, smth_lambda, choose_loss, mode, epochs, optimizer, loss_similarity, loss_smooth, transform, training_generator, validation_generator, test_generator, device, earlyStop):
    
    # path to save model parameters to
    model_dir = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(dataset,model_name,diffeo_name,choose_loss,start_channel,FT_size[0],FT_size[1],smth_lambda,learning_rate,mode)
    model_dir_png = './ModelParameters-{}/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}_Png/'.format(dataset,model_name,diffeo_name,choose_loss,start_channel,FT_size[0],FT_size[1],smth_lambda,learning_rate,mode)

    if not isdir(model_dir_png):
        mkdir(model_dir_png)

    if not isdir(model_dir):
        mkdir(model_dir)
    
    ##############
    ## Training ##
    ##############

    if earlyStop:
        # counter and best SSIM for early stopping
        counter_earlyStopping = 0
        if dataset == 'CMRxRecon':
            best_SSIM = 0
        else:
            best_Dice = 0

    print('\nStarted training on ', time.ctime())

    for epoch in range(epochs):
        losses = np.zeros(training_generator.__len__())
        for i, image_pair in enumerate(training_generator):
            if choose_loss == 5:    # contrastive loss
                mov_img_fullySampled = image_pair[0].to(device).float()
                fix_img_fullySampled = image_pair[1].to(device).float()
                mov_img_subSampled   = image_pair[2].to(device).float()
                fix_img_subSampled   = image_pair[3].to(device).float()                 
                
                with torch.no_grad():
                    # get result for fully sampled image pairs (do not back-propagate!!)
                    Df_xy_fullySampled, features_disp_fullySampled = model(mov_img_fullySampled, fix_img_fullySampled)
                    #grid, warped_mov_fullySampled = transform(mov_img_fullySampled, Df_xy_fullySampled.permute(0, 2, 3, 1))
                
                # get result for subsampled image pairs
                Df_xy, features_disp        = model(mov_img_subSampled, fix_img_subSampled)
                grid, warped_mov_subSampled = transform(mov_img_subSampled, Df_xy.permute(0, 2, 3, 1)) 
                
                # transform features from [1,32,x,y] to [x*y,32]
                features_disp_fullySampled  = torch.flatten(features_disp_fullySampled.squeeze(), start_dim=1, end_dim=2).permute(1,0)
                features_disp               = torch.flatten(features_disp.squeeze(), start_dim=1, end_dim=2).permute(1,0)

                # take samples each 10 steps from the feature map with 32 channels
                indixes  = torch.arange(start=0,end=features_disp_fullySampled.shape[0],step=10)
                queries  = features_disp[indixes,:]
                pos_keys = features_disp_fullySampled[indixes,:]

                # compute image similarity loss
                loss1 = loss_similarity(fix_img_subSampled, warped_mov_subSampled) 
                # compute contrastive loss between fully sampled and subsampled results
                loss2 = loss_smooth(queries, pos_keys)
            else:    
                mov_img = image_pair[0].to(device).float()
                fix_img = image_pair[1].to(device).float()

                Df_xy, _ = model(mov_img, fix_img)
                grid, warped_mov = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
                
                # compute similarity loss in the image space
                loss1 = loss_similarity(fix_img, warped_mov)    
                
                # compute smoothness loss
                loss2 = loss_smooth(Df_xy)
                
            loss = loss1 + smth_lambda * loss2
            losses[i] = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ################
        ## Validation ##
        ################

        with torch.no_grad():
            model.eval()
            MSE_Validation = []
            SSIM_Validation = []
            if dataset != 'CMRxRecon':
                Dice_Validation = []
            
            for i, image_pair in enumerate(validation_generator): 
                fix_img = image_pair[0].cuda().float()
                mov_img = image_pair[1].cuda().float()
                
                if dataset != 'CMRxRecon':
                    mov_seg = image_pair[2].cuda().float()
                    fix_seg = image_pair[3].cuda().float()
                
                Df_xy, __ = model(mov_img, fix_img)
                # get warped image and segmentation
                grid, warped_mov_img = transform(mov_img, Df_xy.permute(0, 2, 3, 1), mod = 'nearest')
                if dataset != 'CMRxRecon':
                    grid, warped_mov_seg = transform(mov_seg, Df_xy.permute(0, 2, 3, 1), mod = 'nearest')

                # calculate Dice, MSE and SSIM 
                if dataset != 'CMRxRecon':
                    Dice_Validation.append(np.mean(dice_ACDC(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy())[1:3]))
                MSE_Validation.append(mean_squared_error(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy()))
                SSIM_Validation.append(structural_similarity(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy(), data_range=1))
            
            # calculate mean of validation metrics
            Mean_MSE = np.mean(MSE_Validation)
            Mean_SSIM = np.mean(SSIM_Validation)
            if dataset != 'CMRxRecon':
                Mean_Dice = np.mean([x for x in Dice_Validation if str(x) != 'nan']) 

            if earlyStop:
                # save best metrics and reset counter if Dice/SSIM got better, else increase counter for early stopping
                if dataset == 'CMRxRecon':
                    if Mean_SSIM>best_SSIM:
                        best_SSIM = Mean_SSIM
                        counter_earlyStopping = 0
                    else:
                        counter_earlyStopping += 1   
                else:
                    if Mean_Dice>best_Dice:
                        best_Dice = Mean_Dice
                        counter_earlyStopping = 0
                    else:
                        counter_earlyStopping += 1    
            
            # log loss and validation metrics to wandb
            if dataset == 'CMRxRecon':
                wandb.log({"Loss": np.mean(losses), "MSE": Mean_MSE, "SSIM": Mean_SSIM})
            else:
                wandb.log({"Loss": np.mean(losses), "Dice": Mean_Dice, "MSE": Mean_MSE, "SSIM": Mean_SSIM})
            
            # save and log model     
            if dataset == 'CMRxRecon':
                modelname = 'SSIM_{:.5f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_SSIM, Mean_MSE, epoch+1)
            else:
                modelname = 'DICE_{:.5f}_SSIM_{:.5f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_Dice, Mean_SSIM, Mean_MSE, epoch+1)
            save_checkpoint(model.state_dict(), model_dir, modelname)
            #wandb.log_model(path=model_dir, name=modelname)

            # save image
            sample_path = join(model_dir_png, 'Epoch_{:04d}-images.jpg'.format(epoch+1))
            save_flow(mov_img, fix_img, warped_mov_img, grid.permute(0, 3, 1, 2), sample_path)
            if dataset == 'CMRxRecon':
                print("epoch {:d}/{:d} - SSIM_val: {:.5f}, MSE_val: {:.6f}".format(epoch+1, epochs, Mean_SSIM, Mean_MSE))
            else:
                print("epoch {:d}/{:d} - DICE_val: {:.5f}, SSIM_val: {:.5f}, MSE_val: {:.6f}".format(epoch+1, epochs, Mean_Dice, Mean_SSIM, Mean_MSE))

            if earlyStop:    
                # stop training if metrics stop improving for three epochs (only on the first run)
                if counter_earlyStopping == 5:
                    epochs = epoch+1      # save number of epochs for other runs
                    break
                
    print('Training ended on ', time.ctime())

    #############
    ## Testing ##
    #############
    #"""
    print('\nTesting started on ', time.ctime())

    csv_name = './TestResults-{}/TestMetrics-Model_{}_Diffeo_{}_Loss_{}_Chan_{}_FT_{}-{}_Smth_{}_LR_{}_Mode_{}.csv'.format(dataset,model_name,diffeo_name,choose_loss,start_channel,FT_size[0],FT_size[1],smth_lambda,learning_rate,mode)
    f = open(csv_name, 'w')
    with f:
        if dataset == 'CMRxRecon':
            fnames = ['Image Pair','MSE','SSIM','Time','Mean MSE','Mean SSIM','Mean Time','Mean NegJ']
        else:
            fnames = ['Image Pair','Dice full','Dice no background','MSE','SSIM','Time','Mean Dice full',' Mean Dice no background','Mean MSE','Mean SSIM','Mean Time','Mean NegJ']
        writer = csv.DictWriter(f, fieldnames=fnames)
        writer.writeheader()

    model.eval()
    transform.eval()
    MSE_test = []
    SSIM_test = []
    NegJ_test=[]
    times = []
    if dataset != 'CMRxRecon':
        Dice_test_full = []
        Dice_test_noBackground = []

    for i, imagePairs in enumerate(test_generator): 
        with torch.no_grad():
            fix_img = imagePairs[0].cuda().float()
            mov_img = imagePairs[1].cuda().float()
            
            if dataset != 'CMRxRecon':
                mov_seg = imagePairs[2].cuda().float()
                fix_seg = imagePairs[3].cuda().float()
            
            start = time.time()
            
            Df_xy, __ = model(mov_img, fix_img)
            
            # get inference time
            inference_time = time.time()-start
            times.append(inference_time)
            
            # get warped image and segmentation
            grid, warped_mov_img = transform(mov_img, Df_xy.permute(0, 2, 3, 1), mod = 'nearest')
            if dataset != 'CMRxRecon':
                grid, warped_mov_seg = transform(mov_seg, Df_xy.permute(0, 2, 3, 1), mod = 'nearest')
            
            # calculate MSE, SSIM and Dice 
            if dataset != 'CMRxRecon':
                dices_temp = dice_ACDC(warped_mov_seg[0,0,:,:].cpu().numpy(),fix_seg[0,0,:,:].cpu().numpy())
                csv_Dice_full = np.mean(dices_temp)
                csv_Dice_noBackground = np.mean(dices_temp[1:3])
            csv_MSE = mean_squared_error(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy())
            csv_SSIM = structural_similarity(warped_mov_img[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy(), data_range=1)
            
            if dataset != 'CMRxRecon':
                Dice_test_full.append(csv_Dice_full)
                Dice_test_noBackground.append(csv_Dice_noBackground)
            MSE_test.append(csv_MSE)
            SSIM_test.append(csv_SSIM)
        
            hh, ww = Df_xy.shape[-2:]
            Df_xy = Df_xy.detach().cpu().numpy()
            Df_xy[:,0,:,:] = Df_xy[:,0,:,:] * hh / 2
            Df_xy[:,1,:,:] = Df_xy[:,1,:,:] * ww / 2

            jac_det = jacobian_determinant_vxm(Df_xy[0, :, :, :])
            negJ = np.sum(jac_det <= 0) / 160 / 192 * 100
            NegJ_test.append(negJ)
            
            # save test results to csv file
            f = open(csv_name, 'a')
            with f:
                writer = csv.writer(f)
                if dataset == 'CMRxRecon':
                    writer.writerow([i, csv_MSE, csv_SSIM, inference_time, '-', '-', '-', '-']) 
                else:
                    writer.writerow([i, csv_Dice_full, csv_Dice_noBackground, csv_MSE, csv_SSIM, inference_time, '-', '-', '-', '-', '-', '-']) 

    # calculate mean and stdof test metrics
    Mean_time = np.mean(times)
    Mean_MSE  = np.mean(MSE_test)
    Mean_SSIM = np.mean(SSIM_test)
    if dataset != 'CMRxRecon':
        Mean_Dice_full = np.mean(Dice_test_full)
        Mean_Dice_noBackground = np.mean(Dice_test_noBackground)
    Mean_NegJ = np.mean(NegJ_test)

    Std_MSE  = np.std(MSE_test)
    Std_SSIM = np.std(SSIM_test)
    if dataset != 'CMRxRecon':
        Std_Dice_full = np.std(Dice_test_full)
        Std_Dice_noBackground = np.std(Dice_test_noBackground)
    Std_NegJ = np.std(NegJ_test)
    
    # save test results to csv file
    f = open(csv_name, 'a')
    with f:
        writer = csv.writer(f)
        if dataset == 'CMRxRecon':
            writer.writerow(['-', '-', '-', '-', Mean_MSE, Mean_SSIM, Mean_time, Mean_NegJ]) 
        else:
            writer.writerow(['-', '-', '-', '-', '-', '-', Mean_Dice_full, Mean_Dice_noBackground, Mean_MSE, Mean_SSIM, Mean_time, Mean_NegJ]) 

    # print results
    if dataset == 'CMRxRecon':
        print('     Mean inference time: {:.4f} seconds\n     MSE: {:.6f} +- {:.6f}\n     SSIM: {:.5f} +- {:.5f}\n     DetJ<0 %: {:.4f} +- {:.4f}'.format(Mean_time, Mean_MSE, Std_MSE, Mean_SSIM, Std_SSIM, Mean_NegJ, Std_NegJ))
        wandb.log({"MSE": Mean_MSE, "SSIM": Mean_SSIM, "NegJ": Mean_NegJ, "Time": Mean_time})
    else:
        print('     Mean inference time: {:.4f} seconds\n     DICE full: {:.5f} +- {:.5f}\n     DICE no background: {:.5f} +- {:.5f}\n     MSE: {:.6f} +- {:.6f}\n     SSIM: {:.5f} +- {:.5f}\n     DetJ<0 %: {:.4f} +- {:.4f}'.format(Mean_time, Mean_Dice_full, Std_Dice_full, Mean_Dice_noBackground, Std_Dice_noBackground,  Mean_MSE, Std_MSE, Mean_SSIM, Std_SSIM, Mean_NegJ, Std_NegJ))
        wandb.log({"Dice full": Mean_Dice_full, "Dice no background": Mean_Dice_noBackground, "MSE": Mean_MSE, "SSIM": Mean_SSIM, "NegJ": Mean_NegJ, "Time": Mean_time})
    
    print('Testing ended on ', time.ctime())
    #"""
    # Mark the run as finished
    wandb.finish()   

    return epochs    

##############################################
### Helper Functions for CMRImageGenerator ###
##############################################

def readfile2numpy(file_name):
    '''
    read the data from mat and convert to numpy array
    '''
    hf = h5py.File(file_name)
    keys = list(hf.keys())
    assert len(keys) == 1, f"Expected only one key in file, got {len(keys)} instead"
    new_value = hf[keys[0]][()]
    data = new_value["real"] + 1j*new_value["imag"]

    return data  

def extract_slice(fullmulti, slice, frame, H, W):
    '''
    generate slice, reconstruct image from k-space, normalize to [0,1], interpolate to [246,512] and save image
    '''
    slice_kspace = fullmulti[frame, slice] 
    # convert to tensor
    slice_kspace = T.to_tensor(slice_kspace) 
    # Apply Inverse Fourier Transform to get the complex image  
    image = fastmri.ifft2c(slice_kspace)
    # Compute absolute value to get a real image      
    image = fastmri.complex_abs(image) 
    # combine the coil images to a coil-combined one
    image = fastmri.rss(image, dim=0)
    # normalize images have data range [0,1]
    image = normalize(image)
    # interpolate images to be the same size
    image = F.interpolate(image.unsqueeze(0).unsqueeze(0), (H, W), mode='bilinear').squeeze(0).squeeze(0)

    return image

def extract_differences(images, frames, slices, H, W):
    ''' take mean of frames and sum them over all slices of a patient  '''
    differences = torch.zeros([slices,frames-1, H, W]) 
    for slice in range(slices):
        for i in range(frames):
            if i>1:
                differences[slice,i-2,:,:] = abs(images[slice,i-1,:,:]-images[slice,i-2,:,:])

    mean_diff_frames = torch.sum(differences, dim=1)/differences.shape[1] # mean over all differences
    diffs_slices = torch.sum(mean_diff_frames, dim=0) # sum over all differences between slices

    return diffs_slices


##############################################
### Helper Functions for creating boxplots ###
##############################################

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    #plt.setp(bp['medians'], color=color)

def create_AB_boxplot(savename=None, title=None, data_A=None, data_B=None, labels=None, legend=['A','B'], figure_size=(10,4)):
    ''' Create boxplots of A and B with legend and labels on the x-axis  '''
    plt.figure(figsize=figure_size)
    bpl = plt.boxplot(data_A, positions=np.array(range(len(data_A)))*2.0-0.4, sym='', widths=0.6, patch_artist=True)
    bpr = plt.boxplot(data_B, positions=np.array(range(len(data_B)))*2.0+0.4, sym='', widths=0.6, patch_artist=True)

    #set_box_color(bpl, '#D7191C') 
    #set_box_color(bpr, '#2C7BB6')

    plt.setp(bpl["boxes"], facecolor='#D7191C')
    plt.setp(bpr["boxes"], facecolor='#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label=legend[0])
    plt.plot([], c='#2C7BB6', label=legend[1])
    plt.legend()

    plt.xticks(range(0, len(labels) * 2, 2), labels)
    plt.xlim(-2, len(labels)*2)
    plt.tight_layout()
    if title != None:
        plt.title(title, y=1.0, pad=-14)
    if savename == None:
        plt.savefig('boxplot.png')   
    else:
        plt.savefig(savename)    

def create_boxplot(savename=None, title=None, data=None, labels=None, legend=None, figure_size=(10,4), offsets=None, width=0.5, anchor_loc=None):
    ''' Create boxplots of any size with legend and labels on the x-axis  '''
    
    # set plot size
    plt.figure(figsize=figure_size)

    # check whether data was input
    if data is None:
        raise ValueError('Data is missing!!')

    # make sure that every data point has a label in the legend
    assert data.shape[0] == len(legend), f"Expected number of data arrays to be the same as number of elements in the legend, but got {data.shape[0]} and {len(legend)}"
    
    # make sure that every data point has a label in the legend
    assert data.shape[1] == len(labels), f"Expected number of data arrays to be the same as number of labels, but got {data.shape[1]} and {len(labels)}"
    
    # make sure that offsets math number of boxplots
    assert data.shape[0] == len(offsets), f"Expected number of data arrays to be the same as number of offsets, but got {data.shape[0]} and {len(offsets)}"
    
    # init colors for plots
    color = cm.rainbow(np.linspace(0, 1, data.shape[0]))

    # interate through size of data
    for i in range(data.shape[0]):
        # reformat data for boxplots (array with array containing the label scores)
        data_newFormat = []
        for j in range(data.shape[1]):
            data_newFormat.append(data[i,j,:]) 
        # create boxplot
        bp = plt.boxplot(data_newFormat, positions=np.array(range(len(data_newFormat)))*2+offsets[i], sym='', widths=width, patch_artist=True) 
        # set colors for boxes
        plt.setp(bp["boxes"], facecolor=color[i])
        # draw temporary lines and use them to create a legend
        plt.plot([], c=color[i], label=legend[i])
    
    if type(anchor_loc) == type(None):
        plt.legend() 
    else:    
        plt.legend(bbox_to_anchor=anchor_loc,loc='center') 
    plt.xticks(range(0, len(labels)*2, 2), labels)
    plt.xlim(-0.9, len(labels)*1.65) 
    plt.tight_layout()
    
    if title is not None:
        plt.title(title, y=1.0, pad=-14)
    
    if savename is None:
        plt.savefig('boxplot.png')   
    else:
        plt.savefig(savename)            

def FFT(image):
    return fftshift(fftn(ifftshift(image, dim=[-2, -1]), dim=[-2, -1]), dim=[-2, -1])

def IFFT(kspace):
    return fftshift(ifftn(ifftshift(kspace, dim=[-2, -1]), dim=[-2, -1]), dim=[-2, -1])   

def crop2D(mov, fix, u, pos, crop_size):
    """crop a given array in spatial domain"""
    x_pos = pos[0]
    y_pos = pos[1]
    window_size = (crop_size, crop_size)
    # if arrays are numpy
    if isinstance(fix, np.ndarray) and isinstance(mov, np.ndarray):
        fix_tmp = view_as_windows(fix, window_size)[x_pos, y_pos]
        mov_tmp = view_as_windows(mov, window_size)[x_pos, y_pos]
    # if they are torch.tensors
    else:
        fix = fix.squeeze().cpu().numpy()
        mov = mov.squeeze().cpu().numpy()
        fix_tmp_0 = view_as_windows(fix[0,:,:], window_size)[x_pos, y_pos]
        fix_tmp_1 = view_as_windows(fix[1,:,:], window_size)[x_pos, y_pos]
        mov_tmp_0 = view_as_windows(mov[0,:,:], window_size)[x_pos, y_pos]
        mov_tmp_1 = view_as_windows(mov[1,:,:], window_size)[x_pos, y_pos]
        fix_tmp_0 = torch.from_numpy(fix_tmp_0).unsqueeze(0).unsqueeze(0)
        fix_tmp_1 = torch.from_numpy(fix_tmp_1).unsqueeze(0).unsqueeze(0)
        mov_tmp_0 = torch.from_numpy(mov_tmp_0).unsqueeze(0).unsqueeze(0)
        mov_tmp_1 = torch.from_numpy(mov_tmp_1).unsqueeze(0).unsqueeze(0)
        fix = torch.cat([fix_tmp_0,fix_tmp_1], dim = 1)
        mov = torch.cat([mov_tmp_0,mov_tmp_1], dim = 1)
    #ref_mov = np.stack((ref_tmp, mov_tmp), axis=-1)
    #flow_out = flowCrop(u, x_pos, y_pos, crop_size)
    return mov, fix #ref_mov, flow_out
