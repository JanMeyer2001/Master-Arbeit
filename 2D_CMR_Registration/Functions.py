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
from scipy.sparse import csr_matrix
import math
import random
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

def normalize_numpy(image):
    """ expects an image as an numpy array and normalizes the values between [0,1] """
    return (image - np.min(image)) / (np.max(image) - np.min(image))

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
    masks = []  
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
                    # get masks 
                    mask = [f.path for f in scandir(join(data_path, 'FullySampled', subset, patient, slice)) if isfile(join(data_path, 'FullySampled', subset, patient, slice, f)) and f.name.startswith('Mask')]*len(list(itertools.combinations(frames_fullySampled, 2)))
                    # add masks to list
                    masks = masks + mask
    return image_pairs_fullySampled, image_pairs_subSampled, masks

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

def load_mask(path):
    mask = imread(path, as_gray=True)
    mask = torch.from_numpy(mask)
    return mask

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

def getIndixes(mask=None, density_inside=None, density_outside=None, sampling='random'):
    """ extracts indixes from mask tensor size (x,y) with sampling density inside and outside the cardiac region """
    assert type(mask) != type(None), f"Mask needs to be input into getIndixes!!"
    # sample cardiac region more than the background
    if torch.count_nonzero(mask) != 0:
        # get all non-zero positions in the mask to find the cardiac region
        positions = np.nonzero(mask.squeeze().cpu().numpy())
        crop_x = int((positions[0].max() - positions[0].min())/2)
        crop_y = int((positions[1].max() - positions[1].min())/2)
        center_x = positions[0].min() + crop_x
        center_y = positions[1].min() + crop_y
        # make mask quadratic around cardiac region
        mask[(center_x-crop_x):(center_x+crop_x),(center_y-crop_y):(center_y+crop_y)] = 1
        # init foreground for cardiac region
        foreground = torch.zeros_like(mask[(center_x-crop_x):(center_x+crop_x),(center_y-crop_y):(center_y+crop_y)])
        # init background
        background = torch.zeros_like(mask)
        if type(density_outside) != type(None):
            # undersampled background by flatting, indexing and resizing
            background = torch.flatten(background, start_dim=0, end_dim=1)
            if sampling=='random':
                indixes = torch.round(torch.rand(int(background.shape[0]/density_outside))*(background.shape[0]-1)).int()
            elif sampling=='linear':    
                indixes = torch.arange(start=0,end=background.shape[0],step=density_outside)
            else:
                print('wrong sampling method in getIndixes!!')
            background[indixes] = 1
            background = torch.unflatten(background,0,mask.shape)
            # cut out foreground
            #background[positions[0].min():positions[0].max(),positions[0].min():positions[0].max()] = 0
        if type(density_inside) != type(None):
            # undersample foreground with density_inside  
            foreground = torch.flatten(foreground, start_dim=0, end_dim=1)
            if sampling=='random':
                indixes = torch.round(torch.rand(int(foreground.shape[0]/density_inside))*(foreground.shape[0]-1)).int()
            elif sampling=='linear':    
                indixes = torch.arange(start=0,end=foreground.shape[0],step=density_inside)
            else:
                print('wrong sampling method in getIndixes!!')
            foreground[indixes] = 1
            foreground = torch.unflatten(foreground,0,mask[(center_x-crop_x):(center_x+crop_x),(center_y-crop_y):(center_y+crop_y)].shape)
            # insert foreground into background
            background[(center_x-crop_x):(center_x+crop_x),(center_y-crop_y):(center_y+crop_y)] = foreground
        # get positions of indixes
        position = np.nonzero(torch.flatten(background, start_dim=0, end_dim=1).cpu().numpy())[0]
    # if there is no segmentation available (aka. the mask is empty)
    else:
        # undersampled background by flatting, indexing and resizing
        background = torch.flatten(mask, start_dim=0, end_dim=1)
        if type(density_outside) != type(None):
            position = torch.arange(start=0,end=background.shape[0],step=density_outside)
        else:
            position = [0]    
        
    return position#, background

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
        self.paths_fullySampled, self.paths_subSampled, self.paths_masks = get_image_pairs_ACDC_ContrastiveLearning(subset='Training', data_path=data_path, subfolder=subfolder)
            
  def __len__(self):
        'total number of training samples'
        return len(self.paths_fullySampled)

  def __getitem__(self, index):
        'Loads fully sampled and subsampled image pairs'
        mov_img_fullySampled, fix_img_fullySampled  = load_image_pair_ACDC_train(self.paths_fullySampled[index][0], self.paths_fullySampled[index][1])
        mov_img_subSampled, fix_img_subSampled      = load_image_pair_ACDC_train(self.paths_subSampled[index][0], self.paths_subSampled[index][1])
        masks = load_mask(self.paths_masks[index])
        return mov_img_fullySampled, fix_img_fullySampled, mov_img_subSampled, fix_img_subSampled, masks

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
            frames = [f.path for f in scandir(join(data_path, subset, subfolder, patient, slice)) if isfile(join(data_path, subset, subfolder, patient, slice, f)) and not (f.name.find('Image_Frame') == -1)]
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
  def __init__(self, data_path, cropping, mode):
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
        if cropping == False:
            self.paths = get_image_pairs_CMRxRecon(subset='TrainingSet/Full', data_path=data_path, subfolder=subfolder)
        else:    
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
  def __init__(self, data_path, cropping, mode):
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
        if cropping == False:
            self.paths = get_image_pairs_CMRxRecon(subset='ValidationSet/Full', data_path=data_path, subfolder=subfolder)
        else:    
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
  def __init__(self, data_path, cropping, mode):
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
        if cropping == False:
            self.paths = get_image_pairs_CMRxRecon(subset='TestSet/Full', data_path=data_path, subfolder=subfolder)
        else:    
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
  def __init__(self, data_path, cropping, mode):
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
        if cropping == False:
            subset='TestSet/Full'
        else: 
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
                        frames_subSampled = [f.path for f in scandir(join(data_path, subset, subfolder, patient, slice)) if isfile(join(data_path, subset, subfolder, patient, slice, f)) and not (f.name.find('Image_Frame') == -1)]
                        frames_fullySampled = [f.path for f in scandir(join(data_path, subset, 'FullySampled', patient, slice)) if isfile(join(data_path, subset, 'FullySampled', patient, slice, f)) and not (f.name.find('Image_Frame') == -1)][0:len(frames_subSampled)]
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

        return mov_img_fullySampled, fix_img_fullySampled, mov_img_subSampled, fix_img_subSampled

class DatasetCMRxReconstruction(Data.Dataset):
    'Dataset for reconstruction of subsampled CMRxRecon data'
    def __init__(self, data_path, cropping, mode):
        'Initialization'
        # choose subfolder according to mode
        assert mode >= 1 and mode <= 3, f"Expected mode for CMRxRecon test benchmark to be one of 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 1:
            self.subfolder = 'AccFactor04'
        elif mode == 2:
            self.subfolder = 'AccFactor08' 
        elif mode == 3:
            self.subfolder = 'AccFactor10' 
        # get names and paths of training data
        if cropping == False:
            subset = 'TestSet/Full' 
        else: 
            subset = 'TestSet/Croped'
        
        self.pairs            = []    # init array for data pairs
        self.H                = 246   # image height
        self.W                = 512   # image width
        self.num_coils        = 10    # number of coils
        # get folder names for patients
        patients_fullySampled = [basename(f.path) for f in scandir(join(data_path, subset, 'FullySampled')) if f.is_dir() and not (f.name.find('P') == -1)]
        patients_subSampled   = [basename(f.path) for f in scandir(join(data_path, subset, self.subfolder)) if f.is_dir() and not (f.name.find('P') == -1)]
        
        for patient in patients_fullySampled: #[0:2]
            if patient in patients_subSampled:
                # get subfolder names for image slices
                slices_fullySampled = [basename(f.path) for f in scandir(join(data_path, subset, 'FullySampled', patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
                slices_subSampled   = [basename(f.path) for f in scandir(join(data_path, subset, self.subfolder, patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
                for slice in slices_fullySampled:
                    if slice in slices_subSampled:
                        # get all frames for each slice
                        frames_subSampled   = [f.path for f in scandir(join(data_path, subset, self.subfolder, patient, slice)) if isfile(join(data_path, subset, self.subfolder, patient, slice, f)) and not (f.name.find('Image_Frame') == -1)]
                        frames_fullySampled = [f.path for f in scandir(join(data_path, subset, 'FullySampled', patient, slice)) if isfile(join(data_path, subset, 'FullySampled', patient, slice, f))][0:len(frames_subSampled)]
                        # get k-space and coil map paths
                        k_space_data = [patient,slice]
                        #k_space_data = [f.path for f in scandir(join(data_path, subset, subfolder, patient, slice)) if isfile(join(data_path, subset, subfolder, patient, slice, f)) and not (f.name.find('k-space_Frame') == -1)]
                        coil_maps    = [f.path for f in scandir(join(data_path, subset, self.subfolder, patient, slice)) if isfile(join(data_path, subset, self.subfolder, patient, slice, f)) and not (f.name.find('SensitivityMaps_Frame') == -1)]
                        # repeat mask for correct size
                        masks = [f.path for f in scandir(join(data_path, subset, self.subfolder, patient)) if f.is_file() and not (f.name.find('Mask') == -1)] * len(frames_subSampled)
                        # add fully sampled, subsampled frames as well as corresponding k-space and coil map data to a list 
                        self.pairs.append([frames_fullySampled,frames_subSampled,masks,k_space_data,coil_maps])
                        #self.pairs = self.pairs + list(zip(frames_fullySampled,frames_subSampled,masks,k_space_data,coil_maps))

    def __len__(self):
        'number of samples'
        return len(self.pairs)

    def __getitem__(self, index):
        'Generates data for an image slice'
        num_frames  = len(self.pairs[index][0])             # number of frames
        patient     = self.pairs[index][3][0]               # number of current patient
        slice       = int(self.pairs[index][3][1][-1])      # number of current slice

        # get path for k-space
        path_origin = join('/home/jmeyer/storage/staff/ziadalhajhemid/CMRxRecon23/MultiCoil/Cine/TestSet',self.subfolder,patient)
        fullmulti   = readfile2numpy(join(path_origin, 'cine_sax.mat'),real=False)

        # init numpy arrays
        images_subsampled   = np.zeros((num_frames,self.H,self.W))
        images_fullysampled = np.zeros((num_frames,self.H,self.W))
        k_spaces            = np.zeros((1,self.num_coils,num_frames,self.H,self.W), dtype=np.complex64)
        coil_maps           = np.zeros((1,self.num_coils,num_frames,self.H,self.W), dtype=np.complex64)
        
        # fill list for all frames
        for i in range(num_frames):
            # read in image data
            images_fullysampled[i,:,:]  = imread(self.pairs[index][0][i], as_gray=True)/255
            images_subsampled[i,:,:]    = imread(self.pairs[index][1][i], as_gray=True)/255
            mask                        = imread(self.pairs[index][2][i], as_gray=True)/255
            if mask.shape[0] != 246 or mask.shape[1] != 512:
                padx  = int((246-mask.shape[0])/2)      # calculate padding for x axis
                pady  = int((512-mask.shape[1])/2)      # calculate padding for y axis
                padxy = (pady, pady, padx, padx)        # adaptive padding
                mask  = F.pad(torch.from_numpy(mask), padxy, "constant", 0).numpy()
                
            # load k-space data and coil maps (size of [C,H,W] with C being coil channels)
            k_space = fullmulti[i, slice]
            if k_space.shape[1] == 246 and k_space.shape[2] == 512:
                k_spaces[0,:,i,:,:] = k_space
            else:    
                padx  = int((246-k_space.shape[1])/2) # calculate padding for x axis
                pady  = int((512-k_space.shape[2])/2) # calculate padding for y axis
                padxy = (pady, pady, padx, padx)      # adaptive padding
                k_spaces[0,:,i,:,:] = F.pad(torch.from_numpy(k_space), padxy, "constant", 0).numpy()
            coil_map = torch.load(self.pairs[index][4][i])    
            if coil_map.shape[1] == 246 and coil_map.shape[2] == 512:
                coil_maps[0,:,i,:,:] = coil_map
            else:    
                coil_map_real = F.interpolate(torch.real(torch.from_numpy(coil_map)).unsqueeze(0), (246,512), mode='nearest').squeeze(0)
                coil_map_imag = F.interpolate(torch.imag(torch.from_numpy(coil_map)).unsqueeze(0), (246,512), mode='nearest').squeeze(0)
                coil_maps[0,:,i,:,:] = torch.complex(coil_map_real, coil_map_imag).numpy()

        # take one mask and repeat it to correct size
        mask_frames = mask
        mask_frames = np.repeat(mask_frames[np.newaxis,:,:], self.num_coils, axis=0)
        mask_frames = np.repeat(mask_frames[np.newaxis,:,np.newaxis,:,:], num_frames, axis=2)
        
        return images_fullysampled, images_subsampled, mask_frames, k_spaces, coil_maps
    
class DatasetMotionReconstruction_LineSwapping(Data.Dataset):
    'Dataset of subsampled CMRxRecon data that is artificially motion-corrupted via k-space line swapping'
    def __init__(self, data_path, cropping, mode, z):
        'Initialization'
        # choose subfolder according to mode
        assert mode >= 1 and mode <= 3, f"Expected mode for CMRxRecon test benchmark to be one of 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 1:
            self.subfolder = 'AccFactor04'
        elif mode == 2:
            self.subfolder = 'AccFactor08' 
        elif mode == 3:
            self.subfolder = 'AccFactor10' 
        # get names and paths of training data
        if cropping == False:
            subset = 'TestSet/Full' 
        else: 
            subset = 'TestSet/Croped'
        
        self.pairs            = []    # init array for data pairs
        self.z                = z     # number of lines that will be swapped to simulate motion
        self.H                = 246   # image height
        self.W                = 512   # image width
        
        # get folder names for patients
        patients_fullySampled = [basename(f.path) for f in scandir(join(data_path, subset, 'FullySampled')) if f.is_dir() and not (f.name.find('P') == -1)]
        
        for patient in patients_fullySampled: #[0:2]
            # get subfolder names for image slices
            slices_fullySampled = [basename(f.path) for f in scandir(join(data_path, subset, 'FullySampled', patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
            for slice in slices_fullySampled:
                frames_fullySampled = [f.path for f in scandir(join(data_path, subset, 'FullySampled', patient, slice)) if isfile(join(data_path, subset, 'FullySampled', patient, slice, f))]
                # get k-space and coil map paths
                k_space_data = [patient,slice]
                #k_space_data = [f.path for f in scandir(join(data_path, subset, subfolder, patient, slice)) if isfile(join(data_path, subset, subfolder, patient, slice, f)) and not (f.name.find('k-space_Frame') == -1)]
                coil_maps    = [f.path for f in scandir(join(data_path, subset, self.subfolder, patient, slice)) if isfile(join(data_path, subset, self.subfolder, patient, slice, f)) and not (f.name.find('SensitivityMaps_Frame') == -1)]
                # repeat mask for correct size
                masks = [f.path for f in scandir(join(data_path, subset, self.subfolder, patient)) if f.is_file() and not (f.name.find('Mask') == -1)] # * len(frames_fullySampled)
                # add fully sampled, subsampled frames as well as corresponding k-space and coil map data to a list 
                self.pairs.append([frames_fullySampled,masks,k_space_data,coil_maps])
                #self.pairs = self.pairs + list(zip(frames_fullySampled,frames_subSampled,masks,k_space_data,coil_maps))

    def __len__(self):
        'number of samples'
        return len(self.pairs)

    def __getitem__(self, index):
        'Generates data for an image slice'
        patient     = self.pairs[index][2][0]               # number of current patient
        slice       = int(self.pairs[index][2][1][-1])      # number of current slice

        # get path for fully sampled k-space
        path_origin = join('/home/jmeyer/storage/staff/ziadalhajhemid/CMRxRecon23/MultiCoil/Cine/TestSet/FullSample',patient)
        fullmulti   = readfile2numpy(join(path_origin, 'cine_sax.mat'),real=False)
        [num_frames, num_slices, num_coils, ny, nx] = fullmulti.shape

        # init numpy arrays and torch tensors
        images_subsampled   = np.zeros((num_frames,self.H,self.W))
        images_fullysampled = np.zeros((num_frames,self.H,self.W))
        k_spaces            = np.zeros((1,num_coils,num_frames,self.H,self.W), dtype=np.complex64)
        coil_maps           = np.zeros((1,num_coils,num_frames,self.H,self.W), dtype=np.complex64)
        
        for frame in range(num_frames):
            # read in image data
            images_fullysampled[frame,:,:] = imread(self.pairs[index][0][frame], as_gray=True)/255
                
            # load k-space data and coil maps (size of [C,H,W] with C being coil channels)
            k_space = fullmulti[frame, slice]
            if k_space.shape[1] == 246 and k_space.shape[2] == 512:
                k_spaces[0,:,frame,:,:] = k_space
            else:    
                padx  = int((246-k_space.shape[1])/2) # calculate padding for x axis
                pady  = int((512-k_space.shape[2])/2) # calculate padding for y axis
                padxy = (pady, pady, padx, padx)      # adaptive padding
                k_spaces[0,:,frame,:,:] = F.pad(torch.from_numpy(k_space), padxy, "constant", 0).numpy()
            
            coil_map = torch.load(self.pairs[index][3][frame])    
            if coil_map.shape[1] == 246 and coil_map.shape[2] == 512:
                coil_maps[0,:,frame,:,:] = coil_map
            else:    
                coil_map_real = F.interpolate(torch.real(torch.from_numpy(coil_map)).unsqueeze(0), (246,512), mode='nearest').squeeze(0)
                coil_map_imag = F.interpolate(torch.imag(torch.from_numpy(coil_map)).unsqueeze(0), (246,512), mode='nearest').squeeze(0)
                coil_maps[0,:,frame,:,:] = torch.complex(coil_map_real, coil_map_imag).numpy()
        
        # add motion artifacts to the k-space data via line swapping
        k_spaces_motion = k_spaces
        for frame in range(num_frames):
            # get random numbers for the original line, the new line to replace it and the frame to use
            original_lines  = random.sample(range(self.H), self.z)
            new_lines       = random.sample(range(self.H), self.z)
            for i, line in enumerate(original_lines):
                new_frame = random.sample(range(num_frames), 1)
                k_spaces_motion[0,:,frame,line,:] = k_spaces[0,:,new_frame,new_lines[i],:] 

        # take one mask and repeat it to correct size
        mask = imread(self.pairs[index][1][0], as_gray=True)/255
        if mask.shape[0] != 246 or mask.shape[1] != 512:
            padx  = int((246-mask.shape[0])/2)      # calculate padding for x axis
            pady  = int((512-mask.shape[1])/2)      # calculate padding for y axis
            padxy = (pady, pady, padx, padx)        # adaptive padding
            mask  = F.pad(torch.from_numpy(mask), padxy, "constant", 0).numpy()
        mask = np.repeat(mask[np.newaxis,:,:], num_coils, axis=0)
        mask = np.repeat(mask[np.newaxis,:,np.newaxis,:,:], num_frames, axis=2)
        
        # subsampling motion-corrupted k-spaces
        k_spaces_motion_subsampled = k_spaces_motion * mask

        # get subsampled images from the motion-corrupted k-space data
        image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(T.to_tensor(k_spaces_motion_subsampled[0,:,:,:,:]))), dim=0)
        images_subsampled = normalize(image)
        
        return images_fullysampled, images_subsampled, mask, k_spaces_motion_subsampled, coil_maps

class DatasetMotionReconstruction_LungMovement(Data.Dataset):
    'Dataset of subsampled CMRxRecon data that is artificially motion-corrupted with simulated lung movement'
    def __init__(self, data_path, cropping, mode, transform, L):
        'Initialization'
        # choose subfolder according to mode
        assert mode >= 1 and mode <= 3, f"Expected mode for CMRxRecon test benchmark to be one of 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
        if mode == 1:
            self.subfolder = 'AccFactor04'
        elif mode == 2:
            self.subfolder = 'AccFactor08' 
        elif mode == 3:
            self.subfolder = 'AccFactor10' 
        # get names and paths of training data
        if cropping == False:
            subset = 'TestSet/Full' 
        else: 
            subset = 'TestSet/Croped'
        
        self.pairs            = []                  # init array for data pairs
        self.H                = 246                 # image height
        self.W                = 512                 # image width
        self.transform        = transform           # select transformation
        self.L                = L                   # number of channels with synthetic lung movement corruption
        
        # get folder names for patients
        patients_fullySampled = [basename(f.path) for f in scandir(join(data_path, subset, 'FullySampled')) if f.is_dir() and not (f.name.find('P') == -1)]
        
        for patient in patients_fullySampled: #[0:2]
            # get subfolder names for image slices
            slices_fullySampled = [basename(f.path) for f in scandir(join(data_path, subset, 'FullySampled', patient)) if f.is_dir() and not (f.name.find('Slice') == -1)]
            for slice in slices_fullySampled:
                frames_fullySampled = [f.path for f in scandir(join(data_path, subset, 'FullySampled', patient, slice)) if isfile(join(data_path, subset, 'FullySampled', patient, slice, f))]
                # get k-space and coil map paths
                k_space_data = [patient,slice]
                #k_space_data = [f.path for f in scandir(join(data_path, subset, subfolder, patient, slice)) if isfile(join(data_path, subset, subfolder, patient, slice, f)) and not (f.name.find('k-space_Frame') == -1)]
                coil_maps    = [f.path for f in scandir(join(data_path, subset, self.subfolder, patient, slice)) if isfile(join(data_path, subset, self.subfolder, patient, slice, f)) and not (f.name.find('SensitivityMaps_Frame') == -1)]
                # repeat mask for correct size
                masks = [f.path for f in scandir(join(data_path, subset, self.subfolder, patient)) if f.is_file() and not (f.name.find('Mask') == -1)] # * len(frames_fullySampled)
                # add fully sampled, subsampled frames as well as corresponding k-space and coil map data to a list 
                self.pairs.append([frames_fullySampled,masks,k_space_data,coil_maps])
                #self.pairs = self.pairs + list(zip(frames_fullySampled,frames_subSampled,masks,k_space_data,coil_maps))

    def __len__(self):
        'number of samples'
        return len(self.pairs)

    def __getitem__(self, index):
        'Generates data for an image slice'
        patient     = self.pairs[index][2][0]               # number of current patient
        slice       = int(self.pairs[index][2][1][-1])      # number of current slice

        # get path for fully sampled k-space
        path_origin = join('/home/jmeyer/storage/staff/ziadalhajhemid/CMRxRecon23/MultiCoil/Cine/TestSet/FullSample',patient)
        fullmulti   = readfile2numpy(join(path_origin, 'cine_sax.mat'),real=False)
        [num_frames, num_slices, num_coils, ny, nx] = fullmulti.shape

        # init numpy arrays and torch tensors
        images_subsampled   = np.zeros((self.L+1,int(num_frames/2),self.H,self.W))
        images_fullysampled = np.zeros((int(num_frames/2),self.H,self.W))
        k_spaces            = torch.zeros(1,num_coils,int(num_frames/2),self.H,self.W,dtype=torch.complex64)
        coil_maps           = torch.zeros(1,num_coils,int(num_frames/2)*(self.L+1),self.H,self.W,dtype=torch.complex64)
        k_spaces_motion     = torch.zeros(self.L+1,num_coils,int(num_frames/2),self.H,self.W,dtype=torch.complex64) 
        
        # select every second frame
        frames_selected = np.arange(0,num_frames,2)

        # fill list for all selected frames
        for i, frame in enumerate(frames_selected):
            # read in image data
            images_fullysampled[i,:,:] = imread(self.pairs[index][0][frame], as_gray=True)/255
                
            # load k-space data and coil maps (size of [C,H,W] with C being coil channels)
            k_space = fullmulti[frame, slice]
            if k_space.shape[1] == 246 and k_space.shape[2] == 512:
                k_spaces[0,:,i,:,:] = torch.from_numpy(k_space)
            else:    
                padx  = int((246-k_space.shape[1])/2) # calculate padding for x axis
                pady  = int((512-k_space.shape[2])/2) # calculate padding for y axis
                padxy = (pady, pady, padx, padx)      # adaptive padding
                k_spaces[0,:,i,:,:] = F.pad(torch.from_numpy(k_space), padxy, "constant", 0)
            
            coil_map = torch.load(self.pairs[index][3][frame])    
            if coil_map.shape[1] == 246 and coil_map.shape[2] == 512:
                coil_map = np.repeat(coil_map[:,np.newaxis,:,:], self.L+1, axis=1)
                coil_maps[0,:,(i*(self.L+1)):((i+1)*(self.L+1)),:,:] = torch.from_numpy(coil_map)
            else:    
                coil_map_real   = F.interpolate(torch.real(torch.from_numpy(coil_map)).unsqueeze(0), (246,512), mode='nearest').squeeze(0)
                coil_map_imag   = F.interpolate(torch.imag(torch.from_numpy(coil_map)).unsqueeze(0), (246,512), mode='nearest').squeeze(0)
                coil_maps_inter = torch.complex(coil_map_real, coil_map_imag)
                coil_map        = np.repeat(coil_maps_inter.numpy()[:,np.newaxis,:,:], self.L+1, axis=1)
                coil_maps[0,:,(i*(self.L+1)):((i+1)*(self.L+1)),:,:] = torch.from_numpy(coil_map)
        
        # take one mask and repeat it to correct size
        mask = imread(self.pairs[index][1][0], as_gray=True)/255
        if mask.shape[0] != 246 or mask.shape[1] != 512:
            padx  = int((246-mask.shape[0])/2)      # calculate padding for x axis
            pady  = int((512-mask.shape[1])/2)      # calculate padding for y axis
            padxy = (pady, pady, padx, padx)        # adaptive padding
            mask  = F.pad(torch.from_numpy(mask), padxy, "constant", 0).numpy()
        mask = np.repeat(mask[np.newaxis,:,:], num_coils, axis=0)
        mask = np.repeat(mask[np.newaxis,:,np.newaxis,:,:], int(num_frames/2)*(self.L+1), axis=2)
        
        # subsample and add motion artifacts to the k-space data 
        k_spaces_motion[0,...] = k_spaces * mask[:,:,0:len(frames_selected),:,:]
        for k, frame in enumerate(frames_selected):
            for i in range(self.L+1):
                if i == 0:
                    # reconstruct images
                    image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(torch.view_as_real(k_spaces_motion[0,:,k,:,:]))), dim=0)
                    images_subsampled[i,k,:,:] = normalize(image)
                else:
                    # get and save random displacement
                    random_disp = random_smooth_Displacement((self.H,self.W),0.1)
                    # get images for all coils
                    coil_images = fastmri.complex_abs(fastmri.ifft2c(torch.view_as_real(k_spaces_motion[0,:,k,:,:])))
                    # motion corrupt the coil images
                    grid, coil_images_motion = self.transform(coil_images.unsqueeze(0).float(),torch.from_numpy(random_disp).unsqueeze(0).float())
                    # save rss image
                    images_subsampled[i,k,:,:] = normalize(fastmri.rss(coil_images_motion.squeeze(), dim=0)) 
                    # convert back to obtain motion corrupted k-space
                    coil_images = torch.zeros(num_coils,self.H,self.W,2)
                    coil_images[:,:,:,0] = coil_images_motion.squeeze()
                    k_spaces_motion[i,:,k,:,:] = torch.view_as_complex(fastmri.fft2c(coil_images))
        #reshape motion k-space to the correct dimensions
        k_spaces_motion = torch.reshape(k_spaces_motion, (num_coils,int(num_frames/2)*(self.L+1),self.H,self.W)).unsqueeze(0)

        return images_fullysampled, images_subsampled, mask, k_spaces_motion, coil_maps

def getExampleMotionCorruptedData(folder,patient,slice,H,W,L,transform):
    # get path for fully sampled k-space
    path_rawData    = '/home/jmeyer/storage/staff/ziadalhajhemid/CMRxRecon23/MultiCoil/Cine/TestSet/'
    path_fullsample = join(path_rawData,'FullSample',patient)
    fullmulti       = readfile2numpy(join(path_fullsample, 'cine_sax.mat'),real=False)
    [num_frames, num_slices, num_coils, ny, nx] = fullmulti.shape
    slice_num       = int(slice[-1])      # number of current slice

    # get paths for fully sampled ground truth images, masks and coils
    data_path                = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/TestSet/Full/'
    paths_images_groundtruth = [f.path for f in scandir(join(data_path, 'FullySampled', patient, slice)) if isfile(join(data_path, 'FullySampled', patient, slice, f))]
    paths_coils              = [f.path for f in scandir(join(data_path, folder, patient, slice)) if isfile(join(data_path, folder, patient, slice, f)) and not (f.name.find('SensitivityMaps_Frame') == -1)]
    path_masks               = [f.path for f in scandir(join(data_path, folder, patient)) if f.is_file() and not (f.name.find('Mask') == -1)]

    # init numpy arrays and torch tensors
    images_subsampled   = torch.zeros((L+1,int(num_frames/2),H,W))
    images_fullysampled = torch.zeros((int(num_frames/2),H,W))
    k_spaces            = torch.zeros(1,num_coils,int(num_frames/2),H,W,dtype=torch.complex64)
    coil_maps           = torch.zeros(1,num_coils,int(num_frames/2)*(L+1),H,W,dtype=torch.complex64)
    k_spaces_motion     = torch.zeros(L+1,num_coils,int(num_frames/2),H,W,dtype=torch.complex64) 
    
    # select every second frame
    frames_selected = np.arange(0,num_frames,2)

    # fill list for all selected frames
    for i, frame in enumerate(frames_selected):
        # read in image data
        images_fullysampled[i,:,:] = torch.from_numpy(imread(paths_images_groundtruth[frame], as_gray=True)/255)
            
        # load k-space data and coil maps (size of [C,H,W] with C being coil channels)
        k_space = fullmulti[frame, slice_num]
        if k_space.shape[1] == 246 and k_space.shape[2] == 512:
            k_spaces[0,:,i,:,:] = torch.from_numpy(k_space)
        else:    
            padx  = int((246-k_space.shape[1])/2) # calculate padding for x axis
            pady  = int((512-k_space.shape[2])/2) # calculate padding for y axis
            padxy = (pady, pady, padx, padx)      # adaptive padding
            k_spaces[0,:,i,:,:] = F.pad(torch.from_numpy(k_space), padxy, "constant", 0)
        
        coil_map = torch.load(paths_coils[frame])    
        if coil_map.shape[1] == 246 and coil_map.shape[2] == 512:
            coil_map = np.repeat(coil_map[:,np.newaxis,:,:], L+1, axis=1)
            coil_maps[0,:,(i*(L+1)):((i+1)*(L+1)),:,:] = torch.from_numpy(coil_map)
        else:    
            coil_map_real   = F.interpolate(torch.real(torch.from_numpy(coil_map)).unsqueeze(0), (246,512), mode='nearest').squeeze(0)
            coil_map_imag   = F.interpolate(torch.imag(torch.from_numpy(coil_map)).unsqueeze(0), (246,512), mode='nearest').squeeze(0)
            coil_maps_inter = torch.complex(coil_map_real, coil_map_imag)
            coil_map        = np.repeat(coil_maps_inter.numpy()[:,np.newaxis,:,:], L+1, axis=1)
            coil_maps[0,:,(i*(L+1)):((i+1)*(L+1)),:,:] = torch.from_numpy(coil_map)
    
    # take one mask and repeat it to correct size
    mask = imread(path_masks[0])/255
    if mask.shape[0] != 246 or mask.shape[1] != 512:
        padx  = int((246-mask.shape[0])/2)      # calculate padding for x axis
        pady  = int((512-mask.shape[1])/2)      # calculate padding for y axis
        padxy = (pady, pady, padx, padx)        # adaptive padding
        mask  = F.pad(torch.from_numpy(mask), padxy, "constant", 0).numpy()
    mask = np.repeat(mask[np.newaxis,:,:], num_coils, axis=0)
    mask = torch.from_numpy(np.repeat(mask[np.newaxis,:,np.newaxis,:,:], int(num_frames/2)*(L+1), axis=2))
    
    # subsample and add motion artifacts to the k-space data 
    k_spaces_motion[0,...] = k_spaces * mask[:,:,0:len(frames_selected),:,:]
    for k, frame in enumerate(frames_selected):
        for i in range(L+1):
            if i == 0:
                # reconstruct images
                image = fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(torch.view_as_real(k_spaces_motion[0,:,k,:,:]))), dim=0)
                images_subsampled[i,k,:,:] = normalize(image)
            else:
                # get and save random displacement
                random_disp = random_smooth_Displacement((H,W),0.1)
                # get images for all coils
                coil_images = fastmri.complex_abs(fastmri.ifft2c(torch.view_as_real(k_spaces_motion[0,:,k,:,:])))
                # motion corrupt the coil images
                grid, coil_images_motion = transform(coil_images.unsqueeze(0).float(),torch.from_numpy(random_disp).unsqueeze(0).float())
                # save rss image
                images_subsampled[i,k,:,:] = normalize(fastmri.rss(coil_images_motion.squeeze(), dim=0)) 
                # convert back to obtain motion corrupted k-space
                coil_images = torch.zeros(num_coils,H,W,2)
                coil_images[:,:,:,0] = coil_images_motion.squeeze()
                k_spaces_motion[i,:,k,:,:] = torch.view_as_complex(fastmri.fft2c(coil_images))
    #reshape motion k-space to the correct dimensions
    k_spaces_motion = torch.reshape(k_spaces_motion, (num_coils,int(num_frames/2)*(L+1),H,W)).unsqueeze(0)

    return images_fullysampled, images_subsampled, mask, k_spaces_motion, coil_maps


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

def readfile2numpy(file_name,real=True):
    '''
    read the data from mat and convert to numpy array
    '''
    hf = h5py.File(file_name)
    keys = list(hf.keys())
    assert len(keys) == 1, f"Expected only one key in file, got {len(keys)} instead"
    new_value = hf[keys[0]][()]
    if real == True:
        data = new_value
    else:    
        data = new_value["real"] + 1j*new_value["imag"]

    return data  

def extract_mask(fullmulti, slice, frame, H, W):
    '''
    generate slice, get mask, interpolate to [246,512] and save image
    '''
    slice_kspace = fullmulti[frame, slice] 
    # convert to tensor
    slice_kspace = T.to_tensor(slice_kspace)
    # interpolate images to be the same size
    image = F.interpolate(image.unsqueeze(0).unsqueeze(0), (H, W), mode='bilinear').squeeze(0).squeeze(0)

    return image

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


################################################################
### Helper Functions for creating boxplots and other figures ###
################################################################

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
        #plt.plot([], c=color[i], label=legend[i])
    """
    if type(anchor_loc) == type(None):
        plt.legend() 
    else:    
        plt.legend(bbox_to_anchor=anchor_loc,loc='center') 
    """
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

def display_and_compare_images(ground_truth, method_images, method_titles, metrics, mode, boldpos, figname="test.png"):
    """ display images (numpy arrays) with metrics"""
    num_methods = len(method_images)
    fig, axes = plt.subplots(2, num_methods + 1, figsize=(20, 5))
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0.0, top=1)

    H, W = ground_truth.shape
    crop = [int(H / 4), int(W / 4)]
    ground_truth_crop = ground_truth[crop[0] : -crop[0], crop[1] : -crop[1]]
    #ground_truth_crop = normalize_numpy(ground_truth_crop)

    #ground_truth = normalize_numpy(ground_truth)
    # Display ground truth
    axes[0, 0].imshow(
        #ground_truth[crop_glob:-crop_glob, crop_glob:-crop_glob],
        ground_truth_crop,
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[0, 0].set_title("Ground Truth", fontsize=24, fontfamily="serif")
    axes[0, 0].margins(0, 0)
    axes[0, 0].axis("off")

    # Compute and display metrics for ground truth (no errors)
    axes[0, 0].axis("off")
    empyt_img = np.ones_like(ground_truth)
    im = axes[1, 0].imshow(empyt_img, cmap="magma", alpha=0.0)
    axes[1, 0].axis("off")
    # Text for mode of the figure
    if mode == 1:
        text = "R=4"
    elif mode == 2:
        text = "R=8"
    elif mode == 3:
        text = "R=10"
    axes[1, 0].set_title(text, fontsize=50, weight="bold", fontfamily="serif", y=0.57)

    # Loop through the method images
    for i in range(num_methods):
        # Display the method image
        method_image = method_images[i]
        method_img_crop = method_image[crop[0] : -crop[0], crop[1] : -crop[1]]
        #method_img_crop = normalize_numpy(method_img_crop)
        # Normalize contrast for better visualization
        # method_img_eq = exposure.equalize_hist(method_img)
        #method_image = normalize_numpy(method_image)
        im = axes[0, i + 1].imshow(
            #method_image[crop_glob:-crop_glob, crop_glob:-crop_glob],
            method_img_crop,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        if i == boldpos:
            axes[0, i + 1].set_title(
                method_titles[i], fontsize=24, weight="bold", fontfamily="serif"
            )
        else:
            axes[0, i + 1].set_title(method_titles[i], fontsize=24, fontfamily="serif")
        axes[0, i + 1].margins(0, 0)
        axes[0, i + 1].axis("off")

        # display metrics for each method image 
        info_text = f"% HaarPSI: {metrics[i,0]:.2f}  PSNR: {metrics[i,1]:.2f}\n% SSIM: {metrics[i,2]:.2f}  MSE (e-3): {metrics[i,3]:.2f}"
        text = axes[0, i + 1].text(
            10,
            156,
            info_text,
            fontsize=14,
            color="black",
            fontfamily="serif",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
        )
        axes[0, i + 1].margins(0, 0)
        axes[0, i + 1].axis("off")

        # Compute and display the difference (error) between method image and ground truth
        #error_img = np.abs(ground_truth[crop_glob:-crop_glob, crop_glob:-crop_glob]-method_image[crop_glob:-crop_glob, crop_glob:-crop_glob])
        error_img = np.abs(ground_truth_crop-method_img_crop)
        cax = axes[1, i + 1].imshow(error_img, cmap="cividis", vmin=0, vmax=0.1)
        axes[1, i + 1].axis("off")

    sm = cm.ScalarMappable(cmap="cividis")  # , norm=plt.Normalize(vmin=0, vmax=0.2))
    sm.set_array([])
    divider = make_axes_locatable(axes[1, 0])
    cax = divider.append_axes("bottom", size="12%", pad=0.1)  # 0.10
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", cmap="cividis")
    # set ticks to "min", "max"
    cbarticks = ["min", "max"]
    # but the axis label o top of the colorbar
    cbar.set_ticks([0, 1.0])
    cbar.set_ticklabels(cbarticks, fontsize=20)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.tick_top()
    cbar.ax.set_xlabel("Error", rotation=0, fontsize=20, labelpad=0, fontfamily="serif")
    cbar.ax.tick_params(axis="both", labelsize=16)
    cbar.ax.tick_params(labelsize="large")

    plt.tight_layout(h_pad=0.1, w_pad=0.3, pad=0)
    plt.show()
    fig.savefig(figname, dpi=600, pad_inches=0.05, bbox_inches="tight")

#############
## ESPIRiT ##
#############

def espirit(X, k, r, t, c):
    """
    Derives the ESPIRiT operator.

    Arguments:
      X: Multi channel k-space data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric 
         dimensions and (nc) is the channel dimension.
      k: Parameter that determines the k-space kernel size. If X has dimensions (1, 256, 256, 8), then the kernel 
         will have dimensions (1, k, k, 8)
      r: Parameter that determines the calibration region size. If X has dimensions (1, 256, 256, 8), then the 
         calibration region will have dimensions (1, r, r, 8)
      t: Parameter that determines the rank of the auto-calibration matrix (A). Singular values below t times the
         largest singular value are set to zero.
      c: Crop threshold that determines eigenvalues "=1".
    Returns:
      maps: This is the ESPIRiT operator. It will have dimensions (sx, sy, sz, nc, nc) with (sx, sy, sz, :, idx)
            being the idx'th set of ESPIRiT maps.
    """
    
    fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax) 
    ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 

    sx = np.shape(X)[0]
    sy = np.shape(X)[1]
    sz = np.shape(X)[2]
    nc = np.shape(X)[3]

    sxt = (sx//2-r//2, sx//2+r//2) if (sx > 1) else (0, 1)
    syt = (sy//2-r//2, sy//2+r//2) if (sy > 1) else (0, 1)
    szt = (sz//2-r//2, sz//2+r//2) if (sz > 1) else (0, 1)

    # Extract calibration region.    
    C = X[sxt[0]:sxt[1], syt[0]:syt[1], szt[0]:szt[1], :].astype(np.complex64)

    # Construct Hankel matrix.
    p = (sx > 1) + (sy > 1) + (sz > 1)
    A = np.zeros([(r-k+1)**p, k**p * nc]).astype(np.complex64)

    idx = 0
    for xdx in range(max(1, C.shape[0] - k + 1)):
      for ydx in range(max(1, C.shape[1] - k + 1)):
        for zdx in range(max(1, C.shape[2] - k + 1)):
          # numpy handles when the indices are too big
          block = C[xdx:xdx+k, ydx:ydx+k, zdx:zdx+k, :].astype(np.complex64) 
          A[idx, :] = block.flatten()
          idx = idx + 1

    # Take the Singular Value Decomposition.
    U, S, VH = np.linalg.svd(A, full_matrices=True)
    V = VH.conj().T

    # Select kernels.
    n = np.sum(S >= t * S[0])
    V = V[:, 0:n]

    kxt = (sx//2-k//2, sx//2+k//2) if (sx > 1) else (0, 1)
    kyt = (sy//2-k//2, sy//2+k//2) if (sy > 1) else (0, 1)
    kzt = (sz//2-k//2, sz//2+k//2) if (sz > 1) else (0, 1)

    # Reshape into k-space kernel, flips it and takes the conjugate
    kernels = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    kerdims = [(sx > 1) * k + (sx == 1) * 1, (sy > 1) * k + (sy == 1) * 1, (sz > 1) * k + (sz == 1) * 1, nc]
    for idx in range(n):
        kernels[kxt[0]:kxt[1],kyt[0]:kyt[1],kzt[0]:kzt[1], :, idx] = np.reshape(V[:, idx], kerdims)

    # Take the iucfft
    axes = (0, 1, 2)
    kerimgs = np.zeros(np.append(np.shape(X), n)).astype(np.complex64)
    for idx in range(n):
        for jdx in range(nc):
            ker = kernels[::-1, ::-1, ::-1, jdx, idx].conj()
            kerimgs[:,:,:,jdx,idx] = fft(ker, axes) * np.sqrt(sx * sy * sz)/np.sqrt(k**p)

    # Take the point-wise eigenvalue decomposition and keep eigenvalues greater than c
    maps = np.zeros(np.append(np.shape(X), nc)).astype(np.complex64)
    for idx in range(0, sx):
        for jdx in range(0, sy):
            for kdx in range(0, sz):

                Gq = kerimgs[idx,jdx,kdx,:,:]

                u, s, vh = np.linalg.svd(Gq, full_matrices=True)
                for ldx in range(0, nc):
                    if (s[ldx]**2 > c):
                        maps[idx, jdx, kdx, :, ldx] = u[:, ldx]

    return maps

def espirit_proj(x, esp):
    """
    Construct the projection of multi-channel image x onto the range of the ESPIRiT operator. Returns the inner
    product, complete projection and the null projection.

    Arguments:
      x: Multi channel image data. Expected dimensions are (sx, sy, sz, nc), where (sx, sy, sz) are volumetric 
         dimensions and (nc) is the channel dimension.
      esp: ESPIRiT operator as returned by function: espirit

    Returns:
      ip: This is the inner product result, or the image information in the ESPIRiT subspace.
      proj: This is the resulting projection. If the ESPIRiT operator is E, then proj = E E^H x, where H is 
            the hermitian.
      null: This is the null projection, which is equal to x - proj.
    """
    ip = np.zeros(x.shape).astype(np.complex64)
    proj = np.zeros(x.shape).astype(np.complex64)
    for qdx in range(0, esp.shape[4]):
        for pdx in range(0, esp.shape[3]):
            ip[:, :, :, qdx] = np.add(ip[:, :, :, qdx],x[:, :, :, pdx]) * esp[:, :, :, pdx, qdx].conj()

    for qdx in range(0, esp.shape[4]):
        for pdx in range(0, esp.shape[3]):
          proj[:, :, :, pdx] = np.add(proj[:, :, :, pdx],ip[:, :, :, qdx]) * esp[:, :, :, pdx, qdx]

    return (ip, proj, x - proj)

def rmse(pred, gt):
    return np.linalg.norm(pred.flatten() - gt.flatten()) ** 2 / np.linalg.norm(gt.flatten()) ** 2


def total_variation(image):
    """
    Compute the Total Variation (TV) of the image.
    """
    tv_h = torch.abs(image[ 1:, :] - image[:-1, :]).sum()
    tv_w = torch.abs(image[:, 1:] - image[:, :-1]).sum()
    return tv_h + tv_w

def reconstruct_denoised_image(reconstructed_images, k_space, flow_fields, sensitivity_maps, undersampling_mask, num_frames, num_repeats, coils, height, width, lr=1e-3, num_iters=100):
    """
    Reconstruct denoised images with TV regularization.
    Args:
        k_space: Input undersampled k-space, shape (num_frames * num_repeats, coils, height, width).
        flow_fields: Flow fields, shape (num_frames * num_repeats, height, width, 2).
        sensitivity_maps: Sensitivity maps, shape (coils, height, width).
        undersampling_mask: Binary undersampling mask, shape (height, width).
        num_frames: Number of cardiac frames.
        num_repeats: Number of respiratory motion repeats.
        lr: Learning rate for the Adam optimizer.
        num_iters: Number of optimization iterations.
    Returns:
        Reconstructed denoised images: shape (num_frames, height, width).
    """
    device = k_space.device

    # Initialize reconstructed images (num_frames, height, width)
    #reconstructed_images = torch.zeros(num_frames, height, width, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([reconstructed_images], lr=lr)

    for iter in range(num_iters):
        optimizer.zero_grad()

        # Apply flow fields and reconstruct x * n images
        deformed_images = []
        for i in range(num_frames):
            base_image = reconstructed_images[i]
            for j in range(num_repeats):
                flow_field = flow_fields[i * num_repeats + j]
                grid = (torch.stack(torch.meshgrid(torch.linspace(-1, 1, height, device=device),torch.linspace(-1, 1, width, device=device),),dim=-1,) + flow_field)
                deformed_image = F.grid_sample(base_image.unsqueeze(0).unsqueeze(0).float(),grid.unsqueeze(0).float(),mode="bilinear",padding_mode="zeros",align_corners=True,).squeeze()
                deformed_images.append(deformed_image)
        deformed_images = torch.stack(deformed_images)  # Shape: (num_frames * num_repeats, height, width)

        # Apply sensitivity maps
        predicted_k_space = []
        for i in range(num_frames * num_repeats):
            image = deformed_images[i].unsqueeze(0)
            coil_images = image * sensitivity_maps  # Shape: (coils, height, width)
            k_space_pred = fft2(coil_images)  # Fourier Transform
            undersampled_k_space_pred = k_space_pred * undersampling_mask
            predicted_k_space.append(undersampled_k_space_pred)
        predicted_k_space = torch.stack(predicted_k_space)  # Shape: (num_frames * num_repeats, coils, height, width)

        # Compute data consistency loss
        data_consistency_loss = torch.norm(predicted_k_space - k_space) ** 2

        # Compute total variation loss
        tv_loss = sum(total_variation(reconstructed_images[i]) for i in range(num_frames))

        # Total loss
        total_loss = data_consistency_loss + tv_loss #

        # Backward and optimize
        total_loss.requires_grad = True
        total_loss.backward(retain_graph=True)
        optimizer.step()
        
        if iter == 0:
            plt.subplot(2, 2, 1)
            plt.imshow(reconstructed_images[0,:,:].cpu().detach().numpy(), cmap='gray')  
            plt.axis('off')
            plt.title('Iter 1')
        elif iter == 9:
            plt.subplot(2, 2, 2)
            plt.imshow(reconstructed_images[0,:,:].cpu().detach().numpy(), cmap='gray')  
            plt.axis('off')
            plt.title('Iter 10')
        elif iter == 49:
            plt.subplot(2, 2, 3)
            plt.imshow(reconstructed_images[0,:,:].cpu().detach().numpy(), cmap='gray')  
            plt.axis('off')
            plt.title('Iter 50')
        elif iter == 99:
            plt.subplot(2, 2, 4)
            plt.imshow(reconstructed_images[0,:,:].cpu().detach().numpy(), cmap='gray')  
            plt.axis('off')
            plt.title('Iter 100')
    plt.savefig('TestIterativeMotion.png')
    plt.close

    return reconstructed_images


############################################################
## MCMR pipeline taken/adapted from                       ##
## https://github.com/JZPeterPan/MCMR-Recon-Driven-Motion ##
############################################################

class ReconDCPM(torch.nn.Module):
    def __init__(self, max_iter, weight_init=1e12, tol=1e-12):

        super().__init__()
        self.A = MulticoilForwardOp(center=True, coil_axis=-4, channel_dim_defined=False)
        self.AH = MulticoilAdjointOp(center=True, coil_axis=-4, channel_dim_defined=False)

        self.DC = DCPM(self.A, self.AH, weight_init=weight_init, max_iter=max_iter, tol=tol, weight_scale=1.0, requires_grad=False)

    def forward(self, img, kspace, mask, smaps):
        if img is None:
            img = self.AH(kspace, mask, smaps)
        recon_im = self.DC([img, kspace, mask, smaps])
        return recon_im


class ReconDCPMMotion(torch.nn.Module):
    def __init__(self, max_iter, coil_axis, weight_init=1e-12, tol=1e-12):
        super().__init__()
        self.A_motion = MulticoilMotionForwardOp(center=True, coil_axis=coil_axis, channel_dim_defined=False)
        self.AH_motion = MulticoilMotionAdjointOp(center=True, coil_axis=coil_axis, channel_dim_defined=False)

        self.DC = DCPM(self.A_motion, self.AH_motion, weight_init=weight_init, max_iter=max_iter, tol=tol, weight_scale=1.0)

    def forward(self, recon_image, kspace, mask, smaps, flow, transform, recon_frames=None, recon_neighbor_frames='all'):

        # modify the input size to fit the DCPM
        if flow.dim() != 6:
            flow = flow[None, ...]
            assert flow.dim() == 6
        if recon_frames is None:
            recon_frames = range(flow.shape[2])    # number of frames is the last dimension
        if recon_image.dim() != 4:
            recon_image = recon_image[None, ...]
        if kspace.dim() != 6:
            kspace_expand = torch.zeros((1, kspace.shape[1], *flow.shape[1:-1])).type(torch.complex64).cuda()
            for idx, slc in enumerate(recon_frames):
                neighbor_kspace = neighboring_frame_select(kspace, slc, recon_neighbor_frames, frame_dim=2)
                neighbor_kspace = neighbor_kspace[:, :, None, ...]
                kspace_expand[:, :, idx:idx+1] = neighbor_kspace
            kspace = kspace_expand
        if mask.dim() != 6:
            mask_expand = torch.zeros((*mask.shape[:2], *flow.shape[1:-1])).cuda()
            for idx, slc in enumerate(recon_frames):
                neighbor_mask = neighboring_frame_select(mask, slc, recon_neighbor_frames, frame_dim=2)
                neighbor_mask = neighbor_mask[:, :, None, ...]
                mask_expand[:, :, idx:idx+1] = neighbor_mask
            mask = mask_expand
        if smaps.dim() != 6:
            smaps = smaps[:, :, None, ...]

        recon_im = self.DC([recon_image, kspace, mask, smaps, flow, transform])
        return recon_im
    
def neighboring_frame_select(input, slc, neighboring_frame, frame_dim=1):
    """
    the input is regarded as cyclic.
    :param input:
    :param slc:
    :param neighboring_frame:
    :param frame_dim:
    :return:
    """
    nfr = input.shape[frame_dim]
    if isinstance(neighboring_frame, int): assert 2*neighboring_frame+1 <= nfr
    # alternative: for neighboring_frame == 'all' we can also shift nothing
    shift_offset = int(nfr/2) - slc if neighboring_frame == 'all' else neighboring_frame - slc

    input_shifted = torch.roll(input, shift_offset, dims=frame_dim)
    output = torch.swapaxes(input_shifted, frame_dim, 0)
    if isinstance(neighboring_frame, int):
        output = output[:2*neighboring_frame+1, ...]
    output = torch.swapaxes(output, 0, frame_dim)
    return output

def fft2(x, dim=(-2,-1)):
    return torch.fft.fft2(x, dim=dim, norm='ortho')


def ifft2(X, dim=(-2,-1)):
    return torch.fft.ifft2(X, dim=dim, norm='ortho')


def fft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(fft2(torch.fft.ifftshift(x, dim), dim), dim)


def ifft2c(x, dim=(-2,-1)):
    return torch.fft.fftshift(ifft2(torch.fft.ifftshift(x, dim), dim), dim)


class MulticoilForwardOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        if center:
            self.fft2 = fft2c
        else:
            self.fft2 = fft2
        self.coil_axis = coil_axis
        self.channel_dim_defined = channel_dim_defined

    def forward(self, image, mask, smaps):
        if self.channel_dim_defined:
            coilimg = torch.unsqueeze(image[:,0], self.coil_axis) * smaps
        else:
            coilimg = torch.unsqueeze(image, self.coil_axis) * smaps
        kspace = self.fft2(coilimg)
        masked_kspace = kspace * mask
        return masked_kspace


class MulticoilAdjointOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        if center:
            self.ifft2 = ifft2c
        else:
            self.ifft2 = ifft2
        self.coil_axis = coil_axis
        self.channel_dim_defined = channel_dim_defined

    def forward(self, kspace, mask, smaps):
        masked_kspace = kspace * mask
        coilimg = self.ifft2(masked_kspace)
        img = torch.sum(torch.conj(smaps) * coilimg, self.coil_axis)

        if self.channel_dim_defined:
            return torch.unsqueeze(img, 1)
        else:
            return img
        
class MulticoilMotionForwardOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        self.W = WarpForward()
        self.A = MulticoilForwardOp(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.channel_dim_defined = channel_dim_defined

    def forward(self, x, mask, smaps, u, transform):
        if self.channel_dim_defined:
            x = self.W(x[:,0], u)
        else:
            x = self.W(x, u)
        y = self.A(x, mask, smaps)
        return y


class MulticoilMotionAdjointOp(torch.nn.Module):
    def __init__(self, center=False, coil_axis=-3, channel_dim_defined=True):
        super().__init__()
        self.AH = MulticoilAdjointOp(center=center, coil_axis=coil_axis, channel_dim_defined=False)
        self.WH = WarpAdjoint()
        self.channel_dim_defined = channel_dim_defined

    def forward(self, y, mask, smaps, u, transform):
        x = self.AH(y, mask, smaps)
        x = self.WH(x, u, transform)
        if self.channel_dim_defined:
            return torch.unsqueeze(x, 1)
        else:
            return x

class WarpForward(torch.nn.Module):
    def forward(self, x, u):
        # we assume that the input does not have any channel dimension
        # x [batch, frames, M, N]
        # u [batch, frames_all, frames, M, N, 2]
        out_shape = u.shape[:-1]
        H, W = u.shape[-3:-1]
        x = torch.repeat_interleave(torch.unsqueeze(x, -3), repeats=u.shape[-5], dim=-3)
        x = torch.reshape(x, (-1, 1, H, W)) # [batch * frames * frames_all, 1, M, N]
        u = torch.reshape(u, (-1, H, W, 2)) # [batch * frames * frames_all, M, N, 2]
        x_re = torch.real(x).contiguous()
        x_im = torch.imag(x).contiguous()
        out_re = warp_torch(x_re.float(), u.permute(0, 3, 1, 2).float())
        out_im = warp_torch(x_im.float(), u.permute(0, 3, 1, 2).float())
        Wx = torch.complex(out_re, out_im)

        return torch.reshape(Wx, out_shape)

def warp_torch(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    mask = torch.ones(x.size(), dtype=x.dtype)
    if x.is_cuda:
        grid = grid.cuda()
        mask = mask.cuda()

    # flo = torch.flip(flo, dims=[1])
    # vgrid = Variable(grid) + flo
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x, vgrid, align_corners=True)
    
    #mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=True)
    #mask[mask < 0.9999] = 0
    #mask[mask > 0] = 1

    # return output * mask
    return output

class WarpAdjoint(torch.nn.Module):
    def forward(self, x, u, transform):
        # we assume that the input does not have any channel dimension
        # x [batch, frames, frames_all, H, W]
        # u [batch, frames, frames_all, H, W, 2]
        # transform is the spatial transformer
        out_shape = u.shape[:-1]
        H, W = u.shape[-3:-1]
        x = torch.reshape(x, (-1, 1, H, W)) # [batch * frames * frames_all, 1, M, N]
        u = torch.reshape(u, (-1, H, W, 2)) # [batch * frames * frames_all, M, N, 2]
        x_re = torch.real(x).contiguous()
        x_im = torch.imag(x).contiguous()
        grid, out_re = transform(x_re.float(), u.float()) 
        grid, out_im = transform(x_im.float(), u.float()) 
        x_warpT = torch.complex(out_re, out_im)
        x_warpT = torch.reshape(x_warpT, out_shape)
        x_warpT = torch.sum(x_warpT, -3) # unnecessary?
        return x_warpT

class DCPM(torch.nn.Module):
    def __init__(self, A, AH, weight_init=1.0, weight_scale=1.0, requires_grad=False, **kwargs):
        super().__init__()

        self.A = A
        self.AH = AH

        self.weight_scale = weight_scale
        self.weight_init = weight_init
        self._weight = torch.tensor(1, dtype=torch.float32) * weight_init
        # self._weight = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32)*weight_init)
        # self._weight.requires_grad_(requires_grad)
        # self._weight.proj = lambda: self._weight.data.clamp_(1e-4, 1000)

        max_iter = kwargs.get('max_iter', 10)
        tol = kwargs.get('tol', 1e-10)
        self.prox = CGClass(A, AH, max_iter=max_iter, tol=tol)

    @property
    def weight(self):
        return self._weight * self.weight_scale

    def forward(self, inputs, scale=1.0):
        x = inputs[0]           # subsampled images
        y = inputs[1]           # k-space
        constants = inputs[2:]  # masks, smaps, flows and spatial transformer
        lambdaa = 1.0 / torch.max(self.weight * scale, torch.ones_like(self.weight)*1e-9)
        return self.prox(lambdaa, x, y, *constants)

    def __repr__(self):
        return f'DCPD(lambda_init={self.weight_init:.4g}, weight_scale={self.weight_scale}, requires_grad={self._weight.requires_grad})'

class CGClass(torch.nn.Module):
    def __init__(self, A, AH, max_iter=10, tol=1e-10):
        super().__init__()
        self.A = A
        self.AH = AH
        self.max_iter = max_iter
        self.tol = tol

        self.cg = ComplexCG()

    def forward(self, lambdaa, x, y, *constants):
        out = torch.zeros_like(x)
        out = self.cg(self.A, self.AH, self.max_iter, self.tol, lambdaa, x, y, *[c for c in constants])
        """
        for n in range(x.shape[0]):
            cg_out = self.cg(self.A, self.AH, self.max_iter, self.tol, lambdaa, x[n::1], y[n::1], *[c[n::1] for c in constants])
            out[n] = cg_out[0]
        """
        return out

class ComplexCG(torch.nn.Module):

    def dotp(self, data1, data2):
        if data1.is_complex():
            mult = torch.conj(data1) * data2
        else:
            mult = data1 * data2
        return torch.sum(mult)

    def solve(self, x0, M, tol, max_iter):
        x = torch.zeros_like(x0)    # size [1,F,H,W]
        r = x0.clone()              # size [1,F,H,W]
        p = x0.clone()              # size [1,F,H,W]

        rTr = torch.norm(r).pow(2)

        it = 0
        while rTr > tol and it < max_iter:
            it += 1
            q = M(p)                        # size [1,F,H,W]
            alpha = rTr / self.dotp(p, q)
            x = x + alpha * p
            r = r - alpha * q

            rTrNew = torch.norm(r).pow(2)

            beta = rTrNew / rTr

            p = r.clone() + beta * p
            rTr = rTrNew.clone()
        return x

    def forward(self, A, AH, max_iter, tol, lambdaa, x, y, *constants):
        def M(p):
            return AH(A(p, *constants), *constants) + lambdaa * p

        rhs = AH(y, *constants) + lambdaa * x
        return self.solve(rhs, M, tol, max_iter)

######################
## Code from LAPNet ##
######################

def random_smooth_Displacement(img_size, amplitude=1):
    """ random smoothed displacements """
    H, W = img_size
    u = -1 + 2 * np.random.rand(H, W, 2)
    # u = np.random.normal(0, 1, (M, N, 2))
    cut_off = 0.01
    w_x_cut = math.floor(cut_off / (1 / H) + (H + 1) / 2)
    w_y_cut = math.floor(cut_off / (1 / W) + (W + 1) / 2)

    LowPass_win = np.zeros((H, W))
    LowPass_win[(H - w_x_cut): w_x_cut, (W - w_y_cut): w_y_cut] = 1

    u[..., 0] = (np.fft.ifft2(np.fft.fft2(u[..., 0]) * np.fft.ifftshift(LowPass_win))).real
    # also equal to u[..., 0] =
    # (np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(u[..., 0])) * LowPass_win))).real
    u[..., 1] = (np.fft.ifft2(np.fft.fft2(u[..., 1]) * np.fft.ifftshift(LowPass_win))).real

    u1 = u[..., 0].flatten()
    u2 = u[..., 1].flatten()
    amplitude = amplitude / max(np.linalg.norm(np.vstack([u1, u2]), axis=0))
    u = u * amplitude
    return u