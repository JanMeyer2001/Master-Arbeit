import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import os
from os import listdir, scandir
from os.path import join, isfile
import matplotlib.pyplot as plt
import itertools
import scipy
import h5py
import fastmri
from fastmri.data import transforms as T

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


def show_coils(data, slice_nums, cmap=None, vmax = 0.0005):
    '''
    plot the figures along the first dims.
    '''
    fig = plt.figure()
    for i, num in enumerate(slice_nums):
        plt.subplot(1, len(slice_nums), i + 1)
        plt.imshow(data[num], cmap=cmap,vmax=vmax)
        plt.axis('off')
        
    plt.savefig('./Thesis/Images/Coils.png') 
    plt.close


def rotate_image(image):
    [w, h] = image.shape
    image_new = np.zeros((h,w))
    for y in range(h):
        for x in range(w):
            #image_new[y, w-x-1] = image[x, y]
            image_new[h-y-1, x] = image[x, y]

    return image_new        


# Data from CMRxRecon
data_path = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/FullSample'
names = [f.path for f in os.scandir(data_path) if f.is_dir()]

# read files from mat to numpy
fullmulti = readfile2numpy(os.path.join(data_path, names[0], 'cine_sax.mat'))
[nframe, nslice, ncoil, ny, nx] = fullmulti.shape

for i in range(nframe):
    n = 1
    for j in range(nslice):
        # choose frame and slice
        slice_kspace = fullmulti[i,j] 

        # Convert from numpy array to pytorch tensor
        slice_kspace2 = T.to_tensor(slice_kspace) 

        # Apply Inverse Fourier Transform to get the complex image       
        image = fastmri.ifft2c(slice_kspace2)  

        # Compute absolute value to get a real image      
        image_abs = fastmri.complex_abs(image)   

        #show_coils(image_abs, [0, 3, 6], cmap='gray', vmax = 0.0005)
        # combine the coil images to a coil-combined one
        slice_image_rss = fastmri.rss(image_abs, dim=0)

        # plot the final image
        plt.subplot(round(nslice/2), round(nslice/2), n)
        plt.imshow(np.abs(slice_image_rss.numpy()), cmap='gray', vmax = 0.0015)
        plt.axis('off')
        #plt.savefig('./Thesis/Images/ImageSlice.png') 
        #plt.savefig('Frame' + str(i) + 'Slice' + str(j) +'.png') #'./Thesis/Images/image' + str(i) +'.png'
        #plt.close

        n = n+1

    plt.savefig('Frame' + str(i) + '.png') #'./Thesis/Images/image' + str(i) +'.png'
    plt.title('Frame' + str(i))
    plt.close

""" 
# Data from OASIS
data_path = '/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS'
names = [f for f in listdir(join(data_path, 'imagesTr')) if isfile(join(data_path, 'imagesTr', f))]

for i in range(5):
    # get images
    nim1 = nib.load(os.path.join(data_path, 'imagesTr', names[i]))
    
    imagex = nim1.get_fdata()[96,:,:]
    imagex = rotate_image(imagex)
    imagex = np.array(imagex, dtype='float32') 
    
    imagey = nim1.get_fdata()[:,96,:]
    imagey = rotate_image(imagey)
    imagey = np.array(imagey, dtype='float32')
    
    imagez = nim1.get_fdata()[:,:,96]
    imagez = rotate_image(imagez)
    imagez = np.array(imagez, dtype='float32')

    # get labels
    nim1 = nib.load(os.path.join(data_path, 'labelsTr', names[i])) 
    labelx = nim1.get_fdata()[96,:,:]
    labelx = rotate_image(labelx)
    labelx = np.array(labelx, dtype='float32') 
    
    labely = nim1.get_fdata()[:,96,:]
    labely = rotate_image(labely)
    labely = np.array(labely, dtype='float32')
    
    labelz = nim1.get_fdata()[:,:,96]
    labelz = rotate_image(labelz)
    labelz = np.array(labelz, dtype='float32')
    
    # plot images and labels
    plt.subplot(2, 3, 1)
    plt.imshow(imagex, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('X Axis')

    plt.subplot(2, 3, 2)
    plt.imshow(imagey, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Y Axis')
    
    plt.subplot(2, 3, 3)
    plt.imshow(imagez, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Z Axis')
    
    plt.subplot(2, 3, 4)
    plt.imshow(labelx)
    plt.axis('off')
    #plt.title('X Axis')

    plt.subplot(2, 3, 5)
    plt.imshow(labely)
    plt.axis('off')
    #plt.title('Y Axis')
    
    plt.subplot(2, 3, 6)
    plt.imshow(labelz)
    plt.axis('off')
    #plt.title('Z Axis')
    
    plt.savefig('./Thesis/Images/image' + str(i) +'.png')
    plt.close
"""