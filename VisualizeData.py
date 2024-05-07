import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import os
from os import listdir
from os.path import join, isfile
import matplotlib.pyplot as plt
import itertools

def rotate_image(image):
    [w, h] = image.shape
    image_new = np.zeros((h,w))
    for y in range(h):
        for x in range(w):
            #image_new[y, w-x-1] = image[x, y]
            image_new[h-y-1, x] = image[x, y]

    return image_new        

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
    plt.imshow(label)
    plt.axis('off')
    plt.savefig('label.png')
    plt.close

    
    nim1 = nib.load(os.path.join(data_path, 'masksTr', names[1])) 
    mask = nim1.get_fdata()[:,:,96]
    mask = np.array(mask, dtype='float32') 

    plt.imshow(mask)
    plt.axis('off')
    plt.savefig('mask.png')
    plt.close
    """