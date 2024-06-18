from os import makedirs, mkdir
from os.path import isdir, join
from argparse import ArgumentParser
import numpy as np
import torch
from Models import *
from Functions import *
import torch.utils.data as Data
import nibabel
from skimage.metrics import structural_similarity, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

path = '/home/jmeyer/storage/students/janmeyer_711878/data/Nifti/ImagePair1'

# load Nifti files
mov_img = nibabel.load(join(path,'MovingImage.nii'))
fix_img = nibabel.load(join(path,'FixedImage.nii'))
warped_img = nibabel.load(join(path,'WarpedImage.nii'))

mov_img_array = np.array(mov_img.get_fdata(), dtype='float32')
fix_img_array = np.array(fix_img.get_fdata(), dtype='float32')
warped_img_array = np.array(warped_img.get_fdata(), dtype='float32')

print('MSE before registration: ', mean_squared_error(mov_img_array,fix_img_array),', SSIM before registration: ', structural_similarity(mov_img_array,fix_img_array, data_range=1))
print('MSE after registration: ', mean_squared_error(warped_img_array,fix_img_array),', SSIM after registration: ', structural_similarity(warped_img_array,fix_img_array, data_range=1))

# plot images before and after registration
plt.subplots(figsize=(7, 4))
plt.axis('off')

plt.subplot(2,3,1) 
plt.imshow(mov_img_array, cmap='gray')
plt.title('Moving')
plt.axis('off')

plt.subplot(2,3,2) 
plt.imshow(fix_img_array, cmap='gray')
plt.title('Fixed')
plt.axis('off')

plt.subplot(2,3,3) 
plt.imshow(warped_img_array, cmap='gray')
plt.title('Warped')
plt.axis('off')

plt.subplot(2,3,5) 
plt.imshow(abs(fix_img_array-mov_img_array), cmap='gray')
plt.title('Difference before')
plt.axis('off')

plt.subplot(2,3,6) 
plt.imshow(abs(fix_img_array-warped_img_array), cmap='gray')
plt.title('Difference after')
plt.axis('off')

plt.tight_layout()
plt.savefig('TestNiftyReg.png') #./Thesis/Images/
plt.close