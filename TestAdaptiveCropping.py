import os
from os import listdir, scandir
from os.path import join, isfile
import matplotlib.pyplot as plt
from skimage.io import imread
import torch
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.feature import canny
import numpy as np
from skimage.morphology import area_opening
import cv2

def normalize(image):
    """ expects an image as tensor and normalizes the values between [0,1] """
    return (image - torch.min(image)) / (torch.max(image) - torch.min(image))

def crop_image(image):
    [x, y] = image.shape
    x_center = int(x/2)
    y_center = int(y/2)
    new_x = int(x/6)        # crop x by 3
    new_y = int(y/4)        # and y by 2
    return image[(x_center-new_x):(x_center+new_x),(y_center-new_y):(y_center+new_y)]

def extract_differences(path):
    # get all frames
    frames = [f.path for f in os.scandir(join(path)) if isfile(join(path, f))]
    images = torch.zeros([len(frames), 246, 512])  
    differences = torch.zeros([len(frames)-1, 246, 512]) 
    i = 1
    for frame in frames:
        image = imread(frame, as_gray=True)
        images[i-1,:,:] = torch.from_numpy(image)
        if i>1:
            differences[i-2,:,:] = abs(images[i-1,:,:]-images[i-2,:,:])
        i += 1

    mean_diff = torch.sum(differences, dim=0)/differences.shape[0] # mean over all differences

    return images[-1,:,:], mean_diff

#data_path = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/ValidationSet/FullySampled/P010' #AccFactor04
data_path = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/TrainingSet/FullySampled/P120' #AccFactor04
slices = [f.path for f in os.scandir(data_path) if f.is_dir() and not (f.name.find('Slice') == -1)]
diffs_frames = torch.zeros([len(slices), 246, 512])

plt.subplots(figsize=(7, 4))
plt.axis('off')

for i, slice in enumerate(slices):
    image, diff = extract_differences(slice)
    diffs_frames[i,:,:] = diff
    plt.subplot(5,4,2*i+1) 
    plt.imshow(image, cmap='gray', vmin=0, vmax = 1)
    plt.axis('off')
    plt.subplot(5,4,2*i+2) 
    plt.imshow(normalize(diff), cmap='gray', vmin=0, vmax = 1)
    plt.axis('off')

"""
mask = canny(mask, sigma=3)
radius = [20,30,40,50]
h_space = hough_circle(mask, radius) #[0,:,:]
accum, cx, cy, radii = hough_circle_peaks(h_space, [radius,])
mask = mask*0
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_x, center_y, radius, shape=mask.shape)
    mask[circy, circx] = 1

plt.subplot(round(len(slices)/2),round(len(slices)/2)+1,2*len(slices)+2) 
plt.imshow(mask, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')

plt.tight_layout()
plt.savefig('AllSlices.png') #./Thesis/Images/
plt.close
"""
image, diff = extract_differences(slices[0])

sum_diffs_slices = torch.sum(diffs_frames, dim=0) # sum over all differences between slices

mask = sum_diffs_slices>0.25        # threshold to get the mask
mask = area_opening(mask.int().numpy(), area_threshold=125) # perform opening to get rid of any noise

positions = np.nonzero(mask)        # get all positions
top = positions[0].max()
bottom = positions[0].min()
left = positions[1].min()
right = positions[1].max()

"""
# get rectangle for ROI 
output = cv2.rectangle(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (left, top), (right, bottom), (0,255,0), 1)
cv2.imwrite('plot_with_bounds.png', output)

plt.subplot(1,5,4) 
plt.imshow(output, cmap='gray', vmin=0, vmax = 1)
plt.title('Extracted ROI')
plt.axis('off')
"""

center_x = bottom + int((top - bottom)/2)
center_y = left + int((right - left)/2)
"""
center = torch.zeros([246,512])
center[center_x, center_y] = 1
plt.subplot(1,5,4) 
plt.imshow(center, cmap='gray', vmin=0, vmax = 1)
#plt.title('Center')
plt.axis('off')
"""
crop_x = int(image.shape[0]/6)
crop_y = int(image.shape[1]/6)
image_crop = image[(center_x-crop_x):(center_x+crop_x),(center_y-crop_y):(center_y+crop_y)]
print('image size before cropping: ', image.shape)
print('image size after cropping: ', image_crop.shape)

#"""
plt.subplots(figsize=(7, 4))
plt.axis('off')

plt.subplot(1,4,1) 
plt.imshow(image, cmap='gray', vmin=0, vmax = 1)
#plt.title('Original image')
plt.axis('off')

plt.subplot(1,4,2) 
plt.imshow(sum_diffs_slices, cmap='gray', vmin=0, vmax = 1)
#plt.title('Sum of Differences for slices')
plt.axis('off')

plt.subplot(1,4,3) 
plt.imshow(mask, cmap='gray', vmin=0, vmax = 1)
#plt.title('Generated Mask')
plt.axis('off')

plt.subplot(1,4,4) 
plt.imshow(image_crop, cmap='gray', vmin=0, vmax = 1)
#plt.title('Cropped image')
plt.axis('off')

plt.tight_layout()
plt.savefig('./Thesis/Images/AdaptiveCropping.png') 
plt.close
#"""

plt.subplot(1,1,1) 
plt.imshow(image, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout()
plt.savefig('./Thesis/Images/FullImage.png') 
plt.close

plt.subplot(1,1,1) 
plt.imshow(sum_diffs_slices, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout()
plt.savefig('./Thesis/Images/DifferencesSlices.png') 
plt.close

plt.subplot(1,1,1) 
plt.imshow(mask, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout()
plt.savefig('./Thesis/Images/MaskForCropping.png') 
plt.close

plt.subplot(1,1,1) 
plt.imshow(image_crop, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout()
plt.savefig('./Thesis/Images/ImageCrop.png') 
plt.close
