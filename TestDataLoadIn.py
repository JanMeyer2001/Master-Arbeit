import os
from os import listdir, scandir
from os.path import join, isfile
import matplotlib.pyplot as plt
from skimage.io import imread
import torch

def crop_image(image):
    [x, y] = image.shape
    x_center = int(x/2)
    y_center = int(y/2)
    new_x = int(x/6)        # crop x by 3
    new_y = int(y/4)        # and y by 2
    return image[(x_center-new_x):(x_center+new_x),(y_center-new_y):(y_center+new_y)]

#data_path = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/ValidationSet/FullySampled/P001/Slice0' #AccFactor04
data_path = '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon/TrainingSet/FullySampled/P120/Slice6' #AccFactor04

# get all frames
frames = [f.path for f in os.scandir(join(data_path)) if isfile(join(data_path, f))]
images = torch.zeros([len(frames), 246, 512])  # cropped sizes of [82, 256] instead of [246, 512]
differences = torch.zeros([len(frames)-1, 246, 512]) 

plt.subplots(figsize=(7, 4))
plt.axis('off')
i = 1
for frame in frames:
    image = imread(frame, as_gray=True) #crop_image()
    images[i-1,:,:] = torch.from_numpy(image)
    framename = os.path.basename(frame)
    plt.subplot(4,3,i)
    plt.imshow(image, cmap='gray', vmin=0, vmax = 1)
    plt.title(framename)
    plt.axis('off')
    if i>1:
        differences[i-2,:,:] = abs(images[i-1,:,:]-images[i-2,:,:])

    i += 1
plt.tight_layout()
plt.savefig('Frames.png') #./Thesis/Images/
plt.close

"""
plt.subplots(figsize=(7, 4))
plt.axis('off')
for i in range(differences.shape[0]):
    plt.subplot(4,3,i+1)
    plt.imshow(differences[i,:,:], cmap='gray', vmin=0, vmax = 1)
    plt.axis('off')

sum_diff = torch.sum(differences, dim=0) # sum over all differences
plt.subplot(4,3,differences.shape[0]+1)
plt.imshow(sum_diff, cmap='gray', vmin=0, vmax = 1)
plt.axis('off')
plt.tight_layout()
plt.savefig('Differences.png') #./Thesis/Images/
plt.close
"""