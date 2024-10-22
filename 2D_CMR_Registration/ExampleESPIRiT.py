import matplotlib.pyplot as plt
import numpy as np
from numpy import transpose
import torch
from fastmri.data import transforms as T
import time
from Functions import *

# Load k-space data
data_path = '/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/AccFactor04'
names = [f.path for f in scandir(data_path) if f.is_dir() and f.name.endswith('P120')]
X = readfile2numpy(join(data_path, names[0], 'cine_sax.mat'))[0,0] # take first slice and frame
X_real = torch.from_numpy(X['real'].astype(np.float32))
X_imag = torch.from_numpy(X['imag'].astype(np.float32))
X = torch.complex(X_real,X_imag)
X = torch.permute(X, (1,2,0)).unsqueeze(dim=0)

# Derive ESPIRiT operator
esp = espirit(X.numpy(), 6, 24, 0.01, 0.9925)

# Display ESPIRiT operator
for idx in range(np.squeeze(esp).shape[2]):
    for jdx in range(np.squeeze(esp).shape[3]):
        plt.subplot(np.squeeze(esp).shape[2], np.squeeze(esp).shape[3], idx * np.squeeze(esp).shape[2] + jdx + 1)
        plt.imshow(np.abs(np.squeeze(esp)[:,:,idx,jdx]), cmap='gray')
        plt.axis('off')
plt.show()
plt.savefig('ESPIRiT1.png') # for saving the image
plt.close

# transform into image space
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax) 
#x = ifft(X, (0, 1, 2))
#x = IFFT(torch.from_numpy(X))

x = fastmri.ifft2c(torch.view_as_real(X))      
x = fastmri.complex_abs(x) 

# Do projections
ip, proj, null = espirit_proj(x, esp)

# Figure code
x    = np.squeeze(x)
ip   = np.squeeze(ip)
proj = np.squeeze(proj)
null = np.squeeze(null)

dspx = np.power(np.abs(np.concatenate((x[:, :, 0], x[:, :, 1], x[:, :, 2], x[:, :, 3], x[:, :, 4], x[:, :, 5], x[:, :, 6], x[:, :, 7]), 0)), 1/3)
dspip = np.power(np.abs(np.concatenate((ip[:, :, 0], ip[:, :, 1], ip[:, :, 2], ip[:, :, 3], ip[:, :, 4], ip[:, :, 5], ip[:, :, 6], ip[:, :, 7]), 0)), 1/3)
dspproj = np.power(np.abs(np.concatenate((proj[:, :, 0], proj[:, :, 1], proj[:, :, 2], proj[:, :, 3], proj[:, :, 4], proj[:, :, 5], proj[:, :, 6], proj[:, :, 7]), 0)), 1/3)
dspnull = np.power(np.abs(np.concatenate((null[:, :, 0], null[:, :, 1], null[:, :, 2], null[:, :, 3], null[:, :, 4], null[:, :, 5], null[:, :, 6], null[:, :, 7]), 0)), 1/3)

print("NOTE: Contrast has been changed")

# Display ESPIRiT projection results 
plt.subplot(1, 4, 1)
plt.imshow(dspx, cmap='gray')
plt.title('Data')
plt.axis('off')
plt.subplot(1, 4, 2)
plt.imshow(dspip, cmap='gray')
plt.title('Inner product')
plt.axis('off')
plt.subplot(1, 4, 3)
plt.imshow(dspproj, cmap='gray')
plt.title('Projection')
plt.axis('off')
plt.subplot(1, 4, 4)
plt.imshow(dspnull, cmap='gray')
plt.title('Null Projection')
plt.axis('off')
plt.show()
plt.savefig('ESPIRiT2.png') # for saving the image
plt.close