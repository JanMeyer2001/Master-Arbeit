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

# define modes and methods
modes = [1,2,3]
num_methods = 4

# image size
H = 246
W = 512

# define metrics
metrics = np.zeros((3,4,4))

# fill in metrics for R=4
metrics[0,0,:] = [48.82,28.79,76.25,0.13]
metrics[0,1,:] = [55.92,29.54,78.24,0.11]
metrics[0,2,:] = [51.83,28.85,75.88,0.13]
metrics[0,3,:] = [50.47,28.87,75.87,0.13]
# fill in metrics for R=8
metrics[1,0,:] = [48.15,27.78,74.69,0.17]
metrics[1,1,:] = [51.03,28.05,75.76,0.16]
metrics[1,2,:] = [51.15,28.20,76.34,0.15]
metrics[1,3,:] = [50.58,27.90,75.87,0.16]
# fill in metrics for R=10
metrics[2,0,:] = [43.27,27.37,74.47,0.18]
metrics[2,1,:] = [46.42,27.47,74.48,0.18]
metrics[2,2,:] = [50.94,28.12,76.27,0.15]
metrics[2,3,:] = [44.43,27.38,74.76,0.18]

for mode in modes:
    save_name = '/home/jmeyer/storage/students/janmeyer_711878/Master-Arbeit/Images/ResultsReconstructionMetrics_mode{}.png'.format(mode)
    fig, axes = plt.subplots(1, num_methods, figsize=(20, 5))
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0.0, top=1)

    # Compute and display metrics for ground truth (no errors)
    empyt_img = np.ones((H,W))
    for i in range(num_methods):
        axes[i].imshow(empyt_img, cmap="magma", alpha=0.0)
        axes[i].axis("off")

        # display metrics for each method image 
        info_text = f"HaarPSI: {metrics[mode-1,i,0]:.2f} %  PSNR: {metrics[mode-1,i,1]:.2f} dB\nSSIM:      {metrics[mode-1,i,2]:.2f} %  MSE:   {metrics[mode-1,i,3]:.2f} e-3"
        text = axes[i].text(
            10,
            156,
            info_text,
            fontsize=14,
            color="black",
            fontfamily="serif",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
        )
        axes[i].margins(0, 0)
        axes[i].axis("off")

    plt.tight_layout(h_pad=0.1, w_pad=0.3, pad=0)
    plt.show()
    fig.savefig(save_name, dpi=600, pad_inches=0.05, bbox_inches="tight")