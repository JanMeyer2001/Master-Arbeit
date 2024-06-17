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

parser = ArgumentParser()
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon',
                    #default='/home/jmeyer/storage/datasets/CMRxRecon/MultiCoil/Cine/TrainingSet/FullSample', #AccFactor04
                    help="data path for training images")
opt = parser.parse_args()
datapath = opt.datapath

bs = 1
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
test_set = TestDatasetCMRBenchmark(data_path=datapath, mode=0)
test_generator = Data.DataLoader(dataset=test_set, batch_size=bs, shuffle=False, num_workers=2)
path = '/home/jmeyer/storage/students/janmeyer_711878/data/Nifti/ImagePair1'
num_image_pair = 1

for mov_img, fix_img, _, _ in test_generator: 
    with torch.no_grad():
        if num_image_pair == 1:
            # convert to numpy arrays
            mov_img = mov_img[0,0,:,:].cpu().numpy()
            fix_img = fix_img[0,0,:,:].cpu().numpy()
            
            # load Nifti files
            mov_img_nifti = nibabel.load(join(path,'MovingImage.nii'))
            fix_img_nifti = nibabel.load(join(path,'FixedImage.nii'))

            mov_img_nifti_array = np.array(mov_img_nifti.get_fdata(), dtype='float32')
            fix_img_nifti_array = np.array(fix_img_nifti.get_fdata(), dtype='float32')

            # plot all image to look for differences
            plt.subplots(figsize=(7, 4))
            plt.axis('off')
            
            plt.subplot(3,2,1) 
            plt.imshow(mov_img, cmap='gray')
            plt.title('Moving')
            plt.axis('off')
            
            plt.subplot(3,2,2) 
            plt.imshow(fix_img, cmap='gray')
            plt.title('Fixed')
            plt.axis('off')
            
            plt.subplot(3,2,3) 
            plt.imshow(mov_img_nifti_array, cmap='gray')
            plt.title('Moving Nifti')
            plt.axis('off')
            
            plt.subplot(3,2,4) 
            plt.imshow(fix_img_nifti_array, cmap='gray')
            plt.title('Fixed Nifti')
            plt.axis('off')
            
            plt.subplot(3,2,5) 
            plt.imshow(abs(mov_img_nifti_array-mov_img), cmap='gray')
            plt.title('Difference Moving')
            plt.axis('off')
            
            plt.subplot(3,2,6) 
            plt.imshow(abs(fix_img_nifti_array-fix_img), cmap='gray')
            plt.title('Difference Fixed')
            plt.axis('off')

            plt.tight_layout()
            plt.savefig('ImageComparisonNifti.png') #./Thesis/Images/
            plt.close

            # increment counter for image pair number
            num_image_pair += 1