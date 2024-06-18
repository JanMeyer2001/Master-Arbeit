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
save_path = '/home/jmeyer/storage/students/janmeyer_711878/data/Nifti'
num_image_pair = 1

for mov_img, fix_img, _, _ in test_generator: 
    with torch.no_grad():
        # convert to numpy arrays
        mov_img = mov_img[0,0,:,:].cpu().numpy()
        fix_img = fix_img[0,0,:,:].cpu().numpy()
        
        # convert to Nifit format
        mov_img_nifti = nibabel.Nifti1Image(mov_img, np.eye(4))
        fix_img_nifti = nibabel.Nifti1Image(fix_img, np.eye(4))
        
        # create folder if it does not exist 
        path = join(save_path,'ImagePair'+str(num_image_pair))
        if not isdir(path):
            mkdir(path)

        # save Nifti images
        nibabel.save(mov_img_nifti, join(path,'MovingImage'))
        nibabel.save(fix_img_nifti, join(path,'FixedImage'))

        # increment counter for image pair number
        num_image_pair += 1