from torchinfo import summary
from Functions import *
from Models import *
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--start_channel", type=int, dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--dataset", type=str, dest="dataset", default="ACDC",
                    help="dataset for training images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--model", type=int,
                    dest="model", default=0, #2
                    help="choose whether to use Fourier-Net (0), Fourier-Net+ (1), cascaded Fourier-Net (2) or VoxelMorph (3) as the model")
parser.add_argument("--diffeo", type=int,
                    dest="diffeo", default=0, 
                    help="choose whether to use a diffeomorphic transform (1) or not (0)")
parser.add_argument("--FT_size_x", type=int,
                    dest="FT_size_x", default=24,
                    help="choose size x of FT crop: Should be smaller than 40.")
parser.add_argument("--FT_size_y", type=int,
                    dest="FT_size_y", default=24,
                    help="choose size y of FT crop: Should be smaller than 84.")
opt = parser.parse_args()

start_channel = opt.start_channel
dataset = opt.dataset
model = opt.model
diffeo = opt.diffeo
FT_size = [opt.FT_size_x,opt.FT_size_y]

if dataset == 'ACDC':
    # load ACDC test data
    test_set = TestDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', 0) 
elif dataset == 'CMRxRecon':
    # load CMRxRecon test data
    test_set = TestDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', 0) 
elif dataset == 'OASIS':
    # path for OASIS test dataset
    test_set = TestDatasetOASIS('/imagedata/Learn2Reg_Dataset_release_v1.1/OASIS') 
else:
    raise ValueError('Dataset should be "ACDC", "CMRxRecon" or "OASIS", but found "%s"!' % dataset)

input_shape = test_set.__getitem__(0)[0].unsqueeze(0).shape

assert model >= 0 and model <= 6, f"Expected F_Net_plus to be either 0, 1 or 2, but got: {model}"
assert diffeo == 0 or diffeo == 1, f"Expected diffeo to be either 0 or 1, but got: {diffeo}"
if model == 0:
    model = Fourier_Net(2, 2, start_channel, diffeo).cuda() 
elif model == 1:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Fourier_Net_plus(2, 2, start_channel, diffeo, FT_size).cuda() 
elif model == 2:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade(2, 2, start_channel, diffeo, FT_size).cuda()
elif model == 3:
    model = Fourier_Net_dense(2, 2, start_channel, diffeo, FT_size).cuda()
    input_shape = [1,1,224,256] 
elif model == 4:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Fourier_Net_plus_dense(2, 2, start_channel, diffeo, FT_size).cuda() 
elif model == 5:
    assert FT_size[0] > 0 and FT_size[0] <= 40 and FT_size[1] > 0 and FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{FT_size[0]}, {FT_size[1]}]"
    model = Cascade_dense(2, 2, start_channel, diffeo, FT_size).cuda() 
elif model == 6:
    model = VxmDense(inshape=input_shape[2:4], nb_unet_features=32, bidir=False, nb_unet_levels=4).cuda() 

summary(model, input_size=[list(input_shape),list(input_shape)])

"""
model_Fourier_Net = Fourier_Net(2, 2, 8, 0, [24,24]).cuda()
model_Fourier_Net_plus = Fourier_Net_plus(2, 2, 8, 0, [24,24]).cuda()
model_Fourier_Net_plus_cascade = Cascade(2, 2, 8, 0, [24,24]).cuda()
model_Fourier_Net_diff = Fourier_Net(2, 2, 8, 1, [24,24]).cuda()
model_Fourier_Net_plus_diff = Fourier_Net_plus(2, 2, 8, 1, [24,24]).cuda()
model_Fourier_Net_plus_cascade_diff = Cascade(2, 2, 8, 1, [24,24]).cuda()

summary(model_Fourier_Net, input_size=[list(input_shape),list(input_shape)])
summary(model_Fourier_Net_plus, input_size=[list(input_shape),list(input_shape)])
summary(model_Fourier_Net_plus_cascade, input_size=[list(input_shape),list(input_shape)])
summary(model_Fourier_Net_diff, input_size=[list(input_shape),list(input_shape)])
summary(model_Fourier_Net_plus_diff, input_size=[list(input_shape),list(input_shape)])
summary(model_Fourier_Net_plus_cascade_diff, input_size=[list(input_shape),list(input_shape)])
"""