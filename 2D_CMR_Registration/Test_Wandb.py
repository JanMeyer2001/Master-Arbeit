import wandb
import numpy as np
import torch
import wandb.sync
from Models import *
from Functions import *
import torch.utils.data as Data
import warnings
warnings.filterwarnings("ignore")

config={
    "bs": 1,
    "learning_rate": 1e-4,
    "start_channel": 8,
    "smth_lambda": 0.01,
    "FT_size": [24,24],
    "choose_loss": 1,
    "mode": 0,
    "F_Net_plus": 2,
    #"dataset": "CMRxRecon",
    "dataset": "ACDC",
    "epochs": 50,
    "diffeo": 0,
}
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
epochs = config["epochs"]

# choose the model
assert config["F_Net_plus"] == 0 or config["F_Net_plus"] == 1 or config["F_Net_plus"] == 2, f"Expected F_Net_plus to be either 0, 1 or 2, but got: {config["F_Net_plus"]}"
assert config["diffeo"] == 0 or config["diffeo"] == 1, f"Expected diffeo to be either 0 or 1, but got: {config["diffeo"]}"
if config["F_Net_plus"] == 0:
    model = Fourier_Net(2, 2, config["start_channel"], config["diffeo"]).cuda() 
elif config["F_Net_plus"] == 1:
    assert config["FT_size"][0] > 0 and config["FT_size"][0] <= 40 and config["FT_size"][1] > 0 and config["FT_size"][1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{config["FT_size"][0]}, {config["FT_size"][1]}]"
    model = Fourier_Net_plus(2, 2, config["start_channel"], config["diffeo"], config["FT_size"]).cuda() 
elif config["F_Net_plus"] == 2:
    assert config["FT_size"][0] > 0 and config["FT_size"][0] <= 40 and config["FT_size"][1] > 0 and config["FT_size"][1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{config["FT_size"][0]}, {config["FT_size"][1]}]"
    model = Cascade(2, 2, config["start_channel"], config["diffeo"], config["FT_size"]).cuda() 

# choose the loss function for similarity
assert config["choose_loss"] >= 0 and config["choose_loss"] <= 3, f"Expected choose_loss to be one of SAD (0), MSE (1), NCC (2) or SSIM (3), but got: {config["choose_loss"]}"
if config["choose_loss"] == 1:
    loss_similarity = MSE().loss
elif config["choose_loss"] == 0:
    loss_similarity = SAD().loss
elif config["choose_loss"] == 2:
    loss_similarity = NCC(win=9)
elif config["choose_loss"] == 3:
    ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1, win_size=9)
    loss_similarity = SAD().loss
loss_smooth = smoothloss

transform = SpatialTransform().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

for param in transform.parameters():
    param.requires_grad = False
    param.volatile = True

# load CMR training, validation and test data
assert config["mode"] >= 0 and config["mode"] <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {config["mode"]}"

if config["dataset"] == "CMRxRecon":
    train_set = TrainDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', config["mode"]) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=config["bs"], shuffle=True, num_workers=4)
    validation_set = ValidationDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', config["mode"]) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=config["bs"], shuffle=True, num_workers=4)
    test_set = TestDatasetCMRxReconBenchmark('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode=config["mode"])
    test_generator = Data.DataLoader(dataset=test_set, batch_size=config["bs"], shuffle=False, num_workers=2)
elif config["dataset"] == "ACDC":
    train_set = TrainDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', config["mode"]) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=config["bs"], shuffle=True, num_workers=4)
    validation_set = ValidationDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', config["mode"]) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=config["bs"], shuffle=True, num_workers=4)
    test_set = TestDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode=config["mode"])
    test_generator = Data.DataLoader(dataset=test_set, batch_size=config["bs"], shuffle=False, num_workers=2)
else:
    raise ValueError('Dataset should be "ACDC" or "CMRxRecon", but found "%s"!' % config["dataset"])

print('Finished Loading!')

epochs = log_TrainTest(None,model,config["F_Net_plus"],config["diffeo"],config["dataset"],config["FT_size"],config["learning_rate"],config["start_channel"],config["smth_lambda"],config["choose_loss"],config["mode"],epochs,optimizer,loss_similarity,loss_smooth,transform,training_generator,validation_generator,test_generator,True)
    