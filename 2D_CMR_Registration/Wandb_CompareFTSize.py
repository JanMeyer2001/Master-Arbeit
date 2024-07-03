import wandb
import numpy as np
import torch
import wandb.sync
from Models import *
from Functions import *
import torch.utils.data as Data
import warnings
warnings.filterwarnings("ignore")

# compare different FT crop sizes
total_runs = 4
project_name = "FTSize_Fourier-Net+"
names = ["FTCropSize_40-84", "FTCropSize_20-42", "FTCropSize_24-24", "FTCropSize_12-12"]

# init array for test results
results_test = np.zeros((total_runs,7))

for run in range(total_runs):
    if run == 0:  
        print(names[run])
        wandb.init(
            # Set the project where this run will be logged
            project = project_name,
            # pass the run name
            name = names[run],
            # track hyperparameters and run metadata
            config={
                "bs": 1,
                "learning_rate": 1e-4,
                "start_channel": 8,
                "smth_lambda": 0.01,
                "FT_size": [40,84],
                "choose_loss": 1,
                "mode": 0,
                "F_Net_plus": True,
                #"dataset": "CMRxRecon",
                "dataset": "ACDC",
                "epochs": 50,
                "diffeo": False,
            }
        )
    elif run == 1:
        print(names[run])
        wandb.init(
            project = project_name,
            name = names[run],
            # track hyperparameters and run metadata
            config={
                "bs": 1,
                "learning_rate": 1e-4,
                "start_channel": 16,
                "smth_lambda": 0.01,
                "FT_size": [20,42],
                "choose_loss": 1,
                "mode": 0,
                "F_Net_plus": True,
                #"dataset": "CMRxRecon",
                "dataset": "ACDC",
                "epochs": 50,
                "diffeo": False,
            }
        )
    elif run == 2:
        print(names[run])
        wandb.init(
            project = project_name,
            name = names[run],
            # track hyperparameters and run metadata
            config={
                "bs": 1,
                "learning_rate": 1e-4,
                "start_channel": 32,
                "smth_lambda": 0.01,
                "FT_size": [24,24],
                "choose_loss": 1,
                "mode": 0,
                "F_Net_plus": True,
                #"dataset": "CMRxRecon",
                "dataset": "ACDC",
                "epochs": 50,
                "diffeo": False,
            }
        )   
    elif run == 3:
        print(names[run])
        wandb.init(
            project = project_name,
            name = names[run],
            # track hyperparameters and run metadata
            config={
                "bs": 1,
                "learning_rate": 1e-4,
                "start_channel": 32,
                "smth_lambda": 0.01,
                "FT_size": [12,12],
                "choose_loss": 1,
                "mode": 0,
                "F_Net_plus": True,
                #"dataset": "CMRxRecon",
                "dataset": "ACDC",
                "epochs": 50,
                "diffeo": False,
            }
        )       
        
    # Copy config 
    config = wandb.config
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    epochs = config.epochs

    # choose the model
    model_name = 0
    if config.F_Net_plus:
        assert config.FT_size[0] > 0 and config.FT_size[0] <= 40 and config.FT_size[1] > 0 and config.FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{config.FT_size[0]}, {config.FT_size[1]}]"
        model = Cascade(2, 2,  config.start_channel,  config.FT_size).cuda() 
        model_name = 1
    else:
        model = Fourier_Net(2, 2,  config.start_channel).cuda()  

    # choose the loss function for similarity
    assert config.choose_loss >= 0 and config.choose_loss <= 3, f"Expected choose_loss to be one of SAD (0), MSE (1), NCC (2) or SSIM (3), but got: {config.choose_loss}"
    if config.choose_loss == 1:
        loss_similarity = MSE().loss
    elif config.choose_loss == 0:
        loss_similarity = SAD().loss
    elif config.choose_loss == 2:
        loss_similarity = NCC(win=9)
    elif config.choose_loss == 3:
        ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=1, win_size=9)
        loss_similarity = SAD().loss
    loss_smooth = smoothloss

    # choose whether to use a diffeomorphic transform or not
    diff_transform = DiffeomorphicTransform(time_step=7).cuda()
    diffeo_name = 0
    if config.diffeo:
        diffeo_name = 1

    transform = SpatialTransform().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # load CMR training, validation and test data
    assert config.mode >= 0 and config.mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {config.mode}"
    if run == 0:
        print('Load in CMR data...')
        
        if config.dataset == "CMRxRecon":
            train_set = TrainDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', config.mode) 
            training_generator = Data.DataLoader(dataset=train_set, batch_size=config.bs, shuffle=True, num_workers=4)
            validation_set = ValidationDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', config.mode) 
            validation_generator = Data.DataLoader(dataset=validation_set, batch_size=config.bs, shuffle=True, num_workers=4)
            test_set = TestDatasetCMRxReconBenchmark('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode=config.mode)
            test_generator = Data.DataLoader(dataset=test_set, batch_size=config.bs, shuffle=False, num_workers=2)

        elif config.dataset == "ACDC":
            train_set = TrainDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', config.mode) 
            training_generator = Data.DataLoader(dataset=train_set, batch_size=config.bs, shuffle=True, num_workers=4)
            validation_set = ValidationDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', config.mode) 
            validation_generator = Data.DataLoader(dataset=validation_set, batch_size=config.bs, shuffle=True, num_workers=4)
            test_set = TestDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode=config.mode)
            test_generator = Data.DataLoader(dataset=test_set, batch_size=config.bs, shuffle=False, num_workers=2)
        
        print('Finished Loading!')

        epochs = log_TrainTest(wandb,model,model_name,diffeo_name,config.FT_size,config.learning_rate,config.start_channel,config.smth_lambda,config.choose_loss,config.diffeo,config.mode,epochs,optimizer,loss_similarity,loss_smooth,diff_transform,transform,training_generator,validation_generator,test_generator,True)
    else:
        log_TrainTest(wandb,model,model_name,diffeo_name,config.FT_size,config.learning_rate,config.start_channel,config.smth_lambda,config.choose_loss,config.diffeo,config.mode,epochs,optimizer,loss_similarity,loss_smooth,diff_transform,transform,training_generator,validation_generator,test_generator,False)
