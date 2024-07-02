import wandb
import numpy as np
import torch
import wandb.sync
from Models import *
from Functions import *
import torch.utils.data as Data
import warnings
warnings.filterwarnings("ignore")


# compare Fourier-Net with Fourier-Net+
total_runs = 2
project_name = "Compare_Models"
names = ["Fourier-Net", "Fourier-Net+"]

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
                "choose_loss": 1,
                "mode": 0,
                "F_Net_plus": False,
                "FT_size": [24,24],
                #"dataset": "CMRxRecon",
                "dataset": "ACDC",
                "epochs": 100,
                "diffeo": False,
            }
        )
    elif run == 1:
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
                "choose_loss": 1,
                "mode": 0,
                "F_Net_plus": True,
                "FT_size": [24,24],
                #"dataset": "CMRxRecon",
                "dataset": "ACDC",
                "epochs": 100,
                "diffeo": False,
            }
        )
        
    # Copy your config 
    config = wandb.config
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")
    epochs = config.epochs

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

        epochs = log_TrainTest(wandb,config.F_Net_plus,config.FT_Size,config.learning_rate,config.start_channel,config.smth_lambda,config.choose_loss,config.diffeo,config.mode,config.epochs,training_generator,validation_generator,test_generator,True)
    else:
        log_TrainTest(wandb,config.F_Net_plus,config.FT_Size,config.learning_rate,config.start_channel,config.smth_lambda,config.choose_loss,config.diffeo,config.mode,config.epochs,training_generator,validation_generator,test_generator,False)
