import wandb
import numpy as np
import torch
import wandb.sync
from Models import *
from Functions import *
import torch.utils.data as Data
import warnings
warnings.filterwarnings("ignore")
from info_nce import InfoNCE

# compare different FT crop sizes
total_runs = 2 #4
project_name = "Fourier-Net_ContrastiveLearning"
names = ["Lambda-0.01", "Lambda-0.05"] #"Lambda-0.0001", "Lambda-0.0005", "Lambda-0.001", "Lambda-0.005"

# init array for test results
results_test = np.zeros((total_runs,7))

for run in range(total_runs):
    if run == 0:
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
                "FT_size": [24,24],
                "choose_loss": 5,
                "mode": 1,
                "model": 0,
                "epochs": 6,
                "diffeo": 0,
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
                "smth_lambda": 0.05,
                "FT_size": [24,24],
                "choose_loss": 5,
                "mode": 1,
                "model": 0,
                "epochs": 6,
                "diffeo": 0,
            }
        )
    """
    elif run == 2:
        print(names[run])
        wandb.init(
            project = project_name,
            name = names[run],
            # track hyperparameters and run metadata
            config={
                "bs": 1,
                "learning_rate": 1e-4,
                "start_channel": 16,
                "smth_lambda": 0.001,
                "FT_size": [24,24],
                "choose_loss": 5,
                "mode": 1,
                "model": 0,
                "epochs": 6,
                "diffeo": 0,
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
                "start_channel": 16,
                "smth_lambda": 0.005,
                "FT_size": [24,24],
                "choose_loss": 5,
                "mode": 1,
                "model": 0,
                "epochs": 6,
                "diffeo": 0,
            }
        )       
    """    
    # Copy config 
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config.epochs

    # choose the model
    assert config.model == 0 or config.model == 1 or config.model == 2, f"Expected model to be either 0, 1 or 2, but got: {config.model}"
    assert config.diffeo == 0 or config.diffeo == 1, f"Expected diffeo to be either 0 or 1, but got: {config.diffeo}"
    if config.model == 0:
        model = Fourier_Net(2, 2, config.start_channel, config.diffeo).cuda() 
    elif config.model == 1:
        assert config.FT_size[0] > 0 and config.FT_size[0] <= 40 and config.FT_size[1] > 0 and config.FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{config.FT_size[0]}, {config.FT_size[1]}]"
        model = Fourier_Net_plus(2, 2, config.start_channel, config.diffeo, config.FT_size).cuda() 
    elif config.model == 2:
        assert config.FT_size[0] > 0 and config.FT_size[0] <= 40 and config.FT_size[1] > 0 and config.FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{config.FT_size[0]}, {config.FT_size[1]}]"
        model = Cascade(2, 2, config.start_channel, config.diffeo, config.FT_size).cuda() 

    # choose the loss function for similarity
    assert config.choose_loss >= 0 and config.choose_loss <= 5, f"Expected choose_loss to be one of SAD (0), MSE (1), NCC (2) or SSIM (3), but got: {config.choose_loss}"
    if config.choose_loss == 1:
        loss_similarity = MSE().loss
        loss_smooth = smoothloss
    elif config.choose_loss == 0:
        loss_similarity = SAD().loss
        loss_smooth = smoothloss
    elif config.choose_loss == 2:
        loss_similarity = NCC(win=9)
        loss_smooth = smoothloss
    elif config.choose_loss == 3:
        loss_similarity = torch.nn.L1Loss()
        loss_smooth = smoothloss    
    elif config.choose_loss == 4:
        loss_similarity = RelativeL2Loss()  
        loss_smooth = smoothloss
    elif config.choose_loss == 5:
        loss_similarity = MSE().loss
        loss_smooth = InfoNCE() # contrative learning loss

    transform = SpatialTransform().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # load CMR training, validation and test data
    assert config.mode >= 0 and config.mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {config.mode}"
    if run == 0:
        print('Load in CMR data...')
        train_set = TrainDatasetACDC_ContrastiveLearning('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', config.mode) 
        training_generator = Data.DataLoader(dataset=train_set, batch_size=config.bs, shuffle=True, num_workers=4)
        validation_set = ValidationDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', config.mode) 
        validation_generator = Data.DataLoader(dataset=validation_set, batch_size=config.bs, shuffle=False, num_workers=4)
        test_set = TestDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode=config.mode)
        test_generator = Data.DataLoader(dataset=test_set, batch_size=config.bs, shuffle=False, num_workers=2)
        print('Finished Loading!')

        log_TrainTest(wandb,model,config.model,config.diffeo,'ACDC',config.FT_size,config.learning_rate,config.start_channel,config.smth_lambda,config.choose_loss,config.mode,epochs,optimizer,loss_similarity,loss_smooth,transform,training_generator,validation_generator,test_generator,device,False)
    else:
        log_TrainTest(wandb,model,config.model,config.diffeo,'ACDC',config.FT_size,config.learning_rate,config.start_channel,config.smth_lambda,config.choose_loss,config.mode,epochs,optimizer,loss_similarity,loss_smooth,transform,training_generator,validation_generator,test_generator,device,False)
