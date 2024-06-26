import wandb
import numpy as np
import torch
import wandb.sync
from Models import *
from Functions import *
import torch.utils.data as Data
import time
from skimage.metrics import structural_similarity, mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import os

# compare Fourier-Net with Fourier-Net+
total_runs = 2
project_name = "Compare_Models"
names = ["Diff-Fourier-Net", "Diff-Fourier-Net+"]

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
                "FT_size": [24,24],
                "choose_loss": 1,
                "mode": 0,
                "F_Net_plus": False,
                "dataset": "CMRxRecon",
                "datapath": '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon',
                "epochs": 100,
                "diffeo": True,
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
                "FT_size": [24,24],
                "choose_loss": 1,
                "mode": 0,
                "F_Net_plus": True,
                "dataset": "CMRxRecon",
                "datapath": '/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon',
                "epochs": 100,
                "diffeo": True,
            }
        )
        
    # Copy your config 
    config = wandb.config

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    # choose the model
    model_name = 0
    if config.F_Net_plus:
        assert config.FT_size[0] > 0 and config.FT_size[0] <= 40 and config.FT_size[1] > 0 and config.FT_size[1] <= 84, f"Expected FT size smaller or equal to [40, 84] and larger than [0, 0], but got: [{config.FT_size[0]}, {config.FT_size[1]}]"
        model = Cascade(2, 2, config.start_channel, config.FT_size).cuda() 
        model_name = 1
    else:
        model = Fourier_Net(2, 2, config.start_channel).cuda()  

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
    diffeo_name = 0
    if config.diffeo:
        diff_transform = DiffeomorphicTransform(time_step=7).cuda()
        diffeo_name = 1

    transform = SpatialTransform().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # load CMR training, validation and test data
    assert config.mode >= 0 and config.mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {config.mode}"
    if run == 0:
        print('Load in CMR data...')
        train_set = TrainDatasetCMR(config.datapath, config.mode) 
        training_generator = Data.DataLoader(dataset=train_set, batch_size=config.bs, shuffle=True, num_workers=4)
        validation_set = ValidationDatasetCMR(config.datapath, config.mode) 
        validation_generator = Data.DataLoader(dataset=validation_set, batch_size=config.bs, shuffle=True, num_workers=4)
        test_set = TestDatasetCMRBenchmark(data_path=config.datapath, mode=config.mode)
        test_generator = Data.DataLoader(dataset=test_set, batch_size=config.bs, shuffle=False, num_workers=2)
        print('Finished Loading!')

    # path to save model parameters to
    model_dir = './ModelParameters/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_Smth_{}_LR_{}_Mode_{}_Pth/'.format(model_name,diffeo_name,config.choose_loss,config.start_channel,config.smth_lambda, config.learning_rate, config.mode)
    model_dir_png = './ModelParameters/Model_{}_Diffeo_{}_Loss_{}_Chan_{}_Smth_{}_LR_{}_Mode_{}_Png/'.format(model_name,diffeo_name,config.choose_loss,config.start_channel,config.smth_lambda, config.learning_rate, config.mode)

    if not os.path.isdir(model_dir_png):
        os.mkdir(model_dir_png)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    ##############
    ## Training ##
    ##############

    print('\nStarted training on ', time.ctime())

    for epoch in range(config.epochs):
        losses = np.zeros(training_generator.__len__())
        for i, image_pair in enumerate(training_generator):
            mov_img = image_pair[0].cuda().float()
            fix_img = image_pair[1].cuda().float()
            
            f_xy = model(mov_img, fix_img)
            if config.diffeo:
                Df_xy = diff_transform(f_xy)
            else:
                Df_xy = f_xy
            grid, warped_mov = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
            
            loss1 = loss_similarity(fix_img, warped_mov) 
            loss5 = loss_smooth(Df_xy)
            
            loss = loss1 + config.smth_lambda * loss5
            losses[i] = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ################
        ## Validation ##
        ################

        with torch.no_grad():
            model.eval()
            MSE_Validation = []
            SSIM_Validation = []
            
            for mov_img, fix_img in validation_generator: 
                fix_img = fix_img.cuda().float()
                mov_img = mov_img.cuda().float()
                
                f_xy = model(mov_img, fix_img)
                if config.diffeo:
                    Df_xy = diff_transform(f_xy)
                else:
                    Df_xy = f_xy
                grid, warped_mov = transform(mov_img, Df_xy.permute(0, 2, 3, 1))
                
                # calculate MSE and SSIM
                MSE_Validation.append(mean_squared_error(warped_mov[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy()))
                SSIM_Validation.append(structural_similarity(warped_mov[0,0,:,:].cpu().numpy(), fix_img[0,0,:,:].cpu().numpy(), data_range=1))
        
            # calculate mean of validation metrics
            Mean_MSE = np.mean(MSE_Validation)
            Mean_SSIM = np.mean(SSIM_Validation)
            
            # log loss and validation metrics to wandb
            wandb.log({"Loss": np.mean(losses), "MSE": Mean_MSE, "SSIM": Mean_SSIM})
            
            # save and log model     
            modelname = 'SSIM_{:.6f}_MSE_{:.6f}_Epoch_{:04d}.pth'.format(Mean_SSIM, Mean_MSE, epoch)
            save_checkpoint(model.state_dict(), model_dir, modelname)
            wandb.log_model(path=model_dir, name=modelname)

            # save image
            sample_path = os.path.join(model_dir_png, 'Epoch_{:04d}-images.jpg'.format(epoch))
            save_flow(mov_img, fix_img, warped_mov, grid.permute(0, 3, 1, 2), sample_path)
            print("epoch {0:d}/{1:d} - MSE_val: {2:.6f}, SSIM_val: {3:.5f}".format(epoch, config.epochs, Mean_MSE, Mean_SSIM))
            
    print('Training ended on ', time.ctime())

    #############
    ## Testing ##
    #############
    
    print('\nTesting started on ', time.ctime())

    model.eval()
    transform.eval()
    MSE_test = []
    SSIM_test = []
    NegJ=[]
    times = []

    for mov_img_fullySampled, fix_img_fullySampled, mov_img_subSampled, fix_img_subSampled in test_generator: 
        with torch.no_grad():
            start = time.time()
            # calculate displacement on subsampled data
            V_xy = model(mov_img_subSampled.float().to(device), fix_img_subSampled.float().to(device))
            # but warp fully sampled data
            __, warped_mov_img_fullySampled = transform(mov_img_fullySampled.float().to(device), V_xy.permute(0, 2, 3, 1), mod = 'nearest')
            
            # get inference time
            inference_time = time.time()-start
            times.append(inference_time)

            # convert to numpy
            warped_mov_img_fullySampled = warped_mov_img_fullySampled[0,0,:,:].cpu().numpy()
            fix_img_fullySampled = fix_img_fullySampled[0,0,:,:].cpu().numpy()
            
            # calculate metrics on fully sampled images
            MSE_test.append(mean_squared_error(warped_mov_img_fullySampled, fix_img_fullySampled))
            SSIM_test.append(structural_similarity(warped_mov_img_fullySampled, fix_img_fullySampled, data_range=1))

            hh, ww = V_xy.shape[-2:]
            V_xy = V_xy.detach().cpu().numpy()
            V_xy[:,0,:,:] = V_xy[:,0,:,:] * hh / 2
            V_xy[:,1,:,:] = V_xy[:,1,:,:] * ww / 2

            jac_det = jacobian_determinant_vxm(V_xy[0, :, :, :])
            negJ = np.sum(jac_det <= 0) / 160 / 192 * 100
            NegJ.append(negJ)

    # get results for the table
    results_test[run,0] = np.mean(MSE_test)
    results_test[run,1] = np.std(MSE_test)

    results_test[run,2] = np.mean(SSIM_test)
    results_test[run,3] = np.std(SSIM_test)

    results_test[run,4] = np.mean(NegJ)
    results_test[run,5] = np.std(NegJ)

    results_test[run,6] = np.mean(times)

    # print results
    print('     Mean inference time: {0:.4f} seconds\n     MSE: {1:.6f} +- {2:.6f}\n     SSIM: {3:.5f} +- {4:.5f}\n     DetJ<0 %: {5:.4f} +- {6:.4f}'.format(results_test[run,6], results_test[run,0], results_test[run,1], results_test[run,2], results_test[run,3], results_test[run,4], results_test[run,5]))
    # save results to wandb table
    TestResults = wandb.Table(columns=["Run", "Mean MSE", "Std MSE", "Mean SSIM", " Std SSIM", "Mean NegJ", " Std NegJ", "Mean Inference Time"], data=[["Diff-Fourier-Net", results_test[0,0], results_test[0,1], results_test[0,2], results_test[0,3], results_test[0,4], results_test[0,5], results_test[0,6]], ["Diff-Fourier-Net+", results_test[1,0], results_test[1,1], results_test[1,2], results_test[1,3], results_test[1,4], results_test[1,5], results_test[1,6]]])
    wandb.log({"Test Results": TestResults})
    
    print('Testing ended on ', time.ctime())
    
    # Mark the run as finished
    wandb.finish()
