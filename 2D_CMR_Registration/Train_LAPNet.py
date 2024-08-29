import math
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
from Models import *
import warnings
warnings.filterwarnings("ignore")
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float,
                    dest="learning_rate", default=1e-4, help="learning rate")
parser.add_argument("--epochs", type=int, dest="epochs", default=50,
                    help="number of total epochs")
parser.add_argument("--lambda", type=float, dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--dataset", type=str,
                    dest="dataset", default='ACDC',
                    help="dataset for training images: Select either ACDC, CMRxRecon or OASIS")
parser.add_argument("--mode", type=int, dest="mode", default=0,
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss", default=1,
                    help="choose similarity loss: MSE (0) or NCC (1)")
parser.add_argument("--earlyStop", type=int,
                    dest="earlyStop", default=3,
                    help="choose after how many epochs early stopping is applied")
opt = parser.parse_args()

learning_rate = opt.learning_rate
batch_size = 1
max_epochs = opt.epochs
smth_lambda = opt.smth_lambda
dataset = opt.dataset
mode = opt.mode
choose_loss = opt.choose_loss
earlyStop = opt.earlyStop

# load CMR training, validation and test data
assert mode >= 0 and mode <= 3, f"Expected mode to be one of fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3), but got: {mode}"
if dataset == 'ACDC':
    # load ACDC data
    train_set = TrainDatasetACDC_kSpace('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetACDC('/home/jmeyer/storage/students/janmeyer_711878/data/ACDC', mode) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=True, num_workers=4)
    #input_shape = [216,256]
elif dataset == 'CMRxRecon':
    # load CMRxRecon data
    train_set = TrainDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode) 
    training_generator = Data.DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=4)
    validation_set = ValidationDatasetCMRxRecon('/home/jmeyer/storage/students/janmeyer_711878/data/CMRxRecon', mode) 
    validation_generator = Data.DataLoader(dataset=validation_set, batch_size=1, shuffle=True, num_workers=4)
    #input_shape = [82,170]
else:
    print('Incorrect dataset selected!! Must be either ACDC or CMRxRecon...')

input_shape = train_set.__getitem__(0)[0].shape[1:3]

# path to save model parameters to
model_dir = './ModelParameters-{}/LAPNet_Mode_{}/'.format(dataset,mode)
model_dir_png = './ModelParameters-{}/LAPNet_Mode_{}_Png/'.format(dataset,mode)

if not isdir(model_dir):
    mkdir(model_dir)
    
if not isdir(model_dir_png):
    mkdir(model_dir_png)

csv_name = model_dir_png + 'LAPNet_Mode_{}.csv'.format(mode)
f = open(csv_name, 'w')
with f:
    if dataset == 'CMRxRecon':
        fnames = ['Epoch','MSE','SSIM'] 
    else:
        fnames = ['Epoch','Dice','MSE','SSIM']
    writer = csv.DictWriter(f, fieldnames=fnames)
    writer.writeheader()

Model = buildLAPNet_model_2D(input_shape)
Model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.999, lr=0.0),
                    loss=LAP_loss_function,
                    metrics=['accuracy'])

num_workers = 1

"""
if experiment_setup['weights_path']:
    weights_path = experiment_setup['weights_path']
    Model.load_weights(weights_path)
"""

# Checkpoints
checkpoints = ModelCheckpoint(
    f'model_dir/' + '{epoch:04d}' + '.pth',
    save_weights_only=True,
    save_freq="epoch")

# learning rate monitoring
scheduler = LearningRateScheduler(step_decay)

# define callbacks
callbacks_list = [scheduler, checkpoints]

# train
Model.fit_generator(generator=training_generator,
                        callbacks=callbacks_list,
                        verbose=1,
                        epochs=max_epochs,
                        workers=num_workers,
                        max_queue_size=20,
                        use_multiprocessing=True)