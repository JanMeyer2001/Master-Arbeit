import matplotlib.pyplot as plt 
import csv 
from argparse import ArgumentParser
  
parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--smth_lambda", type=float,
                    dest="smth_lambda", default=0.01,
                    help="lambda loss: suggested range 0.1 to 10")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=8,
                    help="number of start channels")
parser.add_argument("--choose_loss", type=int,
                    dest="choose_loss",
                    default=1,
                    help="choose similarity loss: SAD (0), MSE (1), NCC (2), SSIM (3)")
parser.add_argument("--mode", type=int,
                    dest="mode",
                    default='0',
                    help="choose dataset mode: fully sampled (0), 4x accelerated (1), 8x accelerated (2) or 10x accelerated (3)")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
smth_lambda = opt.smth_lambda
choose_loss = opt.choose_loss
mode = opt.mode

foldername = 'Loss_' + str(choose_loss) + '_Chan_' + str(start_channel) + '_Smth_' + str(smth_lambda) + '_LR_' + str(lr) + '_Mode_' + str(mode)
filename = foldername + '_Png/' + foldername + '.csv'
# for debugging
#path = './2D Fourier-Net+/'
#filename = path + foldername + '_Png/' + foldername + '.csv'

index = [] 
MSE = [] 
SSIM = []

with open(filename,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for i, row in enumerate(lines): 
        if i>0:
            index.append(int(row[0])) 
            MSE.append(float(row[1])) 
            SSIM.append(float(row[2]))

plt.subplots(figsize=(8, 4))  
plt.title('Validation Benchmark', fontsize = 20) 
plt.axis('off')

plt.subplot(1,2,1)
plt.plot(index, MSE, color = 'g', label = "MSE") #
plt.xlabel('Iterations')
plt.legend()
#plt.ylabel('MSE') 
#plt.ylim([0,0.001])

plt.subplot(1,2,2)
plt.plot(index, SSIM, color = 'r', label = "SSIM") #
plt.xlabel('Iterations') 
plt.legend()
#plt.ylabel('SSIM') 
#plt.ylim([0.75,1])

plt.savefig('/home/jmeyer/storage/students/janmeyer_711878/Master-Arbeit/Thesis/Images/ValidationBenchmark-' + foldername + '.png')
plt.close()