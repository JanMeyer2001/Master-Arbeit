import matplotlib.pyplot as plt 
import csv 

foldername1 = 'Loss_1_Chan_8_Smth_0.01_LR_0.0001_Mode_0'
filename1 = foldername1 + '_Png/' + foldername1 + '.csv'

foldername2 = 'Loss_1_Chan_8_Smth_0.01_LR_0.0001_Mode_1'
filename2 = foldername2 + '_Png/' + foldername2 + '.csv'

foldername3 = 'Loss_1_Chan_8_Smth_0.01_LR_0.0001_Mode_2'
filename3 = foldername3 + '_Png/' + foldername3 + '.csv'

# for debugging
#path = './2D Fourier-Net+/'
#filename1 = path + foldername1 + '_Png/' + foldername1 + '.csv'

index1 = [] 
MSE1 = [] 
SSIM1 = []

with open(filename1,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for i, row in enumerate(lines): 
        if i>0:
            index1.append(int(row[0])) 
            MSE1.append(float(row[1])) 
            SSIM1.append(float(row[2]))

index2 = [] 
MSE2 = [] 
SSIM2 = []

with open(filename2,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for i, row in enumerate(lines): 
        if i>0:
            index2.append(int(row[0])) 
            MSE2.append(float(row[1])) 
            SSIM2.append(float(row[2]))

index3 = [] 
MSE3 = [] 
SSIM3 = []

with open(filename3,'r') as csvfile: 
    lines = csv.reader(csvfile, delimiter=',') 
    for i, row in enumerate(lines): 
        if i>0:
            index3.append(int(row[0])) 
            MSE3.append(float(row[1])) 
            SSIM3.append(float(row[2]))

plt.subplots(figsize=(10, 4))  
plt.title('Validation Benchmark Results', fontsize = 20) 
plt.axis('off')

plt.subplot(1,2,1)
#plt.title('MSE', fontsize = 16) 
plt.plot(index1, MSE1, color = 'g', label = "Fully Sampled") 
plt.plot(index2, MSE2, color = 'r', label = "Subsampled (Acc4)") 
plt.plot(index3, MSE3, color = 'b', label = "Subsampled (Acc8)") 
plt.xlabel('Iterations')
plt.ylabel('MSE') 
plt.legend()

plt.subplot(1,2,2)
#plt.title('SSIM', fontsize = 16) 
plt.plot(index1, SSIM1, color = 'g', label = "Fully Sampled") 
plt.plot(index2, SSIM2, color = 'r', label = "Subsampled (Acc4)") 
plt.plot(index3, SSIM3, color = 'b', label = "Subsampled (Acc8)") 
plt.xlabel('Iterations') 
plt.ylabel('SSIM') 
plt.legend()

plt.savefig('/home/jmeyer/storage/students/janmeyer_711878/Master-Arbeit/Thesis/Images/ValidationBenchmark-' + foldername1 + '+Mode1+Mode2.png')
plt.close()