import torch
from torch import nn
from torch import empty
import matplotlib.pyplot as plt
from torch.nn.functional import fold
from torch.nn.functional import unfold
import os

'''
ATTENTION: 

 To be able to run this code val_data.pkl and train_data.pkl must be next to this code file (same directory path)
'''

def psnr(denoised, ground_truth):
    #Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
    mse = torch.mean((denoised - ground_truth )** 2)
    psnr = -10 * torch . log10 ( mse + 10** -8)
    return psnr.item()

model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
                      nn.ReLU(),
                      nn.UpsamplingNearest2d(scale_factor = 2),
                      nn.Conv2d(32, 32, kernel_size=(3,3),padding=1),
                      nn.ReLU(),
                      nn.UpsamplingNearest2d(scale_factor = 2),
                      nn.Conv2d(32, 3, kernel_size=(3,3),padding=1),
                      nn.Sigmoid()
                      )

file_abs_path = os.path.dirname(os.path.abspath(__file__))
        
full_path = os.path.join(file_abs_path, 'train_data.pkl')
noisy_imgs_1 , noisy_imgs_2 = torch.load(full_path)

subset_train = 50000
subset_test = 10000
val_subste = 10000


max_pxl = noisy_imgs_1.max().item()
#We neet float type to 
noisy_imgs_1 = noisy_imgs_1[0:subset_train,:,:,:]
noisy_imgs_1 = noisy_imgs_1.float()
noisy_imgs_2 = noisy_imgs_2[0:subset_train,:,:,:]
noisy_imgs_2 = noisy_imgs_2.float()

file_abs_path = os.path.dirname(os.path.abspath(__file__))        
full_path = os.path.join(file_abs_path, 'val_data.pkl')
test_imgs , clean_imgs = torch.load(full_path)
test_imgs = test_imgs[0:subset_test,:,:,:]
test_imgs = test_imgs.float()
clean_imgs = clean_imgs[0:subset_test,:,:,:]
clean_imgs = clean_imgs.float()

#If your computer is equiped with a GPU, the computation will happen there
if torch.cuda.is_available():
    model.cuda()
    noisy_imgs_1 = noisy_imgs_1.cuda()
    noisy_imgs_2 = noisy_imgs_2.cuda()
    test_imgs = test_imgs.cuda()
    clean_imgs = clean_imgs.cuda()

lr = 2.5
batch_size=40
nb_epoch=200

optimizer = torch.optim.SGD(model.parameters(), lr)
criterion = nn.MSELoss()

if test_imgs!=None:
            initial_psnr = psnr(test_imgs/255, clean_imgs/255)
            print('Psnr value between clean and noisy images is: {:.02f}'.format(initial_psnr))

for e in range(nb_epoch):
    for input, targets in zip(noisy_imgs_1.split(batch_size),  
                              noisy_imgs_2.split(batch_size)):
        model.train()
        output = model(input/255)
        loss = criterion(output, targets/255)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    output_ = test_imgs/255
    output_ = model(output_ )
    output_ = torch.clip(output_*255, 0, 255)
    psnr_=psnr(output_/255, clean_imgs/255)
    print('Nb of epoch: {:d}    psnr: {:.02f}    loss: {:.08f}'.format(e, psnr_, loss))