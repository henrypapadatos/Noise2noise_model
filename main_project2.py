
import torch
from torch import nn
from Miniproject_2 import model
import numpy as np
import math
from torch import empty
from torch import set_grad_enabled
import random
set_grad_enabled(False)
'''

input_tensor = torch.normal(0, 1, size=(3,2,2), requires_grad=True)
target = torch.normal(0, 1, size=(3,2,2), requires_grad=True)

sequential_torch = nn.Sequential(nn.ReLU(),nn.Sigmoid())

criterion = nn.MSELoss()

y = sequential_torch(input_tensor)

loss = criterion(y, target)

# loss_val = loss(y, target)

loss.backward()
with torch.no_grad():
    loss.grad
'''

# loss = MSE()

# our_loss = loss.forward(input_tensor, target)
# torch_loss = nn.MSELoss()

# torch_loss_val = torch_loss(input_tensor, target)

# print(our_loss-torch_loss_val)

# relu = Relu()
# sigmoid = Sigmoid()
# sequential = Seq(relu,sigmoid)

# sequential_torch = nn.Sequential(nn.ReLU(),nn.Sigmoid())


# y = sequential.forward(input_tensor)

# y_torch = sequential_torch(input_tensor)

# print(y-y_torch)

# y = sequental.backward(y)

# # y_torch.backward(gradient)
# print(y-torch.gradient(y_torch))
################################################################ OUR DATASET ##########################################################
subset_train = 10000
subset_test = 100

noisy_imgs_1 , noisy_imgs_2 = torch.load('train_data.pkl')

noisy_imgs_1 = noisy_imgs_1[0:subset_train,:,:,:]
noisy_imgs_2 = noisy_imgs_2[0:subset_train,:,:,:]

test_imgs , clean_imgs = torch.load ('val_data.pkl')
test_imgs = test_imgs[0:subset_test,:,:,:]
clean_imgs = clean_imgs[0:subset_test,:,:,:]


################################################################ RANDOM DATASET FOR LINEAR #############################################

# input_rand = torch.randn(10, 20)
# target_rand = torch.randn(10, 20)

# test_input_rand = torch.randn(1, 20)
# test_target_rand = torch.randn(1, 20)

########################################################### DATASET LIKE IN EXERCISE SESSION FOR LINEAR #################################

# def reshapeLabel(label):
#     """'
#     Reshape 1-D [0,1,...] to 2-D [[1,-1],[-1,1],...].
#     """
#     n = label.size(0)
#     y = empty(n, 2)
#     y[:, 0] = 2 * (0.5 - label)
#     y[:, 1] = - y[:, 0]
#     return y.float()

# def generate_disk_dataset(nb_points):
#     """
#     Inspired by the practical 5, this method generates points uniformly in the unit square, with label 1 if the points are in the disc centered at (0.5,0.5) of radius 1/sqrt(2pi), and 0 otherwise
#     """
#     input = empty(nb_points,2).uniform_(0,1)
#     label = input.sub(0.5).pow(2).sum(1).lt(1./2./math.pi).float()
#     target = reshapeLabel(label)
#     return input,target

# # Generate train set and test set
# nb_points = 1000
# input_rand, target_rand = generate_disk_dataset(nb_points)
# test_input_rand ,test_target_rand = generate_disk_dataset(nb_points)
##############################################################################################################"""
# def psnr(denoised, ground_truth):
#     #Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
#     mse = torch.mean((denoised - ground_truth )** 2)
#     psnr = -10 * torch . log10 ( mse + 10** -8)
#     return psnr.item()

# model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3,3), stride=2, padding=1),
#                       nn.ReLU(),
#                       nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1),
#                       nn.ReLU(),
#                       #nn.Upsample(scale_factor= 2, mode = "nearest"),
#                       nn.UpsamplingNearest2d(scale_factor = 2),
#                       nn.Conv2d(32, 32, kernel_size=(3,3),padding=1),
#                       nn.ReLU(),
#                       nn.UpsamplingNearest2d(scale_factor = 2),
#                       #nn.Upsample(scale_factor= 2, mode = "nearest")
#                       nn.Conv2d(32, 3, kernel_size=(3,3),padding=1),
#                       nn.Sigmoid()
#                       )

# lr = 1.0
# batch_size=500
# nb_epoch=100

# optimizer = torch.optim.SGD(model.parameters(), lr)
# criterion = nn.MSELoss()

# if test_imgs!=None:
#             initial_psnr = psnr(test_imgs/255, clean_imgs/255)
#             print('Psnr value between clean and noisy images is: {:.02f}'.format(initial_psnr))

# for e in range(nb_epoch):
#     for input, targets in zip(noisy_imgs_1.split(batch_size),  
#                               noisy_imgs_2.split(batch_size)):
#         model.train()
#         output = model(input/255)
#         loss = criterion(output, targets/255)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     model.eval()
#     output_ = test_imgs/255
#     output_ = model(output_ )
#     output_ = torch.clip(output_*255, 0, 255)
#     psnr_=psnr(output_/255, clean_imgs/255)
#     print('Nb of epoch: {:d}    psnr: {:.02f}    loss: {:.08f}'.format(e, psnr_, loss))

        



            
            
############################################################################################################################################

model = model.Model()

# model.train(input_rand, target_rand, test_input=test_input_rand,test_target=test_target_rand)
model.train(noisy_imgs_1, noisy_imgs_2, test_input=test_imgs,test_target=clean_imgs)

