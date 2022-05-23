
import torch
from torch import nn
from Miniproject_2 import model
import numpy as np
import math
from torch import empty
from torch import set_grad_enabled
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

# y = sequential.backward(y)

# # y_torch.backward(gradient)
# print(y-torch.gradient(y_torch))
################################################################ OUR DATASET ##########################################################
subset_train = 2
subset_test = 10000

noisy_imgs_1 , noisy_imgs_2 = torch.load(r'C:\Users\Usuario\OneDrive - epfl.ch\Documents\EPFL\Semester II\Deep Learning\Project\Noise2noise_model\train_data.pkl')

noisy_imgs_1 = noisy_imgs_1[0:subset_train,:,:,:]
noisy_imgs_2 = noisy_imgs_2[0:subset_train,:,:,:]

test_imgs , clean_imgs = torch.load (r'C:\Users\Usuario\OneDrive - epfl.ch\Documents\EPFL\Semester II\Deep Learning\Project\Noise2noise_model\val_data.pkl')
test_imgs = test_imgs[0:subset_test,:,:,:]
clean_imgs = clean_imgs[0:subset_test,:,:,:]

################################################################ RANDOM DATASET FOR LINEAR #############################################

input_rand = torch.randn(10, 20)
target_rand = torch.randn(10, 20)

test_input_rand = torch.randn(1, 20)
test_target_rand = torch.randn(1, 20)

########################################################### DATASET LIKE IN EXERCISE SESSION FOR LINEAR #################################

def reshapeLabel(label):
    """'
    Reshape 1-D [0,1,...] to 2-D [[1,-1],[-1,1],...].
    """
    n = label.size(0)
    y = empty(n, 2)
    y[:, 0] = 2 * (0.5 - label)
    y[:, 1] = - y[:, 0]
    return y.float()

def generate_disk_dataset(nb_points):
    """
    Inspired by the practical 5, this method generates points uniformly in the unit square, with label 1 if the points are in the disc centered at (0.5,0.5) of radius 1/sqrt(2pi), and 0 otherwise
    """
    input = empty(nb_points,2).uniform_(0,1)
    label = input.sub(0.5).pow(2).sum(1).lt(1./2./math.pi).float()
    target = reshapeLabel(label)
    return input,target

# Generate train set and test set
nb_points = 1000
input_rand, target_rand = generate_disk_dataset(nb_points)
test_input_rand ,test_target_rand = generate_disk_dataset(nb_points)

############################################################################################################################################

model = model.Model()

model.train(input_rand, target_rand, test_input=test_input_rand,test_target=test_target_rand)
