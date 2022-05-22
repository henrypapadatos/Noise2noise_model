
import torch
from torch import nn
from Miniproject_2 import model
import numpy as np

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

subset_train = 2
subset_test = 10000

noisy_imgs_1 , noisy_imgs_2 = torch.load(r'C:\Users\Usuario\OneDrive - epfl.ch\Documents\EPFL\Semester II\Deep Learning\Project\Noise2noise_model\train_data.pkl')

noisy_imgs_1 = noisy_imgs_1[0:subset_train,:,:,:]
noisy_imgs_2 = noisy_imgs_2[0:subset_train,:,:,:]

test_imgs , clean_imgs = torch.load (r'C:\Users\Usuario\OneDrive - epfl.ch\Documents\EPFL\Semester II\Deep Learning\Project\Noise2noise_model\val_data.pkl')
test_imgs = test_imgs[0:subset_test,:,:,:]
clean_imgs = clean_imgs[0:subset_test,:,:,:]
input_rand = torch.randn(10, 20)
target_rand = torch.randn(10, 20)

test_input_rand = torch.randn(1, 20)
test_target_rand = torch.randn(1, 20)

model = model.Model()

model.train(input_rand, target_rand, test_input=test_input_rand,test_target=test_target_rand)
