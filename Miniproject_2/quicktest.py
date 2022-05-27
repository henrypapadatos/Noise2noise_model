# -*- coding: utf-8 -*
"""
Created on Sat May 21 18:49:25 2022

@author: papad
"""
import torch
import model
import torch.nn.functional as F
from torch import set_grad_enabled

set_grad_enabled(True)
x = torch.randn(1, 3, 32, 32)

'''
sigmoid = model.Sigmoid()
print(torch.allclose(sigmoid.forward(x), torch.sigmoid(x)))



unfolded = torch.nn.functional.unfold(x, kernel_size = 2, dilation = 2, padding = 0, stride = 1)
print(unfolded.shape)

Sequential = model.Sequential
seq = Sequential(conv, sigmoid)
print(torch.allclose(seq.forward(x), F.conv2d(x, conv.weight, conv.bias).sigmoid()))
'''
Conv2d = model.Conv2d
conv = Conv2d(3, 3, 3, padding=1, dilation =2, stride =2)
print(torch.allclose(conv.forward(x), F.conv2d(x, conv.weight, conv.bias, padding=1, dilation =2, stride =2)))



from torch import randn
in_channels = 3
out_channels = 7
kernel_size = (2, 2)

from torch.nn.functional import conv2d

torch.manual_seed(0)

input = randn((1, in_channels, 7, 7), dtype=torch.double, requires_grad=True)


our_conv = Conv2d(in_channels, out_channels, kernel_size)
weight = our_conv.weight
bias = our_conv.bias
pytorch_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
pytorch_conv.weight = torch.nn.Parameter(weight)
pytorch_conv.bias = torch.nn.Parameter(bias)

our_y_pred = our_conv.forward(input.float())
pytorch_conv = pytorch_conv.float()
pytorch_y_pred = pytorch_conv(input.float())
pytorch_y_pred.retain_grad()

print(torch.allclose(our_y_pred, pytorch_y_pred)) #test forward

gradient = torch.rand_like(our_y_pred)

pytorch_y_pred.backward(gradient=gradient.float())

dl_dx = our_conv.backward(gradient.float())

#check weight
print(torch.allclose(pytorch_conv.weight.grad, our_conv.gradweight))

#check dl_db
print(torch.allclose(pytorch_conv.bias.grad, our_conv.gradbias))

#check dl_dx
print(torch.allclose(input.grad.float(), dl_dx))

Conv2d = model.Conv2d
conv = Conv2d(3, 3, 3, padding=1, dilation =2, stride =2)
print(torch.allclose(conv.forward(x), F.conv2d(x, conv.weight, conv.bias, padding=1, dilation =2, stride =2)))

print('Now testing MSE')
from torch import randn
in_channels = 3
out_channels = 7
kernel_size = (2, 2)

import torch

torch.manual_seed(0)

input = randn((1, in_channels, 7, 7), dtype=torch.float, requires_grad=True)
x_target = randn((1, in_channels, 7, 7), dtype=torch.float, requires_grad=True)


our_conv = model.MSE()
pytorch_conv = torch.nn.MSELoss()

our_y_pred = our_conv.forward(input, x_target)
pytorch_conv = pytorch_conv.float()
pytorch_y_pred = pytorch_conv(input, x_target)
pytorch_y_pred.retain_grad()

print(torch.allclose(our_y_pred, pytorch_y_pred)) #test forward

#gradient = torch.rand_like(our_y_pred)

pytorch_y_pred.backward()

dl_dx = our_conv.backward()

#check dl_dx
print(torch.allclose(input.grad.float(), dl_dx))
