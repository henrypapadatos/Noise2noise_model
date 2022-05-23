# -*- coding: utf-8 -*-
"""
Created on Sat May 21 18:49:25 2022

@author: papad
"""
import torch
import model
import torch.nn.functional as F

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



