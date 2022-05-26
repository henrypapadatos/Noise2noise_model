# -*- coding: utf-8 -*-

import sys
sys.path.append(r'C:\Users\papad\OneDrive\Documents\MA2\Deep_learning\Project\Noise2noise_model\Miniproject_1\others')
import torch
from torch import nn
from Miniproject_1 import model


subset_train = 10000
subset_test = 10000

noisy_imgs_1 , noisy_imgs_2 = torch.load('train_data.pkl')

noisy_imgs_1 = noisy_imgs_1[0:subset_train,:,:,:]
noisy_imgs_2 = noisy_imgs_2[0:subset_train,:,:,:]

test_imgs , clean_imgs = torch.load ('val_data.pkl')
test_imgs = test_imgs[0:subset_test,:,:,:]
clean_imgs = clean_imgs[0:subset_test,:,:,:]

noise2noise = model.Model()

noise2noise.train(noisy_imgs_1, noisy_imgs_2, num_epochs=100, test_input=test_imgs, test_target=clean_imgs, vizualisation_flag=True)    
    
noise2noise.load_pretrained_model()

output = noise2noise.predict(noisy_imgs_1[0:2])
print('here')
model.display_img(output[0])
