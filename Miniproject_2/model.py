
import torch
from torch import nn
from torch import empty
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import fold
from torch.nn.functional import unfold
import os


class Module ( object ) :
    def forward ( self , x ) :
        raise NotImplementedError
    def backward ( self , y) :
        raise NotImplementedError
    def SGD ( self , lr):
        pass
    def param ( self ) :
        pass

class Relu(Module):
    def forward ( self , x) :
        return x*(x>0)
    def backward ( self , y  ) :
        return (y>0)

class Sigmoid(Module):
    def forward (self , x) :
        return 1/(1+(-x).exp())
    def backward ( self , y) :
        return (-y).exp()/(1+(-y).exp()).pow(2)
    
class MSE(Module):
    def forward (self , x, x_target) :
        return (x - x_target).pow(2).mean()
    def backward ( self , y, y_target) :
        return 2* (y-y_target).mean()

class Sequential(Module): #MODIFY: supposed run sequentially all the stuff you are asking it to 
    def __init__(self, *type_layer):
        #super().__init__()
        self.type_layer = type_layer
        self.loss = MSE()
    # ex: 1 Sequential ( Conv( stride 2) , ReLU , Conv ( stride 2) , ReLU , Upsampling , ReLU , Upsampling , Sigmoid )

    def forward (self,x):
        # for looop from start to begning
        #print("y",x)
        x_ = x
        for layer in self.type_layer:
            #later we will have to call layer.param
            x_ = layer.forward(x_)
            #layer.forward(layer.param())
            #print("forward_layer",x_)
        return x_
    
    def backward (self,y, target):
        #print("y",y)
        y_ =y 
        y_ = self.loss.backward(y_, target)
        for layer in reversed(self.type_layer):
            y_ = layer.backward(y_)
            #print("backward_layer",y_)

        return y_

    def param ( self ) :
        return []



class Conv2d(Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride = 1, padding = 1, dilation= 1):
        super().__init__()

        if type(kernel_size) == int:
            kernel_size = (kernel_size,kernel_size)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_channel = output_channel
        
        k = np.sqrt(1/(input_channel*kernel_size[0]*kernel_size[1]))
        self.weight = torch.empty(output_channel, input_channel,kernel_size[0],kernel_size[1]).uniform_(-k,k)
        self.bias = torch.empty(output_channel).uniform_(-k,k)
        self.gradweight = torch.empty(output_channel, input_channel,kernel_size[0],kernel_size[1])*0
        self.gradbias = torch.empty(output_channel)*0
        
    #initialize them here and gradients should at 0

    def conv (self,x):
        weight2 = self.weight.copy
        h_in, w_in = x.shape[2:]
        h_out = ((h_in+2*self.padding-self.dilation*(self.kernel_size[0]-1)-1)/self.stride+1)
        w_out = ((w_in+2*self.padding-self.dilation*(self.kernel_size[1]-1)-1)/self.stride+1)
        unfolded = torch.nn.functional.unfold(x, kernel_size = self.kernel_size, dilation = self.dilation, padding = self.padding, stride = self.stride)
        out = unfolded.transpose(1, 2).matmul(weight2.view(weight2.size(0), -1).t()).transpose(1, 2) + self.bias.view(1,-1,1)
        output = out.view(x.shape[0], self.output_channel, int(h_out), int(w_out))
        return output
        
    def forward (self,x):
        self.conv(x)
        return[]

    def backward (self,y):
        #taking the deriative of "linear conv"
        return[]
    def param ( self ) :
        return [self.weight, self.bias, self.gradweight, self.gradbias]

class Model():
    def __init__(self):
        ## instantiate model + optimizer + loss function + any other stuff you need

        self.lr = 0.002
        self.nb_epoch = 100
        self.batch_size = 1000

        self.model = Sequential()
           
        #self.optimizer = 
        self.criterion = MSE()
    
    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model$
        #full_path = os.path.join('Miniproject_2', 'bestmodel.pth')
        #self.model.load_state_dict(torch.load(full_path,map_location=torch.device('cpu')))
        pass 
    def train(self, train_input, train_target, test_input, test_target, vizualisation_flag = False):
        pass
    def predict(self, input_imgs):
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network .
        
        #normalize image between 0 and 1
        input_imgs_ = input_imgs/255
        output = self.model.forward(input_imgs_)
        return output
    
    def psnr(self, denoised, ground_truth):
        #Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
        mse = torch.mean((denoised - ground_truth )** 2)
        psnr = -10 * torch . log10 ( mse + 10** -8)
        return psnr.item()
