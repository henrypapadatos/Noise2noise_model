
import torch
from torch import empty
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import fold
from torch.nn.functional import unfold

class Module ( object ) :
    def forward ( self , x ) :
        raise NotImplementedError
    def backward ( self , y) :
        raise NotImplementedError
    def param ( self ) :
        return []


torch.set_grad_enabled(False)

class Relu():
    def forward ( self , x) :
        return max(0.0, x)
    def backward ( self , y  ) :
        if y>0:
            return 1
        else:
            return 0

    def param ( self ) :
        return []

class Sigmoid():
    def forward (self , x) :
        return 1/(1+(-x).exp())
    def backward ( self , y) :
        return (-y).exp()/(1+(-y).exp()).pow(2)
    def param ( self ) :
        return []
    
class MSE():
    def forward (self , x, x_target) :
        return (x - x_target).pow(2).mean()
    def backward ( self , y, y_target) :
        return 2* (y-y_target).mean()
    def param ( self ) :
        return []
    
class Seq(): #MODIFY: supposed run sequentially all the stuff you are asking it to 
    # ex: 1 Sequential ( Conv ( stride 2) , ReLU , Conv ( stride 2) , ReLU , Upsampling , ReLU , Upsampling , Sigmoid )
    def forward (self,x):
        #ConvLayer(stride = 2)
        Relu.forward(x)
        #ConvLayer(stride = 2)
        Relu.forward(x)
        #Upsampling()
        Relu.forward(x)
        #Upsampling()
        Sigmoid.forward(x)
    
    def backward (self,y):
        #ConvLayer(stride = 2)
        Relu.backward(y)
        #ConvLayer(stride = 2)
        Relu.backward(y)
        #Upsampling()
        Relu.backward(y)
        #Upsampling()
        Sigmoid.backward(y)

    def param ( self ) :
        return []


class ConvLayer():
    def __init__(self, input_channel, output_channel, kernel_size, stride):
        super().__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        
        k = np.sqrt(1/(iput_channel*kernel_size[0]*kernel_size[1]))
        self.weights = torch.empty(output_channel, input_channel,kernel_size[0],kernel_size[1]).uniform_(-k,k)
        self.bias = torch.empty(output_channel).uniform_(-k,k)
        self.gradweights = torch.empty(output_channel, input_channel,kernel_size[0],kernel_size[1])*0
        self.gradbias = torch.empty(output_channel)*0
        
    #initialize them here and gradients should at 0 
    def forward (self,x):
        # unfold(x,) and liearize it to be able to use what we did in the exercise session
        unfold = torch.nn.Unfold(kernel_size = self.kernel_size)
        output = unfold(x)

        
        return 

    def backward (self,y):
        #taking the deriative of "linear conv"
        return
    def param ( self ) :
        return [self.weights, self.bias, self.gradweights, self.gradbias]











    


    
