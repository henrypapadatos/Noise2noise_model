
import torch
from torch import nn
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
    def SGD ( self , lr):
        pass
    def param ( self ) :
        pass


torch.set_grad_enabled(False)

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

class Seq(Module): #MODIFY: supposed run sequentially all the stuff you are asking it to 
    def __init__(self, *type_layer):
        #super().__init__()
        self.type_layer = type_layer
    # ex: 1 Sequential ( Conv( stride 2) , ReLU , Conv ( stride 2) , ReLU , Upsampling , ReLU , Upsampling , Sigmoid )

    def forward (self,x):
        # for looop from start to begning
        print("y",x)
        for layer in self.type_layer:
            #later we will have to call layer.param
            x = layer.forward(x)
            #layer.forward(layer.param())
            print("forward_layer",x)
        return x
    
    def backward (self,y):
        print("y",y)
        for layer in reversed(self.type_layer):
            y = layer.backward(y)
            print("backward_layer",y)

        return y

    def param ( self ) :
        return []



class ConvLayer(Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride):
        super().__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        
        k = np.sqrt(1/(input_channel*kernel_size[0]*kernel_size[1]))
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

print("First Try")
y = torch.normal(0, 1, size=(3,2,2))
relu = Relu()
sigmoid = Sigmoid()
sequential = Seq(relu,sigmoid)
y = sequential.forward(y)
y = sequential.backward(y)

sequentialll = nn.Sequential(nn.ReLU(),nn.Sigmoid())
y = sequentialll(y)
y.backward()
y.grad
