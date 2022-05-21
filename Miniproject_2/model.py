
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
        self.loss = MSE()
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
    
    def backward (self,y, target):
        print("y",y)
        y = loss.backward(y, target)
        for layer in reversed(self.type_layer):
            y = layer.backward(y)
            print("backward_layer",y)

        return y

    def param ( self ) :
        return []



class Conv2d(Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0, dilation= 1):
        super().__init__()

        if type(kernel_size) == int:
            kernel_size = (kernel_size,kernel_size)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_channel = output_channel
        
        k = np.sqrt(1/(input_channel*kernel_size[0]*kernel_size[1]))
        #initializing them
        self.weight = torch.empty(output_channel, input_channel,kernel_size[0],kernel_size[1]).uniform_(-k,k)
        self.bias = torch.empty(output_channel).uniform_(-k,k)
        self.gradweight = torch.empty(output_channel, input_channel,kernel_size[0],kernel_size[1])*0
        self.gradbias = torch.empty(output_channel)*0
        

    def conv (self,x):
        weight2 = self.weight.clone()
        h_in, w_in = x.shape[2:]
        h_out = ((h_in+2*self.padding-self.dilation*(self.kernel_size[0]-1)-1)/self.stride+1)
        w_out = ((w_in+2*self.padding-self.dilation*(self.kernel_size[1]-1)-1)/self.stride+1)
        unfolded = torch.nn.functional.unfold(x, kernel_size = self.kernel_size, dilation = self.dilation, padding = self.padding, stride = self.stride)
        out = unfolded.transpose(1, 2).matmul(weight2.view(weight2.size(0), -1).t()).transpose(1, 2) + self.bias.view(1,-1,1)
        output = out.view(x.shape[0], self.output_channel, int(h_out), int(w_out))
        return output
        
    def forward (self,x):
        print("TYPE",x.type)
        x_ = x
        x_ = self.conv(x_)
        return x_

    def backward (self,y):
        #taking the deriative of "linear conv"
        return[]
    def param ( self ) :
        return [self.weight, self.bias, self.gradweight, self.gradbias]

input_tensor = torch.normal(0, 1, size=(3,2,2), requires_grad=True)
target = torch.normal(0, 1, size=(3,2,2), requires_grad=True)

sequential_torch = nn.Sequential(nn.ReLU(),nn.Sigmoid())

criterion = nn.MSELoss()

y = sequential_torch(input_tensor)

loss = criterion(y, target)

# loss_val = loss(y, target)

loss.backward()



torch.set_grad_enabled(True)



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
