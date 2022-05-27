from torch import set_default_dtype, float64 , empty, cuda
from torch.nn.functional import fold, unfold
import pickle
import math
import os
set_default_dtype(float64)

class Module ( object ) :
    # Architecture suggested in the assignment
    def __call__(self, x):
        # We added this module to implement to the same behaviour as pytorch modules
        # If no specific function is called, forward is called by default
        pass

    def forward ( self , x ) :
        # In charge of the forward pass of the Module,
        # get for input and returns, a tensor or a tuple of tensors.
        raise NotImplementedError

    def backward ( self , y) :
        # In charge of the backward pass of the Module,
        # get as input a tensor or a tuple of tensors containing the 
        # gradient of the loss with respect to the module’s output, 
        # accumulate the gradient wrt the parameters, and return a tensor
        # or a tuple of tensors containing the gradient of the loss wrt the module’s input
        raise NotImplementedError

    def param ( self ) :
        # Returns a list of pairs composed of a parameter tensor and a gradient tensor of the same size.
        # This list should be empty for parameterless modules (such as ReLU). 
        return []

class Relu(Module):
    # Rectified Linear Unit activation function module
    def __init___(self):
        self.input = None

    def __call__(self, x):
        return self.forward(x)

    def forward ( self , x) :
        # Applies max(0,x)
        self.input = x.clone()
        return self.input.mul((self.input>0))

    def backward ( self , y) :
        # Computes the gradient of ReLU with respect to the input
        return y.mul(self.input>0)

    def param ( self ) :
        # Returns an empty list since there are no parameters to update for the ReLU module
        return []

class Sigmoid(Module):
    # Sigmoid activation function module
    def __init__(self):
        self.input = None
        
    def __call__(self, x):
        return self.forward(x)

    def forward (self, x) :
        # Applies the Sigmoid to the input
        self.input = x.clone()
        return 1/(1+(-self.input).exp())

    def backward ( self , y) :
        # Computes the gradient of Sigmoid with respect to the input
        grad = (-self.input ).exp()/(1+(-self.input).exp()).pow(2)
        return y.mul(grad)

    def param(self):
        # Returns an empty list since there are no parameters to update for the Sigmoid module
        return []
    
class MSE(Module):
    # Mean Squared Error loss function module equivalent to the pytorch torch.nn.MSELoss(reduction ='mean'), 
    # meaning in both the forward and backward we divide over all the elements
    def __call__(self, x, x_target):
        return self.forward(x, x_target)

    def forward (self, x, x_target) :
        # Computes the average error over all batches, returns a scalar
        self.prediction = x.clone()
        self.target = x_target.clone()
        return (self.prediction - self.target).pow(2).mean()

    def backward (self) :
        # Computes the gradient of MSE loss function with respect to the input, returns a tensor
        return 2* (self.prediction-self.target)/self.prediction.numel()     

    def param ( self ) :
        # Returns an empty list since there are no parameters to update for the MSE module
        return []


class Conv2d(Module):
    # Computes the 2D convolution of an input
    def __init__(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0, dilation= 1):

        super().__init__()

        if type(kernel_size) == int:
            kernel_size = (kernel_size,kernel_size)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_channel = output_channel
        self.input_channel = input_channel
        self.input = None

        # Uniform initialisation of the weights and biases, based on pytorch initilisation of these in torch.nn.Conv2d
        self.weight = empty(output_channel, input_channel,kernel_size[0],kernel_size[1]).fill_(0)
        k = math.sqrt(1/(input_channel*kernel_size[0]*kernel_size[1]))
        self.weight.uniform_(-k,k).nan_to_num_()
        self.bias = empty(output_channel).fill_(0)
        self.bias.uniform_(-k,k).nan_to_num_()
        self.gradweight = empty(output_channel, input_channel,kernel_size[0],kernel_size[1]).fill_(0).nan_to_num_()
        self.gradbias = empty(output_channel).fill_(0).nan_to_num_()
        
        if cuda.is_available():
            self.weight = self.weight.cuda()
            self.bias = self.bias.cuda()
            self.gradweight = self.weight.cuda()
            self.gradbias = self.bias.cuda()
        
    def __call__(self, x):
        return self.forward(x)
        
    def forward (self,x):
        # Applies a 2D convolution, transforming it in a linear operation Y = X*w + b
        # using unfold 
        self.input = x.clone()
        
        weights = self.weight.clone()
        bias = self.bias.clone()
        
        h_in, w_in = self.input.shape[2:]
        h_out = ((h_in+2*self.padding-self.dilation*(self.kernel_size[0]-1)-1)/self.stride+1)
        w_out = ((w_in+2*self.padding-self.dilation*(self.kernel_size[1]-1)-1)/self.stride+1)
        unfolded = unfold(self.input, kernel_size = self.kernel_size, dilation = self.dilation, padding = self.padding, stride = self.stride).double()
        self.unfolded_x = unfolded.clone()
        out = unfolded.transpose(1, 2).matmul(weights.view(weights.size(0), -1).t()).transpose(1, 2) + bias.view(1,-1,1)
        output = out.view(self.input.shape[0], self.output_channel, int(h_out), int(w_out))
        
        return output

    def backward (self,y):
        # y : dL/dy
        # Applies the backprogation based on the linearisation of the 2d convolution
        Y = y.clone()
        
        # Compute the derivative of the output with respect to the biases dL/db, summing over all batches
        self.gradbias=Y.sum((0,2,3))

        # Compute the derivative of the output with respect to the weights dL/dw = X^(T)*dy, summing over all batches
        lin_Y = Y.view(Y.shape[0],self.output_channel,Y.shape[2]*Y.shape[3])
        lin_weights_grad = lin_Y.matmul(self.unfolded_x.transpose(1,2)).sum(0)
        self.gradweight = lin_weights_grad.view(lin_weights_grad.size(0),self.input_channel, self.kernel_size[0],self.kernel_size[1])

        # Computes the gradient with respect to the input dL/dx = dY*w^(T) 
        lin_w = self.weight.view(self.weight.size(0), -1)
        lin_grad_wrt_input = lin_w.transpose(0,1).matmul(lin_Y)
        grad_wrt_input = fold(lin_grad_wrt_input,output_size=(self.input.shape[-2],self.input.shape[-1]), kernel_size = self.kernel_size, dilation = self.dilation, padding = self.padding, stride = self.stride).double()

        return grad_wrt_input
    
    def param( self ) :
        # Returns the updated parameters of the module
        return [(self.weight, self.gradweight), (self.bias, self.gradbias)]


class Upsampling(Module):
    # Implements the equivalent of a nn.UpsamplingNearest2d + nn.torch.Conv2d
    def __init__(self, input_channel, output_channel, kernel_size, scale = 1, stride = 1, padding = 0, dilation= 1):
        self.scale = scale
        self.conv2d = Conv2d(input_channel, output_channel, kernel_size, stride = stride, padding=padding, dilation=dilation)
    
    def __call__(self, x):
        return self.forward(x)
    
    def forward (self , x) :
    # Implements an upscaling of the input followed by a 2d convolution   
        input_ = x.clone()
        [b,c_i,h_i,w_i] = input_.shape
        u1 = empty(w_i,w_i*self.scale).fill_(0).nan_to_num_()
        u2 = empty(h_i,h_i*self.scale).fill_(0).nan_to_num_()

        if cuda.is_available():
                    u1 = u1.cuda()
                    u2 = u2.cuda()
                    
        for i in range(w_i):
            u1[i,i*self.scale:(i*self.scale+self.scale)] = 1
        self.u1 = u1
        for j in range(h_i):
            u2[j,j*self.scale:(j*self.scale+self.scale)] = 1
        self.u2 = u2

        u1_i = input_.matmul(u1)
        u1_i_t = u1_i.transpose(2,3)
        out = u1_i_t.matmul(u2)
        output = out.transpose(2,3)
        output = self.conv2d(output)

        return output 
    
    def backward (self , y) :
    # Computes the gradient of UpscalingNN and of the convolution with respect to the input
    
        y_ = y.clone()
        y_ = self.conv2d.backward(y_)
        v1 = self.u1.t()
        v2 = self.u2.t()
        y_ = y_.transpose(2,3)
        v2_y = y_.matmul(v2)
        v2_y_t = v2_y.transpose(2,3)
        output2 = v2_y_t.matmul(v1)
        return output2

    def param ( self ) :
    # Returns the updated parameters of the 2d convolution only, since the upscaling doesn't have parameters
        return self.conv2d.param()
    
class Optimizer(Module):
    # Implements a Stochastic Gradient Descent 
    def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False):
        self.maximize = maximize
        self.lr = lr
        self.params = None
        
    def step(self, params):
    # Updates the parameters of each module used in our model

        for param_layer in params:
            for param in param_layer:
                g_t = param[1]
                if self.maximize:
                    param[0].add_(self.lr * g_t)
                else:
                    param[0].sub_(self.lr * g_t)
                
                #After a step, we reinitialise the gradients at 0
                param[1].fill_(0)
        return params
    
class Sequential(Module):  
    # Implementation of a Sequential module. This function gets as input a
    # sequence of modules that will be applied in the order given in the instructor
    def __init__(self, *type_layer):
        self.type_layer = type_layer
        
    def __call__(self, x):
       return self.forward(x)
   
    def forward (self,x):
        # Applies the forward passes of the given modules, one by one 
        # and in order
        x_ = x.clone()
        for layer in self.type_layer:
            x_ = layer(x_)
        return x_
    
    def backward (self,y):
        # Computes the gradient by going backward through the given modules
        y_ = y.clone()
        for layer in reversed(self.type_layer):
            y_ = layer.backward(y_)
        return y_

    def param(self):
        # Returns a list containing the updated parameters of each one of the given modules
        parameters = []
        for layer in self.type_layer:
            parameters.append(layer.param())

        return parameters
    
class Model():
    # Module of the architecture of our model
    def __init__(self):
        self.lr = 2.5 
        self.nb_epoch = 100
        self.batch_size = 40 
        self.optimizer = Optimizer(lr= self.lr)
        self.criterion = MSE()
        self.channel = 32
        self.model = Sequential(Conv2d(input_channel = 3,output_channel = self.channel,kernel_size = 3, padding = 1, stride = 2)
                                ,Relu()
                                ,Conv2d(input_channel = self.channel,output_channel = self.channel,kernel_size = 3, padding = 1, stride = 2)
                                ,Relu()
                                ,Upsampling(self.channel,self.channel,kernel_size = 3, scale= 2, padding = 1)
                                ,Relu()
                                ,Upsampling(self.channel,3,kernel_size = 3, scale= 2, padding = 1)
                                ,Sigmoid())

    def load_pretrained_model(self):

        # Loading the parameters saved in bestmodel.pkl into the model
        file_abs_path = os.path.dirname(os.path.abspath(__file__))
        
        full_path = os.path.join(file_abs_path, 'bestmodel.pth')
        
        params = self.model.param()
        
        with open(full_path, 'rb') as file:          
            loaded_params = pickle.load(file)
        
        for param_layer, loaded_param_layer in zip(params, loaded_params):
            for param, loaded_param in zip(param_layer, loaded_param_layer):
                
                param[1].copy_(loaded_param[1])
                param[0].copy_(loaded_param[0])
    
    def train(self, train_input, train_target, num_epochs=100):
        # Trains the models over a dataset(train_input, train_target), given a certain number of epochs
        # train_input : tensor of size (N, C, H, W) containing a noisy version of the images
        # train_target : tensor of size (N, C, H, W) containing another noisy version of the
        # same images , which only differs from the input by their noise .
        
        # Cheking if a GPU is available
        if cuda.is_available():
            train_input = train_input.cuda()
            train_target = train_target.cuda()
       
        for e in range(num_epochs):
            
            for input, targets in zip(train_input.split(self.batch_size),  
                                      train_target.split(self.batch_size)):

                # Forward pass
                output = self.predict(input)
                # Computing initial loss
                loss = self.criterion(output/255, targets/255)
                # Computing the gradient of the loss
                grad_loss = self.criterion.backward()
                # Backpropagation pass
                self.model.backward(grad_loss)
                # Update of the parameters 
                self.optimizer.step(self.model.param())

            print('Nb of epoch: {:d}   loss: {:.04f}'.format(e, loss))
        
    def predict(self, input_imgs):
        # Applies to trained model to a test input
        # input_imgs : tensor of size (N1 , C, H, W) with values in range 0 -255 that has to
        # be denoised by the trained or the loaded network .
        # returns a tensor of the size (N1 , C, H, W) with values in range 0 -255.

        moved_to_GPU = False
        # If input_imgs is on the cpu even though there is a GPU, it means that the model is 
        # on the GPU. Therefore, we need to send input_imgs on the GPU, apply the model to it and then 
        # send it back to the CPU
        if cuda.is_available() and not input_imgs.is_cuda:
            moved_to_GPU = True
            input_imgs = input_imgs.cuda()

        # Normalize image between 0 and 1 
        input_imgs_ = (input_imgs/255)
        output = self.model(input_imgs_)

        # Output should be an int between [0,255]
        output = output.mul(255).clip(0, 255)
        
        if moved_to_GPU:
            output = output.cpu()
            
        return output
    
    def save_model(self):
        # Saves the model as a .pth file
        full_path = os.path.join('Miniproject_2', 'bestmodel.pkl')
        params = self.model.param()
        
        with open(full_path, 'wb') as file:          
            pickle.dump(params, file)