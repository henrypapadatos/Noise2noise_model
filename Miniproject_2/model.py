
import torch
from torch import nn
from torch import empty
import matplotlib.pyplot as plt
import math
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
    def __init___(self):
        self.input = None
    def forward ( self , x) :
        self.input = x.clone()
        return self.input*((self.input>0).float())
    def backward ( self , y) :
        return y.mul((self.input>0).float())
    def param ( self ) :
        return []

class Sigmoid(Module):
    def __init__(self):
        self.input = None

    def forward (self, x) :
        self.input = x.clone()
        return 1/(1+(-self.input).exp())

    def backward ( self , y) :
        
        grad = (-self.input ).exp()/(1+(-self.input).exp()).pow(2)
        return y.mul(grad)
    def param(self):
        return []
    
class MSE(Module):
    def forward (self, x, x_target) :
        self.prediction = x 
        self.target = x_target
        return (self.prediction - self.target).pow(2).mean()
    def backward (self) :
        batch_size = self.prediction.size()[0]
        return 2* (self.prediction-self.target)/batch_size
    def param ( self ) :
        return []

class Sequential(Module):  #MODIFY: supposed run sequentially all the stuff you are asking it to 
    def __init__(self, *type_layer):
        #super().__init__()
        self.type_layer = type_layer
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
    
    def backward (self,y):
        #print("y",y)
        y_ = y 
        for layer in reversed(self.type_layer):
            y_ = layer.backward(y_)
            #print("backward_layer",y_)
            #print("backward_layer",y_)

        return y_

    def param(self):

        parameters = []
        for layer in self.type_layer:
            parameters.append(layer.param())

        return parameters



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
        self.input = None
        
        k = math.sqrt(1/(input_channel*kernel_size[0]*kernel_size[1]))
        #initializing them
        self.weight = torch.empty(output_channel, input_channel,kernel_size[0],kernel_size[1]).uniform_(-k,k)
        self.bias = torch.empty(output_channel).uniform_(-k,k)
        self.gradweight = torch.empty(output_channel, input_channel,kernel_size[0],kernel_size[1])*0
        self.gradbias = torch.empty(output_channel)*0
        

    def conv (self,x, weights=None, bias=None):
        if weights==None:
            weights = self.weight.clone()
            bias = self.bias.clone()
        elif bias==None:
            bias=torch.empty(weights.size()[0])*0
        h_in, w_in = x.shape[2:]
        h_out = ((h_in+2*self.padding-self.dilation*(self.kernel_size[0]-1)-1)/self.stride+1)
        w_out = ((w_in+2*self.padding-self.dilation*(self.kernel_size[1]-1)-1)/self.stride+1)
        unfolded = torch.nn.functional.unfold(x, kernel_size = self.kernel_size, dilation = self.dilation, padding = self.padding, stride = self.stride)
        out = unfolded.transpose(1, 2).matmul(weights.view(weights.size(0), -1).t()).transpose(1, 2) + bias.view(1,-1,1)
        output = out.view(x.shape[0], self.output_channel, int(h_out), int(w_out))
        return output
        
    def forward (self,input):
        self.input = input.clone()
        output = self.conv(input.clone())
        return output

    def backward (self,y):
        #taking the deriative of "linear conv"
        output = y.clone()
        
        #compute de derivative of the output wrt the bias
        dydbias = y.sum((0,2,3))
        self.bias+=dydbias
        
        #compute de derivative of the output wrt the weights
        output = output.permute(1, 2, 3, 0).reshape(self.output_channel, -1)
        modified_input= torch.nn.functional.unfold(self.input.clone(), kernel_size = self.kernel_size, dilation = self.dilation, padding = self.padding, stride = self.stride)
        modified_input = modified_input.permute(1,2,0)
        modified_input = modified_input.reshape(modified_input.size(0), -1)
        #dy/dkernel = conv(self.input, y)
        dydkernel = torch.matmul(output, modified_input.t())
        
        #dydkernel has the wrond dimension :(
        #self.gradweight += dydkernel
        
        modified_weights = self.weight.clone().permute(1,2,3,0).reshape(self.output_channel, -1)
        dydinput = torch.matmul(modified_weights.t(), output)
        #dy/input 
        return[]
    
    def param ( self ) :
        return [(self.weight, self.gradweight), (self.bias, self.gradbias)]


class Optimizer(Module):
    def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False):
        #self.momentum = momentum
        #self.dampening = dampening
        #self.weight_decay = weight_decay
        #self.nesterov = nesterov
        self.maximize = maximize
        self.lr = lr
        #self.prev_b = 0
    def step (self, params) :

        for param_layer in params:
            for param in param_layer:
                g_t = param[1]
                #if self.weight_decay:
                    #g_t = g_t + self.weight_decay*param[0]
                #if self.momentum:
                    #if not self.prev_b: #if self.prev_b = 0
                        #b_t = g_t
                    #else:
                        #b_t = self.momentum*self.prev_b + (1-self.dampening)*g_t  #eta is learning rate
                        #self.prev_b = b_t
                    #if self.nesterov:
                        #g_t = g_t+self.momentum*b_t #help its g_t-1
                    #else:
                        #g_t = b_t

                
                if self.maximize:
                    param[0].add_(self.lr * g_t)
                else:
                    param[0].sub_(self.lr * g_t)
                    

    def param ( self ) :
        return [self.params]

class Upsampling(Module):
    def __init__(self,scale):
        self.scale = scale
        self.conv2d = Conv2d()
    def forward (self , x) :
        x_ = x.clone()
        [b,c_i,h_i,w_i] = x_.shape
        h_out = self.scale*h_i
        w_out = self.scale*w_i
        out = torch.empty(b,c_i,h_out,w_out)
        for k in range(b):
            for l in range(c_i):
                for i in range(h_i):
                    for j in range(w_i):
                        out[k,l,i*self.scale:(i*self.scale+self.scale),j*self.scale:(j*self.scale+self.scale)] = x_[k,l,i,j]
        return self.conv2d.forward(out)

    def backward (self,y):
        return self.conv2d(y, kernel_size = 1, stride = self.scale)

    def param ( self ) :
        return[]


def training_visualisation(imgs):
    #Plot the 4 first images of imgs in a subplot way
    fig=plt.figure()
    nb_image = 4
    for a in range(nb_image):
        with torch.no_grad():     
            b = imgs[a,:,:,:].int()
            if torch.cuda.is_available():
                b = b.to('cpu')
            b = b.permute(1,2,0)    
            ax = fig.add_subplot(1, nb_image, a+1)
            subplot_title=("Image"+str(a))
            ax.set_title(subplot_title)  
            plt.imshow(b)
    fig.tight_layout() 
    plt.pause(0.001)
    
class Model():
    def __init__(self):
        self.lr = 0.002
        self.nb_epoch = 100
        self.batch_size = 1000
        self.optimizer = Optimizer(lr= self.lr)
        self.criterion = MSE()
        self.Conv = Conv2d
        self.Linear = Linear
        #self.Upsampling = Upsampling
        #self.model = Sequential(self.Conv(3,3,3) , self.ReLU , self.Conv(3,3,3) , self.ReLU , self.Upsampling , self.ReLU , self.Upsampling , self.Sigmoid)
        #self.Conv(3,3,kernel_size = 2, padding = 1, dilation = 2, stride = 1 ), self.Sigmoid
        #self.model = Sequential(self.Conv(3,3,kernel_size = 2, padding = 1, dilation = 2, stride = 1 ), self.ReLU)

        # Linear(25,25),ReLU(),Linear(25,25),ReLU(),Linear(25,2),Tanh()
        self.model = Sequential(Linear(20,25), Relu(), Linear(25,25),Relu(),Linear(25,20),Sigmoid())

    
    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model$
        #full_path = os.path.join('Miniproject_2', 'bestmodel.pth')
        #self.model.load_state_dict(torch.load(full_path,map_location=torch.device('cpu')))
        pass 
    def train(self, train_input, train_target, num_epochs=100 ,test_input=None, test_target=None, vizualisation_flag = False):
        num_epochs = 10

        if vizualisation_flag and test_input!=None:
            plt.ion()
            plt.show()
            training_visualisation(test_target)
            training_visualisation(test_input)
        
        if test_input!=None:
            initial_psnr = self.psnr(test_input/255, test_target/255)
            print('Psnr value between clean and noisy images is: {:.02f}'.format(initial_psnr))

        for e in range(num_epochs):
            i = 0
            for input, targets in zip(train_input.split(self.batch_size),  
                                      train_target.split(self.batch_size)):

                
                output = self.predict(input)
                loss = self.criterion.forward(output/255, targets/255)
                grad_loss = self.criterion.backward()
                self.model.backward(grad_loss)
                # add SGD here
                self.optimizer.step(self.model.param())
                i+=1
            
            

            if test_input!=None:
                denoised = self.model.forward(test_input)    
                psnr = self.psnr(denoised/255, test_target/255)                
                print('Nb of epoch: {:d}    psnr: {:.02f}'.format(e, psnr))
                
            #display denoised images 25 times during training
            if vizualisation_flag and (e%(num_epochs//10)==0) and test_input!=None:
                training_visualisation(denoised)
            
        
    def predict(self, input_imgs):
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network .
        #normalize image between 0 and 1
        input_imgs_ = input_imgs/255
        output = self.model.forward(input_imgs_)

        # output should be an int between [0,255]
        output = torch.clip(output*255, 0, 255)
        return output
    
    def psnr(self, denoised, ground_truth):
        #Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
        mse = torch.mean((denoised - ground_truth )** 2)
        psnr = -10 * torch . log10 ( mse + 10** -8)
        return psnr.item()

class Linear(Module):
    """ A Module implementing the sequential combination of several modules. It stores the individual modules in a list modules.
    """
    
    def __init__(self,input_features,output_features,bias=True):
        super(Linear).__init__()
        
        self.input_features = input_features
        self.output_features = output_features
        
        self.weights = empty(input_features,output_features)
        self.weights_grad = empty(input_features,output_features)
        
        if bias :
            self.bias = empty(output_features)
            self.bias_grad = empty(output_features)
        else : 
            self.bias = None
            self.bias_grad = None
        
        self.reset()
    
    def reset(self):
        """ Initializes the weights at random and the biases at 0 if they are defined
        """
        
        std = 1. / math.sqrt(self.weights.size(0))
        self.weights.normal_(-std,std)
        self.weights_grad.fill_(0)
        if self.bias is not None : 
            self.bias.fill_(0)
            self.bias_grad.fill_(0)
    def forward(self,input):
        """ Implements the forward pass for the Linear Module
        Saves the input for the backward pass when we will need to compute gradients, under self.input  
        Computes Y = X*w + b
        """
        self.input = input.clone()
        if self.bias is not None:
            return self.input.matmul(self.weights).add(self.bias)
        else :
            return self.input.matmul(self.weights)
        
    def backward(self,gradwrtoutput):
        """ Implements the backward pass for the Linear Module
        Uses the chain rule to compute the gradients wrt the weights, bias, and input and stores them in the instance parameters
        Arguments:
             #gradwrtoutput : dL/dy in the backprogation formulas 
        """
        
        # Computes gradient wrt weights dw = X^(T) * dy using the backpropagation formulas
        self.weights_grad = self.input.t().matmul(gradwrtoutput)
        
        # Computes gradient wrt bias iff bias is not none
        # db = dy ^ (T) * 1  using the backpropagation formulas, 
        # The sum is to take for account the possibility that we process our inputs by batches of size greater than 1, we sum the gradient contributions of independent points
        if self.bias is not None : 
                self.bias_grad = gradwrtoutput.t().sum(1)
            
        # Computes gradient wrt input X dX = dY * w^(T) using the backpropagation formulas
        return gradwrtoutput.matmul(self.weights.t())
    
    
    def param(self):
        """ Returns a list of pairs, each composed of a parameter tensor and its corresponding gradient tensor of same size
        """
        if self.bias is not None : 
            return [(self.weights,self.weights_grad),(self.bias,self.bias_grad)]
        else : 
            return [(self.weights,self.weights_grad)]
        
        
        
    
    
