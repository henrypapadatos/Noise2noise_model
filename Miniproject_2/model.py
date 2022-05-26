
import torch
from torch import empty
import matplotlib.pyplot as plt
from torch.nn.functional import fold
from torch.nn.functional import unfold
import math

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
        return self.input.mul((self.input>0))
    def backward ( self , y) :
        return y.mul(self.input>0)
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
        
        self.prediction = x.clone()
        self.target = x_target.clone()
        return (self.prediction - self.target).pow(2).mean()
        #return torch.rand(100,4,32,32)
    def backward (self) :
        return 2* (self.prediction-self.target)/self.prediction.numel()     
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
        x_ = x.clone()
        for layer in self.type_layer:
            #later we will have to call layer.param
            x_ = layer.forward(x_)
            #layer.forward(layer.param())
            #print("forward_layer",x_)
        return x_
    
    def backward (self,y):
        #print("y",y)
        y_ = y.clone()
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
        self.input_channel = input_channel
        self.input = None
        
       
        #initializing them
        '''
        std = 1. / math.sqrt(self.weights.size(0))
        self.weights.normal_(-std,std)
        
        '''
        self.weight = empty(output_channel, input_channel,kernel_size[0],kernel_size[1]).fill_(0)
        k = math.sqrt(1/(input_channel*kernel_size[0]*kernel_size[1]))
        self.weight.uniform_(-k,k).nan_to_num_()
        #self.weight = self.weight.fill_(0).uniform_(-k,k)
        self.bias = empty(output_channel).fill_(0)
        self.bias.uniform_(-k,k).nan_to_num_()
        #self.bias = self.bias.fill_(0).uniform_(-k,k)
        self.gradweight = empty(output_channel, input_channel,kernel_size[0],kernel_size[1]).fill_(0).nan_to_num_()
        self.gradbias = empty(output_channel).fill_(0).nan_to_num_()
        
    def __call__(self, input):
        return self.forward(input)

    def conv (self,x, weights=None, bias=None):
        x_ = x.clone()

        if weights==None:
            weights = self.weight.clone()
            bias = self.bias.clone()
        elif bias==None:
            bias= empty(weights.size()[0]).fill_(0).nan_to_num()

        h_in, w_in = x_.shape[2:]
        h_out = ((h_in+2*self.padding-self.dilation*(self.kernel_size[0]-1)-1)/self.stride+1)
        w_out = ((w_in+2*self.padding-self.dilation*(self.kernel_size[1]-1)-1)/self.stride+1)
        unfolded = unfold(x_, kernel_size = self.kernel_size, dilation = self.dilation, padding = self.padding, stride = self.stride)
        self.unfolded_x = unfolded.clone()
        out = unfolded.transpose(1, 2).matmul(weights.view(weights.size(0), -1).t()).transpose(1, 2) + bias.view(1,-1,1)
        output = out.view(x_.shape[0], self.output_channel, int(h_out), int(w_out))
        
        return output
        
    def forward (self,x):
        self.input = x.clone()
    
        '''
        self.weight.requires_grad_(requires_grad = True)
        self.input.requires_grad_(requires_grad = True)
        self.bias.requires_grad_(requires_grad = True)
        '''
        self.output = self.conv(self.input)
        return self.output


    def backward (self,y):
        
        Y = y.clone()
        # print(" MAX Y", Y.abs().mean())
        #compute de derivative of the output wrt the bias
        self.gradbias=Y.sum((0,2,3))
        # print(" MAX gradbias", self.gradbias.abs().mean())

        lin_Y = Y.view(Y.shape[0],self.output_channel,Y.shape[2]*Y.shape[2])
        lin_weights_grad = lin_Y.matmul(self.unfolded_x.transpose(1,2)).sum(0)
        self.gradweight = lin_weights_grad.view(lin_weights_grad.size(0),self.input_channel, self.kernel_size[0],self.kernel_size[0])

        # print(" MAX gradweight",self.gradweight.abs().mean())
        # Computes gradient wrt input X dX = dY * w^(T) using the backpropagation formulas
        lin_w = self.weight.view(self.weight.size(0), -1)
        lin_grad_wrt_input = lin_w.transpose(0,1).matmul(lin_Y)
        grad_wrt_input = fold(lin_grad_wrt_input,output_size=(self.input.shape[-1],self.input.shape[-1]), kernel_size = self.kernel_size, dilation = self.dilation, padding = self.padding, stride = self.stride)
        #print(" MAX grad_wrt_input", grad_wrt_input.abs().max())

        return grad_wrt_input
    
    def param( self ) :
        return [(self.weight, self.gradweight), (self.bias, self.gradbias)]


class Optimizer(Module):
    def __init__(self, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False):
        self.maximize = maximize
        self.lr = lr
        self.params = None
        #self.prev_b = 0
    def step(self, params):

        for param_layer in params:
            for param in param_layer:
                g_t = param[1]
                if self.maximize:
                    param[0].add_(self.lr * g_t)
                else:
                    param[0].sub_(self.lr * g_t)
                
                #After gradient step reanitialise that gradients of the wieghts.
                param[1].fill_(0)
        return params

    def param ( self ) :
        return self.params

class Upsampling(Module):
    def __init__(self, input_channel= None, output_channel = None, kernel_size = None, scale = 1, stride = None, padding = None, dilation= None):
        ##add inpput and output param + other to be able to do convolution
        self.scale = scale
        #self.conv2d = Conv2d(input_channel, output_channel, kernel_size, stride = stride, padding=padding, dilation=dilation)

    def forward (self , x) :
        
        input_ = x.clone().float()
        [b,c_i,h_i,w_i] = input_.shape
        u1 = empty(w_i,w_i*self.scale).fill_(0).nan_to_num_()
        for i in range(w_i):
            u1[i,i*self.scale:(i*self.scale+self.scale)] = 1
        self.u1 = u1
        u2 = empty(h_i,h_i*self.scale).fill_(0).nan_to_num_()
        for j in range(h_i):
            u2[j,j*self.scale:(j*self.scale+self.scale)] = 1
        self.u2 = u2
        u1_i = input_.matmul(u1)
        u1_i_t = u1_i.transpose(2,3)
        out = u1_i_t.matmul(u2)
        output = out.transpose(2,3)
        #output = self.conv2d.forward(output)
        return output 

    
    def backward (self , y) :
        y_ = y.clone().float()
        #y_ = self.conv2d.backward(y_)
        v1 = self.u1.t()
        v2 = self.u2.t()
        y_ = y_.transpose(2,3)
        v2_y = y_.matmul(v2)
        v2_y_t = v2_y.transpose(2,3)
        output2 = v2_y_t.matmul(v1)
        return output2

    def param ( self ) :
        return []#self.conv2d.param()


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
        self.lr = 4.0 #0.001 best  0.00001
        self.nb_epoch = 100
        self.batch_size = 500
        #self.batch_size = 50
        self.optimizer = Optimizer(lr= self.lr)
        self.criterion = MSE()
        self.channel = 32
        ### LINEAR ##
        #self.model = Sequential(Linear(20,25), Relu(), Linear(25,25),Relu(),Linear(25,20),Sigmoid())
        #self.model = Sequential(Linear(2,25),Relu(),Linear(25,25),Relu(),Linear(25,25),Relu(),Linear(25,2),Sigmoid())

        ### CONV2D ##
        # self.model = Sequential(Conv2d(input_channel = 3,output_channel = 16,kernel_size = 3, padding = 1, stride = 1),Relu(), Conv2d(16,16,kernel_size = 3, padding = 1, stride = 1),Relu(), Conv2d(16,3,kernel_size = 3, padding = 1, stride = 1),Sigmoid())
        self.model = Sequential(Conv2d(input_channel = 3,output_channel = self.channel,kernel_size = 3, padding = 1, stride = 2)
                                ,Relu()
                                # ,Conv2d(input_channel = self.channel,output_channel = self.channel,kernel_size = 3, padding = 1, stride = 2)
                                # ,Relu()
                                # ,Upsampling(self.channel,self.channel,kernel_size = 3, scale= 2, padding = 1)
                                # ,Relu()
                                ,Upsampling(self.channel,3,kernel_size = 3, scale= 2, padding = 1)
                                ,Sigmoid())

    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model$
        #full_path = os.path.join('Miniproject_2', 'bestmodel.pth')
        #self.model.load_state_dict(torch.load(full_path,map_location=torch.device('cpu')))
        pass 
    def train(self, train_input, train_target, num_epochs=100 ,test_input=None, test_target=None, vizualisation_flag = False):
        #num_epochs = 10

        if vizualisation_flag and test_input!=None:
            plt.ion()
            plt.show()
            training_visualisation(test_target)
            training_visualisation(test_input)
        
        if test_input!=None:
            initial_psnr = self.psnr(test_input/255, test_target/255)
            print('Psnr value between clean and noisy images is: {:.02f}'.format(initial_psnr))

        for e in range(num_epochs):
            
            for input, targets in zip(train_input.split(self.batch_size),  
                                      train_target.split(self.batch_size)):

                
                output = self.predict(input)
                loss = self.criterion.forward(output/255, targets/255)
                grad_loss = self.criterion.backward()
                self.model.backward(grad_loss)
                # add SGD here
                #with torch.no_grad():  
                params =  self.optimizer.step(self.model.param())


            if test_input!=None:
                denoised = self.predict(test_input)    
                psnr = self.psnr(denoised/255, test_target/255)                
                print('Nb of epoch: {:d}    psnr: {:.04f}    loss: {:.08f}'.format(e, psnr, loss))
                
            #display denoised images 25 times during training
            if vizualisation_flag and (e%(num_epochs//10)==0) and test_input!=None:
                training_visualisation(denoised)
            
        
    def predict(self, input_imgs):
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network .
        #normalize image between 0 and 1
        input_imgs_ = (input_imgs/255).float()
        output = self.model.forward(input_imgs_)

        # output should be an int between [0,255]
        output = output.mul(255).clip(0, 255)
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
        
        self.weights = empty(input_features,output_features).nan_to_num()
        self.weights_grad = empty(input_features,output_features).nan_to_num()
        
        if bias :
            self.bias = empty(output_features).nan_to_num()
            self.bias_grad = empty(output_features).nan_to_num()
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
        
        
        
    
    
