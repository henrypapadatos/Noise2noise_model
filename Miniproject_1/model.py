import torch
from torch import nn
import os

    
class NetBlock(nn.Module):
    """
    This class implements a residual block constituted of a conv2d (or ConvTranspose2d)
    Followed by a batch normalization and a Relu
    If the parameter "transpose_flag" is True, ConvTranspose2d will be used. And if not, 
    a Conv2d will be used
    The skip connection has to take into account the stride. It is implemented 
    a simple convolution with kernel size = 1
    """
    def __init__(self, input_channel, output_channel, kernel_size, stride, transpose_flag):
        super().__init__()

        self.stride = stride
        self.transpose_flag = transpose_flag
        
        #Uses a transpose convolution
        if self.transpose_flag:
            if self.stride==1:
                self.convSkipTrans = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=(1,1), stride=(stride,stride))
                self.convTrans1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=(kernel_size,kernel_size), stride=(stride,stride), padding = (kernel_size -1)//2)
            elif self.stride==2:
                self.convTrans1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=(kernel_size,kernel_size), stride=(stride,stride), padding = (kernel_size -1)//2, output_padding = 1)
                self.convSkipTrans = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=(1,1), stride=(stride,stride), output_padding = 1)
        #Uses a convolution 
        else:
            self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=(kernel_size,kernel_size), stride=(stride,stride), padding = (kernel_size -1)//2)
            self.convSkip = nn.Conv2d(input_channel, output_channel, kernel_size=(1,1), stride=(stride,stride))
        
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.Relu = nn.ReLU()

        

    def forward(self, x):
        """
        Implements the forward path of NetBlock

        Parameters
        ----------
        x : TYPE: Tensor
            Input tensor on which our Res block is applied.

        Returns
        -------
        y : tensor
            Resulting tensor.

        """
        if self.transpose_flag:
            y = self.convTrans1(x)
        else:
            y = self.conv1(x)
        y = self.bn1(y)
        y = self.Relu(y) 
        
        #Apply the custom skip connection
        if self.transpose_flag:
            x = self.convSkipTrans(x)
        else:
            x = self.convSkip(x)
        y = y + x
        
        return y 


class Net(nn.Module):
    """
    This class creates our network
    """
    def __init__(self):
        """
        Initialize all the blocks needed in the model
        """
        
        super().__init__()
        
        nb_channel = 32
        
        self.conv1 = nn.Conv2d(3, nb_channel, kernel_size=(3,3), stride=(1,1), padding = (3 -1)//2)
        self.conv5t = nn.ConvTranspose2d(nb_channel, 3, kernel_size=(3,3), stride=(1,1), padding = (3 -1)//2)

       
        #Theses 2 blocks are our custom residual blocks implemented in the class NetBlock
        self.NetBLock1 = NetBlock(nb_channel,nb_channel,5,2,transpose_flag=0) 
        self.TransNetBlock1 = NetBlock(nb_channel,nb_channel,5,2,transpose_flag=1)
        
        self.Dropout = nn.Dropout(0.2)


        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)

        self.Relu =  nn.ReLU()
        self.sigmoid = nn.Sigmoid()



        
    def forward(self,x):
        """
        Implements the forward path of Net

        Parameters
        ----------
        x : TYPE: Tensor
            Input noisy image on which our model is applied.

        Returns
        -------
        y : tensor
            Denoised image.
            """
        
        y = self.conv1(x)
        
        y = self.bn1(y)

        y = self.Relu(y)
        
        y = self.Dropout(y)

        y = self.NetBLock1(y)
        
        y = self.bn2(y)

        y = self.Dropout(y)

        y =  self.TransNetBlock1(y)

        y = self.conv5t(y)

        y = self.sigmoid(y)
        
        return y

  
class Model():
    """
    Main class to access our model from outside this file. 
    """
    def __init__(self):
        """
        Instantiate model + optimizer + loss function
        """

        self.lr = 0.002
        self.batch_size = 1000

        self.model = Net()
           
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = nn.MSELoss()

    def load_pretrained_model(self):
        """
        This loads the parameters saved in bestmodel.pth into the model
        """
        
        #This is needed to have the right absolute path of bestmodel.pth
        file_abs_path = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(file_abs_path, 'bestmodel.pth')
        
        self.model.load_state_dict(torch.load(full_path,map_location=torch.device('cpu')))
        
        #If the model is run on a machine equipped with a GPU, the model is transfered to the GPU
        if torch.cuda.is_available():
            self.model.cuda()


    def train(self, train_input, train_target, num_epochs=100):
        """

        Parameters
        ----------
        train_input : Tensor
                      tensor of size (N, C, H, W) containing a noisy version of the images
        train_target : Tensor
                       tensor of size (N, C, H, W) containing another noisy version of the
                       same images , which only differs from the input by their noise .
        num_epochs : int, optional
                     Number of epochs that we want to train our model with. The default is 100.

        """
        
        #If the model is run on a machine equipped with a GPU, trasnfer the images to the GPU
        if torch.cuda.is_available():
            self.model.cuda()
            train_input = train_input.cuda()
            train_target = train_target.cuda()
            
        for e in range(num_epochs):
            i = 0
            for input, targets in zip(train_input.split(self.batch_size),  
                                      train_target.split(self.batch_size)):
                self.model.train()
                output = self.predict(input)
                loss = self.criterion(output/255, targets/255)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i+=1
              
            print('Nb of epoch: {:d}    loss: {:.04f}'.format(e, loss))

    def predict(self, input_imgs):
        """
        Parameters
        ----------
        input_imgs : Tensor
                     input_imgs : tensor of size (N1 , C, H, W) that has to be denoised by the trained
                     or the loaded network.
        """
        
        moved_to_GPU = False
        
        #if input_imgs is on the cpu even though there is a GPU, it means that the model is 
        #on the GPU. Therefore, we need to send input_imgs on the GPU, applay the model to it and then 
        #send it back to the cpu
        if torch.cuda.is_available() and not input_imgs.is_cuda:
            moved_to_GPU = True
            input_imgs = input_imgs.cuda()
        
        #normalize image between 0 and 1
        input_imgs_ = input_imgs/255
        output = self.model(input_imgs_)
        
        # output should be an int between [0,255]
        output = torch.clip(output*255, 0, 255)
        
        if moved_to_GPU:
            output = output.cpu()
        
        return output