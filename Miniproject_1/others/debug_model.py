import torch
from torch import nn
import matplotlib.pyplot as plt
import os

    
class NetBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, transpose_flag):
        super().__init__()

        self.stride = stride
        self.transpose_flag = transpose_flag
        
        if self.transpose_flag:
            if self.stride==1:
                self.convSkipTrans = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=(1,1), stride=(stride,stride))
                self.convTrans1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=(kernel_size,kernel_size), stride=(stride,stride), padding = (kernel_size -1)//2)
            elif self.stride==2:
                self.convTrans1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=(kernel_size,kernel_size), stride=(stride,stride), padding = (kernel_size -1)//2, output_padding = 1)
                self.convSkipTrans = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=(1,1), stride=(stride,stride), output_padding = 1)
         
        else:
            self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=(kernel_size,kernel_size), stride=(stride,stride), padding = (kernel_size -1)//2)
            self.convSkip = nn.Conv2d(input_channel, output_channel, kernel_size=(1,1), stride=(stride,stride))
        
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.Relu = nn.ReLU()

        

    def forward(self, x):
        if self.transpose_flag:
            y = self.convTrans1(x)
        else:
            y = self.conv1(x)
        y = self.bn1(y)
        y = self.Relu(y) 
        if self.transpose_flag:
            x = self.convSkipTrans(x)
        else:
            x = self.convSkip(x)
        y = y + x
        
        return y 


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        nb_channel = 32
        
        self.conv1 = nn.Conv2d(3, nb_channel, kernel_size=(3,3), stride=(1,1), padding = (3 -1)//2)
        self.conv5t = nn.ConvTranspose2d(nb_channel, 3, kernel_size=(3,3), stride=(1,1), padding = (3 -1)//2)

       
        
        self.NetBLock1 = NetBlock(nb_channel,nb_channel,5,2,transpose_flag=0) 
        self.TransNetBlock1 = NetBlock(nb_channel,nb_channel,5,2,transpose_flag=1)
        
        self.Dropout = nn.Dropout(0.2)


        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)

        self.Relu =  nn.ReLU()
        self.sigmoid = nn.Sigmoid()



        
    def forward(self,x):
        
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
    def __init__(self):
        ## instantiate model + optimizer + loss function + any other stuff you need

        self.lr = 0.002
        self.batch_size = 1000

        self.model = Net()
           
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = nn.MSELoss()

    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model$
        full_path = os.path.join('Miniproject_1', 'bestmodel.pth')
        self.model.load_state_dict(torch.load(full_path,map_location=torch.device('cpu')))
        
        if torch.cuda.is_available():
            self.model.cuda()


    def train(self, train_input, train_target, num_epochs=100 ,test_input=None, test_target=None, vizualisation_flag = False):
        #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the
        # same images , which only differs from the input by their noise .
        
        if torch.cuda.is_available():
            self.model.cuda()
            train_input = train_input.cuda()
            train_target = train_target.cuda()
            if test_input is not None: 
                test_input = test_input.cuda()
                test_target = test_target.cuda()
        
        #If flag is true, plot the clean image and the noisy image
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
                self.model.train()
                output = self.predict(input)
                loss = self.criterion(output/255, targets/255)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i+=1
            
            
            if test_input!=None:  
                self.model.eval()
                denoised = self.predict(test_input)
                psnr = self.psnr(denoised/255, test_target/255)                
                print('Nb of epoch: {:d}    psnr: {:.02f}'.format(e, psnr))
                
            #display denoised images 25 times during training
            if vizualisation_flag and (e%(num_epochs//10)==0) and test_input!=None:
                training_visualisation(denoised)

    def predict(self, input_imgs):
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network .
        
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
    
    def psnr(self, denoised, ground_truth):
        #Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
        mse = torch.mean((denoised - ground_truth)** 2)
        psnr = -10 * torch . log10 ( mse + 10** -8)
        return psnr.item()
    
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


    
def display_img(img):
    
    if torch.cuda.is_available():
        img = img.to('cpu')
    
    with torch.no_grad(): 
        image = img.permute(1,2,0)
        plt.figure()
        plt.imshow(image)
        
#######################################################################################
