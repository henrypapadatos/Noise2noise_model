import torch
from torch import nn
import matplotlib.pyplot as plt

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x
    
    

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
                #self.convSkipTrans = nn.Upsample(scale_factor=2, mode='bilinear')
         
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
    def __init__(self, batch_size):
        super().__init__() #parent class refers to nn.module
        
        nb_channel = 32
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(3, nb_channel, kernel_size=(3,3), stride=(1,1), padding = (3 -1)//2)
        self.conv5t = nn.ConvTranspose2d(nb_channel, 3, kernel_size=(3,3), stride=(1,1), padding = (3 -1)//2)
        self.Relu =  nn.ReLU()
        
        
        self.NetBLock1 = NetBlock(nb_channel,nb_channel,5,2,transpose_flag=0) 
        self.TransNetBlock1 = NetBlock(nb_channel,nb_channel,5,2,transpose_flag=1)
        self.NetBLock2 = NetBlock(32,32,3,1,transpose_flag=0) 
        self.TransNetBLock2 = NetBlock(32,32,3,1,transpose_flag=1) 
        self.Dropout = nn.Dropout(0.2)

        self.Pool = nn.MaxPool2d(kernel_size = 2, return_indices = True)
        self.unPool = nn.MaxUnpool2d(kernel_size = 2)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)

        self.sigmoid = nn.Sigmoid()

        self.linear = nn.Linear(32*16*16,32*16*16)


        # https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
        #self.downsample = downsample
    
        
    def forward(self,x):
        
        verbose = False
        
        if verbose:
            print("x_shape : ", x.shape)   
        y = self.conv1(x)
        if verbose:
            print("y_shape : ", y.shape)
        y = self.bn1(y)

        y = self.Relu(y)
        y = self.Dropout(y)

        y = self.NetBLock1(y)
        if verbose:
            print("y_shape : ", y.shape)
        #y, indices = self.Pool(y)
        if verbose:
            print("y_shape : ", y.shape)
        y = self.bn2(y)

        #y = self.Relu(y)
        y = self.Dropout(y)

        #y = self.NetBLock2(y)
        if verbose:
            print("y_shape : ", y.shape)
        #y = self.Relu(y)
        #y = self.Dropout(y)
        #y = self.bn3(y)

        #y = self.linear(y.view(self.batch_size,32*16*16))
        #y = self.Relu(y).view(self.batch_size,32,16,16)

        #y = self.TransNetBLock2(y)
        if verbose:
            print("y_shape : ", y.shape)
        #y = self.unPool(y, indices)
        if verbose:
            print("y_shape : ", y.shape)
        y =  self.TransNetBlock1(y)
        if verbose:
            print("y_shape : ", y.shape)
        y = self.conv5t(y)
        if verbose:
            print("y_shape : ", y.shape)

        y = self.sigmoid(y)
        return y

  
class Model():
    def __init__(self):
        ## instantiate model + optimizer + loss function + any other stuff you need

        super().__init__()
        self.lr = 0.002
        self.nb_epoch = 100
        self.batch_size = 1000

        self.model = Net(self.batch_size)
           
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = nn.MSELoss()

    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model
        pass

    def train(self, train_input, train_target, test_input, test_target, vizualisation_flag = False):
        #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the
        # same images , which only differs from the input by their noise .

        #If flag is true, plot the clean image and the noisy image
        if vizualisation_flag:
            plt.ion()
            plt.show()
            training_visualisation(test_target)
            training_visualisation(test_input)

        initial_psnr = self.psnr(test_input, test_target)
        print('Psnr value between clean and noisy images is: {:.02f}'.format(initial_psnr))

        for e in range(self.nb_epoch):
            i = 0
            for input, targets in zip(train_input.split(self.batch_size),  
                                      train_target.split(self.batch_size)):
                self.model.train()
                output = self.predict(input)
                loss = self.criterion(output, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i+=1
            
            self.model.eval()
            denoised = self.predict(test_input)
            psnr = self.psnr(denoised, test_target)
            
            print('Nb of epoch: {:d}    psnr: {:.02f}'.format(e, psnr))
            
            #display denoised images 25 times during training
            if vizualisation_flag and (e%(self.nb_epoch//10)==0):
                training_visualisation(denoised)

    def predict(self, input_imgs):
        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network .
        output = self.model(input_imgs)
        return output
    
    def psnr(self, denoised, ground_truth):
        #Peak Signal to Noise Ratio : denoised and ground˙truth have range [0 , 1]
        mse = torch.mean((denoised - ground_truth )** 2)
        psnr = -10 * torch . log10 ( mse + 10** -8)
        return psnr.item()
    
def training_visualisation(imgs):
    #Plot the 4 first images of imgs in a subplot way
    fig=plt.figure()
    nb_image = 4
    for a in range(nb_image):
        with torch.no_grad():     
            b = imgs[a,:,:,:]
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

subset_train = 1000
subset_test = 100
val_subste = 100
noisy_imgs_1 , noisy_imgs_2 = torch.load('train_data.pkl')

max_pxl = noisy_imgs_1.max().item()
#We neet float type to 
noisy_imgs_1 = noisy_imgs_1[0:subset_train,:,:,:]/max_pxl
noisy_imgs_1 = noisy_imgs_1.float()
noisy_imgs_2 = noisy_imgs_2[0:subset_train,:,:,:]/max_pxl
noisy_imgs_2 = noisy_imgs_2.float()
print(noisy_imgs_1.size())
test_imgs , clean_imgs = torch.load ('val_data.pkl')
test_imgs = test_imgs[0:subset_test,:,:,:]/max_pxl
test_imgs = test_imgs.float()
clean_imgs = clean_imgs[0:subset_test,:,:,:]/max_pxl
clean_imgs = clean_imgs.float()

noise2noise = Model()

#If your computer is equiped with a GPU, the computation will happen there
if torch.cuda.is_available():
    noise2noise.model.cuda()
    noisy_imgs_1 = noisy_imgs_1.cuda()
    noisy_imgs_2 = noisy_imgs_2.cuda()
    test_imgs = test_imgs.cuda()
    clean_imgs = clean_imgs.cuda()

noise2noise.train(noisy_imgs_1, noisy_imgs_2, test_imgs, clean_imgs, vizualisation_flag=True)    
    
image_number = 0
img = test_imgs[image_number,:,:,:]
display_img(img)
predicted = noise2noise.predict(img.unsqueeze(0))
display_img(predicted.squeeze())
plt.show
