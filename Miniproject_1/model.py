import torch
from torch import nn
import matplotlib.pyplot as plt

class Model():
    def __init__(self):
        ## instantiate model + optimizer + loss function + any other stuff you need

        self.lr = 0.001

        self.model = nn.Sequential(
                          nn.Conv2d(3, 32, kernel_size=(5,5), stride=(1,1)),
                          nn.ReLU(),
                          nn.Conv2d(32, 32, kernel_size=(5,5), stride=(1,1)),
                          nn.ReLU(),
                          nn.Conv2d(32, 32, kernel_size=(4,4), stride=(2,2)),
                          nn.ReLU(),
                          nn.Conv2d(32, 32, kernel_size=(3,3), stride=(2,2)),
                          nn.ReLU(),
                          nn.Conv2d(32, 8, kernel_size=(4,4), stride=(1,1)),
                          nn.ReLU(),
                          nn.ConvTranspose2d(8, 32, kernel_size=(4,4), stride=(1,1)),
                          nn.ReLU(),
                          nn.ConvTranspose2d(32, 32, kernel_size=(3,3), stride=(2,2)),
                          nn.ReLU(),
                          nn.ConvTranspose2d(32, 32, kernel_size=(4,4), stride=(2,2)),
                          nn.ReLU(), 
                          nn.ConvTranspose2d(32, 32, kernel_size=(5,5), stride=(1,1)),
                          nn.ReLU(),
                          nn.ConvTranspose2d(32, 3, kernel_size=(5,5), stride=(1,1)),
                          nn.ReLU(),                       
                          )

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        self.criterion = nn.MSELoss()
        pass

    def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel .pth into the model
        pass

    def train(self, train_input, train_target):
        #: train˙input : tensor of size (N, C, H, W) containing a noisy version of the images
        #: train˙target : tensor of size (N, C, H, W) containing another noisy version of the
        # same images , which only differs from the input by their noise .
        
        

        pass

    def predict(self, test_input):

        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network .
        pass

noisy_imgs_1 , noisy_imgs_2 = torch.load('train_data.pkl')
noisy_imgs_1 = noisy_imgs_1.float()
noisy_imgs_2 = noisy_imgs_2.float()

mean, std = noisy_imgs_1.mean(), noisy_imgs_1.std()
noisy_imgs_1.sub_(mean).div_(std)

sample = noisy_imgs_1[0,:,:,:].view(1,3,32,32)

model = Model()
prediction = model.model(sample)

def display_img(img):
    plt.figure()
    image = img.permute(1,2,0)
    plt.imshow(image)

image_number = 1
display_img(noisy_imgs_1[image_number,:,:,:])
display_img(noisy_imgs_2[image_number,:,:,:])



plt.show()

print("test")