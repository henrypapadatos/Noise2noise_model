import torch
from torch import nn
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self):
        ## instantiate model + optimizer + loss function + any other stuff you need

        super().__init__()
        self.lr = 0.001
        self.nb_epoch = 100
        self.batch_size = 500

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
        for e in range(self.nb_epoch):
            i = 0
            for input, targets in zip(train_input.split(self.batch_size),  
                                      train_target.split(self.batch_size)):
                output = self.model(input)
                loss = self.criterion(output, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                i+=1
                #print("Nb of batch is: "+str(i))
            print("Nb of epoch is: "+str(e))
        

        pass

    def predict(self, test_input):

        #: test˙input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network .
        pass
    
def display_img(img):
    
    image = img.permute(1,2,0)

    #if the image is float type, it has been normalized. We need to denormalize it before showing
    if image.dtype == torch.float32:
        image = (image*std+mean).int()
    
    plt.figure()
    plt.imshow(image)
    
#######################################################################################

subset = 1000
val_subste = 100
noisy_imgs_1 , noisy_imgs_2 = torch.load('train_data.pkl')
#We neet float type to 
noisy_imgs_1 = noisy_imgs_1[0:subset,:,:,:].float()
noisy_imgs_2 = noisy_imgs_2[0:subset,:,:,:].float()

mean, std = noisy_imgs_1.mean(), noisy_imgs_1.std()
noisy_imgs_1.sub_(mean).div_(std)
noisy_imgs_2.sub_(mean).div_(std)

noisy_imgs , clean_imgs = torch.load ('val_data.pkl')
noisy_imgs = noisy_imgs_1[0:subset,:,:,:].float()
clean_imgs = noisy_imgs_2[0:subset,:,:,:].float()

noisy_imgs.sub_(mean).div_(std)
clean_imgs.sub_(mean).div_(std)

model = Model()

#If your computer is equiped with a GPU, the computation will happen there
if torch.cuda.is_available():
    model.model.cuda()
    noisy_imgs_1 =noisy_imgs_1.cuda()
    noisy_imgs_2 = noisy_imgs_2.cuda()

model.train(noisy_imgs_1, noisy_imgs_2)

image_number = 0
display_img(noisy_imgs_1[image_number,:,:,:])
display_img(noisy_imgs_2[image_number,:,:,:])



plt.show()

print("test")