# Noise2noise_model
The goal of the mini-projects is to implement a Noise2Noise model. A Noise2Noise model is an image denoising network trained without a clean reference image. 
The original paper can be found at https://arxiv.org/abs/1803.04189. 

The project has two parts, focusing on two different facets of deep learning. 
The exact description of the project can be found in "Project_description.pdf"

## Miniproject_1 
The first one is to build a network that denoises images using the PyTorch framework, in particular the torch.nn modules and autograd. 

## Miniproject_2
The second one is to understand and build a framework, its constituent modules, that are the standard building blocks of deep networks without PyTorch’s autograd.

Our results can be found in hte "report" file in each project file. 

"main_project_1.py" and "main_project_2.py" provide an example of how to run the models.

To reproduce the experiment you can download the dataset here: https://www.dropbox.com/sh/4brth6vhbyozo2s/AAARSsespGzzsjY3cVfp8mBMa?dl=0
Then place "train_data.pkl" and "val_data.pkl" in the top folder.

This project was realised in the scope of the class Deep learning (EE-559) thaught by François Fleuret. 