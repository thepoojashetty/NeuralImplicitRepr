from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch.nn as nn
import torch 
import torch.nn.functional as F
import  pytorch_lightning as pl
import torch.optim as optim
import numpy as np
from helpers import *
    
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class NeuralSignedDistanceModel(pl.LightningModule):
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.lr=learning_rate
        self.hidden=32
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=self.hidden,kernel_size=5,stride=2,padding=3),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden,out_channels=self.hidden*2,kernel_size=5,stride=2,padding=2),
            nn.BatchNorm2d(self.hidden*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden*2,out_channels=self.hidden*3,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(self.hidden*3),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden*3,out_channels=self.hidden*4,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(self.hidden*4),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden*4,out_channels=self.hidden*4,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(self.hidden*4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3200,64)
        )

        self.siren_net=self.siren(in_features=66, out_features=1, hidden_features=256, hidden_layers=5, outermost_linear=True)

        self.loss=nn.MSELoss()
        self.save_hyperparameters()
    
    def siren(self,in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, first_omega_0=30, hidden_omega_0=30.):
        net = []
        net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            net.append(final_linear)
        else:
            net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        return nn.Sequential(*net)

    def forward(self,x,pixel_coord):
        output=self.layers(x)
        #scale between -1 and 1
        #minval=output.min().item()
        #maxval=output.max().item()
        #output=-1+2*(output-minval)/(maxval-minval)
        output=torch.cat((pixel_coord.to(torch.float32),output),dim=1)
        output=self.siren_net(output)
        #print(f"Out shape {output.shape}")
        return output

    def training_step(self, batch, batch_idx):
        loss,pred=self.common_step(batch,batch_idx)
        self.log('train_loss',loss,on_epoch=True)
        if self.current_epoch%20==0 and batch_idx==0:
            self.eval()
            with torch.no_grad():
                skel_imgs=self.generateSkeleton(batch=batch)
                self.logger.experiment.add_images("Glyphs",batch['image'])
                self.logger.experiment.add_images("Skeletons",skel_imgs)
            self.train()
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss,pred=self.common_step(batch,batch_idx)
        self.log('validation_loss',loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss,pred=self.common_step(batch,batch_idx)
        self.log('test_loss',loss)
        return loss

    def common_step(self,batch,batch_idx):
        image,pixel_coord=batch['image'],batch['pixel_coord']
        pred=self.forward(image,pixel_coord)
        loss=self.loss(pred,batch['sdv'])
        return loss,pred#
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(),lr=self.lr)
    
    def generateSkeleton(self,batch):
        image=batch['image']
        skel_imgs=torch.zeros_like(image)
        for i in range(skel_imgs.shape[2]):
            for j in range(skel_imgs.shape[3]):
                pixel_coord=torch.tensor(normalize([j,i])).expand(skel_imgs.shape[0],-1)
                out = self.forward(image,pixel_coord).view(skel_imgs.shape[0],1,1,1)
                #print("out:",out[:,0,0,0])
                skel_imgs[:,0,i,j]=out[:,0,0,0]
        #print("skel_img:",skel_imgs)
        skel_imgs=np.where(np.array(skel_imgs)<4,0,255)
        return skel_imgs
