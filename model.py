from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch.nn as nn
import torch 
import torch.nn.functional as F
import  pytorch_lightning as pl
import torch.optim as optim
import numpy as np

class SineLayer(nn.Module):
    def __init__(self,in_features,out_features,omega_0=30):
        super().__init__()
        self.omega_0=omega_0
        self.in_features=in_features
        self.out_features=out_features
        self.linear=nn.Linear(self.in_features,self.out_features)

        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)

    def forward(self,x):
        return torch.sin(self.omega_0*self.linear(x))

class NeuralSignedDistanceModel(pl.LightningModule):
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.lr=learning_rate
        self.hidden=32
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=self.hidden,kernel_size=5,stride=2,padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden,out_channels=self.hidden*2,kernel_size=5,stride=2,padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden*2,out_channels=self.hidden*3,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden*3,out_channels=self.hidden*4,kernel_size=3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden*4,out_channels=self.hidden*4,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.hidden*4,out_channels=self.hidden*4,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3200,64)
        )

        self.siren=nn.Sequential(
            SineLayer(66,32),
            SineLayer(32,10),
            nn.Linear(10,1)
        )

        self.loss=nn.MSELoss()
        self.save_hyperparameters()

    def forward(self,x,pixel_coord):
        output=self.layers(x)
        output=torch.cat((pixel_coord,output),dim=1)
        output=self.siren(output)
        #print(f"Out shape {output.shape}")
        return output

    def training_step(self, batch, batch_idx):
        loss,pred=self.common_step(batch,batch_idx)
        self.log('train_loss',loss)
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
        return loss,pred
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(),lr=self.lr)
    
    def generateSkeleton(self,batch):
        image=batch['image']
        skel_imgs=torch.zeros_like(image)
        for i in range(skel_imgs.shape[2]):
            for j in range(skel_imgs.shape[3]):
                pixel_coord=torch.tensor([i,j]).expand(skel_imgs.shape[0],-1)
                out = self.forward(image,pixel_coord).view(skel_imgs.shape[0],1,1,1)
                print("out:",out[:,0,0,0])
                skel_imgs[:,0,i,j]=out[:,0,0,0]
        print("skel_img:",skel_imgs)
        return skel_imgs
