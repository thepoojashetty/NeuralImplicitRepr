from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch.nn as nn
import torch 
import torch.nn.functional as F
import  pytorch_lightning as pl
import torch.optim as optim
import numpy as np
import torchvision
from unet import UNet


class AutoencoderModel(pl.LightningModule):
    def __init__(self, learning_rate):
        super().__init__()
        self.lr=learning_rate

        self.unet = UNet()

        self.loss=nn.MSELoss()
        self.save_hyperparameters()

    def forward(self,x):
        unet_output = self.unet(x) # compute reconstruction loss
        return unet_output

    def training_step(self, batch, batch_idx):
        loss=self.common_step(batch,batch_idx)
        self.log('train_loss',loss,on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss=self.common_step(batch,batch_idx)
        self.log('validation_loss',loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss=self.common_step(batch,batch_idx)
        self.log('test_loss',loss)
        return loss

    def common_step(self,batch,batch_idx):
        image=batch['image']
        unet_output=self.forward(image)
        reconstruction_loss= self.loss(unet_output,image)
        return reconstruction_loss
    
    def configure_optimizers(self):
        return optim.AdamW(lr=1e-4, params=self.parameters())
