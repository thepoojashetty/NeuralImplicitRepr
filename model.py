from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch.nn as nn
import torch 
import torch.nn.functional as F
import  pytorch_lightning as pl
import torch.optim as optim
import numpy as np
import torchvision
    
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
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
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class NeuralSignedDistanceModel(pl.LightningModule):
    def __init__(self, learning_rate,in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.lr=learning_rate

        self.siren_net = []
        self.siren_net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.siren_net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.siren_net.append(final_linear)
        else:
            self.siren_net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.siren_net = nn.Sequential(*self.siren_net)

        resnet_model = torchvision.models.resnet18(pretrained=True)
        self.img_encoder =  nn.Sequential(*list(resnet_model.children())[:-1])
        self.img_encoder.add_module('Flatten', nn.Flatten())
        self.img_encoder.add_module('Linear', nn.Linear(512, 128))
        self.img_encoder.add_module('ReLU', nn.ReLU())
        self.img_encoder.add_module('Linear2', nn.Linear(128, 32))

        self.loss=nn.MSELoss()
        self.save_hyperparameters()

    def forward(self,x,pixel_coord):
        #our image has only one channel. We need to increase the number of channels to 3
        encoded_img = self.img_encoder(x.expand(-1, 3, -1, -1))
        siren_input = torch.cat((encoded_img.unsqueeze(1).expand(-1,4096,-1), pixel_coord), dim=-1)
        siren_input = siren_input.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        model_output = self.siren_net(siren_input)
        return model_output

    def training_step(self, batch, batch_idx):
        loss,pred=self.common_step(batch,batch_idx)
        self.log('train_loss',loss,on_epoch=True)
        # if self.current_epoch%20==0 and batch_idx==0:
        #     self.eval()
        #     with torch.no_grad():
        #         skel_imgs=self.generateSkeleton(batch=batch)
        #         self.logger.experiment.add_images("Glyphs",batch['image'])
        #         self.logger.experiment.add_images("Skeletons",skel_imgs)
        #     self.train()
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
        return optim.AdamW(lr=1e-4, params=self.parameters())
    
    def generateSkeleton(self,batch):
        image=batch['image'][0]
        pixel_coord=batch['pixel_coord'][0]
        model_out=self.forward(image.unsqueeze(0),pixel_coord.unsqueeze(0))
        skel_imgs=model_out.cpu().view(64,64).numpy()
        return skel_imgs
