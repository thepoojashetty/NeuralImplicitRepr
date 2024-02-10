import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time
from collections import OrderedDict
from skimage import io
from scipy.ndimage import distance_transform_edt
import torchvision
import itertools

import config

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

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

class NIR_net(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
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

        resnet_model = torchvision.models.resnet50(pretrained=True)
        resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.img_encoder =  nn.Sequential(*list(resnet_model.children())[:-1])
        self.img_encoder.add_module('Flatten', nn.Flatten())
        self.img_encoder.add_module('Linear', nn.Linear(2048, 256))
        self.img_encoder.add_module('ReLU', nn.ReLU())
        self.img_encoder.add_module('Linear2', nn.Linear(256, 32))

        
    def forward(self, images,coords):
        encoded_img = self.img_encoder(images)
        siren_input = torch.cat((encoded_img.unsqueeze(1).expand(-1,4096,-1), coords), dim=-1)
        siren_input = siren_input.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        model_output = self.siren_net(siren_input)
        return model_output, siren_input


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def normalize(data):
    max_value=63
    norm_data= -1 + 2 * np.array(data) / max_value
    return norm_data

class HistoricalGlyphDataset(Dataset):
    def __init__(self,data_dir,num_of_coord=2):
        super().__init__()
        #number of pixels to sample
        self.num_of_coord=num_of_coord
        self.img_dir= os.path.join(data_dir,"img")
        self.skel_dir= os.path.join(data_dir,"skel")
        # self.images=np.repeat(np.array(sorted(os.listdir(self.img_dir))),self.num_of_coord)
        self.images=sorted(os.listdir(self.img_dir))
        # self.skel=np.repeat(np.array(sorted(os.listdir(self.skel_dir))),self.num_of_coord)
        self.skel=sorted(os.listdir(self.skel_dir))
        self.data=np.stack((self.images,self.skel),axis=1)
        #shuffle the data
        np.random.shuffle(self.data)
        self.transform=Compose([
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        self.mgrid=get_mgrid(64,2)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img_path=os.path.join(self.img_dir,self.data[idx][0].item())
        image=io.imread(img_path)
        image=self.transform(image)
        skel_path=os.path.join(self.skel_dir,self.data[idx][1].item())
        skel=io.imread(skel_path)
        skel_signed_dist=distance_transform_edt(skel)
        skel_signed_dist=self.transform(skel_signed_dist).to(torch.float32)
        skel_signed_dist = skel_signed_dist.permute(1, 2, 0).view(-1, 1)
        #index=idx%4096
        # index=np.random.randint(0,4096)

        sample= {'image':image,'pixel_coord': self.mgrid,'sdv':skel_signed_dist}
        return sample

if __name__ == "__main__":
    glyphData = HistoricalGlyphDataset(config.DATA_DIR,num_of_coord=config.NUM_OF_COORD)
    dataloader = DataLoader(glyphData, batch_size=128, pin_memory=True, num_workers=0)
    
    model = NIR_net(in_features=34, out_features=1, hidden_features=512,
                    hidden_layers=3, outermost_linear=True)

    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
    
    epochs=200

    for i in range(epochs):
        print("Epoch "+str(i)+"\n")
        for batch,sample in enumerate(dataloader):
            image,pixel_coord, ground_truth = sample['image'],sample['pixel_coord'],sample['sdv']
            model_output, coords = model(image,pixel_coord)
            loss = ((model_output - ground_truth)**2).mean()
            
            # if not i % 100:
            #     print("Epoch %d, Total loss %0.6f" % (i, loss))
            #     img_grad = gradient(model_output, coords)
            #     img_laplacian = laplace(model_output, coords)

            #     fig, axes = plt.subplots(1,3, figsize=(18,6))
            #     axes[0].imshow(model_output.cpu().view(64,64).detach().numpy())
            #     axes[1].imshow(img_grad.norm(dim=-1).cpu().view(64,64).detach().numpy())
            #     axes[2].imshow(img_laplacian.cpu().view(64,64).detach().numpy())
            #     plt.show()

            optim.zero_grad()
            loss.backward()
            optim.step()

        if(i%10==0):
            torch.save(model.state_dict(), config.CKPT_DIR_PATH+"model.pth")
        
    torch.save(model.state_dict(), config.CKPT_DIR_PATH+"model.pth")

    #inference
    model.load_state_dict(torch.load(config.CKPT_DIR_PATH+"model.pth"))
    model.eval()

    with torch.no_grad():
        out_of_range_coords = get_mgrid(64, 2).unsqueeze(0)
        image = io.imread(config.TEST_DATA)
        transform=Compose([
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        image=transform(image).unsqueeze(0)

        model_out, _ = model(image,out_of_range_coords)
        fig, ax = plt.subplots(figsize=(16,16))
        img=model_out.cpu().view(64,64).numpy()
        # img=np.where(img<10,0,255)
        ax.imshow(img)
        # plt.show()
        #plt.savefig("output/final_prediction.png")
        plt.imsave(config.GENERATED_SKEL,img)
        

