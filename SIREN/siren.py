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

import  NIR_config_inf as config

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
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords 
    

def get_cameraman_tensor(sidelength):
    #img = Image.fromarray(skimage.data.camera())   
    skel= io.imread("/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_A_8.png")
    img=distance_transform_edt(skel)
    img=Image.fromarray(img)
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


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
        self.images=np.repeat(np.array(sorted(os.listdir(self.img_dir))),self.num_of_coord)
        #print(self.data)
        self.skel=np.repeat(np.array(sorted(os.listdir(self.skel_dir))),self.num_of_coord)
        #print(self.skel)
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
        index=np.random.randint(0,4096)
        #Normalize 0 to 1
        #sampling random row and column value
        # i=np.random.randint(0,skel.shape[0])
        # j=np.random.randint(0,skel.shape[1])
        
        #Normalize the coordinates in the range -1 and 1
        # pixel_coord=normalize(torch.tensor([i,j],dtype=torch.float32))
        #pixel_coord=[i,j]
        #plotMat(skel_signed_dist)
        #plt.clf()
        #plt.imshow(skel)
        #plt.scatter(i,j,color="red")
        #plt.show()
        sample= {'image':image,'pixel_coord': self.mgrid[index],'sdv':skel_signed_dist[index]}
        return sample

if __name__ == "__main__":
    #cameraman = ImageFitting(64)
    # glyphData = HistoricalGlyphDataset("/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata",num_of_coord=4096) 
    glyphData = HistoricalGlyphDataset(config.DATA_DIR,num_of_coord=config.NUM_OF_COORD)
    dataloader = DataLoader(glyphData, batch_size=4096, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=12, out_features=1, hidden_features=512,
                    hidden_layers=3, outermost_linear=True)

    img_encoder = nn.Sequential(
        nn.Conv2d(1, 32, 4, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(64, 128, 4, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(128, 256, 4, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(256, 512, 4, stride=2, padding=1),
        nn.GELU(),
        nn.Flatten(),
        nn.Linear(2048, 256),
        nn.GELU(),
        nn.Linear(256, 10),
    )

    # img_encoder =  torchvision.models.resnet18(pretrained=True)
    

    #img_siren.cuda()

    # total_steps = 500 # Since the whole image is our dataset, this just means 500 gradient descent steps.
    # steps_til_summary = 100

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    # model_input, ground_truth = next(iter(dataloader))
    # model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
    
    epochs=500

    # for step in range(total_steps):
    for i in range(epochs):
        print("Epoch "+str(i)+"\n")
        for batch,sample in enumerate(dataloader):
            image,pixel_coord, ground_truth = sample['image'],sample['pixel_coord'],sample['sdv']
            encoded_img = img_encoder(image)
            siren_input = torch.cat((encoded_img, pixel_coord), dim=-1)
            model_output, coords = img_siren(siren_input)
            loss = ((model_output - ground_truth)**2).mean()
            
            # if not i % 50:
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
        # model_output, coords = img_siren(model_input)
        # loss = ((model_output - ground_truth)**2).mean()
        
        # if not step % steps_til_summary:
        #     print("Step %d, Total loss %0.6f" % (step, loss))
        #     img_grad = gradient(model_output, coords)
        #     img_laplacian = laplace(model_output, coords)

        #     fig, axes = plt.subplots(1,3, figsize=(18,6))
        #     axes[0].imshow(model_output.cpu().view(64,64).detach().numpy())
        #     axes[1].imshow(img_grad.norm(dim=-1).cpu().view(64,64).detach().numpy())
        #     axes[2].imshow(img_laplacian.cpu().view(64,64).detach().numpy())
        #     plt.show()

        # optim.zero_grad()
        # loss.backward()
        # optim.step()

    # torch.save(img_siren.state_dict(), config.CKPT_DIR_PATH+"siren_img.pth")
    #inference
    img_siren.load_state_dict(torch.load(config.CKPT_DIR_PATH+"siren_img.pth"))
    with torch.no_grad():
        out_of_range_coords = get_mgrid(64, 2)
        # image=io.imread("/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_a_44.png")
        # image=io.imread("/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_A_8.png")
        image = io.imread(config.TEST_DATA)
        transform=Compose([
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        image=transform(image).unsqueeze(0)

        encoded_img = img_encoder(image)
        encoded_img = encoded_img.repeat(4096, 1)
        siren_input = torch.cat((encoded_img, out_of_range_coords), dim=-1)

        model_out, _ = img_siren(siren_input)
        fig, ax = plt.subplots(figsize=(16,16))
        img=model_out.cpu().view(64,64).numpy()
        # img=np.where(img<10,0,255)
        ax.imshow(img)
        # plt.show()
        #plt.savefig("output/final_prediction.png")
        plt.imsave(config.GENERATED_SKEL,img)
        

