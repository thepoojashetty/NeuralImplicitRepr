import numpy as np
import torch
from torch import nn 
from torch.utils.data import DataLoader,Dataset,random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from skimage import io
from scipy.ndimage import distance_transform_edt
import pytorch_lightning as pl

import matplotlib.pyplot as plt

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class HistoricalGlyphDataset(Dataset):
    def __init__(self,data_dir,transform):
        super().__init__()
        #number of pixels to sample
        self.img_dir= os.path.join(data_dir,"img")
        self.skel_dir= os.path.join(data_dir,"skel")
        # self.images=np.repeat(np.array(sorted(os.listdir(self.img_dir))),self.num_of_coord)
        self.images=sorted(os.listdir(self.img_dir))
        # self.skel=np.repeat(np.array(sorted(os.listdir(self.skel_dir))),self.num_of_coord)
        self.skel=sorted(os.listdir(self.skel_dir))
        self.data=np.stack((self.images,self.skel),axis=1)
        #shuffle the data
        #np.random.shuffle(self.data)
        self.transform=transform
        self.transform_skel= transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
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
        skel_signed_dist=self.transform_skel(skel_signed_dist).to(torch.float32)
        skel_signed_dist = skel_signed_dist.permute(1, 2, 0).view(-1, 1)
        #index=idx%4096
        # index=np.random.randint(0,4096)

        sample= {'image':image,'pixel_coord': self.mgrid,'sdv':skel_signed_dist}
        return sample

class GlyphDataModule(pl.LightningDataModule):
    def __init__(self,data_dir,batch_size,num_workers,transform):
        super().__init__()
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.transform=transform

    def setup(self, stage):
        self.glyphDataset = HistoricalGlyphDataset(
            data_dir= self.data_dir,
            transform=self.transform
        )

        train_size = int(0.8*len(self.glyphDataset))
        validation_size = int(0.1*len(self.glyphDataset))
        test_size = len(self.glyphDataset)-train_size-validation_size

        self.train_subset,self.valid_subset,self.test_subset=random_split(self.glyphDataset,[train_size,validation_size,test_size])

    def train_dataloader(self) :
        return DataLoader (
            dataset=self.glyphDataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self) :
        return DataLoader (
            dataset=self.valid_subset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def test_dataloader(self) :
        return DataLoader (
            dataset=self.test_subset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
def plotMat(skel_signed_dist):
    fig,ax=plt.subplots(figsize=(16,16))
    ax.matshow(skel_signed_dist,cmap=plt.cm.Blues)
    for i in range(64):
        for j in range(64):
            ax.text(j, i, str(round(skel_signed_dist[i, j],1)), ha='center', va='center',fontsize=5)
    plt.show()