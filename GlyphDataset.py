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
from helpers import *

class HistoricalGlyphDataset(Dataset):
    def __init__(self,data_dir,num_of_coord=2,transform=None):
        super().__init__()
        #number of pixels to sample
        self.num_of_coord=num_of_coord
        self.img_dir= os.path.join(data_dir,"img")
        self.skel_dir= os.path.join(data_dir,"skel")
        self.data=np.repeat(np.array(sorted(os.listdir(self.img_dir))),self.num_of_coord).tolist()
        #print(self.data)
        self.skel=np.repeat(np.array(sorted(os.listdir(self.skel_dir))),self.num_of_coord).tolist()
        #print(self.skel)
        self.transform=transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        img_path=os.path.join(self.img_dir,self.data[idx])
        image=io.imread(img_path)
        skel_path=os.path.join(self.skel_dir,self.skel[idx])
        skel=io.imread(skel_path)
        skel_signed_dist=distance_transform_edt(skel)

        #Normalize 0 to 1
        #sampling random row and column value
        row=np.random.randint(0,skel.shape[0])
        col=np.random.randint(0,skel.shape[1])
        
        #Normalize the coordinates in the range -1 and 1
        pixel_coord=normalize([row,col])
        sample= {'image':image,'pixel_coord': torch.tensor(pixel_coord),'sdv':torch.tensor(skel_signed_dist[row][col],dtype=torch.float32)}
        if self.transform:
            sample['image']=self.transform(sample['image'])
        return sample

class GlyphDataModule(pl.LightningDataModule):
    def __init__(self,data_dir,batch_size,num_workers,transform, num_of_coord):
        super().__init__()
        self.data_dir=data_dir
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.transform=transform
        self.num_of_coord=num_of_coord

    def setup(self, stage):
        glyphDataset = HistoricalGlyphDataset(
            data_dir= self.data_dir,
            transform=self.transform,
            num_of_coord= self.num_of_coord
        )

        train_size = int(0.8*len(glyphDataset))
        validation_size = int(0.1*len(glyphDataset))
        test_size = len(glyphDataset)-train_size-validation_size

        self.train_subset,self.valid_subset,self.test_subset=random_split(glyphDataset,[train_size,validation_size,test_size])

    def train_dataloader(self) :
        return DataLoader (
            dataset=self.train_subset,
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