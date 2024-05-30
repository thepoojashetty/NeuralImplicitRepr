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

class HistoricalGlyphDataset(Dataset):
    def __init__(self,data_dir,transform):
        super().__init__()
        #number of pixels to sample
        self.img_dir= os.path.join(data_dir,"img")
        self.images=sorted(os.listdir(self.img_dir))
        self.transform=transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        img_path=os.path.join(self.img_dir,self.images[idx])
        image=io.imread(img_path)
        image=self.transform(image)
        sample= {'image':image}
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