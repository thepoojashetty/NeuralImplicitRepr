import numpy as np
import torch
from torch import nn 
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from skimage import io
from scipy.ndimage import distance_transform_edt

class HistoricalGlyphDataset(Dataset):
    def __init__(self,data_dir,n=2,transform=None) -> None:
        super().__init__()
        #number of pixels to sample
        self.n=n
        self.img_dir= os.path.join(data_dir,"img")
        self.skel_dir= os.path.join(data_dir,"skel")
        self.data=np.repeat(np.array(sorted(os.listdir(self.img_dir))),self.n).tolist()
        #print(self.data)
        self.skel=np.repeat(np.array(sorted(os.listdir(self.skel_dir))),self.n).tolist()
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

        #sampling random row and column value
        row=np.random.randint(0,skel.shape[0])
        col=np.random.randint(0,skel.shape[1])

        sample= {'image':image,'pixel_coord': torch.tensor([row,col]),'sdv':torch.tensor(skel_signed_dist[row][col],dtype=torch.float32)}
        if self.transform:
            sample['image']=self.transform(sample['image'])
        return sample


