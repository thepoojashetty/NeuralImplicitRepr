
from GlyphDataset import GlyphDataModule
from model import NeuralSignedDistanceModel
import NIR_config_inf

from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from skimage import io
import matplotlib.pyplot as plt
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import os

#env name : ptorch

"""
def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to generate image skeletons')
    parser.add_argument('--data_path', type=str, help="path to data",default=path)
    parser.add_argument('--batch_size', type=int, default=128,help="number of samples in the batch")
    parser.add_argument('--epochs', type=int, default=5,help="number of epochs")
    parser.add_argument('--model_path',type=str,default="./Model/model.pt")
    return parser.parse_args()
"""

def generateSkeleton(img_path,model,device,transform):
    model.eval()
    image=io.imread(img_path)
    skel_img=np.zeros_like(image)
    model.to(device)
    image=transform(image).unsqueeze(0).to(device)
    #print("Shape:",image.shape,"\n")
    with torch.no_grad():
        for i in range(image.shape[2]):
            #print("i:",i,"\n")
            for j in range(image.shape[3]):
                #print("j:",j,"\n")
                pixel_coord=torch.tensor([i,j]).unsqueeze(0).to(device)
                skel_img[i][j] = model(image,pixel_coord)

    skel_img=np.where(np.array(skel_img)<4,0,255)
    plt.imshow(skel_img)
    print(skel_img)
    plt.savefig("Output_skel.png")

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform=transforms.Compose([
            transforms.ToTensor()
        ])
    model=NeuralSignedDistanceModel.load_from_checkpoint(NIR_config_inf.CKPT_DIR_PATH+"train_loss(train_loss=23.324462890625)_best_epoch=176.ckpt")
    img_path=NIR_config_inf.DATA_DIR+"/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_A_8.png"
    generateSkeleton(img_path,model,device,transform)

 