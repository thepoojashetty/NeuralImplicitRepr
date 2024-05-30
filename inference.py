
from GlyphDataset import GlyphDataModule
from model import NeuralSignedDistanceModel
import NIR_config_inf as config
from helpers import *

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

skel_paths = ["/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/skel/skel_Fust & Schoeffer Durandus Gotico-Antiqua 118G_E_14.png",
              "/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/skel/skel_Fust & Schoeffer Durandus Gotico-Antiqua 118G_D_13.png"]

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

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def generateSkeleton(img_path,model,transform):
    model.eval()
    coords = get_mgrid(64, 2).unsqueeze(0)
    out_imges=[[],[]]
    for i,img_p in enumerate(img_path):
        image = io.imread(img_p)
        out_imges[i].append(image)
        out_imges[i].append(io.imread(skel_paths[i]))
        image=transform(image).unsqueeze(0)

        with torch.no_grad():
            model_out= model(image,coords)
            fig, ax = plt.subplots(figsize=(16,16))
            img=model_out.cpu().view(64,64).numpy()
            out_imges[i].append(img)
            out_imges[i].append(np.where(img<1,0,255))
            out_imges[i].append(np.where(img<2,0,255))
            out_imges[i].append(np.where(img<3,0,255))
            out_imges[i].append(np.where(img<4,0,255))
            out_imges[i].append(np.where(img<5,0,255))
            # img=np.where(img<1,0,255)
            # ax.imshow(img)
            # plt.show()
            #plt.savefig("output/final_prediction.png")
            # plt.imsave(config.GENERATED_SKEL,img)
    
    fig,ax=plt.subplots(2,8,figsize=(16,5))
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(2):
        for j in range(8):
            if j in [2]:
                ax[i,j].imshow(out_imges[i][j])
            else:
                ax[i,j].imshow(out_imges[i][j],cmap='gray')
            ax[i,j].axis('off')
    #set the title for each column at the bottom
    #increase font size
    ax[0,0].set_title("Glyph",fontsize=24)
    ax[0,1].set_title("Skeleton",fontsize=24)
    ax[0,2].set_title("SDV",fontsize=24)
    ax[0,3].set_title("T=1",fontsize=24)
    ax[0,4].set_title("T=2",fontsize=24)
    ax[0,5].set_title("T=3",fontsize=24)
    ax[0,6].set_title("T=4",fontsize=24)
    ax[0,7].set_title("T=5",fontsize=24)


    plt.savefig(config.GENERATED_SKEL+"final_prediction.png")


if __name__=="__main__":
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
    model=NeuralSignedDistanceModel.load_from_checkpoint(config.CKPT_DIR_PATH+"stepLR_B256_model_loss(train_loss=1.016628384590149)_best_epoch=457.ckpt")
    img_path=[config.TEST_DATA1,config.TEST_DATA2]
    generateSkeleton(img_path,model,transform)

 