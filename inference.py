
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
    out_of_range_coords = get_mgrid(64, 2).unsqueeze(0)
    image = io.imread(img_path)
    image=transform(image).unsqueeze(0)

    with torch.no_grad():
        model_out= model(image,out_of_range_coords)
        fig, ax = plt.subplots(figsize=(16,16))
        img=model_out.cpu().view(64,64).numpy()
        # img=np.where(img<10,0,255)
        ax.imshow(img)
        # plt.show()
        #plt.savefig("output/final_prediction.png")
        plt.imsave(config.GENERATED_SKEL,img)

if __name__=="__main__":
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
    model=NeuralSignedDistanceModel.load_from_checkpoint(config.CKPT_DIR_PATH+"model_loss(train_loss=158.92906188964844)_best_epoch=499.ckpt")
    img_path=config.TEST_DATA
    generateSkeleton(img_path,model,transform)

 