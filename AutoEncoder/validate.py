
from model import AutoencoderModel
import NIR_config_inf as config
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

if __name__=="__main__":
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
    model=AutoencoderModel.load_from_checkpoint(config.AE_CKPT_DIR_PATH+"aemodel_loss(validation_loss=7.97853545009275e-07)_best_epoch=457.ckpt")
    img_path=config.TEST_DATA
    model.eval()
    image = io.imread(img_path)
    image=transform(image).unsqueeze(0)

    with torch.no_grad():
        out_img= model(image)
        #i want to plot image and out_img
        fig, ax = plt.subplots(1,2,figsize=(8,8))
        ax[0].imshow(image.squeeze().numpy())
        ax[1].imshow(out_img.squeeze().numpy())
        plt.show()
        #plt.savefig("output/final_prediction.png")


 