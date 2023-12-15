
from GlyphDataset import GlyphDataModule
from model import NeuralSignedDistanceModel
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
#from callbacks import LogImageCallback
import os

#env name : tenv

if __name__=='__main__':
    #args=parse_args()
    os.makedirs(config.CKPT_DIR_PATH, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
    model= NeuralSignedDistanceModel(learning_rate=config.LEARNING_RATE)
    data= GlyphDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        transform=transform,
        num_of_coord=config.NUM_OF_COORD
        )
    logger=TensorBoardLogger(save_dir=config.LOG_PATH,name="NIR_model")
    trainer=pl.Trainer(
        logger=logger,
        min_epochs=1,
        max_epochs= config.NUM_EPOCHS,
        overfit_batches=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=config.CKPT_DIR_PATH,
                filename="train_loss({train_loss})_best_{epoch}",
                monitor='train_loss',
                mode="min"
            ),
            #LogImageCallback()
            #EarlyStopping(monitor="validation_loss")
        ]
    )

    trainer.fit(model,data)
    trainer.validate(model,data)
    trainer.test(model,data)




    