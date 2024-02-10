
from GlyphDataset import GlyphDataModule
from model import NeuralSignedDistanceModel
import config

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
from pytorch_lightning.profilers import PyTorchProfiler
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
    model= NeuralSignedDistanceModel(learning_rate=config.LEARNING_RATE,in_features=34, out_features=1, hidden_features=128,
                    hidden_layers=3, outermost_linear=True)
    data= GlyphDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        transform=transform
        )
    logger=TensorBoardLogger(save_dir=config.LOG_PATH,name="NIR_model")
    profiler = PyTorchProfiler(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(config.LOG_PATH+"/profiler0"),
        schedule=torch.profiler.schedule(skip_first=3, wait=1,warmup=1,active=20)
    )
    trainer=pl.Trainer(
        logger=logger,
        # profiler=profiler,
        min_epochs=1,
        max_epochs= config.NUM_EPOCHS,
        callbacks=[
            ModelCheckpoint(
                dirpath=config.CKPT_DIR_PATH,
                filename="model_loss({train_loss})_best_{epoch}",
                monitor='train_loss',
                mode="min"
            ),
            #LogImageCallback()
            #EarlyStopping(monitor="validation_loss")
            ModelCheckpoint(
                dirpath=config.CKPT_DIR_PATH,
                filename="model_loss({val_loss:.2f})_lastepoch({epoch})",
            )
        ]
    )

    trainer.fit(model,data)
    trainer.validate(model,data)
    trainer.test(model,data)




    