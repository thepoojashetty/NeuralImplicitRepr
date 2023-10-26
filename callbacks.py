import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import numpy as np
import torch
from skimage import io
from torchvision import transforms
import config


class LogImageCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.transform=transforms.Compose([
                transforms.ToTensor()
            ])
        self.image_to_test=io.imread(config.DATA_DIR+"/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_A_8.png")
        self.image_to_test=self.transform(self.image_to_test).unsqueeze(0)

    def on_train_epoch_end(self, trainer, pl_module):
        #save image to logger
        if trainer.current_epoch %20==0:
            print(f"Saving image at epoch {trainer.current_epoch}")
            pl_module.eval()
            with torch.no_grad():
                skel_img=self.generateSkeleton(trainer,pl_module)
                trainer.logger.experiment.add_image(f"Skeletons/{trainer.current_epoch}",skel_img)
            pl_module.train()

    def generateSkeleton(self,trainer,pl_module):
        skel_img=np.zeros((1,64,64))
        for i in range(skel_img.shape[1]):
            #print("i:",i,"\n")
            for j in range(skel_img.shape[2]):
                #print("j:",j,"\n")
                pixel_coord=torch.tensor([i,j]).unsqueeze(0)
                skel_img[0][i][j] = pl_module(self.image_to_test,pixel_coord)
        skel_img=np.where(np.array(skel_img)<4,0,255)
        return skel_img
        

        