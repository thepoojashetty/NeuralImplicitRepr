from torchvision import datasets,transforms
import NIR_config_inf as config
from torchvision.utils import save_image
import torch
from skimage import io
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import numpy as np




if __name__ == '__main__':
    skel_path="/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/skel/skel_Fust & Schoeffer Durandus Gotico-Antiqua 118G_E_14.png"
    skel=io.imread(skel_path)
    skel=distance_transform_edt(skel)
    plt.axis('off')
    # plt.imshow(skel)
    plt.imsave(config.GENERATED_SKEL+"skel_sd_E.png",skel)
    # plt.show()