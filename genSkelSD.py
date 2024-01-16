from skimage import io
from scipy.ndimage import distance_transform_edt
import numpy as np
import matplotlib.pyplot as plt

skel= io.imread("/Users/poojashetty/Documents/AI_SS23/Project/NeuralImplicitRepr/NIR/Testdata/img/Fust & Schoeffer Durandus Gotico-Antiqua 118G_A_8.png")
skel_signed_dist=distance_transform_edt(skel)

#save the signed distance map as png
#io.imsave throws error
#convert skel_signed_dist to numpy array
#save the image as png
plt.imshow(skel_signed_dist)
plt.show()



