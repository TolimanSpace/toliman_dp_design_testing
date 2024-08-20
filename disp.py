import numpy as np 
import matplotlib.pyplot as plt

fname = "/Volumes/Morgana2/gpir9156/toliman/non_glued/15_08_green_starphire1_80us_0gain_img_stack_batch_0.npy"
stretch = 0.5

plt.figure(figsize=(20,6))
data = np.load(fname)[-1,:,:]

plt.subplot(1,3,1)
plt.imshow(data**stretch)
plt.title("Starphire 1")

fname = "/Volumes/Morgana2/gpir9156/toliman/non_glued/15_08_green_starphire2_80us_0gain_img_stack_batch_0.npy"
data = np.load(fname)[-1,:,:]

plt.subplot(1,3,2)
plt.imshow(data**stretch)
plt.title("Starphire 2")

fname = "/Volumes/Morgana2/gpir9156/toliman/glued/15_08_green_149us_19.5gain_img_stack_batch_0.npy"
data = np.load(fname)[-1,:,:]

plt.subplot(1,3,3)
plt.imshow(data**stretch)
plt.title("Plates Combined")

plt.show()