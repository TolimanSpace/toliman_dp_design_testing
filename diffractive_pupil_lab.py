import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

fname = "/Volumes/Morgana2/gpir9156/toliman/diffractive_pupil/"

img_names = ["diffractive_pupil_img__80us_0.84gain_img_stack_batch_0.npy",
             "diffractive_pupil_img_wspider_160us_1.84gain_img_stack_batch_0.npy",
             "diffractive_pupil_img_47137us_11.56gain_img_stack_batch_0.npy",
             ] 
titles = ["No Spider", "Spider"]
cen = [1728,2680] #[row,col]
hlf_sz = 100
stretch = 0.5

plt.figure(figsize=(10,6))
for i in range(len(titles)):
    plt.subplot(1,len(titles),i+1)
    data = np.load(fname + img_names[i])[0,:,:]

    norm_psf = PowerNorm(gamma=stretch, vmin=data.min(), vmax=data.max())   
    plt.xlim([cen[1]-hlf_sz, cen[1]+hlf_sz])
    plt.ylim([cen[0]-hlf_sz, cen[0]+hlf_sz])
    plt.xticks([])
    plt.yticks([])  
    plt.imshow(data, norm=norm_psf, cmap='inferno')
    plt.title(titles[i])

plt.tight_layout()


mosaic = """ 
            ABDEGH
            C.F.IJ
    """
fig = plt.figure(constrained_layout=True, figsize=(10,3))
axes = fig.subplot_mosaic(mosaic,gridspec_kw={"wspace": 0.0,"hspace": 0.0},)  
axes_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
lobe_cens = [[[1525,926],[1943,4440],[3489,2475]], 
             [[1527,931],[1941,4443],[3490,2480]] , 
             [[548,1403],[427,3892],[3040,1490],[2922,3999]]
             ]
lobe_hlf_sz = 25
stretch = 0.5
plt_it = 0
for i in range(len(img_names)):
    data = np.load(fname + img_names[i])[0,:,:]
    norm_psf = PowerNorm(gamma=stretch, vmin=data.min(), vmax=data.max())  

    if i == 2:
        lobe_hlf_sz = lobe_hlf_sz*3

    centers = lobe_cens[i]
    for cen in centers:
        cropped_data = data[cen[0]-lobe_hlf_sz:cen[0]+lobe_hlf_sz, cen[1]-lobe_hlf_sz:cen[1]+lobe_hlf_sz]
        axes[axes_labels[plt_it]].imshow(cropped_data, norm=norm_psf, cmap='inferno')
        axes[axes_labels[plt_it]].set_xticks([])
        axes[axes_labels[plt_it]].set_yticks([])
        plt_it += 1

axes["A"].set_title("                               No Spider")
axes["D"].set_title("                               Spider")
axes["G"].set_title("                               Low laser")
plt.show()