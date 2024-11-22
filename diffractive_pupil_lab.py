import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

date = "11_11_24"

if date == "10_10_24":
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

if date == "11_11_24":
    dir = "/Volumes/Morgana2/gpir9156/toliman/diffractive_pupil/11_11_2024/"

    img_names = ["LHCP_red_img_133us_0gain_img_stack_batch_0.npy",
                "RHCP_red_img_69us_0gain_img_stack_batch_0.npy",
                "LHCP_thermal_img_47849us_0gain_img_stack_batch_0.npy",
                "RHCP_thermal_img_41414us_0gain_img_stack_batch_0.npy",
                ] 
    titles = ["LHCP Red", "RHCP Red", "LHCP Thermal", "RHCP Thermal"]
    hlf_sz = 100
    stretch = 0.5

    plt.figure(figsize=(10,10))
    subtplt_it = 1
    for im_file in img_names:
        data = np.load(dir + im_file)[-1,:,:]
        cen = np.unravel_index(np.argmax(data, axis=None), data.shape)
        
        norm_psf = PowerNorm(gamma=stretch, vmin=data.min(), vmax=data.max())   
        plt.subplot(int(len(img_names)/2),2,subtplt_it)
        plt.title(titles[subtplt_it-1])
        plt.xlim([cen[1]-hlf_sz, cen[1]+hlf_sz])
        plt.ylim([cen[0]-hlf_sz, cen[0]+hlf_sz])
        plt.xticks([])
        plt.yticks([])  
        plt.imshow(data, norm=norm_psf, cmap='inferno')
        subtplt_it += 1
    
    
    plt.show()