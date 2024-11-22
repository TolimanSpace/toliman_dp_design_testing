import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from skimage import feature
from decimal import Decimal

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
    # dir = "/Volumes/Morgana2/gpir9156/toliman/diffractive_pupil/11_11_2024/" # when running off local
    dir = "/import/morgana2/gpir9156/toliman/diffractive_pupil/11_11_2024/" #when ssh'ed into morgana


    img_names = ["LHCP_red_img_133us_0gain_img_stack_batch_0.npy",
                "RHCP_red_img_69us_0gain_img_stack_batch_0.npy",
                "LHCP_thermal_img_47849us_0gain_img_stack_batch_0.npy",
                "RHCP_thermal_img_41414us_0gain_img_stack_batch_0.npy",
                ] 
    titles = ["LHCP Red", "RHCP Red", "LHCP Thermal", "RHCP Thermal"]

    imgs = [np.load(dir + im_file)[-1,:,:] for im_file in img_names]

    hlf_sz_psf = 80 # for psf sz
    hlf_sz_lobe = 200 # for side lobes sz
    stretch = 0.5

    calib_frame = imgs[0]
    cen = np.unravel_index(np.argmax(calib_frame, axis=None), calib_frame.shape)
    nulled_data = calib_frame.copy() # null the central psf to find lobes
    nulled_data[cen[0]-200:cen[0]+200, cen[1]-200:cen[1]+200] = 0
    norm_psf = PowerNorm(gamma=0.1, vmin=nulled_data.min(), vmax=nulled_data.max())   

    # plt.imshow(nulled_data)
    # plt.colorbar()

    # find lobes
    maxima_idxs = feature.peak_local_max(np.asarray(nulled_data), threshold_abs=800, min_distance=200) 
    # for i, max_coord in enumerate(maxima_idxs):
    #     plt.plot(max_coord[1], max_coord[0], "yx")
    #     plt.text(max_coord[1], max_coord[0], str(i), c='y')
    # plt.savefig("test.png")
    # exit()
    plt.figure(figsize=(10,10))
    subtplt_it = 1
    for i, img in enumerate(imgs):
        norm_psf = PowerNorm(gamma=stretch, vmin=img.min(), vmax=img.max())   
        plt.subplot(int(len(img_names)/2),2,subtplt_it)
        plt.title(titles[i])
        plt.xlim([cen[1]-hlf_sz_psf, cen[1]+hlf_sz_psf])
        plt.ylim([cen[0]-hlf_sz_psf, cen[0]+hlf_sz_psf])
        plt.xticks([])
        plt.yticks([])  
        plt.imshow(img, norm=norm_psf, cmap='inferno', origin='upper')
        subtplt_it += 1

        # if i == 1 or i == 3:
        #     # div by 100 to avoid overflow errs
        #     im_1 = imgs[i-1]/100 
        #     im_2 = imgs[i]/100
        #     diff = im_1-im_2
        #     max_diff = np.max(diff)
        #     max_diff *=100
        #     diff *= 100
        #     norm_resid = PowerNorm(gamma=1, vmin=-max_diff, vmax=max_diff)
        #     plt.subplot(int(len(img_names)/2),3,subtplt_it)
        #     plt.imshow(diff, norm = norm_resid, cmap='bwr', origin='upper')
        #     plt.title("Residuals")
        #     plt.colorbar(label="Intensity")
        #     plt.xlim([cen[1]-hlf_sz_psf, cen[1]+hlf_sz_psf])
        #     plt.ylim([cen[0]-hlf_sz_psf, cen[0]+hlf_sz_psf])
        #     plt.xticks([])
        #     plt.yticks([])
        #     subtplt_it += 1

    
    
    plt.savefig("11_11_24_center.png")
        
    # lobes
    mosaic = """
                AB.EF
                CD.GH
                IJ.MN
                KL.OP
            """
    fig = plt.figure(constrained_layout=True, figsize=(10,10))
    axes = fig.subplot_mosaic(mosaic,gridspec_kw={"wspace": 0.0,"hspace": 0.0},)  
    axes_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", 
                   "M", "N", "O","P"]
    ax_it = 0
    offset = 70
    offsets = [[offset,offset],[-offset, offset], [-offset,-offset], [offset,-offset]]
    for i, img in enumerate(imgs):
        norm_psf = PowerNorm(gamma=0.2, vmin=img.min(), vmax=img.max())   
        for j, max_idx in enumerate(maxima_idxs):
            if i > 1:
                max_idx[0] = max_idx[0] + offsets[j][0]
                max_idx[1] = max_idx[1] + offsets[j][1] 
                # axes[axes_labels[ax_it]].plot(tx,ty,'yx')

            axes[axes_labels[ax_it]].imshow(img, norm=norm_psf, cmap="inferno", origin='upper')
            axes[axes_labels[ax_it]].set_xlim([max_idx[1]-hlf_sz_lobe, max_idx[1]+hlf_sz_lobe])
            axes[axes_labels[ax_it]].set_ylim([max_idx[0]-hlf_sz_lobe, max_idx[0]+hlf_sz_lobe])
            axes[axes_labels[ax_it]].set_xticks([])
            axes[axes_labels[ax_it]].set_yticks([])

            ax_it +=1
    axes["A"].set_title("                               " + titles[0])
    axes["E"].set_title("                               "+ titles[1])
    axes["I"].set_title("                               "+ titles[2])
    axes["M"].set_title("                               "+ titles[3])
    plt.savefig("11_11_24_data_lobes.png")