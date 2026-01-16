import PySpin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import os
import csv

# Number of images to take
n_images = 1000
n_image_batch_to_save = 1 # save n_images in n_image_batch_to_save batches (memory override otherwise)

# Path to save images
# pth = "data/toliman/non_glued/08_09_2025/08_09_2025_plate2_0deg_red_"#"data/toliman/glued/28_07_2025/28_07_90deg_green_bckgnd_"
#pth = "data/toliman/spider/12_01_2026/12_01_2026_mask_red_"
#pth = "data/toliman/non_glued/12_01_2026/12_01_2026_plate#4_90deg_red_"
pth = "data/toliman/japan_data/test"


save_img = True
mono16 = True # control bit-depth
plt_max = False
cropping = False

if cropping:
    pth = pth + "cropped"
else:
    pth = pth + "fullframe"

# Exposure time
manual_exposure_time = 173#103 #(us) 
#manual_exposure_time = 300000
# Gain
manual_gain = 0 #(dB)

# Frame rate
frame_rate = 9

# Cropping
width = 300
height = 300
x_offset = 3352
y_offset = 1660

# Retrieve singleton reference to system object
system = PySpin.System.GetInstance()

# Get current library version
version = system.GetLibraryVersion()
print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

# Retrieve list of cameras from the system
cam_list = system.GetCameras()

num_cameras = cam_list.GetSize()

print('Number of cameras detected: %d' % num_cameras)

# Finish if there are no cameras
if num_cameras == 0:

    # Clear camera list before releasing system
    cam_list.Clear()

    # Release system instance
    system.ReleaseInstance()

    print('Not enough cameras!')

else:
    cam = cam_list[0] # default to taking first camera

    image_stack = []
    background_stack = []

    # Initialize camera
    cam.Init()

    # ---- Setting up Exposure ------
    # Turn off auto exposure
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    # Set exposure mode to "Timed"
    cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
    # Set exposure time
    cam.ExposureTime.SetValue(manual_exposure_time)
    # ----------------------------------

    # ---- Setting up Gain ------
    # Turn off auto gain
    cam.GainAuto.SetValue(PySpin.GainAuto_Off)
    # Set gain to 10.5 dB
    cam.Gain.SetValue(manual_gain)
    # ----------------------------------

    # ---- Setting up Frame rate ------
    cam.AcquisitionFrameRateEnable()
    cam.AcquisitionFrameRate.SetValue(frame_rate)

    # ---- Setting up Cropping ------
    cam.OffsetX.SetValue(0)
    cam.OffsetY.SetValue(0)
    cam.Height.SetValue(cam.Height.GetMax())
    cam.Width.SetValue(cam.Width.GetMax())

    if cropping:
        cam.Height.SetValue(height)
        cam.Width.SetValue(width)
        cam.OffsetX.SetValue(x_offset)
        cam.OffsetY.SetValue(y_offset)
        cam.AcquisitionFrameRate.SetValue(10)

    # ---- Setting up ADC Bit Depth ------
    cam.AdcBitDepth.SetValue(PySpin.AdcBitDepth_Bit10)

    # # ---- Setting up Black level ------
    # # Brightness is called black level in GenICam
    # cam.BlackLevelSelector.SetValue(PySpin.BlackLevelSelector_All)
    # # Set the absolute value of brightness to 1.5%.
    # cam.BlackLevel.SetValue(1.5)
    # # ----------------------------------

    # Retrieve GenICam nodemap
    nodemap = cam.GetNodeMap()

    # --- Set Pixel depth ---------------------------------------------------
    if mono16:
        str_id = 'Mono16'
    else:
        str_id = 'Mono8'

    # ---- Get Detector pos ------
    # Necessary if looking at px map
    node_offset_x = PySpin.CIntegerPtr(nodemap.GetNode('OffsetX')).GetValue()
    node_offset_y = PySpin.CIntegerPtr(nodemap.GetNode('OffsetY')).GetValue()
    print(node_offset_y, node_offset_x)

    node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
    if PySpin.IsReadable(node_pixel_format) and PySpin.IsWritable(node_pixel_format):

        # Retrieve the desired entry node from the enumeration node
        node_pixel_format_mono_x = PySpin.CEnumEntryPtr(node_pixel_format.GetEntryByName(str_id))
        if PySpin.IsReadable(node_pixel_format_mono_x):

            # Retrieve the integer value from the entry node
            pixel_format_mono_x = node_pixel_format_mono_x.GetValue()

            # Set integer as new value for enumeration node
            node_pixel_format.SetIntValue(pixel_format_mono_x)

            print('Pixel format set to %s...' % node_pixel_format.GetCurrentEntry().GetSymbolic())

        else:
            print('Pixel format not readable...')

    else:
        print('Pixel format not readable or writable...', node_pixel_format.GetCurrentEntry().GetSymbolic())

    # -----------------------------------------------------------------------------------------

    cam.BeginAcquisition()
    
    # Take n-images
    if plt_max:
        plt.figure()
    for i in range(n_images):
        image_result = cam.GetNextImage(1000)
        if image_result.IsIncomplete():
            print(
                "Image incomplete with image status %d ..."
                % image_result.GetImageStatus()
            )
        img = image_result.GetNDArray()

        # plt.imshow(img,cmap="gray") #5472x3648, mono16
        # plt.show()
        print("Maximum Intensity value: {}".format(img.max()))
        if plt_max:
            plt.scatter(i, img.max(), marker='.', c='k')
            plt.pause(0.02)

        image_stack.append(img)
        image_result.Release()

    cam.EndAcquisition()

    # Save png for quick determination of data too
    if save_img:
        im.imsave(fname= pth + "_image_test.png", 
                arr = image_stack[0], 
                vmin=0,vmax=(2**16-1),cmap="gray")

        save_n = int(n_images/n_image_batch_to_save)
        for i in range(n_image_batch_to_save):
            arr_image_stack = np.asarray(image_stack[i*save_n:(i*save_n + save_n)], dtype='uint16')
            print(arr_image_stack.shape)
            print(pth + str(manual_exposure_time) + "us_" + str(manual_gain) + "gain_Xoff"+ str(node_offset_x)+ "_Yoff"+str(node_offset_y)+ "_img_stack_batch_" + str(i)+ ".npy")
            # np.save(file = pth + str(manual_exposure_time) + "us_" + str(manual_gain) + "gain_Xoff"+ str(node_offset_x)+ "_Yoff"+str(node_offset_y)+ "_img_stack_batch_" + str(i)+ ".npy", 
            #         arr = arr_image_stack)
            np.save(file = pth + ".npy",arr = arr_image_stack)
            with open(pth+".csv","w") as metadata_file:
                writer = csv.writer(metadata_file, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(str(manual_exposure_time) + "us")
                writer.writerow(str(manual_gain) + "gain")
                writer.writerow(str(node_offset_x)+ "Xoff")
                writer.writerow(str(node_offset_y)+ "Yoff")
                writer.writerow(str(i)+ "img_stack_batch")

    del cam

# Clear camera list before releasing system
cam_list.Clear()

# Release system instance
system.ReleaseInstance()

if plt_max:
    plt.xlabel("Frame #")
    plt.ylabel("Maximum intensity")
    plt.show()
