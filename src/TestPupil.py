import numpy as np
from pathlib import Path
import math
import dLux as dl
import dLux.utils as dlu
import dLuxToliman as dlT
import OpticsSupport as OpticsSupport
import PlottingSupport as PlottingSupport
import matplotlib.pyplot as plt

# Aperture parameters
ratio = 1  # Ratio to scale the aperture by (e.g. 5 => 5-inch aperture becomes
# 1-inch aperture)
aperture_npix = 2000  # Number of pixels across the aperture
aperture_diameter = 0.125 / ratio  # Clear aperture diameter (m)
secondary_diameter = 0.032 / ratio  # Secondary mirror diameter (m)
spider_width = 0.002 / ratio  # Spider width (m)

# Observations wavelengths (bandpass of 530-640nm)
wavelengths = np.linspace(530e-9, 640e-9, 100)  # Wavelengths to simulate (m)

# Subtrate parameters
n1 = 1  # Refractive index of free space
n2 = 1.5424  # Refractive index of Zerodur

# Grating parameters
det_npixels = 3600  # DO NOT TOUCH
pixel_scale = dlu.arcsec2rad(0.375) * ratio  # 0.375 arcsec per pixel
max_reach = 0.691443  # Max wavelength to diffract to 80% of the diagonal

wf_npixels = aperture_npix  # Number of pixels across the wavefront
peak_wavelength = np.mean(wavelengths)  # Peak wavelength of the bandpass (m)

# For psf regions of interest
r = det_npixels
c = r // 2
s = 64

source = dlT.AlphaCen(n_wavels=3, separation=8, position_angle=30)

# Manufacturing tolerance on phase
abs_tolerance = 2  #(deg)
tolerances_deg = np.array([0, -abs_tolerance, abs_tolerance]) # ideal tolerance always first entry
tolerances = tolerances_deg* np.pi/180  #(rad)

# Mask(s)
mask_path_1 = Path("Generated Files/Engineering/modified_mask_lower.npy").absolute()
mask_path_2 = Path("Generated Files/Engineering/modified_mask_upper.npy").absolute()
phase_mask_1 = np.load(mask_path_1)  # Load the mask and convert to phase
phase_mask_2 = np.load(mask_path_2)

central_PSFs = [] #dict of tuples with (flux %, psf) produced by Optical system per mask 
sidelobe_PSFs = []
for tolerance in tolerances:
    # Phase of mask calculated according to manufacturing tolerance
    phase_mask = phase_mask_1 + np.where(phase_mask_2 > 0,
                                     phase_mask_2+tolerance, 0)
    
    # Creating the dLux objects
    mask = dl.Optic(phase=phase_mask)
    optics = dlT.TolimanOpticalSystem(
        wf_npixels=wf_npixels,
        mask=mask,
        psf_npixels=det_npixels,
        oversample=1,
    )

    instrument = OpticsSupport.HelperFunctions.createTolimanTelescope(
        optics, source, ratio
    )

    # Check the phase mask
    # print(optics)

    # PlottingSupport.Plotting.printColormap(
    #     phase_mask, title="Phase Mask", colorbar=True, colormap="viridis"
    # )

    # Simulate the PSF
    psf = instrument.model()

    # Crop to regions of interest (side lobes and center)
    central_psf = psf[c - s:c + s, c - s:c + s]
    sidelobe_psf = psf[
        math.floor(r / 7.5):math.floor(r / 7.5) + r // 10,
        math.floor(r / 7.5):math.floor(r / 7.5) + r // 10,
    ]

    # Store (flux %, psf) per region per mask
    central_PSFs.append( ((np.sum(central_psf)/np.sum(psf))*100, central_psf) )
    sidelobe_PSFs.append( ((np.sum(sidelobe_psf)/np.sum(psf))*4*100, sidelobe_psf) )

    # Status (for my sanity because this is so slow...)
    print("{} Tolerance Simulation Complete...".format(tolerance))


# Calc and plot residuals 
plt.rcParams['image.cmap'] = 'bwr'
plt.rcParams['axes.titlesize'] = 10
plt.figure(figsize=(10, 7))
sub_fig_it = 1
print("\n")
for i in range(1,len(tolerances)):
    # Central PSF residuals:
    c_flux_ideal, c_psf_ideal = central_PSFs[0]
    c_flux_tol, c_psf_tol = central_PSFs[i]
    central_psf_residual = c_psf_ideal - c_psf_tol
    c_flux_change_pc = (c_flux_tol - c_flux_ideal)/c_flux_ideal*100 # % increase/decrease in flux within region between psf's

    # Sidelobe PSF residuals:
    s_flux_ideal, s_psf_ideal = sidelobe_PSFs[0]
    s_flux_tol, s_psf_tol = sidelobe_PSFs[i]
    sidelobes_psf_residual = s_psf_ideal - s_psf_tol
    s_flux_change_pc = (s_flux_tol - s_flux_ideal)/s_flux_ideal*100

    # Plot
    ax = plt.subplot(len(tolerances) - 1, 2, sub_fig_it)
    im = ax.imshow(central_psf_residual)
    ax.set_title("Residual Central PSF\nIdeal vs {}deg shift of Phase Mask".format(tolerances_deg[i]))
    plt.colorbar(im, label="Intensity", ax = ax)
    sub_fig_it += 1 

    ax = plt.subplot(len(tolerances) - 1, 2, sub_fig_it)
    im = ax.imshow(sidelobes_psf_residual)
    ax.set_title("Residual Sidelobes PSF\nIdeal vs {}deg shift of Phase Mask".format(tolerances_deg[i]))
    plt.colorbar(im, label="Intensity", ax = ax)
    sub_fig_it += 1 

    # Print change in % flux
    print("{}deg shift resulting in {} % difference from ideal flux percentage in central psf".format(tolerances_deg[i], round(float(c_flux_change_pc),4)))
    print("{}deg shift resulting in {} % difference from ideal flux percentage in sidelobes psf\n".format(tolerances_deg[i], round(float(s_flux_change_pc),4)))

plt.tight_layout()
plt.show()
# # PlottingSupport.Plotting.printColormap(psf**0.2, title="PSF", colorbar=True)
# # plt.imsave("psf_full.png", psf**0.2, cmap="inferno")
# # PlottingSupport.Plotting.printColormap(
# #     central_psf, title="PSF Central", colorbar=True
# # )
# # print(f"Central Flux: {(np.sum(central_psf)/np.sum(psf))*100:.2f}%")
# PlottingSupport.Plotting.printColormap(
#     sidelobe_psf, title="PSF Sidelobe", colorbar=True
# )
# print(f"Sidelobe Flux: {(np.sum(sidelobe_psf)/np.sum(psf))*4*100:.2f}%")

# # bwr 