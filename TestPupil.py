import numpy as np
import math
import dLux as dl
import dLux.utils as dlu
import dLuxToliman as dlT
import OpticsSupport
import PlottingSupport
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

# Mask
mask_path_1 = "modified_mask_lower.npy"
mask_path_2 = "modified_mask_upper.npy"
phase_mask_1 = np.load(mask_path_1)  # Load the mask and convert to phase
phase_mask_2 = np.load(mask_path_2)
phase_mask = phase_mask_1 + np.where(phase_mask_2 > 0,
                                     phase_mask_2-np.pi/18, 0)

# Grating parameters
det_npixels = 3600  # DO NOT TOUCH
pixel_scale = dlu.arcsec2rad(0.375) * ratio  # 0.375 arcsec per pixel
max_reach = 0.691443  # Max wavelength to diffract to 80% of the diagonal
# length of the detector

wf_npixels = aperture_npix  # Number of pixels across the wavefront
peak_wavelength = np.mean(wavelengths)  # Peak wavelength of the bandpass (m)

mask = dl.Optic(phase=phase_mask)
optics = dlT.TolimanOpticalSystem(
    wf_npixels=wf_npixels,
    mask=mask,
    psf_npixels=det_npixels,
    oversample=1,
)
source = dlT.AlphaCen(n_wavels=20, separation=8, position_angle=30)
instrument = OpticsSupport.HelperFunctions.createTolimanTelescope(
    optics, source, ratio
)

print(optics)

PlottingSupport.Plotting.printColormap(
    phase_mask, title="Phase Mask", colorbar=True, colormap="viridis"
)

psf = instrument.model()

r = psf.shape[0]
c = r // 2
s = 64

central_psf = psf[c - s:c + s, c - s:c + s]
sidelobe_psf = psf[
    math.floor(r / 7.5):math.floor(r / 7.5) + r // 10,
    math.floor(r / 7.5):math.floor(r / 7.5) + r // 10,
]
PlottingSupport.Plotting.printColormap(psf**0.2, title="PSF", colorbar=True)
plt.imsave("psf_full.png", psf**0.2, cmap="inferno")
PlottingSupport.Plotting.printColormap(
    central_psf, title="PSF Central", colorbar=True
)
print(f"Central Flux: {(np.sum(central_psf)/np.sum(psf))*100:.2f}%")
PlottingSupport.Plotting.printColormap(
    sidelobe_psf, title="PSF Sidelobe", colorbar=True
)
print(f"Sidelobe Flux: {(np.sum(sidelobe_psf)/np.sum(psf))*4*100:.2f}%")
