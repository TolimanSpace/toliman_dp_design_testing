"""
    Test orientation of dLux simulation. Should pupil plane be constructed the way 
    the incoming WF sees it ?

    Is the detector image oriented the way the incoming WF sees it ?

    NOTE: yes, detector image is oriented the way thze incoming WF sees it.

    so simple flip about y-axis for data when comparing with real data from a detector
"""

import dLux as dl
import dLux.utils as dlu

import numpy as np
import matplotlib.pyplot as plt

# need to understand det orientation first


# single slit rotated 45 degrees
diam = 2.54e-2 # (m) 1inch diameter
npix = 256
coords = dlu.pixel_coords(npixels=npix, diameter=diam)
rect = dlu.rectangle(coords=coords, width=2e-3, height=0.5e-2)
centers = [[-2.5e-3,0],[2.5e-3,0]]# center of slits (m)

slits = []
for cen in centers:
    tf = dl.CoordTransform(translation=cen)
    slits.append(dl.RectangularAperture(height=0.5e-2, width=2e-3, transformation=tf))

tf = dl.CoordTransform(rotation=45*np.pi/180)
multi_slit = dl.MultiAperture(apertures=slits, transformation=tf)

trans = multi_slit.transmission(coords=coords, pixel_scale=diam/npix)

layers =[('aperture', dl.TransmissiveLayer(transmission=trans))]

os = dl.AngularOpticalSystem(wf_npixels=npix,
                             diameter=diam,
                             layers=layers,
                             psf_npixels=npix,
                             psf_pixel_scale=500e-3,
                             )
psf = os.propagate_mono(wavelength=635e-9)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(trans)
plt.subplot(1,2,2)
plt.imshow(psf)
plt.show()

