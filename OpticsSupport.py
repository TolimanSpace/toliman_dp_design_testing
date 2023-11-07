import jax.numpy as np
import dLux.utils as dlu
import dLuxToliman as dlT


class HelperFunctions:
    def __init__(self) -> None:
        pass

    @staticmethod
    def generate_sinusoids(
        aperture_diameter, npixels, period, amplitude, rotation=np.pi / 4
    ):
        """
        Calcaultes the sinusoidal grating and anti-grating patterns and their
        corresponding pixel positions.

        Parameters
        ----------
        aperture_diameter: float, meters
            Size of the aperture in metres
        npixels: int
            Number of pixels to calculate the grating on
        period: float, meters
            Period of the grating in metres
        amplitude: float, meters
            Amplitude of the grating in metres
        rotation: float, radians
            Rotation of the grating in radians

        Returns
        -------
        grating: np.ndarray
            2D array of the grating pattern
        anti_grating: np.ndarray
            2D array of the anti-grating pattern
        X: np.ndarray
            2D array of the X pixel positions
        Y: np.ndarray
            2D array of the Y pixel positions
        """
        # Cacluate coordinates
        X, Y = dlu.pixel_coords(npixels, aperture_diameter)

        # Rotate the coordinates
        R, PHI = dlu.cart2polar([X, Y])
        PHI += rotation
        X, Y = dlu.polar2cart([R, PHI])

        # Define the sine argument value to get the correct period
        B = 2 * np.pi / period

        # Creating gratings
        grating = (amplitude * np.cos(B * Y) + amplitude * np.cos(B * X)) / 4
        anti_grating = (
            amplitude * np.cos(B * Y + np.pi)
            + amplitude * np.cos(B * X + np.pi)
        ) / 4

        # Note here we divide the output by 4:
        # amplitude is a peak to trough value so we must halve the output once
        # To get orthogonal sine waves we sum two seperate 1D sine patterns
        # so the amplitude is doubled so we must halve again

        return grating, anti_grating, X, Y

    @staticmethod
    def phase_to_depth(phase_array, wavelength, n1, n2):
        """
        Converts phase values to a depth for a specified wavelength through
        a substrate with refractive index n2, relative to the 'null' substrate
        with refractive index n1 (typically 1 for free space).

        Parameters
        ----------
        phase_array: np.ndarray
            2D array of phase values
        wavelength: float, meters
            Wavelength of the light in metres
        n1: float
            Refractive index of the substrate
        n2: float
            Refractive index of the top layer

        Returns
        -------
        depth_array: np.ndarray
            2D array of the depth values
        """
        return (phase_array * wavelength) / (2 * np.pi * (n2 - n1))

    @staticmethod
    def scale_binary_mask(phase_mask, aperture_npix):
        """
        Scales the phase mask to the output size, enforcing 0-pi values

        Parameters
        ----------
        phase_mask: np.ndarray
            2D array of phase values
        aperture_npix: int
            Number of pixels in the output array

        Returns
        -------
        scaled_mask: np.ndarray
            2D array of the scaled phase values
        """
        ratio = phase_mask.shape[0] / aperture_npix
        scaled_mask = dlu.scale(phase_mask, aperture_npix, ratio)
        scaled_mask = scaled_mask.at[np.where(scaled_mask >= np.pi / 2)].set(
            np.pi
        )
        scaled_mask = scaled_mask.at[np.where(scaled_mask < np.pi / 2)].set(0)
        return scaled_mask

    @staticmethod
    def calculate_grating_period(
        max_reach,
        pixel_scale,
        det_npixels,
        wavelengths,
        aperture_diameter,
        aperture_npix,
    ):
        """
        Calculates the grating period for a given set of parameters

        Parameters
        ----------
        max_reach: float, ratio
            Maximum wavelength to diffract to 'max_reach' to the edge of the
            chip
        pixel_scale: float, radians
            Pixel scale of the detector
        det_npixels: int
            Number of pixels in the full detector
        wavelengths: np.ndarray, meters
            Array of wavelengths to diffract
        aperture_diameter: float, meters
            Diameter of the aperture
        aperture_npix: int
            Number of pixels in the aperture

        Returns
        -------
        period: float
            Period of the grating
        """
        diffraction_angle = (
            max_reach * pixel_scale * np.sqrt(2) * (det_npixels / 2)
        )
        period = wavelengths.max() / np.sin(diffraction_angle)
        grating_sampling = period / (aperture_diameter / aperture_npix)
        print(f"Grating amplitude: {grating_sampling}")
        print(f"Nyquist Ratio: {grating_sampling/2}")
        return period

    @staticmethod
    def impose_grating(mask, grating, anti_grating):
        """
        Impose the grating onto the mask

        Parameters
        ----------
        mask: np.ndarray
            2D array of the mask
        grating: np.ndarray
            2D array of the grating
        anti_grating: np.ndarray
            2D array of the anti grating

        Returns
        -------
        full_mask: np.ndarray
            2D array of the full mask
        """
        Gmask = grating.at[np.where(mask == 0)].set(0.0)
        AGmask = anti_grating.at[np.where(mask != 0)].set(0.0)
        full_grating = Gmask + AGmask
        full_grating -= full_grating.min()
        full_mask = mask + full_grating
        return full_mask

    @staticmethod
    def ApertureFactory(
        aperture_npix: int,
        aperture_diameter: float,
        secondary_diameter: float,
        oversample: int = 1,
    ):
        """
        Creates an Aperture using new dLux method

        Args:
            aperture_npix (int): Number of pixels in the aperture
            aperture_diameter (float): Diameter of the aperture in meters
            secondary_diameter (float): Diameter of the secondary in meters
            oversample (int, optional): Oversample factor. Defaults to 1.

        Returns:
            Aperture: Aperture object
        """

        # Generate a set of coordinates
        coords = dlu.pixel_coords(
            aperture_npix * oversample, aperture_diameter
        )

        # Create the aperture and mirror
        primary = dlu.circle(coords, aperture_diameter)
        secondary = dlu.circle(coords, secondary_diameter)

        aperture = dlu.combine([primary, secondary], oversample)

        return aperture

    @staticmethod
    def make_grating_mask(
        phase_mask: np.ndarray,
        aperture_npix: int,
        aperture_diameter: float,
        secondary_diameter: float,
        spider_width: float,
        wavelengths: np.ndarray,
        amplitude: float,
        det_npixels: int,
        pixel_scale: float,
        max_reach: float,
        n1: float,
        n2: float,
        out: float = np.nan,
        apply_spiders: bool = True,
        return_raw: bool = False,
    ):
        """
        Makes a full grating mask for a given set of parameters

        Parameters
        ----------
        phase_mask: np.ndarray
            2D array of phase values
        aperture_npix: int
            Number of pixels in the output array
        aperture_diameter: float, meters
            Diameter of the aperture
        secondary_diameter: float, meters
            Diameter of the secondary
        spider_width: float, meters
            Width of the spiders
        wavelengths: np.ndarray, meters
            Array of wavelengths to diffract
        amplitude: float
            Amplitude of the grating
        det_npixels: int
            Number of pixels in the full detector
        pixel_scale: float, radians
            Pixel scale of the detector
        max_reach: float, ratio
            Maximum wavelength to diffract to 'max_reach' to the edge of the
            chip
        n1: float
            Refractive index of the substrate
        n2: float
            Refractive index of the top layer
        out: float = np.nan
            Value to use for the outside of the aperture
        apply_spiders: bool = True
            Apply spiders to the aperture
        return_raw: bool = False
            Return the mask, full mask and support

        Returns
        -------
        full_mask: np.ndarray
            2D array of the full mask
        """

        # Scale mask
        scaled_mask = HelperFunctions.scale_binary_mask(
            phase_mask, aperture_npix
        )

        # Calculate depth
        mask = HelperFunctions.phase_to_depth(
            scaled_mask, wavelengths.mean(), n1, n2
        )

        # Calculate grating period
        period = HelperFunctions.calculate_grating_period(
            max_reach,
            pixel_scale,
            det_npixels,
            wavelengths,
            aperture_diameter,
            aperture_npix,
        )
        print(f"Grating period: {period}m")

        # Create Grating
        grating, anti_grating, X, Y = HelperFunctions.generate_sinusoids(
            aperture_diameter, aperture_npix, period, amplitude
        )

        # Impose grating
        mask = HelperFunctions.impose_grating(mask, grating, anti_grating)

        full_mask = mask

        if return_raw:
            return full_mask, mask, X, Y
        else:
            return full_mask

    @staticmethod
    def createPhaseMask(
        raw_mask: np.ndarray, mean_wavelength: float, n1: float, n2: float
    ):
        """Creates a phase mask from a raw mask

        Args:
            raw_mask (np.ndarray): Raw grating mask
            mean_wavelength (float): Mean wavelength of the bandpass
            n1 (float): Refractive index of free space
            n2 (float): Refractive index of the substrate

        Returns:
            phase_mask (np.ndarray): Full phase mask
        """
        opd_mask = raw_mask * (n2 - n1)
        phase_mask = dlu.opd2phase(opd_mask, mean_wavelength)

        return phase_mask

    @staticmethod
    def createTolimanTelescope(optics, source, ratio: float = 1):
        """Creates a Toliman Telescope

        Args:
            optics (Optics): Optics object
            source (Source): Source object
            ratio (float, optional): Ratio that scales the size of the pupil
            and telescope. Defaults to 1.
        Returns:
            instrument: Telescope object
        """
        instrument = dlT.Toliman(optics, source)
        instrument = instrument.set("diameter", instrument.diameter / ratio)
        instrument = instrument.set(
            "psf_pixel_scale", instrument.psf_pixel_scale * ratio
        )
        instrument = instrument.set(
            "source.separation", instrument.source.separation * ratio
        )

        return instrument

    @staticmethod
    def createCircularMask(h, w, center=None, radius=None):
        if center is None:  # use the middle of the image
            center = (int(w / 2), int(h / 2))
        if (
            radius is None
        ):  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w - center[0], h - center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

        mask = dist_from_center <= radius
        return mask

    @staticmethod
    def createCircularCutout(diameter_ratio, centre, aperture_npix):
        radius = aperture_npix * diameter_ratio / 2
        mask = HelperFunctions.createCircularMask(
            aperture_npix, aperture_npix, center=centre, radius=radius
        )
        return mask
