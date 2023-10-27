import jax.numpy as np
import stl
import dLux.utils as dlu
import dLuxToliman as dlT


class HelperFunctions:
    def __init__(self) -> None:
        pass

    @staticmethod
    def mesh_to_stl(
        X, Y, Z, N, file_name, unit_in="m", unit_out="mm", binary=True
    ):
        """
        Inputs:
            X: 2D numpy array of values defining the x position of each pixel
            ie for a 1x1m sized mesh object with a 0.1m resolution we expect
            X = [[0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
                [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
                                    ...
                [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]
                [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9]]

            Y: 2D numpy array of values defining the y position of each pixel
            ie for a 1x1m sized mesh object with a 0.1m resolution we expect
            Y = [[[0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]
                [0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1]
                                    ...
                [0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8]
                [0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9]]

            Z: 2D numpy array of values defining the z position (height) of
            each pixel using plt.imshow(Z) should yeild a 'surface map' type
            image

            N: The number of pixels in the array, ie X, Y, Z should all be NxN

            file_name: The path/name of the file name to save under

            unit_in: The input units of the X, Y, Z arrays. Default is meters.
            Can be 'm', 'mm', 'um', 'nm'

            unit_out: The output units of the stl file. ie for a 1x1m sized
            mesh object with units in mm we expect the X, Y output to range
            from 0 to 900. Default is milimeters. Can be 'm', 'mm', 'um', 'nm'

            binary: If True file will be output as binary (smaller), else
            ASCII (readable)

        Output:
            returns None, saves an stl file to disk

        -------------------------------------------------------------------
        Algorithm:
            Method: Decompose the traversal pattern into a series of simpler
            patterns and combine them to get the final pattern.

            For a gridsize of N, we have N-1 squares in each row/column,
            giving a total of (N-1)^2 sqaures and 2*(N-1)^2 triangles

            Each triangle is composed of three points and a so a square is
            made of 6 points

            Within the series of points there is a repeating pattern which can
            be decomposed into thee different sequences and combines

            S1: [0,0,0], [1,1,1], [1,1,1], [2,2,2], [2,2,2], [3,3,3], ...
            S2: [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,0,1], ...
            S3: [N,N,N], [0,0,0], [N,N,N], [0,0,0], [N,N,N], [0,0,0], ...
            Combining these together gives us our repeating pattern,
            S = S1 + S2 + S3

            Note: These brackes are there to help define the repeating
            sequence and do not relate to the final grouping into triangles.

            The final sequence is then [0, 1, S, N-1] and represents a full
            row of triangles. This final sequence is then repeated again
            with N added to each value, then again with 2*N added to each
            value and so on.
        """
        import numpy as np

        Z = np.array(Z)

        # Dictionary of indexes for scaling values
        dic = {
            "m": 0,
            "mm": 1,
            "um": 2,
            "nm": 3,
        }

        # Array of relevant scaling values
        # X axis is unit in, Y axis is unit out
        scale_arr = [
            [1, 1e-3, 1e-6, 1e-9],
            [1e3, 1, 1e-3, 1e-6],
            [1e6, 1e3, 1, 1e-3],
            [1e9, 1e6, 1e3, 1],
        ]

        points = (
            np.array([X.flatten(), Y.flatten(), Z.flatten()])
            * scale_arr[dic[unit_out]][dic[unit_in]]
        )

        # Other Numer of trianges and squares
        squares_per_row = N - 1
        num_squares = squares_per_row**2
        num_triangles = 2 * num_squares

        # Points per triangle and square
        points_per_triangle = 3
        points_per_square = 2 * points_per_triangle

        # Number of points per row
        points_per_row = squares_per_row * points_per_square

        # Size of the sequences
        sequence_size = points_per_row - 3
        full_sequence_size = sequence_size + 3

        # Create indexes used to create sequences
        i = np.linspace(0, sequence_size, num=sequence_size, endpoint=False)
        j = np.floor_divide(i, 3)

        # Sequence 1
        S1 = np.floor_divide(j + 1, 2)

        # Sequence 2
        S2_mask = (i + 1) % 3 == 0
        S2 = np.ones(sequence_size) * S2_mask

        # Sequence 3
        S3_mask = j % 2 == 0
        S3 = N * np.ones(sequence_size) * S3_mask

        # Add together
        S = S1 + S2 + S3

        # Non repeating start and end of array
        start = np.array([0, 1])
        end = np.array([squares_per_row])

        # Combine
        sequence = np.concatenate([start, S, end])
        base_sequence = np.tile(sequence, reps=squares_per_row)

        # Create meshgrid of integer N values for the different rows
        arr_x = np.linspace(
            0, full_sequence_size, num=full_sequence_size, endpoint=False
        )
        arr_y = np.linspace(
            0, squares_per_row, num=squares_per_row, endpoint=False
        )
        X, Y = np.meshgrid(arr_x, arr_y)
        incremental_sequence = (
            N * Y.flatten()
        )  # Flatten here in order to combine

        # Combine sequences together to create final array
        full_sequence_flat = base_sequence + incremental_sequence
        full_sequence = full_sequence_flat.reshape([num_triangles, 3])

        # Force integer values
        full_sequence = np.array(full_sequence, dtype=int)

        # Create the vertices and points arrays
        vertices = points.T
        faces = full_sequence

        # Create the mesh object
        grid_mesh = stl.mesh.Mesh(
            np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype)
        )

        # input mesh vectors
        for i, f in enumerate(faces):
            for j in range(3):
                grid_mesh.vectors[i][j] = vertices[f[j], :]

        # Write the mesh to file
        if binary:
            grid_mesh.save("{}.stl".format(file_name))
        else:
            grid_mesh.save("{}.stl".format(file_name), mode=stl.Mode.ASCII)

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
        # Generate Aperture support
        # secondary_ratio = secondary_diameter/aperture_diameter
        # if apply_spiders:
        #     spider_ratio    = spider_width/aperture_diameter
        #     aperture = dl.ApertureFactory(aperture_npix,
        #                                 secondary_ratio=secondary_ratio,
        #                                 nstruts=3,
        #                                 strut_rotation=-np.pi/2,
        #                                 strut_ratio=spider_ratio)
        # else:
        # aperture = self.ApertureFactory(
        #     aperture_npix, aperture_diameter, secondary_diameter
        # )
        # support = aperture.transmission

        # Scale mask
        scaled_mask = HelperFunctions.scale_binary_mask(
            phase_mask, aperture_npix
        )

        # Calculate depth
        # mask = phase_to_opd(scaled_mask, wavelengths.mean())
        # mask = phase_to_depth(scaled_mask, wavelengths.mean() * 1e-9, n1, n2)
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

        # plt.imshow(grating)
        # plt.colorbar()
        # plt.show()

        # plt.imshow(mask)
        # plt.colorbar()
        # plt.show()

        # Impose grating
        mask = HelperFunctions.impose_grating(mask, grating, anti_grating)

        # plt.imshow(mask)
        # plt.colorbar()
        # plt.show()

        # Apply Support
        # full_mask = mask.at[np.where(support == 0)].set(out)
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
