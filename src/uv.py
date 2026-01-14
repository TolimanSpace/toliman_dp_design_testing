
"""
    A collection of classes and functions to handle Fourier analysis
"""

import jax.numpy as jnp
from jax import vmap

import dLux.utils as dlu
import dLux as dl

import numpy as np
from skimage import feature
from skimage import draw

import matplotlib.pyplot as plt
from typing import Union
import copy

__all__ = ["BasiiLayer", "mf", "compute_complex_vis", "SplodgeZernikeBasis"]

# function to hold a collection of bases that can update and eval net basis
class BasesLayer(dl.optical_layers.OpticalLayer):
    layers: dict

    def __init__(
            self: dl.optical_layers.OpticalLayer,
            layers: list[dl.layers.BasisLayer, tuple]
            ):
        
        self.layers = dlu.list2dictionary(layers, True, dl.layers.BasisLayer)

        for layer in self.layers.values():
            if not isinstance(layer, dl.layers.BasisLayer):
                raise TypeError(
                    f"Expected a list of BasisLayer objects, got {type(layer)}."
                )

    def __getattr__(self, key: str):
        """
        Raises both the individual layers and the attributes of the layers via
        their keys.

        Parameters
        ----------
        key : str
            The key of the item to be searched for in the layers dictionary.

        Returns
        -------
        item : object
            The item corresponding to the supplied key in the layers dictionary.
        """
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute " f"{key}."
        )
    

    def eval_basii(self):
        """
            Returns list of basis for each BasisLayer in layers.
        """
        eval_fn = lambda basis, coeff: dlu.eval_basis(basis, coeff)

        basii = [eval_fn(BasisLayer.basis, BasisLayer.coefficients) for BasisLayer in self.layers.values()]
        
        return basii
    
    def eval_basis(self):
        return jnp.asarray(self.eval_basii()).sum(axis=0)
    
    @property
    def coefficients(self):
        """
        Returns the coefficients of the aberrated apertures.

        Returns
        -------
        coefficients : list[Array]
            The coefficients of the aberrated apertures.
        """
        return [BasisLayer.coefficients for BasisLayer in self.layers.values()]

    @property
    def transmissions(self):
        # use basii shape to our advantage here
        # using simple binary transmission for now
        eval_fn = lambda basis, coeff: dlu.eval_basis(basis, coeff)

        basii = [jnp.asarray(eval_fn(BasisLayer.basis, jnp.ones(shape=BasisLayer.coefficients.shape))) for BasisLayer in self.layers.values()]
        trans = [b.at[b != 0.0].set(1.0) for b in basii]

        return trans


    @property
    def transmission(self):
        return jnp.asarray(self.transmissions).sum(axis=0)
    
    def apply(self: dl.layers.optical_layers.OpticalLayer, wavefront: dl.wavefronts.Wavefront) -> dl.wavefronts.Wavefront:
        """
        Apply the basis layers to a wavefront.

        """
        # NOTE no option for phase/normalisation implemented yet
        wavefront += self.eval_basis()

        return wavefront


def mf(
       pixel_sz: float, 
       n_pix: int,
       center_wl: float,
       mask_coords: jnp.array,
       stretch: float = 1.0,    
       rot: float = 0.0,
    ):
    """
        Match filter (u,v) coords for splodge sampling given mask dimensions and filter parameters.
        Method taken from AMICAL but stripped of BS and CPs info + jit compatible.
        
        Parameters:
        ------------
        pixel_sz: float
            Detector plate scale (arcsec/pixel)
        n_pix: int
            Number of pixels in detector (along one dimension, assuming square)
        center_wl: float
            Central wavelength of filter (m)
        mask_coords: jnp.array
            2D array containing the [x,y] mask coordinates of the sub-aperture centers (m),
            shape (n, 2) for n sub-apertures.
        stretch: float
            Factor to stretch the splodge sampling grid by
        rot: float
            Rotation to apply to uv coords (deg)
        
        Returns:
        --------
        dict with keys (and corresponding values):
        uv_real: (jnp.array, jnp.array)
            Splodge coordinates in u,v space
        uv_px: (jnp.array, jnp.array)
            Splodge coordinates in pixel space (as pixel indices).
        uv_coords: jnp.array
            (2, n_pix, n_pix) array of u,v coords for each pixel.
        h2bl: jnp.array
            2D array mapping hole pairs to baseline indices.
            h2bl[i,j] gives the baseline index for holes i,j.
        bl2h: jnp.array
            2D array mapping baseline indices to hole pairs.
            bl2h[0,i], bl2h[1,i] gives the hole pair for baseline i.
        n_holes: int
            Number of holes in mask.
        n_baselines: int
            Number of baselines in mask

    """
    pixel_sz *= 1/60 * 1/60 * jnp.pi/180  # arcsec to rad
    n_holes = mask_coords.shape[0]
    n_baselines = int(n_holes * (n_holes - 1) / 2)

    mask_coords = mask_coords * stretch
    mask_coords = mask_coords @ jnp.array([[jnp.cos(rot*jnp.pi/180), -jnp.sin(rot*jnp.pi/180)],[jnp.sin(rot*jnp.pi/180), jnp.cos(rot*jnp.pi/180)]]).T

    # ---- AMICAL (w Jax implementation) ----
    # Given a pair of holes i,j h2bl_ix(i,j) gives the number of the baseline
    h2bl_ix = jnp.zeros([n_holes, n_holes], dtype=int)
    count = 0
    for i in range(n_holes - 1):
        for j in jnp.arange(i + 1, n_holes):
            h2bl_ix = h2bl_ix.at[i, j].set(int(count))
            count = count + 1

    # Given a baseline, bl2h_ix gives the 2 holes that go to make it up
    bl2h_ix = jnp.zeros([2, n_baselines], dtype=int)

    count = 0
    for i in range(n_holes - 1):
        for j in jnp.arange(i + 1, n_holes):
            bl2h_ix = bl2h_ix.at[0, count].set(int(i))
            bl2h_ix = bl2h_ix.at[1, count].set(int(j))
            count = count + 1
    # ----------------
    # Get u,v coords for each baseline
    u_real = jnp.asarray([(mask_coords[bl2h_ix[0, i], 0] - mask_coords[bl2h_ix[1, i], 0]) / center_wl for i in range(n_baselines)])
    v_real = jnp.asarray([(mask_coords[bl2h_ix[0, i], 1] - mask_coords[bl2h_ix[1, i], 1]) / center_wl for i in range(n_baselines)])
    u_px = jnp.asarray(u_real*center_wl/pixel_sz + int(n_pix/2), dtype=int)
    v_px = jnp.asarray(v_real*center_wl/pixel_sz + int(n_pix/2), dtype=int)

    # BL length (m)
    bl_length = jnp.zeros(shape=(n_baselines,), dtype=float)
    for bl_id in range(n_baselines):
        x1 = mask_coords[bl2h_ix[0, bl_id], 0]
        y1 = mask_coords[bl2h_ix[0, bl_id], 1]
        x2 = mask_coords[bl2h_ix[1, bl_id], 0]
        y2 = mask_coords[bl2h_ix[1, bl_id], 1]

        bl_length = bl_length.at[bl_id].set(jnp.sqrt((x1 - x2)**2 + (y1 - y2)**2))

    out_dict = {
                "uv_real": (u_real, v_real),
                "uv_px": (u_px, v_px),
                "uv_coords": dlu.pixel_coords(npixels=n_pix, diameter=(n_pix*pixel_sz)/center_wl),
                "h2bl": h2bl_ix,
                "bl2h": bl2h_ix,
                "n_holes": n_holes, 
                "n_baselines": n_baselines,
                "bl_length": bl_length,
                }

    return out_dict

def compute_complex_vis(im_array: jnp.array):
    """
        Compute normalised v2 and phase from complex visibility calc on
        image array.
        TODO implement optional cleaning method
        
        Parameters:
        ------------
        im_array: Array
            2D square image array
        
        Returns:
        _________
        v: np.array
            Complex visibility function
        v2: np.array
            Normalised square mod of complex visibility function

        phase: np.array
             Complex visibility function
        
    """
    ft = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(im_array)))

    v2 = jnp.abs(ft)**2
    v2 /= v2[int(ft.shape[0]/2),int(ft.shape[0]/2)]
    phase = jnp.arctan2(ft.imag,ft.real) # [-pi, pi]

    return ft, v2, phase

def measured_phase(intensity_data: jnp.array):
    ft = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.fftshift(intensity_data)))
    phase = jnp.arctan2(ft.imag,ft.real) # [-pi, pi]
    return phase


def SplodgeZernikeBasis(
        uv_coords: jnp.array,
        splodge_radius: float,
        splodge_coords: jnp.array,
        zernike_idxs: jnp.array,
    ):

    """
        Create a set of zernikes basii, one for each splodge,
        stored in BasesLayer object.

        Parameters:
        ------------
        uv_coords: jnp.array
            Pixel coords in u-v space.
        splodge_radius: float
            Radius of splodge in u-v space. If None, single pixel splodge.
        splodge_coords: jnp.array
            2D array of splodge centers [u,v], in u-v space. 
        zernike_idxs: jnp.array
            Zernike Noll indices to use for each splodge.

        Returns:
        --------
        basii: BasesLayer object
            Object holding BasisLayer objects, one for each splodge.
    """

    basii = []
    if splodge_radius is None:
        wh = (2*uv_coords.max())/uv_coords.shape[1]*1/2 # single pixel width
        for i, (u, v) in enumerate(splodge_coords):
            # round to nearest pixel 
            d_u,d_v = jnp.abs(uv_coords[0]-u), jnp.abs(uv_coords[1]-v)
            u_ix, v_ix = jnp.unravel_index(jnp.argmin(d_u + d_v), d_u.shape)

            basis=jnp.zeros(shape=uv_coords[0].shape)
            basis= basis.at[u_ix, v_ix].set(1.0)
            basis_layer = dl.BasisLayer(basis=jnp.asarray([basis]))
            basii.append((f"splodge_{i}", basis_layer))

    else:
        for i, (u, v) in enumerate(splodge_coords):
            tf = dl.CoordTransform(translation=(u,v))
            circ = dl.CircularAperture(radius=splodge_radius,transformation=tf)
            ab_circ = dl.AberratedAperture(aperture=circ, noll_inds=zernike_idxs) 
            basis_layer = dl.BasisLayer(basis=ab_circ.calc_basis(coords=uv_coords))

            basii.append((f"splodge_{i}", basis_layer))

    return BasesLayer(basii)

class UVComponents():
    """
        Class for managing u,v analysis of images, given mask parameters and data. 
        Works for multiple masks/patterns simultaneously.
        
        More specifically class can be used to:
            - Gather u,v coords for the splodges given mask coordinates
            - Fit zernike polynomials across splodges 
            - Esimtate the piston, tip and tilt at each aperture

        Parameters:
        ------------
        mask_coords: list[np.array]
            List with each element an array containing the [x,y] mask coordinates 
            of the sub-aperture centers of shape (n, 2) for n sub-apertures. 
            Length X for X masks (m).
        uv_coords: list[np.array]
            Pixel coords defining u-v space. 
        bl2h: list[np.array]
            Each element contains a 2D array mapping baseline indices 
            to hole pairs.
            bl2h[0,i], bl2h[1,i] gives the hole pair for baseline i.
        n_holes: list[int]
            Number of holes in mask, per mask
        n_bl: list[int]
            Number of baselines in mask, per mask.
        sp_coord_sptl: list[np.array]
            Each element contains 2D array of splodge centers [u,v], 
            in u-v space for a given mask.
        sp_coord_px: list[np.array]
            Each element contains a 2D array of splodge centers [u,v], 
            in pixel space for a given mask.
        bl_length: list[np.array]
            Each element contains a 1D array of baseline lengths (m) for a given mask.
        central_wl: float
            Central wavelength of filter (m).
        px_scale: float
            Detector plate scale (arcsec/pixel).
        splodge_bases: list[BasesLayer]
            List of BasesLayer objects, one for each mask.
        uv_envelopes: list[jnp.array]
            List of 2D arrays for the splodge envelopes in u-v space,
            1's where splodes are, jnp.nans elsewhere.
        single_splodge_masks: list[list[jnp.array]]
            Each element contains a list of 2D arrays, with each
            array representing the mask for a single splodge. 1's
            where splodge exists, jnp.nans elsewhere.
        zernike_param_matrices: list[list[jnp.array]]
            Each element contains a list of 2D zernike parameter arrays each
            for a given splodge where columns correspond to the basis vector
            for each zernike polynomial modelled across the splodge.
        inv_est_matrices: list[jnp.array]
            Coefficient matrix for the estimation for tip and tilt across each aperture,
            per mask/pattern.

    """

    mask_coords: list[jnp.array]

    uv_coords: jnp.array
    bl2h: list[jnp.array]
    n_holes: list[int]
    n_bl: list[int]
    sp_coord_sptl: list[jnp.array]
    sp_coord_px: list[jnp.array]
    bl_length: list[jnp.array]

    central_wl: float
    px_scale: float
    n_pix: int

    splodge_bases: list[BasesLayer]
    uv_envelopes: list[jnp.array] 
    single_splodge_masks: list[list[jnp.array]]
    zernike_param_matrices: list[list[jnp.array]]

    inv_est_matrices: list[jnp.array]

    def __init__(self,
                 px_scale: float,
                 central_wl: float,
                 n_pix: int,
                 mask_coords: list,
                 sub_divide: dict = None,
                 ):
        """
            Parameters:
            ------------
            px_scale: float
                Detector plate scale (arcsec/pixel)
            central_wl: float
                Central wavelength of filter (m)
            n_pix: int
                Number of pixels across raw image (along one dimension, assuming square).
            mask_coords: list
                List with each element an array containing the [x,y] mask coordinates 
                of the sub-aperture centers of shape (n, 2) for n sub-apertures. 
                Length X for X masks (m).
            sub_divide: dict
                Dictionary with keys "PA", "d", where
                PA: list
                    The position angles (in degrees) to sub-divide each aperture into,
                    effectively creating sub-sub-apertures.
                    len(PA) is the number of sub-divisions per aperture.
                d: list
                    The distance from the center of the aperture to each of the 
                    sub-divided regions (m). len(d) is the number of sub-divisions per aperture.
        """
        
        self.mask_coords = list(mask_coords)
        assert len(self.mask_coords) > 0 and self.mask_coords[0].shape[1],\
              "Mask coordinates must be a list of arrays with shape (n, 2) for n sub-apertures."
        
        if sub_divide is not None:
            self.subdiv_mask_coords(sub_divide=sub_divide)

        self.central_wl = central_wl
        self.px_scale = px_scale
        self.n_pix = n_pix

        # Create match filter per given mask
        self.uv_coords, self.bl2h, self.n_holes, self.n_bl, self.sp_coord_px, self.sp_coord_sptl, self.bl_length = \
            self.get_mfs(
                pixel_sz=px_scale,
                n_pix=n_pix,
                canter_wl=central_wl,
                mask_coords=self.mask_coords,
                stretch=1.0, # default stretch factor for now until set in set_splodge_bases()
                rot=0.0,
            )
        
        self.splodge_bases = None # to be set after init - iterative process

        self.inv_est_matrices = self.get_inv_est_arr() # for tt estimation 

    def get_inv_est_arr(self):
        inv_matrices = []
        for pat_i in range(len(self.mask_coords)):
            A = jnp.zeros(shape=(self.n_bl[pat_i],self.n_holes[pat_i]))
            for baseline_i in range(self.bl2h[pat_i].shape[1]):
                hole_i, hole_j = self.bl2h[pat_i][0,baseline_i], self.bl2h[pat_i][1,baseline_i]
                A = A.at[baseline_i,[hole_i,hole_j]].set(1)
            A *= 0.5
            inv_A = jnp.linalg.pinv(A) 
            inv_matrices.append(inv_A)
        return inv_matrices
        
    def subdiv_mask_coords(self, sub_divide: dict):
        """
            Sub-divide the mask coordinates into circular sub-sub-apertures

            Parameters:
            ------------
            sub_divide: dict
                Dictionary with keys "PA", "d", where
                PA: list
                    The position angles (in degrees) to sub-divide each aperture into,
                    effectively creating sub-sub-apertures.
                    len(PA) is the number of sub-divisions per aperture.
                d: list
                    The distance from the center of the aperture to each of the 
                    sub-divided regions (m). len(d) is the number of sub-divisions per aperture.

            Returns:
            --------
                New mask coordinates describing the sub-divided apertures.
        """

        assert set(sub_divide.keys()) == set(["PA", "d"]),"""Sub-divide dictionary must contain keys 'PA' for sub-sub-aperture position 
                                    angles and 'd' distance from the center of the sub-aperture."""
        
        assert len(list(sub_divide["PA"])) == len(list(sub_divide["d"])), "Sub-divide dictionary must have equal length lists for 'PA' and 'd'."

        # create base template from origin and translate for each sub-aperture
        tf = [[d*jnp.cos(2*jnp.pi-theta_deg*jnp.pi/180), d*jnp.sin(2*jnp.pi-theta_deg*jnp.pi/180)]for d, theta_deg in zip(sub_divide["d"],sub_divide["PA"])] #[x,y]
        new_mask_coords = []
        for pat_coords in self.mask_coords:
            sub_div_coords = [jnp.asarray(sub_coord) + jnp.asarray(tf) for sub_coord in pat_coords]
            new_mask_coords.append(jnp.vstack(sub_div_coords))

        self.mask_coords = new_mask_coords

        return self.mask_coords


    def set_splodge_bases(self, 
                          splodge_radius: float, 
                          stretch: float = 1.0, 
                          rot: float = 0.0,
                          n_noll: int = 3,
                          plot: bool = False,
                          ):
        """
            Get splodge bases for each mask.
            Parameters:
            ------------
            splodge_radius: float
                Radius of splodge in u-v space.
            stretch: float
                Factor to stretch the splodge sampling grid by. Default 1.0.
            rot: float
                Rotation to apply to mask coords (deg), rotating uv coords.
            n_noll: int
                The highest order noll index to fit the phase over each splodge with.
                Default is 3 (i.e. noll indices 1, 2, 3).
            plot: bool
                If True, plot the splodge bases for each mask, including splodge envelope
                and center splodge coordinates in pixel space.
            Returns:
            --------
            splodge_bases: list[BasesLayer]
                List of BasesLayer objects, one for each mask.
        """
        zernike_idxs = jnp.arange(1, n_noll+1)

        # Update splodge coords
        _, self.bl2h, _, self.n_bl, self.sp_coord_px, self.sp_coord_sptl, self.bl_length = \
            self.get_mfs(
                pixel_sz=self.px_scale,
                n_pix=self.n_pix,
                canter_wl=self.central_wl,
                mask_coords=self.mask_coords,
                stretch=stretch, # default stretch factor for now until set in set_splodge_bases()
                rot=rot,
            )
        
        # Filter for unique uv samples 
        u_idxs = [jnp.unique(jnp.round(self.sp_coord_px[pat_i]),axis=0,return_index=True) for pat_i in range(len(self.sp_coord_px))]
        self.sp_coord_px = [self.sp_coord_px[pat_i][u_idxs[pat_i][1]] for pat_i in range(len(self.sp_coord_px))]
        self.sp_coord_sptl = [self.sp_coord_sptl[pat_i][u_idxs[pat_i][1]] for pat_i in range(len(self.sp_coord_sptl))]
        self.n_bl = [self.sp_coord_px[pat_i].shape[0] for pat_i in range(len(self.sp_coord_px))]
        self.bl_length = [self.bl_length[pat_i][u_idxs[pat_i][1]] for pat_i in range(len(self.bl_length))]
        self.bl2h = [self.bl2h[pat_i][:,u_idxs[pat_i][1]] for pat_i in range(len(self.bl2h))]

        splodge_bases = []
        for pat_i in range(len(self.mask_coords)):
            BasiiLayer = SplodgeZernikeBasis(
                            uv_coords=self.uv_coords,
                            splodge_radius=splodge_radius,
                            splodge_coords=self.sp_coord_sptl[pat_i],
                            zernike_idxs=zernike_idxs,
                            )
            splodge_bases.append(BasiiLayer)

        if plot:
            for pat_i in range(len(splodge_bases)):
                plt.figure(figsize=(5,4))
                e = self.uv_coords.max()
                extent = [-e,e,-e,e]
                plt.imshow(splodge_bases[pat_i].transmission, cmap='gray', origin='lower', extent=extent)
                plt.title("Splodge Envelope")
                # for ith_splodge, uv in enumerate(sp_coord_px[pat_i]):
                for ith_splodge, uv in enumerate(self.sp_coord_sptl[pat_i]):
                    plt.plot(uv[0], uv[1], 'rx')
                    plt.text(uv[0], uv[1], str(ith_splodge), color='b')

                plt.xlabel("u")
                plt.ylabel("v")

        self.splodge_bases = splodge_bases

        self.uv_envelopes, self.single_splodge_masks, self.zernike_param_matrices = self.get_splodge_fitting_matrices()

        return splodge_bases
    
    def get_splodge_fitting_matrices(self):
        """
            Returns the splodge fitting matrices for each mask.

            Returns:
            -------
            UV_ENVELOPES: list[jnp.array]
                List of 2D arrays for the splodge envelopes in u-v space,
                1's where splodges are, jnp.nans elsewhere.
            SINGLE_SPLODGES: list[list[jnp.array]]
                Each element contains a list of 2D arrays, with each
                array representing the mask for a single splodge. 1's
                where splodge exists, jnp.nans elsewhere.
            ZERNIKE_PARAM_MATRICES: list[list[jnp.array]]
                Each element contains a list of 2D zernike parameter arrays, each 
                for a given splodge. Columns correspond to the basis vector
                for each zernike polynomial modelled across the splodge.
        """
        if self.splodge_bases is None:
            raise ValueError("Splodge bases not set. Call set_splodge_bases() first.")

        UV_ENVELOPES, SINGLE_SPLODGES, ZERNIKE_PARAM_MATRICES = [], [], []
        for ModelBasiiLayer in self.splodge_bases:
            basii_layers = list(ModelBasiiLayer.layers.values())
            single_splodge_trans = ModelBasiiLayer.transmissions
            nan_single_splodge_trans = [t.at[t ==0.0].set(jnp.nan) for t in single_splodge_trans]
            SINGLE_SPLODGES.append(nan_single_splodge_trans)

            nan_mask_transmission = ModelBasiiLayer.transmission
            nan_mask_transmission= nan_mask_transmission.at[nan_mask_transmission ==0.0].set(jnp.nan)
            UV_ENVELOPES.append(nan_mask_transmission)

            zern_matrices=[]
            for j in range(len(single_splodge_trans)):
                b = basii_layers[j]

                zernike_basis_layers = [b.basis[k]*nan_single_splodge_trans[j]for k in range(b.basis.shape[0])]
                zernike_basis_layers = [basis_layer.flatten() for basis_layer in zernike_basis_layers]
                zernike_basis_layers = [basis_layer[~np.isnan(basis_layer)] for basis_layer in zernike_basis_layers] # remove nans

                A = jnp.column_stack(zernike_basis_layers)
                zern_matrices.append(A)   
    
            ZERNIKE_PARAM_MATRICES.append(zern_matrices)

        return UV_ENVELOPES, SINGLE_SPLODGES, ZERNIKE_PARAM_MATRICES
    
    def fit_splodge(self, ph_data, plot: bool=False):
        """
            Measures the phase accross each splodge of each mask as described
            by Zernike polynomials up to order n_noll specified in set_splodge_bases().

            Parameters:
            ------------
            ph_data: list[np.array]
                List of 2D arrays, each an array of size (n_pix, n_pix) 
                of phase of the complex visibilities of the data produced
                by each mask.
            
            Returns:
            -------
            FIT_UV_COEFFS: list[np.array]
                List of zernike noll coefficients from the phase fitting results to each
                splodge in each mask.

        """
        if self.splodge_bases is None:
            raise ValueError("Splodge bases not set. Call set_splodge_bases() first.")
        
        assert ph_data[0].shape == (self.n_pix,self.n_pix), \
        "Data of incorrect shape to detector size specification. Expected ({},{})".format(self.n_pix, self.n_pix)
        
        FIT_UV_COEFFS = []
        for i, ModelBasiiLayer in enumerate(self.splodge_bases):
            params=[key+".coefficients" for key in ModelBasiiLayer.layers.keys()]

            data = [ph_data[i]*trans for trans in self.single_splodge_masks[i]]

            # For each splodge, solve zernike coeffs 
            coeffs = []
            for j, splodge_data in enumerate(data):
                d1 = splodge_data.flatten()
                d1 = d1[~np.isnan(d1)] # remove nans

                A = self.zernike_param_matrices[i][j] 

                # LLS
                x_hat = jnp.linalg.lstsq(A, d1)[0]

                coeffs.append(x_hat)

            FIT_UV_COEFFS.append(jnp.asarray(coeffs))

            # Update bases 
            for j, param_str in enumerate(params):
                ModelBasiiLayer = ModelBasiiLayer.set(param_str, jnp.asarray(coeffs[j]))
            self.splodge_bases[i] = ModelBasiiLayer

            if plot:
                data_full = ph_data[i]*self.uv_envelopes[i]

                plt.figure(figsize=(18,5))
                plt.subplot(1,3,1)
                model_basii = ModelBasiiLayer.eval_basis()
                min_basii = min(model_basii.min(), data_full.min())
                max_basii = max(model_basii.max(), data_full.max())
                plt.imshow(model_basii, cmap='viridis', vmin=min_basii, vmax=max_basii)
                plt.title("Model")
                plt.colorbar()
                plt.subplot(1,3,2)
                plt.imshow(data_full, cmap='viridis', vmin=min_basii, vmax=max_basii)
                plt.title("Data")
                plt.colorbar()
                plt.subplot(1,3,3)
                diff = ModelBasiiLayer.eval_basis()-data_full
                diff_max= jnp.nanmax(jnp.abs(diff))
                plt.imshow(diff, cmap='bwr', vmin=-diff_max, vmax=diff_max)
                plt.colorbar()
                plt.title("Difference")

        return FIT_UV_COEFFS
    
    def read_splodge(self, ph_data: Union[list,jnp.array], mask_idx: int=None):
        """
            Direct pixel value read (no fitting for modes across splodge)
            ------------
            ph_data: list[jnp.array] or jnp.array
                List of 2D arrays, each an array of size (n_pix, n_pix) 
                of phase of the complex visibilities of the data produced
                by each mask OR a single 2D array if only reading a single mask.
            mask_idx: int
                If specified, only read splodge values for given mask index.
                If None, read for all masks.
            
        """
        if mask_idx is not None:
            return ph_data[self.sp_coord_px[mask_idx][:,1], self.sp_coord_px[mask_idx][:,0]]
        else:
            return [ph_data[i][self.sp_coord_px[i][:,1], self.sp_coord_px[i][:,0]] for i in range(len(self.sp_coord_px))]

    def filt_sp_coords(self, bool_filt: list[jnp.array]):
        """
            Filter the splodge coodinates an re-init new set.
            Parameters:
            ------------
            bool_filt: list[jnp.array]
                List of boolean arrays, one for each mask, 
                each of size (n_splodges,) indicating which splodges to keep.
            Returns:
            --------
            newUVHelper: UVComponents
                New UVComponents object with filtered splodge coordinates.
        """

        assert len(bool_filt) == len(self.mask_coords), \
        "Boolean filter list length must match number of masks."

        newUVHelper = copy.copy(self)

        newUVHelper.sp_coord_px = [self.sp_coord_px[i][bool_filt[i]] for i in range(len(self.mask_coords))]
        newUVHelper.sp_coord_sptl = [self.sp_coord_sptl[i][bool_filt[i]] for i in range(len(self.mask_coords))]

        return newUVHelper

    def estimate_tt(self, baseline_tt: list[jnp.array]):
        """
            Estimate the tip and tilt across each aperture given an ensemble
            of baseline tip/tilt fits. 
            
            For N > 3 sub-apertures, there are more baseline measurements 
            than sub-apertures, use SVD/PINV for overdetermined, usually 
            inconsistent system.  

            Parameters:
            ------------
            baseline_tt: list[np.array]
                List of 2D arrays, one for each pattern, each an array of size 
                (n_splodges, 2) of measured tip and tilt values for each splodge.

            Returns:
            -------
            ESTIMATED_TT: list[np.array]
                List of arrays, each an array of size (n_splodges,2) 
                containing estimated tip and tilt values for each splodge in each mask.
        """
        
        TT_ESTIMATES = []
        for pat_i in range(len(self.mask_coords)):
            tip_est_ph = jnp.matmul(self.inv_est_matrices[pat_i], baseline_tt[pat_i][:,0]).flatten()
            tilt_est_ph = jnp.matmul(self.inv_est_matrices[pat_i], baseline_tt[pat_i][:,1]).flatten()

            TT_ESTIMATES.append(jnp.vstack([tip_est_ph, tilt_est_ph]).T)

        return TT_ESTIMATES

    @property
    def splodge_masks(self):
        """
            Returns the splodge envelope for each mask.
            Each mask is a binary mask of splodges in image (1 = splodge, 0 = no splodge).

            Returns:
            -------
            splodge_masks: list[np.array]
                List of 2D arrays, each an array of size (n_pix, n_pix) 
                of splodge masks for each mask.
        """
        if self.splodge_bases is None:
            raise ValueError("Splodge bases not set. Call set_splodge_bases() first.")
        
        splodge_masks = [splodge_basis.transmission for splodge_basis in self.splodge_bases]
        
        return splodge_masks

    @staticmethod
    def get_mfs(pixel_sz: float,n_pix: int,canter_wl: float,mask_coords: list, stretch: float, rot: float = 0.0):
        """
            Get match filter information for each mask.

            Returns:
            ------- 
            UV_COORDS: np.array
                An array of size (2, n_pix, n_pix) 
                of u,v coords for each pixel.
            BL2H: list[np.array]
                X elements, one for each mask, each an array of size (2, n_baselines) 
                mapping baseline indices to hole pairs.
            N_HOLES: list[int]
                X elements, one for each mask, each an int of number of holes in mask.
            N_BL: list[int]
                X elements, one for each mask, each an int of number of baselines in mask.
            SP_COORD_PX: list[np.array]
                X elements, one for each mask, each an array of size (n_splodges, 2) 
                of splodge centers [u,v] in pixel space.
            SP_COORD_SPTL: list[np.array]
                X elements, one for each mask, each an array of size (n_splodges, 2) 
                of splodge centers [u,v] in u-v space.
            

        """
        UV_COORDS, BL2H, N_HOLES, N_BL, SP_COORD_PX, SP_COORD_SPTL, BL_L = [], [] ,[] ,[] ,[] ,[], []
        for mask_pattern in mask_coords:
            mf_dict = mf(
                pixel_sz=pixel_sz, 
                n_pix=n_pix,    
                center_wl=canter_wl,
                mask_coords=mask_pattern,
                stretch=stretch,
                rot=rot,
            )

            (u_real, v_real) = mf_dict["uv_real"]
            (u_px,v_px) = mf_dict["uv_px"]

            UV_COORDS.append(mf_dict["uv_coords"])
            BL2H.append(mf_dict["bl2h"]) 
            N_HOLES.append(mf_dict["n_holes"])
            N_BL.append(mf_dict["n_baselines"])
            SP_COORD_PX.append(jnp.column_stack((u_px, v_px)))
            SP_COORD_SPTL.append(jnp.column_stack((u_real, v_real)))
            BL_L.append(mf_dict["bl_length"])

        return UV_COORDS[0], BL2H, N_HOLES, N_BL, SP_COORD_PX, SP_COORD_SPTL, BL_L
    
def dummy_splodge_mask(im_array: np.array, min_peak: float, radius:int, prox_thresh: int):
    """
    Quick alternative to mask around splodges we care about (instead of calc splodge coords)

    Parameters:
    ------------
    im_array: np.array
        2D square image array
    min_peak: float
        Minimum intensity value for a peak (/local maxima) to be considered
    radius: int
        Radius of splodges in pixels
    prox_thresh: int
        Minimal distance allowed separating peaks
        
    Returns:
    _________
    splodge_mask: np.array
        Binary mask of splodges in image (1 = splodge, 0 = no splodge)
    
    """
    splodge_coords = feature.peak_local_max(np.asarray(im_array), threshold_abs=min_peak, min_distance=prox_thresh) 
    assert len(splodge_coords) > 0 and len(splodge_coords) < 1000, "None or too many splodges"

    # Draw mask 
    splodge_mask = jnp.zeros(im_array.shape)
    for cen in splodge_coords:
        rr, cc = draw.disk((cen[0], cen[1]), radius)
        splodge_mask = splodge_mask.at[rr, cc].set(1.0)
    
    return splodge_mask