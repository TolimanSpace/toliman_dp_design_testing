"""
    Set of classes to streamline phase retrieval of the toliman plates
"""

import dLux as dl
import dLux.utils as dlu
import zodiax as zdx

import jax.numpy as jnp
from jax import Array
import numpy as np

from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import pickle

# version of jax.scipy.stats.logppmf where k can be non-discrete (helpful if data is not discrete photon counts)
from jax.scipy.special import xlogy, gammaln
import jax.scipy as jsp
from jax._src.lax.lax import _const as _lax_const
from jax import lax, Array
from jax._src.numpy.util import promote_args_inexact
from jax.tree_util import tree_map, tree_flatten

from src.uv import compute_complex_vis

def jax_0_4_24_logpmf(k: jnp.array, mu: jnp.array, loc: jnp.array = 0) -> jnp.array:
    r"""Poisson log probability mass function.

    JAX implementation of :obj:`scipy.stats.poisson` ``logpmf``.

    The Poisson probability mass function is given by

    .. math::

        f(k) = e^{-\mu}\frac{\mu^k}{k!}

    and is defined for :math:`k \ge 0` and :math:`\mu \ge 0`.

    Args:
    k: arraylike, value at which to evaluate the PMF
    mu: arraylike, distribution shape parameter
    loc: arraylike, distribution offset parameter

    Returns:
    array of logpmf values.

    See Also:
    - :func:`jax.scipy.stats.poisson.cdf`
    - :func:`jax.scipy.stats.poisson.pmf`
    """
    k, mu, loc = promote_args_inexact("poisson.logpmf", k, mu, loc)
    zero = _lax_const(k, 0)
    x = lax.sub(k, loc)
    log_probs = xlogy(x, mu) - gammaln(x + 1) - mu
    #   return jnp.where(jnp.logical_or(lax.lt(x, zero),
    #                                   lax.ne(jnp.round(k), k)), -jnp.inf, log_probs)

    return jnp.where(lax.lt(x, zero), -jnp.inf, log_probs) # key diff in exit conditions to above

class OptimManager():
    """
        OptimManager handles input models, data and 
        optimisation specifications to calculate gradients,
        update models and run optimisation loops.

        Attributes:
        ------------
    """
    models: list
    data: list
    params: list
    optimisers: list
    jointly_fit_params: list

    param_update_dict: dict

    loss_fn: callable
    loss_fn_kwargs: dict

    def __init__(
            self,
            models: list,
            data: list,
            params: list,
            loss_fn: str,
            optimisers: list,
            jointly_fit_params: list = [],
            loss_fn_kwargs: dict = {}
            ):
        """
        Parameters
        ----------
        models : list
            List of dLux BaseOpticalSystem's or dLux Instruments to be optimised.
        data : list
            List of data to be used for optimisation. 
            Data should be the same length as models, each model is optimised
            uniquely against its corresponding data.
        params : list
            List of parameters to be optimised. Applies to all models
        loss_fn : str
            Loss function to be used for optimisation. 
            Options are 'poisson', 'diff2' and 'norm'.
        optimisers : list
            List of optax optimisers to be used for optimisation. 
            Here learning rates and schedules can be specified. 
            One optimiser per parameter. 
        jointly_fit_params : list, default: []
            List of parameters to be jointly fitted across all models. 
            Gradients are updated by averaging across the graidents 
            of each model.
            If none given, all parameters are optimised individually.
        loss_fn_kwargs : dict, default: {}
            Additional keyword arguments to be passed to the loss function.

        """
        models = list(models)
        data = list(data)
        optimisers = list(optimisers)
        jointly_fit_params = list(jointly_fit_params)
        assert len(models) == len(data), "Must have same number of models as frames of data. \
            Got {} models and {} frames of data.".format(len(models), len(data))
        assert len(params) > 0, "Must have at least one parameter to optimise."
        assert len(params) == len(optimisers), "Must have same number of optimisers as params."
        if len(jointly_fit_params) > 0:
            assert set(list(jointly_fit_params)).issubset(set(list(params))), \
                "Jointly fit parameters must be a subset of the parameters to optimise."

        self.models = models
        self.data = data
        self.params = params
        self.jointly_fit_params = jointly_fit_params
        self.optimisers = optimisers
        self.loss_fn_kwargs = loss_fn_kwargs
        self.loss_fn = self.get_loss_fn(loss_fn)

        self.param_update_dict = dict.fromkeys(params)
        for key in self.param_update_dict:
            self.param_update_dict[key] = []
        self.param_update_dict["net loss"] = []

    @property
    def get_models(self):
        """
            Returns the list of models being optimised in their current state.
            Models are BaseOpticalSystem's or Instruments.
        """
        return self.models

    def loss_and_grads(self):
        """
            Returns the net loss and individual model gradients as described by the loss_fn.
            Gradients of jointly fit parameters will be averaged across all models to 
            produce a single set of gradients for those parameters.
            Returns:
            --------
            net_loss : float
                The total loss across all models.
            grads : list
                List of gradients for each model. Gradients take same type as model
        """
        loss_grads = [self.loss_fn(model = self.models[k], data = self.data[k]) for k in range(len(self.models))]
        individual_loss = jnp.asarray([loss_grads[k][0] for k in range(len(self.models))])
        net_loss = jnp.sum(individual_loss)
        grads = [loss_grads[k][1] for k in range(len(self.models))]

        for joint_param_str in self.jointly_fit_params:
            all_grads = [g.get(joint_param_str) for g in grads]
            joint_grad = jnp.asarray(all_grads).mean(axis=0)

            grads = [g.set(joint_param_str,joint_grad) for g in grads]

        return individual_loss, net_loss, grads

    def update_models(self, grads, optim, opt_state):
        """
            Updates the models using the gradients and optimiser state.

            Parameters:
            ----------
            grads : list
                List of gradients for each model. Gradients take same type as model.
            optim : optax.GradientTransformation
                The optimiser to be used for updating the models.
            opt_state : optax.OptState
                The state of the optimiser, used to update the models.
        """
        updates_n_opt_state = [optim.update(grads[k], opt_state) for k in range(len(self.models))]
        updates = [updates_n_opt_state[k][0] for k in range(len(self.models))]
        new_opt_state= [updates_n_opt_state[k][1] for k in range(len(self.models))][0] # assumes same across models (frames)
        

        updated_models = [zdx.apply_updates(self.models[k], updates[k]) for k in range(len(self.models))]
        self.models = updated_models
        
        return updated_models, new_opt_state

    def run_optimisation(self, num_steps: int = 1000):
        """
            Runs the optimisation loop for the specified number of steps.

            Parameters:
            ----------
            num_steps : int, default: 1000
                The number of optimisation steps to run.
        """
        optim, opt_state = zdx.get_optimiser(self.models[0], self.params, self.optimisers)
        progress_bar = tqdm(range(num_steps), desc='Loss: ')
        for j in progress_bar:
            # Calculate loss and gradients
            individual_losses, net_loss, grads = self.loss_and_grads()

            # Update models
            _, opt_state = self.update_models(grads, optim, opt_state);

            # store parameters of interest
            self.store_params(net_loss,individual_losses);
    
            progress_bar.set_description(f'Loss: {net_loss:.4f}')

    def store_params(self, net_loss, individual_loss):
        """
            Store optimised parameters and their updates in a dictionary.
            Parameters:
            ----------
            net_loss : float
                The total loss across all models, to be stored in the dictionary.
        """
        for key in self.params:
                self.param_update_dict[key].append([model.get(key) for model in self.models])
        self.param_update_dict["net loss"].append(net_loss)
        self.param_update_dict["final losses"] = individual_loss # will keep updating and save the last update

        return self.param_update_dict
    
    def save_stored_params(self, filename: str):
        """
            Saves the stored parameters to a file.
            Parameters:
            ----------
            filename : str
                The name of the file to save the parameters to.
        """
        
        with open(filename, 'wb') as f:
            pickle.dump(self.param_update_dict, f)
        print(f"Stored parameters saved to {filename}")

    def calc_CRLB(self, param: str):
        """
            Calculates the Cramer-Rao Lower Bound (CRLB) for the given parameters.
            This is calculated for each model.

            Returns:
            --------
            crlb: Array
                Shape (n_models, n_params)
        """
        assert param in self.params, "Parameter {} not in params {}".format(param, self.params)

        CRLB = [jnp.linalg.pinv(zdx.fisher_matrix(self.models[k],
                            param,
                            self.loss_fn, 
                            self.data[k])[0]).diagonal()**0.5 
                            for k in range(len(self.models))]
        
        return jnp.asarray(CRLB)    

    def get_loss_fn(self, str_name: str):
        """
            Returns a loss function which is decorated with zodiax's filter_jit and filter_value_and_grad
            and returns loss, gradient.
            Inputs are model and data, where model is a dLux BaseOpticalSystem or Instrument. 
        """
        if str_name=='poisson':
            @zdx.filter_jit
            @zdx.filter_value_and_grad(self.params)
            def loss_fn(model, data):
                simu_psf = model.model()
                poiss = -jax_0_4_24_logpmf(k=data, mu=simu_psf)
                # loss = jnp.sum(poiss)
                loss = jnp.sum(jnp.where(data<=256, 0, poiss))
                return loss
        elif str_name=='diff2':
            @zdx.filter_jit
            @zdx.filter_value_and_grad(self.params)
            def loss_fn(model, data):
                simu_psf = model.model()
                loss = jnp.nansum((data - simu_psf) ** 2)
                return loss
        elif str_name=='norm':
            @zdx.filter_jit
            @zdx.filter_value_and_grad(self.params)
            def loss_fn(model, data, stdev):
                simu_psf = model.model()
                # loss = -jsp.stats.norm.logpdf(x=data, loc=simu_psf, scale=stdev).sum()
                loss = -jsp.stats.norm.logpdf(x=simu_psf, loc=data, scale=stdev).sum()
                return loss
        elif str_name=='diff2_ph_reg':
            pupil_phase = self.loss_fn_kwargs.get('pupil_phase')
            pupil_mask = self.loss_fn_kwargs.get('pupil_mask')
            @zdx.filter_jit
            @zdx.filter_value_and_grad(self.params)
            def loss_fn(model, data):
                opd_waves = model.aperture.eval_basis()/model.source.wavelengths[0]*pupil_mask
                simu_psf = model.model()

                loss_i = jnp.nansum((data - simu_psf) ** 2)
                loss_ph = jnp.nansum((pupil_phase - opd_waves) ** 2)

                return loss_i*loss_ph
            
        elif str_name=='chi2':
            stdev = self.loss_fn_kwargs.get('stdev')
            RN = self.loss_fn_kwargs.get('readnoise')
            @zdx.filter_jit
            @zdx.filter_value_and_grad(self.params)
            def loss_fn(model, data):
                simu_psf = model.model()
                chi2=((data - simu_psf)/jnp.sqrt(stdev**2+RN**2))**2
                valid_mask = (
                    jnp.isfinite(chi2) & 
                    (data > 256) 
                )
                loss = jnp.sum(jnp.where(valid_mask, chi2, 0))
                return loss
            
        elif str_name=='chi2_poiss':
            RN = self.loss_fn_kwargs.get('readnoise')
            mask = self.loss_fn_kwargs.get('data_mask')
            @zdx.filter_jit
            @zdx.filter_value_and_grad(self.params)
            def loss_fn(model, data):
                simu_psf = model.model()
                # poiss_err = simu_psf**0.5
                # loss = jnp.nansum(( (data - simu_psf)/(poiss_err**2+RN**2) )**2)
                loss = jnp.nansum(( (data - simu_psf)/(RN) )**2 * mask)

                return loss

        else: 
            Warning("No valid loss function given." \
            "available are 'poisson', 'diff2' and 'norm' ")

        return loss_fn

class JointOptimManager():
    """
        JointOptimManager has-a list of OptimManager's.
        Useful for jointly fitting across models of different configurations.
    """

    OptimManagers: list[OptimManager]
    jointly_fit_params: list

    param_update_dict:list[dict]

    def __init__(self, 
                 optim_managers: list,
                 jointly_fit_params: list,
                 ):
        """
        Parameters
        ----------
        optim_managers : list
            List of OptimManager's to be used for optimisation.
            Each OptimManager will be optimised co-dependently 
            with respect to the jointly_fit_params.
        jointly_fit_params : list
            List of parameters to be jointly fitted across all OptimManager's.
            Gradients are updated by averaging across the graidents 
            of each OptimManager.
        """
        
        self.OptimManagers = list(optim_managers)
        self.jointly_fit_params = list(jointly_fit_params)

        assert set(jointly_fit_params).issubset(set(optim_managers[0].params)), \
            "Jointly fit parameters must be a subset of the parameters to optimise."
        
        self.param_update_dict = [None]*len(self.OptimManagers)

    def loss_and_grads(self):
        """
            Returns the net loss and individual model gradients as described by the loss_fn.
            Gradients of jointly fit parameters will be averaged across all models to 
            produce a single set of gradients for those parameters.
            Returns:
            --------
            net_loss : float
                The total loss across all models.
            grads : list
                List of gradients for each model. Gradients take same type as model
        """
        loss_grads = [optim_manager.loss_and_grads() for optim_manager in self.OptimManagers]
        indiviudal_losses = jnp.asarray([loss_grads[k][0] for k in range(len(self.OptimManagers))])
        net_loss = np.sum([loss_grads[k][1] for k in range(len(self.OptimManagers))])
        grads = [loss_grads[k][2] for k in range(len(self.OptimManagers))]

        for joint_param_str in self.jointly_fit_params:
            all_grads = [g.get(joint_param_str) for config_grads in grads for g in config_grads]
            joint_grad = jnp.asarray(all_grads).mean(axis=0)

            for k in range(len(self.OptimManagers)):
                grads[k] = [g.set(joint_param_str,joint_grad) for g in grads[k]]

        return indiviudal_losses, net_loss, grads

    def update_models(self, grads, optim, opt_state):
        """
            Updates the models using the gradients and optimiser state.

            Parameters:
            ----------
            grads : list
                List of gradients for each OptimManager model. Gradients take same type as model.
            optim : optax.GradientTransformation
                The optimiser to be used for updating the models. Assumes that all models in the OptimManagers
                use the same optimiser.
            opt_state : optax.OptState
                The state of the optimiser, used to update the models.
        """
        updated_models_n_opt_state = [optim_manager.update_models(grads=grads[i], optim=optim[i], opt_state=opt_state[i]) for i, optim_manager in enumerate(self.OptimManagers)]
        updated_models = [updated_models_n_opt_state[k][0] for k in range(len(self.OptimManagers))]
        updated_opt_state =  [updated_models_n_opt_state[k][1] for k in range(len(self.OptimManagers))]

        return updated_models, updated_opt_state

    def run_optimisation(self, num_steps: int = 1000):
        """
            Runs the optimisation loop for the specified number of steps.

            Parameters:
            ----------
            num_steps : int, default: 1000
                The number of optimisation steps to run.
        """
        optims = [zdx.get_optimiser(self.OptimManagers[k].models[0], self.OptimManagers[k].params, self.OptimManagers[k].optimisers)[0] for k in range(len(self.OptimManagers))]
        opt_states = [zdx.get_optimiser(self.OptimManagers[k].models[0], self.OptimManagers[k].params, self.OptimManagers[k].optimisers)[1] for k in range(len(self.OptimManagers))]
        progress_bar = tqdm(range(num_steps), desc='Loss: ')
        for j in progress_bar:
            # Calculate loss and gradients
            individual_losses, net_loss, grads = self.loss_and_grads()

            # Update models
            self.models, opt_states = self.update_models(grads, optims, opt_states)

            # store parameters of interest
            self.store_params(net_loss, individual_losses);
    
            progress_bar.set_description(f'Loss: {net_loss:.4f}')

    def store_params(self, net_loss, individual_loss):
        """
            Store optimised parameters and their updates in a dictionary.
            Parameters:
            ----------
            net_loss : float
                The total loss across all models, to be stored in the dictionary.
        """
        for i, optim_manager in enumerate(self.OptimManagers):
            optim_manager.store_params(net_loss, individual_loss[i])
            self.param_update_dict[i] = optim_manager.param_update_dict

        return self.param_update_dict
    
    def save_stored_params(self, filenames: list[str]):
        """
            Saves the stored parameters to a file.
            Parameters:
            ----------
            filename : list[str]
                A name per OptimManager to save the parameters to.
        """
        assert len(filenames) == len(self.OptimManagers), "Need a filename for each OptimManager to save the parameters to."
        for i, filename in enumerate(filenames):
            self.OptimManagers[i].save_stored_params(filename)

    def calc_CRLB(self, param: str):
        """
            Calculates the Cramer-Rao Lower Bound (CRLB) for a given parameter.
            This is calculated for each OptimManager.

            Returns:
            --------
            crlb: list
                List of CRLB arrays called from OptimManager.calc_CRLB 
        """
        CRLB = [optim_manager.calc_CRLB(param) for optim_manager in self.OptimManagers]
        
        return CRLB

class PointSource(dl.sources.Source):
    """
    Identical functionality to dLux PointSource - with a key difference
    that the flux parameter is held as a float array (0 grads when calc
    with Zodiax otherwise).

    ??? abstract "UML"
        ![UML](../../assets/uml/PointSource.png)

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : Array, photons
        The flux of the object.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    position: jnp.array
    flux: jnp.array
    logflux: jnp.array
    bool_logflux: bool = False

    def __init__(
        self: dl.sources.Source,
        wavelengths: jnp.array = None,
        position: jnp.array = jnp.zeros(2),
        flux: jnp.array = None,
        logflux: jnp.array = None,
        weights: jnp.array = None,
        spectrum: dl.Spectrum = None,
    ):
        """
        Parameters
        ----------
        wavelengths : Array, metres = None
            The array of wavelengths at which the spectrum is defined. This input is
            ignored if a Spectrum object is provided.
        position : Array, radians = np.zeros(2)
            The (x, y) on-sky position of this object.
        flux : float, photons = 1.
            The flux of the object.
        logflux: float
            Log of the number of photons to prop. Convenient 
            for MCMC's and fitting processes.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        """
        # Position and Flux
        self.position = jnp.asarray(position, dtype=float)

        if logflux:
            self.bool_logflux = True

        self.logflux = jnp.asarray(logflux, dtype=float)
        self.flux = jnp.asarray(flux, dtype=float)

        if self.position.shape != (2,):
            raise ValueError("position must be a 1d array of shape (2,).")

        super().__init__(
            wavelengths=wavelengths, weights=weights, spectrum=spectrum
        )

    def model(
        self: dl.sources.Source,
        optics: dl.optical_systems.BaseOpticalSystem,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> jnp.array:
        """
        Models the source object through the provided optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source object.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """
        self = self.normalise()

        if self.bool_logflux:
            weights = self.weights * 10**self.logflux
        else:
            weights = self.weights * self.flux

        return optics.propagate(
            self.wavelengths, self.position, weights, return_wf, return_psf
        )

class TransmissiveLayer(dl.layers.TransmissiveLayer):
    """
    Base class to hold transmissive layers imbuing them with a transmission and
    normalise parameter. Same as dl.layers.TransmisssiveLayer with the exception
    of a rotation parameter which is updated on each apply().

    ??? abstract "UML"
        ![UML](../../assets/uml/TransmissiveLayer.png)

    Attributes
    ----------
    transmission: Array
        The Array of transmission values to be applied to the input wavefront.
    normalise: bool
        Whether to normalise the wavefront after passing through the optic.
    rotation: Array([float])
        Single value for rotation of transmissive layer (radians). 
        Array of shape (1,) (zodiax artefact requires array to 
        optimise on single value). Rotation applied CW
    """

    rotation: jnp.array


    def __init__(
        self: dl.layers.optical_layers.OpticalLayer,
        transmission: jnp.array = None,
        normalise: bool = False,
        rotation: jnp.array = np.array([0.0]),
        **kwargs,
    ):
        """
        Parameters
        ----------
        transmission: Array = None
            The array of transmission values to be applied to the input wavefront.
        normalise : bool = False
            Whether to normalise the wavefront after passing through the optic.
        rotation: Array([float])
            Single value for rotation of transmissive layer (radians). 
            Array of shape (1,) (zodiax artefact requires array to 
            optimise on single value). Rotation applied CW
        """
        self.rotation = rotation
        super().__init__(transmission=transmission, normalise=normalise,**kwargs)

    def apply(self: dl.layers.optical_layers.OpticalLayer, wavefront: dl.wavefronts.Wavefront) -> dl.wavefronts.Wavefront:
        """
        Applies the layer to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The transformed wavefront.
        """
        wavefront *= dlu.rotate(self.transmission, self.rotation) 
        if self.normalise:
            wavefront = wavefront.normalise()
        return wavefront

# Global useful functions
def rotation_matrix(theta: float):
    """
        Parameters:
        ----------
        theta: float
            Angle of rotation in radians
        Returns:
        --------
        jnp.array:
            2D rotation matrix (CCW) for given angle
    """
    return jnp.array([[jnp.cos(theta), jnp.sin(theta)], [-jnp.sin(theta), jnp.cos(theta)]])

class BasisLayer(dl.layers.BasisLayer):
    """
    Inherits from dLux BasisLayer, with identical functionality 
    but additional feature of rotation of the basis by a specified angle.

    ??? abstract "UML"
        ![UML](../../assets/uml/BasisLayer.png)

    Attributes
    ----------
    basis: Union[Array, list]
        The set of basis vectors. Should in generate be a 3 dimensional array.
    coefficients: Array
        The array of coefficients to be applied to each basis vector.
    as_phase: bool = False
        Whether to apply the basis as a phase or OPD. If True the output is applied as
        a phase, else it is applied as an OPD.
    rotation: Array([float])
        Single value for rotation of evaluated basis (radians). 
        Array of shape (1,) 
    """

    rotation: jnp.array

    def __init__(
        self: dl.layers.BasisLayer,
        basis: Array = None,
        coefficients: Array = None,
        as_phase: bool = False,
        rotation: jnp.array = np.array([0.0]),
    ):
        """
        Parameters
        ----------
        basis: Union[Array, list]
            The set of basis vectors. Should in generate be a 3 dimensional array.
        coefficients: Array
            The Array of coefficients to be applied to each basis vector.
        as_phase: bool = False
            Whether to apply the basis as a phase or OPD. If True the output is applied
            as a phase, else it is applied as an OPD.
        rotation: Array([float])
            Single value for rotation of evalutated basis (radians). 
            Array of shape (1,). Default 0.0
        """
        super().__init__(basis=basis,
                        coefficients=coefficients,
                        as_phase=as_phase,
                        )

        self.rotation = rotation    


    def eval_basis(self) -> jnp.array:
        """
        Override parent eval_basis()
        Calculates the dot product of the basis vectors and coefficients.

        Returns
        -------
        output : Array
            The output of the dot product between the basis vectors and coefficients.
        """ 
        return dlu.rotate(dlu.eval_basis(self.basis, self.coefficients), self.rotation)


class DynamicAperture(dl.layers.apertures.BaseDynamicAperture):

    """
        Inherits from dLux BaseDynamicAperture, with almost identical functionality
        to CompositeAperture with... (personal prefs, _pattern_rot, _rmaxes update etc)

        Attributes
        ----------
        transmission: Array
            The Array of transmission values to be applied to the input wavefront.
        normalise: bool
            Whether to normalise the wavefront after passing through the optic.
        _pattern_rot: float
            Rotation of aperture pattern in radians. Initially set to 0.
        apertures: dict
            Dictionary of either dLux CircularAperture or dLux AberratedAperture objects representing 
            the circular sub-apertures. 
            Each segment is defined by the center and rmax.
        _pixel_coords: jnp.array
            2D coordinates for each pixel center describing the aperture (m).
        _prim_diam: float
            Primary diameter of Jewel mask in meters
        _npix: int
            Number of pixels spanning vertically/horizontally over the primary mask diameter
    """
    n_sides = 0                     # circ
    rot = 0                         # Default rotation of segments - redudant for circular ap but leaving it here 
                                    # for completeness 
    
    sub_apertures: dict
    _pattern_rot: jnp.array
    _ap_centers: jnp.array
    _ap_noll_idxs: jnp.array


    _pixel_coords: jnp.array
    _prim_diam: float
    _npix: int
    _rmax: jnp.array
    # _rmax_param: str = "rmax" # parameter path to update in sub_apertures
    _rmax_param: str = "radius" # parameter path to update in sub_apertures



    def __init__(self: dl.layers.apertures.BaseDynamicAperture, 
                subap_centers: list, 
                rmax: jnp.array,
                pixel_coords: jnp.array,
                npix: int,
                prim_diam: float,   
                normalise: bool = False,
                pattern_rot: jnp.array = jnp.array([0.0]),
                ap_noll_idxs: jnp.array = None,
                ):
        """
        Parameters:
        ----------
            subap_centers: list
                List describing the cartesian [x,y] coordinates of __every__ segments'
                center in every tiling pattern in the entire mask. Element i contains an array
                of size (n_seg, 2) where n_seg = the # of segments in each tiling pattern
                on the Jewel mask. Units in meters.
                # NOTE I copied this class over from Jewel repo so if things are formatted 
                weird that's why.
            rmax: float
                Max radius to vertices from center of hexagonal segment, meters. If single value
                is given, same rmax applied to all segements. Else, must be of shape (n,) where
                n is the length of hex_centers.
            pixel_coords: jnp.array
                2D coordinates for each pixel center in JewelWedge (cartesian or polar)
            n_pix: int
                Number of pixels spanning vertically/horizontally over the primary mask diameter
            prim_diam: float
                Primary diameter of Jewel mask in meters
            normalise: bool
                Whether to normalise the wavefront after passing through the optic.
            pattern_rot: Array
                Rotation of shim pattern in radians. Default 0.0
            ap_noll_idxs: jnp.array = None
                1D array containing the Zernike (Noll) indices to be used to descibe the aberrations
                across each segment. If None, _sub_apertures will be a dictonary of CircularAperture objects,
                otherwise AberratedAperture objects will be used.
        """
        
        assert subap_centers[0].shape[-1] == 2, """Hexagonal centers dimensions are incorrect. Expecting list of length n_patterns with each entry
                                                an (n_segments,2) array. Got shape {}""".format(subap_centers[0].shape)
        
        list_centers = []
        for pattern in subap_centers:
            for coord in pattern:
                list_centers.append(coord)
        self._ap_centers = jnp.asarray(list_centers) # change to 2d array 

        rmax = jnp.asarray([rmax], dtype=float)
        if rmax.shape == (1,):
            self._rmax = jnp.ones((self._ap_centers.shape[0],), dtype=float)*rmax
        else:
            assert rmax.shape == (self._ap_centers.shape[0],), "rmax must be single value or array of length equal to hex_centers"
            self._rmax = jnp.asarray(rmax, dtype=float)

        self._pixel_coords=pixel_coords
        self._prim_diam=prim_diam
        self._npix = npix
        self._ap_noll_idxs = ap_noll_idxs
        sub_aperture_list = self.init_sub_apertures()
        self.sub_apertures = dlu.list2dictionary(sub_aperture_list, True, dl.layers.apertures.ApertureLayer) 

        self._pattern_rot = jnp.asarray(pattern_rot)

        super().__init__()

    def __getattr__(self: dl.layers.apertures.ApertureLayer, key: str):
        """
        Raises the contained apertures via their dictionary keys.

        Parameters
        ----------
        key: str
            The attribute to get.

        Returns
        -------
        attribute: Any
            The aperture found at the given key.
        """
        sub_apertures = self.__dict__.get("sub_apertures", None)
        if sub_apertures is not None and key in list(self.sub_apertures.keys()):
            return self.sub_apertures[key]
        else:
            raise AttributeError(key)
        
    def init_sub_apertures(self): 
        """
            Parameters:
            ----------
            ap_noll_idxs: jnp.array
                1D array containing the Zernike (Noll) indices to be used to descibe the basis
                at each segment. If None, RegularPolygon objects will be used to describe the
                hexagonal segments.
            Returns:
            --------
            sub_apertures: list
                List of dLux RegularPolygon or AberratedAperture objects representing the hexagonal segments
                in the shim pattern. Each segment is defined by the center and rmax.
        """
        sub_apertures = [] #vmap this?
        for i, cen_tf in enumerate(self._ap_centers):
            # tf = dl.CoordTransform(translation=(cen_tf[0], cen_tf[1]), rotation=0.0) # rotation changes basis axes (not just envelope)
            tf = dl.CoordTransform(translation=(cen_tf[0], cen_tf[1]), rotation=self.rot) # rotation changes basis axes (not just envelope)
            
            # sub_apertures.append(dl.RegPolyAperture(nsides=self.n_sides, rmax=self._rmax[i], transformation=tf)) # to extend for polygons
            sub_apertures.append(dl.CircularAperture(radius=self._rmax[i], transformation=tf))


        if self._ap_noll_idxs is not None:
            # self._rmax_param = "aperture.rmax" # valid for RegPolyAperture not CircularAperture
            self._rmax_param = "aperture.radius"
            sub_apertures = [dl.AberratedAperture(aperture, self._ap_noll_idxs) for aperture in sub_apertures] 

        return sub_apertures

    def _aberrated_apertures(self: dl.layers.apertures.ApertureLayer) -> list[dl.layers.apertures.ApertureLayer]:
        """
        Returns the list of aberrated apertures, from the full set of apertures, along with their 
        corresponding transmission arrays

        Returns
        -------
        apertures : list
            The list of aberrated apertures.
        """

        apertures = self.update_radii()
        # to rotate basis (envelope only and NOT basis axis) we need to rotate the centers of the hexagons
        rotated_centers = jnp.asarray([jnp.matmul(cen, rotation_matrix(self._pattern_rot)) for cen in  self._ap_centers])
        aberrated_aps = []
        for i, aper in enumerate(apertures.keys()): 
            tf = dl.CoordTransform(translation=(rotated_centers[i,0], rotated_centers[i,1]), rotation=0.0) # no segment rotation on axis here to keep basis axis correctly aligned (circle is rotationally symm anyways)
            aberrated_aps.append(apertures[aper].set("aperture.transformation", tf)) 

        # hex_trans = [aper.transmission(self._pixel_coords, self._prim_diam/self._npix) for aper in aberrated_aps] #TODO dlu.rotate
        hex_trans = [dlu.rotate(apertures[aper].transmission(self._pixel_coords, self._prim_diam/self._npix), self._pattern_rot) for aper in apertures.keys()]
        # aberrated_aps = [aper.set("aperture.nsides",0) for aper in aberrated_aps]  # circ basis (to cut from - hexikeys unideal)

        # aberrated_aps = [apertures[aper].set("aperture.nsides",0) for aper in apertures.keys() if isinstance(apertures[aper], dl.layers.apertures.AberratedAperture)]
        # hex_trans = [apertures[aper].transmission(self._pixel_coords, self._prim_diam/self._npix) for aper in apertures.keys() if isinstance(apertures[aper], dl.layers.apertures.AberratedAperture)]

        return aberrated_aps, hex_trans

    @property
    def coefficients(self: dl.layers.apertures.ApertureLayer) -> list[Array]:
        """
        Returns the coefficients of the aberrated apertures.

        Returns
        -------
        coefficients : list[Array]
            The coefficients of the aberrated apertures.
        """
        apertures, _ = self._aberrated_apertures()
        return [ap.coefficients for ap in apertures]


    def eval_basis(self: dl.layers.apertures.ApertureLayer) -> Array:
        """
        Calculates the basis vectors at the given coordinates.

        Parameters
        ----------
        coords : Array
            The coordinates to calculate the basis vectors on.

        Returns
        -------
        basis : Array
            The basis vectors at the given coordinates.
        """
        aberrated_apertures, aperture_trans = self._aberrated_apertures()
        basis_fn = lambda ap: ap.calc_basis(self._pixel_coords) #default to circ
        basii = [basis_fn(aperture) for aperture in aberrated_apertures]
        coeffs = self.coefficients
        eval_fn = lambda basis, coeff: dlu.eval_basis(basis, coeff)
        basis= jnp.array([eval_fn(basis, coeff)*trans for basis, coeff, trans in zip(basii, coeffs, aperture_trans)]).sum(axis=0)   

        return basis #TODO basis envelope should rotate but not the basis itself


    def transmission(self): # pragma: no cover
        apertures = self.update_radii()

        eval_fn = lambda ap: ap.transmission(self._pixel_coords, self._prim_diam/self._npix)
        leaf_fn = lambda ap: isinstance(ap, dl.layers.apertures.ApertureLayer)
        transmissions = tree_map(eval_fn, apertures, is_leaf=leaf_fn)
        return dlu.rotate(np.squeeze(np.array(tree_flatten(transmissions)[0])).sum(axis=0), self._pattern_rot)

    def update_radii(self):
        """
            Updates the radii of the apertures and returns new apertures dict.
        """
        # Update radii
        # we don't do through each aperturees own function because can't optimise on floats (only arr of floats)
        # and single vector to descibe all segements would be nice here.
        return {key: self.sub_apertures[key].set(self._rmax_param, rmax) for key, rmax in zip(self.sub_apertures.keys(), self._rmax)}
    

    def apply(self: dl.layers.apertures.ApertureLayer, wavefront: dl.wavefronts.Wavefront) -> dl.wavefronts.Wavefront:
        """
        Applies the layer to the wavefront.

        Parameters
        ----------
        wavefront : Wavefront
            The wavefront to operate on.

        Returns
        -------
        wavefront : Wavefront
            The transformed wavefront.
        """

        # Transmission
        wavefront *= self.transmission()

        if self.normalise:
            return wavefront.normalise()

        if self._hex_noll_idxs is not None:
            # Aberrations
            aberrations = self.eval_basis() 
            wavefront += aberrations #no option to apply as phase

        return wavefront
    
class ParametricOpticalSystem(dl.optical_systems.OpticalSystem):
    """
    Implements the attributes required for an optical system with a specific output
    pixel scale and number of pixels.

    NOTE - adding to overcome zodiax float incompatibility 

    Attributes
    ----------
    psf_npixels : int
        The number of pixels of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    psf_pixel_scale : float
        The pixel scale of the final PSF.
    """

    psf_npixels: int
    oversample: int
    psf_pixel_scale: jnp.array

    def __init__(
        self: dl.optical_systems.OpticalSystem,
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        psf_npixels : int
            The number of pixels of the final PSF.
        psf_pixel_scale : float
            The pixel scale of the final PSF.
        oversample : int = 1.
            The oversampling factor of the final PSF. Decreases the psf_pixel_scale
            parameter while increasing the psf_npixels parameter.
        """
        self.psf_npixels = int(psf_npixels)
        self.oversample = int(oversample)
        self.psf_pixel_scale = jnp.array(psf_pixel_scale, dtype=float)
        super().__init__(**kwargs)

class AngularOpticalSystem(ParametricOpticalSystem, dl.optical_systems.LayeredOpticalSystem):
    """
    Implements the attributes required for an optical system with a specific output
    pixel scale and number of pixels.

    NOTE - adding to overcome zodiax float incompatibility 

    Attributes
    ----------
    psf_npixels : int
        The number of pixels of the final PSF.
    oversample : int
        The oversampling factor of the final PSF. Decreases the psf_pixel_scale
        parameter while increasing the psf_npixels parameter.
    psf_pixel_scale : float
        The pixel scale of the final PSF.
    """

    def __init__(
        self: dl.optical_systems.OpticalSystem,
        wf_npixels: int,
        diameter: float,
        layers: list,
        psf_npixels: int,
        psf_pixel_scale: float,
        oversample: int = 1,
    ):
        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            layers=layers,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )

    def propagate_mono(
        self: dl.optical_systems.OpticalSystem,
        wavelength: Array,
        offset: Array = jnp.zeros(2),
        return_wf: bool = False,
    ) -> Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        wf = super().propagate_mono(wavelength, offset, return_wf=True)

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = dlu.arcsec2rad(true_pixel_scale)
        psf_npixels = self.psf_npixels * self.oversample
        wf = wf.propagate(psf_npixels, pixel_scale)

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf