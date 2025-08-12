"""
    Set of classes to streamline phase retrieval of the toliman plates
"""

import dLux as dl
import dLux.utils as dlu
import zodiax as zdx

import jax.numpy as jnp
import numpy as np

from tqdm.notebook import tqdm

import pickle

# version of jax.scipy.stats.logppmf where k can be non-discrete (helpful if data is not discrete photon counts)
from jax.scipy.special import xlogy, gammaln
import jax.scipy as jsp
from jax._src.lax.lax import _const as _lax_const
from jax import lax
from jax._src.numpy.util import promote_args_inexact

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

    def __init__(
            self,
            models: list,
            data: list,
            params: list,
            loss_fn: str,
            optimisers: list,
            jointly_fit_params: list = [],
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
        updated_models = [zdx.apply_updates(self.models[k], optim.update(grads[k], opt_state)[0]) for k in range(len(self.models))]
        self.models = updated_models

        return updated_models

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
            _, net_loss, grads = self.loss_and_grads()

            # Update models
            self.update_models(grads, optim, opt_state);

            # store parameters of interest
            self.store_params(net_loss);
    
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
                loss = -jax_0_4_24_logpmf(k=data, mu=simu_psf).sum()
                return loss
        elif str_name=='diff2':
            @zdx.filter_jit
            @zdx.filter_value_and_grad(self.params)
            def loss_fn(model, data):
                simu_psf = model.model()
                loss = ((data-simu_psf)**2).sum()
                return loss
        elif str_name=='norm':
            @zdx.filter_jit
            @zdx.filter_value_and_grad(self.params)
            def loss_fn(model, data, stdev):
                simu_psf = model.model()
                # loss = -jsp.stats.norm.logpdf(x=data, loc=simu_psf, scale=stdev).sum()
                loss = -jsp.stats.norm.logpdf(x=simu_psf, loc=data, scale=stdev).sum()
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
        updated_models = [optim_manager.update_models(grads=grads[i], optim=optim, opt_state=opt_state) for i, optim_manager in enumerate(self.OptimManagers)]

        return updated_models

    def run_optimisation(self, num_steps: int = 1000):
        """
            Runs the optimisation loop for the specified number of steps.

            Parameters:
            ----------
            num_steps : int, default: 1000
                The number of optimisation steps to run.
        """
        optim, opt_state = zdx.get_optimiser(self.OptimManagers[0].models[0], self.OptimManagers[0].params, self.OptimManagers[0].optimisers)
        progress_bar = tqdm(range(num_steps), desc='Loss: ')
        for j in progress_bar:
            # Calculate loss and gradients
            individual_losses, net_loss, grads = self.loss_and_grads()

            # Update models
            self.models = self.update_models(grads, optim, opt_state)

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

    def __init__(
        self: dl.sources.Source,
        wavelengths: jnp.array = None,
        position: jnp.array = jnp.zeros(2),
        flux: jnp.array = None,
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
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        """
        # Position and Flux
        self.position = jnp.asarray(position, dtype=float)
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
