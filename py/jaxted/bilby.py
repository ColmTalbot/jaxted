import os
from functools import partial

import jax
import numpy as np
import pandas as pd
from bilby.core.sampler.base_sampler import Sampler
from bilby.core.result import Result
from bilby.compat.jax import generic_bilby_likelihood_function

from .nest import run_nest
from .smc import run_nssmc_anssmc
from .utils import apply_boundary, generic_bilby_ln_prior

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["BILBY_ARRAY_API"] = "1"
os.environ["SCIPY_ARRAY_API"] = "1"

__all__ = ["Jaxted"]


class Jaxted(Sampler):
    """
    bilby wrapper of :code:`JAXted`
    (https://jaxted.readthedocs.io/en/latest/)

    Parameters
    ==========
    likelihood: bilby.core.likelihood.Likelihood
        A Bilby likelihood object
    priors: bilby.core.prior.PriorDict, dict
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    outdir: str, optional
        Name of the output directory
    label: str, optional
        Naming scheme of the output files
    use_ratio: bool, optional
        Switch to set whether or not you want to use the log-likelihood ratio
        or just the log-likelihood
    skip_import_verification: bool
        Skips the check if the sampler is installed if true. This is
        only advisable for testing environments
    method: str
        The sampling method, should be one of :code:`nest` or :code:`smc`
    nsteps: int
        The number of steps to take in each ensemble MCMC chain
    nlive: int
        The size of the live population
    rseed: int
        The random seed
    alpha: float
        The compression fraction for the :code:`smc` method
    """

    sampler_name = "jaxted"
    sampling_seed_key = "rseed"
    default_kwargs = dict(
        method="nest", nsteps=500, nlive=500, rseed=1, alpha=np.exp(-1)
    )

    def _time_likelihood(self, n_evaluations=100):
        return np.nan

    def run_sampler(self):
        likelihood_fn, ln_prior_fn, sample_fn, boundary_fn = jaxted_inputs_from_bilby(
            self.likelihood, self.priors, use_ratio=self.use_ratio
        )

        method = self.kwargs.pop("method", "nest")
        if method == "nest":
            sampler = run_nest
            self.kwargs.pop("alpha")
        elif method == "smc":
            sampler = run_nssmc_anssmc
        else:
            raise ValueError("Unknown sampling method")
        ln_z, ln_zerr, samples = sampler(
            likelihood_fn=likelihood_fn,
            ln_prior_fn=ln_prior_fn,
            sample_prior=sample_fn,
            boundary_fn=boundary_fn,
            **self.kwargs,
        )
        self.result = self.create_result(samples, ln_z, ln_zerr)
        self.kwargs["method"] = method
        return self.result

    def create_result(self, samples, ln_z, ln_zerr):
        posterior_samples = pd.DataFrame(samples)
        return Result(
            label=self.label,
            outdir=self.outdir,
            posterior=posterior_samples,
            log_evidence=float(ln_z),
            search_parameter_keys=self.search_parameter_keys,
            priors=self.priors,
            log_noise_evidence=self.likelihood.noise_log_likelihood(),
            log_evidence_err=ln_zerr,
            log_bayes_factor=float(ln_z) - self.likelihood.noise_log_likelihood(),
            injection_parameters=self.injection_parameters,
            meta_data=self.meta_data,
        )

    def _setup_pool(self):
        """
        In addition to the usual steps, we need to set the sampling kwargs on
        every process. To make sure we get every process, run the kwarg setting
        more times than we have processes.
        """
        self.kwargs["npool"] = 1
        super()._setup_pool()


def jaxted_inputs_from_bilby(likelihood, priors, use_ratio=False):
    likelihood_fn = jax.vmap(
        partial(
            generic_bilby_likelihood_function,
            likelihood,
            use_ratio=use_ratio,
        )
    )
    boundary_fn = partial(apply_boundary, priors=priors)
    ln_prior_fn = partial(generic_bilby_ln_prior, priors=priors)
    sample_fn = priors.sample
    return likelihood_fn, ln_prior_fn, sample_fn, boundary_fn
