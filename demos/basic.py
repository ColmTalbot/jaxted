"""
A very simple example of using jaxted.

We need to provide three functions:

- a vectorized (log-)likelihood function that takes a dictionary as input and
  returns the log-likelihood at each point. Note that this can be vectorized
  using :code:`jax.vmap`.
- a function that takes a dictionary of values for the parameters and returns
  the log-prior at each point.
- a function that samples from the prior for initialization.

We can then call :code:`run_nest` or :code:`run_nssmc_anssmc` to perform
the sampling.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jaxted.nest import run_nest
from jaxted.smc import run_nssmc_anssmc


def ln_likelihood_fn(parameters):
    """
    A vectorized likelihood function that takes as input a dictionary of
    values for the parameters.

    Parameters
    ==========
    parameters: dict[str, array-like]
        A dictionary of parameter names and array of asccoiated values

    Returns
    =======
    array-like
        The log-likelihood of the data given the parameters at each point
    """
    ln_l = 0
    for value in parameters.values():
        ln_l -= value**2 / 2
    return ln_l


def ln_prior_fn(parameters):
    ln_l = 0
    for value in parameters.values():
        ln_l -= value**2 / 2
    return ln_l


def sample_prior(n_samples):
    return {key: jnp.array(np.random.normal(0, 1, n_samples)) for key in ["a", "b"]}


for fn in [run_nest, run_nssmc_anssmc]:
    ln_z, ln_zerr, samples = fn(
        likelihood_fn=ln_likelihood_fn,
        ln_prior_fn=ln_prior_fn,
        sample_prior=sample_prior,
        boundary_fn=None,
        nsteps=20,
        nlive=1000,
    )
    print(f"ln evidence: {ln_z} +/- {ln_zerr}")

    samples = pd.DataFrame(samples)
    print(samples.describe())
