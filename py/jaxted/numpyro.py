import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.util import log_likelihood, trace, substitute, soft_vmap, Predictive


def log_prior(model, posterior_samples, *args, parallel=False, batch_ndims=1, **kwargs):
    """
    Returns log prior at observation nodes of model, given samples of all latent variables.
    This is a minimally edited version of `numpyro.infer.util.log_likelihood`.

    :param model: Python callable containing Pyro primitives.
    :param dict posterior_samples: dictionary of samples from the posterior.
    :param args: model arguments.
    :param batch_ndims: the number of batch dimensions in posterior samples. Some usages:

        + set `batch_ndims=0` to get log likelihoods for 1 single sample

        + set `batch_ndims=1` to get log likelihoods for `posterior_samples`
          with shapes `(num_samples x ...)`

        + set `batch_ndims=2` to get log likelihoods for `posterior_samples`
          with shapes `(num_chains x num_samples x ...)`

    :param kwargs: model kwargs.
    :return: dict of log likelihoods at observation sites.
    """

    def single_loglik(samples):
        substituted_model = (
            substitute(model, samples) if isinstance(samples, dict) else model
        )
        model_trace = trace(substituted_model).get_trace(*args, **kwargs)
        return {
            name: site["fn"].log_prob(site["value"])
            for name, site in model_trace.items()
            if site["type"] == "sample" and not site["is_observed"]
        }

    prototype_site = batch_shape = None
    for name, sample in posterior_samples.items():
        if batch_shape is not None and jnp.shape(sample)[:batch_ndims] != batch_shape:
            raise ValueError(
                f"Batch shapes at site {name} and {prototype_site} "
                f"should be the same, but got "
                f"{sample.shape[:batch_ndims]} and {batch_shape}"
            )
        else:
            prototype_site = name
            batch_shape = jnp.shape(sample)[:batch_ndims]

    if batch_shape is None:  # posterior_samples is an empty dict
        batch_shape = (1,) * batch_ndims
        posterior_samples = np.zeros(batch_shape)

    batch_size = int(np.prod(batch_shape))
    chunk_size = batch_size if parallel else 1
    return soft_vmap(single_loglik, posterior_samples, len(batch_shape), chunk_size)


def sample_prior(num_samples, model, rng_key, *args, **kwargs):
    pred = Predictive(model, num_samples=num_samples)
    return pred(jax.random.PRNGKey(np.random.randint(100000)), *args, **kwargs)


def jaxted_inputs_from_numpyro(model, *args, **kwargs):
    """
    Convert numpyro model to jaxted inputs.

    :param model: Python callable containing Pyro primitives.
    :param args: model arguments.
    :param kwargs: model kwargs.
    :return: tuple of log likelihood, log prior, sample prior, and boundary function.
    """

    def likelihood_fn(x, *new_args, **new_kwargs):
        lnls = sum(
            log_likelihood(model, x, *new_args, *args, **new_kwargs, **kwargs).values()
        )
        while lnls.ndim > 1:
            lnls = lnls.sum(axis=-1)
        return lnls

    def ln_prior_fn(x, *new_args, **new_kwargs):
        lnps = sum(
            log_prior(model, x, *new_args, *args, **new_kwargs, **kwargs).values()
        )
        while lnps.ndim > 1:
            lnps = lnps.sum(axis=-1)
        return lnps

    def sample_fn(n, *new_args, **new_kwargs):
        vals = sample_prior(n, model, *new_args, *args, **new_kwargs, **kwargs)
        if "ln_l" in vals:
            del vals["ln_l"]
        return vals

    boundary_fn = None

    return likelihood_fn, ln_prior_fn, sample_fn, boundary_fn
