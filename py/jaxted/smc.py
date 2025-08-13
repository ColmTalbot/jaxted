from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax_tqdm import scan_tqdm

from .nssmc import initialize, mutate
from .utils import logsubexp, while_tqdm

__all__ = ["anssmc", "nssmc", "smc_step", "resample", "run_nssmc_anssmc"]


def anssmc(
    likelihood_fn,
    ln_prior_fn,
    sample_prior,
    *,
    boundary_fn=None,
    transform=None,
    alpha=np.exp(-1),
    max_iterations=100,
    nlive=1000,
    rseed=10,
    verbose=False,
    nsteps=500,
):
    """
    Run the adaptive nested sampling via sequential Monte Carlo algorithm as
    described in algorithm 3 of [1]_.

    Parameters
    ==========
    likelihood_fn: callable
        A vectorized likelihood function that takes as input a dictionary of
        values for the parameters and returns the log-likelihood at each point.
    ln_prior_fn: callable
        A function that takes a dictionary of values for the parameters and
        returns the log-prior at each point.
    sample_prior: callable
        A function that samples from the prior for initialization.
    boundary_fn: callable, optional
        A function that takes a dictionary of values for the parameters and
        performs any appropriate boundary conditions, e.g., periodicity.
        If None, no boundary is applied.
    alpha: float, default 1 / e
        The prior compression fraction at each iteration
    max_iterations: int, default 100
        The maximum number of iterations to run.
    nlive: int, default 1000
        The number of live points to use.
    rseed: int, default 10
        The random seed to use.
    verbose: bool, default False
        Whether to print progress messages.
    nsteps: int, default 500
        The number of MCMC iterations to run during each mutation stage.

    Returns
    =======
    ln_evidence: float
        The log-evidence.
    ln_evidence_err: float
        The error on the log-evidence.
    levels: array-like
        The levels used in each iteration.

    References
    ==========
    [1]_ https://arxiv.org/abs/1805.03924
    """

    # @while_tqdm()
    def cond_func(state):
        (
            _,
            ln_normalization,
            ln_evidence,
            ln_variance,
            _,
            ln_likelihoods,
            _,
            iteration,
        ) = state
        level = jnp.quantile(ln_likelihoods, 1 - alpha, method="lower")
        ln_posterior_weights = (
            ln_normalization + ln_likelihoods + jnp.log(ln_likelihoods > level)
        )
        ln_remaining = logsumexp(ln_posterior_weights, b=1 / len(ln_likelihoods))
        ln_remaining_squared = logsumexp(
            2 * ln_posterior_weights, b=1 / len(ln_likelihoods)
        )
        ln_evidence = jnp.logaddexp(ln_evidence, ln_remaining)
        condition = ln_remaining - ln_evidence
        threshold = jnp.log(1e-1)
        temp_variance = logsubexp(ln_remaining_squared, 2 * ln_remaining)
        ln_variance = jnp.logaddexp(ln_variance, temp_variance)
        variance = jnp.exp(ln_variance - 2 * ln_evidence)
        if verbose:
            jax.debug.print(
                "iteration: {:n}, condition {:.2f} > threshold {:.2f}, ln evidence: {:.2f} +/- {:.2f}",
                iteration,
                jnp.exp(condition),
                jnp.exp(threshold),
                ln_evidence,
                variance**0.5,
            )
        return (condition > threshold) & (iteration < max_iterations)

    def body_func(state):
        ln_likelihoods, levels, iteration = state[-3:]
        level = jnp.quantile(ln_likelihoods, 1 - alpha, method="lower")
        levels = levels.at[iteration].set(level)
        iteration += 1

        inner_state = state[:-2]
        inner_state, _ = smc_step(
            *inner_state, level, likelihood_fn, ln_prior_fn, boundary_fn, transform, nsteps=nsteps
        )
        return inner_state + (levels, iteration)

    levels = jnp.full(max_iterations, jnp.nan)
    iteration = 0
    state = initialize(likelihood_fn, sample_prior, transform, nlive, rseed) + (
        levels,
        iteration,
    )
    state = jax.lax.while_loop(
        cond_func,
        body_func,
        state,
    )
    _, _, ln_evidence, ln_variance, _, _, levels, _ = state
    levels = levels[jnp.isfinite(levels)]
    variance = jnp.exp(ln_variance - 2 * ln_evidence)
    ln_evidence_err = variance**0.5

    return ln_evidence, ln_evidence_err, levels


def nssmc(
    likelihood_fn,
    ln_prior_fn,
    sample_prior,
    *,
    boundary_fn=None,
    transform,
    levels,
    nlive=1000,
    rseed=10,
    nsteps=500,
):
    """
    Run the nested sampling via sequential Monte Carlo algorithm as described
    in algorithm 2 of [1]_.

    Parameters
    ==========
    likelihood_fn: callable
        A vectorized likelihood function that takes as input a dictionary of
        values for the parameters and returns the log-likelihood at each point.
    ln_prior_fn: callable
        A function that takes a dictionary of values for the parameters and
        returns the log-prior at each point.
    sample_prior: callable
        A function that samples from the prior for initialization.
    boundary_fn: callable, optional
        A function that takes a dictionary of values for the parameters and
        performs any appropriate boundary conditions, e.g., periodicity.
    levels: array-like
        The sequence of log-likelihood levels to use.
    nlive: int, default 1000
        The number of live points to use.
    rseed: int, default 10
        The random seed to use.
    nsteps: int, default 500
        The number of MCMC iterations to run during each mutation stage.

    Returns
    =======
    ln_evidence: float
        The log-evidence.
    ln_evidence_err: float
        The error on the log-evidence.
    output: dict[str, array-like]
        The output of the sampler, containing the posterior samples and other
        information.

    References
    ==========
    [1]_ https://arxiv.org/abs/1805.03924
    """

    @scan_tqdm(len(levels), print_rate=1)
    def body_func(state, idx):
        level = levels[idx]
        return smc_step(
            *state, level, likelihood_fn, ln_prior_fn, boundary_fn, transform, nsteps=nsteps
        )

    state = initialize(likelihood_fn, sample_prior, transform, nlive, rseed)
    state, output = jax.lax.scan(body_func, state, jnp.arange(len(levels)))
    rng_key, ln_normalization, ln_evidence, ln_variance, samples, ln_likelihoods = state

    output = {key: value.reshape((-1,) + value.shape[2:]) for key, value in output.items()}

    ln_post_weights = ln_normalization + ln_likelihoods
    for key in samples:
        output[key] = jnp.concatenate([output[key], samples[key]])
    output["ln_weights"] = jnp.concatenate([output["ln_weights"], ln_post_weights])
    output["ln_likelihood"] = jnp.concatenate([output["ln_likelihood"], ln_likelihoods])

    ln_weights = output["ln_weights"]
    ln_weights -= jnp.max(ln_weights)
    keep = ln_weights > jnp.log(jax.random.uniform(rng_key, ln_weights.shape))
    output = {key: values[keep] for key, values in output.items()}

    temp_evidence = logsumexp(ln_post_weights, b=1 / len(ln_likelihoods))
    temp_evidence_squared = logsumexp(2 * ln_post_weights, b=1 / len(ln_likelihoods))
    ln_evidence = jnp.logaddexp(ln_evidence, temp_evidence)
    temp_variance = logsubexp(temp_evidence_squared, 2 * temp_evidence)
    ln_variance = jnp.logaddexp(ln_variance, temp_variance)
    variance = jnp.exp(ln_variance - 2 * ln_evidence)
    ln_evidence_err = variance**0.5

    return ln_evidence, ln_evidence_err, output


@partial(
    jax.jit, static_argnames=("likelihood_fn", "ln_prior_fn", "boundary_fn", "transform", "nsteps")
)
def smc_step(
    rng_key,
    ln_normalization,
    ln_evidence,
    ln_variance,
    samples,
    ln_likelihoods,
    level,
    likelihood_fn,
    ln_prior_fn,
    boundary_fn,
    transform,
    nsteps=500,
):
    """
    Perform a single sequential Monte Carlo step.

    This involves the following steps:

    - identify the particles that are below the threshold and remove them
      from the live population.
    - update the run state, evidence, etc., using the removed particles.
    - resample the particles above the threshold.
    - mutate the particles using MCMC.

    Parameters
    ==========
    rng_key: jax.random.PRNGKey
        The random key to use for sampling.
    ln_normalization: float
        The current value of the log-normalization.
    ln_evidence: float
        The current value of the log-evidence.
    ln_variance: float
        The current value of the log-variance.
    samples: dict[str, array-like]
        A dictionary of parameter samples.
    ln_likelihoods: array-like
        The log-likelihoods of the samples.
    level: float
        The current log-likelihood threshold.
    likelihood_fn: callable
        A vectorized likelihood function that takes as input a dictionary of
        values for the parameters and returns the log-likelihood at each point.
    ln_prior_fn: callable
        A function that takes a dictionary of values for the parameters and
        returns the log-prior at each point.
    boundary_fn: callable, optional
        A function that takes a dictionary of values for the parameters and
        performs any appropriate boundary conditions, e.g., periodicity.
    nsteps: int, default 500
        The number of MCMC iterations to run during each mutation stage.

    Returns
    =======
    state: tuple
        The updated state after performing the SMC step.
    output: dict[str, array-like]
        A dictionary containing information about the step, including the
        removed samples and their log-likelihoods.
    """
    sequential_weights = ln_likelihoods > level
    ln_post_weights = (
        ln_normalization + ln_likelihoods + jnp.log(ln_likelihoods <= level)
    )
    temp_evidence = logsumexp(ln_post_weights, b=1 / len(ln_likelihoods))
    temp_evidence_squared = logsumexp(2 * ln_post_weights, b=1 / len(ln_likelihoods))
    ln_evidence = jnp.logaddexp(ln_evidence, temp_evidence)
    temp_variance = logsubexp(temp_evidence_squared, 2 * temp_evidence)
    ln_variance = jnp.logaddexp(ln_variance, temp_variance)
    ln_normalization += jnp.log(sequential_weights.mean())
    output = {key: samples[key].copy() for key in samples}
    output["ln_likelihood"] = ln_likelihoods
    output["ln_weights"] = ln_post_weights

    # resample the particles
    rng_key, samples, ln_likelihoods = resample(rng_key, samples, ln_likelihoods, level)

    # mutate the particles
    # run a short MCMC over the constrained prior to update the samples
    proposal_points = {key: samples[key].copy() for key in samples}
    rng_key, samples, ln_likelihoods, _ = mutate(
        rng_key,
        samples,
        ln_likelihoods,
        proposal_points,
        level,
        likelihood_fn=likelihood_fn,
        ln_prior_fn=ln_prior_fn,
        boundary_fn=boundary_fn,
        transform=transform,
        nsteps=nsteps,
    )
    state = (
        rng_key,
        ln_normalization,
        ln_evidence,
        ln_variance,
        samples,
        ln_likelihoods,
    )
    return state, output


@jax.jit
def resample(rng_key, samples, ln_likelihoods, level):
    weights = ln_likelihoods > level
    rng_key, temp = jax.random.split(rng_key)
    idxs = jax.random.choice(
        temp,
        len(ln_likelihoods),
        (len(ln_likelihoods),),
        p=weights / weights.sum(),
    )
    ln_likelihoods = ln_likelihoods[idxs]
    for key in samples:
        samples[key] = samples[key][idxs]
    return rng_key, samples, ln_likelihoods


def run_nssmc_anssmc(
    likelihood_fn,
    ln_prior_fn,
    sample_prior,
    *,
    boundary_fn=None,
    transform=None,
    verbose=False,
    nlive=1000,
    nsteps=400,
    rseed=1,
    alpha=np.exp(-1),
):
    """
    Run the full adaptive nested sampling via sequential Monte Carlo algorithm
    as proposed in [1]_. This involves first running the adaptive nested
    sampling algorithm, and then using the levels found to run the
    regular nested sampling via sequential Monte Carlo algorithm.

    Parameters
    ==========
    likelihood_fn: callable
        A vectorized likelihood function that takes as input a dictionary of
        values for the parameters and returns the log-likelihood at each point.
    ln_prior_fn: callable
        A function that takes a dictionary of values for the parameters and
        returns the log-prior at each point.
    sample_prior: callable
        A function that samples from the prior for initialization.
    boundary_fn: callable, optional
        A function that takes a dictionary of values for the parameters and
        performs any appropriate boundary conditions, e.g., periodicity.
        If None, no boundary is applied.
    alpha: float, default 1 / e
        The prior compression fraction at each iteration
    nlive: int, default 1000
        The number of live points to use.
    rseed: int, default 1
        The random seed to use.
    verbose: bool, default False
        Whether to print progress messages.
    nsteps: int, default 400
        The number of MCMC iterations to run during each mutation stage.

    Returns
    =======
    ln_evidence: float
        The log-evidence.
    ln_evidence_err: float
        The error on the log-evidence.
    output: dict[str, array-like]
        The output of the sampler, containing the posterior samples and other
        information.

    References
    ==========
    [1]_ https://arxiv.org/abs/1805.03924
    """
    if verbose:
        print("Starting NS-SMC, there may be some compilation delay")
    _, _, levels = anssmc(
        likelihood_fn,
        ln_prior_fn,
        sample_prior,
        boundary_fn=boundary_fn,
        transform=transform,
        nlive=nlive // 10,
        nsteps=nsteps,
        rseed=rseed,
        alpha=alpha,
    )
    if verbose:
        print(f"Adaptive stage complete, running with {len(levels)} levels")
    return nssmc(
        likelihood_fn,
        ln_prior_fn,
        sample_prior,
        boundary_fn=boundary_fn,
        transform=transform,
        levels=levels,
        nlive=nlive,
        nsteps=nsteps,
        rseed=rseed,
    )
