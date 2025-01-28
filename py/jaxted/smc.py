from functools import partial

import numpy as np
import jax
import pandas as pd
from jax_tqdm import scan_tqdm

from .nssmc import initialize, mutate
from .utils import logsubexp, while_tqdm

__all__ = [
    "anssmc", "nssmc", "smc_step", "resample", "run_nssmc_anssmc"
]


def anssmc(
    likelihood_fn,
    ln_prior_fn,
    sample_prior,
    *,
    boundary_fn=None,
    alpha=np.exp(-1),
    max_iterations=100,
    population_size=1000,
    rseed=10,
    verbose=False,
    nsteps=500,
):

    @while_tqdm()
    def cond_func(state):
        _, ln_normalization, ln_evidence, ln_variance, _, ln_likelihoods, _, iteration = state
        level = jax.numpy.quantile(ln_likelihoods, 1 - alpha, method="lower")
        ln_posterior_weights = (
            ln_normalization
            + ln_likelihoods
            + jax.numpy.log(ln_likelihoods > level)
        )
        ln_remaining = jax.scipy.special.logsumexp(ln_posterior_weights, b=1 / len(ln_likelihoods))
        ln_remaining_squared = jax.scipy.special.logsumexp(2 * ln_posterior_weights, b=1 / len(ln_likelihoods))
        ln_evidence = jax.numpy.logaddexp(ln_evidence, ln_remaining)
        condition = ln_remaining - ln_evidence
        threshold = jax.numpy.log(1e-1)
        temp_variance = logsubexp(ln_remaining_squared, 2 * ln_remaining)
        ln_variance = jax.numpy.logaddexp(ln_variance, temp_variance)
        variance = jax.numpy.exp(ln_variance - 2 * ln_evidence)
        if verbose:
            jax.debug.print(
                "iteration: {:n}, condition {:.2f} > threshold {:.2f}, ln evidence: {:.2f} +/- {:.2f}",
                iteration,
                jax.numpy.exp(condition),
                jax.numpy.exp(threshold),
                ln_evidence,
                variance**0.5,
            )
        return (condition > threshold) & (iteration < max_iterations)

    def body_func(state):
        ln_likelihoods, levels, iteration = state[-3:]
        level = jax.numpy.quantile(ln_likelihoods, 1 - alpha, method="lower")
        levels = levels.at[iteration].set(level)
        iteration += 1

        inner_state = state[:-2]
        inner_state, _ = smc_step(
            *inner_state, level, likelihood_fn, ln_prior_fn, boundary_fn, nsteps=nsteps
        )
        return inner_state + (levels, iteration)

    levels = jax.numpy.full(max_iterations, jax.numpy.nan)
    iteration = 0
    state = initialize(likelihood_fn, sample_prior, population_size, rseed) + (levels, iteration)
    state = jax.lax.while_loop(
        cond_func,
        body_func,
        state,
    )
    _, _, ln_evidence, ln_variance, _, _, levels, _ = state
    levels = levels[jax.numpy.isfinite(levels)]
    variance = jax.numpy.exp(ln_variance - 2 * ln_evidence)
    ln_evidence_err = variance**0.5

    return ln_evidence, ln_evidence_err, levels


def nssmc(
    likelihood_fn,
    ln_prior_fn,
    sample_prior,
    *,
    boundary_fn=None,
    levels,
    population_size=1000,
    rseed=10,
    nsteps=500,
):

    @scan_tqdm(len(levels), print_rate=1)
    def body_func(state, idx):
        level = levels[idx]
        return smc_step(*state, level, likelihood_fn, ln_prior_fn, boundary_fn, nsteps=nsteps)

    state = initialize(likelihood_fn, sample_prior, population_size, rseed)
    state, output = jax.lax.scan(body_func, state, jax.numpy.arange(len(levels)))
    rng_key, ln_normalization, ln_evidence, ln_variance, samples, ln_likelihoods = state

    output = {key: output[key].flatten() for key in output}

    ln_post_weights = ln_normalization + ln_likelihoods
    for key in samples:
        output[key] = jax.numpy.concatenate([output[key], samples[key]])
    output["ln_weights"] = jax.numpy.concatenate([output["ln_weights"], ln_post_weights])
    output["ln_likelihood"] = jax.numpy.concatenate([output["ln_likelihood"], ln_likelihoods])

    ln_weights = output["ln_weights"]
    ln_weights -= jax.numpy.max(ln_weights)
    keep = ln_weights > jax.numpy.log(jax.random.uniform(rng_key, ln_weights.shape))
    output = {key: values[keep] for key, values in output.items()}

    temp_evidence = jax.scipy.special.logsumexp(ln_post_weights, b=1 / len(ln_likelihoods))
    temp_evidence_squared = jax.scipy.special.logsumexp(2 * ln_post_weights, b=1 / len(ln_likelihoods))
    ln_evidence = jax.numpy.logaddexp(ln_evidence, temp_evidence)
    temp_variance = logsubexp(temp_evidence_squared, 2 * temp_evidence)
    ln_variance = jax.numpy.logaddexp(ln_variance, temp_variance)
    variance = jax.numpy.exp(ln_variance - 2 * ln_evidence)
    ln_evidence_err = variance**0.5

    return ln_evidence, ln_evidence_err, output


@partial(jax.jit, static_argnames=("likelihood_fn", "ln_prior_fn", "boundary_fn", "nsteps"))
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
    nsteps=500,
):
    sequential_weights = ln_likelihoods > level
    ln_post_weights = (
        ln_normalization
        + ln_likelihoods
        + jax.numpy.log(ln_likelihoods <= level)
    )
    temp_evidence = jax.scipy.special.logsumexp(ln_post_weights, b=1 / len(ln_likelihoods))
    temp_evidence_squared = jax.scipy.special.logsumexp(2 * ln_post_weights, b=1 / len(ln_likelihoods))
    ln_evidence = jax.numpy.logaddexp(ln_evidence, temp_evidence)
    temp_variance = logsubexp(temp_evidence_squared, 2 * temp_evidence)
    ln_variance = jax.numpy.logaddexp(ln_variance, temp_variance)
    ln_normalization += jax.numpy.log(sequential_weights.mean())
    output = {key: samples[key].copy() for key in samples}
    output["ln_likelihood"] = ln_likelihoods
    output["ln_weights"] = ln_post_weights

    # resample the particles
    rng_key, samples, ln_likelihoods = resample(rng_key, samples, ln_likelihoods, level)

    # mutate the particles
    # run a short MCMC over the constrained prior to update the samples
    proposal_points = {key: samples[key].copy() for key in samples}
    rng_key, samples, ln_likelihoods, total_accepted = mutate(
        rng_key, samples, ln_likelihoods, proposal_points, level,
        likelihood_fn=likelihood_fn, ln_prior_fn=ln_prior_fn, boundary_fn=boundary_fn,
        nsteps=nsteps,
    )
    state = (rng_key, ln_normalization, ln_evidence, ln_variance, samples, ln_likelihoods)
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
    boundary_fn,
    *,
    verbose=False,
    population_size=1000,
    nsteps=400,
    rseed=1,
    alpha=np.exp(-1),
):
    if verbose:
        print("Starting NS-SMC, there may be some compilation delay")
    _, _, levels = anssmc(
        likelihood_fn,
        ln_prior_fn,
        sample_prior,
        boundary_fn=boundary_fn,
        population_size=population_size // 10,
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
        levels=levels,
        population_size=population_size,
        nsteps=nsteps,
        rseed=rseed,
    )

