from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

__all__ = ["initialize", "step", "mutate", "new_step"]


def initialize(likelihood_fn, sample_prior, nlive, rseed):
    rng_key = jax.random.PRNGKey(rseed)
    samples = sample_prior(nlive)
    ln_likelihoods = likelihood_fn(samples)

    ln_normalization = 0.0
    ln_evidence = -np.inf
    ln_variance = -np.inf

    return rng_key, ln_normalization, ln_evidence, ln_variance, samples, ln_likelihoods


@partial(jax.jit, static_argnames=("likelihood_fn", "ln_prior_fn", "boundary_fn"))
def step(
    rng_key,
    samples,
    proposal_points,
    ln_likelihoods,
    level,
    likelihood_fn,
    ln_prior_fn,
    boundary_fn,
):
    sequential_weights = ln_likelihoods > level

    old_priors = ln_prior_fn(samples)
    rng_key, subkey_1, subkey_2, subkey_3 = jax.random.split(rng_key, 4)
    if ln_likelihoods.size == 1:
        key = list(proposal_points.keys())[0]
        prop_idxs = jax.random.choice(
            subkey_1, len(proposal_points[key]), (2,), replace=False
        )
        deltas = jax.random.uniform(subkey_2)
    else:
        prop_idxs = jax.random.choice(
            subkey_1,
            len(sequential_weights),
            (2, len(ln_likelihoods)),
            p=sequential_weights / sequential_weights.sum(),
        )
        deltas = jax.random.uniform(subkey_2, (len(ln_likelihoods),))
    proposed = dict()

    for key in samples:
        proposed[key] = jnp.atleast_1d(
            samples[key]
            + deltas
            * (proposal_points[key][prop_idxs[0]] - proposal_points[key][prop_idxs[1]])
        )

    proposed = boundary_fn(proposed)
    proposed_ln_likelihoods = likelihood_fn(proposed)
    proposed_priors = ln_prior_fn(proposed) + jnp.log((proposed_ln_likelihoods > level))
    mh_ratio = proposed_priors - old_priors
    accept = mh_ratio > jnp.log(jax.random.uniform(subkey_3, mh_ratio.shape))
    for key in samples:
        samples[key] = jnp.where(accept, proposed[key], samples[key])
    ln_likelihoods = jnp.where(accept, proposed_ln_likelihoods, ln_likelihoods)
    return rng_key, samples, ln_likelihoods, accept


@partial(
    jax.jit, static_argnames=("likelihood_fn", "ln_prior_fn", "boundary_fn", "nsteps")
)
def mutate(
    rng_key,
    samples,
    ln_likelihoods,
    proposal_points,
    level,
    likelihood_fn,
    ln_prior_fn,
    boundary_fn,
    nsteps=500,
):
    total_accepted = jnp.zeros(ln_likelihoods.shape)
    (rng_key, samples, _, ln_likelihoods, _), accepted = jax.lax.scan(
        partial(
            new_step,
            likelihood_fn=likelihood_fn,
            ln_prior_fn=ln_prior_fn,
            boundary_fn=boundary_fn,
        ),
        (rng_key, samples, proposal_points, ln_likelihoods, level),
        length=nsteps,
    )
    total_accepted = accepted.sum()
    return rng_key, samples, ln_likelihoods, total_accepted


@partial(jax.jit, static_argnames=("likelihood_fn", "ln_prior_fn", "boundary_fn"))
def new_step(state, x, likelihood_fn, ln_prior_fn, boundary_fn):
    _, _, proposal_points, _, level = state
    rng_key, samples, ln_likelihoods, accept = step(
        *state,
        likelihood_fn=likelihood_fn,
        ln_prior_fn=ln_prior_fn,
        boundary_fn=boundary_fn,
    )
    return (rng_key, samples, proposal_points, ln_likelihoods, level), accept.mean()
