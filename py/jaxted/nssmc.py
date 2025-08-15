from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

__all__ = ["initialize", "differential_evolution", "mutate", "new_step"]


def initialize(likelihood_fn, sample_prior, transform, nlive, rseed, **args):
    rng_key = jax.random.PRNGKey(rseed)
    samples = sample_prior(nlive, rng_key=rng_key, **args)

    if transform is not None:
        transformed = transform(samples)
    else:
        transformed = samples

    ln_likelihoods = likelihood_fn(transformed, **args)

    ln_normalization = 0.0
    ln_evidence = -np.inf
    ln_variance = -np.inf

    return rng_key, ln_normalization, ln_evidence, ln_variance, samples, ln_likelihoods


@partial(
    jax.jit,
    static_argnames=("likelihood_fn", "ln_prior_fn", "boundary_fn", "transform"),
)
def uniform(
    rng_key,
    samples,
    proposal_points,
    ln_likelihoods,
    level,
    likelihood_fn,
    ln_prior_fn,
    boundary_fn,
    transform,
    **args,
):
    old_priors = ln_prior_fn(samples, **args)

    rng_key, subkey, *sample_keys = jax.random.split(rng_key, len(samples) + 2)

    proposed = {
        key: jax.random.uniform(sample_keys[ii], ln_likelihoods.shape).astype(values)
        for ii, (key, values) in enumerate(samples.items())
    }

    if transform is not None:
        transformed = transform(proposed)
    else:
        transformed = proposed

    proposed_ln_likelihoods = likelihood_fn(transformed, **args)
    proposed_priors = ln_prior_fn(proposed, **args) + jnp.log(
        proposed_ln_likelihoods > level
    )

    mh_ratio = proposed_priors - old_priors
    accept = mh_ratio > jnp.log(jax.random.uniform(subkey, mh_ratio.shape))

    for key in samples:
        samples[key] = jax.vmap(jnp.where)(accept, proposed[key], samples[key])
    ln_likelihoods = jnp.where(
        accept, proposed_ln_likelihoods, level * jnp.ones(ln_likelihoods.shape)
    )

    return rng_key, proposed, ln_likelihoods, accept


@partial(
    jax.jit,
    static_argnames=("likelihood_fn", "ln_prior_fn", "boundary_fn", "transform"),
)
def differential_evolution(
    rng_key,
    samples,
    proposal_points,
    ln_likelihoods,
    level,
    likelihood_fn,
    ln_prior_fn,
    boundary_fn,
    transform,
    **args,
):
    valid_points = ln_likelihoods > level
    old_priors = ln_prior_fn(samples, **args)

    rng_key, subkey_1, subkey_2, subkey_3, subkey_4 = jax.random.split(rng_key, 5)

    prop_idxs = jax.vmap(jax.random.choice, in_axes=(0, None, None, None, None))(
        jax.random.split(subkey_1, len(ln_likelihoods)),
        len(valid_points),
        (2,),
        False,
        valid_points / valid_points.sum(),
    ).T
    scale = 2.38 / (2 * len(samples)) ** 0.5 / 4
    deltas = jnp.where(
        jax.random.choice(subkey_3, 2, (len(ln_likelihoods),)).astype(bool),
        jax.random.gamma(subkey_2, 4, shape=(len(ln_likelihoods),)) * scale,
        1.0,
    )

    proposed = dict()
    for key in samples:
        diffs = proposal_points[key][prop_idxs[0]] - proposal_points[key][prop_idxs[1]]
        proposed[key] = jax.vmap(lambda sample, delta, diff: sample + delta * diff)(
            samples[key], deltas, diffs
        ).astype(samples[key])

    if boundary_fn is not None:
        proposed = boundary_fn(proposed)

    if transform is not None:
        transformed = transform(proposed)
    else:
        transformed = proposed

    proposed_ln_likelihoods = likelihood_fn(transformed, **args)
    proposed_priors = ln_prior_fn(proposed, **args) + jnp.log(
        proposed_ln_likelihoods > level
    )

    mh_ratio = proposed_priors - old_priors
    accept = mh_ratio > jnp.log(jax.random.uniform(subkey_4, mh_ratio.shape))

    for key in samples:
        samples[key] = jax.vmap(jnp.where)(accept, proposed[key], samples[key])
    ln_likelihoods = jnp.where(accept, proposed_ln_likelihoods, ln_likelihoods)

    return rng_key, samples, ln_likelihoods, accept


@partial(
    jax.jit,
    static_argnames=(
        "likelihood_fn",
        "ln_prior_fn",
        "boundary_fn",
        "transform",
        "proposal",
    ),
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
    transform,
    proposal=differential_evolution,
    nsteps=500,
    **args,
):
    body_fn = partial(
        new_step,
        likelihood_fn=likelihood_fn,
        ln_prior_fn=ln_prior_fn,
        boundary_fn=boundary_fn,
        transform=transform,
        step_fn=proposal,
        **args,
    )

    state = (
        rng_key,
        samples,
        proposal_points,
        ln_likelihoods,
        level,
        jnp.zeros(ln_likelihoods.shape),
    )
    (rng_key, samples, _, ln_likelihoods, _, total_accepted) = jax.lax.fori_loop(
        jnp.array(0),
        nsteps,
        body_fn,
        state,
    )
    return rng_key, samples, ln_likelihoods, total_accepted


@partial(
    jax.jit,
    static_argnames=(
        "likelihood_fn",
        "ln_prior_fn",
        "boundary_fn",
        "transform",
        "step_fn",
    ),
)
def new_step(
    ii,
    state,
    likelihood_fn,
    ln_prior_fn,
    boundary_fn,
    transform,
    step_fn=differential_evolution,
    **args,
):
    _, _, proposal_points, _, level, n_accept = state
    rng_key, samples, ln_likelihoods, accept = step_fn(
        *state[:-1],
        likelihood_fn=likelihood_fn,
        ln_prior_fn=ln_prior_fn,
        boundary_fn=boundary_fn,
        transform=transform,
        **args,
    )
    n_accept += accept
    return (rng_key, samples, proposal_points, ln_likelihoods, level, n_accept)
