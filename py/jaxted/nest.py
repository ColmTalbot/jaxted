from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax_tqdm import scan_tqdm

from .utils import insertion_index_test, logsubexp, likelihood_insertion_index
from .nssmc import initialize, mutate

__all__ = [
    "digest",
    "null_func",
    "outer_step",
    "replace_func",
    "run_nest",
]


def run_nest(
    likelihood_fn,
    ln_prior_fn,
    sample_prior,
    *,
    boundary_fn=None,
    sub_iterations=10,
    nlive=1000,
    nsteps=500,
    rseed=20,
    dlogz=0.1,
):
    state = initialize(likelihood_fn, sample_prior, nlive, rseed)

    @scan_tqdm(sub_iterations, print_rate=1)
    def body_func(state, level):
        return outer_step(
            state,
            likelihood_fn=likelihood_fn,
            ln_prior_fn=ln_prior_fn,
            boundary_fn=boundary_fn,
            nsteps=nsteps,
        )

    state, output = jax.lax.scan(body_func, state, jnp.arange(sub_iterations))
    _, ln_normalization, ln_evidence, _, _, ln_likelihoods = state
    dlogz_ = jnp.log1p(ln_normalization + jnp.max(ln_likelihoods) - ln_evidence)
    output = {key: output[key].reshape((nlive * sub_iterations,) + output[key].shape[2:]) for key in output}
    while dlogz_ > dlogz:
        print(
            f"dlogz = {dlogz_:.2f} > {dlogz:.2f} running again ({ln_evidence:.2f}, {ln_normalization:.2f})"
        )
        state, new_output = jax.lax.scan(body_func, state, jnp.arange(sub_iterations))
        _, ln_normalization, ln_evidence, _, _, ln_likelihoods = state
        dlogz_ = jnp.log1p(ln_normalization + jnp.max(ln_likelihoods) - ln_evidence)
        output = {
            key: jnp.concatenate([output[key], new_output[key].reshape((nlive * sub_iterations,) + output[key].shape[2:])])
            for key in output
        }

    rng_key, ln_normalization, _, _, samples, ln_likelihoods = state
    ln_post_weights = ln_normalization + ln_likelihoods - jnp.log(nlive)

    insertion_indices = output.pop("insertion_index")
    pvalue = insertion_index_test(insertion_indices, nlive)
    print(f"Likelihood insertion test p-value: {pvalue:.4f}")

    output["ln_weights"] = jnp.concatenate([output["ln_weights"], ln_post_weights])
    output["ln_likelihood"] = jnp.concatenate([output["ln_likelihood"], ln_likelihoods])
    for key in samples:
        output[key] = jnp.concatenate([output[key], samples[key]])

    ln_weights = output["ln_weights"]
    ln_evidence = logsumexp(ln_weights)
    ln_variance = logsumexp(2 * ln_weights)
    variance = jnp.exp(ln_variance - 2 * ln_evidence)
    ln_evidence_err = variance**0.5

    ln_weights -= jnp.max(ln_weights)
    keep = ln_weights > jnp.log(jax.random.uniform(rng_key, ln_weights.shape))
    output = {key: values[keep] for key, values in output.items()}

    return ln_evidence, ln_evidence_err, output


def null_func(args):
    state, _ = args
    _, _, _, _, samples, _ = state
    output = {key: samples[key][0] for key in samples}
    output["ln_likelihood"] = jnp.nan
    output["ln_weights"] = -jnp.inf
    output["insertion_index"] = -1
    return state, output


def replace_func(args):
    state, proposed = args
    rng_key, ln_normalization, ln_evidence, ln_variance, samples, ln_likelihoods = state
    level = jnp.min(ln_likelihoods)
    replace = jnp.argmin(ln_likelihoods)

    ln_compression = -1 / len(ln_likelihoods)
    ln_post_weight = ln_normalization + level + logsubexp(0, ln_compression)
    ln_evidence = jnp.logaddexp(ln_evidence, ln_post_weight)
    ln_variance = jnp.logaddexp(2 * ln_post_weight, ln_variance)
    ln_normalization += ln_compression

    output = {key: samples[key][replace] for key in samples}
    output["ln_likelihood"] = level
    output["ln_weights"] = ln_post_weight

    ln_l = proposed.pop("ln_likelihood")
    for key in samples:
        samples[key] = samples[key].at[replace].set(proposed[key])
    output["insertion_index"] = likelihood_insertion_index(ln_likelihoods, ln_l)
    ln_likelihoods = ln_likelihoods.at[replace].set(ln_l)

    state = (
        rng_key,
        ln_normalization,
        ln_evidence,
        ln_variance,
        samples,
        ln_likelihoods,
    )
    return state, output


def digest(state, proposed):
    ln_l = proposed["ln_likelihood"]
    _, _, _, _, _, ln_likelihoods = state
    level = jnp.min(ln_likelihoods)
    return jax.lax.cond(ln_l > level, replace_func, null_func, (state, proposed))


@partial(
    jax.jit, static_argnames=("likelihood_fn", "ln_prior_fn", "boundary_fn", "nsteps")
)
def outer_step(state, likelihood_fn, ln_prior_fn, boundary_fn, nsteps):
    rng_key, _, _, _, samples, ln_likelihoods = state
    proposal_points = {key: samples[key].copy() for key in samples}
    level = jnp.min(ln_likelihoods)
    rng_key, new_samples, new_ln_likelihoods, total_accepted = mutate(
        rng_key,
        samples,
        ln_likelihoods,
        proposal_points,
        level,
        likelihood_fn=likelihood_fn,
        ln_prior_fn=ln_prior_fn,
        boundary_fn=boundary_fn,
        nsteps=nsteps,
    )
    new_samples["ln_likelihood"] = new_ln_likelihoods
    return jax.lax.scan(
        digest,
        state,
        new_samples,
    )
