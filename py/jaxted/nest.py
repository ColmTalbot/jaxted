from functools import partial

import jax
from jax_tqdm import scan_tqdm

from .utils import logsubexp
from .nssmc import initialize, mutate

__all__ = [
    "nest",
    "null_func",
    "replace_func",
    "digest",
    "outer_step",
    "run_nest",
]


def nest(
    likelihood_fn,
    ln_prior_fn,
    sample_prior,
    *,
    boundary_fn=None,
    levels,
    population_size=1000,
    nsteps=500,
    rseed=20,
    dlogz=0.1,
):
    state = initialize(likelihood_fn, sample_prior, population_size, rseed)

    @scan_tqdm(len(levels), print_rate=1)
    def body_func(state, level):
        return outer_step(
            state,
            likelihood_fn=likelihood_fn,
            ln_prior_fn=ln_prior_fn,
            boundary_fn=boundary_fn,
            nsteps=nsteps,
        )

    state, output = jax.lax.scan(
        body_func,
        state,
        levels,
    )
    rng_key, ln_normalization, ln_evidence, ln_variance, samples, ln_likelihoods = state
    dlogz_ = jax.numpy.log1p(
        ln_normalization + jax.numpy.max(ln_likelihoods) - ln_evidence
    )
    output = {key: output[key].flatten() for key in output}
    while dlogz_ > dlogz:
        print(
            f"dlogz = {dlogz_:.2f} > {dlogz:.2f} running again ({ln_evidence:.2f}, {ln_normalization:.2f})"
        )
        state, new_output = jax.lax.scan(
            body_func,
            state,
            levels,
        )
        rng_key, ln_normalization, ln_evidence, ln_variance, samples, ln_likelihoods = (
            state
        )
        dlogz_ = jax.numpy.log1p(
            ln_normalization + jax.numpy.max(ln_likelihoods) - ln_evidence
        )
        output = {
            key: jax.numpy.concatenate([output[key], new_output[key].flatten()])
            for key in output
        }

    ln_post_weights = ln_normalization + ln_likelihoods - jax.numpy.log(population_size)
    ln_evidence = jax.numpy.logaddexp(
        ln_evidence, jax.scipy.special.logsumexp(ln_post_weights)
    )
    ln_variance = jax.numpy.logaddexp(
        ln_variance, jax.scipy.special.logsumexp(2 * ln_post_weights)
    )
    variance = jax.numpy.exp(ln_variance - 2 * ln_evidence)
    ln_evidence_err = variance**0.5

    output["ln_weights"] = jax.numpy.concatenate(
        [output["ln_weights"], ln_post_weights]
    )
    output["ln_likelihood"] = jax.numpy.concatenate(
        [output["ln_likelihood"], ln_likelihoods]
    )
    output = {key: output[key].flatten() for key in output}
    for key in samples:
        output[key] = jax.numpy.concatenate([output[key], samples[key]])

    ln_weights = output["ln_weights"]
    ln_weights -= jax.numpy.max(ln_weights)
    keep = ln_weights > jax.numpy.log(jax.random.uniform(rng_key, ln_weights.shape))
    output = {key: values[keep] for key, values in output.items()}

    return ln_evidence, ln_evidence_err, output


def null_func(args):
    state, _ = args
    _, _, _, _, samples, _ = state
    output = {key: samples[key][0] for key in samples}
    output["ln_likelihood"] = jax.numpy.nan
    output["ln_weights"] = -jax.numpy.inf
    return state, output


def replace_func(args):
    state, proposed = args
    rng_key, ln_normalization, ln_evidence, ln_variance, samples, ln_likelihoods = state
    level = jax.numpy.min(ln_likelihoods)
    replace = jax.numpy.argmin(ln_likelihoods)

    ln_compression = -1 / len(ln_likelihoods)
    ln_post_weight = ln_normalization + level + logsubexp(0, ln_compression)
    ln_evidence = jax.numpy.logaddexp(ln_evidence, ln_post_weight)
    ln_variance = jax.numpy.logaddexp(2 * ln_post_weight, ln_variance)
    ln_normalization += ln_compression

    output = {key: samples[key][replace] for key in samples}
    output["ln_likelihood"] = level
    output["ln_weights"] = ln_post_weight

    ln_l = proposed.pop("ln_likelihood")
    for key in samples:
        samples[key] = samples[key].at[replace].set(proposed[key])
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
    level = jax.numpy.min(ln_likelihoods)
    return jax.lax.cond(ln_l > level, replace_func, null_func, (state, proposed))


@partial(
    jax.jit, static_argnames=("likelihood_fn", "ln_prior_fn", "boundary_fn", "nsteps")
)
def outer_step(state, likelihood_fn, ln_prior_fn, boundary_fn, nsteps):
    rng_key, _, _, _, samples, ln_likelihoods = state
    proposal_points = {key: samples[key].copy() for key in samples}
    level = jax.numpy.min(ln_likelihoods)
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


def run_nest(
    likelihood_fn,
    ln_prior_fn,
    sample_prior,
    boundary_fn,
    *,
    population_size=1000,
    nsteps=400,
    rseed=1,
    sub_iterations=10,
):
    return nest(
        likelihood_fn,
        ln_prior_fn,
        sample_prior,
        boundary_fn=boundary_fn,
        levels=jax.numpy.arange(sub_iterations),
        population_size=population_size,
        nsteps=nsteps,
        rseed=rseed,
    )
