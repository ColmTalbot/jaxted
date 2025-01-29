"""
A collection of useful functions.
"""

import typing
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import randint, ks_1samp
from jax_tqdm import PBar
from jax_tqdm.pbar import build_tqdm

__all__ = [
    "apply_boundary",
    "distance_insertion_index",
    "generic_bilby_ln_prior",
    "insertion_index_test",
    "likelihood_insertion_index",
    "logsubexp",
    "while_tqdm",
]


@jax.jit
def logsubexp(aa, bb):
    r"""
    .. math:

        \log\left( e^{a} - e^{b} \right)

    Parameters
    ==========
    aa: array-like
    bb: array-like

    Returns
    =======
    array-like

    Notes
    =====
    This function does not include safety checks that :math:`a > b`.
    """
    return aa + jnp.log(1 - jnp.exp(bb - aa))


@partial(jax.jit, static_argnames=("priors",))
def generic_bilby_ln_prior(samples, priors):
    """
    Wrapper to function to evaluate the log prior density from a :code:`Bilby`
    prior dictionary.

    Parameters
    ==========
    samples: dict[str, array-like]
    priors: bilby.core.prior.PriorDict

    Returns
    =======
    array-like
    """
    return jnp.log(priors.prob(samples, axis=0))


@partial(jax.jit, static_argnames=("priors",))
def apply_boundary(samples, priors):
    """
    Apply periodic boundary conditions to the input samples using the provided
    prior dictionary.

    Parameters
    ==========
    samples: dict[str, array-like]
    priors: bilby.core.prior.PriorDict

    Returns
    =======
    dict[str, array-like]

    Notes
    =====
    This modifies the input samples in place.
    """
    for key in samples:
        if priors[key].boundary == "periodic":
            samples[key] -= priors[key].minimum
            samples[key] %= priors[key].width
            samples[key] += priors[key].minimum
    return samples


def while_tqdm(
    print_rate: typing.Optional[int] = None,
    tqdm_type: str = "auto",
    **kwargs,
) -> typing.Callable:
    """
    tqdm progress bar for a JAX fori_loop

    Parameters
    ----------
    n: int
        Number of iterations.
    print_rate: int
        Optional integer rate at which the progress bar will be updated,
        by default the print rate will 1/20th of the total number of steps.
    tqdm_type: str
        Type of progress-bar, should be one of "auto", "std", or "notebook".
    **kwargs
        Extra keyword arguments to pass to tqdm.

    Returns
    -------
    typing.Callable:
        Progress bar wrapping function.
    """

    kwargs["desc"] = kwargs.get("desc", "Running while loop")
    update_progress_bar, close_tqdm = build_tqdm(-100, print_rate, tqdm_type, **kwargs)

    def _while_tqdm(func):
        """
        Decorator that adds a tqdm progress bar to `cond_fun`
        used in `jax.lax.while_loop`.
        """

        def wrapper_progress_bar(val):
            if isinstance(val, PBar):
                bar_id = val.id
                val = val.carry
                i = val[-1]
                i, val = update_progress_bar((i, val), i, bar_id=bar_id)
                result = func(val)
                output = PBar(id=bar_id, carry=result)
            else:
                i = val[-1]
                bar_id = 0
                i, val = update_progress_bar((i, val), i)
                result = func(val)
                output = result
            i = jax.lax.select(result, i, -101)
            return close_tqdm(output, i, bar_id=bar_id)

        return wrapper_progress_bar

    return _while_tqdm


def insertion_index_test(result, kind="likelihood", ax=None):
    """
    Compute the p-value comparing the distribution of insertion indices with
    the discrete uniform distribution as described in arxiv:2006.03371.
    Parameters
    ----------
    result: dynesty.utils.Results
        The result of a NS analysis
    kind: str
        The name of the quantity for which to test the insertion indices.
        The allowed values are:
        - likelihood
        - distance
    ax: matplotlib.Axis
        If passed, the insertion indices will be histogramed on the axis.
    Returns
    -------
    pval: float, array-like
        The p value(s) comparing the insertion indices to the discrete uniform
        distribution
        If analyzing a dynamic NS run, one p value is returned for each
        distinct number of live points, typically two.
    """

    def compute_pvalue(_vals, _nlive):
        dist = randint(1, _nlive + 1)
        return ks_1samp(_vals, dist.cdf).pvalue

    key = f"{kind}_insertion_index"
    vals = np.array(result[key])
    select = vals >= 0

    if sum(select) == 0:
        return np.nan

    vals = vals[select]
    if "batch_nlive" in result:
        pvals = list()
        nlives = np.array(result["batch_nlive"])[result["samples_batch"]]
        nlives = nlives[select]
        for nlive in np.unique(result["batch_nlive"]):
            vals_ = vals[nlives == nlive]
            pval = compute_pvalue(vals_, nlive)
            pvals.append(pval)
            label = f"{kind.title()}: $p_{{\\rm value }}={pval:.2f}, n_{{\\rm live}}={nlive}$"
            if ax is not None:
                ax.hist(
                    vals_ / nlive, bins=30, density=True, histtype="step", label=label
                )
        return pvals
    else:
        nlive = result["nlive"]
        pval = compute_pvalue(vals, result["nlive"])
        label = (
            f"{kind.title()}: $p_{{\\rm value }}={pval:.2f}, n_{{\\rm live}}={nlive}$"
        )
        if ax is not None:
            ax.hist(vals / nlive, bins=30, density=True, histtype="step", label=label)
        return pval


@jax.jit
def distance_insertion_index(live_u, start, point):
    """
    Compute the distance insertion index as defined in XXX
    """
    norms = jnp.std(live_u, axis=0)
    distance = jnp.linalg.norm((point - start) / norms)
    all_distances = jnp.array([jnp.linalg.norm((start - u) / norms) for u in live_u])
    return jnp.sum(all_distances < distance)


@jax.jit
def likelihood_insertion_index(live_logl, logl):
    """
    Compute the likelihood insertion index as defined in arxiv:2006.03371
    """
    return jnp.sum(jnp.array(live_logl) < logl)
