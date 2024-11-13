#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A collection of useful functions.

"""

import sys
import warnings
import math
import copy
import os
import shutil
import pickle as pickle_module

os.environ["SCIPY_ARRAY_API"] = "1"  # noqa  # flag for scipy backend switching
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from scipy.stats import randint, ks_1samp
from scipy._lib._array_api import array_namespace
from dynesty.utils import (
    LogLikelihood,
    DelayTimer,
    get_enlarge_bootstrap,
    _RESULTS_STRUCTURE as _RESULTS_STRUCTURE_,
    Results as Results_,
)

from ._version import __version__ as DYNESTY_VERSION

__all__ = [
    "unitcheck", "resample_equal", "mean_and_cov", "quantile", "jitter_run",
    "resample_run", "reweight_run", "unravel_run", "merge_runs", "kld_error",
    "get_enlarge_bootstrap", "LoglOutput", "LogLikelihood", "RunRecord",
    "insertion_index_test", "DelayTimer"
]

SQRTEPS = math.sqrt(float(np.finfo(np.float64).eps))


class LoglOutput:
    """
    Class that encapsulates the output of the likelihood function.
    The reason we need this wrapper is to preserve the blob associated with
    the likelihood function.

    """

    def __init__(self, v, blob_flag):
        """
        Initialize the object

        Parameters
        ----------
        v: float or tuple
            if blob_flag is true v have to be a tuple of logl and blob
            if it is False v is just logl
        blob_flag: boolean
            flag to mark whether the v has a blob or not
        """
        if blob_flag:
            self.val = v[0]
            self.blob = v[1]
        else:
            self.val = v
            self.blob = None
        self.blob_flag = blob_flag

    def __init_subclass__(cls):
        from jax.tree_util import register_pytree_node
        register_pytree_node(cls, cls.pytree_flatten, cls.pytree_unflatten)

    def pytree_flatten(self):
        return (self.val, self.blob), dict(blob_flag=self.blob_flag)

    @classmethod
    def pytree_unflatten(cls, aux_data, data):
        if aux_data['blob_flag']:
            return cls(data, aux_data["blob_flag"]), dict()
        else:
            return cls(data[0], None), dict()

    def __lt__(self, v1):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return self.val < v1

    def __gt__(self, v1):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return self.val > v1

    def __le__(self, v1):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return self.val <= v1

    def __ge__(self, v1):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return self.val >= v1

    def __eq__(self, v1):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return self.val == v1

    def __float__(self):
        """
        Comparison override, we just use .val attribute in the comparison
        """
        return self.val


class RunRecord:
    """
    This is the dictionary like class that saves the results of the nested
    run so it is basically a collection of various lists of
    quantities
    """

    def __init_subclass__(cls):
        from jax.tree_util import register_pytree_node
        register_pytree_node(cls, cls.pytree_flatten, cls.pytree_unflatten)

    def pytree_flatten(self):
        return (self.D,), dict()

    @classmethod
    def pytree_unflatten(cls, aux_data, data):
        new = cls()
        new.D = data[0]
        return new        

    def __init__(self, dynamic=False):
        """
        If dynamic is true. We initialize the class for
        a dynamic nested run
        """
        D = {}
        keys = [
            'id',  # live point labels
            'u',  # unit cube samples
            'v',  # transformed variable samples
            'logl',  # loglikelihoods of samples
            'logvol',  # expected ln(volume)
            'logwt',  # ln(weights)
            'logz',  # cumulative ln(evidence)
            'logzvar',  # cumulative error on ln(evidence)
            'h',  # cumulative information
            'nc',  # number of calls at each iteration
            'boundidx',  # index of bound dead point was drawn from
            'it',  # iteration the live (now dead) point was proposed
            'n',  # number of live points interior to dead point
            'bounditer',  # active bound at a specific iteration
            'scale',  # scale factor at each iteration
            'distance_insertion_index',
            # number of points less distant from the starting point than the last inserted point
            'likelihood_insertion_index',
            # number of points with lower likelihood than the last inserted point
            'blob'  # blobs output by the log-likelihood
        ]
        if dynamic:
            keys.extend([
                'batch',  # live point batch ID
                # these are special since their length
                # is == the number of batches
                'batch_nlive',  # number of live points added in batch
                'batch_bounds'  # loglikelihood bounds used in batch
            ])
        for k in keys:
            D[k] = jax.numpy.array([])
        self.D = D

    def append(self, newD):
        """
        append new information to the RunRecord in the form a dictionary
        i.e. run.append(dict(batch=3, niter=44))
        """
        new = RunRecord()
        for k, new_ in newD.items():
            if new_ is None:
                new_ = jax.numpy.nan
            new.D[k] = jax.numpy.append(self.D[k], new_)
        return new

    def __getitem__(self, k):
        return self.D[k]

    def __setitem__(self, k, v):
        self.D[k] = v

    def keys(self):
        return self.D.keys()


_RESULTS_STRUCTURE = _RESULTS_STRUCTURE_ + [
    ('distance_insertion_index', 'array[int]',
     "The number of live points closer to the start point than "
     "the new point", 'niter'),
    ('likelihood_insertion_index', 'array[int]',
     "The number of live points with likelihood less than "
     "the new point", 'niter'),
]


class Results(Results_):
    """
    Contains the full output of a run along with a set of helper
    functions for summarizing the output.
    The object is meant to be unchangeable record of the static or
    dynamic nested run.

    Results attributes (name, type, description, array size):
    """

    _ALLOWED = set([_[0] for _ in _RESULTS_STRUCTURE])

    def __init__(self, key_values):
        """
        Initialize the results using the list of key value pairs
        or a dictionary
        Results([('logl', [1, 2, 3]), ('samples_it',[1,2,3])])
        Results(dict(logl=[1, 2, 3], samples_it=[1,2,3]))
        """
        self._keys = []
        self._initialized = False
        if isinstance(key_values, dict):
            key_values_list = key_values.items()
        else:
            key_values_list = key_values
        for k, v in key_values_list:
            assert k not in self._keys  # ensure no duplicates
            assert k in self.__class__._ALLOWED, k
            self._keys.append(k)
            setattr(self, k, copy.copy(v))
        required_keys = ['samples_u', 'samples_id', 'logl', 'samples']
        # TODO I need to add here logz, logzerr
        # but that requires ensuring that merge_runs always computes logz
        for k in required_keys:
            if k not in self._keys:
                raise ValueError('Key %s must be provided' % k)
        if 'nlive' in self._keys:
            self._dynamic = False
        elif 'samples_n' in self._keys:
            self._dynamic = True
        else:
            raise ValueError(
                'Trying to construct results object without nlive '
                'or samples_n information')
        self._initialized = True


Results.__doc__ += '\n\n' + str('\n'.join(
    ['| ' + str(_) for _ in _RESULTS_STRUCTURE])) + '\n'


def results_substitute(results, kw_dict):
    """ This is an utility method that takes a Result object and
substituted certain keys in it. It returns a copy object!
    """
    new_list = []
    for k, w in results.items():
        if k not in kw_dict:
            new_list.append((k, w))
        else:
            new_list.append((k, kw_dict[k]))
    return Results(new_list)


class JAXGenerator:

    def __init__(self, seed):
        self.key = seed

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node(
            cls,
            cls.pytree_flatten,
            cls.pytree_unflatten,
        )

    @property
    def key(self):
        self._key, subkey = jax.random.split(self._key)
        return subkey

    @key.setter
    def key(self, value):
        self._key = value

    def uniform(self, size=()):
        if isinstance(size, int):
            size = (size,)
        return jax.random.uniform(self.key, shape=size)

    def integers(self, low, high=None, size=()):
        if isinstance(size, int):
            size = (size,)
        elif size is None:
            size = ()
        if high is None:
            high = low
            low = 0
        return jax.random.randint(self.key, minval=low, maxval=high, shape=size).squeeze()

    def beta(self, a, b, size=()):
        if isinstance(size, int):
            size = (size,)
        return jax.random.beta(self.key, a, b, shape=size)

    def exponential(self, scale=1.0, size=()):
        if isinstance(size, int):
            size = (size,)
        return jax.random.exponential(self.key, shape=size) * scale

    def random(self, size=()):
        return self.uniform(size=size)

    def standard_normal(self, size=()):
        if isinstance(size, int):
            size = (size,)
        return jax.random.normal(self.key, shape=size)

    def choice(self, a, size=(), **kwargs):
        if isinstance(size, int):
            size = (size,)
        if isinstance(a, int):
            a = jax.numpy.arange(a)
        return jax.random.choice(self.key, a, shape=size, **kwargs)

    def pytree_flatten(self):
        return (self.key,), dict()
    
    @classmethod
    def pytree_unflatten(cls, aux_data, data):
        return cls(*data, **aux_data)


def get_random_generator(seed=None):
    """
    Return a random generator (using the seed provided if available)
    """
    if isinstance(seed, (np.random.Generator, JAXGenerator)):
        return seed
    elif isinstance(seed, jax.Array):
        return JAXGenerator(seed)
    return np.random.Generator(np.random.PCG64(seed))


def get_seed_sequence(rstate, nitems):
    """
    Return the list of seeds to initialize random generators
    This is useful when distributing work across a pool
    """
    if isinstance(rstate, np.random.Generator):
        seeds = rstate.integers(0, 2**63 - 1, size=nitems)
    elif isinstance(rstate, JAXGenerator):
        if jax.config.jax_enable_x64:
            inttype = jnp.uint64
        else:
            inttype = jnp.uint32
        seeds = rstate.integers(0, 2**31 - 1, size=(nitems, 2)).astype(inttype)
    elif isinstance(rstate, jax.Array):
        seeds = jax.random.split(rstate, nitems)
    return seeds


@jax.jit
def get_neff_from_logwt(logwt):
    """
    Compute the number of effective samples from an array of unnormalized
    log-weights. We use Kish Effective Sample Size (ESS)  formula.

    Parameters
    ----------
    logwt: numpy array
        Array of unnormalized weights

    Returns
    -------
    int
        The effective number of samples
    """

    # If weights are normalized to the sum of 1,
    # the estimate is  N = 1/\sum(w_i^2)
    # if the weights are not normalized
    # N = (\sum w_i)^2 / \sum(w_i^2)
    xp = array_namespace(logwt)
    W = xp.exp(logwt - logwt.max())
    return W.sum()**2 / (W**2).sum()


@jax.jit
def unitcheck(u, nonbounded=None):
    """Check whether `u` is inside the unit cube. Given a masked array
    `nonbounded`, also allows periodic boundaries conditions to exceed
    the unit cube."""

    if nonbounded is None:
        # No periodic boundary conditions provided.
        return u.min() > 0 and u.max() < 1
    else:
        # Alternating periodic and non-periodic boundary conditions.
        unb = u[nonbounded]
        # pylint: disable=invalid-unary-operand-type
        ub = u[~nonbounded]
        return (unb.min() > 0 and unb.max() < 1 and ub.min() > -0.5
                and ub.max() < 1.5)


@jax.jit
def apply_reflect(u):
    """
    Iteratively reflect a number until it is contained in [0, 1].

    This is for priors with a reflective boundary condition, all numbers in the
    set `u = 2n +/- x` should be mapped to x.

    For the `+` case we just take `u % 1`.
    For the `-` case we take `1 - (u % 1)`.

    E.g., -0.9, 1.1, and 2.9 should all map to 0.9.

    Parameters
    ----------
    u: array-like
        The array of points to map to the unit cube

    Returns
    -------
    u: array-like
       The input array, modified in place.
    """
    xp = array_namespace(u)
    return xp.minimum(u % 2, -u % 2)


@jax.jit
def mean_and_cov(samples, weights):
    """
    Compute the weighted mean and covariance of the samples.

    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples, ndim)
        2-D array containing data samples. This ordering is equivalent to
        using `rowvar=False` in `~numpy.cov`.

    weights : `~numpy.ndarray` with shape (nsamples,)
        1-D array of sample weights.

    Returns
    -------
    mean : `~numpy.ndarray` with shape (ndim,)
        Weighted sample mean vector.

    cov : `~numpy.ndarray` with shape (ndim, ndim)
        Weighted sample covariance matrix.

    Notes
    -----
    Implements the formulae found `here <https://goo.gl/emWFLR>`_.

    """
    xp = array_namespace(samples)

    # Compute the weighted mean.
    mean = xp.average(samples, weights=weights, axis=0)

    # Compute the weighted covariance.
    dx = samples - mean
    wsum = xp.sum(weights)
    w2sum = xp.sum(weights**2)
    cov = wsum / (wsum**2 - w2sum) * xp.einsum('i,ij,ik', weights, dx, dx)

    return mean, cov


def resample_equal(samples, weights, rstate=None):
    """
    Resample a set of points from the weighted set of inputs
    such that they all have equal weight. The points are also
    randomly shuffled.

    Each input sample appears in the output array either
    `floor(weights[i] * nsamples)` or `ceil(weights[i] * nsamples)` times,
    with `floor` or `ceil` randomly selected (weighted by proximity).

    Parameters
    ----------
    samples : `~numpy.ndarray` with shape (nsamples,)
        Set of unequally weighted samples.

    weights : `~numpy.ndarray` with shape (nsamples,)
        Corresponding weight of each sample.

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.

    Returns
    -------
    equal_weight_samples : `~numpy.ndarray` with shape (nsamples,)
        New set of samples with equal weights in random order.

    Examples
    --------
    >>> x = np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
    >>> w = np.array([0.6, 0.2, 0.15, 0.05])
    >>> utils.resample_equal(x, w)
    array([[ 1.,  1.],
           [ 1.,  1.],
           [ 1.,  1.],
           [ 3.,  3.]])

    Notes
    -----
    Implements the systematic resampling method described in `Hol, Schon, and
    Gustafsson (2006) <doi:10.1109/NSSPW.2006.4378824>`_.
    """
    if rstate is None:
        rstate = get_random_generator()
    elif isinstance(rstate, jax.Array):
        rstate = JAXGenerator(rstate)

    idxs = jax.random.choice(
        rstate.key, len(weights), (len(weights),), p=weights, replace=True
    )
    return samples[idxs]


@jax.jit
def quantile(x, q, weights=None):
    """
    Compute (weighted) quantiles from an input set of samples.

    Parameters
    ----------
    x : `~numpy.ndarray` with shape (nsamps,)
        Input samples.

    q : `~numpy.ndarray` with shape (nquantiles,)
       The list of quantiles to compute from `[0., 1.]`.

    weights : `~numpy.ndarray` with shape (nsamps,), optional
        The associated weight from each sample.

    Returns
    -------
    quantiles : `~numpy.ndarray` with shape (nquantiles,)
        The weighted sample quantiles computed at `q`.

    """
    xp = array_namespace(x)

    # Initial check.
    x = xp.atleast_1d(x)
    q = xp.atleast_1d(q)

    # Quantile check.
    if xp.any(q < 0.0) or xp.any(q > 1.0):
        raise ValueError("Quantiles must be between 0. and 1.")

    if weights is None:
        # If no weights provided, this simply calls `np.percentile`.
        return xp.percentile(x, 100.0 * q)
    else:
        # If weights are provided, compute the weighted quantiles.
        weights = xp.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x).")
        idx = xp.argsort(x)  # sort samples
        sw = weights[idx]  # sort weights
        cdf = xp.cumsum(sw)[:-1]  # compute CDF
        cdf /= cdf[-1]  # normalize CDF
        cdf = xp.append(0, cdf)  # ensure proper span
        quantiles = xp.interp(q, cdf, x[idx])
        return quantiles


def _get_nsamps_samples_n(res):
    """ Helper function for calculating the number of samples

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    Returns
    -------
    nsamps: int
        The total number of samples/iterations
    samples_n: array
        Number of live points at a given iteration

    """
    if res.isdynamic():
        # Check if the number of live points explicitly changes.
        samples_n = res.samples_n
        nsamps = len(samples_n)
    else:
        # If the number of live points is constant, compute `samples_n`.
        niter = res.niter
        nlive = res.nlive
        nsamps = len(res.logvol)
        if nsamps == niter:
            samples_n = np.ones(niter, dtype=int) * nlive
        elif nsamps == (niter + nlive):
            samples_n = np.minimum(np.arange(nsamps, 0, -1), nlive)
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points.")
    return nsamps, samples_n


def _find_decrease(samples_n):
    """
    Find all instances where the number of live points is either constant
    or increasing.
    Return the mask,
    the values of nlive when nlives starts to decrease
    The ranges of decreasing nlives
    v=[3,2,1,13,13,12,23,22];
    > print(dynesty.utils._find_decrease(v))
    (array([ True, False, False,  True,  True, False,  True, False]),
    [3, 13, 23],
    [[0, 3], [4, 6], (6, 8)])

    """
    nsamps = len(samples_n)
    nlive_flag = np.zeros(nsamps, dtype=bool)
    nlive_start, bounds = [], []
    nlive_flag[1:] = np.diff(samples_n) < 0

    # For all the portions that are decreasing, find out where they start,
    # where they end, and how many live points are present at that given
    # iteration.
    ids = np.nonzero(nlive_flag)[0]
    if len(ids) > 0:
        boundl = ids[0] - 1
        last = ids[0]
        nlive_start.append(samples_n[boundl])
        for curi in ids[1:]:
            if curi == last + 1:
                last += 1
                # we are in the interval of continuisly decreasing values
                continue
            else:
                # we need to close the last interval
                bounds.append([boundl, last + 1])
                nlive_start.append(samples_n[curi - 1])
                last = curi
                boundl = curi - 1
        # we need to close the last interval
        bounds.append((boundl, last + 1))
        nlive_start = np.array(nlive_start)
    return ~nlive_flag, nlive_start, bounds


def jitter_run(res, rstate=None, approx=False):
    """
    Probes **statistical uncertainties** on a nested sampling run by
    explicitly generating a *realization* of the prior volume associated
    with each sample (dead point). Companion function to :meth:`resample_run`.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.

    approx : bool, optional
        Whether to approximate all sets of uniform order statistics by their
        associated marginals (from the Beta distribution). Default is `False`.

    Returns
    -------
    new_res : :class:`~dynesty.results.Results` instance
        A new :class:`~dynesty.results.Results` instance with corresponding
        weights based on our "jittered" prior volume realizations.

    """

    if rstate is None:
        rstate = get_random_generator()

    # Initialize evolution of live points over the course of the run.
    nsamps, samples_n = _get_nsamps_samples_n(res)
    logl = res.logl

    # Simulate the prior volume shrinkage associated with our set of "dead"
    # points. At each iteration, if the number of live points is constant or
    # increasing, our prior volume compresses by the maximum value of a set
    # of `K_i` uniformly distributed random numbers (i.e. as `Beta(K_i, 1)`).
    # If instead the number of live points is decreasing, that means we're
    # instead  sampling down a set of uniform random variables
    # (i.e. uniform order statistics).

    if approx:
        nlive_flag = np.ones(nsamps, dtype=bool)
        nlive_start, bounds = [], []
    else:
        nlive_flag, nlive_start, bounds = _find_decrease(samples_n)

    # The maximum out of a set of `K_i` uniformly distributed random variables
    # has a marginal distribution of `Beta(K_i, 1)`.
    t_arr = jnp.zeros(nsamps)
    a, b = jnp.broadcast_arrays(samples_n[nlive_flag], 1.0)
    t_arr = t_arr.at[nlive_flag].set(rstate.beta(a=a, b=b, size=a.shape))

    # If we instead are sampling the set of uniform order statistics,
    # we note that the jth largest value is marginally distributed as
    # `Beta(j, K_i-j+1)`. The full joint distribution is::
    #
    #     X_(j) / X_N = (Y_1 + ... + Y_j) / (Y_1 + ... + Y_{K+1})
    #
    # where X_(j) is the prior volume of the live point with the `j`-th
    # *best* likelihood (i.e. prior volume shrinks as likelihood increases)
    # and the `Y_i`'s are i.i.d. exponentially distributed random variables.
    nunif = len(nlive_start)
    for i in range(nunif):
        nstart = int(nlive_start[i])
        bound = bounds[i]
        sn = samples_n[bound[0]:bound[1]]
        y_arr = rstate.exponential(scale=1.0, size=nstart + 1)
        ycsum = y_arr.cumsum()
        ycsum /= ycsum[-1]
        uorder = ycsum[np.append(nstart, sn - 1)]
        rorder = uorder[1:] / uorder[:-1]
        t_arr = t_arr.at[bound[0]:bound[1]].set(rorder)

    # These are the "compression factors" at each iteration. Let's turn
    # these into associated ln(volumes).
    logvol = np.log(t_arr).cumsum()

    (saved_logwt, saved_logz, saved_logzvar,
     saved_h) = compute_integrals(logl=logl, logvol=logvol)

    # Overwrite items with our new estimates.
    substitute = {
        'logvol': logvol,
        'logwt': saved_logwt,
        'logz': saved_logz,
        'logzerr': np.sqrt(np.maximum(saved_logzvar, 0)),
        'h': saved_h
    }

    new_res = results_substitute(res, substitute)
    return new_res


@jax.jit
def compute_integrals(logl, logvol, reweight=None):
    """
    Compute weights, logzs and variances using quadratic estimator.
    Returns logwt, logz, logzvar, h

    Parameters:
    -----------
    logl: array
        array of log likelihoods
    logvol: array
        array of log volumes
    reweight: array (or None)
        (optional) reweighting array to reweight posterior
    """
    xp = array_namespace(logl)

    loglstar_pad = xp.concatenate([xp.array([-1.e300]), logl])

    # we want log(exp(logvol_i)-exp(logvol_(i+1)))
    # assuming that logvol0 = 0
    # log(exp(LV_{i})-exp(LV_{i+1})) =
    # = LV{i} + log(1-exp(LV_{i+1}-LV{i}))
    # = LV_{i+1} - (LV_{i+1} -LV_i) + log(1-exp(LV_{i+1}-LV{i}))
    dlogvol = xp.diff(logvol, prepend=0)
    logdvol = logvol - dlogvol + xp.log1p(-xp.exp(dlogvol))
    # logdvol is log(delta(volumes)) i.e. log (X_i-X_{i-1})
    logdvol2 = logdvol + math.log(0.5)
    # These are log(1/2(X_(i+1)-X_i))

    dlogvol = -xp.diff(logvol, prepend=0)
    # this are delta(log(volumes)) of the run

    # These are log((L_i+L_{i_1})*(X_i+1-X_i)/2)
    saved_logwt = xp.logaddexp(loglstar_pad[1:], loglstar_pad[:-1]) + logdvol2
    if reweight is not None:
        saved_logwt = saved_logwt + reweight
    # saved_logz = xp.logaddexp.accumulate(saved_logwt)
    saved_logz = jax.lax.scan(
        lambda x, new: (jax.numpy.logaddexp(x, new), x),
        -xp.inf,
        saved_logwt
    )[1]
    # This implements eqn 16 of Speagle2020

    logzmax = saved_logz[-1]
    # we'll need that to just normalize likelihoods to avoid overflows

    # H is defined as
    # H = 1/z int( L * ln(L) dX,X=0..1) - ln(z)
    # incomplete H can be defined as
    # H = int( L/Z * ln(L) dX,X=0..x) - z_x/Z * ln(Z)
    h_part1 = xp.cumsum(xp.nan_to_num(
        xp.exp(loglstar_pad[1:] - logzmax + logdvol2) * loglstar_pad[1:] +
        xp.exp(loglstar_pad[:-1] - logzmax + logdvol2) * loglstar_pad[:-1]
    ))
    # here we divide the likelihood by zmax to avoid to overflow
    saved_h = h_part1 - logzmax * xp.exp(saved_logz - logzmax)
    # changes in h in each step
    dh = xp.diff(saved_h, prepend=0)

    # I'm applying abs() here to avoid nans down the line
    # because partial H integrals could be negative
    saved_logzvar = xp.abs(xp.cumsum(dh * dlogvol))
    return saved_logwt, saved_logz, saved_logzvar, saved_h


@jax.jit
def progress_integration(loglstar, loglstar_new, logz, logzvar, logvol,
                         dlogvol, h):
    """
    This is the calculation of weights and logz/var estimates one step at the
    time.
    Importantly the calculation of H is somewhat different from
    compute_integrals as incomplete integrals of H() of require knowing Z

    Return logwt, logz, logzvar, h
    """
    xp = array_namespace(loglstar)
    # Compute relative contribution to results.
    logdvol = logsumexp(
        a=xp.array([logvol + dlogvol, logvol]),
        b=xp.array([0.5, -0.5]),
    )
    logwt = xp.logaddexp(loglstar_new, loglstar) + logdvol  # weight
    logz_new = xp.logaddexp(logz, logwt)  # ln(evidence)
    lzterm = (xp.exp(loglstar - logz_new + logdvol) * loglstar +
              xp.exp(loglstar_new - logz_new + logdvol) * loglstar_new)
    h_new = (lzterm + xp.nan_to_num(xp.exp(logz - logz_new) * (h + logz)) - logz_new
             )  # information
    dh = h_new - h

    logzvar_new = logzvar + dh * dlogvol
    # var[ln(evidence)] estimate
    return logwt, logz_new, logzvar_new, h_new


def resample_run(res, rstate=None, return_idx=False):
    """
    Probes **sampling uncertainties** on a nested sampling run using bootstrap
    resampling techniques to generate a *realization* of the (expected) prior
    volume(s) associated with each sample (dead point). This effectively
    splits a nested sampling run with `K` particles (live points) into a
    series of `K` "strands" (i.e. runs with a single live point) which are then
    bootstrapped to construct a new "resampled" run. Companion function to
    :meth:`jitter_run`.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.

    return_idx : bool, optional
        Whether to return the list of resampled indices used to construct
        the new run. Default is `False`.

    Returns
    -------
    new_res : :class:`~dynesty.results.Results` instance
        A new :class:`~dynesty.results.Results` instance with corresponding
        samples and weights based on our "bootstrapped" samples and
        (expected) prior volumes.

    """

    if rstate is None:
        rstate = get_random_generator()

    # Check whether the final set of live points were added to the
    # run.
    nsamps = len(res.ncall)
    if res.isdynamic():
        # Check if the number of live points explicitly changes.
        samples_n = res.samples_n
        samples_batch = res.samples_batch
        batch_bounds = res.batch_bounds
        added_final_live = True
    else:
        # If the number of live points is constant, compute `samples_n` and
        # set up the `added_final_live` flag.
        nlive = res.nlive
        niter = res.niter
        if nsamps == niter:
            samples_n = np.ones(niter, dtype=int) * nlive
            added_final_live = False
        elif nsamps == (niter + nlive):
            samples_n = np.minimum(np.arange(nsamps, 0, -1), nlive)
            added_final_live = True
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points.")
        samples_batch = np.zeros(len(samples_n), dtype=int)
        batch_bounds = np.array([(-np.inf, np.inf)])
    batch_llmin = batch_bounds[:, 0]
    # Identify unique particles that make up each strand.
    ids = np.unique(res.samples_id)

    # Split the set of strands into two groups: a "baseline" group that
    # contains points initially sampled from the prior, which gives information
    # on the evidence, and an "add-on" group, which gives additional
    # information conditioned on our baseline strands.
    base_ids = []
    addon_ids = []
    for i in ids:
        sbatch = samples_batch[res.samples_id == i]
        if np.any(batch_llmin[sbatch] == -np.inf):
            base_ids.append(i)
        else:
            addon_ids.append(i)
    nbase, nadd = len(base_ids), len(addon_ids)
    base_ids, addon_ids = np.array(base_ids), np.array(addon_ids)

    # Resample strands.
    if nbase > 0 and nadd > 0:
        live_idx = np.append(base_ids[rstate.integers(0, nbase, size=nbase)],
                             addon_ids[rstate.integers(0, nadd, size=nadd)])
    elif nbase > 0:
        live_idx = base_ids[rstate.integers(0, nbase, size=nbase)]
    elif nadd > 0:
        raise ValueError("The provided `Results` does not include any points "
                         "initially sampled from the prior!")
    else:
        raise ValueError("The provided `Results` does not appear to have "
                         "any particles!")

    # Find corresponding indices within the original run.
    samp_idx = np.arange(len(res.ncall))
    samp_idx = np.concatenate(
        [samp_idx[res.samples_id == idx] for idx in live_idx])

    # Derive new sample size.
    nsamps = len(samp_idx)

    # Sort the loglikelihoods (there will be duplicates).
    logls = res.logl[samp_idx]
    idx_sort = np.argsort(logls)
    samp_idx = samp_idx[idx_sort]
    logl = res.logl[samp_idx]

    if added_final_live:
        # Compute the effective number of live points for each sample.
        samp_n = np.zeros(nsamps, dtype=int)
        uidxs, uidxs_n = np.unique(live_idx, return_counts=True)
        for uidx, uidx_n in zip(uidxs, uidxs_n):
            sel = (res.samples_id == uidx)  # selection flag
            sbatch = samples_batch[sel][0]  # corresponding batch ID
            lower = batch_llmin[sbatch]  # lower bound
            upper = max(res.logl[sel])  # upper bound

            # Add number of live points between endpoints equal to number of
            # times the strand has been resampled.
            samp_n[(logl > lower) & (logl < upper)] += uidx_n

            # At the endpoint, divide up the final set of points into `uidx_n`
            # (roughly) equal chunks and have live points decrease across them.
            endsel = (logl == upper)
            endsel_n = np.count_nonzero(endsel)
            chunk = endsel_n / uidx_n  # define our chunk
            counters = np.array(np.arange(endsel_n) / chunk, dtype=int)
            nlive_end = counters[::-1] + 1  # decreasing number of live points
            samp_n[endsel] += nlive_end  # add live point sequence
    else:
        # If we didn't add the final set of live points, the run has a constant
        # number of live points and can simply be re-ordered.
        samp_n = samples_n[samp_idx]

    # Assign log(volume) to samples.
    logvol = np.cumsum(np.log(samp_n / (samp_n + 1.)))

    saved_logwt, saved_logz, saved_logzvar, saved_h = compute_integrals(
        logl=logl, logvol=logvol)

    # Compute sampling efficiency.
    eff = 100. * len(res.ncall[samp_idx]) / sum(res.ncall[samp_idx])

    # Copy results.
    # Overwrite items with our new estimates.
    new_res_dict = dict(niter=len(res.ncall[samp_idx]),
                        ncall=res.ncall[samp_idx],
                        eff=eff,
                        blob=res.blob[samp_idx],
                        samples=res.samples[samp_idx],
                        samples_id=res.samples_id[samp_idx],
                        samples_it=res.samples_it[samp_idx],
                        samples_u=res.samples_u[samp_idx],
                        samples_n=samp_n,
                        logwt=np.asarray(saved_logwt),
                        logl=logl,
                        logvol=logvol,
                        logz=np.asarray(saved_logz),
                        logzerr=np.sqrt(
                            np.maximum(np.asarray(saved_logzvar), 0)),
                        information=np.asarray(saved_h))
    new_res = Results(new_res_dict)

    if return_idx:
        return new_res, samp_idx
    else:
        return new_res


def reweight_run(res, logp_new, logp_old=None):
    """
    Reweight a given run based on a new target distribution.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    logp_new : `~numpy.ndarray` with shape (nsamps,)
        New target distribution evaluated at the location of the samples.

    logp_old : `~numpy.ndarray` with shape (nsamps,)
        Old target distribution evaluated at the location of the samples.
        If not provided, the `logl` values from `res` will be used.

    Returns
    -------
    new_res : :class:`~dynesty.results.Results` instance
        A new :class:`~dynesty.results.Results` instance with corresponding
        weights based on our reweighted samples.

    """

    # Extract info.
    if logp_old is None:
        logp_old = res['logl']
    logrwt = logp_new - logp_old  # ln(reweight)
    logvol = res['logvol']
    logl = res['logl']

    saved_logwt, saved_logz, saved_logzvar, saved_h = compute_integrals(
        logl=logl, logvol=logvol, reweight=logrwt)

    # Overwrite items with our new estimates.
    substitute = {
        'logvol': logvol,
        'logwt': saved_logwt,
        'logz': saved_logz,
        'logzerr': np.sqrt(np.maximum(saved_logzvar, 0)),
        'h': saved_h
    }

    new_res = results_substitute(res, substitute)
    return new_res


def unravel_run(res, print_progress=True):
    """
    Unravels a run with `K` live points into `K` "strands" (a nested sampling
    run with only 1 live point). **WARNING: the anciliary quantities provided
    with each unraveled "strand" are only valid if the point was initialized
    from the prior.**

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance taken from a previous
        nested sampling run.

    print_progress : bool, optional
        Whether to output the current progress to `~sys.stderr`.
        Default is `True`.

    Returns
    -------
    new_res : list of :class:`~dynesty.results.Results` instances
        A list of new :class:`~dynesty.results.Results` instances
        for each individual strand.

    """

    idxs = res.samples_id  # label for each live/dead point

    # Check if we added in the last set of dead points.
    added_live = True
    try:
        if len(idxs) != (res.niter + res.nlive):
            added_live = False
    except AttributeError:
        pass

    if (np.diff(res.logl) == 0).sum() == 0:
        warnings.warn('The likelihood seem to have plateaus. '
                      'The unraveling such runs may be inaccurate')

    # Recreate the nested sampling run for each strand.
    new_res = []
    nstrands = len(np.unique(idxs))
    for counter, idx in enumerate(np.unique(idxs)):
        # Select strand `idx`.
        strand = (idxs == idx)
        nsamps = sum(strand)
        logl = res.logl[strand]

        # Assign log(volume) to samples. With K=1 live point, the expected
        # shrinking in `logvol` at each iteration is `-log(2)` (i.e.
        # shrinking by 1/2). If the final set of live points were added,
        # the expected value of the final live point is a uniform
        # sample and so has an expected value of half the volume
        # of the final dead point.
        if added_live:
            niter = nsamps - 1
            logvol_dead = -math.log(2) * (1. + np.arange(niter))
            if niter > 0:
                logvol_live = logvol_dead[-1] + math.log(0.5)
                logvol = np.append(logvol_dead, logvol_live)
            else:  # point always live
                logvol = np.array([math.log(0.5)])
        else:
            niter = nsamps
            logvol = -math.log(2) * (1. + np.arange(niter))

        saved_logwt, saved_logz, saved_logzvar, saved_h = compute_integrals(
            logl=logl, logvol=logvol)

        # Compute sampling efficiency.
        eff = 100. * nsamps / sum(res.ncall[strand])

        # Save results.
        rdict = dict(nlive=1,
                     niter=niter,
                     ncall=res.ncall[strand],
                     eff=eff,
                     samples=res.samples[strand],
                     samples_id=res.samples_id[strand],
                     samples_it=res.samples_it[strand],
                     samples_u=res.samples_u[strand],
                     blob=res.blob[strand],
                     logwt=saved_logwt,
                     logl=logl,
                     logvol=logvol,
                     logz=saved_logz,
                     logzerr=np.sqrt(saved_logzvar),
                     information=saved_h)

        # Add on batch information (if available).
        try:
            rdict['samples_batch'] = res.samples_batch[strand]
            rdict['batch_bounds'] = res.batch_bounds
        except AttributeError:
            pass

        # Append to list of strands.
        new_res.append(Results(rdict))

        # Print progress.
        if print_progress:
            sys.stderr.write(f'\rStrand: {counter+1}/{nstrands}     ')

    return new_res


def merge_runs(res_list, print_progress=True):
    """
    Merges a set of runs with differing (possibly variable) numbers of
    live points into one run.

    Parameters
    ----------
    res_list : list of :class:`~dynesty.results.Results` instances
        A list of :class:`~dynesty.results.Results` instances returned from
        previous runs.

    print_progress : bool, optional
        Whether to output the current progress to `~sys.stderr`.
        Default is `True`.

    Returns
    -------
    combined_res : :class:`~dynesty.results.Results` instance
        The :class:`~dynesty.results.Results` instance for the combined run.

    """

    ntot = len(res_list)
    counter = 0

    # Establish our set of baseline runs and "add-on" runs.
    rlist_base = []
    rlist_add = []
    for r in res_list:
        try:
            if np.any(r.samples_batch == 0):
                rlist_base.append(r)
            else:
                rlist_add.append(r)
        except AttributeError:
            rlist_base.append(r)
    nbase, nadd = len(rlist_base), len(rlist_add)
    if nbase == 1 and nadd == 1:
        rlist_base = res_list
        rlist_add = []

    # Merge baseline runs while there are > 2 remaining results.
    if len(rlist_base) > 1:
        while len(rlist_base) > 2:
            rlist_new = []
            nruns = len(rlist_base)
            i = 0
            while i < nruns:
                try:
                    # Ignore posterior quantities while merging the runs.
                    r1, r2 = rlist_base[i], rlist_base[i + 1]
                    res = _merge_two(r1, r2, compute_aux=False)
                    rlist_new.append(res)
                except IndexError:
                    # Append the odd run to the new list.
                    rlist_new.append(rlist_base[i])
                i += 2
                counter += 1
                # Print progress.
                if print_progress:
                    sys.stderr.write(f'\rMerge: {counter}/{ntot}     ')
            # Overwrite baseline set of results with merged results.
            rlist_base = copy.copy(rlist_new)

        # Compute posterior quantities after merging the final baseline runs.
        res = _merge_two(rlist_base[0], rlist_base[1], compute_aux=True)
    else:
        res = rlist_base[0]

    # Iteratively merge any remaining "add-on" results.
    nruns = len(rlist_add)
    for i, r in enumerate(rlist_add):
        if i < nruns - 1:
            res = _merge_two(res, r, compute_aux=False)
        else:
            res = _merge_two(res, r, compute_aux=True)
        counter += 1
        # Print progress.
        if print_progress:
            sys.stderr.write(f'\rMerge: {counter}/{ntot}     ')

    res = check_result_static(res)

    return res


def check_result_static(res):
    """ If the run was from a dynamic run but had constant
    number of live points, return a new Results object with
    nlive parameter, so we could use it as static run
    """
    samples_n = _get_nsamps_samples_n(res)[1]
    nlive = max(samples_n)
    niter = res.niter
    standard_run = False

    # Check if we have a constant number of live points.
    if samples_n.size == niter and np.all(samples_n == nlive):
        standard_run = True

    # Check if we have a constant number of live points where we have
    # recycled the final set of live points.
    nlive_test = np.minimum(np.arange(niter, 0, -1), nlive)
    if samples_n.size == niter and np.all(samples_n == nlive_test):
        standard_run = True
    # If the number of live points is consistent with a standard nested
    # sampling run, slightly modify the format to keep with previous usage.
    if standard_run:
        resdict = res.asdict()
        resdict['nlive'] = nlive
        resdict['niter'] = niter - nlive
        res = Results(resdict)
    return res


def kld_error(res,
              error='jitter',
              rstate=None,
              return_new=False,
              approx=False):
    """
    Computes the `Kullback-Leibler (KL) divergence
    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`_ *from* the
    discrete probability distribution defined by `res` *to* the discrete
    probability distribution defined by a **realization** of `res`.

    Parameters
    ----------
    res : :class:`~dynesty.results.Results` instance
        :class:`~dynesty.results.Results` instance for the distribution we
        are computing the KL divergence *from*.

    error : {`'jitter'`, `'resample'`}, optional
        The error method employed, corresponding to :meth:`jitter_run` or
        :meth:`resample_run`. Default is `'jitter'`.

    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.

    return_new : bool, optional
        Whether to return the realization of the run used to compute the
        KL divergence. Default is `False`.

    approx : bool, optional
        Whether to approximate all sets of uniform order statistics by their
        associated marginals (from the Beta distribution). Default is `False`.

    Returns
    -------
    kld : `~numpy.ndarray` with shape (nsamps,)
        The cumulative KL divergence defined *from* `res` *to* a
        random realization of `res`.

    new_res : :class:`~dynesty.results.Results` instance, optional
        The :class:`~dynesty.results.Results` instance corresponding to
        the random realization we computed the KL divergence *to*.

    """

    # Define our original importance weights.
    logp2 = res.logwt - res.logz[-1]

    # Compute a random realization of our run.
    if error == 'jitter':
        new_res = jitter_run(res, rstate=rstate, approx=approx)
    elif error == 'resample':
        new_res, samp_idx = resample_run(res, rstate=rstate, return_idx=True)
        logp2 = logp2[samp_idx]  # re-order our original results to match
    else:
        raise ValueError("Input `'error'` option '{error}' is not valid.")

    # Define our new importance weights.
    logp1 = new_res['logwt'] - new_res['logz'][-1]

    # Compute the KL divergence.
    kld = np.cumsum(np.exp(logp1) * (logp1 - logp2))

    if return_new:
        return kld, new_res
    else:
        return kld


def _merge_two(res1, res2, compute_aux=False):
    """
    Internal method used to merges two runs with differing (possibly variable)
    numbers of live points into one run.

    Parameters
    ----------
    res1 : :class:`~dynesty.results.Results` instance
        The "base" nested sampling run.

    res2 : :class:`~dynesty.results.Results` instance
        The "new" nested sampling run.

    compute_aux : bool, optional
        Whether to compute auxiliary quantities (evidences, etc.) associated
        with a given run. **WARNING: these are only valid if `res1` or `res2`
        was initialized from the prior *and* their sampling bounds overlap.**
        Default is `False`.

    Returns
    -------
    res : :class:`~dynesty.results.Results` instances
        :class:`~dynesty.results.Results` instance from the newly combined
        nested sampling run.

    """

    # Initialize the first ("base") run.
    base_info = dict(id=res1.samples_id,
                     u=res1.samples_u,
                     v=res1.samples,
                     logl=res1.logl,
                     nc=res1.ncall,
                     it=res1.samples_it,
                     blob=res1.blob)
    nbase = len(base_info['id'])

    # Number of live points throughout the run.
    if res1.isdynamic():
        base_n = res1.samples_n
    else:
        niter, nlive = res1.niter, res1.nlive
        if nbase == niter:
            base_n = np.ones(niter, dtype=int) * nlive
        elif nbase == (niter + nlive):
            base_n = np.minimum(np.arange(nbase, 0, -1), nlive)
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points in `res1`.")

    # Batch information (if available).
    # note we also check for existance of batch_bounds
    # because unravel_run makes 'static' runs of 1 livepoint
    # but some will have bounds
    if res1.isdynamic() or 'batch_bounds' in res1.keys():
        base_info['batch'] = res1.samples_batch
        base_info['bounds'] = res1.batch_bounds
    else:
        base_info['batch'] = np.zeros(nbase, dtype=int)
        base_info['bounds'] = np.array([(-np.inf, np.inf)])

    # Initialize the second ("new") run.
    new_info = dict(id=res2.samples_id,
                    u=res2.samples_u,
                    v=res2.samples,
                    logl=res2.logl,
                    nc=res2.ncall,
                    it=res2.samples_it,
                    blob=res2.blob)
    nnew = len(new_info['id'])

    # Number of live points throughout the run.
    if res2.isdynamic():
        new_n = res2.samples_n
    else:
        niter, nlive = res2.niter, res2.nlive
        if nnew == niter:
            new_n = np.ones(niter, dtype=int) * nlive
        elif nnew == (niter + nlive):
            new_n = np.minimum(np.arange(nnew, 0, -1), nlive)
        else:
            raise ValueError("Final number of samples differs from number of "
                             "iterations and number of live points in `res2`.")

    # Batch information (if available).
    # note we also check for existance of batch_bounds
    # because unravel_run makes 'static' runs of 1 livepoint
    # but some will have bounds
    if res2.isdynamic() or 'batch_bounds' in res2.keys():
        new_info['batch'] = res2.samples_batch
        new_info['bounds'] = res2.batch_bounds
    else:
        new_info['batch'] = np.zeros(nnew, dtype=int)
        new_info['bounds'] = np.array([(-np.inf, np.inf)])

    # Initialize our new combind run.
    combined_info = dict(id=[],
                         u=[],
                         v=[],
                         logl=[],
                         logvol=[],
                         logwt=[],
                         logz=[],
                         logzvar=[],
                         h=[],
                         nc=[],
                         it=[],
                         n=[],
                         batch=[],
                         blob=[])

    # Check if batch info is the same and modify counters accordingly.
    if np.all(base_info['bounds'] == new_info['bounds']):
        bounds = base_info['bounds']
        boffset = 0
    else:
        bounds = np.concatenate((base_info['bounds'], new_info['bounds']))
        boffset = len(base_info['bounds'])

    # Start our counters at the beginning of each set of dead points.
    idx_base, idx_new = 0, 0
    logl_b, logl_n = base_info['logl'][idx_base], new_info['logl'][idx_new]
    nlive_b, nlive_n = base_n[idx_base], new_n[idx_new]

    # Iteratively walk through both set of samples to simulate
    # a combined run.
    ntot = nbase + nnew
    llmin_b = np.min(base_info['bounds'][base_info['batch']])
    llmin_n = np.min(new_info['bounds'][new_info['batch']])
    for i in range(ntot):
        if logl_b > llmin_n and logl_n > llmin_b:
            # If our samples from the both runs are past the each others'
            # lower log-likelihood bound, both runs are now "active".
            nlive = nlive_b + nlive_n
        elif logl_b <= llmin_n:
            # If instead our collection of dead points from the "base" run
            # are below the bound, just use those.
            nlive = nlive_b
        else:
            # Our collection of dead points from the "new" run
            # are below the bound, so just use those.
            nlive = nlive_n

        # Increment our position along depending on
        # which dead point (saved or new) is worse.

        if logl_b <= logl_n:
            add_idx = idx_base
            from_run = base_info
            idx_base += 1
            combined_info['batch'].append(from_run['batch'][add_idx])
        else:
            add_idx = idx_new
            from_run = new_info
            idx_new += 1
            combined_info['batch'].append(from_run['batch'][add_idx] + boffset)

        for curk in ['id', 'u', 'v', 'logl', 'nc', 'it', 'blob']:
            combined_info[curk].append(from_run[curk][add_idx])

        combined_info['n'].append(nlive)

        # Attempt to step along our samples. If we're out of samples,
        # set values to defaults.
        try:
            logl_b = base_info['logl'][idx_base]
            nlive_b = base_n[idx_base]
        except IndexError:
            logl_b = np.inf
            nlive_b = 0
        try:
            logl_n = new_info['logl'][idx_new]
            nlive_n = new_n[idx_new]
        except IndexError:
            logl_n = np.inf
            nlive_n = 0

    plateau_mode = False
    plateau_counter = 0
    plateau_logdvol = 0
    logvol = 0.
    logl_array = np.array(combined_info['logl'])
    nlive_array = np.array(combined_info['n'])
    for i, (curl, nlive) in enumerate(zip(logl_array, nlive_array)):
        # Save the number of live points and expected ln(volume).
        if not plateau_mode:
            plateau_mask = (logl_array[i:] == curl)
            nplateau = plateau_mask.sum()
            if nplateau > 1:
                # the number of live points should not change throughout
                # the plateau
                # assert nlive_array[i:][plateau_mask].ptp() == 0
                # TODO currently I disabled this check

                plateau_counter = nplateau
                plateau_logdvol = logvol + np.log(1. / (nlive + 1))
                plateau_mode = True
        if not plateau_mode:
            logvol -= math.log((nlive + 1.) / nlive)
        else:
            logvol = logvol + np.log1p(-np.exp(plateau_logdvol - logvol))
        combined_info['logvol'].append(logvol)
        if plateau_mode:
            plateau_counter -= 1
            if plateau_counter == 0:
                plateau_mode = False
    # Compute sampling efficiency.
    eff = 100. * ntot / sum(combined_info['nc'])

    # Save results.
    r = dict(niter=ntot,
             ncall=np.asarray(combined_info['nc']),
             eff=eff,
             samples=np.asarray(combined_info['v']),
             logl=np.asarray(combined_info['logl']),
             logvol=np.asarray(combined_info['logvol']),
             batch_bounds=np.asarray(bounds),
             blob=np.asarray(combined_info['blob']))

    for curk in ['id', 'it', 'n', 'u', 'batch']:
        r['samples_' + curk] = np.asarray(combined_info[curk])

    # Compute the posterior quantities of interest if desired.
    if compute_aux:

        (r['logwt'], r['logz'], combined_logzvar,
         r['information']) = compute_integrals(logvol=r['logvol'],
                                               logl=r['logl'])
        r['logzerr'] = np.sqrt(np.maximum(combined_logzvar, 0))

        # Compute batch information.
        combined_id = np.asarray(combined_info['id'])
        batch_nlive = [
            len(np.unique(combined_id[combined_info['batch'] == i]))
            for i in np.unique(combined_info['batch'])
        ]

        # Add to our results.
        r['batch_nlive'] = np.array(batch_nlive, dtype=int)

    # Combine to form final results object.
    res = Results(r)

    return res


def _kld_error(args):
    """ Internal `pool.map`-friendly wrapper for :meth:`kld_error`
    used by :meth:`stopping_function`."""

    # Extract arguments.
    results, error, approx, rseed = args
    rstate = get_random_generator(rseed)
    return kld_error(results,
                     error,
                     rstate=rstate,
                     return_new=True,
                     approx=approx)


def old_stopping_function(results,
                          args=None,
                          rstate=None,
                          M=None,
                          return_vals=False):
    """
    The old stopping function utilized by :class:`DynamicSampler`.
    Zipped parameters are passed to the function via :data:`args`.
    Assigns the run a stopping value based on a weighted average of the
    stopping values for the posterior and evidence::
        stop = pfrac * stop_post + (1.- pfrac) * stop_evid
    The evidence stopping value is based on the estimated evidence error
    (i.e. standard deviation) relative to a given threshold::
        stop_evid = evid_std / evid_thresh
    The posterior stopping value is based on the fractional error (i.e.
    standard deviation / mean) in the Kullback-Leibler (KL) divergence
    relative to a given threshold::
        stop_post = (kld_std / kld_mean) / post_thresh
    Estimates of the mean and standard deviation are computed using `n_mc`
    realizations of the input using a provided `'error'` keyword (either
    `'jitter'` or `'resample'`).
    Returns the boolean `stop <= 1`. If `True`, the :class:`DynamicSampler`
    will stop adding new samples to our results.
    Parameters
    ----------
    results : :class:`Results` instance
        :class:`Results` instance.
    args : dictionary of keyword arguments, optional
        Arguments used to set the stopping values. Default values are
        `pfrac = 1.0`, `evid_thresh = 0.1`, `post_thresh = 0.02`,
        `n_mc = 128`, `error = 'jitter'`, and `approx = True`.
    rstate : `~numpy.random.Generator`, optional
        `~numpy.random.Generator` instance.
    M : `map` function, optional
        An alias to a `map`-like function. This allows users to pass
        functions from pools (e.g., `pool.map`) to compute realizations in
        parallel. By default the standard `map` function is used.
    return_vals : bool, optional
        Whether to return the stopping value (and its components). Default
        is `False`.
    Returns
    -------
    stop_flag : bool
        Boolean flag indicating whether we have passed the desired stopping
        criteria.
    stop_vals : tuple of shape (3,), optional
        The individual stopping values `(stop_post, stop_evid, stop)` used
        to determine the stopping criteria.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings("once")
        warnings.warn(
            "This an old stopping function that will "
            "be removed in future releases", DeprecationWarning)
    # Initialize values.
    if args is None:
        args = {}
    if M is None:
        M = map

    # Initialize hyperparameters.
    pfrac = args.get('pfrac', 1.0)
    if not 0. <= pfrac <= 1.:
        raise ValueError(
            f"The provided `pfrac` {pfrac} is not between 0. and 1.")
    evid_thresh = args.get('evid_thresh', 0.1)
    if pfrac < 1. and evid_thresh < 0.:
        raise ValueError(
            f"The provided `evid_thresh` {evid_thresh} is not non-negative "
            f"even though `pfrac` is {pfrac}.")
    post_thresh = args.get('post_thresh', 0.02)
    if pfrac > 0. and post_thresh < 0.:
        raise ValueError(
            f"The provided `post_thresh` {post_thresh} is not non-negative "
            f"even though `pfrac` is {pfrac}.")
    n_mc = args.get('n_mc', 128)
    if n_mc <= 1:
        raise ValueError(f"The number of realizations {n_mc} must be greater "
                         "than 1.")
    if n_mc < 20:
        warnings.warn("Using a small number of realizations might result in "
                      "excessively noisy stopping value estimates.")
    error = args.get('error', 'jitter')
    if error not in {'jitter', 'resample'}:
        raise ValueError(f"The chosen `'error'` option {error} is not valid.")
    approx = args.get('approx', True)

    # Compute realizations of ln(evidence) and the KL divergence.
    rlist = [results for i in range(n_mc)]
    error_list = [error for i in range(n_mc)]
    approx_list = [approx for i in range(n_mc)]
    seeds = get_seed_sequence(rstate, n_mc)
    args = zip(rlist, error_list, approx_list, seeds)
    outputs = list(M(_kld_error, args))
    kld_arr, lnz_arr = np.array([(kld[-1], res.logz[-1])
                                 for kld, res in outputs]).T

    # Evidence stopping value.
    lnz_std = np.std(lnz_arr)
    stop_evid = lnz_std / evid_thresh

    # Posterior stopping value.
    kld_mean, kld_std = np.mean(kld_arr), np.std(kld_arr)
    stop_post = (kld_std / kld_mean) / post_thresh

    # Effective stopping value.
    stop = pfrac * stop_post + (1. - pfrac) * stop_evid

    if return_vals:
        return stop <= 1., (stop_post, stop_evid, stop)
    else:
        return stop <= 1.


def restore_sampler(fname):
    """
    Restore the dynamic sampler from a file.
    It is assumed that the file was created using .save() method
    of DynamicNestedSampler or as a result of checkpointing during
    run_nested()

    Parameters
    ----------
    fname: string
        Filename of the save file.

    Returns
    -------
    Static or dynamic nested sampling object

    """
    with open(fname, 'rb') as fp:
        res = pickle_module.load(fp)
    sampler = res['sampler']
    save_ver = res['version']
    dynesty_format_version = 1
    file_format_version = res['format_version']
    if file_format_version != dynesty_format_version:
        raise RuntimeError('Incorrect format version')
    if save_ver != DYNESTY_VERSION:
        warnings.warn(
            f'The dynesty version in the checkpoint file ({save_ver})'
            f'does not match the current dynesty version'
            '({DYNESTY_VERSION}). That is *NOT* guaranteed to work')
    if hasattr(sampler, 'sampler'):
        # This is the case of th dynamic sampler
        # this is better be written as isinstanceof()
        # but I couldn't do it due to circular imports
        # TODO

        # Here we are dealing with the special case of dynamic sampler
        # where it has internal samplers that also need their pool configured
        # this is the initial sampler
        samplers = [sampler, sampler.sampler]
        if sampler.batch_sampler is not None:
            samplers.append(sampler.batch_sampler)
    else:
        samplers = [sampler]

    return sampler


def save_sampler(sampler, fname):
    """
    Save the state of the dynamic sampler in a file

    Parameters
    ----------
    sampler: object
        Dynamic or Static nested sampler
    fname: string
        Filename of the save file.

    """
    format_version = 1
    # this is an internal version of the format we are
    # using. Increase this if incompatible changes are being made
    D = {
        'sampler': sampler,
        'version': DYNESTY_VERSION,
        'format_version': format_version
    }
    tmp_fname = fname + '.tmp'
    try:
        with open(tmp_fname, 'wb') as fp:
            pickle_module.dump(D, fp)
        try:
            os.rename(tmp_fname, fname)
        except FileExistsError:
            # this can happen in Windows, See #450
            shutil.move(tmp_fname, fname)
    except:  # noqa
        try:
            os.unlink(tmp_fname)
        except:  # noqa
            pass
        raise


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
                ax.hist(vals_ / nlive, bins=30, density=True, histtype="step", label=label)
        return pvals
    else:
        nlive = result["nlive"]
        pval = compute_pvalue(vals, result["nlive"])
        label = f"{kind.title()}: $p_{{\\rm value }}={pval:.2f}, n_{{\\rm live}}={nlive}$"
        if ax is not None:
            ax.hist(vals / nlive, bins=30, density=True, histtype="step", label=label)
        return pval


@jax.jit
def distance_insertion_index(live_u, start, point):
    """
    Compute the distance insertion index as defined in XXX
    """
    xp = array_namespace(point)
    norms = xp.std(live_u, axis=0)
    distance = xp.linalg.norm((point - start) / norms)
    all_distances = xp.array([xp.linalg.norm((start - u) / norms) for u in live_u])
    return xp.sum(all_distances < distance)


@jax.jit
def likelihood_insertion_index(live_logl, logl):
    """
    Compute the likelihood insertion index as defined in arxiv:2006.03371
    """
    xp = array_namespace(live_logl)
    return xp.sum(xp.array(live_logl) < logl)
