#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions for proposing new live points used by
:class:`~dynesty.sampler.Sampler` (and its children from
:mod:`~dynesty.nestedsamplers`) and
:class:`~dynesty.dynamicsampler.DynamicSampler`.

"""
from functools import partial

import warnings
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfinv
from scipy._lib._array_api import array_namespace

from .utils import unitcheck, get_random_generator
from .bounding import Ellipsoid

__all__ = [
    "sample_unif", "fixed_rwalk_jax", "sample_slice", "sample_rslice",
    "sample_hslice"
]


@partial(jax.jit, static_argnames=("prior_transform", "loglikelihood"))
def sample_unif(live, rseed, prior_transform, loglikelihood, loglstar, scale, kwargs):
    """
    Evaluate a new point sampled uniformly from a bounding proposal
    distribution.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the initial sample.

    loglstar : float
        Ln(likelihood) bound. **Not applicable here.**

    axes : `~numpy.ndarray` with shape (ndim, ndim)
        Axes used to propose new points. **Not applicable here.**

    scale : float
        Value used to scale the provided axes. **Not applicable here.**

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    kwargs : dict
        A dictionary of additional method-specific parameters.
        **Not applicable here.**

    Returns
    -------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the final proposed point within the unit cube. **For
        uniform sampling this is the same as the initial input position.**

    v : `~numpy.ndarray` with shape (ndim,)
        Position of the final proposed point in the target parameter space.

    logl : float
        Ln(likelihood) of the final proposed point.

    nc : int
        Number of function calls used to generate the sample. For uniform
        sampling this is `1` by construction.

    blob : dict
        Collection of ancillary quantities used to tune :data:`scale`. **Not
        applicable for uniform sampling.**

    """
    xp = array_namespace(live)
    rstate = get_random_generator(rseed)
    ell = Ellipsoid(xp.mean(live, axis=0), xp.cov(live.T) * 10)
    # ell.update(live, rstate)
    u = ell.samples(live.shape[0], rstate)
    # u = ell.samples(100, rstate)
    # u = jax.random.uniform(rseed, live.shape)
    v = prior_transform(u.T).T
    logl = jax.vmap(lambda x: loglikelihood(x).val)(v)
    if jax.config.jax_enable_x64:
        inttype = xp.int64
    else:
        inttype = xp.int32
    logl = jax.lax.select(
        (xp.min(u, axis=-1) >= 0) & (xp.max(u, axis=-1) <= 1),
        logl,
        xp.full_like(logl, loglstar - 100),
    )
    nc = xp.ones(u.shape[0], dtype=inttype)
    blob = None

    return u, v, logl, nc, blob


@jax.jit
def doubling_cond_fun(val, loglstar):
    rhat, lhat, x1, f_lhat, f_rhat, _ = val
    outer_condition = (rhat - lhat) > 1.1
    midpoint = (lhat + rhat) / 2
    done = (
        (jnp.array(0.0) < midpoint) & (midpoint <= x1)
    ) | (
        (x1 < midpoint) & (midpoint <= jnp.array(0.0))
    )
    inner_condition = (loglstar >= f_lhat) & (loglstar >= f_rhat)
    return outer_condition & done & inner_condition


@partial(jax.jit, static_argnums=(1,))
def doubling_body_fun(val, F):
    rhat, lhat, x1, f_lhat, f_rhat, nc = val
    midpoint = (lhat + rhat) / 2
    fhat = F(midpoint)
    nc += 1
    lhat, rhat, f_lhat, f_rhat = jax.lax.select(
        x1 < midpoint,
        jnp.array([lhat, midpoint, f_lhat, fhat]),
        jnp.array([midpoint, rhat, fhat, f_rhat]),
    )
    return rhat, lhat, x1, f_lhat, f_rhat, nc


@partial(jax.jit, static_argnums=(1,))
def _slice_doubling_accept(x1, F, loglstar, L, R, fL, fR):
    """
    Acceptance test of slice sampling when doubling mode is used.
    This is an exact implementation of algorithm 6 of Neal 2003
    here w=1 and x0=0 as we are working in the
    coordinate system of F(A) = f(x0+A*w)

    Arguments are
    1) candidate location x1
    2) wrapped logl function (see generic_slice_step)
    3) threshold logl value
    4) left edge of the full interval
    5) right edge of the full interval
    6) value at left edge
    7) value at right edge
    """
    rhat, lhat, x1, _, _, nc = jax.lax.while_loop(
        partial(doubling_cond_fun, loglstar=loglstar),
        partial(doubling_body_fun, F=F),
        (R, L, x1, fL, fR, 0),
    )
    return (rhat - lhat) < 1.1, nc


def generic_slice_step(u, direction, nonperiodic, loglstar, loglikelihood,
                       prior_transform, doubling, rstate):
    """
    Do a slice generic slice sampling step along a specified dimension

    Arguments
    u: ndarray (ndim sized)
        Starting point in unit cube coordinates
        It MUST satisfy the logl>loglstar criterion
    direction: ndarray (ndim sized)
        Step direction vector
    nonperiodic: ndarray(bool)
        mask for nonperiodic variables
    loglstar: float
        the critical value of logl, so that new logl must be >loglstar
    loglikelihood: function
    prior_transform: function
    rstate: random state
    """
    xp = array_namespace(u)
    nc, nexpand, ncontract = 0, 0, 0
    nexpand_threshold = 1000  # Threshold for warning the user
    n = len(u)
    rand0 = rstate.random()  # initial scale/offset
    dirlen = xp.linalg.norm(direction)
    maxlen = xp.sqrt(n) / 2.
    # maximum initial interval length (the diagonal of the cube)
    if dirlen > maxlen:
        # I stopped giving warnings, as it was too noisy
        dirnorm = dirlen / maxlen
    else:
        dirnorm = 1
    direction = direction / dirnorm

    #  The function that evaluates the logl at the location of
    # u0 + x*direction0
    def F(x):
        nonlocal nc
        u_new = u + x * direction
        if unitcheck(u_new, nonperiodic):
            logl = loglikelihood(prior_transform(u_new)).val
        else:
            logl = -np.inf
        nc += 1
        return u_new, logl

    # asymmetric step size on the left/right (see Neal 2003)
    nstep_l = -rand0
    nstep_r = (1 - rand0)

    logl_l = F(nstep_l)[1]
    logl_r = F(nstep_r)[1]
    expansion_warning = False
    if not doubling:
        # "Stepping out" the left and right bounds.
        while logl_l > loglstar:
            nstep_l -= 1
            logl_l = F(nstep_l)[1]
            nexpand += 1
        while logl_r > loglstar:
            nstep_r += 1
            logl_r = F(nstep_r)[1]
            nexpand += 1
        if nexpand > nexpand_threshold:
            expansion_warning = True
            warnings.warn('The slice sample interval was expanded more '
                          f'than {nexpand_threshold} times')

    else:
        # "Stepping out" the left and right bounds.
        K = 1
        while (logl_l > loglstar or logl_r > loglstar):
            V = rstate.random()
            if V < 0.5:
                nstep_l -= (nstep_r - nstep_l)
                logl_l = F(nstep_l)[1]
            else:
                nstep_r += (nstep_r - nstep_l)
                logl_r = F(nstep_r)[1]
            nexpand += K
            K *= 2
        L = nstep_l
        R = nstep_r
        fL = logl_l
        fR = logl_r

    # Sample within limits. If the sample is not valid, shrink
    # the limits until we hit the `loglstar` bound.

    doubling_accept = not doubling

    logl_prop = loglstar - 1
    while (logl_prop <= loglstar) or not doubling_accept:
        # Define slice and window.
        nstep_hat = nstep_r - nstep_l

        # Propose new position.
        nstep_prop = nstep_l + rstate.random() * nstep_hat  # scale from left
        u_prop, logl_prop = F(nstep_prop)
        ncontract += 1

        if doubling:
            accept, nc_ = _slice_doubling_accept(
                nstep_prop, F, loglstar, L, R, fL, fR
            )
            nc += nc_
            doubling_accept = not doubling or accept
        else:
            doubling_accept = True

        if nstep_prop < 0:
            nstep_l = nstep_prop
        elif nstep_prop > 0:  # right
            nstep_r = nstep_prop
        else:
            # If `nstep_prop = 0` something has gone horribly wrong.
            raise RuntimeError("Slice sampler has failed to find "
                                "a valid point. Some useful "
                                "output quantities:\n"
                                f"u: {u}\n"
                                f"nstep_left: {nstep_l}\n"
                                f"nstep_right: {nstep_r}\n"
                                f"nstep_hat: {nstep_hat}\n"
                                f"u_prop: {u_prop}\n"
                                f"loglstar: {loglstar}\n"
                                f"logl_prop: {logl_prop}\n"
                                f"direction: {direction}\n")
    v_prop = prior_transform(u_prop)
    return u_prop, v_prop, logl_prop, nc, nexpand, ncontract, expansion_warning


def sample_slice(args):
    """
    Return a new live point proposed by a series of random slices
    away from an existing live point. Standard "Gibs-like" implementation where
    a single multivariate "slice" is a combination of `ndim` univariate slices
    through each axis.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the initial sample. **This is a copy of an existing live
        point.**

    loglstar : float
        Ln(likelihood) bound.

    axes : `~numpy.ndarray` with shape (ndim, ndim)
        Axes used to propose new points. For slices new positions are
        proposed along the arthogonal basis defined by :data:`axes`.

    scale : float
        Value used to scale the provided axes.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    kwargs : dict
        A dictionary of additional method-specific parameters.

    Returns
    -------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the final proposed point within the unit cube.

    v : `~numpy.ndarray` with shape (ndim,)
        Position of the final proposed point in the target parameter space.

    logl : float
        Ln(likelihood) of the final proposed point.

    nc : int
        Number of function calls used to generate the sample.

    blob : dict
        Collection of ancillary quantities used to tune :data:`scale`.

    """

    # Unzipping.
    (u, loglstar, axes, scale, prior_transform, loglikelihood,
     kwargs) = (args.u, args.loglstar, args.axes, args.scale,
                args.prior_transform, args.loglikelihood, args.kwargs)
    rstate = get_random_generator(args.rseed)
    # Periodicity.
    nonperiodic = kwargs.get('nonperiodic', None)
    doubling = kwargs.get('slice_doubling', False)
    # Setup.
    n = len(u)
    assert axes.shape[0] == n
    slices = kwargs.get('slices', 5)  # number of slices
    nc = 0
    nexpand = 0
    ncontract = 0

    # Modifying axes and computing lengths.
    axes = scale * axes.T  # scale based on past tuning
    expansion_warning_set = False
    # Slice sampling loop.
    for _ in range(slices):

        # Shuffle axis update order.
        idxs = np.arange(n)
        rstate.shuffle(idxs)

        # Slice sample along a random direction.
        for idx in idxs:

            # Select axis.
            axis = axes[idx]
            (u_prop, v_prop, logl_prop, nc1, nexpand1, ncontract1,
             expansion_warning) = generic_slice_step(u, axis, nonperiodic,
                                                     loglstar, loglikelihood,
                                                     prior_transform, doubling,
                                                     rstate)
            u = u_prop
            nc += nc1
            nexpand += nexpand1
            ncontract += ncontract1
            if expansion_warning and not doubling:
                # if we expanded the interval by more than
                # the threshold we set the warning and enable doubling
                expansion_warning_set = True
                doubling = True
                warnings.warn('Enabling doubling strategy of slice '
                              'sampling from Neal(2003)')
    blob = {
        'nexpand': nexpand,
        'ncontract': ncontract,
        'expansion_warning_set': expansion_warning_set
    }

    return u_prop, v_prop, logl_prop, nc, blob


def sample_rslice(args):
    """
    Return a new live point proposed by a series of random slices
    away from an existing live point. Standard "random" implementation where
    each slice is along a random direction based on the provided axes.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the initial sample. **This is a copy of an existing live
        point.**

    loglstar : float
        Ln(likelihood) bound.

    axes : `~numpy.ndarray` with shape (ndim, ndim)
        Axes used to propose new slice directions.

    scale : float
        Value used to scale the provided axes.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    kwargs : dict
        A dictionary of additional method-specific parameters.

    Returns
    -------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the final proposed point within the unit cube.

    v : `~numpy.ndarray` with shape (ndim,)
        Position of the final proposed point in the target parameter space.

    logl : float
        Ln(likelihood) of the final proposed point.

    nc : int
        Number of function calls used to generate the sample.

    blob : dict
        Collection of ancillary quantities used to tune :data:`scale`.

    """

    # Unzipping.
    (u, loglstar, axes, scale, prior_transform, loglikelihood,
     kwargs) = (args.u, args.loglstar, args.axes, args.scale,
                args.prior_transform, args.loglikelihood, args.kwargs)
    rstate = get_random_generator(args.rseed)
    # Periodicity.
    nonperiodic = kwargs.get('nonperiodic', None)
    doubling = kwargs.get('slice_doubling', False)

    xp = array_namespace(u)

    # Setup.
    n = len(u)
    assert axes.shape[0] == n
    slices = kwargs.get('slices', 5)  # number of slices
    nc = 0
    nexpand = 0
    ncontract = 0
    expansion_warning_set = False

    # Slice sampling loop.
    for _ in range(slices):

        # Propose a direction on the unit n-sphere.
        drhat = rstate.standard_normal(size=n)
        drhat /= xp.linalg.norm(drhat)

        # Transform and scale based on past tuning.
        direction = axes @ drhat * scale

        (u_prop, v_prop, logl_prop, nc1, nexpand1, ncontract1,
         expansion_warning) = generic_slice_step(u, direction, nonperiodic,
                                                 loglstar, loglikelihood,
                                                 prior_transform, doubling,
                                                 rstate)
        u = u_prop
        nc += nc1
        nexpand += nexpand1
        ncontract += ncontract1
        if expansion_warning and not doubling:
            doubling = True
            expansion_warning_set = True
            warnings.warn('Enabling doubling strategy of slice '
                          'sampling from Neal(2003)')

    blob = {
        'nexpand': nexpand,
        'ncontract': ncontract,
        'expansion_warning_set': expansion_warning_set
    }

    return u_prop, v_prop, logl_prop, nc, blob


def sample_hslice(args):
    """
    Return a new live point proposed by "Hamiltonian" Slice Sampling
    using a series of random trajectories away from an existing live point.
    Each trajectory is based on the provided axes and samples are determined
    by moving forwards/backwards in time until the trajectory hits an edge
    and approximately reflecting off the boundaries.
    Once a series of reflections has been established, we propose a new live
    point by slice sampling across the entire path.

    Parameters
    ----------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the initial sample. **This is a copy of an existing live
        point.**

    loglstar : float
        Ln(likelihood) bound.

    axes : `~numpy.ndarray` with shape (ndim, ndim)
        Axes used to propose new slice directions.

    scale : float
        Value used to scale the provided axes.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    kwargs : dict
        A dictionary of additional method-specific parameters.

    Returns
    -------
    u : `~numpy.ndarray` with shape (npdim,)
        Position of the final proposed point within the unit cube.

    v : `~numpy.ndarray` with shape (ndim,)
        Position of the final proposed point in the target parameter space.

    logl : float
        Ln(likelihood) of the final proposed point.

    nc : int
        Number of function calls used to generate the sample.

    blob : dict
        Collection of ancillary quantities used to tune :data:`scale`.

    """

    # Unzipping.
    (u, loglstar, axes, scale, prior_transform, loglikelihood,
     kwargs) = (args.u, args.loglstar, args.axes, args.scale,
                args.prior_transform, args.loglikelihood, args.kwargs)
    rstate = get_random_generator(args.rseed)

    xp = array_namespace(u)

    # Periodicity.
    nonperiodic = kwargs.get('nonperiodic', None)

    # Setup.
    n = len(u)
    assert axes.shape[0] == len(u)
    slices = kwargs.get('slices', 5)  # number of slices
    grad = kwargs.get('grad', None)  # gradient of log-likelihood
    max_move = kwargs.get('max_move', 100)  # limit for `ncall`
    compute_jac = kwargs.get('compute_jac', False)  # whether Jacobian needed
    jitter = 0.25  # 25% jitter
    nc = 0
    nmove = 0
    nreflect = 0
    ncontract = 0

    # Slice sampling loop.
    for _ in range(slices):
        # Define the left, "inner", and right "nodes" for a given chord.
        # We will plan to slice sampling using these chords.
        nodes_l, nodes_m, nodes_r = [], [], []

        # Propose a direction on the unit n-sphere.
        drhat = rstate.standard_normal(size=n)
        drhat /= xp.linalg.norm(drhat)

        # Transform and scale based on past tuning.
        axis = axes @ drhat * scale * 0.01

        # Create starting window.
        vel = xp.array(axis)  # current velocity
        u_l = u.copy()
        u_r = u.copy()
        u_l -= rstate.uniform(1. - jitter, 1. + jitter) * vel
        u_r += rstate.uniform(1. - jitter, 1. + jitter) * vel
        nodes_l.append(xp.array(u_l))
        nodes_m.append(xp.array(u))
        nodes_r.append(xp.array(u_r))

        # Progress "right" (i.e. "forwards" in time).
        reverse, reflect = False, False
        u_r = xp.array(u)
        ncall = 0
        while ncall <= max_move:

            # Iterate until we can bracket the edge of the distribution.
            nodes_l.append(xp.array(u_r))
            u_out, u_in = None, []
            while True:
                # Step forward.
                u_r += rstate.uniform(1. - jitter, 1. + jitter) * vel
                # Evaluate point.
                if unitcheck(u_r, nonperiodic):
                    v_r = prior_transform(xp.asarray(u_r))
                    logl_r = loglikelihood(xp.asarray(v_r))
                    nc += 1
                    ncall += 1
                    nmove += 1
                else:
                    logl_r = -np.inf
                # Check if we satisfy the log-likelihood constraint
                # (i.e. are "in" or "out" of bounds).
                if logl_r < loglstar:
                    if reflect:
                        # If we are out of bounds and just reflected, we
                        # reverse direction and terminate immediately.
                        reverse = True
                        nodes_l.pop()  # remove since chord does not exist
                        break
                    else:
                        # If we're already in bounds, then we're safe.
                        u_out = xp.array(u_r)
                        logl_out = logl_r
                    # Check if we could compute gradients assuming we
                    # terminated with the current `u_out`.
                    if xp.isfinite(logl_out.val):
                        reverse = False
                    else:
                        reverse = True
                else:
                    reflect = False
                    u_in.append(xp.array(u_r))
                # Check if we've bracketed the edge.
                if u_out is not None:
                    break
            # Define the rest of our chord.
            if len(nodes_l) == len(nodes_r) + 1:
                if len(u_in) > 0:
                    u_in = u_in[rstate.choice(
                        len(u_in))]  # pick point randomly
                else:
                    u_in = xp.array(u)
                    pass
                nodes_m.append(xp.array(u_in))
                nodes_r.append(xp.array(u_out))
            # Check if we have turned around.
            if reverse:
                break

            # Reflect off the boundary.
            u_r, logl_r = u_out, logl_out
            # If the gradient is provided, evaluate it.
            h = grad(v_r)
            if compute_jac:
                jac = kwargs["jac"](u_r)
                h = xp.linalg.inv(jac) @ h
            nc += 1
            # Compute specular reflection off boundary.
            vel_ref = vel - 2 * h * vel @ h / xp.linalg.norm(h)**2
            dotprod = vel_ref @ vel
            dotprod /= xp.linalg.norm(vel_ref) * xp.linalg.norm(vel)
            # Check angle of reflection.
            if dotprod < -0.99:
                # The reflection angle is sufficiently small that it might
                # as well be a reflection.
                reverse = True
                break
            else:
                # If the reflection angle is sufficiently large, we
                # proceed as normal to the new position.
                vel = vel_ref
                u_out = None
                reflect = True
                nreflect += 1

        # Progress "left" (i.e. "backwards" in time).
        reverse, reflect = False, False
        vel = -xp.array(axis)  # current velocity
        u_l = xp.array(u)
        ncall = 0
        while ncall <= max_move:

            # Iterate until we can bracket the edge of the distribution.
            # Use a doubling approach to try and locate the bounds faster.
            nodes_r.append(xp.array(u_l))
            u_out, u_in = None, []
            while True:
                # Step forward.
                u_l += rstate.uniform(1. - jitter, 1. + jitter) * vel
                # Evaluate point.
                if unitcheck(u_l, nonperiodic):
                    v_l = prior_transform(xp.asarray(u_l))
                    logl_l = loglikelihood(xp.asarray(v_l))
                    nc += 1
                    ncall += 1
                    nmove += 1
                else:
                    logl_l = -np.inf
                # Check if we satisfy the log-likelihood constraint
                # (i.e. are "in" or "out" of bounds).
                if logl_l < loglstar:
                    if reflect:
                        # If we are out of bounds and just reflected, we
                        # reverse direction and terminate immediately.
                        reverse = True
                        nodes_r.pop()  # remove since chord does not exist
                        break
                    else:
                        # If we're already in bounds, then we're safe.
                        u_out = xp.array(u_l)
                        logl_out = logl_l
                    # Check if we could compute gradients assuming we
                    # terminated with the current `u_out`.
                    if xp.isfinite(logl_out.val):
                        reverse = False
                    else:
                        reverse = True
                else:
                    reflect = False
                    u_in.append(xp.array(u_l))
                # Check if we've bracketed the edge.
                if u_out is not None:
                    break
            # Define the rest of our chord.
            if len(nodes_r) == len(nodes_l) + 1:
                if len(u_in) > 0:
                    u_in = u_in[rstate.choice(
                        len(u_in))]  # pick point randomly
                else:
                    u_in = xp.array(u)
                    pass
                nodes_m.append(xp.array(u_in))
                nodes_l.append(xp.array(u_out))
            # Check if we have turned around.
            if reverse:
                break

            # Reflect off the boundary.
            u_l, logl_l = u_out, logl_out
            # If the gradient is provided, evaluate it.
            h = grad(v_l)
            if compute_jac:
                jac = kwargs["jac"](u_r)
                h = xp.linalg.inv(jac) @ h
            nc += 1
            # Compute specular reflection off boundary.
            vel_ref = vel - 2 * h * vel @ h / xp.linalg.norm(h)**2
            dotprod = vel_ref @ vel
            dotprod /= xp.linalg.norm(vel_ref) * xp.linalg.norm(vel)
            # Check angle of reflection.
            if dotprod < -0.99:
                # The reflection angle is sufficiently small that it might
                # as well be a reflection.
                reverse = True
                break
            else:
                # If the reflection angle is sufficiently large, we
                # proceed as normal to the new position.
                vel = vel_ref
                u_out = None
                reflect = True
                nreflect += 1

        # Initialize lengths of chords.
        if len(nodes_l) > 1:
            # remove initial fallback chord
            nodes_l.pop(0)
            nodes_m.pop(0)
            nodes_r.pop(0)
        nodes_l, nodes_m, nodes_r = (xp.array(nodes_l), xp.array(nodes_m),
                                     xp.array(nodes_r))
        Nchords = len(nodes_l)
        axlen = xp.array([
            xp.linalg.norm(nr - nl) for nl, nr in zip(nodes_l, nodes_r)
        ])

        # Slice sample from all chords simultaneously. This is equivalent to
        # slice sampling in *time* along our trajectory.
        axlen_init = xp.array(axlen)
        while True:
            # Safety check.
            if xp.any(axlen < 1e-5 * axlen_init):
                raise RuntimeError("Hamiltonian slice sampling appears to be "
                                   "stuck! Some useful output quantities:\n"
                                   f"u: {u}\n u_left: {u_l}\n"
                                   f"u_right: {u_r}\n loglstar: {loglstar}.")

            # Select chord.
            axprob = axlen / xp.sum(axlen)
            idx = rstate.choice(Nchords, p=axprob)
            # Define chord.
            u_l, u_m, u_r = nodes_l[idx], nodes_m[idx], nodes_r[idx]
            u_hat = u_r - u_l
            rprop = rstate.random()
            u_prop = u_l + rprop * u_hat  # scale from left
            if unitcheck(u_prop, nonperiodic):
                v_prop = prior_transform(xp.asarray(u_prop))
                logl_prop = loglikelihood(xp.asarray(v_prop)).val
            else:
                logl_prop = -np.inf
            nc += 1
            ncontract += 1
            # If we succeed, move to the new position.
            if logl_prop > loglstar:
                u = u_prop
                break
            # If we fail, check if the new point is to the left/right of
            # the point interior to the bounds (`u_m`) and update
            # the bounds accordingly.
            else:
                s = (u_prop - u_m) @ u_hat  # check sign (+/-)
                if s < 0:  # left
                    nodes_l[idx] = u_prop
                    axlen[idx] *= 1 - rprop
                elif s > 0:  # right
                    nodes_r[idx] = u_prop
                    axlen[idx] *= rprop
                else:
                    raise RuntimeError(
                        "Slice sampler has failed to find "
                        "a valid point. Some useful "
                        "output quantities:\n"
                        f"u: {u}\n u_left: {u_l}\n"
                        f"u_right: {u_r}\n u_hat: {u_hat}\n"
                        f"u_prop: {u_prop}\n loglstar: {loglstar}\n"
                        f"logl_prop: {logl_prop}")

    blob = {'nmove': nmove, 'nreflect': nreflect, 'ncontract': ncontract}

    return u_prop, v_prop, logl_prop, nc, blob



@partial(jax.jit, static_argnames=("ptform", "lnl"))
def _eval(u_prop, loglstar, ptform, lnl):
    v_prop = ptform(norm_to_unif(u_prop))
    logl_prop = lnl(v_prop).val
    return jax.lax.select(logl_prop > loglstar, 1, 0)


def _null(*args):
    return jnp.array(0)


@partial(jax.jit, static_argnames=("ptform", "lnl"))
def _accept(u_prop, loglstar, ptform, lnl):
    return jax.lax.cond(
        jnp.isfinite(u_prop[0]),
        partial(_eval, ptform=ptform, lnl=lnl),
        _null,
        u_prop,
        loglstar,
    )


@partial(jax.jit, static_argnames=("ptform", "lnl"))
def for_step(idx, args, ptform, lnl):
    points, rng_key, loglstar, naccepted, periodic, reflective = args
    length = points.shape[0]
    rng_key, subkey = jax.random.split(rng_key)
    idxs = jax.random.permutation(subkey, length)
    first, second = jnp.split(idxs, 2)
    for active, passive in [(first, second), (second, first)]:
        rng_key, *keys = jax.random.split(rng_key, 5)
        a = points[active]
        b = points[passive]
        diffs = jnp.diff(
            jax.vmap(jax.random.choice, in_axes=(0, None, None, None))(
                jax.random.split(keys[0], length // 2), b, (2,), False
            ),
            axis=1,
        ).squeeze()
        scale = (
            jax.random.gamma(keys[1], 4, (length // 2,))
            * 0.25
            * 2.38
            / (2 * points.shape[1]) ** 0.5
        )
        scale **= jax.random.choice(keys[2], 2, (length // 2,))
        diffs *= scale[:, None] / 1
        proposed = a + diffs
        accept = jax.vmap(
            partial(_accept, ptform=ptform, lnl=lnl), in_axes=(0, None)
        )(proposed, loglstar)
        ln_mh = (
            jnp.sum(a**2, axis=-1) / 2
            - jnp.sum(proposed**2, axis=-1) / 2
        )
        accept = accept & (
            jnp.exp(ln_mh) > jax.random.uniform(keys[3], accept.shape)
        )
        a = a + diffs * accept[:, None]
        points = points.at[active].set(a)
        subset = naccepted[active] + accept
        naccepted = naccepted.at[active].set(subset)
    return points, rng_key, loglstar, naccepted, periodic, reflective


def unif_to_norm(val):
    return erfinv(2 * val - 1) * 2**0.5


def norm_to_unif(val):
    return (erf(val / 2**0.5) + 1) / 2


@partial(jax.jit, static_argnames=("prior_transform", "loglikelihood"))
def fixed_rwalk_jax(
    live, rseed, prior_transform, loglikelihood, loglstar, scale, kwargs
):
    walks = kwargs.get("walks", 1000)

    u, rng_key, _, naccept, _, _ = jax.lax.fori_loop(
        0,
        walks,
        partial(for_step, ptform=prior_transform, lnl=loglikelihood),
        (
            unif_to_norm(live),
            rseed,
            loglstar,
            jnp.zeros(live.shape[0]),
            kwargs["periodic"],
            kwargs["reflective"],
        ),
    )
    u = norm_to_unif(u)
    if jax.config.jax_enable_x64:
        inttype = jnp.int64
    else:
        inttype = jnp.int32
    ncall = walks * jnp.ones_like(naccept, dtype=inttype)

    current_u = jax.vmap(jax.lax.select)(
        naccept > 0, u, jax.random.uniform(rng_key, u.shape)
    )
    current_v = prior_transform(current_u.T).T
    logl = jax.vmap(lambda x: loglikelihood(x).val)(current_v)
    jax.debug.print("mean accepted: {}", jnp.mean(naccept))

    blob = {
        "accept": naccept,
        "reject": walks - naccept,
        "scale": scale * jnp.ones_like(naccept),
    }

    return current_u, current_v, logl, ncall, blob
