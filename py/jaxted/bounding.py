#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bounding classes used when proposing new live points, along with a number of
useful helper functions. Bounding objects include:

    UnitCube:
        The unit N-cube (unconstrained draws from the prior).

    Ellipsoid:
        Bounding ellipsoid.

    MultiEllipsoid:
        A set of (possibly overlapping) bounding ellipsoids.

    RadFriends:
        A set of (possibly overlapping) balls centered on each live point.

    SupFriends:
        A set of (possibly overlapping) cubes centered on each live point.

"""

import warnings
import numpy as np
from scipy import spatial
from scipy import cluster
from scipy import linalg as lalg
from scipy.special import logsumexp, gammaln
from scipy.cluster.vq import kmeans2
from scipy._lib._array_api import scipy_namespace_for, array_namespace
from .utils import unitcheck, get_seed_sequence, get_random_generator

__all__ = [
    "UnitCube", "Ellipsoid", "MultiEllipsoid", "RadFriends", "SupFriends",
    "logvol_prefactor", "randsphere", "bounding_ellipsoid",
    "bounding_ellipsoids", "_bounding_ellipsoids",
    "_ellipsoid_bootstrap_expand", "_friends_bootstrap_radius",
    "_friends_leaveoneout_radius"
]


class UnitCube:
    """
    An N-dimensional unit cube.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the unit cube.

    """

    def __init__(self, ndim):
        self.n = ndim  # dimension
        self.logvol = 0.  # volume
        self.funit = 1.  # overlap with the unit cube

    def contains(self, x):
        """Checks if unit cube contains the point `x`."""

        return unitcheck(x)

    def sample(self, rstate):
        """
        Draw a sample uniformly distributed within the unit cube.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the unit cube.

        """

        return rstate.random(size=self.n)

    def samples(self, nsamples, rstate):
        """
        Draw `nsamples` samples randomly distributed within the unit cube.

        Returns
        -------
        x : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the unit cube.

        """

        return rstate.random(size=(nsamples, self.n))

    def update(self, points, rstate, bootstrap=0, pool=None):
        """Filler function."""
        pass


class Ellipsoid:
    """
    An N-dimensional ellipsoid defined by::

        (x - v)^T A (x - v) = 1

    where the vector `v` is the center of the ellipsoid and `A` is a
    symmetric, positive-definite `N x N` matrix.

    Parameters
    ----------
    ctr : `~numpy.ndarray` with shape (N,)
        Coordinates of ellipsoid center.

    cov : `~numpy.ndarray` with shape (N, N)
        Covariance matrix describing the axes.

    """

    def __init__(self, ctr, cov, am=None, axes=None):
        xp = array_namespace(ctr, cov)
        self.n = len(ctr)  # dimension
        self.ctr = xp.asarray(ctr)  # center coordinates
        self.cov = xp.asarray(cov)  # covariance matrix

        # The eigenvalues (l) of `a` are (a^-2, b^-2, ...) where
        # (a, b, ...) are the lengths of principle axes.
        # The eigenvectors (v) are the normalized principle axes.
        eigvals, eigvecs = xp.linalg.eigh(self.cov)
        self.axlens = xp.sqrt(eigvals)
        # Volume of ellipsoid is the volume of an n-sphere
        # is a product of squares of eigen values
        self.logvol = logvol_prefactor(self.n) + 0.5 * xp.log(eigvals).sum()
        if axes is None:
            self.axes = eigvecs * self.axlens
        else:
            self.axes = axes
        if am is None:
            self.am = (eigvecs * (1. / eigvals)) @ eigvecs.T
            # precision matrix (inverse of covariance)
        else:
            self.am = am

        # Scaled eigenvectors are the principle axes, where `paxes[:,i]` is the
        # i-th axis. Multiplying this matrix by a vector will transform a
        # point in the unit n-sphere to a point in the ellipsoid.
        self.paxes = eigvecs * self.axlens

        # Amount by which volume was increased after initialization (i.e.
        # cumulative factor from `scale_to_vol`).
        self.expand = 1.
        self.funit = 1

    def scale_to_logvol(self, logvol):
        """Scale ellipsoid to a target volume."""
        xp = array_namespace(self.cov)

        logf = (logvol - self.logvol)
        # log of the maxium axis length of the ellipsoid
        max_log_axlen = xp.log(xp.sqrt(self.n) / 2)
        log_axlen = xp.log(self.axlens)
        if log_axlen.max() < max_log_axlen - logf / self.n:
            # we are safe to inflate the ellipsoid isothropically
            # without hitting boundaries
            f = xp.exp(logf / self.n)
            self.cov *= f**2
            self.am *= 1. / f**2
            self.axlens *= f
            self.axes *= f
        else:
            logfax = xp.zeros(self.n)
            curlogf = logf  # how much we have left to inflate
            curn = self.n  # how many dimensions left
            eigvals, eigvecs = xp.linalg.eigh(self.cov)

            # here we start from largest and go to smallest
            for curi in xp.argsort(eigvals)[::-1]:
                delta = max(
                    min(max_log_axlen - log_axlen[curi], curlogf / curn), 0)
                logfax[curi] = delta
                curlogf -= delta
                curn -= 1
            fax = xp.exp(logfax)  # linear inflation of each dimension
            l1 = eigvals * fax**2  # eigen values are squares of axes
            self.cov = (eigvecs * l1) @ eigvecs.T
            self.am = (eigvecs * (1. / l1)) @ eigvecs.T
            self.axlens *= fax
            self.axes = self.axes * fax
        self.logvol = logvol

    def major_axis_endpoints(self):
        """Return the endpoints of the major axis."""
        xp = array_namespace(self.cov)

        i = xp.argmax(self.axlens)  # find the major axis
        v = self.paxes[:, i]  # vector from center to major axis endpoint

        return self.ctr - v, self.ctr + v

    def distance(self, x):
        """Compute the normalized distance to `x` from the center of the
        ellipsoid."""
        d = x - self.ctr

        return (d @ self.am @ d)**0.5

    def distance_many(self, x):
        """Compute the normalized distance to `x` from the center of the
        ellipsoid."""

        d = x - self.ctr[None, :]

        return (d @ self.am @ d.T).diagonal()**0.5

    def contains(self, x):
        """Checks if ellipsoid contains `x`."""

        return self.distance(x) <= 1.0

    def sample(self, rstate, nsamples=None):
        """
        Draw a sample uniformly distributed within the ellipsoid.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the ellipsoid.

        """
        if nsamples is not None:
            mean = self.ctr[:, None]
        else:
            mean = self.ctr
        return mean + self.axes @ randsphere(self.n, rstate=rstate, nsamples=nsamples)

    def samples(self, nsamples, rstate):
        """
        Draw `nsamples` samples uniformly distributed within the ellipsoid.

        Returns
        -------
        x : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the ellipsoid.

        """
        return self.sample(rstate, nsamples).T

    def unitcube_overlap(self, ndraws=10000, rstate=None):
        """Using `ndraws` Monte Carlo draws, estimate the fraction of
        overlap between the ellipsoid and the unit cube."""

        samples = [self.sample(rstate=rstate) for i in range(ndraws)]
        nin = sum((unitcheck(x) for x in samples))

        return 1. * nin / ndraws

    def update(self,
               points,
               rstate,
               bootstrap=0,
               pool=None,
               mc_integrate=False):
        """
        Update the ellipsoid to bound the collection of points.

        Parameters
        ----------
        points : `~numpy.ndarray` with shape (npoints, ndim)
            The set of points to bound.

        rstate : `~numpy.random.Generator`, optional
            `~numpy.random.Generator` instance.

        bootstrap : int, optional
            The number of bootstrapped realizations of the ellipsoid. The
            maximum distance to the set of points "left out" during each
            iteration is used to enlarge the resulting volumes.
            Default is `0`.

        pool : user-provided pool, optional
            Use this pool of workers to execute operations in parallel.

        mc_integrate : bool, optional
            Whether to use Monte Carlo methods to compute the effective
            overlap of the final ellipsoid with the unit cube.
            Default is `False`.

        """

        # Compute new bounding ellipsoid.
        ell = bounding_ellipsoid(points)
        self.n = ell.n
        self.ctr = ell.ctr
        self.cov = ell.cov
        self.am = ell.am
        self.logvol = ell.logvol
        self.axlens = ell.axlens
        self.axes = ell.axes
        self.paxes = ell.paxes
        self.expand = ell.expand

        # Use bootstrapping to determine the volume expansion factor.
        if bootstrap > 0:

            # If provided, compute bootstraps in parallel using a pool.
            if pool is None:
                M = map
            else:
                M = pool.map
            multis = [False for it in range(bootstrap)]
            ps = [points for it in range(bootstrap)]
            seeds = get_seed_sequence(rstate, bootstrap)
            args = zip(multis, ps, seeds)
            expands = list(M(_ellipsoid_bootstrap_expand, args))

            # Conservatively set the expansion factor to be the maximum
            # factor derived from our set of bootstraps.
            expand = max(expands)

            # If our ellipsoid is over-constrained, expand it.
            if expand > 1.:
                lv = self.logvol + self.n * np.log(expand)
                self.scale_to_logvol(lv)

        # Estimate the fractional overlap with the unit cube using
        # Monte Carlo integration.
        if mc_integrate:
            self.funit = self.unitcube_overlap(rstate=rstate)


class MultiEllipsoid:
    """
    A collection of M N-dimensional ellipsoids.

    Parameters
    ----------
    ells : list of `Ellipsoid` objects with length M, optional
        A set of `Ellipsoid` objects that make up the collection of
        N-ellipsoids. Used to initialize :class:`MultiEllipsoid` if provided.

    ctrs : `~numpy.ndarray` with shape (M, N), optional
        Collection of coordinates of ellipsoid centers. Used to initialize
        :class:`MultiEllipsoid` if :data:`ams` is also provided.

    covs : `~numpy.ndarray` with shape (M, N, N), optional
        Collection of matrices describing the axes of the ellipsoids. Used to
        initialize :class:`MultiEllipsoid` if :data:`ctrs` also provided.

    """

    def __init__(self, ells=None, ctrs=None, covs=None):
        if ells is not None:
            # Try to initialize quantities using provided `Ellipsoid` objects.
            if (ctrs is None) and (covs is None):
                self.nells = len(ells)
                self.ells = ells
            else:
                raise ValueError("You cannot specific both `ells` and "
                                 "(`ctrs`, `covs`)!")
        else:
            # Try to initialize quantities using provided `ctrs` and `covs`.
            if (ctrs is None) and (covs is None):
                raise ValueError("You must specify either `ells` or "
                                 "(`ctrs`, `covs`).")
            else:
                self.nells = len(ctrs)
                self.ells = [
                    Ellipsoid(ctrs[i], covs[i]) for i in range(self.nells)
                ]
        self.__update_arrays()

        # Compute quantities.
        xp = array_namespace(self.ctrs)
        self.expands = xp.ones(self.nells)
        scipy = scipy_namespace_for(xp)
        self.logvol_tot = scipy.special.logsumexp(self.logvols)
        self.expand_tot = 1.
        self.funit = 1

    def __update_arrays(self):
        """
        Update internal arrays to ensure that in sync with ells
        """
        xp = array_namespace(*(ell.ctr for ell in self.ells))
        self.ctrs = xp.array([ell.ctr for ell in self.ells])
        self.covs = xp.array([ell.cov for ell in self.ells])
        self.ams = xp.array([ell.am for ell in self.ells])
        self.logvols = xp.array([ell.logvol for ell in self.ells])

    def scale_to_logvol(self, logvols):
        """Scale ellipoids to a corresponding set of
        target volumes.
        """
        for i in range(self.nells):
            self.ells[i].scale_to_logvol(logvols[i])

        # IMPORTANT We must also update arrays ams, covs
        self.__update_arrays()

        self.expands = np.array(
            [self.ells[i].expand for i in range(self.nells)])
        logvol_tot = logsumexp(logvols)
        self.expand_tot *= np.exp(logvol_tot - self.logvol_tot)
        self.logvol_tot = logvol_tot

    def major_axis_endpoints(self):
        """Return the endpoints of the major axis of each ellipsoid."""
        xp = array_namespace(self.ctrs)
        return xp.array([ell.major_axis_endpoints() for ell in self.ells])

    def within(self, x, j=None):
        """Checks which ellipsoid(s) `x` falls within, skipping the `j`-th
        ellipsoid if need be."""

        delt = x[None, :] - self.ctrs
        xp = array_namespace(delt)
        mask = xp.einsum('ai,aij,aj->a', delt, self.ams, delt) < 1
        if j is not None:
            mask[j] = False
        return xp.nonzero(mask)[0]

    def overlap(self, x, j=None):
        """Checks how many ellipsoid(s) `x` falls within, skipping the `j`-th
        ellipsoid."""

        q = len(self.within(x, j=j))

        return q

    def contains(self, x):
        """Checks if the set of ellipsoids contains `x`."""
        delt = x[None, :] - self.ctrs
        xp = array_namespace(delt)
        return xp.any(xp.einsum('ai,aij,aj->a', delt, self.ams, delt) < 1)

    def sample(self, rstate, return_q=False):
        """
        Sample a point uniformly distributed within the *union* of ellipsoids.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the set of ellipsoids.

        idx : int
            The index of the ellipsoid `x` was sampled from.

        q : int, optional
            The number of ellipsoids `x` falls within.

        """
        xp = array_namespace(self.ctrs)

        # If there is only one ellipsoid, sample from it.
        if self.nells == 1:
            x = self.ells[0].sample(rstate=rstate)
            idx = 0
            q = 1
            if return_q:
                return x, idx, q
            else:
                return x, idx

        probs = xp.exp(self.logvols - self.logvol_tot)
        while True:
            # Select an ellipsoid at random proportional to its volume.
            idx = rand_choice(probs, rstate)

            # Select a point from the chosen ellipsoid.
            x = self.ells[idx].sample(rstate=rstate)

            # Check how many ellipsoids the point lies within
            delts = (x[None, :] - self.ctrs)
            ell_masks = xp.einsum('ai,aij,aj->a', delts, self.ams, delts)
            q = xp.max((ell_masks < 1).sum(), 1)

            if return_q:
                # If `q` is being returned, assume the user wants to
                # explicitly apply the `1. / q` acceptance criterion to
                # properly sample from the union of ellipsoids.
                return x, idx, q
            else:
                # If `q` is not being returned, assume the user wants this
                # done internally so we repeat the loop if needed
                # random is faster than uniform
                if q == 1 or rstate.random() < (1. / q):
                    return x, idx

    def samples(self, nsamples, rstate):
        """
        Draw `nsamples` samples uniformly distributed within the *union* of
        ellipsoids.

        Returns
        -------
        xs : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the set of ellipsoids.

        """
        xp = array_namespace(self.ctrs)
        xs = xp.array([self.sample(rstate=rstate)[0] for i in range(nsamples)])

        return xs

    def monte_carlo_logvol(self,
                           ndraws=10000,
                           rstate=None,
                           return_overlap=True):
        """Using `ndraws` Monte Carlo draws, estimate the log volume of the
        *union* of ellipsoids. If `return_overlap=True`, also returns the
        estimated fractional overlap with the unit cube."""

        # Estimate volume using Monte Carlo integration.
        xp = array_namespace(self.ctrs)
        samples = [
            self.sample(rstate=rstate, return_q=True) for i in range(ndraws)
        ]
        qsum = sum((1. / q for (x, idx, q) in samples))
        logvol = xp.log(qsum / ndraws) + self.logvol_tot

        if return_overlap:
            # Estimate the fractional amount of overlap with the
            # unit cube using the same set of samples.
            qin = sum((1. / q * unitcheck(x) for (x, idx, q) in samples))
            overlap = qin / qsum
            return logvol, overlap
        else:
            return logvol

    def update(self,
               points,
               rstate,
               bootstrap=0,
               pool=None,
               mc_integrate=False):
        """
        Update the set of ellipsoids to bound the collection of points.

        Parameters
        ----------
        points : `~numpy.ndarray` with shape (npoints, ndim)
            The set of points to bound.

        rstate : `~numpy.random.Generator`, optional
            `~numpy.random.Generator` instance.

        bootstrap : int, optional
            The number of bootstrapped realizations of the ellipsoids. The
            maximum distance to the set of points "left out" during each
            iteration is used to enlarge the resulting volumes.
            Default is `0`.

        pool : user-provided pool, optional
            Use this pool of workers to execute operations in parallel.

        mc_integrate : bool, optional
            Whether to use Monte Carlo methods to compute the effective
            volume and fractional overlap of the final union of ellipsoids
            with the unit cube. Default is `False`.

        """
        xp = array_namespace(points)
        scipy = scipy_namespace_for(xp)

        npoints, ndim = points.shape
        if npoints == 1:
            raise RuntimeError('Cannot compute the bounding ellipsoid of '
                               'a single point.')
        LOG10_EXPAND_VOL_WARN = 2
        # maximum volume enhancement from bootstrap
        # Calculate the bounding ellipsoid for the points, possibly
        # enlarged to a minimum volume.
        firstell = bounding_ellipsoid(points)

        # Recursively split the bounding ellipsoid
        ells = _bounding_ellipsoids(points, firstell)

        # Update the set of ellipsoids.
        self.nells = len(ells)
        self.ells = ells
        self.__update_arrays()
        # Sanity check: all points must be contained in some ellipsoid
        if not all(self.contains(p) for p in points):
            # refuse to update
            raise RuntimeError('Rejecting invalid MultiEllipsoid region')
        self.logvol_tot = scipy.special.logsumexp(self.logvols)

        # Compute expansion factor.
        expands = xp.array([ell.expand for ell in self.ells])
        logvols_orig = self.logvols - xp.log(expands)
        logvol_tot_orig = scipy.special.logsumexp(logvols_orig)
        self.expand_tot = xp.exp(self.logvol_tot - logvol_tot_orig)

        # Use bootstrapping to determine the volume expansion factor.
        if bootstrap > 0:

            # If provided, compute bootstraps in parallel using a pool.
            if pool is None:
                M = map
            else:
                M = pool.map
            multis = [True for it in range(bootstrap)]
            ps = [points for it in range(bootstrap)]
            seeds = get_seed_sequence(rstate, bootstrap)
            args = zip(multis, ps, seeds)
            expands = list(M(_ellipsoid_bootstrap_expand, args))

            # Conservatively set the expansion factor to be the maximum
            # factor derived from our set of bootstraps.
            expand = max(expands)
            # Put a warning if a boostrap leads to 100 times larger volume
            if xp.log10(expand) * firstell.n > LOG10_EXPAND_VOL_WARN:
                warnings.warn(
                    'The enlargement factor for the ellipsoidal bounds'
                    ' determined'
                    ' from boostrapping is very large. If you are using'
                    ' uniform sampling that may mean that the sampling'
                    ' will be inefficient. This may be caused by a very'
                    ' complex posterior shape. You may consider using more'
                    ' liveponts or different sampler (i.e. rslice or rwalk)'
                    ' or alternatively disable bootstrap (bootstrap=0)')
            # If our ellipsoids are overly constrained, expand them.
            if expand > 1.:
                lvs = self.logvols + ndim * xp.log(expand)
                self.scale_to_logvol(lvs)

        # Estimate the volume and fractional overlap with the unit cube
        # using Monte Carlo integration.
        if mc_integrate:
            self.logvol_tot, self.funit = self.monte_carlo_logvol(
                rstate=rstate, return_overlap=True)


class RadFriends:
    """
    A collection of N-balls of identical size centered on each live point.

    Parameters
    ----------
    ndim : int
        The number of dimensions of each ball.

    cov : `~numpy.ndarray` with shape `(ndim, ndim)`, optional
        Covariance structure (correlation and size) of each ball.

    """

    def __init__(self, ndim, cov=None, xp=np):
        scipy = scipy_namespace_for(cov)
        self.n = ndim

        if cov is None:
            cov = xp.identity(self.n)
        self.cov = cov
        self.am = xp.linalg.pinv(self.cov)
        self.axes = scipy.linalg.sqrtm(self.cov)
        self.axes_inv = xp.linalg.pinv(self.axes)

        detsign, detln = xp.linalg.slogdet(self.am)
        assert detsign > 0
        self.logvol_ball = logvol_prefactor(self.n) - 0.5 * detln
        self.expand = 1.
        self.funit = 1

    def scale_to_logvol(self, logvol):
        """Scale ball to encompass a target volume."""
        xp = array_namespace(logvol)

        f = xp.exp((logvol - self.logvol_ball) * (1.0 / self.n))
        # linear factor
        self.cov *= f**2
        self.am /= f**2
        self.axes *= f
        self.axes_inv /= f
        self.logvol_ball = logvol

    def within(self, x, ctrs):
        """Check which balls `x` falls within."""
        xp = array_namespace(x, ctrs)

        # Execute a brute-force search over all balls.
        idxs = xp.where(
            xp.linalg.norm(xp.dot(ctrs - x, self.axes_inv), axis=1) <= 1.)[0]

        return idxs

    def overlap(self, x, ctrs):
        """Check how many balls `x` falls within."""

        q = len(self.within(x, ctrs))

        return q

    def contains(self, x, ctrs):
        """Check if the set of balls contains `x`."""

        return self.overlap(x, ctrs) > 0

    def sample(self, ctrs, rstate, return_q=False):
        """
        Sample a point uniformly distributed within the *union* of balls.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the set of balls.

        q : int, optional
            The number of balls `x` falls within.

        """

        nctrs = len(ctrs)  # number of balls

        while True:
            ds = randsphere(self.n, rstate=rstate)
            dx = ds @ self.axes

            # If there is only one ball, sample from it.
            if nctrs == 1:
                q = 1
                x = ctrs[0] + dx
            else:
                # Select a ball at random.
                idx = rstate.integers(nctrs)
                x = ctrs[idx] + dx
                q = self.overlap(x, ctrs)
            # random is faster than uniform
            if q == 1 or return_q or rstate.random() < (1. / q):
                if return_q:
                    return x, q
                else:
                    return x

    def samples(self, nsamples, ctrs, rstate):
        """
        Draw `nsamples` samples uniformly distributed within the *union* of
        balls.

        Returns
        -------
        xs : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the set of balls.

        """
        xp = array_namespace(ctrs)

        xs = xp.array(
            [self.sample(ctrs, rstate=rstate) for _ in range(nsamples)])

        return xs

    def monte_carlo_logvol(self,
                           ctrs,
                           ndraws=10000,
                           rstate=None,
                           return_overlap=True):
        """Using `ndraws` Monte Carlo draws, estimate the log volume of the
        *union* of balls. If `return_overlap=True`, also returns the
        estimated fractional overlap with the unit cube."""
        xp = array_namespace(ctrs)

        # Estimate volume using Monte Carlo integration.
        samples = ([
            self.sample(ctrs, rstate=rstate, return_q=True)
            for _ in range(ndraws)
        ])
        qs = xp.array([_[1] for _ in samples])
        qsum = xp.sum(1. / qs)
        logvol = xp.log(1. / ndraws * qsum * len(ctrs)) + self.logvol_ball

        if return_overlap:
            # Estimate the fractional amount of overlap with the
            # unit cube using the same set of samples.
            qin = sum((1. / q * unitcheck(x) for (x, q) in samples))
            overlap = qin / qsum
            return logvol, overlap
        else:
            return logvol

    def update(self,
               points,
               rstate,
               bootstrap=0,
               pool=None,
               mc_integrate=False,
               use_clustering=True):
        """
        Update the radii of our balls.

        Parameters
        ----------
        points : `~numpy.ndarray` with shape (npoints, ndim)
            The set of points to bound.

        rstate : `~numpy.random.Generator`, optional
            `~numpy.random.Generator` instance.

        bootstrap : int, optional
            The number of bootstrapped realizations of the ellipsoids. The
            maximum distance to the set of points "left out" during each
            iteration is used to enlarge the resulting volumes.
            Default is `0`.

        pool : user-provided pool, optional
            Use this pool of workers to execute operations in parallel.

        mc_integrate : bool, optional
            Whether to use Monte Carlo methods to compute the effective
            volume and fractional overlap of the final union of balls
            with the unit cube. Default is `False`.

        use_clustering : bool, optional
            Whether to use clustering to avoid issues with widely-seperated
            modes. Default is `True`.

        """
        xp = array_namespace(points)
        scipy = scipy_namespace_for(xp)

        # If possible, compute bootstraps in parallel using a pool.
        if pool is None:
            M = map
        else:
            M = pool.map

        # Get new covariance.
        if use_clustering:
            self.cov = self._get_covariance_from_clusters(points)
        else:
            self.cov = self._get_covariance_from_all_points(points)
        self.am = xp.linalg.pinv(self.cov)
        self.axes = scipy.linalg.sqrtm(self.cov)
        self.axes_inv = xp.linalg.pinv(self.axes)

        # Decorrelate and re-scale points.
        points_t = xp.dot(points, self.axes_inv)

        if bootstrap == 0.:
            # Construct radius using leave-one-out if no bootstraps used.
            radii = _friends_leaveoneout_radius(points_t, 'balls')
        else:
            # Bootstrap radius using the set of live points.
            ps = [points_t for it in range(bootstrap)]
            ftypes = ['balls' for it in range(bootstrap)]
            seeds = get_seed_sequence(rstate, bootstrap)
            args = zip(ps, ftypes, seeds)
            radii = list(M(_friends_bootstrap_radius, args))

        # Conservatively set radius to be maximum of the set.
        rmax = max(radii)

        # Re-scale axes.
        self.cov *= rmax**2
        self.am /= rmax**2
        self.axes *= rmax
        self.axes_inv /= rmax

        # Compute volume.
        detsign, detln = xp.linalg.slogdet(self.am)
        assert detsign > 0
        self.logvol_ball = (logvol_prefactor(self.n) - 0.5 * detln)
        self.expand = 1.

        # Estimate the volume and fractional overlap with the unit cube
        # using Monte Carlo integration.
        if mc_integrate:
            self.funit = self.monte_carlo_logvol(points,
                                                 return_overlap=True,
                                                 rstate=rstate)[1]

    def _get_covariance_from_all_points(self, points):
        """Compute covariance using all points."""
        xp = array_namespace(points)
        return xp.cov(points, rowvar=False)

    def _get_covariance_from_clusters(self, points):
        """Compute covariance from re-centered clusters."""

        # Compute pairwise distances.
        distances = spatial.distance.pdist(points,
                                           metric='mahalanobis',
                                           VI=self.am)

        # Identify conglomerates of points by constructing a linkage matrix.
        linkages = cluster.hierarchy.single(distances)

        # Cut when linkage between clusters exceed the radius.
        clusteridxs = cluster.hierarchy.fcluster(linkages,
                                                 1.0,
                                                 criterion='distance')
        nclusters = np.max(clusteridxs)
        if nclusters == 1:
            return self._get_covariance_from_all_points(points)
        else:
            i = 0
            overlapped_points = np.empty_like(points)
            for idx in np.unique(clusteridxs):
                group_points = points[clusteridxs == idx, :]
                group_mean = group_points.mean(axis=0).reshape((1, -1))
                j = i + len(group_points)
                overlapped_points[i:j, :] = group_points - group_mean
                i = j
            return self._get_covariance_from_all_points(overlapped_points)


class SupFriends:
    """
    A collection of N-cubes of identical size centered on each live point.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the cube.

    cov : `~numpy.ndarray` with shape `(ndim, ndim)`, optional
        Covariance structure (correlation and size) of each cube.

    """

    def __init__(self, ndim, cov=None, xp=np):
        scipy = scipy_namespace_for(xp)
        self.n = ndim

        if cov is None:
            cov = xp.identity(self.n)
        self.cov = cov
        self.am = xp.linalg.pinv(self.cov)
        self.axes = scipy.linalg.sqrtm(self.cov)
        self.axes_inv = xp.linalg.pinv(self.axes)

        detsign, detln = xp.linalg.slogdet(self.am)
        assert detsign > 0
        self.logvol_cube = self.n * np.log(2.) - 0.5 * detln
        self.expand = 1.
        self.funit = 1

    def scale_to_logvol(self, logvol):
        """Scale cube to encompass a target volume."""
        xp = array_namespace(logvol)

        f = xp.exp((logvol - self.logvol_cube) * (1.0 / self.n))
        # linear factor
        self.cov *= f**2
        self.am /= f**2
        self.axes *= f
        self.axes_inv /= f
        self.logvol_cube = logvol

    def within(self, x, ctrs):
        """Checks which cubes `x` falls within."""
        xp = array_namespace(ctrs)

        # Execute a brute-force search over all cubes.
        idxs = xp.where(
            xp.max(xp.abs(xp.dot(ctrs - x, self.axes_inv)), axis=1) <= 1.)[0]

        return idxs

    def overlap(self, x, ctrs):
        """Checks how many cubes `x` falls within, skipping the `j`-th
        cube."""

        q = len(self.within(x, ctrs))

        return q

    def contains(self, x, ctrs):
        """Checks if the set of cubes contains `x`."""

        return self.overlap(x, ctrs) > 0

    def sample(self, ctrs, rstate, return_q=False):
        """
        Sample a point uniformly distributed within the *union* of cubes.

        Returns
        -------
        x : `~numpy.ndarray` with shape (ndim,)
            A coordinate within the set of cubes.

        q : int, optional
            The number of cubes `x` falls within.

        """

        nctrs = len(ctrs)  # number of cubes

        while True:
            ds = rstate.uniform(-1, 1, size=self.n)
            dx = ds @ self.axes
            # If there is only one cube, sample from it.
            if nctrs == 1:
                x = ctrs[0] + dx
                q = 1
            else:
                # Select a cube at random.
                idx = rstate.integers(nctrs)
                x = ctrs[idx] + dx
                # Check how many cubes the point lies within, passing over
                # the `idx`-th cube `x` was sampled from.
                q = self.overlap(x, ctrs)
            # random() is faster than uniform()
            if q == 1 or return_q or rstate.random() < (1. / q):
                if return_q:
                    return x, q
                else:
                    return x

    def samples(self, nsamples, ctrs, rstate):
        """
        Draw `nsamples` samples uniformly distributed within the *union* of
        cubes.

        Returns
        -------
        xs : `~numpy.ndarray` with shape (nsamples, ndim)
            A collection of coordinates within the set of cubes.

        """
        xp = array_namespace(ctrs)

        xs = xp.array(
            [self.sample(ctrs, rstate=rstate) for _ in range(nsamples)])

        return xs

    def monte_carlo_logvol(self,
                           ctrs,
                           ndraws=10000,
                           rstate=None,
                           return_overlap=True):
        """Using `ndraws` Monte Carlo draws, estimate the log volume of the
        *union* of cubes. If `return_overlap=True`, also returns the
        estimated fractional overlap with the unit cube."""
        xp = array_namespace(ctrs)

        # Estimate the volume using Monte Carlo integration.
        samples = [
            self.sample(ctrs, rstate=rstate, return_q=True)
            for i in range(ndraws)
        ]
        qsum = sum((1. / q for (x, q) in samples))
        logvol = xp.log(1. * qsum / ndraws * len(ctrs)) + self.logvol_cube

        if return_overlap:
            # Estimate the fractional overlap with the unit cube using
            # the same set of samples.
            qin = sum((1. / q * unitcheck(x) for (x, q) in samples))
            overlap = qin / qsum
            return logvol, overlap
        else:
            return logvol

    def update(self,
               points,
               rstate,
               bootstrap=0,
               pool=None,
               mc_integrate=False,
               use_clustering=True):
        """
        Update the half-side-lengths of our cubes.

        Parameters
        ----------
        points : `~numpy.ndarray` with shape (npoints, ndim)
            The set of points to bound.

        rstate : `~numpy.random.Generator`, optional
            `~numpy.random.Generator` instance.

        bootstrap : int, optional
            The number of bootstrapped realizations of the ellipsoids. The
            maximum distance to the set of points "left out" during each
            iteration is used to enlarge the resulting volumes.
            Default is `0`.

        pool : user-provided pool, optional
            Use this pool of workers to execute operations in parallel.

        mc_integrate : bool, optional
            Whether to use Monte Carlo methods to compute the effective
            volume and fractional overlap of the final union of cubes
            with the unit cube. Default is `False`.

        use_clustering : bool, optional
            Whether to use clustering to avoid issues with widely-seperated
            modes. Default is `True`.

        """
        xp = array_namespace(points)
        scipy = scipy_namespace_for(xp)

        # If possible, compute bootstraps in parallel using a pool.
        if pool is None:
            M = map
        else:
            M = pool.map

        # Get new covariance.
        if use_clustering:
            self.cov = self._get_covariance_from_clusters(points)
        else:
            self.cov = self._get_covariance_from_all_points(points)
        self.am = xp.linalg.pinv(self.cov)
        self.axes = scipy.linalg.sqrtm(self.cov)
        self.axes_inv = xp.linalg.pinv(self.axes)

        # Decorrelate and re-scale points.
        points_t = xp.dot(points, self.axes_inv)

        if bootstrap == 0.:
            # Construct radius using leave-one-out if no bootstraps used.
            hsides = _friends_leaveoneout_radius(points_t, 'cubes')
        else:
            # Bootstrap radius using the set of live points.
            ps = [points_t for it in range(bootstrap)]
            ftypes = ['cubes' for it in range(bootstrap)]
            seeds = get_seed_sequence(rstate, bootstrap)
            args = zip(ps, ftypes, seeds)
            hsides = list(M(_friends_bootstrap_radius, args))

        # Conservatively set half-side-length to be maximum of the set.
        hsmax = max(hsides)

        # Re-scale axes.
        self.cov *= hsmax**2
        self.am /= hsmax**2
        self.axes *= hsmax
        self.axes_inv /= hsmax

        detsign, detln = xp.linalg.slogdet(self.am)
        assert detsign > 0
        self.logvol_cube = (self.n * np.log(2.) - 0.5 * detln)
        self.expand = 1.

        # Estimate the volume and fractional overlap with the unit cube
        # using Monte Carlo integration.
        if mc_integrate:
            self.funit = self.monte_carlo_logvol(points,
                                                 return_overlap=True,
                                                 rstate=rstate)[1]

    def _get_covariance_from_all_points(self, points):
        """Compute covariance using all points."""
        xp = array_namespace(points)

        return xp.cov(points, rowvar=False)

    def _get_covariance_from_clusters(self, points):
        """Compute covariance from re-centered clusters."""

        # Compute pairwise distances.
        distances = spatial.distance.pdist(points,
                                           metric='mahalanobis',
                                           VI=self.am)

        # Identify conglomerates of points by constructing a linkage matrix.
        linkages = cluster.hierarchy.single(distances)

        # Cut when linkage between clusters exceed the radius.
        clusteridxs = cluster.hierarchy.fcluster(linkages,
                                                 1.0,
                                                 criterion='distance')
        nclusters = np.max(clusteridxs)
        if nclusters == 1:
            return self._get_covariance_from_all_points(points)
        else:
            i = 0
            overlapped_points = np.empty_like(points)
            for idx in np.unique(clusteridxs):
                group_points = points[clusteridxs == idx, :]
                group_mean = group_points.mean(axis=0).reshape((1, -1))
                j = i + len(group_points)
                overlapped_points[i:j, :] = group_points - group_mean
                i = j
            return self._get_covariance_from_all_points(overlapped_points)


##################
# HELPER FUNCTIONS
##################


def logvol_prefactor(n, p=2.):
    """
    Returns the ln(volume constant) for an `n`-dimensional sphere with an
    :math:`L^p` norm. The constant is defined as::

        lnf = n * ln(2.) + n * LogGamma(1./p + 1) - LogGamma(n/p + 1.)

    By default the `p=2.` norm is used (i.e. the standard Euclidean norm).

    """

    p *= 1.  # convert to float in case user inputs an integer
    lnf = (n * np.log(2.) + n * gammaln(1. / p + 1.) - gammaln(n / p + 1))

    return lnf


def randsphere(n, rstate, nsamples=None):
    """Draw a point uniformly within an `n`-dimensional unit sphere."""
    if nsamples is None:
        size = n
        nsamples = ()
    else:
        size = (n, nsamples)
    z = rstate.standard_normal(size=size)  # initial n-dim vector
    # notice I use random () instead of uniform
    # and standard_norm instead of normal as those are faster
    # as this is a time-critical function
    xp = array_namespace(z)
    xhat = z * (rstate.random(nsamples)**(1. / n) / xp.linalg.norm(z, axis=0)
                )  # scale
    return xhat


def rand_choice(pb, rstate):
    """ Optimized version of numpy's random.choice
    Return an index of a point selected with the probability pb
    The pb must sum to 1
    """
    xp = array_namespace(pb)
    p1 = xp.cumsum(pb)
    # random is faster than uniform
    xr = rstate.random()
    return min(xp.searchsorted(p1, xr), len(pb) - 1)


def improve_covar_mat(covar0, ntries=100, max_condition_number=1e12):
    """
    Given the covariance matrix improve it, if it is not invertable
    or eigen values are negative or condition number that is above the limit
    Returns:
    a tuple with three elements
    1) a boolean flag if a matrix is 'good', so it didn't need adjustments
    2) updated matrix
    3) its inverse
    """
    xp = array_namespace(covar0)
    ndim = covar0.shape[0]
    covar = xp.array(covar0)
    coeffmin = 1e-10
    # this will a starting point for the modification
    # of the form (1-coeff)*M + (coeff)*E
    # eig_mult = 10  # we want the condition number to be at least that much
    # smaller than the max_condition_number

    # here we are trying to check if we compute cholesky transformation
    # and all eigenvals > 0 and condition number is good
    # if the only problem are the eigenvalues we just increase the lowest ones
    # if we are getting linalg exceptions we use add diag matrices

    for trial in range(ntries):
        # failed = 0
        try:
            # Check if matrix is invertible.
            eigval, eigvec = xp.linalg.eigh(covar)
            # compute eigenvalues/vectors
            # maxval = eigval.max()
            # minval = eigval.min()
            axes = eigvec * eigval**0.5
            # # Check if eigen values are good
            # if xp.isfinite(eigval).all():
            #     if maxval <= 0:
            #         # no positive eigvalues
            #         # not much to fix
            #         failed = 2
            #     else:
            #         if minval < maxval / max_condition_number:
            #             # some eigen values are too small
            #             failed = 1
            #         else:
            #             axes = eigvec * eigval**.5
            #             break
            # else:
            #     # complete failure
            #     failed = 2
        except lalg.LinAlgError:
            # There is some kind of massive failure
            # we suppress the off-diagonal elements
            pass
            # failed = 2
        # if failed > 0:
        #     if failed == 1:
        #         eigval_fix = xp.maximum(
        #             eigval, eig_mult * maxval / max_condition_number)
        #         covar = (eigvec * eigval_fix) @ eigvec.T
        #     else:
        coeff = coeffmin * (1. / coeffmin)**(trial * 1. / (ntries - 1))
        # this starts at coeffmin when trial=0 and ends at 1
        # when trial == ntries-1
        covar = (1. - coeff) * covar + coeff * np.eye(ndim)

    # if failed > 0:
    #     warnings.warn("Failed to guarantee the ellipsoid axes will be "
    #                   "non-singular. Defaulting to a sphere.")
    #     covar = np.eye(ndim)  # default to identity
    #     am = covar.copy()
    #     axes = covar.copy()
    # else:
    #     # invert the matrix using eigen decomposition
    am = (eigvec * (1. / eigval)) @ eigvec.T
    good_mat = trial == 0
    # if True it means no adjustments were necessary
    return good_mat, covar, am, axes


def bounding_ellipsoid(points):
    """
    Calculate the bounding ellipsoid containing a collection of points.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        A set of coordinates.

    Returns
    -------
    ellipsoid : :class:`Ellipsoid`
        The bounding :class:`Ellipsoid` object.

    """
    xp = array_namespace(points)
    npoints, ndim = points.shape

    if npoints == 1:
        raise ValueError("Cannot compute a bounding ellipsoid of a "
                         "single point.")

    # Calculate covariance of points.
    ctr = xp.mean(points, axis=0)
    covar = xp.cov(points, rowvar=False)
    # delta = points - ctr

    # When ndim = 1, `np.cov` returns a 0-d array. Make it a 1x1 2-d array.
    if ndim == 1:
        covar = xp.atleast_2d(covar)

    # ROUND_DELTA = 1e-3
    # numerical experiments show that round off errors can reach large
    # values if the matrix is poorly conditioned
    # Note that likely the delta here must be related to maximum
    # condition number parameter in improve_covar_mat()
    #
    # one_minus_a_bit = 1. - ROUND_DELTA
    good_mat, covar, am, axes = improve_covar_mat(covar)

    # for i in range(2):
    #     # If the matrix needs improvement
    #     # we improve the matrix twice, first before rescaling
    #     # and second after rescaling. If matrix is okay, we do
    #     # the loop once
    #     good_mat, covar, am, axes = improve_covar_mat(covar)

    #     # Calculate expansion factor necessary to bound each point.
    #     # Points should obey `(x-v)^T A (x-v) <= 1`, so we calculate this for
    #     # each point and then scale A up or down to make the
    #     # "outermost" point obey `(x-v)^T A (x-v) = 1`.

    #     fmax = xp.einsum('ij,jk,ik->i', delta, am, delta).max()

    #     # Due to round-off errors, we actually scale the ellipsoid so the
    #     # outermost point obeys `(x-v)^T A (x-v) < 1 - (a bit) < 1`.
    #     # in the first iteration we just try to adjust the matrix
    #     # if it didn't work again, we bail out
    #     if i == 0 and fmax > one_minus_a_bit:
    #         mult = fmax / one_minus_a_bit
    #         # IMPORTANT that we need to update the cov, its inverse and axes
    #         # as those are used directly
    #         covar *= mult
    #         am /= mult
    #         axes *= xp.sqrt(mult)
    #     if i == 1 and fmax >= 1:
    #         raise RuntimeError(
    #             "Failed to initialize the ellipsoid to contain all the points")
    #     if good_mat:
    #         # I only need to run through the loop twice if the matrix
    #         # is problematic
    #         break
    # Initialize our ellipsoid with *safe* covariance matrix.
    ell = Ellipsoid(ctr, covar, am=am, axes=axes)

    return ell


def _bounding_ellipsoids(points, ell, scale=None):
    """
    Internal method used to compute a set of bounding ellipsoids when a
    bounding ellipsoid for the entire set has already been calculated.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        A set of coordinates.

    ell : Ellipsoid
        The bounding ellipsoid containing :data:`points`.

    Returns
    -------
    ells : list of :class:`Ellipsoid` objects
        List of :class:`Ellipsoid` objects used to bound the
        collection of points. Used to initialize the :class:`MultiEllipsoid`
        object returned in :meth:`bounding_ellipsoids`.

    """
    xp = array_namespace(points)
    npoints, ndim = points.shape

    # We do not allow clusters with less than 2*ndim points,
    # as the bounding ellipsoid
    # will be poorly-constrained. Reject the split and simply return the
    # original ellipsoid bounding all the points.
    min_size = 2 * ndim
    if npoints < min_size * 2:
        # if we have less then min_size*2 pts, it's pointless to
        # even run clustering
        return [ell]

    # Starting cluster centers are initialized using the major-axis
    # endpoints of the original bounding ellipsoid.
    p1, p2 = ell.major_axis_endpoints()
    start_ctrs = xp.vstack((p1, p2))  # shape is (k, ndim) = (2, ndim)

    if scale is None:
        scale = points.std(axis=0)[None, :]
        # scale factor across different dimensions
        # to make things more isothropic
    # Split points into two clusters using k-means clustering with k=2.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k2_res = kmeans2(points / scale,
                         k=start_ctrs / scale,
                         iter=10,
                         minit='matrix',
                         check_finite=False)
    labels = k2_res[1]  # cluster identifier ; shape is (npoints,)

    # Get points in each cluster.
    points_k = [points[labels == k, :] for k in (0, 1)]

    # if the smallest cluster is too small refuse
    if min(points_k[0].shape[0], points_k[1].shape[0]) < min_size:
        return [ell]

    # Bounding ellipsoid for each cluster
    ells = [bounding_ellipsoid(points_j) for points_j in points_k]

    # If the total volume decreased significantly, we accept
    # the split into subsets. We then recursively split each subset.
    # The condition for the volume decrease is motivated by the BIC values
    # assuming that the number of parameter of the ellipsoid is X (it is
    # Ndim*(Ndim+3)/2, the number of points is N
    # then the BIC of the k bounding ellipsoid model is
    # 2 * N * log(V0) + k * X * ln(N)
    # where V0 is the total volume of the ellipsoids
    # then the condition for bic improvement is for k1 ellipsoids vs k0
    # 2 * N * log(V1) + k1 * X * ln (N) <  2 * N * log(V0) + k0 * X * ln (N)
    # log(V1) - log(V0) < (k0-k1) * X * ln(N)/2/N
    # The choice of BIC is motivated by Xmeans algo from Pelleg 2000
    # See also Feroz2008

    nparam = (ndim * (ndim + 3)) // 2
    log_vol_dec = nparam * np.log(npoints) / npoints
    # this is the log vol decrement for one extra ellipsoid
    # note that this is missing a factor of two to mimick previous behaviour
    # this makes splitting less agressive

    # now we try to split again
    out_ells = (_bounding_ellipsoids(points_k[0], ells[0], scale=scale) +
                _bounding_ellipsoids(points_k[1], ells[1], scale=scale))

    # if the first volume test was successful we accept the results
    if (xp.logaddexp(ells[0].logvol, ells[1].logvol) -
            ell.logvol) < -log_vol_dec:
        return out_ells

    # if it was not we check again if the volume decreased significantly
    # after the recursion
    if ((logsumexp([e.logvol for e in out_ells]) - ell.logvol)
            < -log_vol_dec * (len(out_ells) - 1)):
        return out_ells

    # Otherwise, we are happy with the single bounding ellipsoid.
    return [ell]


def bounding_ellipsoids(points):
    """
    Calculate a set of ellipsoids that bound the collection of points.

    Parameters
    ----------
    points : `~numpy.ndarray` with shape (npoints, ndim)
        A set of coordinates.

    Returns
    -------
    mell : :class:`MultiEllipsoid` object
        The :class:`MultiEllipsoid` object used to bound the
        collection of points.

    """

    # Calculate the bounding ellipsoid for the points possibly
    # enlarged to a minimum volume.
    ell = bounding_ellipsoid(points)

    # Recursively split the bounding ellipsoid
    ells = _bounding_ellipsoids(points, ell)

    return MultiEllipsoid(ells=ells)


def _bootstrap_points(points, rseed):
    """
    Select the bootstrap set from points.
    Return:
    Tuple with selected, and not-selected points
    """
    rstate = get_random_generator(rseed)
    npoints = points.shape[0]
    xp = array_namespace(points)

    # Resampling.
    idxs = rstate.integers(npoints, size=npoints)
    idx_in = xp.unique(idxs)  # selected objects
    sel_in = xp.zeros(npoints, dtype=bool)
    sel_in = sel_in.at[idx_in].set(True)
    # in the crazy case of not having selected more than one
    # point I just arbitrary add points to have at least two in idx_in
    # and at least 1 in idx_out
    n_in = sel_in.sum()
    if n_in < 2:
        sel_in = sel_in.at[:2].set(True)
    if n_in > npoints - 1:
        sel_in = sel_in.at[0].set(False)
    points_in, points_out = points[sel_in], points[~sel_in]
    return points_in, points_out


def _ellipsoid_bootstrap_expand(args):
    """Internal method used to compute the expansion factor for a bounding
    ellipsoid or ellipsoids based on bootstrapping.
    The argument is a tuple:
    multi: boolean flag if we are doing multiell or single ell decomposition
    points: 2d array of points
    rseed: seed to initialize the random generator
    """

    # Unzipping.
    multi, points, rseed = args
    xp = array_namespace(points)

    points_in, points_out = _bootstrap_points(points, rseed)

    # Compute bounding ellipsoid.
    ell = bounding_ellipsoid(points_in)

    if not multi:
        # Compute normalized distances to missing points.
        dists = ell.distance_many(points_out)
    else:
        ells = _bounding_ellipsoids(points_in, ell)
        # Compute normalized distances to missing points.
        dists = xp.min(xp.array([el.distance_many(points_out) for el in ells]),
                       axis=0)

    # Compute expansion factor.
    expand = max(1., xp.max(dists))

    return expand


def _friends_bootstrap_radius(args):
    """Internal method used to compute the radius (half-side-length) for each
    ball (cube) used in :class:`RadFriends` (:class:`SupFriends`) using
    bootstrapping."""

    # Unzipping.
    points, ftype, rseed = args

    points_in, points_out = _bootstrap_points(points, rseed)

    # Construct KDTree to enable quick nearest-neighbor lookup for
    # our resampled objects.
    kdtree = spatial.KDTree(points_in)

    if ftype == 'balls':
        # Compute distances from our "missing" points its closest neighbor
        # among the resampled points using the Euclidean norm
        # (i.e. "radius" of n-sphere).
        dists = kdtree.query(points_out, k=1, eps=0, p=2)[0]
    elif ftype == 'cubes':
        # Compute distances from our "missing" points its closest neighbor
        # among the resampled points using the Euclidean norm
        # (i.e. "half-side-length" of n-cube).
        dists = kdtree.query(points_out, k=1, eps=0, p=np.inf)[0]

    # Conservative upper-bound on radius.
    dist = max(dists)

    return dist


def _friends_leaveoneout_radius(points, ftype):
    """Internal method used to compute the radius (half-side-length) for each
    ball (cube) used in :class:`RadFriends` (:class:`SupFriends`) using
    leave-one-out (LOO) cross-validation."""

    # Construct KDTree to enable quick nearest-neighbor lookup for
    # our resampled objects.
    kdtree = spatial.KDTree(points)

    if ftype == 'balls':
        # Compute radius to two nearest neighbors (self + neighbor).
        dists = kdtree.query(points, k=2, eps=0, p=2)[0]
    elif ftype == 'cubes':
        # Compute half-side-length to two nearest neighbors (self + neighbor).
        dists = kdtree.query(points, k=2, eps=0, p=np.inf)[0]

    dist = dists[:, 1]  # distances to LOO nearest neighbor

    return dist
