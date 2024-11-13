#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
The base `Sampler` class containing various helpful functions. All other
samplers inherit this class either explicitly or implicitly.


neff = \frac{(\sum w)^2}{\sum w^2}

"""

import sys
import warnings
import math
import copy
from dataclasses import dataclass
from functools import partial
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
from dynesty.utils import (
    IteratorResult,
    get_print_func,
    _LOWL_VAL,
    print_fn,
    DelayTimer,
)

from .bounding import UnitCube
from .sampling import sample_unif
from .utils import (progress_integration,
                    RunRecord,
                    compute_integrals,
                    distance_insertion_index, likelihood_insertion_index,
                    Results, JAXGenerator, LoglOutput,
                    )


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["u", "v", "logl", "nc","blob"],
    meta_fields=[],
)
@dataclass
class Point:
    u: jax.Array
    v: jax.Array
    logl: jax.Array
    nc: jax.Array
    blob: dict | None


__all__ = ["Sampler"]


class Sampler:
    """
    The basic sampler object that performs the actual nested sampling.

    Parameters
    ----------
    loglikelihood : function
        Function returning ln(likelihood) given parameters as a 1-d `~numpy`
        array of length `ndim`.

    prior_transform : function
        Function transforming a sample from the a unit cube to the parameter
        space of interest according to the prior.

    npdim : int, optional
        Number of parameters accepted by `prior_transform`.

    live_points : list of 3 `~numpy.ndarray` each with shape (nlive, ndim)
        Initial set of "live" points. Contains `live_u`, the coordinates
        on the unit cube, `live_v`, the transformed variables, and
        `live_logl`, the associated loglikelihoods.

    update_interval : int
        Only update the bounding distribution every `update_interval`-th
        likelihood call.

    first_update : dict
        A dictionary containing parameters governing when the sampler should
        first update the bounding distribution from the unit cube to the one
        specified by the user.

    rstate : `~numpy.random.Generator`
        `~numpy.random.Generator` instance.

    queue_size: int
        Carry out likelihood evaluations in parallel by queueing up new live
        point proposals using (at most) this many threads/members.

    """

    def __init__(self,
                 loglikelihood,
                 prior_transform,
                 npdim,
                 live_points,
                 update_interval,
                 first_update,
                 rstate,
                 queue_size,
                 ncdim,
                 logvol_init=0,
                 blob=False):

        # distributions
        self.loglikelihood = loglikelihood
        self.prior_transform = prior_transform
        self.npdim = npdim
        self.ncdim = ncdim
        self.blob = blob

        self.live_u, self.live_v, self.live_logl = live_points[:3]
        self.live_u = jnp.array(self.live_u)
        self.live_v = jnp.array(self.live_v)
        self.live_logl = jnp.array(self.live_logl)
        if blob:
            self.live_blobs = live_points[3]
        else:
            self.live_blobs = None
        self.nlive = len(self.live_u)
        self.live_bound = jnp.zeros(self.nlive, dtype=int)
        self.live_it = jnp.zeros(self.nlive, dtype=int)

        # random state
        if isinstance(rstate, np.random.Generator):
            rstate = JAXGenerator(jax.random.PRNGKey(rstate.integers(10000)))
        self.rstate = rstate

        # set to none just for qa
        self.scale = None
        self.distance_insertion_index = -1
        self.likelihood_insertion_index = -1
        self.method = None
        self.kwargs = {}

        self.queue = []  # proposed live point queue
        self.nqueue = 0  # current size of the queue
        self.unused = 0  # total number of proposals unused
        self.used = 0  # total number of proposals used

        # sampling
        self.it = 1  # current iteration
        self.ncall = self.nlive  # number of function calls
        self.dlv = math.log((self.nlive + 1.) / self.nlive)  # shrinkage/iter
        self.added_live = False  # whether leftover live points were used
        self.eff = 0.  # overall sampling efficiency
        self.cite = ''  # Default empty
        self.save_samples = True
        self.save_bounds = True

        # bounding updates
        self.bound_update_interval = update_interval
        self.first_bound_update_ncall = first_update.get(
            'min_ncall', 2 * self.nlive)
        self.first_bound_update_eff = first_update.get('min_eff', 10.)
        self.logl_first_update = None
        self.unit_cube_sampling = True
        self.bound = [UnitCube(self.ncdim)]  # bounding distributions
        self.nbound = 1  # total number of unique bounding distributions
        self.ncall_at_last_update = 0

        self.logvol_init = logvol_init

        self.plateau_mode = False
        self.plateau_counter = None
        self.plateau_logdvol = None
        self.cumulative_ln_weight = -np.inf
        self.cumulative_ln_weight_squared = -np.inf
        # results
        # self.saved_run = RunRecord().D
        self.saved_run = RunRecord()

    def save(self, fname):
        raise RuntimeError('Should be overriden')

    def propose_point(self, *args):
        raise RuntimeError('Should be overriden')

    def evolve_point(self, *args):
        raise RuntimeError('Should be overriden')

    def update_proposal(self, *args, **kwargs):
        raise RuntimeError('Should be overriden')

    def update(self, subset=None):
        raise RuntimeError('Should be overriden')

    def reset(self):
        """Re-initialize the sampler."""

        # live points
        self.live_u = self.rstate.random(size=(self.nlive, self.npdim))
        self.live_v = self.prior_transform(self.live_u.T).T
        self.live_logl = jax.vmap(lambda x: self.loglikelihood(x).val)(self.live_v)

        self.live_bound = jnp.zeros(self.nlive, dtype=int)
        self.live_it = jnp.zeros(self.nlive, dtype=int)

        # parallelism
        self.queue = []
        self.nqueue = 0
        self.unused = 0
        self.used = 0

        # sampling
        self.it = 1
        self.ncall = self.nlive
        self.bound = [UnitCube(self.ncdim)]
        self.nbound = 1
        self.unit_cube_sampling = False
        self.added_live = False

        self.plateau_mode = False
        self.plateau_counter = None
        self.plateau_logdvol = None

        # results
        self.saved_run = RunRecord()

    @property
    def results(self):
        """Saved results from the nested sampling run. If bounding
        distributions were saved, those are also returned."""

        d = {}
        for k in [
                'nc', 'v', 'id', 'it', 'u', 'logwt', 'logl', 'logvol', 'logz',
                'logzvar', 'h', 'blob'
        ]:
            d[k] = np.array(self.saved_run[k])

        # Add all saved samples to the results.
        if self.save_samples:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = [('nlive', self.nlive), ('niter', self.it - 1),
                           ('ncall', d['nc'].astype(jnp.int32)), ('eff', self.eff),
                           ('samples', d['v']), ('blob', d['blob'])]
                for k in ['id', 'it', 'u']:
                    results.append(('samples_' + k, d[k]))
                for k in ['logwt', 'logl', 'logvol', 'logz']:
                    results.append((k, d[k]))
                results.append(('logzerr', np.sqrt(d['logzvar'])))
                results.append(('information', d['h']))
        else:
            raise ValueError("You didn't save any samples!")

        # Add any saved bounds (and ancillary quantities) to the results.
        if self.save_bounds:
            results.append(('bound', copy.deepcopy(self.bound)))
            results.append(
                ('bound_iter', np.array(self.saved_run['bounditer'],
                                        dtype=int)))
            results.append(('samples_bound',
                            np.array(self.saved_run['boundidx'], dtype=int)))
        for key in ['scale', 'distance_insertion_index', 'likelihood_insertion_index']:
            results.append((key, np.array(self.saved_run[key])))

        return Results(results)

    @property
    def n_effective(self):
        """
        Estimate the effective number of posterior samples using the Kish
        Effective Sample Size (ESS) where `ESS = sum(wts)^2 / sum(wts^2)`.
        Note that this is `len(wts)` when `wts` are uniform and
        `1` if there is only one non-zero element in `wts`.

        """
        return jnp.nan_to_num(jnp.exp(
            2 * self.cumulative_ln_weight - self.cumulative_ln_weight_squared
        ))

    @property
    def citations(self):
        """
        Return list of papers that should be cited given the specified
        configuration of the sampler.

        """

        return self.cite

    def update_bound_if_needed(self, loglstar, ncall=None, force=False):
        """
        Here we update the bound depending on the situation
        The arguments are the loglstar and number of calls
        if force is true we update the bound no matter what
        """

        if ncall is None:
            ncall = self.ncall
        call_check_first = (ncall >= self.first_bound_update_ncall)
        call_check = (ncall >= self.bound_update_interval +
                      self.ncall_at_last_update)
        efficiency_check = (self.eff < self.first_bound_update_eff)
        # there are three cases when we update the bound
        # * if we are still using uniform cube sampling and both efficiency is
        # lower than the threshold and the number of calls is larger than the
        # threshold
        # * if we are sampling from uniform cube and loglstar is larger than
        # the previously saved logl_first_update
        # * if we are not uniformly cube sampling and the ncall is larger
        # than the ncall of the previous update by the update_interval
        # * we are forced
        if ((self.unit_cube_sampling and efficiency_check and call_check_first)
                or (not self.unit_cube_sampling and call_check) or
            (self.unit_cube_sampling and self.logl_first_update is not None
             and loglstar > self.logl_first_update)) or force:
            if loglstar == _LOWL_VAL:
                # in the case we just started and we have some
                # LOWL_VAL points we don't want to use them for the
                # boundary
                subset = self.live_logl > loglstar
            else:
                subset = slice(None)
            bound = self.update(subset=subset)
            if self.save_bounds:
                self.bound.append(bound)
            self.nbound += 1
            self.ncall_at_last_update = ncall
            if self.unit_cube_sampling:
                self.unit_cube_sampling = False
                self.logl_first_update = loglstar

    def _fill_queue(self, loglstar):
        """Sequentially add new live point proposals to the queue."""

        # All the samplers should have have a starting point
        # satisfying a strict logl>loglstar criterion
        # The slice sampler will just fail if it's not the case
        # therefore we provide those subsets of points to choose from.
        if self.method != 'unif':
            args = (jnp.nonzero(self.live_logl > loglstar)[0], )
            if len(args[0]) == 0:
                raise RuntimeError(
                    'No live points are above loglstar. '
                    'Do you have a likelihood plateau ? '
                    'It is also possible that you are trying to sample '
                    'excessively around the very peak of the posterior')
        else:
            args = ()
        if not self.unit_cube_sampling:
            evolve_point = self.evolve_point
        else:
            evolve_point = sample_unif
        point_queue = self.live_u.copy()

        for key, value in self.kwargs.items():
            if isinstance(value, (np.ndarray, list)):
                self.kwargs[key] = jnp.asarray(value)
        kwargs = {
            k: v for k, v in self.kwargs.copy().items()
            if isinstance(v, (float, int, jax.Array)) or v is None
        }
        ptform = self.prior_transform
        lnl = self.loglikelihood
        u, v, logl, nc, blob = evolve_point(
            self.live_u, self.rstate.key, ptform, lnl, loglstar, self.scale, kwargs
        )
        if blob is None:
            blob = [None] * len(u)
        else:
            blob = [
                {k: v for k, v in zip(blob.keys(), values)}
                for values in zip(*[value for value in blob.values()])
            ]
        self.queue = list(zip(u, v, logl, nc, blob))
        keys = ["u", "v", "logl", "nc", "blob"]
        self.queue = [{k: v for k, v in zip(keys, vals)} for vals in self.queue]
        self.nqueue = len(self.queue)
        self._add_insertion_indices_to_queue(point_queue)

    def _add_insertion_indices_to_queue(self, point_queue):
        for start, entry in zip(point_queue, self.queue):
            blob = entry["blob"]
            if blob is not None:
                blob["distance_insertion"] = distance_insertion_index(self.live_u, start, entry["u"])
                blob["likelihood_insertion"] = likelihood_insertion_index(self.live_logl, entry["logl"])
        self.queue = [Point(**point) for point in self.queue]

    def _distance_insertion_index(self, start, point):
        """
        Compute the distance insertion index as defined in XXX
        """
        norms = jnp.std(self.live_u, axis=0)
        distance = jnp.linalg.norm((point - start) / norms)
        all_distances = jnp.array([jnp.linalg.norm((start - u) / norms) for u in self.live_u])
        return jnp.sum(all_distances < distance)

    def _likelihood_insertion_index(self, logl):
        """
        Compute the likelihood insertion index as defined in arxiv:2006.03371
        """
        return jnp.sum(jnp.array(self.live_logl) < logl)

    def _get_point_value(self, loglstar):
        """Grab the first live point proposal in the queue."""

        # If the queue is empty, refill it.
        if self.nqueue <= 0:
            self._fill_queue(loglstar)

        # Grab the earliest entry.
        # u, v, logl, nc, blob = self.queue.pop(0)
        entry = self.queue.pop(0)
        self.used += 1  # add to the total number of used points
        self.nqueue -= 1

        # return u, v, logl, nc, blob
        return entry

    def _new_point(self, loglstar):
        """Propose points until a new point that satisfies the log-likelihood
        constraint `loglstar` is found."""

        ncall = self.ncall
        ncall_accum = 0
        while True:
            # Get the next point from the queue
            # u, v, logl, nc, blob = self._get_point_value(loglstar)
            entry = self._get_point_value(loglstar)
            u = entry["u"]
            v = entry["v"]
            logl = entry["logl"]
            nc = entry["nc"]
            blob = entry["blob"]
            ncall += nc
            if blob is not None:
                distance_index = blob.get("distance_insertion", 0)
                likelihood_index = blob.get("likelihood_insertion", 0)
            ncall_accum += nc

            if blob is not None and not self.unit_cube_sampling:
                # If our queue is empty, update any tuning parameters
                # associated
                # with our proposal (sampling) method.
                # If it's not empty we are just accumulating the
                # the history of evaluations
                self.update_proposal(blob, update=self.nqueue <= 0)

            # the reason I'm not using self.ncall is that it's updated at
            # higher level
            # also on purpose this is placed in nqueue==0
            # because we only want update if we are planning to generate
            # new points
            if self.nqueue == 0 and self.unit_cube_sampling:
                self.update_bound_if_needed(loglstar, ncall=ncall)

            # If we satisfy the log-likelihood constraint, we're done!
            if logl > loglstar:
                break

        return u, v, logl, ncall_accum, distance_index, likelihood_index

    def add_live_points(self):
        """Add the remaining set of live points to the current set of dead
        points. Instantiates a generator that will be called by
        the user. Returns the same outputs as :meth:`sample`."""

        # Check if the remaining live points have already been added
        # to the output set of samples.
        if self.added_live:
            raise ValueError("The remaining live points have already "
                             "been added to the list of samples!")
        else:
            self.added_live = True
        if len(self.saved_run['logz']) > 0:
            logz = self.saved_run['logz'][-1]
            logzvar = self.saved_run['logzvar'][-1]
            h = self.saved_run['h'][-1]
            loglstar = self.saved_run['logl'][-1]
            logvol = self.saved_run['logvol'][-1]
        else:
            # this is special case if we didn't do any running
            # just sampled uniformly and bailed out
            h = 0.  # information, initially *0.*
            logz = -1.e300  # ln(evidence), initially *0.*
            logzvar = 0.  # var[ln(evidence)], initially *0.*
            logvol = self.logvol_init
            # initially contains the whole prior (volume=1.)
            loglstar = -1.e300  # initial ln(likelihood)

        # After N samples have been taken out, the remaining volume is
        # `e^(-N / nlive)`. The remaining points are distributed uniformly
        # within the remaining volume so that the expected volume enclosed
        # by the `i`-th worst likelihood is
        # `e^(-N / nlive) * (nlive + 1 - i) / (nlive + 1)`.
        # The tricky bit here is what to do if we have a plateau that we
        # haven't fully exhausted
        # then we first use the old delta(V) till we are done with the plateau
        if not self.plateau_mode:
            logvols = jnp.log(1. - (jnp.arange(self.nlive) + 1.) /
                             (self.nlive + 1.))
            # Defining change in `logvol` used in `logzvar` approximation.
        else:
            # we first just use old delta(v)'s associated with each point
            # in the plateau
            logvols = jnp.log1p(-((1 + jnp.arange(self.plateau_counter)) *
                                 jnp.exp(self.plateau_logdvol - logvol)))
            # after we're done with it we just assign 1/(nrest+1) fraction of
            # the remaining volume to each leftover point
            nrest = self.nlive - self.plateau_counter
            logvols = jnp.concatenate([
                logvols,
                logvols[-1] + jnp.log1p(-(1 + jnp.arange(nrest)) / (nrest + 1))
            ])
        # IMPORTANT in those caclulations I keep logvol separate
        # and add it later to ensure the first dlv=0
        dlvs = -jnp.diff(logvols, prepend=0)
        logvols += logvol
        # Sorting remaining live points.
        lsort_idx = jnp.argsort(self.live_logl)
        loglmax = jnp.max(self.live_logl)

        # Grabbing relevant values from the last dead point.
        if not self.unit_cube_sampling:
            bounditer = self.nbound - 1
        else:
            bounditer = 0

        # Add contributions from the remaining live points in order
        # from the lowest to the highest log-likelihoods.
        to_add = list()
        for i in range(self.nlive):

            # Grab live point with `i`-th lowest log-likelihood along with
            # ancillary quantities.
            idx = lsort_idx[i]
            logvol, dlv = logvols[i], dlvs[i]
            # we are doing copies here, because live_u/live_v are
            # updated in place
            ustar = self.live_u[idx].copy()
            vstar = self.live_v[idx].copy()
            if self.blob:
                old_blob = self.live_blobs[idx].copy()
            else:
                old_blob = None
            loglstar_new = self.live_logl[idx]
            boundidx = self.live_bound[idx]
            point_it = self.live_it[idx]

            (logwt, logz, logzvar,
             h) = progress_integration(loglstar, loglstar_new, logz, logzvar,
                                       logvol, dlv, h)
            loglstar = loglstar_new
            delta_logz = jnp.logaddexp(0, loglmax + logvol - logz)

            # Save results.
            if self.save_samples:
                to_add.append(
                    (
                        idx,
                        ustar,
                        vstar,
                        loglstar,
                        logvol,
                        logwt,
                        logz,
                        logzvar,
                        h,
                        1,  # this is technically a lie
                        # as we didn't call the likelihood even once
                        # however because we lose track of ncs if we start
                        # from points that are not sampled from unit cube
                        # it can lead to sum(nc)!=ncall
                        point_it,
                        1,
                        bounditer,
                        -1,
                        -1,
                        ))
            self.eff = 100. * (self.it + i) / self.ncall  # efficiency

            # Return our new "dead" point and ancillary quantities.
            ret = IteratorResult(worst=idx,
                                 ustar=ustar,
                                 vstar=vstar,
                                 loglstar=loglstar,
                                 logvol=logvol,
                                 logwt=logwt,
                                 logz=logz,
                                 logzvar=logzvar,
                                 h=h,
                                 nc=1,
                                 blob=old_blob,
                                 worst_it=point_it,
                                 boundidx=boundidx,
                                 bounditer=bounditer,
                                 eff=self.eff,
                                 delta_logz=delta_logz)
        
        to_add = (jax.numpy.array(arr) for arr in zip(*to_add))
        self.saved_run.D = update_saved_run(*to_add, self.saved_run.D)
        return [ret] * self.nlive

    def _remove_live_points(self):
        """Remove the final set of live points if they were
        previously added to the current set of dead points."""

        if self.added_live:
            self.added_live = False
            if self.save_samples:
                for k in [
                        'id', 'u', 'v', 'logl', 'logvol', 'logwt', 'logz',
                        'logzvar', 'h', 'nc', 'boundidx', 'it', 'bounditer',
                        'scale', 'distance_insertion_index', 'likelihood_insertion_index',
                        "n", "blob",
                ]:
                    self.saved_run[k] = self.saved_run[k][:-self.nlive]
        else:
            raise ValueError("No live points were added to the "
                             "list of samples!")

    def sample(self,
               maxiter=None,
               maxcall=None,
               dlogz=0.01,
               logl_max=np.inf,
               n_effective=np.inf,
               add_live=True,
               save_bounds=True,
               save_samples=True,
               resume=False):
        """
        **The main nested sampling loop.** Iteratively replace the worst live
        point with a sample drawn uniformly from the prior until the
        provided stopping criteria are reached. Instantiates a generator
        that will be called by the user.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations. Iteration may stop earlier if the
            termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations. Iteration may stop
            earlier if termination condition is reached. Default is
            `sys.maxsize` (no limit).

        dlogz : float, optional
            Iteration will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below
            this threshold. Ejnplicitly, the stopping criterion is
            `ln(z + z_est) - ln(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. Default is `0.01`.

        logl_max : float, optional
            Iteration will stop when the sampled ln(likelihood) exceeds the
            threshold set by `logl_max`. Default is no bound (`np.inf`).

        n_effective: int, optional
            Minimum number of effective posterior samples. If the estimated
            "effective sample size" (ESS) exceeds this number,
            sampling will terminate. Default is no ESS (`np.inf`).

        add_live : bool, optional
            Whether or not to add the remaining set of live points to
            the list of samples when calculating `n_effective`.
            Default is `True`.

        save_bounds : bool, optional
            Whether or not to save past distributions used to bound
            the live points internally. Default is `True`.

        save_samples : bool, optional
            Whether or not to save past samples from the nested sampling run
            (along with other ancillary quantities) internally.
            Default is `True`.

        Returns
        -------
        worst : int
            Index of the live point with the worst likelihood. This is our
            new dead point sample.

        ustar : `~numpy.ndarray` with shape (npdim,)
            Position of the sample.

        vstar : `~numpy.ndarray` with shape (ndim,)
            Transformed position of the sample.

        loglstar : float
            Ln(likelihood) of the sample.

        logvol : float
            Ln(prior volume) within the sample.

        logwt : float
            Ln(weight) of the sample.

        logz : float
            Cumulative ln(evidence) up to the sample (inclusive).

        logzvar : float
            Estimated cumulative variance on `logz` (inclusive).

        h : float
            Cumulative information up to the sample (inclusive).

        nc : int
            Number of likelihood calls performed before the new
            live point was accepted.

        worst_it : int
            Iteration when the live (now dead) point was originally proposed.

        boundidx : int
            Index of the bound the dead point was originally drawn from.

        bounditer : int
            Index of the bound being used at the current iteration.

        eff : float
            The cumulative sampling efficiency (in percent).

        delta_logz : float
            The estimated remaining evidence expressed as the ln(ratio) of the
            current evidence.

        """
        # Initialize quantities.
        if maxcall is None:
            maxcall = 2**31 - 1
        if maxiter is None:
            maxiter = 2**31 - 1
        self.save_samples = save_samples
        self.save_bounds = save_bounds
        ncall = 0
        # Check whether we're starting fresh or continuing a previous run.
        if self.it == 1 or len(self.saved_run['logl']) == 0:
            # Initialize values for nested sampling loop.
            h = 0.  # information, initially *0.*
            logz = -1.e300  # ln(evidence), initially *0.*
            logzvar = 0.  # var[ln(evidence)], initially *0.*
            logvol = self.logvol_init
            # initially contains the whole prior (volume=1.)
            loglstar = -1.e300  # initial ln(likelihood)
            delta_logz = 1.e300  # ln(ratio) of total/current evidence

        else:
            # Remove live points (if added) from previous run.
            if self.added_live and not resume:
                self._remove_live_points()

            # Get final state from previous run.
            h, logz, logzvar, logvol, loglstar = [
                self.saved_run[_][-1]
                for _ in ['h', 'logz', 'logzvar', 'logvol', 'logl']
            ]
            delta_logz = jnp.logaddexp(0, jnp.max(self.live_logl) + logvol - logz)

        stop_iterations = False
        # # The main nested sampling loop.
        for it in range(sys.maxsize):
            delta_logz = jnp.logaddexp(0, jnp.max(self.live_logl) + logvol - logz)

            stop_iterations = check_complete(
                it, maxiter, ncall, maxcall, dlogz, delta_logz,
                loglstar, logl_max, n_effective, self.n_effective,
                self.live_logl,
            )

            if stop_iterations:
                # if not self.save_samples:
                #     # If dumping past states, save only the required quantities
                #     # TODO I don't quite understand why we do this
                #     add_info = dict(logz=logz,
                #                     logzvar=logzvar,
                #                     h=h,
                #                     logvol=logvol,
                #                     logl=loglstar)
                #     self.saved_run = self.saved_run.append(add_info)
                break

            self._fill_queue(loglstar)
            to_add = list()
            nc = 0
            big_blob = dict(accept=0, reject=0)
            for point in self.queue:
                if point.blob is not None:
                    for key in big_blob:
                        big_blob[key] += point.blob.get(key, 0)
                loglstar_old = loglstar
                worst = jnp.argmin(self.live_logl)
                ustar = self.live_u[worst]
                vstar = self.live_v[worst]
                worst_it = self.live_it[worst]
                bounditer = jax.lax.select(
                    self.unit_cube_sampling,
                    0,
                    self.nbound - 1
                )
                self.ncall += point.nc
                (
                    self.live_logl,
                    self.live_it,
                    self.live_bound,
                    self.live_u,
                    self.live_v,
                    self.unit_cube_sampling,
                    self.nbound,
                    self.it,
                ), success = apply_step(
                    self.live_logl,
                    self.live_it,
                    self.live_bound,
                    self.live_u,
                    self.live_v,
                    self.unit_cube_sampling,
                    self.nbound,
                    self.it,
                    point,
                )
                nc += point.nc
                if success:
                    loglstar = jnp.min(self.live_logl)
                    logvol -= self.dlv
                    # loglstar = jnp.min(self.live_logl)
                    logwt, logz, logzvar, h = progress_integration(
                        loglstar_old, loglstar, logz, logzvar,
                        logvol, self.dlv, h)
                    self.cumulative_ln_weight = jnp.logaddexp(self.cumulative_ln_weight, logwt)
                    self.cumulative_ln_weight_squared = jnp.logaddexp(
                        self.cumulative_ln_weight_squared, 2 * logwt
                    )

                    if point.blob is None:
                        blob = dict()
                    else:
                        blob = point.blob

                    to_add.append((
                        worst,
                        ustar,
                        vstar,
                        loglstar,
                        logvol,
                        logwt,
                        logz,
                        logzvar,
                        h,
                        nc,
                        worst_it,
                        blob.get("accept", 1),
                        bounditer,
                        blob.get("distance_insertion_index", -1),
                        blob.get("likelihood_insertion_index", -1),
                    ))
                    last_nc = nc
                    nc = 0
            if len(to_add) > 0:
                to_add = (jax.numpy.array(arr) for arr in zip(*to_add))
                self.saved_run.D = update_saved_run(*to_add, self.saved_run.D)
                last_nc = nc

            nc = last_nc
            worst = jnp.argmin(self.live_logl)  # index
            # # Locate the "live" point with the lowest `logl`.
            worst_it = self.live_it[worst]  # when point was proposed
            boundidx = self.live_bound[worst]  # associated bound index
            if not self.unit_cube_sampling:
                self.update_proposal(big_blob, update=True)
            self.update_bound_if_needed(loglstar, ncall=self.ncall)

            # # Set our new worst likelihood constraint.
            # # Notice we are doing copies here because live_u and live_v
            # # are updated in-place
            ustar = self.live_u[worst].copy()  # unit cube position
            vstar = self.live_v[worst].copy()  # transformed position
            if self.blob:
                old_blob = self.live_blobs[worst].copy()
            else:
                old_blob = None

            # # Compute our sampling efficiency.
            self.eff = 100. * self.it / self.ncall

            # Return dead point and ancillary quantities.
            yield IteratorResult(worst=worst,
                                 ustar=ustar,
                                 vstar=vstar,
                                 loglstar=loglstar,
                                 logvol=logvol,
                                 logwt=np.nan,
                                 logz=logz,
                                 logzvar=logzvar,
                                 h=h,
                                 nc=nc,
                                 blob=old_blob,
                                 worst_it=worst_it,
                                 boundidx=boundidx,
                                 bounditer=bounditer,
                                 eff=self.eff,
                                 delta_logz=delta_logz)

    def run_nested(self,
                   maxiter=None,
                   maxcall=None,
                   dlogz=None,
                   logl_max=np.inf,
                   n_effective=np.inf,
                   add_live=True,
                   print_progress=True,
                   print_func=None,
                   save_bounds=True,
                   checkpoint_file=None,
                   checkpoint_every=60,
                   resume=False):
        """
        **A wrapper that executes the main nested sampling loop.**
        Iteratively replace the worst live point with a sample drawn
        uniformly from the prior until the provided stopping criteria
        are reached.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations. Iteration may stop earlier if the
            termination condition is reached. Default is `sys.maxsize`
            (no limit).

        maxcall : int, optional
            Maximum number of likelihood evaluations. Iteration may stop
            earlier if termination condition is reached. Default is
            `sys.maxsize` (no limit).

        dlogz : float, optional
            Iteration will stop when the estimated contribution of the
            remaining prior volume to the total evidence falls below
            this threshold. Ejnplicitly, the stopping criterion is
            `ln(z + z_est) - ln(z) < dlogz`, where `z` is the current
            evidence from all saved samples and `z_est` is the estimated
            contribution from the remaining volume. If `add_live` is `True`,
            the default is `1e-3 * (nlive - 1) + 0.01`. Otherwise, the
            default is `0.01`.

        logl_max : float, optional
            Iteration will stop when the sampled ln(likelihood) exceeds the
            threshold set by `logl_max`. Default is no bound (`np.inf`).

        n_effective: int, optional
            Minimum number of effective posterior samples. If the estimated
            "effective sample size" (ESS) exceeds this number,
            sampling will terminate. Default is no ESS (`np.inf`).
            This option is deprecated and will be removed in a future release.

        add_live : bool, optional
            Whether or not to add the remaining set of live points to
            the list of samples at the end of each run. Default is `True`.

        print_progress : bool, optional
            Whether or not to output a simple summary of the current run that
            updates with each iteration. Default is `True`.

        print_func : function, optional
            A function that prints out the current state of the sampler.
            If not provided, the default :meth:`results.print_fn` is used.

        save_bounds : bool, optional
            Whether or not to save past bounding distributions used to bound
            the live points internally. Default is *True*.

        checkpoint_file: string, optional
            if not None The state of the sampler will be saved into this
            file every checkpoint_every seconds

        checkpoint_every: float, optional
            The number of seconds between checkpoints that will save
            the internal state of the sampler. The sampler will also be
            saved in the end of the run irrespective of checkpoint_every.
        """

        # Define our stopping criteria.
        if dlogz is None:
            if add_live:
                dlogz = 1e-3 * (self.nlive - 1.) + 0.01
            else:
                dlogz = 0.01
        if resume and self.added_live:
            warnings.warn('You are resuming a finished static run. '
                          'This will not do anything')
            # TODO I should create a separate STATE Enum
            # here like to rely on that rather than added_live
            return

        # Run the main nested sampling loop.
        pbar, print_func = get_print_func(print_func, print_progress)
        if checkpoint_file is not None:
            timer = DelayTimer(checkpoint_every)
        try:
            ncall = self.ncall
            for it, results in enumerate(
                    self.sample(maxiter=maxiter,
                                maxcall=maxcall,
                                dlogz=dlogz,
                                logl_max=logl_max,
                                save_bounds=save_bounds,
                                save_samples=True,
                                n_effective=n_effective,
                                resume=resume,
                                add_live=add_live)):
                ncall += results.nc

                # Print progress.
                if print_progress:
                    i = self.it - 1
                    print_func(results,
                               i,
                               ncall,
                               dlogz=dlogz,
                               logl_max=logl_max)

                if checkpoint_file is not None and timer.is_time():
                    self.save(checkpoint_file)

            # Add remaining live points to samples.
            if add_live:
                self.add_live_points()
                # it = self.it - 1
                # for i, results in enumerate(self.add_live_points()):
                #     ncall += results.nc

                #     # Print progress.
                #     if print_progress:
                #         print_func(results,
                #                    it,
                #                    ncall,
                #                    add_live_it=i + 1,
                #                    dlogz=dlogz,
                #                    logl_max=logl_max)

            # Here we recompute the integrals using the full run
            new_logwt, new_logz, new_logzvar, new_h = compute_integrals(
                logl=self.saved_run['logl'], logvol=self.saved_run['logvol'])
            self.saved_run['logwt'] = new_logwt
            self.saved_run['logz'] = new_logz
            self.saved_run['logzvar'] = new_logzvar
            self.saved_run['h'] = new_h
            if checkpoint_file is not None:
                # I don't check the time timer here
                self.save(checkpoint_file)

        finally:
            if pbar is not None:
                pbar.close()
            self.loglikelihood.history_save()

    def add_final_live(self, print_progress=True, print_func=None):
        """
        **A wrapper that executes the loop adding the final live points.**
        Adds the final set of live points to the pre-existing sequence of
        dead points from the current nested sampling run.

        Parameters
        ----------
        print_progress : bool, optional
            Whether or not to output a simple summary of the current run that
            updates with each iteration. Default is `True`.

        print_func : function, optional
            A function that prints out the current state of the sampler.
            If not provided, the default :meth:`results.print_fn` is used.

        """

        if print_func is None:
            print_func = print_fn

        # Add remaining live points to samples.
        pbar, print_func = get_print_func(print_func, print_progress)
        try:
            ncall = self.ncall
            it = self.it - 1
            for i, results in enumerate(self.add_live_points()):

                # Print progress.
                if print_progress:
                    print_func(results,
                               it,
                               ncall,
                               add_live_it=i + 1,
                               dlogz=0.01)
        finally:
            if pbar is not None:
                pbar.close()


@jax.jit
def check_complete(
    it, maxiter, ncall, maxcall, dlogz, delta_logz,
    loglstar, logl_max, n_effective, current_n_effective,
    live_logl,
):
    # Stopping criterion 1: current number of iterations
    # exceeds `maxiter`.
    # Stopping criterion 2: current number of `loglikelihood`
    stop_iterations = (it > maxiter) | (ncall > maxcall)

    stop_iterations = stop_iterations | (dlogz is not None and delta_logz < dlogz)
    # Stopping criterion 3: estimated (fractional) remaining evidence
    # lies below some threshold set by `dlogz`.
    
    # Stopping criterion 4: last dead point exceeded the upper
    # `logl_max` bound.
    stop_iterations = stop_iterations | (loglstar > logl_max)

    # Stopping criterion 5: the number of effective posterior
    # samples has been achieved.
    stop_iterations = stop_iterations | (current_n_effective > n_effective)
    # if (n_effective is not None) and not jnp.isposinf(n_effective):
    #     current_n_effective = self.n_effective
    #     # TODO This needs to be refactored
    #     # here we are adding final live points then checking
    #     # if n_effective is large enough then removing them again
    #     # this is slow and not a good logic
    #     if current_n_effective > n_effective:
    #         # if add_live:
    #         #     self.add_final_live(print_progress=False)

    #         #     # Recompute n_effective after adding live points
    #         #     current_n_effective = self.n_effective
    #         #     self._remove_live_points()
    #         #     self.added_live = False
    #         if current_n_effective > n_effective:
    #             stop_iterations = True

    stop_iterations = stop_iterations | (jnp.ptp(live_logl) == 0)
    
    return stop_iterations


@jax.jit
def apply_step(
    live_logl,
    live_it,
    live_bound,
    live_u,
    live_v,
    unit_cube_sampling,
    nbound,
    it,
    point,
):
    success = point.logl > jnp.min(live_logl)
    args = jax.lax.cond(
        success,
        actually_apply_step,
        lambda x: x,
        (
            live_logl,
            live_it,
            live_bound,
            live_u,
            live_v,
            unit_cube_sampling,
            nbound,
            it,
            point.u,
            point.v,
            point.logl,
        ),
    )
    return args[:-3], success


@jax.jit
def actually_apply_step(args):
    (
        live_logl,
        live_it,
        live_bound,
        live_u,
        live_v,
        unit_cube_sampling,
        nbound,
        it,
        u,
        v,
        logl,
    ) = args

    # Compute bound index at the current iteration.
    bounditer = jax.lax.select(
        unit_cube_sampling,
        0,
        nbound - 1
    )

    worst = jnp.argmin(live_logl)  # index
    live_u = live_u.at[worst].set(u)
    live_v = live_v.at[worst].set(v)
    if isinstance(logl, LoglOutput):
        live_logl = live_logl.at[worst].set(logl.val)
    else:
        live_logl = live_logl.at[worst].set(logl)
    live_bound = live_bound.at[worst].set(bounditer)
    live_it = live_it.at[worst].set(it)
    # Compute our sampling efficiency.
    return (
        live_logl,
        live_it,
        live_bound,
        live_u,
        live_v,
        unit_cube_sampling,
        nbound,
        it + 1,
        u,
        v,
        logl,
    )


@jax.jit
def update_saved_run(
    worst,
    ustar,
    vstar,
    loglstar,
    logvol,
    logwt,
    logz,
    logzvar,
    h,
    nc,
    worst_it,
    scale,
    bounditer,
    distance_index,
    likelihood_index,
    saved_run,
):
    # Save the worst live point. It is now a "dead" point.
    update = dict(id=worst,
                u=ustar,
                v=vstar,
                logl=loglstar,
                logvol=logvol,
                logwt=logwt,
                logz=logz,
                logzvar=logzvar,
                h=h,
                nc=nc,
                it=worst_it,
                bounditer=bounditer,
                scale=scale,
                blob=jax.numpy.full_like(loglstar, jax.numpy.nan),
                boundidx=bounditer,
                distance_insertion_index=distance_index,
                likelihood_insertion_index=likelihood_index,
                n=jax.numpy.full_like(loglstar, jax.numpy.nan),
                )
    for key, value in update.items():
        if value is None:
            value = jax.numpy.full_like(loglstar, jax.numpy.nan)
        value = jax.numpy.array(value)
        this = saved_run.get(key, jax.numpy.array(jax.numpy.nan))
        this = jax.numpy.array(this)
        if key in saved_run and this.ndim == value.ndim:
            saved_run[key] = jax.numpy.concatenate([this, value])
        elif key in saved_run:
            saved_run[key] = value
    return saved_run
