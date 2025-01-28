import os
from functools import partial

import jax
import numpy as np
import pandas as pd
from bilby.core.sampler.base_sampler import Sampler
from bilby.core.result import Result
from bilby.compat.jax import generic_bilby_likelihood_function

from .nest import run_nest
from .smc import run_nssmc_anssmc
from .utils import apply_boundary, generic_bilby_ln_prior

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["BILBY_ARRAY_API"] = "1"
os.environ["SCIPY_ARRAY_API"] = "1"


class Jaxted(Sampler):
    """
    bilby wrapper of `dynesty.NestedSampler`
    (https://dynesty.readthedocs.io/en/latest/)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `dynesty.NestedSampler`, see
    documentation for that class for further help. Under Other Parameters below,
    we list commonly used kwargs and the Bilby defaults.

    Parameters
    ==========
    likelihood: likelihood.Likelihood
        A  object with a log_l method
    priors: bilby.core.prior.PriorDict, dict
        Priors to be used in the search.
        This has attributes for each parameter to be sampled.
    outdir: str, optional
        Name of the output directory
    label: str, optional
        Naming scheme of the output files
    use_ratio: bool, optional
        Switch to set whether or not you want to use the log-likelihood ratio
        or just the log-likelihood
    plot: bool, optional
        Switch to set whether or not you want to create traceplots
    skip_import_verification: bool
        Skips the check if the sampler is installed if true. This is
        only advisable for testing environments
    print_method: str ('tqdm')
        The method to use for printing. The options are:
        - 'tqdm': use a `tqdm` `pbar`, this is the default.
        - 'interval-$TIME': print to `stdout` every `$TIME` seconds,
          e.g., 'interval-10' prints every ten seconds, this does not print every iteration
        - else: print to `stdout` at every iteration
    exit_code: int
        The code which the same exits on if it hasn't finished sampling
    check_point: bool,
        If true, use check pointing.
    check_point_plot: bool,
        If true, generate a trace plot along with the check-point
    check_point_delta_t: float (600)
        The minimum checkpoint period (in seconds). Should the run be
        interrupted, it can be resumed from the last checkpoint.
    n_check_point: int, optional (None)
        The number of steps to take before checking whether to check_point.
    resume: bool
        If true, resume run from checkpoint (if available)
    maxmcmc: int (5000)
        The maximum length of the MCMC exploration to find a new point
    nact: int (2)
        The number of autocorrelation lengths for MCMC exploration.
        For use with the :code:`act-walk` and :code:`rwalk` sample methods.
        See the dynesty guide in the Bilby docs for more details.
    naccept: int (60)
        The expected number of accepted steps for MCMC exploration when using
        the :code:`acceptance-walk` sampling method.
    rejection_sample_posterior: bool (True)
        Whether to form the posterior by rejection sampling the nested samples.
        If False, the nested samples are resampled with repetition. This was
        the default behaviour in :code:`Bilby<=1.4.1` and leads to
        non-independent samples being produced.
    proposals: iterable (None)
        The proposal methods to use during MCMC. This can be some combination
        of :code:`"diff", "volumetric"`. See the dynesty guide in the Bilby docs
        for more details. default=:code:`["diff"]`.
    rstate: numpy.random.Generator (None)
        Instance of a numpy random generator for generating random numbers.
        Also see :code:`seed` in 'Other Parameters'.

    Other Parameters
    ================
    nlive: int, (1000)
        The number of live points, note this can also equivalently be given as
        one of [nlive, nlives, n_live_points, npoints]
    bound: {'live', 'live-multi', 'none', 'single', 'multi', 'balls', 'cubes'}, ('live')
        Method used to select new points
    sample: {'act-walk', 'acceptance-walk', 'unif', 'rwalk', 'slice',
             'rslice', 'hslice', 'rwalk_dynesty'}, ('act-walk')
        Method used to sample uniformly within the likelihood constraints,
        conditioned on the provided bounds
    walks: int (100)
        Number of walks taken if using the dynesty implemented sample methods
        Note that the default `walks` in dynesty itself is 25, although using
        `ndim * 10` can be a reasonable rule of thumb for new problems.
        For :code:`sample="act-walk"` and :code:`sample="rwalk"` this parameter
        has no impact on the sampling.
    dlogz: float, (0.1)
        Stopping criteria
    seed: int (None)
        Use to seed the random number generator if :code:`rstate` is not
        specified.
    """

    sampler_name = "jaxted"
    sampling_seed_key = "rseed"
    default_kwargs = dict(
        method="nest", nsteps=500, nlive=500, rseed=1, alpha=np.exp(-1)
    )

    def run_sampler(self):
        likelihood_fn = jax.vmap(
            partial(
                generic_bilby_likelihood_function,
                self.likelihood,
                use_ratio=self.use_ratio,
            )
        )
        boundary_fn = partial(apply_boundary, priors=self.priors)
        ln_prior_fn = partial(generic_bilby_ln_prior, priors=self.priors)
        sample_fn = self.priors.sample

        method = self.kwargs.pop("method", "nest")
        if method == "nest":
            sampler = run_nest
            self.kwargs.pop("alpha")
        elif method == "smc":
            sampler = run_nssmc_anssmc
        else:
            raise ValueError("Unknown sampling method")
        ln_z, ln_zerr, samples = sampler(
            likelihood_fn=likelihood_fn,
            ln_prior_fn=ln_prior_fn,
            sample_prior=sample_fn,
            boundary_fn=boundary_fn,
            **self.kwargs,
        )
        self.result = self.create_result(samples, ln_z, ln_zerr)
        self.kwargs["method"] = method
        return self.result

    def create_result(self, samples, ln_z, ln_zerr):
        posterior_samples = pd.DataFrame(samples)
        return Result(
            label=self.label,
            outdir=self.outdir,
            posterior=posterior_samples,
            log_evidence=float(ln_z),
            search_parameter_keys=self.search_parameter_keys,
            priors=self.priors,
            log_noise_evidence=self.likelihood.noise_log_likelihood(),
            log_evidence_err=ln_zerr,
            log_bayes_factor=float(ln_z) - self.likelihood.noise_log_likelihood(),
        )

    def _setup_pool(self):
        """
        In addition to the usual steps, we need to set the sampling kwargs on
        every process. To make sure we get every process, run the kwarg setting
        more times than we have processes.
        """
        self.kwargs["npool"] = 1
        super()._setup_pool()
