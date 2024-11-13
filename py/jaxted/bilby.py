import os
from importlib.metadata import version

import dill
from bilby.core.utils import (
    check_directory_exists_and_if_not_mkdir,
    logger,
    safe_file_dump,
)
from bilby.core.sampler.dynesty import Dynesty, _set_sampling_kwargs


class Jaxted(Dynesty):
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
    sampling_seed_key = "seed"

    @property
    def _dynesty_init_kwargs(self):
        kwargs = Dynesty._dynesty_init_kwargs.fget(self)
        kwargs["sample"] = "rwalk_jax"
        kwargs["bound"] = "live"
        return kwargs

    @property
    def sampler_init(self):
        from .dynesty import NestedSampler

        return NestedSampler

    @property
    def sampler_class(self):
        from .sampler import Sampler

        return Sampler

    def _set_sampling_method(self):
        """This is retained to clobber the parent class method"""
        _set_sampling_kwargs((self.nact, self.maxmcmc, self.proposals, self.naccept))

    def _setup_pool(self):
        """
        In addition to the usual steps, we need to set the sampling kwargs on
        every process. To make sure we get every process, run the kwarg setting
        more times than we have processes.
        """
        self.kwargs["npool"] = 1
        super()._setup_pool()

    def read_saved_state(self, continuing=False):
        """
        Read a pickled saved state of the sampler to disk.

        If the live points are present and the run is continuing
        they are removed.
        The random state must be reset, as this isn't saved by the pickle.
        `nqueue` is set to a negative number to trigger the queue to be
        refilled before the first iteration.
        The previous run time is set to self.

        Parameters
        ==========
        continuing: bool
            Whether the run is continuing or terminating, if True, the loaded
            state is mostly written back to disk.
        """
        jaxted_version = version("jaxted")
        bilby_version = version("bilby")

        versions = dict(bilby=bilby_version, jaxted=jaxted_version)
        if os.path.isfile(self.resume_file):
            logger.info(f"Reading resume file {self.resume_file}")
            with open(self.resume_file, "rb") as file:
                try:
                    sampler = dill.load(file)
                except EOFError:
                    sampler = None

                if not hasattr(sampler, "versions"):
                    logger.warning(
                        f"The resume file {self.resume_file} is corrupted or "
                        "the version of bilby has changed between runs. This "
                        "resume file will be ignored."
                    )
                    return False
                version_warning = (
                    "The {code} version has changed between runs. "
                    "This may cause unpredictable behaviour and/or failure. "
                    "Old version = {old}, new version = {new}."
                )
                for code in versions:
                    if not versions[code] == sampler.versions.get(code, None):
                        logger.warning(
                            version_warning.format(
                                code=code,
                                old=sampler.versions.get(code, "None"),
                                new=versions[code],
                            )
                        )
                del sampler.versions
                self.sampler = sampler
                if getattr(self.sampler, "added_live", False) and continuing:
                    self.sampler._remove_live_points()
                self.sampler.nqueue = -1
                self.start_time = self.sampler.kwargs.pop("start_time")
                self.sampling_time = self.sampler.kwargs.pop("sampling_time")
                self.sampler.queue_size = self.kwargs["queue_size"]
                self.sampler.pool = None
                self.sampler.M = map
            return True
        else:
            logger.info(f"Resume file {self.resume_file} does not exist.")
            return False

    def write_current_state(self):
        """
        Write the current state of the sampler to disk.

        The sampler is pickle dumped using `dill`.
        The sampling time is also stored to get the full CPU time for the run.

        The check of whether the sampler is picklable is to catch an error
        when using pytest. Hopefully, this message won't be triggered during
        normal running.
        """
        jaxted_version = version("jaxted")
        bilby_version = version("bilby")

        if getattr(self, "sampler", None) is None:
            # Sampler not initialized, not able to write current state
            return

        check_directory_exists_and_if_not_mkdir(self.outdir)
        if hasattr(self, "start_time"):
            self._update_sampling_time()
            self.sampler.kwargs["sampling_time"] = self.sampling_time
            self.sampler.kwargs["start_time"] = self.start_time
        self.sampler.versions = dict(bilby=bilby_version, jaxted=jaxted_version)
        self.sampler.pool = None
        self.sampler.M = map
        if dill.pickles(self.sampler):
            safe_file_dump(self.sampler, self.resume_file, dill)
            logger.info(f"Written checkpoint file {self.resume_file}")
        else:
            logger.warning(
                "Cannot write pickle resume file! "
                "Job will not resume if interrupted."
            )
