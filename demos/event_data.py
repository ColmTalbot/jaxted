#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.

We optionally use ripple waveforms and a JIT-compiled likelihood.
"""

import os
import bilby
import numpy as np
import jax

import ripple_utils

os.environ["OMP_NUM_THREADS"] = "1"
jax.config.update("jax_enable_x64", True)

bilby.core.utils.setup_logger()


def setup_cbc_likelihood(use_jax=True, model="regular"):
    # Set the duration and sampling frequency of the data segment that we're
    # going to inject the signal into
    minimum_frequency = 20.0

    # Set up a random seed for result reproducibility.  This is optional!
    bilby.core.utils.random.seed(88170235)

    ifos = bilby.gw.detector.InterferometerList(["L1"])
    ifo = ifos[0]
    frequency_array, psd_array = np.genfromtxt("L1_psd.txt").T
    ifo.power_spectral_density = bilby.gw.detector.psd.PowerSpectralDensity(
        frequency_array=frequency_array,
        psd_array=psd_array,
    )
    if os.path.exists("L1_data.txt"):
        from gwpy.timeseries import TimeSeries

        data = TimeSeries.read("L1_data.txt")
        ifo.set_strain_data_from_gwpy_timeseries(data)
    else:
        ifo.strain_data.set_from_open_data(
            name="L1",
            start_time=1369419192.7,
            duration=128,
        )
    if use_jax:
        ifos.set_array_backend(jax.numpy)

    if use_jax:
        minimum_frequency = jax.numpy.array(minimum_frequency)

    # Fixed arguments passed into the source model
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency,
    )

    if use_jax:
        match model:
            case "relbin":
                fdsm = ripple_utils.ripple_bbh_relbin
            case _:
                fdsm = ripple_utils.ripple_bbh
    else:
        match model:
            case "relbin":
                fdsm = bilby.gw.source.lal_binary_black_hole_relative_binning
            case "mb":
                fdsm = bilby.gw.source.binary_black_hole_frequency_sequence
            case _:
                fdsm = bilby.gw.source.lal_binary_black_hole

    waveform_generator = bilby.gw.WaveformGenerator(
        duration=ifo.duration,
        sampling_frequency=ifo.sampling_frequency,
        frequency_domain_source_model=fdsm,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
        use_cache=not use_jax,
    )

    if model == "mb":
        del waveform_generator.waveform_arguments["minimum_frequency"]

    priors = bilby.gw.prior.BBHPriorDict(aligned_spin=False)
    del priors["mass_1"], priors["mass_2"]
    priors["L1_time"] = bilby.core.prior.Uniform(
        1369419318.6, 1369419318.8, latex_label="$t_{L}$"
    )
    priors["chirp_mass"] = bilby.core.prior.Uniform(
        2.0214, 2.0331, latex_label="$\\mathcal{M}$"
    )
    priors["mass_ratio"] = bilby.core.prior.Uniform(
        0.125, 1.0, boundary="periodic", latex_label="$q$"
    )
    priors["luminosity_distance"].minimum = 1
    priors["luminosity_distance"].maximum = 500
    priors["a_2"].maximum = 0.05

    match model:
        case "relbin":
            likelihood_class = (
                bilby.gw.likelihood.RelativeBinningGravitationalWaveTransient
            )
            likelihood_kwargs = dict(epsilon=0.1)
        case "mb":
            likelihood_class = bilby.gw.likelihood.MBGravitationalWaveTransient
            likelihood_kwargs = dict()
        case _:
            likelihood_class = bilby.gw.likelihood.GravitationalWaveTransient
            likelihood_kwargs = dict()
    with jax.explain_cache_misses():
        likelihood = likelihood_class(
            interferometers=ifos,
            waveform_generator=waveform_generator,
            priors=priors,
            phase_marginalization=True,
            distance_marginalization=True,
            # reference_frame=ifos,
            time_reference="L1",
            **likelihood_kwargs,
        )
    return likelihood, priors


if __name__ == "__main__":
    likelihood, priors = setup_cbc_likelihood(model="relbin")
    result = bilby.run_sampler(
        likelihood,
        priors,
        nlive=100,
        nsteps=2000,
        naccept=50,
        sampler="jaxted",
        outdir="230529",
        label="jaxted",
        # conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        save="hdf5",
    )
