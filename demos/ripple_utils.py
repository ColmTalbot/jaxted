"""
Generic dumping ground for jax-specific functions that we need.
This should find a home somewhere down the line, but gives an
idea of how much pain is being added.
"""

from functools import partial

import jax
import jax.numpy as jnp
from astropy import constants
from ripplegw.waveforms import (
    IMRPhenomD,
    IMRPhenomD_NRTidalv2,
    IMRPhenomPv2,
    IMRPhenomXAS,
    TaylorF2,
)

MTSUN_SI = (constants.GM_sun / constants.c**3).value


@jax.jit
def rotate_z(angle):
    shape = jnp.shape(angle)
    return jnp.array(
        [
            [jnp.cos(angle), -jnp.sin(angle), jnp.zeros(shape)],
            [jnp.sin(angle), jnp.cos(angle), jnp.zeros(shape)],
            [jnp.zeros(shape), jnp.zeros(shape), jnp.ones(shape)],
        ]
    )


@jax.jit
def rotate_y(angle):
    shape = jnp.shape(angle)
    return jnp.array(
        [
            [jnp.cos(angle), jnp.zeros(shape), jnp.sin(angle)],
            [jnp.zeros(shape), jnp.ones(shape), jnp.zeros(shape)],
            [-jnp.sin(angle), jnp.zeros(shape), jnp.cos(angle)],
        ]
    )


@jax.jit
def rotate_x(angle):
    shape = jnp.shape(angle)
    return jnp.array(
        [
            [jnp.zeros(shape), jnp.zeros(shape), jnp.zeros(shape)],
            [jnp.zeros(shape), jnp.cos(angle), -jnp.sin(angle)],
            [jnp.zeros(shape), jnp.sin(angle), jnp.cos(angle)],
        ]
    )


@partial(jax.jit, static_argnames=("axis",))
def rotate(angle, axis):
    match axis:
        case "x":
            return rotate_x(angle)
        case "y":
            return rotate_y(angle)
        case "z":
            return rotate_z(angle)
        case _:
            raise ValueError(f"Invalid rotation axis: {axis}")


@jax.jit
def apply_rotation(arr, rot):
    return jnp.einsum("ij...,j...->i...", rot, arr, optimize=True), None


@partial(jax.jit, static_argnames=("convention",))
def euler_rotation(array, angles, convention="zyz"):
    rotations = jnp.array(
        [rotate(angle, axis) for angle, axis in zip(angles, convention)]
    )
    return jax.lax.scan(apply_rotation, array, rotations)[0]


@partial(jax.jit, static_argnames=("frame",))
def transform_precessing_spins(
    thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2, m1, m2, fRef, phiRef, frame="LN"
):
    """
    Vectorized version of XLALSimInspiralTransformPrecessingSpin

    There is an additional option to specify whether the spins should be defined with
    respect to the total orbital momentum (:code:`JN`, as in :code:`lalsimulation`)
    or orbital angular momentum (:code:`LN`, as recommended in https://arxiv.org/abs/2207.03508)
    """
    args = thetaJN, phiJL, theta1, theta2, phi12, chi1, chi2, m1, m2, fRef, phiRef
    shape = jnp.broadcast_shapes(*(jnp.shape(arr) for arr in args))

    spin_1 = chi1 * jnp.array(
        [
            jnp.sin(theta1) * jnp.cos(phiRef),
            jnp.sin(theta1) * jnp.sin(phiRef),
            jnp.cos(theta1),
        ]
    )
    spin_2 = chi2 * jnp.array(
        [
            jnp.sin(theta2) * jnp.cos(phi12 + phiRef),
            jnp.sin(theta2) * jnp.sin(phi12 + phiRef),
            jnp.cos(theta2),
        ]
    )
    LNhat = jnp.array([jnp.zeros(shape), jnp.zeros(shape), jnp.ones(shape)])

    total_mass = m1 + m2
    eta = m1 * m2 / total_mass**2
    v0 = (total_mass * MTSUN_SI * jnp.pi * fRef) ** (1 / 3)
    Lmag = total_mass**2 * eta / v0 * (1 + v0**2 * (9 + eta) / 6)
    dimensionful_spin_1 = m1**2 * spin_1
    dimensionful_spin_2 = m2**2 * spin_2
    orbital_angular_momentum = Lmag * LNhat
    J = dimensionful_spin_1 + dimensionful_spin_2 + orbital_angular_momentum

    Jhat = J / jnp.linalg.norm(J, axis=0, keepdims=True)
    theta0 = jnp.arccos(Jhat[2])
    phi0 = jnp.arctan2(Jhat[1], Jhat[0])

    euler_angles = [-phi0, -theta0, phiJL - jnp.pi]
    LNhat = euler_rotation(LNhat, euler_angles)
    spin_1 = euler_rotation(spin_1, euler_angles)
    spin_2 = euler_rotation(spin_2, euler_angles)

    # Compute inclination angle
    line_of_sight = jnp.array([jnp.zeros(shape), jnp.sin(thetaJN), jnp.cos(thetaJN)])
    theta_ln = jnp.arccos(jnp.einsum("i...,i...->...", line_of_sight, LNhat))

    # rotate to align N along z-axis
    theta_lj = jnp.arccos(LNhat[2])
    phi_l = jnp.arctan2(LNhat[1], LNhat[0])
    line_of_sight = euler_rotation(line_of_sight, [-phi_l, -theta_lj, 0])
    phi_n = jnp.arctan2(line_of_sight[1], line_of_sight[0])

    angle_3 = jnp.pi / 2 - phi_n
    if frame == "JN":
        angle_3 -= phiRef

    euler_angles = [-phi_l, -theta_lj, angle_3]
    spin_1 = euler_rotation(spin_1, euler_angles)
    spin_2 = euler_rotation(spin_2, euler_angles)

    return theta_ln, spin_1, spin_2


def ripple_bbh_roq(
    frequency,
    mass_1,
    mass_2,
    luminosity_distance,
    theta_jn,
    phase,
    a_1,
    a_2,
    tilt_1,
    tilt_2,
    phi_12,
    phi_jl,
    **kwargs,
):
    return ripple_cbc_roq(
        frequency,
        mass_1,
        mass_2,
        luminosity_distance,
        theta_jn,
        phase,
        a_1,
        a_2,
        tilt_1,
        tilt_2,
        phi_12,
        phi_jl,
        lambda_1=0.0,
        lambda_2=0.0,
        **kwargs,
    )


def ripple_bns_roq(
    frequency,
    mass_1,
    mass_2,
    luminosity_distance,
    theta_jn,
    phase,
    a_1,
    a_2,
    tilt_1,
    tilt_2,
    phi_12,
    phi_jl,
    lambda_1,
    lambda_2,
    **kwargs,
):
    return ripple_cbc_roq(
        frequency,
        mass_1,
        mass_2,
        luminosity_distance,
        theta_jn,
        phase,
        a_1,
        a_2,
        tilt_1,
        tilt_2,
        phi_12,
        phi_jl,
        lambda_1,
        lambda_2,
        **kwargs,
    )


@partial(jax.jit, static_argnames=("spin_reference_frame", "waveform_approximant"))
def ripple_cbc_roq(*args, **kwargs):
    if "frequency_nodes" in kwargs:
        kwargs["frequencies"] = kwargs["frequency_nodes"]
        roq_mode = True
    else:
        roq_mode = False
    waveform = ripple_cbc(*args, **kwargs)
    if roq_mode:
        output = dict()
        for key, indices in kwargs.items():
            if not key.endswith("_indices"):
                continue
            subset = key.rsplit("_", maxsplit=1)[0]
            output[subset] = {key: value[indices] for key, value in waveform.items()}
        waveform = output
    return waveform


def ripple_bbh_relbin(
    frequency,
    mass_1,
    mass_2,
    luminosity_distance,
    theta_jn,
    phase,
    a_1,
    a_2,
    tilt_1,
    tilt_2,
    phi_12,
    phi_jl,
    fiducial,
    **kwargs,
):
    return ripple_cbc_relbin(
        frequency,
        mass_1,
        mass_2,
        luminosity_distance,
        theta_jn,
        phase,
        a_1,
        a_2,
        tilt_1,
        tilt_2,
        phi_12,
        phi_jl,
        lambda_1=0.0,
        lambda_2=0.0,
        fiducial=fiducial,
        **kwargs,
    )


def ripple_bns_relbin(
    frequency,
    mass_1,
    mass_2,
    luminosity_distance,
    theta_jn,
    phase,
    a_1,
    a_2,
    tilt_1,
    tilt_2,
    phi_12,
    phi_jl,
    lambda_1,
    lambda_2,
    fiducial,
    **kwargs,
):
    return ripple_cbc_relbin(
        frequency,
        mass_1,
        mass_2,
        luminosity_distance,
        theta_jn,
        phase,
        a_1,
        a_2,
        tilt_1,
        tilt_2,
        phi_12,
        phi_jl,
        lambda_1,
        lambda_2,
        fiducial=fiducial,
        **kwargs,
    )


@partial(
    jax.jit,
    static_argnames=("spin_reference_frame", "waveform_approximant", "fiducial"),
)
def ripple_cbc_relbin(frequency, *args, **kwargs):
    if kwargs.get("fiducial", 0) == 1:
        kwargs["frequencies"] = frequency
    else:
        kwargs["frequencies"] = kwargs.pop("frequency_bin_edges")
    return ripple_cbc(frequency, *args, **kwargs)


def ripple_bbh(
    frequency,
    mass_1,
    mass_2,
    luminosity_distance,
    theta_jn,
    phase,
    a_1,
    a_2,
    tilt_1,
    tilt_2,
    phi_12,
    phi_jl,
    **kwargs,
):
    return ripple_cbc(
        frequency,
        mass_1,
        mass_2,
        luminosity_distance,
        theta_jn,
        phase,
        a_1,
        a_2,
        tilt_1,
        tilt_2,
        phi_12,
        phi_jl,
        lambda_1=0.0,
        lambda_2=0.0,
        **kwargs,
    )


def ripple_bns(
    frequency,
    mass_1,
    mass_2,
    luminosity_distance,
    theta_jn,
    phase,
    a_1,
    a_2,
    tilt_1,
    tilt_2,
    phi_12,
    phi_jl,
    lambda_1,
    lambda_2,
    **kwargs,
):
    return ripple_cbc(
        frequency,
        mass_1,
        mass_2,
        luminosity_distance,
        theta_jn,
        phase,
        a_1,
        a_2,
        tilt_1,
        tilt_2,
        phi_12,
        phi_jl,
        lambda_1,
        lambda_2,
        **kwargs,
    )


@partial(jax.jit, static_argnames=("spin_reference_frame", "waveform_approximant"))
def ripple_cbc(
    frequency,
    mass_1,
    mass_2,
    luminosity_distance,
    theta_jn,
    phase,
    a_1,
    a_2,
    tilt_1,
    tilt_2,
    phi_12,
    phi_jl,
    lambda_1,
    lambda_2,
    **kwargs,
):
    reference_frequency = kwargs.get("reference_frequency", 50.0)
    iota, spin_1, spin_2 = transform_precessing_spins(
        theta_jn,
        phi_jl,
        tilt_1,
        tilt_2,
        phi_12,
        a_1,
        a_2,
        mass_1,
        mass_2,
        reference_frequency,
        phase,
        frame=kwargs.get("spin_reference_frame", "JN"),
    )
    if "frequencies" in kwargs:
        frequencies = kwargs["frequencies"]
    elif "minimum_frequency" in kwargs:
        frequencies = jnp.maximum(frequency, kwargs["minimum_frequency"])
    else:
        frequencies = frequency

    component_mass = (mass_1, mass_2)
    mc_eta = (
        (mass_1 * mass_2) ** (3 / 5) / (mass_1 + mass_2) ** (1 / 5),
        mass_1 * mass_2 / (mass_1 + mass_2) ** 2,
    )
    precessing_spins = (*spin_1, *spin_2)
    aligned_spins = (spin_1[2], spin_2[2])
    tidal = (lambda_1, lambda_2)
    time = jnp.array(0.0)
    extrinsic = (luminosity_distance, time, phase, iota)

    match kwargs["waveform_approximant"]:
        case "IMRPhenomD":
            wf_func = IMRPhenomD.gen_IMRPhenomD_hphc
            theta = mc_eta + aligned_spins + extrinsic
        case "IMRPhenomD_NRTidalv2":
            wf_func = IMRPhenomD_NRTidalv2.gen_IMRPhenomD_NRTidalv2_hphc
            theta = mc_eta + aligned_spins + tidal + extrinsic
        case "IMRPhenomPv2":
            wf_func = IMRPhenomPv2.gen_IMRPhenomPv2
            theta = component_mass + precessing_spins + extrinsic
        case "IMRPhenomXAS":
            wf_func = IMRPhenomXAS.gen_IMRPhenomXAS_hphc
            theta = mc_eta + aligned_spins + extrinsic
        case "TaylorF2":
            wf_func = TaylorF2.gen_TaylorF2_hphc
            theta = mc_eta + aligned_spins + extrinsic
        case _:
            raise ValueError(
                f"Unsupported waveform approximant: {kwargs['waveform_approximant']}"
            )

    hp, hc = wf_func(frequencies, theta, reference_frequency)
    return dict(plus=hp, cross=hc)
