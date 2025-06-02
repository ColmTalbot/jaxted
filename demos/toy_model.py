import bilby
import jax
import numpy as np


def func(data, x, y):
    return data[:, None] - jax.numpy.array([x, y])[None, :]


def setup_toy_likelihood_and_priors():
    priors = bilby.core.prior.PriorDict()
    priors['x'] = bilby.core.prior.Uniform(0, 1, 'x')
    priors['y'] = bilby.core.prior.Uniform(0, 1, 'y')

    data = np.random.normal(0, 1, (1000, 2))
    points = np.linspace(0, 1, 1000)
    data += func(points, *[0.5 for _ in priors])

    likelihood = bilby.core.likelihood.GaussianLikelihood(
        x=jax.numpy.array(points),
        y=jax.numpy.array(data),
        func=func,
        sigma=1,
    )
    return likelihood, priors


if __name__ == "__main__":

    likelihood, priors = setup_toy_likelihood_and_priors()
    result = bilby.run_sampler(
        likelihood,
        priors,
        sampler="jaxted",
        outdir="test",
        label="toy",
        injection_parameters={'x': 0.5, 'y': 0.5},
        save="hdf5",
    )
    result.plot_corner()
