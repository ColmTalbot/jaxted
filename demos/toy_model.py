import bilby
import jax
import numpy as np


def func(data, x, y):
    return data[:, None] - jax.numpy.array([x, y])[None, :]


def setup_toy_likelihood_and_priors():
    priors = bilby.core.prior.PriorDict()
    priors['x'] = bilby.core.prior.Uniform(-1, 2, 'x')
    priors['y'] = bilby.core.prior.Uniform(-1, 2, 'y')

    data = np.random.normal(0, 1, (1000, 2))
    points = np.linspace(0, 1, 1000)
    data += func(points, *[0.7 for _ in priors])

    likelihood = bilby.core.likelihood.GaussianLikelihood(
        x=jax.numpy.array(points),
        y=jax.numpy.array(data),
        func=func,
        sigma=1,
    )
    return likelihood, priors


if __name__ == "__main__":

    likelihood, priors = setup_toy_likelihood_and_priors()
    results = list()
    for nsteps in [3, 10, 30, 100]:
        results.append(bilby.run_sampler(
            likelihood,
            priors,
            sampler="jaxted",
            outdir="test",
            label="toy",
            injection_parameters={'x': 0.7, 'y': 0.7},
            nsteps=nsteps,
            save="hdf5",
        ))
    bilby.core.result.plot_multiple(results, filename="test/toy_model.png")
