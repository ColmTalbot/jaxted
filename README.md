# jaxted

**jaxted** is a JAX-native library for performing nested sampling and nested sampling via Sequential Monte Carlo (SMC). It is designed for scalable, efficient Bayesian computation and evidence estimation, leveraging JAX's hardware acceleration and automatic vectorization.

## Features

- **Nested Sampling**: Estimate Bayesian evidence and sample from complex posteriors.
- **SMC Nested Sampling**: Sequential Monte Carlo approach for improved efficiency and flexibility.
- **JAX-native**: Fully compatible with JAX transformations (`jit`, `vmap`, etc.) and accelerators (GPU/TPU).

## Installation

`Jaxted` is under active development and can be installed directly from the GitHub repository.

```bash
pip install git+https://github.com/colmtalbot/jaxted.git@main
```

## Usage

..include:: demos/basic.py

### Numpyro

Jaxted can be used with [Numpyro](https://num.pyro.ai/en/latest/index.html#introductory-tutorials).
Given a `numpyro` model, you can generate jaxted-compatible log-likelihood and prior functions:

```python
from jaxted.numpyro import jaxted_inputs_from_numpyro
jaxted_inputs_from_numpyro

jaxted_loglikelihood, jaxted_prior = jaxted_inputs_from_numpyro(model)
```

### Bilby

Jaxted can be used with [Bilby](https://github.com/bilby-dev/bilby) for nested sampling tasks by specifying `sampler="jaxted"` in the call to `bilby.run_sampler`. Currently this requires installing `Bilby` from github

```bash
pip install git+https://github.com/ColmTalbot/bilby.git@bilback
```

## License

MIT License

## Acknowledgements

Built with [JAX](https://github.com/google/jax).
This project is not affiliated with JAX or Google.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.