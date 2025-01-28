"""
A jax-friendly numpy-style random number generator.
"""

import numpy as np
import jax
import jax.numpy as jnp

__all__ = [
    "JAXGenerator",
    "get_random_generator",
    "get_seed_sequence",
]


class JAXGenerator:
    def __init__(self, seed):
        self.key = seed

    def __init_subclass__(cls):
        jax.tree_util.register_pytree_node(
            cls,
            cls.pytree_flatten,
            cls.pytree_unflatten,
        )

    @property
    def key(self):
        self._key, subkey = jax.random.split(self._key)
        return subkey

    @key.setter
    def key(self, value):
        self._key = value

    def uniform(self, size=()):
        if isinstance(size, int):
            size = (size,)
        return jax.random.uniform(self.key, shape=size)

    def integers(self, low, high=None, size=()):
        if isinstance(size, int):
            size = (size,)
        elif size is None:
            size = ()
        if high is None:
            high = low
            low = 0
        return jax.random.randint(
            self.key, minval=low, maxval=high, shape=size
        ).squeeze()

    def beta(self, a, b, size=()):
        if isinstance(size, int):
            size = (size,)
        return jax.random.beta(self.key, a, b, shape=size)

    def exponential(self, scale=1.0, size=()):
        if isinstance(size, int):
            size = (size,)
        return jax.random.exponential(self.key, shape=size) * scale

    def random(self, size=()):
        return self.uniform(size=size)

    def standard_normal(self, size=()):
        if isinstance(size, int):
            size = (size,)
        return jax.random.normal(self.key, shape=size)

    def choice(self, a, size=(), **kwargs):
        if isinstance(size, int):
            size = (size,)
        if isinstance(a, int):
            a = jax.numpy.arange(a)
        return jax.random.choice(self.key, a, shape=size, **kwargs)

    def pytree_flatten(self):
        return (self.key,), dict()

    @classmethod
    def pytree_unflatten(cls, aux_data, data):
        return cls(*data, **aux_data)


def get_random_generator(seed=None):
    """
    Return a random generator (using the seed provided if available)
    """
    if isinstance(seed, (np.random.Generator, JAXGenerator)):
        return seed
    elif isinstance(seed, jax.Array):
        return JAXGenerator(seed)
    return np.random.Generator(np.random.PCG64(seed))


def get_seed_sequence(rstate, nitems):
    """
    Return the list of seeds to initialize random generators
    This is useful when distributing work across a pool
    """
    if isinstance(rstate, np.random.Generator):
        seeds = rstate.integers(0, 2**63 - 1, size=nitems)
    elif isinstance(rstate, JAXGenerator):
        if jax.config.jax_enable_x64:
            inttype = jnp.uint64
        else:
            inttype = jnp.uint32
        seeds = rstate.integers(0, 2**31 - 1, size=(nitems, 2)).astype(inttype)
    elif isinstance(rstate, jax.Array):
        seeds = jax.random.split(rstate, nitems)
    return seeds
