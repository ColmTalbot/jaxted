#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
jaxted is a jax-based nested sampling package based on :code:`dynesty`.
The main functionality of is performed by the
:code:`jaxted.NestedSampler` and :code:`jaxted.DynamicNestedSampler`
classes.

There is additionally a plugin for :code:`Bilby`.
"""
from .dynesty import NestedSampler, DynamicNestedSampler
from . import bounding
from . import utils

from ._version import __version__

__all__ = ['NestedSampler', 'DynamicNestedSampler', 'bounding', 'utils', '__version__']
