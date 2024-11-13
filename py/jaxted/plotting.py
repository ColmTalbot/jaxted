#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of built-in plotting functions to help visualize ``dynesty`` nested
sampling :class:`~dynesty.results.Results`.

"""

import matplotlib.pyplot as pl
from .utils import insertion_index_test

__all__ = ["insertionplot"]


def insertionplot(results, figsize=(8, 5)):
    """
    Plot the fractional insertion indices for likelihood and distance.

    A trace is added for each unique number of live points.
    The p-value comparing with the expected distribution is shown in the
    legend.

    Parameters
    ----------
    results : :class:`~dynesty.results.Results` instance
        A :class:`~dynesty.results.Results` instance from a nested
        sampling run.

    Returns
    -------
    insertionplot : (`~matplotlib.figure.Figure`, `~matplotlib.axes.Axis`)
        Output insertion index plot.
    """
    fig = pl.figure(figsize=figsize)
    ax = pl.gca()
    for kind in ["likelihood", "distance"]:
        insertion_index_test(result=results, kind=kind, ax=ax)
    ax.set_ylabel("Insertion index")
    ax.set_xlim(0, 1)
    ax.legend(loc="best")
    ax.set_xlabel("Index / $n_{\\rm live}$")
    return fig, ax
