#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

from measure_extinction.plotting.plot_ext import (
    plot_multi_extinction,
    plot_average,
)
from measure_extinction.extdata import ExtData, AverageExtData


def plot_extinction_curves():
    # define the path and the names of the star pairs in the format "reddenedstarname_comparisonstarname" (first the main sequence stars and then the giant stars, sorted by spectral type from B8 to O4)
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    starpair_list = [
        "HD017505_HD214680",
        "BD+56d524_HD051283",
        "HD013338_HD031726",
        "HD014250_HD032630",
        "HD014422_HD214680",
        "HD014956_HD188209",
        "HD029309_HD051283",
        "HD029647_HD078316",
        "HD034921_HD214680",
        "HD037020_HD034816",
        "HD037022_HD214680",
        "HD037023_HD036512",
        "HD037061_HD034816",
        "HD038087_HD003360",
        "HD052721_HD036512",
        "HD156247_HD032630",
        "HD166734_HD036512",
        "HD183143_HD188209",
        "HD185418_HD034816",
        "HD192660_HD091316",
        "HD204827_HD204172",
        "HD206773_HD047839",
        "HD229238_HD091316",
        "HD283809_HD003360",
        "HD294264_HD031726",
    ]

    # plot the extinction curves
    plot_multi_extinction(
        starpair_list,
        path,
        range=[0.76, 5.5],
        alax=True,
        spread=True,
        exclude=["IRS"],
        pdf=True,
    )

    # plot the average extinction curve in a separate plot
    plot_average(
        starpair_list,
        path,
        alax=False,
        powerlaw=True,
        extmodels=True,
        range=[0.78, 5.1],
        exclude=["IRS", "BAND"],
        pdf=True,
    )


if __name__ == "__main__":
    plot_extinction_curves()
