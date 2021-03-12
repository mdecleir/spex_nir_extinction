#!/usr/bin/env python
# This script creates plots with SpeX extinction curves for the NIR extinction paper (Decleir et al. 2021).

from measure_extinction.plotting.plot_ext import (
    plot_multi_extinction,
    plot_average,
)
from measure_extinction.extdata import ExtData, AverageExtData


def plot_extinction_curves(path):
    # define the names of the star pairs in the format "reddenedstarname_comparisonstarname"
    starpairs = [
        "BD+56d524_HD034816",
        "HD013338_HD031726",
        "HD014250_HD042560",
        # "HD014422_HD214680",
        "HD014956_HD188209",
        "HD017505_HD214680",
        "HD029309_HD042560",
        "HD029647_HD042560",
        # "HD034921_HD214680",
        # "HD037020_HD034816",
        # "HD037022_HD034816",
        # "HD037023_HD034816",
        "HD037061_HD034816",
        "HD038087_HD034816",
        # "HD052721_HD091316",
        "HD156247_HD031726",
        "HD166734_HD188209",
        "HD183143_HD188209",
        "HD185418_HD034816",
        "HD192660_HD204172",
        "HD204827_HD204172",
        # "HD206773_HD003360",
        "HD229238_HD214680",
        "HD283809_HD003360",
        "HD294264_HD034759",
    ]

    # plot the extinction curves in E(lambda-V)
    plot_multi_extinction(
        starpairs,
        path,
        range=[0.76, 5.5],
        # spread=True,
        exclude=["IRS"],
        pdf=True,
    )

    # specify the offsets and angles for the star names
    offsets = [
        0,
        0.03,
        0.03,
        0.01,
        0.04,
        0.01,
        0.01,
        0.01,
        0.02,
        0,
        0.03,
        0.02,
        -0.05,
        0.04,
        0.03,
        0.05,
        0.02,
        0.02,
    ]
    angles = [
        -38,
        -44,
        -10,
        -32,
        -30,
        -46,
        -44,
        -42,
        -36,
        -46,
        -40,
        -44,
        -42,
        -42,
        -44,
        -46,
        -46,
        -38,
    ]

    # plot the extinction curves in A(lambda)/A(V)
    fig, ax = plot_multi_extinction(
        starpairs,
        path,
        alax=True,
        range=[0.76, 5.45],
        spread=True,
        exclude=["IRS"],
        text_offsets=offsets,
        text_angles=angles,
        pdf=True,
    )
    ax.set_ylim(-0.1, 5.1)
    fig.savefig("Figures/ext_curves_alav.pdf", bbox_inches="tight")


def plot_average_curve(path):
    """
    Plot the average extinction curve

    Parameters
    ----------
    path : string
        Path to the data files
    """
    starpair_list = [
        "BD+56d524_HD034816",
        "HD013338_HD031726",
        "HD014250_HD042560",
        # "HD014422_HD214680",
        "HD014956_HD188209",
        "HD017505_HD214680",
        "HD029309_HD042560",
        "HD029647_HD042560",
        # "HD034921_HD214680",
        # "HD037020_HD034816",
        # "HD037022_HD034816",
        # "HD037023_HD034816",
        "HD037061_HD034816",
        "HD038087_HD034816",
        # "HD052721_HD091316",
        "HD156247_HD031726",
        "HD166734_HD188209",
        "HD183143_HD188209",
        "HD185418_HD034816",
        "HD192660_HD204172",
        "HD204827_HD204172",
        # "HD206773_HD003360",
        "HD229238_HD214680",
        "HD283809_HD003360",
        "HD294264_HD034759",
    ]
    plot_average(
        starpair_list,
        path,
        range=[0.78, 6.1],
        exclude=["IRS"],
        pdf=True,
    )


if __name__ == "__main__":
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    plot_extinction_curves(path)
    # plot_average_curve(path)
