#!/usr/bin/env python
# This script creates plots with SpeX extinction curves for the NIR extinction paper (Decleir et al. 2021).

import numpy as np

from astropy.modeling.powerlaws import PowerLaw1D

from measure_extinction.plotting.plot_ext import (
    plot_multi_extinction,
    plot_average,
)
from measure_extinction.extdata import ExtData, AverageExtData


def plot_extinction_curves(inpath, outpath):
    """
    Plot the NIR extinction curves for all stars

    Parameters
    ----------
    inpath : string
        Path to the data files

    outpath : string
        Path to save the plots

    Returns
    -------
    Plot with all extinction curves:
        - in E(lambda-V)
        - in A(lambda)/A(V)
    """
    # define the names of the star pairs in the format "reddenedstarname_comparisonstarname"
    starpairs = [
        "BD+56d524_HD034816",
        "HD013338_HD031726",
        # "HD014250_HD042560",
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
        inpath,
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
        inpath,
        alax=True,
        range=[0.76, 5.45],
        spread=True,
        exclude=["IRS"],
        text_offsets=offsets,
        text_angles=angles,
        pdf=True,
    )
    ax.set_ylim(-0.1, 5.1)
    fig.savefig(outpath + "ext_curves_alav.pdf", bbox_inches="tight")


def plot_average_curve(inpath, outpath):
    """
    Plot the average extinction curve

    Parameters
    ----------
    inpath : string
        Path to the data files

    outpath : string
        Path to save the plots

    Returns
    -------
    Plot with the average diffuse extinction curve
    """
    fig, ax = plot_average(
        inpath,
        fitmodel=True,
        range=[0.78, 5.5],
        exclude=["IRS"],
        pdf=True,
    )

    # add literature curves
    waves = np.arange(0.7, 5.4, 0.01)

    # Martin&Whittet 1990 data points
    MW_waves = np.array([0.36, 0.44, 0.55, 0.7, 0.9, 1.25, 1.65, 2.2, 3.5, 4.8])
    MW_exts = (
        np.array([1.8, 1, 0, -0.78, -1.6, -2.25, -2.58, -2.77, -2.93, -2.98]) / 3.05 + 1
    )
    ax.scatter(MW_waves, MW_exts, s=60)

    # Martin&Whittet 1990 powerlaw
    MW90 = PowerLaw1D(x_0=0.55, amplitude=1.19, alpha=1.84)
    ax.plot(waves, MW90(waves), lw=1.5, ls="--", label="Martin&Whittet 1990")

    # Rieke&Lebofsky 1985 data points
    RL_exts = np.array(
        [1.531, 1.324, 1, 0.748, 0.482, 0.282, 0.175, 0.112, 0.058, 0.023]
    )
    ax.scatter(MW_waves, RL_exts, marker="P", s=60, label="Rieke&Lebofsky 1985")

    # Rieke&Lebofsky 1985 powerlaw (taken from Martin&Whittet 1990)
    RL85 = PowerLaw1D(x_0=0.55, amplitude=1.26, alpha=1.62)
    ax.plot(waves, RL85(waves), lw=1.5, ls="--", label="Rieke&Lebofsky 1985")

    # van de Hulst No. 15
    VDH_exts = np.array(
        [1.555, 1.329, 1, 0.738, 0.469, 0.246, 0.155, 0.0885, 0.045, 0.033]
    )
    ax.scatter(MW_waves, VDH_exts, marker="X", s=60, label="van de Hulst No. 15")

    # finish and save the figure
    ax.legend()
    fig.savefig(outpath + "average_ext.pdf", bbox_inches="tight")


if __name__ == "__main__":
    inpath = "/Users/mdecleir/Documents/NIR_ext/Data/"
    outpath = "/Users/mdecleir/spex_nir_extinction/Figures/"
    plot_extinction_curves(inpath, outpath)
    plot_average_curve(inpath, outpath)
