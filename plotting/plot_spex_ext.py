#!/usr/bin/env python
# This script creates plots with SpeX extinction curves for the NIR extinction paper (Decleir et al. 2021).

import numpy as np
import astropy.units as u

from astropy.modeling.powerlaws import PowerLaw1D
from astropy.table import Table
from matplotlib import pyplot as plt

from measure_extinction.plotting.plot_ext import (
    plot_multi_extinction,
    plot_average,
)

from dust_extinction.averages import RL85_MWGC, G21_MWAvg


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


def plot_average_curve(inpath, table_path, outpath):
    """
    Plot the average extinction curve, together with literature curves

    Parameters
    ----------
    inpath : string
        Path to the data files

    table_path : string
        Path to the tables

    outpath : string
        Path to save the plot

    Returns
    -------
    Plot with the average diffuse extinction curve
    """
    fig, ax = plot_average(
        inpath,
        fitmodel=True,
        range=[0.77, 5.2],
        exclude=["IRS"],
        pdf=True,
    )

    # add literature curves
    waves = np.arange(0.75, 5.2, 0.01)

    # Martin&Whittet 1990 data points, not needed if power law model is plotted
    # MW_waves = np.array([0.36, 0.44, 0.55, 0.7, 0.9, 1.25, 1.65, 2.2, 3.5, 4.8])
    # MW_exts = (
    #     np.array([1.8, 1, 0, -0.78, -1.6, -2.25, -2.58, -2.77, -2.93, -2.98]) / 3.05 + 1
    # )
    # ax.scatter(MW_waves, MW_exts, s=60, label="Martin&Whittet 1990")

    # Martin&Whittet 1990 powerlaw
    MW90 = PowerLaw1D(x_0=0.55, amplitude=1.19, alpha=1.84)
    ax.plot(
        waves,
        MW90(waves),
        color="tab:orange",
        lw=1.7,
        ls="-.",
        alpha=0.8,
        label="Martin & Whittet 1990",
    )

    # Rieke&Lebofsky 1985 data points, not needed because dust_extinction model for this paper is plotted
    # RL_exts = np.array(
    #     [1.531, 1.324, 1, 0.748, 0.482, 0.282, 0.175, 0.112, 0.058, 0.023]
    # )
    # ax.scatter(
    #     MW_waves,
    #     RL_exts,
    #     marker="P",
    #     s=70,
    #     color="tab:green",
    #     zorder=10,
    #     label="Rieke&Lebofsky 1985",
    # )

    # Rieke&Lebofsky 1985 powerlaw (taken from Martin&Whittet 1990)
    # this curve is far from the data points of Rieke&Lebofsky 1985
    # RL85 = PowerLaw1D(x_0=0.55, amplitude=1.26, alpha=1.62)
    # ax.plot(waves, RL85(waves), lw=1.5, ls="--", label="Rieke&Lebofsky 1985")

    # van de Hulst No. 15, probably not useful
    # VDH_exts = np.array(
    #    [1.555, 1.329, 1, 0.738, 0.469, 0.246, 0.155, 0.0885, 0.045, 0.033]
    # )
    # ax.scatter(MW_waves, VDH_exts, marker="X", s=60, label="van de Hulst No. 15")

    # Maiz Apellaniz et al. 2020, for RC stars, not useful, don't know what amplitude to use
    # MA20 = PowerLaw1D(x_0=1, amplitude=0.4, alpha=2.27)
    # ax.plot(waves, MA20(waves), lw=1.5, ls="--", label="Maiz Apellaniz+2020")

    # average curves from dust_extinction package
    x = np.arange(0.75, 5.2, 0.01) * u.micron

    models = [RL85_MWGC, G21_MWAvg]
    styles = ["--", ":"]
    colors = ["tab:green", "tab:purple"]
    labels = ["Rieke & Lebofsky 1985", "Gordon et al. 2021"]
    for i, cmodel in enumerate(models):
        ext_model = cmodel()
        (indxs,) = np.where(
            np.logical_and(
                1 / x.value >= ext_model.x_range[0], 1 / x.value <= ext_model.x_range[1]
            )
        )
        yvals = ext_model(x[indxs])
        ax.plot(
            x[indxs],
            yvals,
            lw=1.7,
            ls=styles[i],
            color=colors[i],
            label=labels[i],
            alpha=0.8,
        )

    # add R(V)-dependent average curves for R(V)=3.1
    RV_dep = Table.read(table_path + "RV_depV.txt", format="ascii")
    # ax.plot(RV_dep["wavelength"], RV_dep["intercept"])

    # zoom the residual plot
    res_ax = fig.axes[1]
    res_ax.set_ylim(-0.025, 0.025)

    # finalize and save the figure
    ax.set_ylim(-0.05, 0.6)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[i] for i in [0, 2, 1, 3]],
        [labels[i] for i in [0, 2, 1, 3]],
        fontsize=18,
    )
    fig.savefig(outpath + "average_ext.pdf", bbox_inches="tight")


if __name__ == "__main__":
    inpath = "/Users/mdecleir/Documents/NIR_ext/Data/"
    outpath = "/Users/mdecleir/spex_nir_extinction/Figures/"
    table_path = "/Users/mdecleir/spex_nir_extinction/Tables/"

    # plotting settings for uniform plots
    fs = 20
    plt.rc("font", size=fs)
    plt.rc("xtick", top=True, direction="in", labelsize=fs)
    plt.rc("ytick", direction="in", labelsize=fs)
    plt.rc("xtick.major", width=1, size=8)
    plt.rc("ytick.major", width=1, size=8)

    plot_extinction_curves(inpath, outpath)
    plot_average_curve(inpath, table_path, outpath)
