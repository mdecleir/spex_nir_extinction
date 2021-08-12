#!/usr/bin/env python
# This script creates plots with SpeX extinction curves for the NIR extinction paper (Decleir et al. 2021).

import numpy as np
import astropy.units as u
from astropy.io import fits

from astropy.stats import sigma_clipped_stats
from astropy.modeling.powerlaws import PowerLaw1D
from astropy.table import Table
from matplotlib import pyplot as plt

from measure_extinction.plotting.plot_ext import (
    plot_extinction,
    plot_multi_extinction,
    plot_average,
)
from measure_extinction.extdata import ExtData

from dust_extinction.averages import RL85_MWGC, I05_MWAvg, G21_MWAvg
from dust_extinction.grain_models import D03, ZDA04, C11, J13
from dust_extinction.parameter_averages import F19, CCM89

# gamma function (wavelength dependent width, replacing the FWHM)
def gamma(x, x_o=1, gamma_o=1, asym=1):
    return 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))


# asymmetric Gaussian
def gauss_asymmetric(x, scale=1, x_o=1, gamma_o=1, asym=1):
    # gamma replaces FWHM, so stddev=gamma/(2sqrt(2ln2))
    y = scale * np.exp(
        -((x - x_o) ** 2)
        / (2 * (gamma(x, x_o, gamma_o, asym) / (2 * np.sqrt(2 * np.log(2)))) ** 2)
    )
    return y


# "asymmetric" Drude
def drude_asymmetric(x, scale=1, x_o=1, gamma_o=1, asym=1):
    y = (
        scale
        * (gamma(x, x_o, gamma_o, asym) / x_o) ** 2
        / ((x / x_o - x_o / x) ** 2 + (gamma(x, x_o, gamma_o, asym) / x_o) ** 2)
    )
    return y


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
        # "HD014250_HD031726",
        # "HD014422_HD214680",
        "HD014956_HD214680",
        "HD017505_HD214680",
        "HD029309_HD042560",
        "HD029647_HD034759",
        # "HD034921_HD214680",
        # "HD037020_HD034816",
        # "HD037022_HD034816",
        # "HD037023_HD036512",
        "HD037061_HD034816",
        "HD038087_HD051283",
        # "HD052721_HD091316",
        "HD156247_HD042560",
        # "HD166734_HD031726",
        "HD183143_HD188209",
        "HD185418_HD034816",
        "HD192660_HD214680",
        "HD204827_HD003360",
        # "HD206773_HD047839",
        "HD229238_HD214680",
        "HD283809_HD003360",
        # "HD294264_HD051283",
    ]

    # plot the extinction curves in E(lambda-V)
    plot_multi_extinction(
        starpairs,
        inpath,
        range=[0.76, 5.5],
        spread=True,
        exclude=["IRS"],
        pdf=True,
    )

    # specify the offsets and angles for the star names
    offsets = [
        0.04,
        0.04,
        0.0,
        0.04,
        0.0,
        0.05,
        0.03,
        0.0,
        -0.07,
        0.02,
        0.01,
        0.05,
        0.01,
        0.05,
        0.02,
    ]
    angles = [
        -40,
        -46,
        -36,
        -30,
        -46,
        -46,
        -42,
        -42,
        -44,
        -46,
        -42,
        -46,
        -46,
        -46,
        -46,
    ]

    # plot the extinction curves in A(lambda)/A(V)
    fig, ax = plot_multi_extinction(
        starpairs,
        inpath,
        alax=True,
        range=[0.76, 5.3],
        spread=True,
        exclude=["IRS", "I", "L", "IRAC1", "IRAC2", "WISE1", "WISE2"],
        text_offsets=offsets,
        text_angles=angles,
        pdf=True,
    )

    # finalize and save the plot
    ax.set_ylim(-0.1, 4.15)
    fig.savefig(outpath + "ext_curves_alav.pdf", bbox_inches="tight")


def plot_average_curve(inpath, outpath):
    """
    Plot the average extinction curve:
     - together with literature curves
     - together with dust grain models

    Parameters
    ----------
    inpath : string
        Path to the data files

    outpath : string
        Path to save the plot

    Returns
    -------
    Plots with the average diffuse extinction curve
    """
    # with literature curves
    # plot the average extinction curve
    fig, ax = plot_average(
        inpath,
        fitmodel=True,
        range=[0.75, 4.9],
        exclude=["IRS", "BAND"],
        pdf=True,
    )

    # add literature curves
    waves = np.arange(0.75, 5, 0.001)

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
        alpha=0.9,
        label="Martin & Whittet (1990)",
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

    # add average curves from the dust_extinction package
    x = np.arange(0.75, 5.0, 0.001) * u.micron
    models = [RL85_MWGC, I05_MWAvg, G21_MWAvg]
    styles = ["--", "-.", ":"]
    colors = ["tab:green", "tab:purple", "tab:cyan"]
    labels = [
        "Rieke & Lebofsky (1985)",
        "Indebetouw et al. (2005)",
        "Gordon et al. (2021)",
    ]
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
            alpha=0.9,
        )

    # finalize and save the figure
    ax.set_ylim(-0.03, 0.6)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[i] for i in [0, 2, 1, 3, 4]],
        [labels[i] for i in [0, 2, 1, 3, 4]],
        fontsize=fs * 0.9,
    )
    fig.savefig(outpath + "average_ext.pdf", bbox_inches="tight")

    # with dust grain models
    # plot the average extinction curve
    fig, ax = plot_average(
        inpath,
        range=[0.75, 4.9],
        exclude=["IRS", "BAND"],
        pdf=True,
    )

    # dust grain models from dust_extinction package
    models = [D03, ZDA04, C11, J13]
    modelnames = [
        "MWRV31",
        "BARE-GR-S",
        "MWRV31",
        "MWRV31",
    ]
    styles = [
        "-",
        "--",
        "-.",
        ":",
    ]
    labels = [
        "Draine (2003)",
        "Zubko et al. (2004)",
        "Compiègne et al. (2011)",
        "Jones et al. (2013)",
    ]
    for i, (cmodel, cname) in enumerate(zip(models, modelnames)):
        ext_model = cmodel(cname)
        yvals = ext_model(x)
        ax.plot(x, yvals, lw=1.7, ls=styles[i], label=labels[i], alpha=0.9)

    # finalize and save the plot
    ax.set_ylim(-0.03, 0.6)
    plt.legend(fontsize=fs * 0.9)
    fig.savefig(outpath + "average_ext_mod.pdf", bbox_inches="tight")


def plot_ave_UV(inpath, outpath):
    """
    Plot the average extinction curve in the UV, together with literature curves

    Parameters
    ----------
    inpath : string
        Path to the data files

    outpath : string
        Path to save the plot

    Returns
    -------
    Plot with the average UV diffuse extinction curve
    """
    # plot the average extinction curve
    fig, ax = plot_average(
        inpath,
        range=[0.11, 0.323],
        exclude=["BAND"],
        pdf=True,
    )

    # add average curves from the dust_extinction package
    x = np.arange(0.11, 0.33, 0.001) * u.micron
    models = [CCM89, F19]
    styles = ["--", "-"]
    colors = ["tab:purple", "tab:green"]
    labels = ["Cardelli et al. (1989)", "Fitzpatrick et al. (2019)"]
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
        )

    # finalize and save the plot
    plt.legend(fontsize=fs * 0.9)
    fig.savefig(outpath + "average_ext_UV.pdf", bbox_inches="tight")


def plot_ave_res(inpath, outpath):
    """
    Plot the residuals of the average curve fit separately

    Parameters
    ----------
    inpath : string
        Path to the data files

    outpath : string
        Path to save the plot

    Returns
    -------
    Plot with the residuals of the average curve fit
    """
    # read in the average extinction curve
    average = ExtData(inpath + "average_ext.fits")

    # plot the residuals
    fig, ax = plt.subplots(
        2,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={
            "height_ratios": [1, 3],
            "hspace": 0,
        },
    )
    waves = average.model["waves"]
    residuals = average.model["residuals"]
    ax[1].scatter(waves, residuals, s=1.5, color="k")

    # calculate the standard deviation of the residuals in different wavelength ranges
    ranges = [(0.79, 1.37), (1.4, 1.82), (1.92, 2.54), (2.85, 4.05), (4.55, 4.8)]
    for range in ranges:
        mask = (waves > range[0]) & (waves < range[1])
        mean, median, stddev = sigma_clipped_stats(residuals[mask])
        ax[1].hlines(
            y=(stddev, -stddev),
            xmin=range[0],
            xmax=range[1],
            colors="magenta",
            ls="--",
            lw=2,
            zorder=5,
        )

    # indicate hydrogen lines and jumps
    ax[1].annotate(
        "Pa\njump",
        xy=(0.85, 0.0185),
        xytext=(0.85, 0.023),
        fontsize=0.7 * fs,
        ha="center",
        va="center",
        color="blue",
        arrowprops=dict(arrowstyle="-[, widthB=.5, lengthB=.5", lw=1, color="blue"),
    )
    ax[1].annotate(
        "Br\njump",
        xy=(1.46, 0.0165),
        xytext=(1.46, 0.021),
        fontsize=0.7 * fs,
        ha="center",
        va="center",
        color="blue",
        arrowprops=dict(arrowstyle="-[, widthB=.55, lengthB=.5", lw=1, color="blue"),
    )

    lines = [
        0.9017385,
        0.9231547,
        0.9548590,
        1.0052128,
        1.0941090,
        1.282159,
        1.5264708,
        1.5443139,
        1.5560699,
        1.5704952,
        1.5884880,
        1.6113714,
        1.6411674,
        1.6811111,
        1.7366850,
        2.166120,
        2.3544810,
        2.3828230,
        2.3924675,
        2.4163852,
        3.0392022,
    ]

    ax[1].vlines(x=lines, ymin=-0.017, ymax=0.017, color="blue", lw=0.5, alpha=0.5)

    # add the atmospheric transmission curve
    hdulist = fits.open("/Users/mdecleir/spex_nir_extinction/data/atran2000.fits")
    data = hdulist[0].data
    ax[0].plot(data[0], data[1], color="k", alpha=0.7, lw=0.5)

    # finalize and save the plot
    ax[1].axhline(ls="-", c="k", alpha=0.5)
    plt.xlim(0.75, 4.9)
    plt.ylim(-0.026, 0.026)
    ax[0].set_ylabel("Atmospheric\ntransmission", fontsize=0.8 * fs)
    ax[1].set_xlabel(r"$\lambda$ [$\mu m$]", fontsize=fs)
    ax[1].set_ylabel("residual $A(\lambda)/A(V)$", fontsize=fs)
    fig.savefig(outpath + "average_res.pdf", bbox_inches="tight")


def plot_features(starpair, inpath, outpath):
    """
    Plot the extinction features separately

    Parameters
    ----------
    starpair : string
        Name of the star pair for which to plot the extinction features, in the format "reddenedstarname_comparisonstarname" (no spaces)

    inpath : string
        Path to the data files

    outpath : string
        Path to save the plot

    Returns
    -------
    Plot with extinction features
    """
    # plot the measured extinction curve with the fitted model
    fig, ax = plot_extinction(
        starpair,
        inpath,
        fitmodel=True,
        range=[2.2, 4.05],
        exclude=["IRS", "BAND"],
        pdf=True,
    )

    # retrieve the model parameters
    extdata = ExtData("%s%s_ext.fits" % (inpath, starpair.lower()))
    waves = extdata.model["waves"]

    # for 1 feature
    (
        amplitude,
        x_0,
        alpha,
        scale_1,
        x_1,
        gamma_1,
        asym_1,
        AV,
    ) = extdata.model["params"]

    # for 2 features
    # (
    #     amplitude,
    #     x_0,
    #     alpha,
    #     scale_1,
    #     x_1,
    #     gamma_1,
    #     asym_1,
    #     scale_2,
    #     x_2,
    #     gamma_2,
    #     asym_2,
    #     AV,
    # ) = extdata.model["params"]

    # plot the power law
    powerlaw = amplitude * AV * waves ** (-alpha) - AV
    ax.plot(
        waves,
        powerlaw,
        ls="--",
        lw=2,
        color="olive",
        alpha=0.6,
        label="power law",
    )

    # plot the first feature
    # ax.plot(
    #     waves,
    #     powerlaw
    #     + drude_asymmetric(
    #         waves,
    #         scale=scale_1 * AV,
    #         x_o=x_1,
    #         gamma_o=gamma_1,
    #         asym=asym_1,
    #     ),
    #     ls=":",
    #     label="Mod. Drude",
    # )

    # plot the second feature (if applicable)
    # ax.plot(
    #     waves,
    #     powerlaw
    #     + drude_asymmetric(
    #         waves,
    #         scale=scale_2 * AV,
    #         x_o=x_2,
    #         gamma_o=gamma_2,
    #         asym=asym_2,
    #     ),
    #     ls="-.",
    #     label="Mod. Drude 2",
    # )

    # zoom the residual plot
    res_ax = fig.axes[1]
    res_ax.set_ylim(-0.04, 0.04)

    # finalize and save the figure
    plt.setp(ax, title="")
    del ax.texts[0]
    ax.legend(loc=3)
    if starpair == "HD029647_HD034759":
        ax.set_ylim(-3.44, -3.2)
    ax.text(0.7, 0.9, starpair.split("_")[0], transform=ax.transAxes, fontsize=1.5 * fs)
    fig.savefig(outpath + starpair + "_ext_features.pdf", bbox_inches="tight")


if __name__ == "__main__":
    inpath = "/Users/mdecleir/Documents/NIR_ext/Data/"
    outpath = "/Users/mdecleir/spex_nir_extinction/Figures/"

    # plotting settings for uniform plots
    fs = 20
    plt.rc("font", size=fs)
    plt.rc("xtick.major", width=1, size=10)
    plt.rc("ytick.major", width=1, size=10)
    plt.rc("xtick.minor", width=1, size=5)
    plt.rc("ytick.minor", width=1, size=5)
    plt.rc("axes.formatter", min_exponent=2)
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 0.8)

    # plot_extinction_curves(inpath, outpath)
    # plot_average_curve(inpath, outpath)
    # plot_ave_UV(inpath, outpath)
    # plot_ave_res(inpath, outpath)
    # plot_features("HD283809_HD003360", inpath, outpath)
    # plot_features("HD029647_HD034759", inpath, outpath)
