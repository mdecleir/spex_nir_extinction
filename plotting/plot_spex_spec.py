#!/usr/bin/env python
# This script creates plots with all SpeX spectra for the NIR extinction paper (Decleir et al. 2021).

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from measure_extinction.plotting.plot_spec import plot_multi_spectra
from measure_extinction.stardata import StarData


def plot_comp_spectra(inpath, outpath, stars):
    # specify the offsets and angles for the star names
    offsets = [
        0.2,
        0.18,
        0.13,
        0.1,
        0.15,
        0.12,
        0.1,
        0.1,
        0.08,
        0.28,
        0.23,
        0.15,
        0.11,
        0.14,
        0.25,
    ]
    angles = [12, 10, 8, 8, 7, 7, 5, 10, 10, 18, 17, 12, 12, 14, 16]

    # plot the spectra
    fig, ax = plot_multi_spectra(
        stars,
        inpath,
        mlam4=True,
        range=[0.75, 5.55],
        norm_range=[0.95, 1.05],
        spread=True,
        exclude=["IRS", "I", "L", "IRAC1", "IRAC2", "WISE1", "WISE2"],
        text_offsets=offsets,
        text_angles=angles,
        pdf=True,
        outname="comp_stars.pdf",
    )

    # annotate the main sequence and giant star spectra with text
    ax.annotate(
        "main sequence",
        xy=(5.27, 3.7),
        xytext=(5.4, 3.7),
        fontsize=fs,
        ha="center",
        va="center",
        rotation=-90,
        arrowprops=dict(arrowstyle="-[, widthB=15, lengthB=1.8", lw=3),
    )
    ax.annotate(
        "giants and supergiants",
        xy=(5.27, 8.6),
        xytext=(5.4, 8.6),
        fontsize=fs,
        ha="center",
        va="center",
        rotation=-90,
        arrowprops=dict(arrowstyle="-[, widthB=7.5, lengthB=1.8", lw=3),
    )

    ax.set_ylim(0.5, 10.3)
    fig.savefig(outpath + "comp_stars.pdf", bbox_inches="tight")


def plot_red_spectra(inpath, outpath, stars):
    # specify the offsets and angles for the star names
    offsets = [
        0.19,
        0.22,
        0.21,
        0.15,
        0.22,
        0.20,
        0.24,
        0.17,
        0.2,
        0.21,
        0.19,
        0.17,
        0.24,
        0.26,
        0.24,
    ]
    angles = [16, 17, 19, 17, 25, 25, 30, 32, 31, 32, 34, 40, 46, 50, 60]

    # plot the spectra
    fig, ax = plot_multi_spectra(
        stars,
        inpath,
        mlam4=True,
        range=[0.75, 5.4],
        norm_range=[0.95, 1.05],
        spread=True,
        exclude=["IRS", "STIS_Opt", "I", "L", "IRAC1", "IRAC2", "WISE1", "WISE2"],
        class_offset=False,
        text_offsets=offsets,
        text_angles=angles,
        pdf=True,
        outname="red_stars.pdf",
    )
    ax.set_ylim(0.6, 16)
    fig.savefig(outpath + "red_stars.pdf", bbox_inches="tight")


def plot_unused_spectra(inpath, outpath, stars):
    # specify the offsets and angles for the star names
    offsets = [0.18, 0.25, 0.08, 0.25, 0.14, -0.03, 0.19, 0.22, 0.35, 0.25]
    angles = [11, 19, 22, 22, 25, 30, 35, 39, 37, 39]
    fig, ax = plot_multi_spectra(
        stars,
        inpath,
        mlam4=True,
        range=[0.75, 5.4],
        norm_range=[0.95, 1.05],
        spread=True,
        exclude=["IRS", "I", "L", "IRAC1", "IRAC2", "WISE1", "WISE2"],
        class_offset=False,
        text_offsets=offsets,
        text_angles=angles,
        pdf=True,
        outname="bad_stars.pdf",
    )
    ax.set_ylim(0.6, 12.5)
    fig.set_figheight(14)
    fig.savefig(outpath + "bad_stars.pdf", bbox_inches="tight")


def plot_color_color(stars, bands, div):
    """
    Make a color-color plot

    Parameters
    ----------
    stars : list of strings
        List of stars

    bands : list of strings
        List of bands

    div : float
        Location of division line

    Returns
    -------
    IR color-color plot
    """
    # create the figure
    plt.rc("axes", linewidth=0.8)
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, star in enumerate(stars):
        # categorize the star
        if star in comp_stars:
            color = "black"
            marker = "P"
        elif star in red_stars:
            color = "green"
            marker = "d"
        elif star == "HD014250":
            color = "purple"
            marker = "s"
        else:
            color = "red"
            marker = "o"

        # obtain the photometry
        star_data = StarData("%s.dat" % star.lower(), path=inpath)
        band_data = star_data.data["BAND"]
        mags = np.full(len(bands), np.nan)
        errs = np.full(len(bands), np.nan)
        for j, band in enumerate(bands):
            if band == "K_S":
                band = "K"
            if band in band_data.get_band_names():
                mags[j] = band_data.get_band_mag(band)[0]
                errs[j] = band_data.get_band_mag(band)[1]

        # plot colors
        ax.errorbar(
            mags[0] - mags[1],
            mags[2] - mags[3],
            xerr=np.sqrt((errs[0] ** 2) + (errs[1] ** 2)),
            yerr=np.sqrt((errs[2] ** 2) + (errs[3] ** 2)),
            marker=marker,
            color=color,
            markersize=6,
            markeredgewidth=0,
            elinewidth=0.8,
            alpha=0.7,
        )

    # finalize and save the plot
    ax.axhline(div, color="grey", ls=":")
    ax.set_xlabel(r"$" + bands[0] + "-" + bands[1] + "$", fontsize=fs)
    ax.set_ylabel(r"$" + bands[2] + "-$" + bands[3], fontsize=fs)
    ax.tick_params(width=1)

    labels = ["comparison", "reddened", "windy", "bad"]
    handle1 = Line2D([], [], lw=1, color="black", marker="P", alpha=0.7)
    handle2 = Line2D([], [], lw=1, color="green", marker="d", alpha=0.7)
    handle3 = Line2D([], [], lw=1, color="red", marker="o", alpha=0.7)
    handle4 = Line2D([], [], lw=1, color="purple", marker="s", alpha=0.7)
    handles = [handle1, handle2, handle3, handle4]
    ax.legend(handles, labels, fontsize=fs * 0.8)
    fig.savefig(outpath + "wind_" + bands[3] + ".pdf", bbox_inches="tight")


def plot_wind(inpath, outpath, comp_stars, red_stars, bad_stars):
    """
    Make IR color-color plots to separate windy stars

    Parameters
    ----------
    inpath : string
        Path to the input data files

    outpath : string
        Path to save the plots

    comp_stars : list of strings
        List of comparison stars

    red_stars : list of strings
        List of reddened stars

    bad_stars : list of strings
        List of bad stars

    Returns
    -------
    IR color-color plots
    """
    stars = comp_stars + red_stars + bad_stars

    # plot K-WISE4 vs. J-K
    plot_color_color(stars, ["J", "K_S", "K_S", "WISE4"], 1)
    # plot K-MIPS24 vs. J-K
    plot_color_color(stars, ["J", "K_S", "K_S", "MIPS24"], 0.6)
    # plot K-IRAC1 vs. J-K
    plot_color_color(stars, ["J", "K_S", "K_S", "IRAC1"], 0.2)
    # plot K-IRAC2 vs. J-K
    plot_color_color(stars, ["J", "K_S", "K_S", "IRAC2"], 0.25)
    # plot K-WISE1 vs. J-K
    plot_color_color(stars, ["J", "K_S", "K_S", "WISE1"], 0.11)
    # plot K-WISE2 vs. J-K
    plot_color_color(stars, ["J", "K_S", "K_S", "WISE2"], 0.4)


if __name__ == "__main__":
    inpath = "/Users/mdecleir/Documents/NIR_ext/Data/"
    outpath = "/Users/mdecleir/spex_nir_extinction/Figures/"
    # define the names of the comparison stars (first the main sequence stars and then the giant stars, sorted by spectral type from B8 to O4)
    comp_stars = [
        "HD034759",
        "HD032630",
        "HD042560",
        "HD003360",
        "HD031726",
        "HD034816",
        "HD036512",
        "HD214680",
        "HD047839",
        "HD164794",
        "HD078316",
        "HD051283",
        "HD091316",
        "HD204172",
        "HD188209",
    ]
    # define the names of the reddened stars (sorted by A(V) from low to high)
    red_stars = [
        "HD156247",
        "HD185418",
        "HD013338",
        "HD017505",
        "HD038087",
        "BD+56d524",
        "HD029309",
        "HD192660",
        "HD204827",
        "HD037061",
        "HD014956",
        "HD229238",
        "HD029647",
        "HD183143",
        "HD283809",
    ]
    # define the names of the reddened stars that cannot be used to measure an extinction curve (sorted by steepness)
    bad_stars = [
        "HD014250",
        "HD037022",
        "HD052721",
        "HD037023",
        "HD206773",
        "HD034921",
        "HD037020",
        "HD294264",
        "HD166734",
        "HD014422",
    ]
    # plotting settings for uniform plots
    fs = 20
    plt.rc("font", size=fs)
    plt.rc("xtick", top=True, direction="in", labelsize=fs)
    plt.rc("ytick", direction="in", labelsize=fs)
    plt.rc("xtick.major", width=1, size=8)
    plt.rc("ytick.major", width=1, size=8)

    plot_comp_spectra(inpath, outpath, comp_stars)
    plot_red_spectra(inpath, outpath, red_stars)
    plot_unused_spectra(inpath, outpath, bad_stars)
    plot_wind(inpath, outpath, comp_stars, red_stars, bad_stars)
