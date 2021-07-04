#!/usr/bin/env python
# This script creates plots with all SpeX spectra for the NIR extinction paper (Decleir et al. 2021).

from matplotlib import pyplot as plt

from measure_extinction.plotting.plot_spec import plot_multi_spectra


def plot_comp_spectra(inpath, outpath):
    # define the names of the comparison stars (first the main sequence stars and then the giant stars, sorted by spectral type from B8 to O4)
    stars = [
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


def plot_red_spectra(inpath, outpath):
    # define the names of the reddened stars (sorted by A(V) from low to high)
    stars = [
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


def plot_unused_spectra(inpath, outpath):
    # define the names of the reddened stars that cannot be used to measure an extinction curve (sorted by steepness)
    stars = [
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


if __name__ == "__main__":
    inpath = "/Users/mdecleir/Documents/NIR_ext/Data/"
    outpath = "/Users/mdecleir/spex_nir_extinction/Figures/"
    # plotting settings for uniform plots
    fs = 20
    plt.rc("font", size=fs)
    plt.rc("xtick", top=True, direction="in", labelsize=fs)
    plt.rc("ytick", direction="in", labelsize=fs)
    plt.rc("xtick.major", width=1, size=8)
    plt.rc("ytick.major", width=1, size=8)

    plot_comp_spectra(inpath, outpath)
    plot_red_spectra(inpath, outpath)
    plot_unused_spectra(inpath, outpath)
