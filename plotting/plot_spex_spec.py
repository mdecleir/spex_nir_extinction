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
        0.15,
        0.11,
        0.1,
        0.12,
        0.06,
        0.1,
        0.08,
        0.08,
        0.28,
        0.23,
        0.15,
        0.11,
        0.12,
        0.25,
    ]
    angles = [12, 10, 8, 8, 7, 7, 5, 10, 10, 18, 15, 12, 12, 14, 16]

    # plot the spectra
    fig, ax = plot_multi_spectra(
        stars,
        inpath,
        mlam4=True,
        range=[0.75, 5.6],
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
        xy=(5.37, 3.7),
        xytext=(5.5, 3.7),
        fontsize=fs,
        ha="center",
        va="center",
        rotation=-90,
        arrowprops=dict(arrowstyle="-[, widthB=15, lengthB=1.8", lw=3),
    )
    ax.annotate(
        "giants and supergiants",
        xy=(5.37, 8.7),
        xytext=(5.5, 8.7),
        fontsize=fs,
        ha="center",
        va="center",
        rotation=-90,
        arrowprops=dict(arrowstyle="-[, widthB=8, lengthB=1.8", lw=3),
    )

    fig.savefig(outpath + "comp_stars.pdf", bbox_inches="tight")


def plot_red_spectra(inpath, outpath):
    # define the names of the reddened stars (sorted by A(V) from low to high)
    stars = [
        "HD156247",
        "HD185418",
        "HD013338",
        "HD017505",
        "BD+56d524",
        "HD192660",
        "HD204827",
        "HD029309",
        "HD014956",
        "HD038087",
        "HD037061",
        "HD294264",
        "HD229238",
        "HD029647",
        "HD183143",
        "HD166734",
        "HD283809",
    ]

    # specify the offsets and angles for the star names
    offsets = [
        0.11,
        0.22,
        0.21,
        0.17,
        0.22,
        0.2,
        0.22,
        0.16,
        0.23,
        0.17,
        0.18,
        0.24,
        0.2,
        0.28,
        0.25,
        0.27,
        0.25,
    ]
    angles = [15, 17, 19, 15, 25, 33, 34, 32, 34, 26, 32, 42, 40, 46, 48, 52, 62]

    # plot the spectra
    fig, ax = plot_multi_spectra(
        stars,
        inpath,
        mlam4=True,
        range=[0.75, 5.5],
        norm_range=[0.95, 1.05],
        spread=True,
        exclude=["IRS", "STIS_Opt", "I", "L", "IRAC1", "IRAC2", "WISE1", "WISE2"],
        text_offsets=offsets,
        text_angles=angles,
        class_offset=False,
        pdf=True,
        outname="red_stars.pdf",
    )
    ax.set_ylim(0.6, 16.9)
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
        "HD014422",
    ]
    # specify the offsets and angles for the star names
    offsets = [0.15, 0.17, 0.05, 0.21, 0.12, -0.1, 0.16, 0.19]
    angles = [8, 15, 18, 18, 19, 23, 28, 33]
    fig, ax = plot_multi_spectra(
        stars,
        inpath,
        mlam4=True,
        range=[0.75, 5.4],
        norm_range=[0.95, 1.05],
        spread=True,
        exclude=["IRS", "I", "L", "IRAC1", "IRAC2", "WISE1", "WISE2"],
        text_offsets=offsets,
        text_angles=angles,
        pdf=True,
        outname="bad_stars.pdf",
    )
    ax.set_ylim(0.6, 11.3)
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
