#!/usr/bin/env python
# This script creates plots with all SpeX spectra for the NIR extinction paper (Decleir et al. 2021).

from measure_extinction.plotting.plot_spec import plot_multi_spectra


def plot_comp_spectra(path):
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
        path,
        mlam4=True,
        range=[0.75, 5.6],
        norm_range=[1, 1.05],
        spread=True,
        exclude=["IRS"],
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
        ha="center",
        va="center",
        rotation=-90,
        arrowprops=dict(arrowstyle="-[, widthB=17, lengthB=1.8", lw=3),
    )
    ax.annotate(
        "giants and supergiants",
        xy=(5.37, 8.6),
        xytext=(5.5, 8.6),
        ha="center",
        va="center",
        rotation=-90,
        arrowprops=dict(arrowstyle="-[, widthB=9, lengthB=1.8", lw=3),
    )

    fig.savefig("../Figures/paper/comp_stars.pdf", bbox_inches="tight")


def plot_red_spectra(path):
    # define the names of the reddened stars (first the main sequence stars and then the giant stars, sorted by A(V) from low to high)
    stars = [
        "HD014250",
        "HD156247",
        "BD+56d524",
        "HD185418",
        "HD013338",
        "HD017505",
        "HD204827",
        "HD192660",
        "HD038087",
        "HD029309",
        "HD037022",
        "HD037023",
        "HD014956",
        "HD052721",
        "HD037061",
        "HD229238",
        "HD206773",
        "HD034921",
        "HD029647",
        "HD037020",
        "HD294264",
        "HD166734",
        "HD014422",
        "HD183143",
        "HD283809",
    ]

    # plot the spectra
    plot_multi_spectra(
        stars,
        path,
        mlam4=True,
        range=[0.75, 5.5],
        norm_range=[1, 1.1],
        spread=True,
        exclude=["IRS", "STIS_Opt"],
        pdf=True,
        outname="red_stars.pdf",
    )


if __name__ == "__main__":
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    plot_comp_spectra(path)
    plot_red_spectra(path)
