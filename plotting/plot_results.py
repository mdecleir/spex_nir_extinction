# This script creates tables and plots of the results of the fitting for the NIR extinction paper (Decleir et al. 2021).

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from astropy.table import Table
import pandas as pd
from collections import Counter
from matplotlib.lines import Line2D

from measure_extinction.extdata import ExtData


def table_results(inpath, outpath, starpair_list, flagged):
    """
    Create tables with the fitting results:
        - One to save as a text file
        - One in the aastex format

    Parameters
    ----------
    inpath : string
        Path to the input data files

    outpath : string
        Path to save the tables

    starpair_list : list of strings
        List of star pairs to include in the tables, in the format "reddenedstarname_comparisonstarname" (no spaces)

    flagged : list of strings
        List of star pairs to exclude from the tables, in the format "reddenedstarname_comparisonstarname" (no spaces)

    Returns
    -------
    Tables with fitting results
    """
    # create empty tables
    table_txt = Table(
        names=(
            "reddened",
            "comparison",
            "amplitude",
            "ampl. left unc.",
            "ampl. right unc.",
            "alpha",
            "alpha left unc.",
            "alpha right unc.",
            "AV",
            "AV left unc.",
            "AV right unc.",
            "EBV",
            "EBV left unc.",
            "EBV right unc.",
            "RV",
            "RV left unc.",
            "RV right unc.",
        ),
        dtype=(
            "str",
            "str",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
        ),
    )
    table_lat = Table(
        names=(
            "reddened",
            "comparison",
            "amplitude",
            r"$\alpha$",
            "A(V)",
            "E(B-V)",
            "R(V)",
        ),
        dtype=("str", "str", "str", "str", "str", "str", "str"),
    )

    # retrieve the fitting results for all stars
    for starpair in starpair_list:
        if starpair in flagged:
            continue
        extdata = ExtData("%s%s_ext.fits" % (inpath, starpair.lower()))
        table_txt.add_row(
            (
                starpair.split("_")[0],
                starpair.split("_")[1],
                extdata.model["params"][0].value,
                extdata.model["params"][0].unc_minus,
                extdata.model["params"][0].unc_plus,
                extdata.model["params"][2].value,
                extdata.model["params"][2].unc_minus,
                extdata.model["params"][2].unc_plus,
                extdata.columns["AV"][0],
                extdata.columns["AV"][1],
                extdata.columns["AV"][2],
                extdata.columns["EBV"][0],
                extdata.columns["EBV"][1],
                extdata.columns["EBV"][2],
                extdata.columns["RV"][0],
                extdata.columns["RV"][1],
                extdata.columns["RV"][2],
            )
        )
        table_lat.add_row(
            (
                starpair.split("_")[0],
                starpair.split("_")[1],
                "${:.2f}".format(extdata.model["params"][0].value)
                + "_{-"
                + "{:.3f}".format(extdata.model["params"][0].unc_minus)
                + "}^{+"
                + "{:.3f}".format(extdata.model["params"][0].unc_plus)
                + "}$",
                "${:.2f}".format(extdata.model["params"][2].value)
                + "_{-"
                + "{:.3f}".format(extdata.model["params"][2].unc_minus)
                + "}^{+"
                + "{:.3f}".format(extdata.model["params"][2].unc_plus)
                + "}$",
                "${:.2f}".format(extdata.columns["AV"][0])
                + "_{-"
                + "{:.3f}".format(extdata.columns["AV"][1])
                + "}^{+"
                + "{:.3f}".format(extdata.columns["AV"][2])
                + "}$",
                "${:.2f}".format(extdata.columns["EBV"][0])
                + "_{-"
                + "{:.3f}".format(extdata.columns["EBV"][1])
                + "}^{+"
                + "{:.3f}".format(extdata.columns["EBV"][2])
                + "}$",
                "${:.2f}".format(extdata.columns["RV"][0])
                + "_{-"
                + "{:.3f}".format(extdata.columns["RV"][1])
                + "}^{+"
                + "{:.3f}".format(extdata.columns["RV"][2])
                + "}$",
            )
        )

    # write the tables to files
    table_txt.write(
        outpath + "fitting_results.dat",
        format="ascii.commented_header",
        overwrite=True,
    )

    table_lat.write(
        outpath + "fitting_results.tex",
        format="aastex",
        latexdict={
            "tabletype": "deluxetable*",
            "caption": r"MCMC fitting results for the extinction curves: the amplitude and index $\alpha$ of the powerlaw, and A(V) are directly obtained from the fitting, while E(B-V) is obtained from the observations, and R(V) is calculated as R(V)=A(V)/E(B-V). \label{tab:fit_results}",
        },
        overwrite=True,
    )


def plot_params(ax, x, y, x_err=None, y_err=None, flagged=None):
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt=".k", zorder=0)
    print(x)
    rho, p = stats.spearmanr(x, y)
    ax.text(
        0.95,
        0.9,
        r"$\rho =$" + "{:1.2f}".format(rho),
        fontsize=12,
        horizontalalignment="right",
        transform=ax.transAxes,
    )
    # indicate the flagged curves in red
    if flagged is not None:
        ax.scatter(x[flagged], y[flagged], color="r", s=10)
        rho2, p2 = stats.spearmanr(x[~flagged], y[~flagged])
        ax.text(
            0.95,
            0.8,
            r"$\rho =$" + "{:1.2f}".format(rho2),
            color="green",
            fontsize=12,
            horizontalalignment="right",
            transform=ax.transAxes,
        )


def plot_param_triangle(inpath, outpath, starpair_list, flagged):
    amplitudes, amp_l, amp_u, alphas, alpha_l, alpha_u, AVs, AV_l, AV_u, RVs = (
        np.zeros(len(starpair_list)) for i in range(10)
    )
    flags = np.zeros(len(starpair_list), dtype=bool)

    # retrieve the fitting results
    for i, starpair in enumerate(starpair_list):
        extdata = ExtData("%s%s_ext.fits" % (inpath, starpair.lower()))
        amplitudes[i] = extdata.model["params"][0].value
        amp_l[i] = extdata.model["params"][0].unc_minus
        amp_u[i] = extdata.model["params"][0].unc_plus
        alphas[i] = extdata.model["params"][2].value
        alpha_l[i] = extdata.model["params"][2].unc_minus
        alpha_u[i] = extdata.model["params"][2].unc_plus
        AVs[i] = extdata.model["params"][3].value
        AV_l[i] = extdata.model["params"][3].unc_minus
        AV_u[i] = extdata.model["params"][3].unc_plus
        RVs[i] = extdata.columns["RV"][0]
        if starpair in flagged:
            flags[i] = True
        # TODO: add RV uncertainties!!

    # create the plot
    fig, ax = plt.subplots(3, 3, figsize=(10, 10), sharex="col", sharey="row")
    fs = 16

    # plot alpha vs. amplitude
    plot_params(ax[0, 0], amplitudes, alphas, (amp_l, amp_u), (alpha_l, alpha_u), flags)

    # plot A(V) vs. amplitude
    plot_params(ax[1, 0], amplitudes, AVs, (amp_l, amp_u), (AV_l, AV_u), flags)

    # plot R(V) vs. amplitude
    plot_params(ax[2, 0], amplitudes, RVs, (amp_l, amp_u), flagged=flags)

    # plot A(V) vs. alpha
    plot_params(ax[1, 1], alphas, AVs, (alpha_l, alpha_u), (AV_l, AV_u), flags)

    # plot R(V) vs. alpha
    plot_params(ax[2, 1], alphas, RVs, (alpha_l, alpha_u), flagged=flags)

    # plot R(V) vs. A(V)
    plot_params(ax[2, 2], AVs, RVs, (AV_l, AV_u), flagged=flags)

    # finalize the plot
    ax[0, 0].set_ylabel(r"$\alpha$", fontsize=fs)
    ax[1, 0].set_ylabel("A(V)", fontsize=fs)
    ax[2, 0].set_ylabel("R(V)", fontsize=fs)
    ax[2, 0].set_xlabel("amplitude", fontsize=fs)
    ax[2, 1].set_xlabel(r"$\alpha$", fontsize=fs)
    ax[2, 2].set_xlabel("A(V)", fontsize=fs)
    ax[0, 1].axis("off")
    ax[0, 2].axis("off")
    ax[1, 2].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(outpath + "params.pdf", bbox_inches="tight")


def compare_AV_lit(ext_path, lit_path, starpair_list):
    """
    Function to create a plot with the comparison of calculated A(V) values and values from the literature

    Parameters
    ----------
    ext_path : string
        Path to the extinction data

    lit_path : string
        Path to the literature data

    starpair_list : list of strings
        List of star pairs to include in the plot, in the format "reddenedstarname_comparisonstarname" (no spaces)
    """
    # Obtain A(V) values from the literature
    # Cardelli, Clayton and Mathis 1989
    CCM89 = pd.read_table(
        lit_path + "Cardelli+89_tab1.dat", sep="\s+", index_col="Star"
    )
    CCM89["A(V)"] = CCM89["R(V)"] * CCM89["E(B-V)"]

    # Valencic, Clayton and Gordon 2004
    VCG04 = Table.read(
        lit_path + "Valencic+04_tab4.dat", format="ascii.cds"
    ).to_pandas()
    VCG04.loc[:355, "Name"] = "HD" + VCG04.loc[:355, "Name"].str.replace(
        "HD ", ""
    ).str.zfill(6)
    VCG04.loc[355:392, "Name"] = VCG04.loc[355:392, "Name"].str.replace(" ", "d")
    VCG04.set_index("Name", inplace=True)

    # Fitzpatrick et al. 2019

    # Gordon et al. 2021
    G21 = pd.read_table(lit_path + "Gordon+21_tab5.dat", sep="\s+", index_col="Star")

    # plot the literature A(V) values against the calculated values
    fig, ax = plt.subplots(figsize=(6, 6))
    for starpair in starpair_list:
        extdata = ExtData("%s%s_ext.fits" % (ext_path, starpair.lower()))
        red_star = starpair.split("_")[0]
        if red_star in CCM89.index:
            ax.scatter(
                extdata.columns["AV"][0],
                CCM89.loc[red_star, "A(V)"],
                color="tab:blue",
            )
        if red_star in VCG04.index:
            ax.scatter(
                extdata.columns["AV"][0],
                VCG04.loc[red_star, "A(V)"],
                color="tab:orange",
                marker="d",
            )
        if red_star.lower() in G21.index:
            ax.scatter(
                extdata.columns["AV"][0],
                G21.loc[red_star.lower(), "A(V)"],
                color="tab:green",
                marker="P",
            )
        ax.errorbar(
            extdata.columns["AV"][0],
            extdata.columns["AV"][0],
            # yerr=([extdata.columns["AV"][1]], [extdata.columns["AV"][2]]),
            yerr=([0.2], [0.2]),
            color="k",
            lw=0.5,
            alpha=0.6,
        )

    handle1 = Line2D([], [], lw=0, color="tab:blue", marker="o")
    handle2 = Line2D([], [], lw=0, color="tab:orange", marker="d")
    handle3 = Line2D([], [], lw=0, color="tab:green", marker="P")
    labels = ["Cardelli+1989", "Valencic+2004", "Gordon+2021"]
    handles = [handle1, handle2, handle3]
    ax.plot([1, 5.7], [1, 5.7], color="k", ls="--")
    ax.set_xlim(0.9, 5.8)
    ax.set_ylim(0.9, 5.8)
    ax.set_aspect("equal", adjustable="box")
    plt.legend(handles, labels, fontsize=fs * 0.8)
    plt.xlabel("A(V) from this work")
    plt.ylabel("A(V) from the literature")
    plt.savefig(plot_path + "AV_comparison.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # define the input and output path and the names of the star pairs in the format "reddenedstarname_comparisonstarname"
    inpath = "/Users/mdecleir/Documents/NIR_ext/Data/"
    data_path = "/Users/mdecleir/spex_nir_extinction/data/"
    plot_path = "/Users/mdecleir/spex_nir_extinction/Figures/"
    table_path = "/Users/mdecleir/spex_nir_extinction/Tables/"
    starpair_list = [
        "BD+56d524_HD034816",
        "HD013338_HD031726",
        "HD014250_HD042560",
        "HD014422_HD214680",
        "HD014956_HD188209",
        "HD017505_HD214680",
        "HD029309_HD042560",
        "HD029647_HD042560",
        # "HD034921_HD214680",
        "HD037020_HD034816",
        "HD037022_HD034816",
        "HD037023_HD034816",
        "HD037061_HD034816",
        "HD038087_HD034816",
        "HD052721_HD091316",
        "HD156247_HD031726",
        "HD166734_HD188209",
        "HD183143_HD188209",
        "HD185418_HD034816",
        "HD192660_HD204172",
        "HD204827_HD204172",
        "HD206773_HD003360",
        "HD229238_HD214680",
        "HD283809_HD003360",
        "HD294264_HD034759",
    ]

    flagged = [
        "HD037023_HD034816",
        "HD034921_HD214680",
        "HD037020_HD034816",
        "HD037022_HD034816",
        "HD052721_HD091316",
        "HD206773_HD003360",
        "HD014250_HD042560",
        "HD014422_HD214680",
    ]

    # subtract the flagged stars from the star pair list
    good_stars = list((Counter(starpair_list) - Counter(flagged)).elements())

    # settings for the plotting
    fs = 18
    plt.rc("font", size=fs)
    plt.rc("xtick", direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", direction="in", labelsize=fs * 0.8)
    plt.rc("xtick.major", width=1, size=8)
    plt.rc("ytick.major", width=1, size=8)

    # create tables
    # table_results(inpath, table_path, starpair_list, flagged)

    # create plots
    # plot_param_triangle(inpath, plot_path, starpair_list, flagged)
    compare_AV_lit(inpath, data_path, good_stars)
