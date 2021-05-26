# This script creates tables and plots of the results of the fitting for the NIR extinction paper (Decleir et al. 2021).

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from astropy.table import Table
import pandas as pd
from collections import Counter
from matplotlib.lines import Line2D

from measure_extinction.extdata import ExtData


def table_results(inpath, outpath, diffuse, dense):
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

    diffuse : list of strings
        List of diffuse star pairs to include in the tables, in the format "reddenedstarname_comparisonstarname" (no spaces)

    dense : list of strings
        List of dense star pairs to include in the tables, in the format "reddenedstarname_comparisonstarname" (no spaces)

    Returns
    -------
    Tables with fitting results
    """
    # create empty tables for the diffuse sightlines
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
            "$S$",
            r"$\alpha$",
            "$A(V)$",
            "$E(B-V)$",
            "$R(V)$",
        ),
        dtype=("str", "str", "str", "str", "str", "str", "str"),
    )

    # retrieve the fitting results for all diffuse sightlines and add them to the table
    for starpair in diffuse:
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
                "{:.2f}".format(extdata.model["params"][0].value)
                + "_{-"
                + "{:.3f}".format(extdata.model["params"][0].unc_minus)
                + "}^{+"
                + "{:.3f}".format(extdata.model["params"][0].unc_plus)
                + "}",
                "{:.2f}".format(extdata.model["params"][2].value)
                + "_{-"
                + "{:.3f}".format(extdata.model["params"][2].unc_minus)
                + "}^{+"
                + "{:.3f}".format(extdata.model["params"][2].unc_plus)
                + "}",
                "{:.2f}".format(extdata.columns["AV"][0])
                + "_{-"
                + "{:.3f}".format(extdata.columns["AV"][1])
                + "}^{+"
                + "{:.3f}".format(extdata.columns["AV"][2])
                + "}",
                "{:.2f}".format(extdata.columns["EBV"][0])
                + "_{-"
                + "{:.3f}".format(extdata.columns["EBV"][1])
                + "}^{+"
                + "{:.3f}".format(extdata.columns["EBV"][2])
                + "}",
                "{:.2f}".format(extdata.columns["RV"][0])
                + "_{-"
                + "{:.3f}".format(extdata.columns["RV"][1])
                + "}^{+"
                + "{:.3f}".format(extdata.columns["RV"][2])
                + "}",
            )
        )

    # write the tables to files
    table_txt.write(
        outpath + "fitting_results_diff.dat",
        format="ascii.commented_header",
        overwrite=True,
    )

    table_lat.write(
        outpath + "fitting_results_diff.tex",
        format="aastex",
        col_align="ll|CCCCC",
        latexdict={
            "tabletype": "deluxetable*",
            "caption": r"MCMC fitting results for the 13 diffuse extinction curves: the amplitude $S$ and index $\alpha$ of the powerlaw, and $A(V)$ are directly obtained from the fitting, while $E(B-V)$ is obtained from the observations, and $R(V)$ is calculated as $R(V)=A(V)/E(B-V)$. \label{tab:fit_results_diff}",
        },
        overwrite=True,
    )

    # create empty tables for the dense sightlines
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
            "$S$",
            r"$\alpha$",
            "$A(V)$",
            "$E(B-V)$",
            "$R(V)$",
        ),
        dtype=("str", "str", "str", "str", "str", "str", "str"),
    )

    # retrieve the fitting results for all dense sightlines and add them to the table
    for starpair in dense:
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
                "{:.2f}".format(extdata.model["params"][0].value)
                + "_{-"
                + "{:.3f}".format(extdata.model["params"][0].unc_minus)
                + "}^{+"
                + "{:.3f}".format(extdata.model["params"][0].unc_plus)
                + "}",
                "{:.2f}".format(extdata.model["params"][2].value)
                + "_{-"
                + "{:.3f}".format(extdata.model["params"][2].unc_minus)
                + "}^{+"
                + "{:.3f}".format(extdata.model["params"][2].unc_plus)
                + "}",
                "{:.2f}".format(extdata.columns["AV"][0])
                + "_{-"
                + "{:.3f}".format(extdata.columns["AV"][1])
                + "}^{+"
                + "{:.3f}".format(extdata.columns["AV"][2])
                + "}",
                "{:.2f}".format(extdata.columns["EBV"][0])
                + "_{-"
                + "{:.3f}".format(extdata.columns["EBV"][1])
                + "}^{+"
                + "{:.3f}".format(extdata.columns["EBV"][2])
                + "}",
                "{:.2f}".format(extdata.columns["RV"][0])
                + "_{-"
                + "{:.3f}".format(extdata.columns["RV"][1])
                + "}^{+"
                + "{:.3f}".format(extdata.columns["RV"][2])
                + "}",
            )
        )

    # write the tables to files
    table_txt.write(
        outpath + "fitting_results_dense.dat",
        format="ascii.commented_header",
        overwrite=True,
    )

    table_lat.write(
        outpath + "fitting_results_dense.tex",
        format="aastex",
        col_align="ll|CCCCC",
        latexdict={
            "tabletype": "deluxetable*",
            "caption": r"MCMC fitting results for the 2 dense extinction curves: the amplitude $S$ and index $\alpha$ of the powerlaw, and $A(V)$ are directly obtained from the fitting, while $E(B-V)$ is obtained from the observations, and $R(V)$ is calculated as $R(V)=A(V)/E(B-V)$. \label{tab:fit_results_dense}",
        },
        overwrite=True,
    )


def plot_params(
    ax,
    dense,
    x,
    y,
    x_err=None,
    y_err=None,
):
    # give the dense sightlines a different color and marker
    ax.errorbar(
        x[dense],
        y[dense],
        xerr=(x_err[0][dense], x_err[1][dense]),
        yerr=(y_err[0][dense], y_err[1][dense]),
        fmt="s",
        color="magenta",
        markersize=4,
        zorder=0,
        label="dense",
    )
    ax.errorbar(
        x[~dense],
        y[~dense],
        xerr=(x_err[0][~dense], x_err[1][~dense]),
        yerr=(y_err[0][~dense], y_err[1][~dense]),
        fmt="ok",
        markersize=4,
        zorder=0,
        label="diffuse",
    )
    rho, p = stats.spearmanr(x, y)
    ax.text(
        0.95,
        0.9,
        r"$\rho =$" + "{:1.2f}".format(rho),
        fontsize=12,
        horizontalalignment="right",
        transform=ax.transAxes,
    )


def plot_param_triangle(inpath, outpath, diffuse, dense):
    (
        amplitudes,
        amp_min,
        amp_plus,
        alphas,
        alpha_min,
        alpha_plus,
        AVs,
        AV_min,
        AV_plus,
        RVs,
        RV_min,
        RV_plus,
    ) = (np.zeros(len(diffuse + dense)) for i in range(12))
    dense_bool = np.full(len(diffuse + dense), False)

    # retrieve the fitting results for all sightlines
    for i, starpair in enumerate(diffuse + dense):
        extdata = ExtData("%s%s_ext.fits" % (inpath, starpair.lower()))
        amplitudes[i] = extdata.model["params"][0].value
        amp_min[i] = extdata.model["params"][0].unc_minus
        amp_plus[i] = extdata.model["params"][0].unc_plus
        alphas[i] = extdata.model["params"][2].value
        alpha_min[i] = extdata.model["params"][2].unc_minus
        alpha_plus[i] = extdata.model["params"][2].unc_plus
        AVs[i] = extdata.columns["AV"][0]
        AV_min[i] = extdata.columns["AV"][1]
        AV_plus[i] = extdata.columns["AV"][2]
        RVs[i] = extdata.columns["RV"][0]
        RV_min[i] = extdata.columns["RV"][1]
        RV_plus[i] = extdata.columns["RV"][2]

        # flag the dense sightlines
        if starpair in dense:
            dense_bool[i] = True

    # create the plot
    fig, ax = plt.subplots(3, 3, figsize=(10, 10), sharex="col", sharey="row")

    # plot alpha vs. amplitude
    plot_params(
        ax[0, 0],
        dense_bool,
        amplitudes,
        alphas,
        (amp_min, amp_plus),
        (alpha_min, alpha_plus),
    )

    # plot A(V) vs. amplitude
    plot_params(
        ax[1, 0], dense_bool, amplitudes, AVs, (amp_min, amp_plus), (AV_min, AV_plus)
    )

    # plot R(V) vs. amplitude
    plot_params(
        ax[2, 0], dense_bool, amplitudes, RVs, (amp_min, amp_plus), (RV_min, RV_plus)
    )

    # plot A(V) vs. alpha
    plot_params(
        ax[1, 1], dense_bool, alphas, AVs, (alpha_min, alpha_plus), (AV_min, AV_plus)
    )

    # plot R(V) vs. alpha
    plot_params(
        ax[2, 1], dense_bool, alphas, RVs, (alpha_min, alpha_plus), (RV_min, RV_plus)
    )

    # plot R(V) vs. A(V)
    plot_params(ax[2, 2], dense_bool, AVs, RVs, (AV_min, AV_plus), (RV_min, RV_plus))

    # finalize the plot
    ax[0, 0].set_ylabel(r"$\alpha$", fontsize=fs)
    ax[1, 0].set_ylabel("$A(V)$", fontsize=fs)
    ax[2, 0].set_ylabel("$R(V)$", fontsize=fs)
    ax[2, 0].set_xlabel("$S$", fontsize=fs)
    ax[2, 1].set_xlabel(r"$\alpha$", fontsize=fs)
    ax[2, 2].set_xlabel("$A(V)$", fontsize=fs)
    ax[0, 1].axis("off")
    ax[0, 2].axis("off")
    ax[1, 2].axis("off")
    handles, labels = ax[0, 0].get_legend_handles_labels()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.legend(handles, labels, bbox_to_anchor=(0.88, 0.88))
    plt.savefig(outpath + "params.pdf", bbox_inches="tight")


def compare_AV_lit(ext_path, lit_path, plot_path, starpair_list):
    """
    Function to create a plot with the comparison of calculated A(V) values and values from the literature

    Parameters
    ----------
    ext_path : string
        Path to the extinction data

    lit_path : string
        Path to the literature data

    plot_path : string
       Path to save the plot

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

    # Gordon et al. 2009
    G09 = pd.read_table(
        lit_path + "Gordon+09_tab2.dat", sep="\s+", skiprows=[0, 1], index_col="#Name"
    )

    # Gordon et al. 2021
    G21 = pd.read_table(lit_path + "Gordon+21_tab5.dat", sep="\s+", index_col="Star")

    # plot the literature A(V) values against the calculated values
    fig, ax = plt.subplots(figsize=(6, 6))
    for starpair in starpair_list:
        extdata = ExtData("%s%s_ext.fits" % (ext_path, starpair.lower()))
        red_star = starpair.split("_")[0]
        if red_star in CCM89.index:
            ax.errorbar(
                extdata.columns["AV"][0],
                CCM89.loc[red_star, "A(V)"],
                xerr=([extdata.columns["AV"][1]], [extdata.columns["AV"][2]]),
                marker="o",
                color="tab:blue",
                elinewidth=1,
            )
        if red_star in VCG04.index:
            ax.errorbar(
                extdata.columns["AV"][0],
                VCG04.loc[red_star, "A(V)"],
                xerr=([extdata.columns["AV"][1]], [extdata.columns["AV"][2]]),
                marker="d",
                color="tab:orange",
                elinewidth=1,
            )
        if red_star in G09.index:
            ax.errorbar(
                extdata.columns["AV"][0],
                G09.loc[red_star, "AV"],
                xerr=([extdata.columns["AV"][1]], [extdata.columns["AV"][2]]),
                marker="^",
                color="tab:green",
                elinewidth=1,
            )
        if red_star.lower() in G21.index:
            ax.errorbar(
                extdata.columns["AV"][0],
                G21.loc[red_star.lower(), "A(V)"],
                xerr=([extdata.columns["AV"][1]], [extdata.columns["AV"][2]]),
                marker="P",
                color="tab:red",
                elinewidth=1,
            )

    # create the legend
    handle1 = Line2D([], [], lw=0, color="tab:blue", marker="o")
    handle2 = Line2D([], [], lw=0, color="tab:orange", marker="d")
    handle3 = Line2D([], [], lw=0, color="tab:green", marker="^")
    handle4 = Line2D([], [], lw=0, color="tab:red", marker="P")
    labels = [
        "Cardelli et al. (1989)",
        "Valencic et al. (2004)",
        "Gordon et al. (2009)",
        "Gordon et al. (2021)",
    ]
    handles = [handle1, handle2, handle3, handle4]

    # finalize and save the plot
    ax.plot([0.5, 5.7], [0.5, 5.7], color="k", ls="--")
    ax.set_xlim(0.45, 5.8)
    ax.set_ylim(0.45, 5.8)
    ax.set_aspect("equal", adjustable="box")
    plt.legend(handles, labels, fontsize=fs * 0.75)
    plt.xlabel(r"$A(V)$ from this work")
    plt.ylabel(r"$A(V)$ from the literature")
    plt.savefig(plot_path + "AV_comparison.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # define the input and output path and the names of the star pairs in the format "reddenedstarname_comparisonstarname"
    inpath = "/Users/mdecleir/Documents/NIR_ext/Data/"
    data_path = "/Users/mdecleir/spex_nir_extinction/data/"
    plot_path = "/Users/mdecleir/spex_nir_extinction/Figures/"
    table_path = "/Users/mdecleir/spex_nir_extinction/Tables/"

    diffuse = [
        "BD+56d524_HD034816",
        "HD013338_HD031726",
        "HD014250_HD042560",
        "HD014422_HD214680",
        "HD014956_HD188209",
        "HD017505_HD214680",
        "HD029309_HD042560",
        # "HD034921_HD214680",
        "HD037020_HD034816",
        "HD037022_HD034816",
        "HD037023_HD034816",
        "HD037061_HD034816",
        "HD038087_HD051283",
        "HD052721_HD091316",
        "HD156247_HD042560",
        "HD166734_HD031726",
        "HD183143_HD188209",
        "HD185418_HD034816",
        "HD192660_HD204172",
        "HD204827_HD003360",
        "HD206773_HD003360",
        "HD229238_HD214680",
        "HD294264_HD034759",
    ]

    dense = ["HD029647_HD042560", "HD283809_HD003360"]

    flagged = [
        "HD014250_HD042560",
        "HD014422_HD214680",
        "HD034921_HD214680",
        "HD037020_HD034816",
        "HD037022_HD034816",
        "HD037023_HD034816",
        "HD052721_HD091316",
        "HD166734_HD031726",
        "HD206773_HD003360",
        "HD294264_HD034759",
    ]

    # subtract the flagged stars from the star pair list
    good_diffuse = list((Counter(diffuse) - Counter(flagged)).elements())
    good_dense = list((Counter(dense) - Counter(flagged)).elements())

    # settings for the plotting
    fs = 18
    plt.rc("font", size=fs)
    plt.rc("xtick", direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", direction="in", labelsize=fs * 0.8)
    plt.rc("xtick.major", width=1, size=8)
    plt.rc("ytick.major", width=1, size=8)

    # create tables
    # table_results(inpath, table_path, good_diffuse, good_dense)

    # create plots
    plot_param_triangle(inpath, plot_path, good_diffuse, good_dense)
    # compare_AV_lit(inpath, data_path, plot_path, good_diffuse + good_dense)
