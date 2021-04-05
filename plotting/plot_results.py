# This script creates tables and plots of the results of the fitting for the NIR extinction paper (Decleir et al. 2021).

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from astropy.table import Table
import pandas as pd

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


def compare_AV_lit():
    # Obtain AVs from the literature
    CCM89 = pd.read_table(
        "/Users/mdecleir/spex_nir_extinction/data/Cardelli+89_tab1.dat", sep="\s"
    )
    CCM89["AV"] = CCM89["RV"] * CCM89["E(B-V)"]
    print(CCM89)


if __name__ == "__main__":
    # define the input and output path and the names of the star pairs in the format "reddenedstarname_comparisonstarname"
    inpath = "/Users/mdecleir/Documents/NIR_ext/Data/"
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
        "HD034921_HD214680",
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

    # create tables
    table_results(inpath, table_path, starpair_list, flagged)

    # create plots
    # plot_param_triangle(inpath, outpath, starpair_list, flagged)
    # plot_rv_dep(inpath, outpath, starpair_list, flagged)
    # compare_AV_lit()
