# This script plots the results of the fitting

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

from measure_extinction.extdata import ExtData


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


def plot_param_triangle(starpair_list, flagged):
    amplitudes, amp_l, amp_u, alphas, alpha_l, alpha_u, AVs, AV_l, AV_u, RVs = (
        np.zeros(len(starpair_list)) for i in range(10)
    )
    flags = np.zeros(len(starpair_list), dtype=bool)

    # retrieve the fitting results
    for i, starpair in enumerate(starpair_list):
        extdata = ExtData("%s%s_ext.fits" % (path, starpair.lower()))
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
    plt.savefig("Figures/params.pdf", bbox_inches="tight")


def plot_rv_dep(inpath, outpath, starpair_list, flagged):
    AVs, RVs = (np.zeros(len(starpair_list)) for i in range(2))
    waves, exts, uncs = [], [], []
    wave_list = [1, 2, 3, 4, 5]
    alavs = np.full((len(wave_list), len(starpair_list)), np.nan)

    # retrieve the information for all stars
    for i, starpair in enumerate(starpair_list):
        if starpair in flagged:
            continue

        # retrieve R(V)
        extdata = ExtData("%s%s_ext.fits" % (inpath, starpair.lower()))
        RVs[i] = extdata.columns["RV"][0]

        # transform the curve from E(lambda-V) to A(lambda)/A(V)
        extdata.trans_elv_alav()

        # get the good data in flat arrays
        (flat_waves, flat_exts, flat_exts_unc) = extdata.get_fitdata(
            ["SpeX_SXD", "SpeX_LXD"]
        )

        # get the A(lambda)/A(V) at certain wavelengths
        for j, wave in enumerate(wave_list):
            indx = np.abs(flat_waves.value - wave).argmin()
            if (np.abs(flat_waves[indx].value - wave)) < 0.01:
                alavs[j][i] = flat_exts[indx]

    # plot A(lambda)/A(V) vs. R(V) at certain wavelengths
    fig, ax = plt.subplots(len(wave_list), figsize=(7, len(wave_list) * 4), sharex=True)
    for j, wave in enumerate(wave_list):
        ax[j].scatter(RVs, alavs[j])
        rho, p = stats.spearmanr(RVs, alavs[j], nan_policy="omit")
        ax[j].text(
            0.95,
            0.9,
            r"$\rho =$" + "{:1.2f}".format(rho),
            fontsize=12,
            horizontalalignment="right",
            transform=ax[j].transAxes,
        )
        ax[j].set_ylabel("$A($" + str(wave) + "$\mu m)/A(V)$")

    # finalize the plot
    plt.xlabel("R(V)")
    plt.subplots_adjust(hspace=0)
    plt.savefig(outpath + "RV_dep.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # define the input and output path and the names of the star pairs in the format "reddenedstarname_comparisonstarname"
    inpath = "/Users/mdecleir/Documents/NIR_ext/Data/"
    outpath = "/Users/mdecleir/spex_nir_extinction/Figures/"
    starpair_list = [
        "HD017505_HD214680",
        "BD+56d524_HD034816",
        "HD013338_HD031726",
        "HD014250_HD042560",
        "HD014422_HD214680",
        "HD014956_HD188209",
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

    # create plot
    plot_param_triangle(starpair_list, flagged)
    plot_rv_dep(inpath, outpath, starpair_list, flagged)
