# This script plots the results of the fitting

from matplotlib import pyplot as plt
from scipy import stats

from measure_extinction.extdata import ExtData


def plot_params(ax, x, y, x_err=None, y_err=None):
    ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt=".k")
    rho, p = stats.spearmanr(x, y)
    ax.text(
        0.95,
        0.9,
        r"$\rho =$" + "{:1.2f}".format(rho),
        fontsize=12,
        horizontalalignment="right",
        transform=ax.transAxes,
    )


def plot_param_triangle(starpair_list):
    amplitudes, amp_l, amp_u, alphas, alpha_l, alpha_u, AVs, AV_l, AV_u, RVs = (
        [] for i in range(10)
    )

    # retrieve the fitting results
    for starpair in starpair_list:
        extdata = ExtData("%s%s_ext.fits" % (path, starpair.lower()))
        amplitudes.append(extdata.model["params"][0].value)
        amp_l.append(extdata.model["params"][0].unc_minus)
        amp_u.append(extdata.model["params"][0].unc_plus)
        alphas.append(extdata.model["params"][2].value)
        alpha_l.append(extdata.model["params"][2].unc_minus)
        alpha_u.append(extdata.model["params"][2].unc_plus)
        AVs.append(extdata.model["params"][3].value)
        AV_l.append(extdata.model["params"][3].unc_minus)
        AV_u.append(extdata.model["params"][3].unc_plus)

        RVs.append(extdata.columns["RV"][0])
        # TODO: add RV uncertainties!!

    # create the plot
    fig, ax = plt.subplots(3, 3, figsize=(10, 10), sharex="col", sharey="row")
    fs = 16

    # plot alpha vs. amplitude
    plot_params(ax[0, 0], amplitudes, alphas, (amp_l, amp_u), (alpha_l, alpha_u))

    # plot A(V) vs. amplitude
    plot_params(ax[1, 0], amplitudes, AVs, (amp_l, amp_u), (AV_l, AV_u))

    # plot R(V) vs. amplitude
    plot_params(ax[2, 0], amplitudes, RVs, (amp_l, amp_u))

    # plot A(V) vs. alpha
    plot_params(ax[1, 1], alphas, AVs, (alpha_l, alpha_u), (AV_l, AV_u))

    # plot R(V) vs. alpha
    plot_params(ax[2, 1], alphas, RVs, (alpha_l, alpha_u))

    # plot R(V) vs. A(V)
    plot_params(ax[2, 2], AVs, RVs, (AV_l, AV_u))

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


if __name__ == "__main__":
    # define the path and the names of the star pairs in the format "reddenedstarname_comparisonstarname"
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
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

    # create plot
    plot_param_triangle(starpair_list)
