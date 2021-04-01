# This script calculates and fits the R(V) dependent relationship

import numpy as np

from matplotlib import pyplot as plt
from collections import Counter
from scipy import stats
from astropy.modeling import models, fitting

from measure_extinction.extdata import ExtData


def get_data(inpath, starpair_list):
    """
    Obtain the required data for all stars in starpair_list:
        - A(lambda)/A(V)
        - R(V)

    Parameters
    ----------
    inpath : string
        Path to the input data files

    starpair_list : list of strings
        List of star pairs for which to collect the data, in the format "reddenedstarname_comparisonstarname" (no spaces)

    Returns
    -------
    R(V) with uncertainties, A(lambda)/A(V) with uncertainties, wavelengths
    """
    RVs = np.zeros((len(starpair_list), 3))

    # determine the wavelengths at which to retrieve the extinction data
    extdata_model = ExtData("%s%s_ext.fits" % (inpath, starpair_list[0].lower()))
    waves = np.concatenate(
        (extdata_model.waves["SpeX_SXD"].value, extdata_model.waves["SpeX_LXD"].value)
    )
    alavs = np.full((len(waves), len(starpair_list)), np.nan)
    alav_uncs = np.full((len(waves), len(starpair_list)), np.nan)

    # retrieve the information for all stars
    for i, starpair in enumerate(starpair_list):
        # retrieve R(V)
        extdata = ExtData("%s%s_ext.fits" % (inpath, starpair.lower()))
        RVs[i] = np.array(extdata.columns["RV"])

        # transform the curve from E(lambda-V) to A(lambda)/A(V)
        extdata.trans_elv_alav()

        # get the good data in flat arrays
        (flat_waves, flat_exts, flat_exts_unc) = extdata.get_fitdata(
            ["SpeX_SXD", "SpeX_LXD"]
        )

        # ind1 = np.abs(flat_waves.value - 3).argmin()
        # print(flat_waves[ind1], flat_exts[ind1], extdata.columns["AV"][0])
        # # go from A(lambda)/A(V) to A(lambda)/A(1)
        # flat_exts = flat_exts / flat_exts[ind1]

        # retrieve A(lambda)/A(V) at all wavelengths
        for j, wave in enumerate(waves):
            if wave in flat_waves.value:
                alavs[j][i] = flat_exts[flat_waves.value == wave]
                alav_uncs[j][i] = flat_exts_unc[flat_waves.value == wave]

    return RVs, alavs, alav_uncs, waves


def plot_rv_dep(outpath, RVs, alavs, alav_uncs, waves, plot_waves, slopes, intercepts):
    """
    Plot the relationship between A(lambda)/A(V) and R(V)-3.1 at wavelengths "plot_waves"

    Parameters
    ----------
    outpath : string
        Path to save the plot

    RVs : np.ndarray
        Numpy array with R(V) values and uncertainties

    alavs : np.ndarray
        Numpy array with A(lambda)/A(V) values

    alav_uncs : np.ndarray
        Numpy array with A(lambda)/A(V) uncertainties

    waves : np.ndarray
        Numpy array with all wavelengths for which A(lambda)/A(V) is given

    plot_waves : list
        List with wavelengths for which to plot the relationship

    slopes : np.ndarray
        Numpy array with the slopes of the linear relationship

    intercepts : np.ndarray
        Numpy array with the intercepts of the linear relationship

    Returns
    -------
    Plot of A(lambda)/A(V) vs. R(V)-3.1 at wavelengths "plot_waves"
    """
    fig, ax = plt.subplots(
        len(plot_waves), figsize=(7, len(plot_waves) * 4), sharex=True
    )

    for j, wave in enumerate(plot_waves):
        indx = np.abs(waves - wave).argmin()
        # plot the data and the fitted line
        ax[j].errorbar(
            RVs[:, 0] - 3.1,
            alavs[indx],
            xerr=(RVs[:, 1], RVs[:, 2]),
            yerr=alav_uncs[indx],
            ms=10,
            fmt=".k",
            elinewidth=0.5,
        )
        ax[j].plot(
            np.arange(-1, 4, 0.1),
            slopes[indx] * np.arange(-1, 4, 0.1) + intercepts[indx],
            color="forestgreen",
            ls="--",
            alpha=0.6,
        )
        rho, p = stats.spearmanr(RVs[:, 0] - 3.1, alavs[indx], nan_policy="omit")
        ax[j].text(
            0.97,
            0.08,
            r"$\rho =$" + "{:1.2f}".format(rho),
            fontsize=fs * 0.8,
            horizontalalignment="right",
            transform=ax[j].transAxes,
        )
        ax[j].set_ylabel("$A($" + "{:1.2f}".format(wave) + "$\mu m)/A(V)$")

    # finalize the plot
    plt.xlabel("R(V) - 3.1")
    plt.subplots_adjust(hspace=0)
    plt.savefig(outpath + "RV_dep.pdf", bbox_inches="tight")


def fit_plot_rv_dep(inpath, outpath, starpair_list):
    """
    Fit and plot the relationship between A(lambda)/A(V) and R(V)

    Parameters
    ----------
    inpath : string
        Path to the input data files

    outpath : string
        Path to save the output

    starpair_list : list of strings
        List of star pairs to include in the fitting, in the format "reddenedstarname_comparisonstarname" (no spaces)
    """
    # collect the data to be fitted
    RVs, alavs, alav_uncs, waves = get_data(inpath, starpair_list)
    RV_vals = RVs[:, 0]

    # for every wavelength, fit a straight line through the A(lambda)/A(V) vs. R(V) data
    fit = fitting.LinearLSQFitter()
    line_func = models.Linear1D()
    slopes, intercepts, stds = np.full((3, len(waves)), np.nan)

    for j, wave in enumerate(waves):
        mask = ~np.isnan(alavs[j])
        npts = np.sum(mask)
        # require at least 5 data points for the fitting
        if npts < 5:
            continue
        fitted_line = fit(line_func, (RV_vals[mask] - 3.1), alavs[j][mask])
        std = np.sqrt(
            np.sum((fitted_line(RV_vals[mask] - 3.1) - alavs[j][mask]) ** 2)
            / (npts - 1)
        )
        slopes[j] = fitted_line.slope.value
        intercepts[j] = fitted_line.intercept.value
        stds[j] = std

    # plot A(lambda)/A(V) vs. R(V) at certain wavelengths
    plot_waves = [0.82, 1.6499686, 2.4502702, 3.5002365, 4.6994233]
    plot_rv_dep(outpath, RVs, alavs, alav_uncs, waves, plot_waves, slopes, intercepts)

    # plot the slopes and intercepts vs. wavelength
    fig, ax = plt.subplots(3, figsize=(9, 8), sharex=True)
    ax[0].scatter(waves, slopes, s=0.3)
    ax[1].scatter(waves, intercepts, s=0.3)
    ax[2].scatter(waves, stds, s=0.3)
    for wave in plot_waves:
        indx = np.abs(waves - wave).argmin()
        ax[0].scatter(wave, slopes[indx], color="tab:orange", marker="x")
        ax[1].scatter(wave, intercepts[indx], color="tab:orange", marker="x")
        ax[2].scatter(wave, stds[indx], color="tab:orange", marker="x")

    # fit the slopes and intercepts vs. wavelength
    # fit2 = fitting.LevMarLSQFitter()
    # powerlaw = models.PowerLaw1D(amplitude=0.3, alpha=1.6)
    # fit_slopes = fit2(powerlaw, waves, slopes)
    # fit_intercepts = fit2(powerlaw, waves, intercepts)
    # print(fit_intercepts)
    # # ax[0].plot(waves, fit_slopes(waves))
    # # ax[1].plot(waves, fit_intercepts(intercepts))
    # pl = models.PowerLaw1D(amplitude=0.35, alpha=1.6)
    # ax[1].plot(waves, pl(waves))

    # finalize the plot
    plt.xlabel(r"$\lambda$ [$\mu m$]")
    ax[0].set_ylim(-0.01, 0.075)
    ax[1].set_ylim(-0.03, 0.6)
    ax[2].set_ylim(0.01, 0.075)
    ax[0].set_ylabel("a")
    ax[1].set_ylabel("b")
    ax[2].set_ylabel(r"$\sigma$")
    ax[0].axhline(ls="--", color="k", lw=1, alpha=0.6)
    ax[1].axhline(ls="--", color="k", lw=1, alpha=0.6)
    plt.subplots_adjust(hspace=0)
    plt.savefig(outpath + "RV_slope_inter.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # define the input and output path and the names of the star pairs in the format "reddenedstarname_comparisonstarname"
    inpath = "/Users/mdecleir/Documents/NIR_ext/Data/"
    outpath = "/Users/mdecleir/spex_nir_extinction/Figures/"
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
    # list the problematic star pairs
    flagged = [
        "HD014422_HD214680",
        "HD014250_HD042560",
        "HD037023_HD034816",
        "HD034921_HD214680",
        "HD037020_HD034816",
        "HD037022_HD034816",
        "HD052721_HD091316",
        "HD206773_HD003360",
    ]

    # settings for the plotting
    fs = 18
    plt.rc("font", size=fs)
    plt.rc("xtick", direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", direction="in", labelsize=fs * 0.8)
    plt.rc("xtick.major", width=1, size=8)
    plt.rc("ytick.major", width=1, size=8)

    # subtract the flagged stars from the star pair list
    good_stars = list((Counter(starpair_list) - Counter(flagged)).elements())

    # fit and plot the RV dependence
    fit_plot_rv_dep(inpath, outpath, good_stars)

    # add CCM89 1/RV dependent relation (can only be used with 1/RV)
    # waves_CCM89 = [0.7, 0.9, 1.25, 1.6, 2.2, 3.4]
    # slopes_CCM89 = [-0.366, -0.6239, -0.3679, -0.2473, -0.1483, -0.0734]
    # intercepts_CCM89 = [0.8686, 0.68, 0.4008, 0.2693, 0.1615, 0.08]
    # ax[0].scatter(waves_CCM89, slopes_CCM89, s=5)
    # ax[1].scatter(waves_CCM89, intercepts_CCM89, s=5)
