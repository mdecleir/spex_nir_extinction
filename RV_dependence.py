# This script calculates and fits the R(V) dependent relationship

import numpy as np

from matplotlib import pyplot as plt
from collections import Counter
from scipy import stats
from astropy.modeling import models, fitting

from measure_extinction.extdata import ExtData


def fit_RV_dep(inpath, outpath, starpair_list):
    """
    Fit the relationship between A(lambda)/A(V) and R(V)

    Parameters
    ----------
    inpath : string
        Path to the input data files

    outpath : string
        Path to save the output

    starpair_list : list of strings
        List of star pairs to include in the fitting, in the format "reddenedstarname_comparisonstarname" (no spaces)
    """
    RVs = np.zeros(len(starpair_list))
    waves, exts, uncs = [], [], []

    # determine the wavelengths at which to calculate the R(V) relationship
    extdata = ExtData("%s%s_ext.fits" % (inpath, starpair_list[2].lower()))
    waves = np.concatenate(
        (extdata.waves["SpeX_SXD"].value, extdata.waves["SpeX_LXD"].value)
    )
    print(len(waves))

    wave_list = np.arange(0.8, 5.5, 0.01)
    alavs = np.full((len(wave_list), len(starpair_list)), np.nan)

    # retrieve the information for all stars
    for i, starpair in enumerate(starpair_list):
        # retrieve R(V)
        extdata = ExtData("%s%s_ext.fits" % (inpath, starpair.lower()))
        RVs[i] = extdata.columns["RV"][0]

        # transform the curve from E(lambda-V) to A(lambda)/A(V)
        extdata.trans_elv_alav()

        # get the good data in flat arrays
        (flat_waves, flat_exts, flat_exts_unc) = extdata.get_fitdata(
            ["SpeX_SXD", "SpeX_LXD"]
        )

        ind1 = np.abs(flat_waves.value - 3).argmin()
        # print(flat_waves[ind1], flat_exts[ind1], extdata.columns["AV"][0])
        # # go from A(lambda)/A(V) to A(lambda)/A(1)
        # flat_exts = flat_exts / flat_exts[ind1]

        # get the A(lambda)/A(V) at certain wavelengths
        for j, wave in enumerate(wave_list):
            indx = np.abs(flat_waves.value - wave).argmin()
            if (np.abs(flat_waves[indx].value - wave)) < 0.01:
                alavs[j][i] = flat_exts[indx]
    print(alavs)

    # for every wavelength, fit a straight line through the A(lambda)/A(V) vs. 1/R(V) data
    fit = fitting.LinearLSQFitter()
    line_func = models.Linear1D()
    slopes = []
    intercepts = []
    coeffs = []
    chi2s = []
    for j, wave in enumerate(wave_list):
        mask = ~np.isnan(alavs[j])
        fitted_line = fit(line_func, 1 / RVs[mask], alavs[j][mask])
        chi2 = np.sum((fitted_line(1 / RVs[mask]) - alavs[j][mask]) ** 2)
        # rho, p = stats.spearmanr(RVs, alavs[j], nan_policy="omit")
        slopes.append(fitted_line.slope.value)
        intercepts.append(fitted_line.intercept.value)
        # coeffs.append(rho)
        chi2s.append(chi2)

    # plot the slopes and intercepts vs. wavelength
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].scatter(wave_list, slopes, s=5)
    ax[1].scatter(wave_list, intercepts, s=5)

    # add CCM89 RV dependent relation
    waves = [0.7, 0.9, 1.25, 1.6, 2.2, 3.4]
    intercepts = [0.8686, 0.68, 0.4008, 0.2693, 0.1615, 0.08]
    slopes = [-0.366, -0.6239, -0.3679, -0.2473, -0.1483, -0.0734]
    ax[0].scatter(waves, slopes, s=5)
    ax[1].scatter(waves, intercepts, s=5)

    plt.xlabel("wavelength")
    ax[0].set_ylabel("slopes")
    ax[1].set_ylabel("intercepts")
    plt.subplots_adjust(hspace=0)
    plt.savefig(outpath + "RV_slope_inter.pdf", bbox_inches="tight")

    # plot A(lambda)/A(V) vs. R(V) at certain wavelengths
    lim_waves = [
        20,
        120,
        220,
        320,
        420,
    ]
    fig, ax = plt.subplots(len(lim_waves), figsize=(7, len(lim_waves) * 4), sharex=True)
    for j, indx in enumerate(lim_waves):
        # print(wave_list[indx], alavs[indx])
        mask = ~np.isnan(alavs[indx])
        fitted_line = fit(line_func, 1 / RVs[mask], alavs[indx][mask])
        ax[j].plot(1 / RVs, fitted_line(1 / RVs))
        ax[j].scatter(1 / RVs, alavs[indx])
        rho, p = stats.spearmanr(1 / RVs, alavs[indx], nan_policy="omit")
        ax[j].text(
            0.95,
            0.9,
            r"$\rho =$" + "{:1.2f}".format(rho),
            fontsize=12,
            horizontalalignment="right",
            transform=ax[j].transAxes,
        )
        ax[j].set_ylabel("$A($" + str(int(wave_list[indx])) + "$\mu m)/A(V)$")
    # finalize the plot
    plt.xlabel("1/R(V)")
    plt.subplots_adjust(hspace=0)
    plt.savefig(outpath + "RV_dep.pdf", bbox_inches="tight")


import emcee
import astropy.units as u
from astropy import uncertainty as unc


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

    # subtract the flagged stars from the star pair list
    good_stars = list((Counter(starpair_list) - Counter(flagged)).elements())

    # fit the RV dependence
    fit_RV_dep(inpath, outpath, good_stars)

    #     mcmcfile = bfile.replace(".fits", ".h5")
    #     reader = emcee.backends.HDFBackend(mcmcfile)
    #     nsteps, nwalkers = reader.get_log_prob().shape
    #     samples = reader.get_chain(discard=int(0.4 * nsteps), flat=True)
    #
    #     avs_dist = unc.Distribution(samples[:, -1])
    #     av_per = avs_dist.pdf_percentiles([16.0, 50.0, 84.0])
    #     avs[k] = av_per[1]
    #     avs_unc[1, k] = av_per[2] - av_per[1]
    #     avs_unc[0, k] = av_per[1] - av_per[0]
    #     # print(avs_dist.pdf_percentiles([33., 50., 87.]))
    #
    #     (indxs,) = np.where(
    #         (cext.waves["BAND"] > 0.4 * u.micron)
    #         & (cext.waves["BAND"] < 0.5 * u.micron)
    #     )
    #     ebvs_dist = unc.normal(
    #         cext.exts["BAND"][indxs[0]],
    #         std=cext.uncs["BAND"][indxs[0]],
    #         n_samples=avs_dist.n_samples,
    #     )
    #     ebvs[k] = ebvs_dist.pdf_mean()
    #     ebvs_unc[k] = ebvs_dist.pdf_std()
    #
    #     rvs_dist = avs_dist / ebvs_dist
    #     rv_per = rvs_dist.pdf_percentiles([16.0, 50.0, 84.0])
    #     rvs[k] = rv_per[1]
    #     rvs_unc[1, k] = rv_per[2] - rv_per[1]
    #     rvs_unc[0, k] = rv_per[1] - rv_per[0]
    #
    #     (indxs,) = np.where(cext.names["BAND"] == args.band)
    #     exts[k] = (cext.exts["BAND"][indxs[0]] / avs[k]) + 1.0
    #     exts_unc[k] = cext.uncs["BAND"][indxs[0]]
    #
    # # plots
    # fontsize = 14
    #
    # font = {"size": fontsize}
    #
    # matplotlib.rc("font", **font)
    #
    # matplotlib.rc("lines", linewidth=1)
    # matplotlib.rc("axes", linewidth=2)
    # matplotlib.rc("xtick.major", width=2)
    # matplotlib.rc("xtick.minor", width=2)
    # matplotlib.rc("ytick.major", width=2)
    # matplotlib.rc("ytick.minor", width=2)
    #
    # figsize = (5.5, 5.0)
    # fig, ax = pyplot.subplots(figsize=figsize)
    #
    # diffuse = []
    # for tname in extnames:
    #     if tname == "hd283809":
    #         diffuse.append(False)
    #     elif tname == "hd029647":
    #         diffuse.append(False)
    #     else:
    #         diffuse.append(True)
    # diffuse = np.array(diffuse)
    # dense = ~diffuse
    #
    # # R(V) versus A(V)
    # ax.errorbar(
    #     rvs[diffuse],
    #     exts[diffuse],
    #     xerr=rvs_unc[:, diffuse],
    #     yerr=exts_unc[diffuse],
    #     fmt="go",
    #     label="diffuse",
    # )
    # ax.errorbar(
    #     rvs[dense],
    #     exts[dense],
    #     xerr=rvs_unc[:, dense],
    #     yerr=exts_unc[dense],
    #     fmt="bo",
    #     markerfacecolor="none",
    #     label="dense",
    # )
    # ax.set_xlabel(r"$R(V)$")
    # ax.set_ylabel(rf"$A({args.band})/A(V)$")
    # ax.tick_params("both", length=10, width=2, which="major")
    # ax.tick_params("both", length=5, width=1, which="minor")
    #
    # ax.legend()
    #
    # fig.tight_layout()
    #
    # save_str = f"_rvdep{args.band}"
    # if args.png:
    #     fig.savefig(args.filelist.replace(".dat", save_str + ".png"))
    # elif args.pdf:
    #     fig.savefig(args.filelist.replace(".dat", save_str + ".pdf"))
    # else:
    #     pyplot.show()
