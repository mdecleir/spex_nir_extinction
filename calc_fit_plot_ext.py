# This script is intended to automate the calculation, fitting and plotting of the extinction curves of all stars in the sample.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from measure_extinction.extdata import ExtData, AverageExtData
from measure_extinction.utils.calc_ext import calc_extinction
from measure_extinction.plotting.plot_ext import plot_multi_extinction, plot_extinction

from fit_spex_ext import fit_spex_ext, fit_features


# function to calculate, fit and plot all extinction curves
def calc_fit_plot(starpair_list):
    for starpair in starpair_list:
        redstar = starpair.split("_")[0]
        compstar = starpair.split("_")[1]
        print("reddened star: ", redstar, "/ comparison star:", compstar)

        # calculate the extinction curve
        calc_extinction(redstar, compstar, path)

        # fit the extinction curve
        fit_spex_ext(starpair, path, ice=False)

        # plot the extinction curve
        plot_extinction(
            starpair,
            path,
            fitmodel=True,
            range=[0.78, 5.55],
            exclude=["IRS"],
            pdf=True,
        )


def fit_plot_features(starpair):
    """
    Fit and plot the features separately with different profiles

    Parameters
    ----------
    starpair : string
        Name of the star pair for which to fit the extinction features, in the format "reddenedstarname_comparisonstarname" (no spaces)

    Returns
    -------
    Plot with the continuum-subtracted extinction data and the fitted models
    """
    # fit the features
    waves, exts, results = fit_features(starpair, path)

    # plot the data
    fig, ax = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [6, 1]}
    )
    ax[0].plot(waves, exts, color="k", lw=0.5, alpha=0.7)

    # plot the fitted models
    # Gaussians (with the two individual profiles)
    ax[0].plot(
        waves,
        results[0](waves),
        lw=2,
        label="2 Gaussians",
    )

    # ax[0].plot(waves, results[0][0](waves), color="#1f77b4", lw=1, ls="--")
    # ax[0].plot(waves, results[0][1](waves), color="#1f77b4", lw=1, ls="--")

    # Asymmetric Gaussians (with the two individual profiles)
    ax[0].plot(
        waves,
        results[3](waves),
        lw=2,
        label="2 mod. Gaussians",
    )
    ax[0].plot(waves, results[3][0](waves), color="C1", lw=1, ls="--")
    ax[0].plot(waves, results[3][1](waves), color="C1", lw=1, ls="--")

    # Drudes
    ax[0].plot(
        waves,
        results[1](waves),
        ls="--",
        lw=1,
        label="2 Drudes",
    )

    # Asymmetric Drudes
    ax[0].plot(
        waves,
        results[4](waves),
        ls="--",
        lw=1,
        label="2 mod. Drudes",
    )

    # Lorentzians
    ax[0].plot(
        waves,
        results[2](waves),
        ls=":",
        lw=1,
        label="2 Lorentzians",
    )

    # Asymmetric Lorentzians
    ax[0].plot(
        waves,
        results[5](waves),
        ls=":",
        lw=1,
        label="2 mod. Lorentzians",
    )

    # finish the upper plot
    ax[0].set_ylim(0.0, 0.176)
    ax[0].set_ylabel("excess extinction")
    ax[0].yaxis.set_major_locator(MaxNLocator(prune="lower"))
    ax[0].legend(fontsize=fs * 0.8)

    # plot the residuals (for the best fitting model)
    ax[1].scatter(waves, results[3](waves) - exts, s=0.7, color="C1")
    ax[1].axhline(ls="--", c="k", alpha=0.5)
    ax[1].axhline(y=0.05, ls=":", c="k", alpha=0.5)
    ax[1].axhline(y=-0.05, ls=":", c="k", alpha=0.5)
    ax[1].set_ylabel("residual")
    ax[1].set_ylim(-0.1, 0.1)

    # finish and save the plot
    plt.xlabel(r"$\lambda$ [$\mu m$]")
    plt.subplots_adjust(hspace=0)
    plt.savefig("Figures/features.pdf", bbox_inches="tight")


# def plot_lab_ice():
# file = "data/H2O_NASA.dat"
# table = pd.read_table(file, comment="#", sep="\s+")
# table = table[2531:]
# waves = 1 / table["Freq."] * 1e4
# norm = np.max(-table["%T,10K"] + 100)
# absorbs = (-table["%T,10K"] + 100) / norm
#
# # plot the spectrum
# # plt.plot(waves, absorbs, color="k", label=r"H$_2$O ice")
#
# file = "data/mixother_NASA.dat"
# table = pd.read_table(file, comment="#", sep="\s+")
#
# table = table[2531:]
# waves = 1 / table["Freq."] * 1e4
# norm = np.max(-table["%T,10K"] + 95)
# absorbs = (-table["%T,10K"] + 95) / norm
#
# # plot the spectrum
# # plt.plot(waves, absorbs, color="k", label=r"mixed ice")
#
# file = "data/mix_Leiden.dat"
# table = pd.read_table(file, comment="#", sep="\s+")
# table = table[4330:]
# waves = 1 / table["Freq."] * 1e4
# norm = np.max(table["Trans."] + 0.02)
# absorbs = (table["Trans."] + 0.02) / norm
# # plt.plot(waves, absorbs)
#
# file = "data/ammon.dat"
# table = pd.read_table(file, comment="#", sep="\s+")
# table = table[4000:]
# waves = 1 / table["Freq."] * 1e4
# norm = np.max(table["Trans."] + 0.01)
# absorbs = (table["Trans."] + 0.01) / norm
# # plt.plot(waves, absorbs)
#
# file = "data/Godd_mix.dat"
# table = pd.read_table(file, comment="#", sep="\s+")
# table = table[500:]
# waves = 1 / table["freq"] * 1e4
# norm = np.max(table["absorbance"] + 0.01)
# absorbs = (table["absorbance"] + 0.01) / norm
#
# # plt.plot(waves, absorbs)


# function to plot all residuals in one figure
def plot_residuals(starpair_list):
    # make a list to add all residuals
    extdata = ExtData("%s%s_ext.fits" % (path, starpair_list[0].lower()))
    full_waves = np.sort(
        np.concatenate(
            (extdata.waves["SpeX_SXD"].value, extdata.waves["SpeX_LXD"].value)
        )
    )
    full_res = np.zeros_like(full_waves)
    full_npts = np.zeros_like(full_waves)

    # compact
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    # spread out
    fig2, ax2 = plt.subplots(figsize=(15, len(starpair_list) * 1.25))
    colors = plt.get_cmap("tab10")

    for i, starpair in enumerate(starpair_list):
        extdata = ExtData("%s%s_ext.fits" % (path, starpair.lower()))

        # sum the residuals
        for wave in full_waves:
            indx_model = np.where(extdata.model["waves"] == wave)[0]
            indx_full = np.where(full_waves == wave)[0]
            if (len(indx_model) > 0) & (
                ~np.isnan(extdata.model["residuals"][indx_model])
            ):
                full_res[indx_full] += extdata.model["residuals"][indx_model]
                full_npts[indx_full] += 1

        # compact
        ax1.scatter(
            extdata.model["waves"], extdata.model["residuals"], s=0.3, alpha=0.3
        )
        ax1.axhline(ls="--", c="k", alpha=0.5)
        ax1.axhline(y=-0.02, ls=":", alpha=0.5)
        ax1.axhline(y=0.02, ls=":", alpha=0.5)
        # ax1.axvline(x=1.354, ls=":", alpha=0.5)
        # ax1.axvline(x=1.411, ls=":", alpha=0.5)
        # ax1.axvline(x=1.805, ls=":", alpha=0.5)
        # ax1.axvline(x=1.947, ls=":", alpha=0.5)
        # ax1.axvline(x=2.522, ls=":", alpha=0.5)
        # ax1.axvline(x=2.875, ls=":", alpha=0.5)
        # ax1.axvline(x=4.014, ls=":", alpha=0.5)
        # ax1.axvline(x=4.594, ls=":", alpha=0.5)

        ax1.set_ylim([-0.2, 0.2])
        ax1.set_xlabel(r"$\lambda$ [$\mu m$]")
        ax1.set_ylabel("residuals")

        # spread out
        offset = 0.2 * i
        ax2.scatter(extdata.model["waves"], extdata.model["residuals"] + offset, s=0.5)
        ax2.axhline(y=offset, ls="--", c="k", alpha=0.5)
        ax2.axhline(y=offset - 0.1, ls=":", alpha=0.5)
        ax2.axhline(y=offset + 0.1, ls=":", alpha=0.5)

        ax2.text(5, offset, starpair.split("_")[0], color=colors(i % 10), fontsize=14)
        ax2.set_ylim([-0.2, offset + 0.2])
        ax2.set_xlabel(r"$\lambda$ [$\mu m$]")
        ax2.set_ylabel("residuals + offset")

    # plot the average of the residuals
    ax1.plot(full_waves, full_res / full_npts, color="k", lw=0.2)
    fig1.savefig(path + "residuals.pdf", bbox_inches="tight")
    fig2.savefig(path + "residuals_spread.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # define the path and the names of the star pairs in the format "reddenedstarname_comparisonstarname" (first the main sequence stars and then the giant stars, sorted by spectral type from B8 to O4)
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

    # # read the list of stars for which to measure and plot the extinction curve
    # table = pd.read_table("red-comp.list", comment="#")
    # redstars = table["reddened"]
    # compstars = table["comparison"]
    # starpair_list = []
    #
    # # calculate and plot the extinction curve for every star
    # for redstar, compstar in zip(redstars, compstars):
    #     # create the starpair_list
    #     starpair_list.append(redstar + "_" + compstar)

    # calculate, fit and plot all extinction curves
    calc_fit_plot(starpair_list)

    # plot_multi_extinction(
    #     starpair_list,
    #     path,
    #     range=[0.76, 5.5],
    #     # alax=True,
    #     # spread=True,
    #     exclude=["IRS"],
    #     pdf=True,
    # )

    # create more plots
    fs = 18
    plt.rc("font", size=fs)
    plt.rc("axes", lw=1)
    plt.rc("xtick", direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", direction="in", labelsize=fs * 0.8)
    plt.rc("xtick.major", width=1, size=8)
    plt.rc("ytick.major", width=1, size=8)

    # plot all residuals in one figure
    # plot_residuals(starpair_list)

    # fit features for HD283809 separately
    # fit_plot_features("HD283809_HD003360")

    # parser.add_argument("--alax", help="plot A(lambda)/A(X)", action="store_true")
    # parser.add_argument(
    #     "--average", help="plot the average extinction curve", action="store_true"
    # )
    # parser.add_argument(
    #     "--extmodels", help="plot extinction curve models", action="store_true"
    # )
    # parser.add_argument(
    #     "--powerlaw", help="plot NIR powerlaw model", action="store_true"
    # )
    # parser.add_argument("--HI_lines", help="indicate the HI-lines", action="store_true")
    # parser.add_argument(
    #     "--onefig",
    #     help="whether or not to plot all curves in the same figure",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--range",
    #     nargs="+",
    #     help="wavelength range to be plotted (in micron)",
    #     type=float,
    #     default=None,
    # )
    # parser.add_argument(
    #     "--spread", help="spread the curves out over the figure", action="store_true"
    # )


#
# from measure_extinction.plotting.plot_ext import (
#     plot_multi_extinction,
#     plot_extinction,
#     plot_average,
# )
#
# def plot_extinction_curves():
#     starpair_list = [
#         "HD017505_HD214680",
#         "BD+56d524_HD051283",
#         "HD013338_HD031726",
#         ## "HD014250_HD032630",
#         ## "HD014422_HD214680",
#         "HD014956_HD188209",
#         "HD029309_HD051283",
#         "HD029647_HD078316",
#         ## "HD034921_HD214680",
#         ## "HD037020_HD034816",
#         ## "HD037022_HD214680",
#         "HD037023_HD036512",
#         "HD037061_HD034816",  # this
#         "HD038087_HD003360",  # this
#         ## "HD052721_HD036512",
#         ## "HD156247_HD032630",
#         "HD166734_HD036512",
#         "HD183143_HD188209",
#         "HD185418_HD034816",
#         "HD192660_HD091316",
#         "HD204827_HD204172",
#         ## "HD206773_HD047839",
#         "HD229238_HD091316",
#         "HD283809_HD003360",
#         "HD294264_HD031726",  # this
#     ]
#
#     # plot the extinction curves
#     # plot_multi_extinction(
#     #     starpair_list,
#     #     path,
#     #     range=[0.76, 5.5],
#     #     alax=True,
#     #     spread=False,
#     #     exclude=["IRS"],
#     #     pdf=True,
#     # )
#
#     # plot the average extinction curve in a separate plot
#     # plot_average(
#     #     starpair_list,
#     #     path,
#     #     powerlaw=True,
#     #     extmodels=True,
#     #     range=[0.78, 6.1],
#     #     exclude=["IRS"],
#     #     pdf=True,
#     # )
#
#     # calculate RV
#     # average.columns["RV"] = 1 / (average.exts["BAND"][1] - 1)
#
#
# def plot_fit():
#
#     # plot_extinction_curves()
#     plot_fit()
#     #
#     # if args.onefig:  # plot all curves in the same figure
#     #     plot_multi_extinction(
#     #         starpair_list,
#     #         args.path,
#     #         args.alax,
#     #         args.average,
#     #         args.extmodels,
#     #         args.powerlaw,
#     #         args.HI_lines,
#     #         args.range,
#     #         args.spread,
#     #         args.exclude,
#     #         pdf=True,
#     #     )
#     # else:  # plot all curves separately
#     #     if args.spread:
#     #         parser.error(
#     #             "The flag --spread can only be used in combination with the flag --onefig. It only makes sense to spread out the curves if there is more than one curve in the same plot."
#     #         )
#     #     if args.average:
#     #         parser.error(
#     #             "The flag --average can only be used in combination with the flag --onefig. It only makes sense to add the average extinction curve to a plot with multiple curves."
#     #         )
#     #     for redstar, compstar in zip(redstars, compstars):
#     #         plot_extinction(
#     #             redstar + "_" + compstar,
#     #             args.path,
#     #             args.alax,
#     #             args.extmodels,
#     #             args.powerlaw,
#     #             args.HI_lines,
#     #             args.range,
#     #             args.exclude,
#     #             pdf=True,
#     #         )
