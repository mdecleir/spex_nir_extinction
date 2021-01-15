# This script is intended to automate the calculation, fitting and plotting of the extinction curves of all stars in the sample.
import numpy as np
import matplotlib.pyplot as plt

from measure_extinction.extdata import ExtData, AverageExtData
from measure_extinction.utils.calc_ext import calc_extinction
from measure_extinction.plotting.plot_ext import plot_multi_extinction, plot_extinction

from fit_spex_ext import fit_spex_ext


# function to calculate, fit and plot all extinction curves
def calc_fit_plot(starpair_list):
    for starpair in starpair_list:
        redstar = starpair.split("_")[0]
        compstar = starpair.split("_")[1]
        print("reddened star: ", redstar, "/ comparison star:", compstar)

        # calculate the extinction curve
        calc_extinction(redstar, compstar, path)

        # fit the extinction curve
        fit_spex_ext(starpair, path, ice=True)

        # plot the extinction curve
        plot_extinction(
            starpair,
            path,
            fitmodel=True,
            range=[0.78, 5.55],
            exclude=["IRS"],
            pdf=True,
        )


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
        ax1.axhline(y=-0.1, ls=":", alpha=0.5)
        ax1.axhline(y=0.1, ls=":", alpha=0.5)
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
        "HD017505_HD214680",  # done
        "BD+56d524_HD051283",  # done
        "HD013338_HD031726",  # done
        "HD014250_HD032630",  # done # bad
        "HD014422_HD214680",  # done # emission
        "HD014956_HD188209",  # done
        "HD029309_HD051283",  # done
        "HD029647_HD078316",  # done
        "HD034921_HD214680",  # done # emission
        "HD037020_HD034816",  # done # bad
        "HD037022_HD214680",  # done # bad
        "HD037023_HD036512",  # done
        "HD037061_HD034816",  # this # done
        "HD038087_HD003360",  # this # done
        "HD052721_HD036512",  # done # bad
        "HD156247_HD032630",  # done # emission
        "HD166734_HD036512",  # done
        "HD183143_HD188209",  # done
        "HD185418_HD034816",  # done
        "HD192660_HD091316",  # done
        "HD204827_HD204172",  # done
        "HD206773_HD047839",  # done
        "HD229238_HD091316",  # done
        "HD283809_HD003360",  # done
        "HD294264_HD031726",  # this # done
    ]

    # calculate, fit and plot all extinction curves
    calc_fit_plot(starpair_list)

    # create more plots
    plt.rc("font", size=18)
    plot_residuals(starpair_list)

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
