# This script is intended to automate the calculation, fitting and plotting of the extinction curves of all stars in the sample.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from measure_extinction.extdata import ExtData, AverageExtData
from measure_extinction.utils.calc_ext import calc_extinction, calc_ave_ext
from measure_extinction.plotting.plot_ext import plot_extinction

from fit_spex_ext import fit_spex_ext, fit_features_ext, fit_features_spec


# function to calculate, fit and plot all extinction curves
def calc_fit_plot(
    starpair_list, path, dense=False, profile="gauss_asym", bootstrap=False, fixed=False
):
    """
    Calculate, fit and plot the extinction curve for all star pairs in "starpair_list"

    Parameters
    ----------
    starpair_list : list of strings
        List of star pairs for which to calculate, fit and plot the extinction curve, in the format "reddenedstarname_comparisonstarname" (no spaces)

    path : string
        Path to the data files

    dense : boolean [default=False]
        Whether or not the sightline is dense

    profile : string [default="gauss_asym"]
        Profile to use for the feature(s) if dense = True (options are "gauss1", "drude1", "lorentz1", "gauss_asym1", "drude_asym1", "lorentz_asym1","gauss2", "drude2", "lorentz2", "gauss_asym2", "drude_asym2", "lorentz_asym2")

    bootstrap : boolean [default=False]
        Whether or not to do a quick bootstrap fitting to get more realistic uncertainties for the fitting results

    fixed : boolean [default=False]
        Whether or not to add a fixed feature around 3 micron (for diffuse sightlines)

    Returns
    -------
    Calculates, saves, fits and plots the extinction curve
    """
    for starpair in starpair_list:
        redstar = starpair.split("_")[0]
        compstar = starpair.split("_")[1]
        print("reddened star: ", redstar, "/ comparison star:", compstar)

        # calculate the extinction curve
        calc_extinction(redstar, compstar, path)

        # fit the extinction curve
        fit_spex_ext(
            starpair,
            path,
            dense=dense,
            profile=profile,
            bootstrap=bootstrap,
            fixed=fixed,
        )

        # plot the extinction curve
        plot_extinction(
            starpair,
            path,
            fitmodel=True,
            range=[0.78, 5.55],
            exclude=["IRS"],
            pdf=True,
        )


def calc_fit_average(starpair_list, path, fixed=False):
    """
    Calculate and fit the average extinction curve

    Parameters
    ----------
    starpair_list : list of strings
        List of star pairs for which to calculate and fit the average extinction curve, in the format "reddenedstarname_comparisonstarname" (no spaces)

    path : string
        Path to the data files


    fixed : boolean [default=False]
        Whether or not to add a fixed feature around 3 micron

    Returns
    -------
    Calculates, saves and fits the average extinction curve (output: path/average_ext.fits)
    """
    # mask wavelength regions at the edges
    mask = [
        (0.805, 0.807),
        (1.34, 1.344),
        (1.41, 1.42),
        (1.949, 1.953),
        (2.9, 2.908),
        (3.996, 4.01),
    ]

    # calculate the average extinction curve
    calc_ave_ext(starpair_list, path, min_number=5, mask=mask)

    # fit the average extinction curve
    fit_spex_ext("average", path, fixed=fixed)


def fit_plot_features_spectrum(star, path):
    """
    Fit and plot the features directly from the spectrum

    Parameters
    ----------
    star : string
        Name of the reddened star for which to fit the features in the spectrum

    path : string
        Path to the data files

    Returns
    -------
    Plot with the continuum-subtracted spectrum and the fitted models
    """
    # fit the features
    waves, fluxes, npts, results = fit_features_spec(star, path)

    # plot the data
    fig, ax = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [6, 1]}
    )
    fluxes[npts == 0] = np.nan
    ax[0].plot(waves, fluxes, color="k", lw=0.5, alpha=0.7)

    # plot the fitted models
    # 2 Gaussians
    ax[0].plot(waves, results[0](waves), lw=2, label="2 Gaussians")

    # 2 asymmetric Gaussians (with the two individual profiles)
    ax[0].plot(
        waves,
        results[3](waves),
        lw=2,
        label="2 mod. Gaussians",
    )
    ax[0].plot(waves, results[3][0](waves), color="C1", lw=1, ls="--")
    ax[0].plot(waves, results[3][1](waves), color="C1", lw=1, ls="--")

    # 2 Drudes
    ax[0].plot(
        waves,
        results[1](waves),
        ls="--",
        lw=1,
        label="2 Drudes",
    )

    # 2 asymmetric Drudes
    ax[0].plot(
        waves,
        results[4](waves),
        ls="--",
        lw=1,
        label="2 mod. Drudes",
    )

    # 2 Lorentzians
    ax[0].plot(
        waves,
        results[2](waves),
        ls=":",
        lw=1,
        label="2 Lorentzians",
    )

    # 2 asymmetric Lorentzians
    ax[0].plot(
        waves,
        results[5](waves),
        ls=":",
        lw=1,
        label="2 mod. Lorentzians",
    )

    # 1 asymmetric Drude
    ax[0].plot(
        waves,
        results[6](waves),
        ls="-.",
        lw=1,
        label="1 mod. Drude",
    )

    # finish the upper plot
    ax[0].set_ylabel("flux")
    ax[0].set_ylim(-0.4e-12, 0.05e-12)
    ax[0].axhline(color="k", ls=":")
    ax[0].yaxis.set_major_locator(MaxNLocator(prune="lower"))
    ax[0].legend(fontsize=fs * 0.6)

    # plot the residuals (for the best fitting model)
    ax[1].scatter(waves, results[3](waves) - fluxes, s=0.7, color="C1")
    ax[1].set_ylim(-1e-13, 1e-13)
    ax[1].axhline(ls="--", c="k", alpha=0.5)
    ax[1].set_ylabel("residual")

    # finish and save the plot
    plt.xlabel(r"$\lambda$ [$\mu m$]")
    plt.subplots_adjust(hspace=0)
    plt.savefig(
        "/Users/mdecleir/spex_nir_extinction/Figures/" + star + "_spec_features.pdf",
        bbox_inches="tight",
    )


def fit_plot_features_ext(starpair, path):
    """
    Fit and plot the extinction features separately

    Parameters
    ----------
    starpair : string
        Name of the star pair for which to fit the extinction features, in the format "reddenedstarname_comparisonstarname" (no spaces)

    path : string
        Path to the data files

    Returns
    -------
    Plot with the continuum-subtracted extinction and the fitted models
    """
    # fit the features
    waves, exts, results = fit_features_ext(starpair, path)

    # plot the data
    fig, ax = plt.subplots(
        2, 1, figsize=(8, 6), sharex=True, gridspec_kw={"height_ratios": [6, 1]}
    )
    ax[0].plot(waves, exts, color="k", lw=0.5, alpha=0.7)

    # plot the fitted models
    # 2 Gaussians
    ax[0].plot(
        waves,
        results[0](waves),
        lw=2,
        label="2 Gaussians",
    )

    # 2 asymmetric Gaussians (with the two individual profiles)
    ax[0].plot(
        waves,
        results[3](waves),
        lw=2,
        label="2 mod. Gaussians",
    )
    ax[0].plot(waves, results[3][0](waves), color="C1", lw=1, ls="--")
    ax[0].plot(waves, results[3][1](waves), color="C1", lw=1, ls="--")

    # 2 Drudes
    ax[0].plot(
        waves,
        results[1](waves),
        ls="--",
        lw=1,
        label="2 Drudes",
    )

    # 2 asymmetric Drudes
    ax[0].plot(
        waves,
        results[4](waves),
        ls="--",
        lw=1,
        label="2 mod. Drudes",
    )

    # 2 Lorentzians
    ax[0].plot(
        waves,
        results[2](waves),
        ls=":",
        lw=1,
        label="2 Lorentzians",
    )

    # 2 asymmetric Lorentzians
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
    plt.savefig(
        "/Users/mdecleir/spex_nir_extinction/Figures/" + starpair + "_features.pdf",
        bbox_inches="tight",
    )


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
    # define the path of the data files
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"

    # define the diffuse and dense sub-samples
    diffuse = [
        "BD+56d524_HD034816",
        "HD013338_HD031726",
        # "HD014250_HD031726",
        # "HD014422_HD214680",
        "HD014956_HD214680",
        "HD017505_HD214680",
        "HD029309_HD042560",
        # "HD034921_HD214680",
        # "HD037020_HD034816",
        # "HD037022_HD034816",
        # "HD037023_HD036512",
        "HD037061_HD034816",
        "HD038087_HD051283",
        # "HD052721_HD091316",
        "HD156247_HD042560",
        # "HD166734_HD031726",
        "HD183143_HD188209",
        "HD185418_HD034816",
        "HD192660_HD214680",
        "HD204827_HD003360",
        # "HD206773_HD047839",
        "HD229238_HD214680",
        # "HD294264_HD051283",
    ]

    dense_samp = ["HD029647_HD034759", "HD283809_HD003360"]

    # calculate, fit and plot all diffuse extinction curves
    # calc_fit_plot(diffuse, path, bootstrap=True)

    # fit all diffuse sightlines with a fixed feature
    # calc_fit_plot(diffuse, path, bootstrap=True, fixed=True)

    # calculate and fit the average diffuse extinction curve
    # calc_fit_average(diffuse, path)

    # calculate and fit the average diffuse extinction curve with a fixed feature
    # calc_fit_average(diffuse, path, fixed=True)

    # calculate, fit and plot all dense extinction curves
    # calc_fit_plot(dense_samp, path, dense=True, profile="drude_asym1", bootstrap=True)

    # create more plots
    # fs = 18
    # plt.rc("font", size=fs)
    # plt.rc("axes", lw=1)
    # plt.rc("xtick", direction="in", labelsize=fs * 0.8)
    # plt.rc("ytick", direction="in", labelsize=fs * 0.8)
    # plt.rc("xtick.major", width=1, size=8)
    # plt.rc("ytick.major", width=1, size=8)

    # ------------------------------------------------------------------
    # EXTRA (eventually not used in the paper)
    # plot all residuals in one figure
    # plot_residuals(starpair_list)

    # fit features from the spectrum instead of the extinction curve
    # fit_plot_features_spectrum("HD283809", path)

    # fit features from the continuum-subtracted extinction curve
    # fit_plot_features_ext("HD283809_HD003360", path)

    # fit ice feature with an assymmetric Gaussian
    # calc_fit_plot(dense_samp, path, dense=True, profile="gauss_asym1")
