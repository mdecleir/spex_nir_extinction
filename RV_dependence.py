# This script calculates and fits the R(V) dependent relationship

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from collections import Counter
from scipy import stats, interpolate
from astropy.modeling import models, fitting
from astropy.table import Table
from matplotlib.lines import Line2D

import astropy.units as u

from measure_extinction.extdata import ExtData
from dust_extinction.parameter_averages import CCM89, F19


def get_data(inpath, starpair_list, norm="V"):
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

    norm : string [default="V"]
        Band or wavelength for the normalization

    Returns
    -------
    R(V) with uncertainties, A(lambda)/A(V) with uncertainties, wavelengths
    """
    RVs = np.zeros((len(starpair_list), 3))

    # determine the wavelengths at which to retrieve the extinction data
    extdata_model = ExtData("%s%s_ext.fits" % (inpath, starpair_list[0].lower()))
    waves = np.sort(
        np.concatenate(
            (
                extdata_model.waves["SpeX_SXD"].value,
                extdata_model.waves["SpeX_LXD"].value,
            )
        )
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

        # convert extinction from A(lambda)/A(V) to A(lambda)/A(norm) if norm is not "V"
        if norm != "V":
            ind1 = np.abs(flat_waves.value - norm).argmin()
            flat_exts = flat_exts / flat_exts[ind1]
            flat_exts_unc = flat_exts_unc / flat_exts[ind1]

        # retrieve A(lambda)/A(V) at all wavelengths
        for j, wave in enumerate(waves):
            if wave in flat_waves.value:
                alavs[j][i] = flat_exts[flat_waves.value == wave]
                alav_uncs[j][i] = flat_exts_unc[flat_waves.value == wave]

    return RVs, alavs, alav_uncs, waves


def plot_rv_dep(
    outpath, RVs, alavs, alav_uncs, waves, plot_waves, slopes, intercepts, norm="V"
):
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

    norm : string [default="V"]
        Band or wavelength for the normalization

    Returns
    -------
    Plot of A(lambda)/A(V) vs. R(V)-3.1 at wavelengths "plot_waves"
    """
    fig, ax = plt.subplots(
        len(plot_waves), figsize=(7, len(plot_waves) * 4), sharex=True
    )
    if norm != "V":
        norm = str(norm) + r"\mu m"

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
        xs = np.arange(-1, 3, 0.1)
        ax[j].plot(
            xs,
            slopes[indx] * xs + intercepts[indx],
            color="forestgreen",
            ls="--",
            alpha=0.6,
            label=r"$%5.3f\, (R_V-3.1) - %5.3f$" % (slopes[indx], intercepts[indx]),
        )

        ax[j].set_ylabel(r"$A(" + "{:1.2f}".format(wave) + "\mu m)/A(" + norm + ")$")
        ax[j].legend(loc="lower right", fontsize=fs * 0.8)

        # add literature curves
        # styles = ["--", ":"]
        # for i, cmodel in enumerate([CCM89, F19]):
        #     if wave > 3.3:
        #         continue
        #     yvals = []
        #     for xval in xs:
        #         ext_model = cmodel(Rv=xval + 3.1)
        #         yvals.append(ext_model(wave * u.micron))
        #     ax[j].plot(
        #         xs,
        #         yvals,
        #         lw=1.5,
        #         ls=styles[i],
        #         alpha=0.8,
        #     )

    # finalize the plot
    plt.xlabel("R(V) - 3.1")
    plt.subplots_adjust(hspace=0)
    plt.savefig(outpath + "RV_dep" + norm.split("\\")[0] + ".pdf", bbox_inches="tight")


def table_rv_dep(outpath, table_waves, fit_slopes, fit_intercepts, fit_stds, norm="V"):
    """
    Create tables with the slopes, intercepts and standard deviations at wavelengths "table_waves", and the measured and fitted average extinction curve

    Parameters
    ----------
    outpath : string
        Path to save the table

    table_waves : list
        List with wavelengths to be included in the table

    fit_slopes : tuple
        The interpolated spline for the slopes

    fit_intercepts : astropy model
        The fitted model for the intercepts

    fit_stds : tuple
        The interpolated spline for the standard deviations

    norm : string [default="V"]
        Band or wavelength for the normalization

    Returns
    -------
    Tables of the R(V)-dependent relationship at wavelengths "table_waves":
        - in aaxtex format for the paper
        - in ascii format
    """
    # obtain the slopes, intercepts and standard deviations at the table wavelengths
    table_slopes = interpolate.splev(table_waves, fit_slopes)
    table_intercepts = fit_intercepts(table_waves)
    table_stds = interpolate.splev(table_waves, fit_stds)

    # obtain the measured average extinction curve
    average = ExtData(inpath + "average_ext.fits")
    (ave_waves, exts, exts_unc) = average.get_fitdata(["SpeX_SXD", "SpeX_LXD"])
    indx = np.argsort(ave_waves)
    ave_waves = ave_waves[indx].value
    exts = exts[indx]
    exts_unc = exts_unc[indx]

    # create wavelength bins and calculate the binned median extinction and uncertainty
    bin_edges = np.insert(table_waves + 0.025, 0, table_waves[0] - 0.025)
    meds, edges, indices = stats.binned_statistic(
        ave_waves,
        (exts, exts_unc),
        statistic="median",
        bins=bin_edges,
    )

    # obtain the fitted average extinction curve
    ave_fit = average.model["params"][0] * table_waves ** (-average.model["params"][2])

    # create the table
    table = Table(
        [
            table_waves,
            meds[0],
            meds[1],
            ave_fit,
            table_slopes,
            table_intercepts,
            table_stds,
        ],
        names=(
            "wavelength[micron]",
            "ave",
            "ave_unc",
            "ave_fit",
            "slope",
            "intercept",
            "std",
        ),
    )

    # save it in ascii format
    table.write(
        outpath + "RV_dep" + str(norm) + ".txt",
        format="ascii.commented_header",
        overwrite=True,
    )

    # save it in aastex format
    table.write(
        outpath + "RV_dep" + str(norm) + ".tex",
        format="aastex",
        names=(
            r"$\lambda [\micron]$",
            "ave",
            "unc",
            "fit",
            r"$b(\lambda$)",
            r"$a(\lambda$)",
            r"$\sigma(\lambda)$",
        ),
        formats={
            r"$\lambda [\micron]$": "{:.2f}",
            "ave": "{:.3f}",
            "unc": "{:.3f}",
            "fit": "{:.3f}",
            r"$b(\lambda$)": "{:.3f}",
            r"$a(\lambda$)": "{:.3f}",
            r"$\sigma(\lambda)$": "{:.3f}",
        },
        latexdict={
            "col_align": "c|ccc|ccc",
            "tabletype": "deluxetable",
            "caption": r"The average diffuse Milky Way extinction curve and parameters of the linear relationship between extinction $A(\lambda)/A(V)$ and $R(V)$. For every wavelength, in columns 2--4 the binned value of the measured average extinction curve, its uncertainty, and the fitted value are given. Columns 5--7 give the slope $b$, intercept $a$, and standard deviation $\sigma$ of the $R(V)$-dependent extinction curve. \label{tab:RV_dep}",
        },
        fill_values=[("nan", "--")],
        overwrite=True,
    )


def fit_slopes_intercepts(slopes, intercepts, stds, waves, norm):
    """
    Fit the slopes, intercepts and standard deviations vs. wavelength

    Parameters
    ----------
    slopes : np.ndarray
        Numpy array with the slopes of the linear relationship

    intercepts : np.ndarray
        Numpy array with the intercepts of the linear relationship

    stds : np.ndarray
        Numpy array with the standard deviations about the linear fit

    waves : np.ndarray
        Numpy array with all wavelengths

    norm : string [default="V"]
        Band or wavelength for the normalization

    Returns
    -------
    spline_wave : np.ndarray
        Numpy array with the anchor wavelengths

    spline_slope : np.ndarray
        Numpy array with the anchor slopes

    spline_std : np.ndarray
        Numpy array with the anchor standard deviations

    fit_slopes : tuple
        The interpolated spline for the slopes

    fit_intercepts : astropy model
        The fitted model for the intercepts

    fit_stds : tuple
        The interpolated spline for the standard deviations
    """
    # define a mask for the good data
    mask = ~np.isnan(slopes)
    short_wave_mask = waves < 4.1

    # fit the intercepts with a power law
    fit_lev = fitting.LevMarLSQFitter()
    powerlaw = models.PowerLaw1D(fixed={"x_0": True})
    fit_intercepts = fit_lev(powerlaw, waves[mask], intercepts[mask])

    # define the anchor points for the spline interpolation
    # divide the data into 25 bins with the same number of data points in every bin
    alloc, bin_edges = pd.qcut(waves[mask * short_wave_mask], q=25, retbins=True)
    # calculate the median wavelength, slope and standard deviation in every bin
    meds, edges, indices = stats.binned_statistic(
        waves[mask * short_wave_mask],
        (
            waves[mask * short_wave_mask],
            slopes[mask * short_wave_mask],
            stds[mask * short_wave_mask],
        ),
        statistic="median",
        bins=bin_edges,
    )

    # use the median values as the anchor points for the spline interpolation
    spline_wave = meds[0][~np.isnan(meds[0])]
    spline_slope = meds[1][~np.isnan(meds[1])]
    spline_std = meds[2][~np.isnan(meds[2])]

    # interpolate the slopes with a spline function
    fit_slopes = interpolate.splrep(spline_wave, spline_slope)

    # interpolate the standard deviations with a spline function
    fit_stds = interpolate.splrep(spline_wave, spline_std)

    # create tables with the fitting results at certain wavelengths
    table_waves = np.arange(0.8, 4.05, 0.05)
    table_rv_dep(table_path, table_waves, fit_slopes, fit_intercepts, fit_stds, norm)

    return spline_wave, spline_slope, spline_std, fit_slopes, fit_intercepts, fit_stds


def plot_RV_lit(outpath, fit_slopes, fit_intercepts):
    """
    Plot the R(V)-dependent extinction curve for different R(V) values, together with R(V)-dependent curves from the literature

    Parameters
    ----------
    outpath : string
        Path to save the plot

    fit_slopes : tuple
        The interpolated spline for the slopes

    fit_intercepts : astropy model
        The fitted model for the intercepts

    Returns
    -------
    Plot with the R(V)-dependent extinction curve for different R(V) values
    """
    waves = np.arange(0.8, 4.01, 0.01)
    fig, ax = plt.subplots(figsize=(10, 9))

    for i, RV in enumerate([2.0, 3.1, 5.5]):
        # plot the extinction curve from this work
        offset = 0.08 * i
        slopes = interpolate.splev(waves, fit_slopes)
        alav = fit_intercepts(waves) + slopes * (RV - 3.1)
        (line,) = ax.plot(waves, alav + offset, lw=1.5, label="R(V) = " + str(RV))

        # plot the literature curves
        color = line.get_color()
        styles = ["--", ":"]
        for i, cmodel in enumerate([CCM89, F19]):
            ext_model = cmodel(Rv=RV)
            (indxs,) = np.where(
                np.logical_and(
                    1 / waves >= ext_model.x_range[0], 1 / waves <= ext_model.x_range[1]
                )
            )
            yvals = ext_model(waves[indxs] * u.micron)
            ax.plot(
                waves[indxs],
                yvals + offset,
                lw=1.5,
                color=color,
                ls=styles[i],
                alpha=0.8,
            )

    # add text
    ax.text(3.45, 0.03, "R(V) = 2.0", fontsize=0.8 * fs, color="tab:blue")
    ax.text(3.45, 0.13, "R(V) = 3.1", fontsize=0.8 * fs, color="tab:orange")
    ax.text(3.45, 0.26, "R(V) = 5.5", fontsize=0.8 * fs, color="tab:green")

    # finalize and save the plot
    ax.set_xlabel(r"$\lambda$ [$\mu m$]")
    ax.set_ylabel(r"$A(\lambda)/A(V)$ + offset")
    ax.set_xlim(0.75, 4.05)
    ax.set_ylim(0.0, 0.9)
    handles = [
        Line2D([0], [0], color="k", lw=1.5),
        Line2D([0], [0], color="k", lw=1.5, ls="--"),
        Line2D([0], [0], color="k", lw=1.5, ls=":"),
    ]
    labels = [
        "this work",
        "Cardelli et al. (1989)",
        "Fitzpatrick et al. (2019)",
    ]
    plt.legend(handles, labels, fontsize=0.8 * fs)
    plt.savefig(outpath + "RV_lit.pdf", bbox_inches="tight")


def fit_plot_rv_dep(inpath, plot_path, table_path, starpair_list, norm="V"):
    """
    Fit and plot the relationship between A(lambda)/A(V) and R(V)

    Parameters
    ----------
    inpath : string
        Path to the input data files

    plot_path : string
        Path to save the plots

    table_path : string
        Path to save the table

    starpair_list : list of strings
        List of star pairs to include in the fitting, in the format "reddenedstarname_comparisonstarname" (no spaces)

    norm : string [default="V"]
        Band or wavelength for the normalization

    Returns
    -------
    Several plots related to the R(V)-dependence
    """
    # collect the data to be fitted
    RVs, alavs, alav_uncs, waves = get_data(inpath, starpair_list, norm)
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
        fitted_line = fit(
            line_func,
            (RV_vals[mask] - 3.1),
            alavs[j][mask],
            weights=1 / alav_uncs[j][mask],
        )
        # calculate the standard deviation about the fit
        # the "residuals" in the fit_info is the sum of the squared residuals
        # std = np.sqrt(fit.fit_info["residuals"] / (npts - 2))
        # this does not work when using weights in the fitting
        # dividing by npts-2 is needed, because there are npts-2 degrees of freedom (subtract 1 for the slope and 1 for the intercept)
        std = np.sqrt(
            np.sum((fitted_line(RV_vals[mask] - 3.1) - alavs[j][mask]) ** 2)
            / (npts - 2)
        )
        slopes[j] = fitted_line.slope.value
        intercepts[j] = fitted_line.intercept.value
        stds[j] = std

    # plot A(lambda)/A(V) vs. R(V) at certain wavelengths
    plot_waves = [0.89864296, 1.6499686, 2.4502702, 3.5002365, 4.697073]
    plot_rv_dep(
        plot_path,
        RVs,
        alavs,
        alav_uncs,
        waves,
        plot_waves,
        slopes,
        intercepts,
        norm=norm,
    )

    # plot the slopes, intercepts and standard deviations vs. wavelength
    # color the data points at wavelengths > 4.03 gray
    fig, ax = plt.subplots(3, figsize=(9, 9), sharex=True)
    short_waves = waves < 4.03
    ax[0].scatter(waves[short_waves], slopes[short_waves], color="k", s=0.3)
    ax[0].scatter(waves[~short_waves], slopes[~short_waves], color="gray", s=0.3)
    ax[1].scatter(waves, intercepts, color="k", s=0.3)
    ax[2].scatter(waves[short_waves], stds[short_waves], color="k", s=0.3)
    ax[2].scatter(waves[~short_waves], stds[~short_waves], color="gray", s=0.3)
    for wave in plot_waves:
        indx = np.abs(waves - wave).argmin()
        ax[0].scatter(wave, slopes[indx], color="lime", s=50, marker="x", zorder=3)
        ax[1].scatter(wave, intercepts[indx], color="lime", marker="x", zorder=3)
        ax[2].scatter(wave, stds[indx], color="lime", marker="x", zorder=3)

    # fit the slopes, intercepts and standard deviations vs. wavelength, and add the fit to the plot
    (
        spline_wave,
        spline_slope,
        spline_std,
        fit_slopes,
        fit_intercepts,
        fit_stds,
    ) = fit_slopes_intercepts(slopes, intercepts, stds, waves, norm)
    slope_spline = interpolate.splev(waves[short_waves], fit_slopes)
    ax[0].scatter(spline_wave, spline_slope, color="r", marker="d", s=10)
    ax[0].plot(
        waves[short_waves],
        slope_spline,
        color="crimson",
        ls="--",
        alpha=0.7,
    )

    ax[1].plot(
        waves[:-120],
        fit_intercepts(waves[:-120]),
        color="crimson",
        ls="--",
        alpha=0.7,
        label=r"$%5.2f \ \lambda ^{-%5.2f}$"
        % (fit_intercepts.amplitude.value, fit_intercepts.alpha.value),
    )

    std_spline = interpolate.splev(waves[short_waves], fit_stds)
    ax[2].scatter(spline_wave, spline_std, color="r", marker="d", s=10)
    ax[2].plot(
        waves[short_waves],
        std_spline,
        color="crimson",
        ls="--",
        alpha=0.7,
    )

    # finalize and save the plot
    ax[1].legend(fontsize=0.8 * fs)
    plt.xlabel(r"$\lambda$ [$\mu m$]")
    plt.xlim(0.75, 5.2)
    ax[0].set_ylim(-0.01, 0.075)
    ax[1].set_ylim(-0.03, 0.6)
    ax[2].set_ylim(0.01, 0.075)
    ax[0].set_ylabel(r"$b$")
    ax[1].set_ylabel(r"$a$")
    ax[2].set_ylabel(r"$\sigma$")
    ax[0].axhline(ls="--", color="k", lw=1, alpha=0.6)
    ax[1].axhline(ls="--", color="k", lw=1, alpha=0.6)
    plt.subplots_adjust(hspace=0)
    plt.savefig(plot_path + "RV_slope_inter" + str(norm) + ".pdf", bbox_inches="tight")

    # compare R(V)-dependent extinction curves to literature curves
    # only useful for extinction curves that are normalized to A(V)
    if norm == "V":
        plot_RV_lit(plot_path, fit_slopes, fit_intercepts)


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
        "HD283809_HD003360",
        "HD294264_HD034759",
    ]
    # list the problematic star pairs
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

    # settings for the plotting
    fs = 20
    plt.rc("font", size=fs)
    plt.rc("xtick", direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", direction="in", labelsize=fs * 0.8)
    plt.rc("xtick.major", width=1, size=8)
    plt.rc("ytick.major", width=1, size=8)

    # subtract the flagged stars from the star pair list
    good_stars = list((Counter(starpair_list) - Counter(flagged)).elements())

    # fit and plot the RV dependence
    fit_plot_rv_dep(inpath, plot_path, table_path, good_stars)

    # fit and plot the RV dependence when normalizing to 1 micron instead of to the V-band
    fit_plot_rv_dep(inpath, plot_path, table_path, good_stars, norm=1)

    # ------------------------------------------------------------------
    # EXTRA (eventually not used in the paper)

    # add CCM89 1/RV dependent relation (can only be used with 1/RV)
    # waves_CCM89 = [0.7, 0.9, 1.25, 1.6, 2.2, 3.4]
    # slopes_CCM89 = [-0.366, -0.6239, -0.3679, -0.2473, -0.1483, -0.0734]
    # intercepts_CCM89 = [0.8686, 0.68, 0.4008, 0.2693, 0.1615, 0.08]
    # ax[0].scatter(waves_CCM89, slopes_CCM89, s=5)
    # ax[1].scatter(waves_CCM89, intercepts_CCM89, s=5)
