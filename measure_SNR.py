# This script is intended to measure the SNR of the spectra in different ways

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import os

from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter
from astropy.stats import sigma_clipped_stats

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData
from measure_extinction.plotting.plot_spec import plot_spectrum


def SNR_ori_spec(data_path, plot_path):
    """
    - Print the SNR of all the original spectra, as they come out of Spextool
    - Plot the median SNRs

    Parameters
    ----------
    data_path : string
        Path to the data files

    plot_path : string
        Path to save the plot

    Returns
    -------
    - Maximum and median SNR for every spectrum
    - Plot with all median SNRs
    """
    medians = []
    stars = np.arange(75)

    for file in os.listdir(data_path):
        if file.startswith("."):
            continue
        data = Table.read(
            data_path + file,
            format="ascii",
        )

        # calculate the SNR
        SNR = data["col2"] / data["col3"]

        # print the maximum and median SNR
        print(file, np.nanmax(SNR), np.nanmedian(SNR))
        medians.append(np.nanmedian(SNR))

    # plot the SNR medians
    plt.scatter(stars, medians)
    plt.savefig(plot_path + "SNR_meds.pdf")


def SNR_final_spec(data_path, plot_path, stars, plot=False):
    """
    - Calculate the median SNR of the final used spectra after the 1% noise addition and after "merging" (in merge_obsspec.py), in certain wavelength regions
    - Plot the SNR of the final used spectra if requested

    Parameters
    ----------
    data_path : string
        Path to the data files

    plot_path : string
        Path to save the plots

    stars : list of strings
        List of stars for which to calculate (and plot) the SNR

    plot : boolean [default=False]
        Whether or not to plot the SNR vs. wavelength for every star

    Returns
    -------
    - Median SNRs in certain wavelength regions
    - Plots of the SNR vs. wavelength (if requested)
    """
    meds = np.zeros((3, len(stars)))
    for j, star in enumerate(stars):
        # obtain the flux values and uncertainties
        starobs = StarData(
            "%s.dat" % star.lower(),
            path=data_path,
            use_corfac=True,
        )
        waves, fluxes, uncs = starobs.get_flat_data_arrays(["SpeX_SXD", "SpeX_LXD"])

        # calculate the median SNR in certain wavelength regions
        ranges = [
            (0.79, 2.54),
            (2.85, 4.05),
            (4.55, 5.5),
        ]
        SNR = fluxes / uncs
        for i, range in enumerate(ranges):
            mask = (waves > range[0]) & (waves < range[1])
            meds[i][j] = np.median(SNR[mask])

        # plot SNR vs. wavelength if requested
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(waves, fluxes / uncs, s=1)
            plt.savefig(plot_path + star + "_SNR.pdf")

    print(ranges[0], np.nanmin(meds[0]), np.nanmax(meds[0]))
    print(ranges[1], np.nanmin(meds[1]), np.nanmax(meds[1]))
    print(ranges[2], np.nanmin(meds[2]), np.nanmax(meds[2]))


def measure_SNR(spex_path, data_path, plot_path, star, ranges):
    """
    Measure the SNR of a spectrum, by fitting straight lines to pieces of the spectrum
    """
    # plot the spectrum to define regions without spectral lines
    fig, ax = plot_spectrum(star, data_path, range=[0.75, 5.6], log=True)

    # read in all bands and spectra for this star
    starobs = StarData("%s.dat" % star.lower(), path=data_path, use_corfac=True)

    # obtain flux values at a few wavelengths and fit a straight line through the data
    waves, fluxes, uncs = starobs.get_flat_data_arrays(["SpeX_SXD", "SpeX_LXD"])
    print(star)
    for range in ranges:
        min_indx = np.abs(waves - range[0]).argmin()
        max_indx = np.abs(waves - range[1]).argmin()
        func = Linear1D()
        fit = LinearLSQFitter()
        fit_result = fit(func, waves[min_indx:max_indx], fluxes[min_indx:max_indx])
        residu = fluxes[min_indx:max_indx] - fit_result(waves[min_indx:max_indx])

        # calculate the SNR from the data
        data_sxd = Table.read(
            spex_path + star + "_sxd.txt",
            format="ascii",
        )
        data_lxd = Table.read(
            spex_path + star + "_lxd.txt",
            format="ascii",
        )
        data = vstack([data_sxd, data_lxd])
        data.sort("col1")
        print("wave_range", range)
        min_indx2 = np.abs(data["col1"] - range[0]).argmin()
        max_indx2 = np.abs(data["col1"] - range[1]).argmin()

        SNR_data = np.nanmedian((data["col2"] / data["col3"])[min_indx2:max_indx2])
        print("SNR from data", SNR_data)

        # calculate the SNR from the noise around the linear fit
        mean, median, stddev = sigma_clipped_stats(residu)
        SNR_fit = np.median(fluxes[min_indx:max_indx] / stddev)
        print("SNR from fit", SNR_fit)

        # plot the fitted lines on top of the spectrum
        ax.plot(
            waves[min_indx:max_indx],
            fit_result(waves[min_indx:max_indx]),
            lw=2,
            alpha=0.8,
            color="k",
        )
        fig.savefig(plot_path + star + "_SNR_measure.pdf")


def SNR_ext(data_path, plot_path, starpair_list, plot=False):
    """
    - Calculate the median SNR of the extinction curves in certain wavelength regions
    - Plot the SNR of the extinction curves if requested

    Parameters
    ----------
    data_path : string
        Path to the data files

    plot_path : string
        Path to save the plots

    starpair_list : list of strings
        List of star pairs for which to calculate (and plot) the SNR, in the format "reddenedstarname_comparisonstarname" (no spaces)

    plot : boolean [default=False]
        Whether or not to plot the SNR vs. wavelength for every curve

    Returns
    -------
    - Median SNRs in certain wavelength regions
    - Plots of the SNR vs. wavelength (if requested)
    """
    meds = np.zeros((3, len(starpair_list)))
    for j, starpair in enumerate(starpair_list):
        # obtain the extinction curve data
        extdata = ExtData("%s%s_ext.fits" % (data_path, starpair.lower()))

        # transform the curve from E(lambda-V) to A(lambda)/A(V)
        extdata.trans_elv_alav()

        # obtain flat arrays
        waves, exts, uncs = extdata.get_fitdata(["SpeX_SXD", "SpeX_LXD"])

        # calculate the median SNR in certain wavelength regions
        ranges = [
            (0.79, 2.54),
            (2.85, 4.05),
            (4.55, 5.5),
        ]
        SNR = exts / uncs
        for i, range in enumerate(ranges):
            mask = (waves.value > range[0]) & (waves.value < range[1])
            meds[i][j] = np.median(np.abs(SNR[mask]))

        # plot SNR vs. wavelength if requested
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(waves, SNR, s=1)
            plt.savefig(plot_path + starpair + "_SNR.pdf")

    print(ranges[0], np.nanmin(meds[0]), np.nanmax(meds[0]))
    print(ranges[1], np.nanmin(meds[1]), np.nanmax(meds[1]))
    print(ranges[2], np.nanmin(meds[2]), np.nanmax(meds[2]))


if __name__ == "__main__":
    # define the path to the data files
    spex_path = "/Users/mdecleir/Documents/NIR_ext/Data/SpeX_Data/Reduced_spectra/"
    data_path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    plot_path = "/Users/mdecleir/spex_nir_extinction/Figures/"

    # print and plot SNR of original spectra
    SNR_ori_spec(spex_path, plot_path)

    # print and plot SNR of used spectra
    stars = [
        "HD034759",
        "HD042560",
        "HD003360",
        "HD031726",
        "HD034816",
        "HD214680",
        "HD051283",
        "HD188209",
        "HD156247",
        "HD185418",
        "HD013338",
        "HD017505",
        "HD038087",
        "BD+56d524",
        "HD029309",
        "HD192660",
        "HD204827",
        "HD037061",
        "HD014956",
        "HD229238",
        "HD029647",
        "HD183143",
        "HD283809",
    ]
    SNR_final_spec(data_path, plot_path, stars)

    # print and plot SNR of extinction curves
    starpair_list = [
        "BD+56d524_HD034816",
        "HD013338_HD031726",
        "HD014956_HD214680",
        "HD017505_HD214680",
        "HD029309_HD042560",
        "HD029647_HD034759",
        "HD037061_HD034816",
        "HD038087_HD051283",
        "HD156247_HD042560",
        "HD183143_HD188209",
        "HD185418_HD034816",
        "HD192660_HD214680",
        "HD204827_HD003360",
        "HD229238_HD214680",
        "HD283809_HD003360",
    ]

    SNR_ext(data_path, plot_path, starpair_list)

    # measure the SNR in a few wavelength ranges for a few stars
    # HD283809
    ranges = [
        (0.805, 0.845),
        (1.114, 1.191),
        (1.413, 1.562),
        (1.966, 2.107),
        (2.176, 2.423),
        (2.406, 2.52),
        (3.36, 3.689),
        (3.772, 4.01),
        (4.705, 5.194),
    ]
    measure_SNR(spex_path, data_path, plot_path, "HD283809", ranges)
    # HD037061
    ranges = [
        (0.806, 0.843),
        (1.108, 1.273),
        (1.413, 1.562),
        (1.966, 2.107),
        (2.176, 2.423),
        (2.406, 2.52),
        (3.36, 3.689),
        (3.772, 3.997),
    ]
    measure_SNR(spex_path, data_path, plot_path, "HD037061", ranges)
    # HD185418
    ranges = [
        (0.807, 0.843),
        (1.108, 1.192),
        (1.413, 1.562),
        (1.966, 2.107),
        (2.176, 2.423),
        (2.418, 2.52),
        (2.88, 3.27),
        (3.33, 3.54),
    ]
    measure_SNR(spex_path, data_path, plot_path, "HD185418", ranges)
