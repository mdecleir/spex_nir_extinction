# This script is intended to measure the SNR of the spectra in different ways

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
import os

from astropy.modeling.models import Linear1D
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter
from astropy.stats import sigma_clipped_stats

from measure_extinction.stardata import StarData
from measure_extinction.plotting.plot_spec import plot_spectrum


def print_SNR(data_path, plot_path):
    """
    Print the SNR of the spectra
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


if __name__ == "__main__":
    # define the path to the data files
    spex_path = "/Users/mdecleir/Documents/NIR_ext/Data/SpeX_Data/Reduced_spectra/"
    data_path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    plot_path = "/Users/mdecleir/spex_nir_extinction/Figures/"

    # print the SNR information for all stars
    print_SNR(spex_path, plot_path)

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
