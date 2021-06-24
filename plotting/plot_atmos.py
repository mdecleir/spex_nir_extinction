# This script plots the atmospheric tranmission at a resolution of 2000.
# The spectrum was taken from the Spextool database, and is a model of the atmospheric transmission computed with ATRAN (Lord 1992, https://ui.adsabs.harvard.edu/abs/1992nstc.rept.....L/abstract, https://atran.arc.nasa.gov/cgi-bin/atran/atran.cgi)

from astropy.io import fits
from matplotlib import pyplot as plt


def plot_atmos(path):
    hdulist = fits.open(path + "data/atran2000.fits")
    data = hdulist[0].data

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(data[0], data[1], color="k", alpha=0.8, lw=0.6)
    plt.axvspan(1.347, 1.415, alpha=0.6, color="red", zorder=3)
    plt.axvspan(1.798, 1.949, alpha=0.6, color="red", zorder=3)
    plt.axvspan(2.514, 2.880, alpha=0.6, color="red", zorder=3)
    plt.axvspan(4.000, 4.594, alpha=0.6, color="red", zorder=3)

    plt.xlim(0.78, 5.55)
    plt.ylim(0.0, 1.01)
    plt.xticks(fontsize=fs * 0.8)
    plt.yticks(fontsize=fs * 0.8)
    plt.xlabel(r"$\lambda$ [$\mu m$]")
    plt.ylabel("Atmospheric transmission")
    plt.savefig(path + "Figures/atmos_trans.pdf", bbox_inches="tight")


if __name__ == "__main__":
    path = "/Users/mdecleir/spex_nir_extinction/"
    # plotting settings for uniform plots
    fs = 20
    plt.rc("font", size=fs)
    plt.rc("xtick", top=True, direction="in")
    plt.rc("ytick", direction="in")
    plt.rc("xtick.major", width=1, size=8)
    plt.rc("ytick.major", width=1, size=8)

    plot_atmos(path)
