# This script plots the atmospheric tranmission at a resolution of 2000.
# The spectrum was taken from the Spextool database, and is a model of the atmospheric transmission computed with ATRAN (Lord 1992, https://ui.adsabs.harvard.edu/abs/1992nstc.rept.....L/abstract, https://atran.arc.nasa.gov/cgi-bin/atran/atran.cgi)

from astropy.io import fits
from matplotlib import pyplot as plt


def plot_atmos():
    hdulist = fits.open("/Users/mdecleir/spex_nir_extinction/atran2000.fits")
    data = hdulist[0].data

    fs = 16
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(data[0], data[1], color="k", alpha=0.8, lw=0.6)
    plt.axvspan(1.354, 1.411, alpha=0.6, color="red", zorder=3)
    plt.axvspan(1.805, 1.947, alpha=0.6, color="red", zorder=3)
    plt.axvspan(2.522, 2.875, alpha=0.6, color="red", zorder=3)
    plt.axvspan(4.014, 4.594, alpha=0.6, color="red", zorder=3)

    plt.xlim(0.78, 5.55)
    plt.ylim(0.0, 1.01)
    plt.xticks(fontsize=fs * 0.8)
    plt.yticks(fontsize=fs * 0.8)
    plt.xlabel(r"$\lambda$ [$\mu m$]", fontsize=fs)
    plt.ylabel("Atmospheric transmission", fontsize=fs)
    plt.savefig("Figures/atmos_trans.pdf", bbox_inches="tight")


if __name__ == "__main__":
    plot_atmos()
