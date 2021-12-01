# This script plots laboratory ice spectra.
# The spectra were taken from the database of the Astrophysics \& Astrochemistry Laboratory at the NASA/Ames Research Center. The measurements of the spectra were carried out by Hudgins et al. (1993) (1993ApJS...86..713H).

import numpy as np
import pandas as pd

from astropy.table import Table
from astropy.modeling.models import Drude1D, custom_model, Gaussian1D, Lorentz1D
from astropy.modeling.fitting import LevMarLSQFitter
from matplotlib import pyplot as plt


def drude_modified(x, scale=1, x_o=1, gamma_o=1, asym=1):
    gamma = 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))
    y = scale * ((gamma / x_o) ** 2) / ((x / x_o - x_o / x) ** 2 + (gamma / x_o) ** 2)
    return y


def plot_ice():
    fs = 16
    plt.rc("xtick", direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", direction="in", labelsize=fs * 0.8)

    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # read the spectrum for the H2O ice
    file = "H2O_NASA.dat"
    table = pd.read_table(file, comment="#", sep="\s+")
    table = table[2531:]
    waves = 1 / table["Freq."] * 1e4
    norm = np.max(-table["%T,10K"] + 100)
    absorbs = (-table["%T,10K"] + 100) / norm

    # plot the spectrum
    ax[0].plot(waves, absorbs, color="k", label=r"H$_2$O ice")
    ax[1].plot(waves, absorbs, color="k", lw=1, ls="--", alpha=0.5, label=r"H$_2$O ice")

    # fit the spectrum
    drude = Drude1D(x_0=3.03, fixed={"amplitude": True})
    gauss = Gaussian1D(mean=3.03, stddev=0.13, fixed={"amplitude": True})
    lorentz = Lorentz1D(x_0=3.03, fixed={"amplitude": True})

    fit = LevMarLSQFitter()
    fit_result1 = fit(drude, waves, absorbs)
    fit_result2 = fit(gauss, waves, absorbs, maxiter=1000)
    fit_result3 = fit(lorentz, waves, absorbs, maxiter=10000)

    # calculate the residuals
    res1 = absorbs - fit_result1(waves)
    res2 = absorbs - fit_result2(waves)
    res3 = absorbs - fit_result3(waves)

    print("D", np.sum(res1 ** 2))
    print("G", np.sum(res2 ** 2))
    print("L", np.sum(res3 ** 2))

    # plot the fits
    ax[0].plot(
        waves,
        fit_result1(waves),
        ls="--",
        label="Drude",
    )
    ax[0].plot(
        waves,
        fit_result2(waves),
        ls=":",
        label="Gaussian",
    )
    ax[0].plot(
        waves,
        fit_result3(waves),
        ls="-.",
        label="Lorentz",
    )
    ax[0].legend()

    # read the spectrum for the mixed ice
    file = "mix_NASA.dat"
    table = pd.read_table(file, comment="#", sep="\s+")
    print(np.max(table["%T,10K"]))

    table = table[2531:]
    waves = 1 / table["Freq."] * 1e4
    norm = np.max(-table["%T,10K"] + 95)
    absorbs = (-table["%T,10K"] + 95) / norm

    # plot the spectrum
    plt.plot(waves, absorbs, color="k", label=r"mixed ice")

    # fit the spectrum
    drude = Drude1D(x_0=3.03, fixed={"amplitude": True})
    gauss = Gaussian1D(mean=3.03, stddev=0.13, fixed={"amplitude": True})
    lorentz = Lorentz1D(x_0=3.03, fixed={"amplitude": True})

    fit = LevMarLSQFitter()
    fit_result1 = fit(drude, waves, absorbs, maxiter=1000)
    fit_result2 = fit(gauss, waves, absorbs, maxiter=10000)
    fit_result3 = fit(lorentz, waves, absorbs, maxiter=10000)

    # calculate the residuals
    res1 = absorbs - fit_result1(waves)
    res2 = absorbs - fit_result2(waves)
    res3 = absorbs - fit_result3(waves)

    print("D", np.sum(res1 ** 2))
    print("G", np.sum(res2 ** 2))
    print("L", np.sum(res3 ** 2))

    print(fit_result1, fit_result2, fit_result3)

    plt.plot(
        waves,
        fit_result1(waves),
        ls="--",
        label="Drude",
    )
    plt.plot(
        waves,
        fit_result2(waves),
        ls=":",
        label="Gaussian",
    )
    plt.plot(
        waves,
        fit_result3(waves),
        ls="-.",
        label="Lorentz",
    )
    ax[1].legend()

    plt.xlim(2.5, 4)
    plt.xlabel(r"$\lambda$ [$\mu m$]", fontsize=fs)
    fig.text(
        0.06,
        0.5,
        "Normalized absorbance",
        rotation="vertical",
        va="center",
        ha="center",
        fontsize=fs,
    )
    plt.subplots_adjust(hspace=0)
    fig.savefig(
        "/Users/mdecleir/spex_nir_extinction/Figures/lab_ice.pdf", bbox_inches="tight"
    )


if __name__ == "__main__":
    plot_ice()
