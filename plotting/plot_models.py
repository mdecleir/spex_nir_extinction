import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.modeling.powerlaws import PowerLaw1D

from dust_extinction.parameter_averages import CCM89, F19
from dust_extinction.conversions import AxAvToExv

from dust_extinction.averages import (
    RL85_MWGC,
    RRP89_MWGC,
    I05_MWAvg,
    CT06_MWLoc,
    CT06_MWGC,
    F11_MWGC,
    G21_MWAvg,
    GCC09_MWAvg,
    B92_MWAvg,
    G03_SMCBar,
    G03_LMCAvg,
    G03_LMC2,
)


def plot_ext_curves():
    """
    Plot model extinction curves.
    """
    # create the plot
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # UV-NIR (Cardelli et al. 1989)
    x = np.arange(0.12, 3.33, 0.001) * u.micron
    curve = CCM89(Rv=3.1)
    ax.plot(x.value, curve(x), label="Cardelli+1989")

    # NIR-MIR (Gordon et al. 2021)
    x = np.arange(3, 30.0, 0.1) * u.micron
    curve = G21_MWAvg()
    ax.plot(x, curve(x), label="Gordon+2021")

    # finalize and save the plot
    ax.legend(loc="best")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.rc("axes.formatter", min_exponent=2)
    plt.xticks(
        [0.1, 0.2, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 8, 10, 20, 30],
        [0.1, 0.2, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 8, 10, 20, 30],
    )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig("Figures/model_ext_curve.pdf")

    # Gordon et al. 2009
    fig, ax = plt.subplots(figsize=(6, 3.5))
    curve = GCC09_MWAvg()
    x = np.arange(0.116, 0.3, 0.001) * u.micron
    ax.plot(x, curve(x), label="Gordon+2009")
    plt.tight_layout()
    plt.savefig("Figures/GCC09_ext_curve.pdf")


def plot_powerlaws():
    """
    Plot model power laws.
    """
    # create the plot
    fig, ax = plt.subplots()
    x = np.arange(0.78, 5.55, 0.001)

    # create and plot some model power laws
    model1 = PowerLaw1D(amplitude=0.4, alpha=1.6)
    model2 = PowerLaw1D(amplitude=0.35, alpha=2)
    plt.plot(x, model1(x))
    plt.plot(x, model2(x))

    # finalize and save the figure
    plt.xlabel(r"$\lambda$ [$\mu m$]")
    plt.ylabel("extinction")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()
    plt.savefig("Figures/powlaws.pdf")


if __name__ == "__main__":
    plot_ext_curves()
    plot_powerlaws()
