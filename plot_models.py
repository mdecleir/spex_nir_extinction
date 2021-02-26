import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.modeling.powerlaws import PowerLaw1D

from dust_extinction.parameter_averages import CCM89, F19
from dust_extinction.conversions import AxAvToExv


def plot_ext_curve():
    x = np.arange(0.12, 3.33, 0.001) * u.micron

    curve = CCM89(Rv=3.1)
    plt.rc("axes.formatter", min_exponent=2)

    plt.plot(x.value, curve(x), label="Cardelli+1989")
    plt.xscale("log")
    plt.xticks(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 2, 3],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.0, 2.0, 3.0],
    )
    plt.legend()
    plt.savefig("Figures/Informative/CCM89_ext_curve.pdf")


def plot_powerlaws():
    fig, ax = plt.subplots()

    x = np.arange(0.78, 5.55, 0.001)

    model1 = PowerLaw1D(amplitude=1, alpha=1)
    model2 = PowerLaw1D(amplitude=0.5, alpha=1)
    model3 = PowerLaw1D(amplitude=1, alpha=2)

    plt.plot(x, model1(x), label=r"amp=1, $\alpha$=1, A(V)=0")
    plt.plot(x, model2(x), label=r"amp=0.5, $\alpha$=1, A(V)=0")
    plt.plot(x, model3(x), label=r"amp=1, $\alpha$=2, A(V)=0")

    ax.vlines(1, -1, 1, ls="--", lw=1, color="k")
    ax.hlines(1, 0.7, 1, ls="--", lw=1, color="k")
    ax.hlines(0.5, 0.7, 1, ls="--", lw=1, color="k")
    ax.hlines(0.0, 0.7, 1, ls="--", lw=1, color="k")

    model4 = PowerLaw1D(amplitude=1, alpha=1) | AxAvToExv(Av=1)
    model5 = PowerLaw1D(amplitude=1, alpha=0.8) | AxAvToExv(Av=1)
    model6 = PowerLaw1D(amplitude=1, alpha=0.8) | AxAvToExv(Av=1.1)
    model7 = PowerLaw1D(amplitude=1.2, alpha=1) | AxAvToExv(Av=1)
    model8 = PowerLaw1D(amplitude=1.2, alpha=1) | AxAvToExv(Av=1.05)
    plt.plot(x, model4(x), label=r"amp=1, $\alpha$=1, A(V)=1")
    plt.plot(x, model5(x), label=r"amp=1, $\alpha$=0.8, A(V)=1")
    plt.plot(x, model6(x), ls="--", label=r"amp=1, $\alpha$=0.8, A(V)=1.1")
    plt.plot(x, model7(x), label=r"amp=1.2, $\alpha$=1, A(V)=1")
    plt.plot(x, model8(x), ls="--", label=r"amp=1.2, $\alpha$=1, A(V)=1.05")

    # add some mock data points
    plt.scatter([1, 2, 3, 4, 5], [0, 1 / 2 - 1, 1 / 3 - 1, 1 / 4 - 1, 1 / 5 - 1])

    plt.xlabel(r"$\lambda$ [$\mu m$]")
    plt.ylabel("extinction")
    plt.xlim(0.7, 5.6)
    plt.ylim(-1, 1.3)
    plt.legend()
    plt.savefig("Figures/Informative/powlaws.pdf")


if __name__ == "__main__":
    plot_powerlaws()
    plot_ext_curve()
