import argparse
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.models import Drude1D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.fitting import _fitter_to_model_params

# from models_mcmc_extension import EmceeFitter
from emcee_fitting import EmceeFitter

from dust_extinction.conversions import AxAvToExv

from measure_extinction.extdata import ExtData


class ModDrude1D(Fittable1DModel):
    r"""
    Modified Drude
    """
    n_inputs = 1
    n_outputs = 1

    scale = Parameter(default=0.1)
    x_o = Parameter(default=3.0, min=0.0)
    gamma_o = Parameter(default=1.0, min=0.0)
    asym = Parameter(default=-0.5)

    @staticmethod
    def evaluate(x, scale, x_o, gamma_o, asym):
        """
        Modified Drude function to have a variable asymmetry.  Drude profiles
        are intrinsically asymmetric with the asymmetry fixed by specific central
        wavelength and width.  This modified Drude introduces an asymmetry
        parameter that allows for variable asymmetry at fixed central wavelength
        and width.

        Parameters
        ----------
        x : float
            input wavelengths

        scale : float
            central amplitude

        x_o : float
            central wavelength

        gamma_o : float
            full-width-half-maximum of profile

        asym : float
            asymmetry where a value of 0 results in a standard Drude profile
        """
        gamma = 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))
        y = (
            scale
            * ((gamma / x_o) ** 2)
            / ((x / x_o - x_o / x) ** 2 + (gamma / x_o) ** 2)
        )

        return y


class ModGauss1D(Fittable1DModel):
    r"""
    Modified Gaussian
    """
    n_inputs = 1
    n_outputs = 1

    scale = Parameter(default=1.0)
    x_o = Parameter(default=3.0, min=0.0)
    gamma_o = Parameter(default=1.0, min=0.0)
    asym = Parameter(default=0.0)

    @staticmethod
    def evaluate(x, scale, x_o, gamma_o, asym):
        """

        Parameters
        ----------
        x : float
            input wavelengths

        scale : float
            central amplitude

        x_o : float
            central wavelength

        gamma_o : float
            full-width-half-maximum of profile

        asym : float
            asymmetry where a value of 0 results in a standard Drude profile
        """
        # gamma replaces FWHM, so stddev=gamma/(2sqrt(2ln2))
        gamma = 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))
        y = scale * np.exp(
            -((x - x_o) ** 2) / (2 * (gamma / (2 * np.sqrt(2 * np.log(2)))) ** 2)
        )
        return y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # plotting setup for easier to read plots
    fontsize = 18
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1.5)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    fig, tax = plt.subplots(
        ncols=2, nrows=2, figsize=(14, 6), gridspec_kw={"height_ratios": [3, 1]}
    )

    # filename = "hd029647_hd034759_ext.fits"
    # filename = "hd029647_hd042560_ext.fits"
    filename = "hd283809_hd003360_ext.fits"
    ext = ExtData(filename)

    (wave, y, y_unc) = ext.get_fitdata(["SpeX_SXD", "SpeX_LXD"])
    # remove units as fitting routines often cannot take numbers with units
    x = wave.to(u.micron).value
    gvals = (0.6 < x) & (x < 6.0)
    # print(y_unc[gvals])
    # gvals = np.logical_or(x < 3.18, x > 3.4)
    weights = 1.0 / (y_unc[gvals])
    # weights = np.full((len(x)), 0.1)
    # weight ice feature
    # weights[(2.9 < x) & (x < 3.2)] *= 5
    # weights[(3.2 < x) & (x < 3.4)] /= 5
    # weights[(2.3 < x) & (x < 2.4)] *= 5
    # weights[(3.35 < x) & (x < 3.45)] *= 5

    ax = tax[0, 0]
    ext.plot(ax)
    ax.set_xlim(0.6, 5.0)
    ax.set_ylabel(r"$E(\lambda -V)$")

    ax = tax[1, 0]
    ax.set_xlim(0.6, 5.0)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel(r"$\lambda$ [$\mu m$]")

    ax = tax[0, 1]
    ext.plot(ax)
    ax.set_xlim(2.0, 4.0)
    ax.set_ylabel(r"$E(\lambda -V)$")

    ax = tax[1, 1]
    ax.set_xlim(2.0, 4.0)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel(r"$\lambda$ [$\mu m$]")

    if "hd029647" in filename:
        tax[0, 0].set_ylim(-4.0, -1.5)
        tax[0, 1].set_ylim(-3.8, -3.1)
    else:
        tax[0, 0].set_ylim(-6.0, -2.0)
        tax[0, 1].set_ylim(-5.5, -5.0)

    # now fit

    av_guess = 5.5
    mod_init1 = (PowerLaw1D() + ModDrude1D()) | AxAvToExv(Av=av_guess)
    mod_init2 = (PowerLaw1D() + ModGauss1D()) | AxAvToExv(Av=av_guess)
    # mod_init1 = (PowerLaw1D() + ModDrude1D() + ModDrude1D() + ModDrude1D()) | AxAvToExv(
    #     Av=av_guess
    # )
    # mod_init2 = (PowerLaw1D() + ModGauss1D() + ModGauss1D() + ModGauss1D()) | AxAvToExv(
    #     Av=av_guess
    # )
    mod_inits = [mod_init1, mod_init2]
    fit = LevMarLSQFitter()

    np.set_printoptions(precision=3, suppress=True)
    print(mod_inits[0].param_names)
    mod_x = np.arange(0.6, 6.0, 0.01)
    psyms = ["k--", "k:"]
    labels = ["ModDrude", "ModGauss"]
    for cmod, cpsym, clabel in zip(mod_inits, psyms, labels):
        print(clabel)
        # print(cmod)

        cmod[0].x_0 = 1.0
        cmod[0].x_0.fixed = True
        cmod[0].amplitude = 0.4
        cmod[0].alpha = 2.0

        cmod[1].scale = 0.1
        cmod[1].scale.bounds = [0.0, None]
        cmod[1].x_o = 3.00
        cmod[1].x_o.bounds = [2.5, 3.5]
        # cmod[1].x_o.fixed = True
        cmod[1].gamma_o = 0.4
        cmod[1].gamma_o.bounds = [0.0, 2.0]
        # cmod[1].asym = -1.0
        cmod[1].asym.bounds = [-100.0, 100.0]
        # cmod[1].asym.fixed = True

        # cmod[2].scale = 0.0
        # cmod[2].scale.fixed = True
        # cmod[2].x_o = 3.4
        # cmod[2].x_o.fixed = True
        # cmod[2].gamma_o = 0.2
        # cmod[2].gamma_o.fixed = True
        # cmod[2].asym = 0.0
        # cmod[2].asym.fixed = True
        #
        # cmod[3].scale = 0.0
        # cmod[3].scale.fixed = True
        # cmod[3].x_o = 2.35
        # cmod[3].x_o.fixed = True
        # cmod[3].gamma_o = 0.2
        # cmod[3].gamma_o.fixed = True
        # cmod[3].asym = 0.0
        # cmod[3].asym.fixed = True

        cfit = fit(
            cmod,
            x[gvals],
            y[gvals],
            maxiter=1000,
            weights=weights[gvals],
        )
        print(cfit.parameters)

        mod_y = cfit(mod_x)

        tax[0, 0].plot(mod_x, mod_y, cpsym, label=clabel)
        tax[0, 1].plot(mod_x, mod_y, cpsym, label=clabel)
        tax[1, 0].plot(x, y - cfit(x), cpsym, alpha=0.5)
        tax[1, 1].plot(x, y - cfit(x), cpsym, alpha=0.5)

        # MCMC
        nsteps = 10000
        burnfrac = 0.25
        emcee_samples_file = filename.replace(".fits", ".h5")
        fit2 = EmceeFitter(
            nsteps=nsteps, burnfrac=burnfrac, save_samples=emcee_samples_file
        )

        # cfit[1].x_o.fixed = True
        # cfit[1].gamma_o.fixed = True
        # cfit[1].asym.fixed = True

        cfit2 = fit2(cfit, x[gvals], y[gvals], weights=weights[gvals])
        print(cfit2.parameters)
        # make the standard mcmc plots
        fit2.plot_emcee_results(cfit2, filebase=filename.replace(".fits", f"_{clabel}"))

        # plot samples from the mcmc chaing
        flat_samples = fit2.fit_info["sampler"].get_chain(
            discard=int(burnfrac * nsteps), flat=True
        )
        inds = np.random.randint(len(flat_samples), size=100)
        model_copy = cfit2.copy()
        for ind in inds:
            sample = flat_samples[ind]
            _fitter_to_model_params(model_copy, sample)
            tax[0, 0].plot(mod_x, model_copy(mod_x), cpsym, alpha=0.05, color="b")
            tax[0, 1].plot(mod_x, model_copy(mod_x), cpsym, alpha=0.05, color="b")

    fig.tight_layout()

    # plot or save to a file
    outname = filename.replace(".fits", "_nirext_fitting")
    if args.png:
        fig.savefig(outname + ".png")
    elif args.pdf:
        fig.savefig(outname + ".pdf")
    else:
        plt.show()
