#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.models import Drude1D, Gaussian1D, Lorentz1D, custom_model
from astropy.modeling.fitting import LevMarLSQFitter

from measure_extinction.extdata import ExtData
from dust_extinction.conversions import AxAvToExv

from emcee_fitting import EmceeFitter


def fit_function(
    dattype="elx",
    functype="pow",
    ice=False,
):
    """
    Define the fitting function

    Parameters
    ----------
    dattype : string [default="elx"]
        Data type to fit ("elx" or "alax")

    functype : string [default="pow"]
        Fitting function type ("pow" for powerlaw or "pol" for polynomial)

    ice : boolean [default=False]
        Whether or not to add the ice feature at 3.05 micron


    Returns
    -------
    func : Astropy CompoundModel
        The fitting function
    """
    # powerlaw model
    if functype == "pow":
        func = PowerLaw1D(fixed={"x_0": True})
    elif functype == "pol":  # polynomial model
        func = Polynomial1D(degree=6)
    else:
        warnings.warn(
            'Unknown function type, choose "pow" for a powerlaw or "pol" for a polynomial',
            stacklevel=2,
        )

    # add a Drude profile for the ice feature if requested
    if ice:
        # func += Drude1D(
        #     x_0=3.05,
        #     fwhm=0.6,
        #     bounds={"fwhm": (0.5, 1)},
        # )

        # def drude_modified(x, scale=1, x_o=1, gamma_o=1, asym=1):
        #     gamma = 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))
        #     y = (
        #         scale
        #         * ((gamma / x_o) ** 2)
        #         / ((x / x_o - x_o / x) ** 2 + (gamma / x_o) ** 2)
        #     )
        #     return y
        #
        # Drude_modified_model = custom_model(drude_modified)
        # func += Drude_modified_model(x_o=3.05, gamma_o=0.3)

        # func += Drude_modified_model(
        #     x_o=3.0146054034063385,
        #     gamma_o=0.4780290123691583,
        #     asym=-3.3249381815320094,
        #     fixed={"x_o": True, "gamma_o": True, "asym": True},
        # )

        # 2 asymmetric (modified) Gaussians
        def gauss_asymmetric(x, scale=1, x_o=1, gamma_o=1, asym=1):
            gamma = 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))
            # gamma is full width, so stddev=gamma/(2sqrt(2ln2))
            y = scale * np.exp(
                -((x - x_o) ** 2) / (2 * (gamma / (2 * np.sqrt(2 * np.log(2)))) ** 2)
            )
            return y

        Gaussian_asym = custom_model(gauss_asymmetric)
        # func += Gaussian_asym(x_o=3, gamma_o=0.3) + Gaussian_asym(x_o=3.4, gamma_o=0.4)

        func += Gaussian_asym(
            x_o=3.011498634873138,
            gamma_o=-0.33575327133587224,
            asym=-3.9652666570924633,
            fixed={"x_o": True, "gamma_o": True, "asym": True},
        ) + Gaussian_asym(
            x_o=3.5278583234132412,
            gamma_o=-1.3852846071334977,
            asym=-11.262827694055785,
            fixed={"x_o": True, "gamma_o": True, "asym": True},
        )

    # convert the function from A(lambda)/A(V) to E(lambda-V)
    if dattype == "elx":
        func = func | AxAvToExv()

    return func


def fit_features(starpair, path):
    """
    Fit the features separately with different profiles

    Parameters
    ----------
    starpair : string
        Name of the star pair for which to fit the extinction features, in the format "reddenedstarname_comparisonstarname" (no spaces)

    path : string
        Path to the data files

    Returns
    -------
    waves : np.ndarray
        Array with wavelengths
    exts_sub : np.ndarray
        Array with continuum subtracted extinctions
    results : list
        List with the fitted models for different profiles
    """
    # first, fit the continuum, excluding the region of the features
    fit_spex_ext(starpair, path, exclude=(2.8, 3.8))

    # retrieve the SpeX data to be fitted, and sort the curve from short to long wavelengths
    extdata = ExtData("%s%s_ext.fits" % (path, starpair.lower()))
    (waves, exts, exts_unc) = extdata.get_fitdata(["SpeX_SXD", "SpeX_LXD"])
    indx = np.argsort(waves)
    waves = waves[indx].value
    exts = exts[indx]
    exts_unc = exts_unc[indx]

    # subtract the fitted (powerlaw) continuum from the data, and select the relevant region
    params = extdata.model["params"]
    exts_sub = exts - (params[0] * params[3] * waves ** (-params[2]) - params[3])
    mask = (waves >= 2.8) & (waves <= 3.8)
    waves = waves[mask]
    exts_sub = exts_sub[mask]
    exts_unc = exts_unc[mask]

    # define different profiles
    # 2 Gaussians (stddev=FWHM/(2sqrt(2ln2)))
    gauss = Gaussian1D(mean=3, stddev=0.13) + Gaussian1D(mean=3.4, stddev=0.17)

    # 2 Drudes
    drude = Drude1D(x_0=3, fwhm=0.3) + Drude1D(x_0=3.4, fwhm=0.4)

    # 2 Lorentzians
    lorentz = Lorentz1D(x_0=3, fwhm=0.3) + Lorentz1D(x_0=3.4, fwhm=0.4)

    # 2 asymmetric Gaussians
    def gauss_asymmetric(x, scale=1, x_o=1, gamma_o=1, asym=1):
        gamma = 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))
        # gamma is full width, so stddev=gamma/(2sqrt(2ln2))
        y = scale * np.exp(
            -((x - x_o) ** 2) / (2 * (gamma / (2 * np.sqrt(2 * np.log(2)))) ** 2)
        )
        return y

    Gaussian_asym = custom_model(gauss_asymmetric)
    gauss_asym = Gaussian_asym(x_o=3, gamma_o=0.3) + Gaussian_asym(x_o=3.4, gamma_o=0.4)

    # 2 "asymmetric" Drudes
    def drude_asymmetric(x, scale=1, x_o=1, gamma_o=1, asym=1):
        gamma = 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))
        y = scale * (gamma / x_o) ** 2 / ((x / x_o - x_o / x) ** 2 + (gamma / x_o) ** 2)
        return y

    Drude_asym = custom_model(drude_asymmetric)
    drude_asym = Drude_asym(x_o=3, gamma_o=0.3) + Drude_asym(x_o=3.4, gamma_o=0.4)

    # 2 asymmetric Lorentzians
    def lorentz_asymmetric(x, scale=1, x_o=1, gamma_o=1, asym=1):
        gamma = 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))
        # gamma is full width, so gamma_formula=gamma/2
        y = scale * (gamma / 2) ** 2 / ((gamma / 2) ** 2 + (x - x_o) ** 2)
        return y

    Lorentzian_asym = custom_model(lorentz_asymmetric)
    lorentz_asym = Lorentzian_asym(x_o=3, gamma_o=0.3) + Lorentzian_asym(
        x_o=3.4, gamma_o=0.4
    )

    profiles = [gauss, drude, lorentz, gauss_asym, drude_asym, lorentz_asym]

    # fit the different profiles
    fit = LevMarLSQFitter()
    results = []
    for profile in profiles:
        fit_result = fit(profile, waves, exts_sub, weights=1 / exts_unc, maxiter=10000)
        results.append(fit_result)
        print(fit_result)
        print("Chi2", np.sum(((exts_sub - fit_result(waves)) / exts_unc) ** 2))

    return waves, exts_sub, results


def fit_spex_ext(starpair, path, functype="pow", ice=False, exclude=None):
    """
    Fit the observed SpeX NIR extinction curve

    Parameters
    ----------
    starpair : string
        Name of the star pair for which to fit the extinction curve, in the format "reddenedstarname_comparisonstarname" (no spaces), or "average" to fit the average extinction curve

    path : string
        Path to the data files

    functype : string [default="pow"]
        Fitting function type ("pow" for powerlaw or "pol" for polynomial)

    ice : boolean [default=False]
        Whether or not to fit the ice feature at 3.05 micron

    exclude : tuple [default=None]
        Wavelength region (min,max) to be excluded from the fitting


    Returns
    -------
    Updates extdata.model["type", "waves", "exts", "residuals", "chi2", "params"] and extdata.columns["AV"] with the fitting results:
        - type: string with the type of model (e.g. "pow_elx_Drude")
        - waves: np.ndarray with the SpeX wavelengths
        - exts: np.ndarray with the fitted model to the extinction curve at "waves" wavelengths
        - residuals: np.ndarray with the residuals, i.e. data-fit, at "waves" wavelengths
        - chi2 : float with the chi square of the fitting
        - params: list with output Parameter objects
    """
    # retrieve the SpeX data to be fitted, and sort the curve from short to long wavelengths
    extdata = ExtData("%s%s_ext.fits" % (path, starpair.lower()))
    (waves, exts, exts_unc) = extdata.get_fitdata(["SpeX_SXD", "SpeX_LXD"])
    indx = np.argsort(waves)
    waves = waves[indx].value
    exts = exts[indx]
    exts_unc = exts_unc[indx]

    # exclude a wavelength region if requested
    if exclude:
        mask = (waves < exclude[0]) | (waves > exclude[1])
        waves = waves[mask]
        exts = exts[mask]
        exts_unc = exts_unc[mask]

    # obtain the function to fit
    if "SpeX_LXD" not in extdata.waves.keys():
        ice = False
    func = fit_function(dattype=extdata.type, functype=functype, ice=ice)

    # use the Levenberg-Marquardt algorithm to fit the data with the model
    fit = LevMarLSQFitter()
    fit_result_lev = fit(func, waves, exts, weights=1 / exts_unc, maxiter=10000)

    # set up the backend to save the samples for the emcee runs
    emcee_samples_file = path + "Fitting_results/" + starpair + "_emcee_samples.h5"

    # do the fitting again, with MCMC, using the results from the first fitting as input
    fit2 = EmceeFitter(nsteps=10000, burnfrac=0.1, save_samples=emcee_samples_file)

    # add parameter bounds
    for param in fit_result_lev.param_names:
        if "amplitude" in param:
            getattr(fit_result_lev, param).bounds = (0, 2)
        elif "alpha" in param:
            getattr(fit_result_lev, param).bounds = (0, 4)
        elif "Av" in param:
            getattr(fit_result_lev, param).bounds = (0, 10)
    fit_result_mcmc = fit2(fit_result_lev, waves, exts, weights=1 / exts_unc)

    # create standard MCMC plots
    fit2.plot_emcee_results(
        fit_result_mcmc, filebase=path + "Fitting_results/" + starpair
    )

    # choose the fit result to save
    fit_result = fit_result_mcmc
    # fit_result = fit_result_lev

    # determine the wavelengths at which to evaluate and save the fitted model curve: all SpeX wavelengths, sorted from short to long (to avoid problems with overlap between SXD and LXD), and shortest and longest wavelength should have data
    if "SpeX_LXD" not in extdata.waves.keys():
        full_waves = extdata.waves["SpeX_SXD"].value
        full_npts = extdata.npts["SpeX_SXD"]
    else:
        full_waves = np.concatenate(
            (extdata.waves["SpeX_SXD"].value, extdata.waves["SpeX_LXD"].value)
        )
        full_npts = np.concatenate((extdata.npts["SpeX_SXD"], extdata.npts["SpeX_LXD"]))
    indxs = np.argsort(full_waves)[
        np.logical_and(full_waves >= np.min(waves), full_waves <= np.max(waves))
    ]
    full_waves = full_waves[indxs]
    full_npts = full_npts[indxs]

    # calculate the residuals and put them in an array of the same length as "full_waves" for plotting
    residuals = exts - fit_result(waves)
    full_res = np.full_like(full_npts, np.nan)
    if exclude:
        new_indx = (full_waves < exclude[0]) | (full_waves > exclude[1])
        full_res[(full_npts > 0) * new_indx] = residuals
    else:
        full_res[(full_npts > 0)] = residuals

    # save the fitting results to the fits file
    if ice:
        functype += "_Drude"
    extdata.model["type"] = functype + "_" + extdata.type
    extdata.model["waves"] = full_waves
    extdata.model["exts"] = fit_result(full_waves)
    extdata.model["residuals"] = full_res
    extdata.model["chi2"] = np.sum((residuals / exts_unc) ** 2)
    print("Chi2", extdata.model["chi2"])
    extdata.model["params"] = []
    for param in fit_result.param_names:
        extdata.model["params"].append(getattr(fit_result, param))
        if "Av" in param:
            extdata.columns["AV"] = (
                getattr(fit_result, param).value,
                getattr(fit_result, param).unc_minus,
                getattr(fit_result, param).unc_plus,
            )
            extdata.calc_RV()
            print(extdata.columns["RV"])
    extdata.save("%s%s_ext.fits" % (path, starpair.lower()))


if __name__ == "__main__":
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    fit_spex_ext("HD283809_HD003360", path)
