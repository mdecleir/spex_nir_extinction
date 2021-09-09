#!/usr/bin/env python

import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.models import (
    Linear1D,
    Drude1D,
    Gaussian1D,
    Lorentz1D,
    custom_model,
)
from astropy.modeling.fitting import LevMarLSQFitter, LinearLSQFitter
from astropy import uncertainty as unc

from measure_extinction.extdata import ExtData
from measure_extinction.stardata import StarData
from measure_extinction.plotting.plot_spec import plot_spectrum
from dust_extinction.conversions import AxAvToExv

from emcee_fitting import EmceeFitter

# gamma function (wavelength dependent width, replacing the FWHM)
def gamma(x, x_o=1, gamma_o=1, asym=1):
    return 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))


# asymmetric Gaussian
def gauss_asymmetric(x, scale=1, x_o=1, gamma_o=1, asym=1):
    # gamma replaces FWHM, so stddev=gamma/(2sqrt(2ln2))
    y = scale * np.exp(
        -((x - x_o) ** 2)
        / (2 * (gamma(x, x_o, gamma_o, asym) / (2 * np.sqrt(2 * np.log(2)))) ** 2)
    )
    return y


# "asymmetric" Drude
def drude_asymmetric(x, scale=1, x_o=1, gamma_o=1, asym=1):
    y = (
        scale
        * (gamma(x, x_o, gamma_o, asym) / x_o) ** 2
        / ((x / x_o - x_o / x) ** 2 + (gamma(x, x_o, gamma_o, asym) / x_o) ** 2)
    )
    return y


# asymmetric Lorentzian
def lorentz_asymmetric(x, scale=1, x_o=1, gamma_o=1, asym=1):
    # gamma replaces FWHM, so gamma_formula=gamma/2
    y = (
        scale
        * (gamma(x, x_o, gamma_o, asym) / 2) ** 2
        / ((gamma(x, x_o, gamma_o, asym) / 2) ** 2 + (x - x_o) ** 2)
    )
    return y


def fit_function(
    dattype="elx",
    functype="pow",
    dense=False,
    profile="gauss_asym",
    fixed=False,
    AV_guess=3,
):
    """
    Define the fitting function

    Parameters
    ----------
    dattype : string [default="elx"]
        Data type to fit ("elx" or "alax")

    functype : string [default="pow"]
        Fitting function type ("pow" for powerlaw or "pol" for polynomial)

    dense : boolean [default=False]
        Whether or not to fit the feature around 3 micron

    profile : string [default="gauss_asym"]
        Profile to use for the features if dense = True (options are "gauss", "drude", "lorentz", "gauss_asym", "drude_asym", "lorentz_asym")

    fixed : boolean [default=False]
        Whether or not to add a fixed feature around 3 micron (for diffuse sightlines)

    AV_guess : float [default=3]
        Initial guess for A(V)

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

    # add profiles for the features if requested
    if dense:
        # define different profiles
        # 1 Gaussian (stddev=FWHM/(2sqrt(2ln2)))
        gauss1 = Gaussian1D(mean=3, stddev=0.13, bounds={"stddev": (0.1, 0.2)})

        # 2 Gaussians (stddev=FWHM/(2sqrt(2ln2)))
        gauss2 = Gaussian1D(
            mean=3, stddev=0.13, bounds={"stddev": (0.12, 0.16)}
        ) + Gaussian1D(
            mean=3.4, stddev=0.14, bounds={"mean": (3.41, 3.45), "stddev": (0.14, 0.2)}
        )

        # 1 Drude
        drude1 = Drude1D(x_0=3, fwhm=0.3, bounds={"fwhm": (0.2, 0.5)})

        # 2 Drudes
        drude2 = Drude1D(x_0=3, fwhm=0.3) + Drude1D(
            x_0=3.4, fwhm=0.15, bounds={"x_0": (3.35, 3.43), "fwhm": (0.14, 0.3)}
        )

        # 1 Lorentzian
        lorentz1 = Lorentz1D(x_0=3, fwhm=0.3, bounds={"x_0": (2.99, 3.1)})

        # 2 Lorentzians
        lorentz2 = Lorentz1D(
            x_0=3, fwhm=0.3, bounds={"x_0": (2.99, 3.1), "fwhm": (0.28, 0.4)}
        ) + Lorentz1D(
            x_0=3.4, fwhm=0.15, bounds={"x_0": (3.35, 3.43), "fwhm": (0.14, 0.3)}
        )

        # 1 asymmetric Gaussian
        Gaussian_asym = custom_model(gauss_asymmetric)
        gauss_asym1 = Gaussian_asym(
            x_o=3,
            gamma_o=0.4,
            bounds={"x_o": (2.9, 3.1), "gamma_o": (0.35, 2), "asym": (-100, 100)},
        )

        # 2 asymmetric Gaussians
        gauss_asym2 = Gaussian_asym(
            x_o=3,
            gamma_o=0.3,
            bounds={"x_o": (2.99, 3.04), "gamma_o": (0.28, 0.5), "asym": (-10, 10)},
        ) + Gaussian_asym(
            x_o=3.4,
            gamma_o=0.15,
            bounds={
                "x_o": (3.3, 3.42),
                "scale": (0.005, None),
                "gamma_o": (0.15, 0.5),
                "asym": (-20, -4),
            },
        )

        # 1 "asymmetric" Drude
        Drude_asym = custom_model(drude_asymmetric)
        drude_asym1 = Drude_asym(
            x_o=3.0,
            gamma_o=0.3,
            bounds={
                "scale": (0, 2),
                "x_o": (2.5, 3.5),
                "gamma_o": (-2, 2),
                "asym": (-50, 50),
            },
        )

        # 2 "asymmetric" Drudes
        drude_asym2 = Drude_asym(x_o=3, gamma_o=0.3) + Drude_asym(
            x_o=3.4, gamma_o=0.15, bounds={"x_o": (3.35, 3.45)}
        )

        # 1 asymmetric Lorentzian
        Lorentzian_asym = custom_model(lorentz_asymmetric)
        lorentz_asym1 = Lorentzian_asym(x_o=3, gamma_o=0.3, bounds={"x_o": (2.95, 3.1)})

        # 2 asymmetric Lorentzians
        lorentz_asym2 = Lorentzian_asym(x_o=3, gamma_o=0.3) + Lorentzian_asym(
            x_o=3.4, gamma_o=0.15
        )

        profiles = {
            "gauss1": gauss1,
            "drude1": drude1,
            "lorentz1": lorentz1,
            "gauss_asym1": gauss_asym1,
            "drude_asym1": drude_asym1,
            "lorentz_asym1": lorentz_asym1,
            "gauss2": gauss2,
            "drude2": drude2,
            "lorentz2": lorentz2,
            "gauss_asym2": gauss_asym2,
            "drude_asym2": drude_asym2,
            "lorentz_asym2": lorentz_asym2,
        }
        func += profiles[profile]

    if fixed:
        # fit a fixed feature for diffuse sightlines
        Drude_asym = custom_model(drude_asymmetric)
        func += Drude_asym(
            x_o=3.017727049,
            gamma_o=0.462375776,
            asym=-2.873011454,
            bounds={"scale": (0, 2)},
            fixed={"x_o": True, "gamma_o": True, "asym": True},
        )

    # convert the function from A(lambda)/A(V) to E(lambda-V)
    if dattype == "elx":
        func = func | AxAvToExv(Av=AV_guess)

    return func


def fit_features_spec(star, path):
    """
    Fit the features directly from the spectrum with different profiles

    Parameters
    ----------
    star : string
        Name of the reddened star for which to fit the features in the spectrum

    path : string
        Path to the data files

    Returns
    -------
    waves : np.ndarray
        Numpy array with wavelengths

    flux_sub : np.ndarray
        Numpy array with continuum subtracted fluxes

    results : list
        List with the fitted models for different profiles
    """
    # obtain the spectrum of the reddened star
    stardata = StarData(star + ".dat", path)
    npts = stardata.data["SpeX_LXD"].npts
    waves = stardata.data["SpeX_LXD"].waves.value
    flux_unc = stardata.data["SpeX_LXD"].uncs

    # "manually" obtain the continuum from the spectrum (i.e. read the flux at 2.4 and 3.6 micron)
    plot_spectrum(
        star,
        path,
        mlam4=True,
        range=[2, 4.5],
        exclude=["IRS", "STIS_Opt"],
    )

    # fit the continuum reference points with a straight line
    ref_waves = [2.4, 3.6]
    fluxes = [3.33268e-12, 4.053e-12]
    func = Linear1D()
    fit = LinearLSQFitter()
    fit_result = fit(func, ref_waves, fluxes)

    # subtract the continuum from the fluxes
    fluxes = stardata.data["SpeX_LXD"].fluxes.value * waves ** 4 - fit_result(waves)

    # define different profiles
    # 2 Gaussians (stddev=FWHM/(2sqrt(2ln2)))
    gauss = Gaussian1D(mean=3, stddev=0.13) + Gaussian1D(
        mean=3.4, stddev=0.06, fixed={"mean": True}
    )

    # 2 Drudes
    drude = Drude1D(x_0=3, fwhm=0.3) + Drude1D(x_0=3.4, fwhm=0.15, fixed={"x_0": True})

    # 2 Lorentzians
    lorentz = Lorentz1D(x_0=3, fwhm=0.3) + Lorentz1D(
        x_0=3.4, fwhm=0.15, fixed={"x_0": True}
    )

    # 2 asymmetric Gaussians
    Gaussian_asym = custom_model(gauss_asymmetric)
    gauss_asym = Gaussian_asym(x_o=3, gamma_o=0.3) + Gaussian_asym(
        x_o=3.4, gamma_o=0.15, fixed={"x_o": True}
    )

    # 2 "asymmetric" Drudes
    Drude_asym = custom_model(drude_asymmetric)
    drude_asym = Drude_asym(x_o=3, gamma_o=0.3) + Drude_asym(
        x_o=3.4, gamma_o=0.15, fixed={"x_o": True}
    )

    # 2 asymmetric Lorentzians
    Lorentzian_asym = custom_model(lorentz_asymmetric)
    lorentz_asym = Lorentzian_asym(x_o=3, gamma_o=0.3) + Lorentzian_asym(
        x_o=3.4, gamma_o=0.15, fixed={"x_o": True}
    )

    # 1 asymmetric Drude
    drude_asym1 = Drude_asym(x_o=3, gamma_o=0.3)

    profiles = [
        gauss,
        drude,
        lorentz,
        gauss_asym,
        drude_asym,
        lorentz_asym,
        drude_asym1,
    ]

    # fit the different profiles
    fit2 = LevMarLSQFitter()
    results = []
    mask1 = (waves > 2.4) & (waves < 3.6)
    mask2 = mask1 * (npts > 0)

    for profile in profiles:
        fit_result = fit2(
            profile,
            waves[mask2],
            fluxes[mask2],
            weights=1 / flux_unc[mask2],
            maxiter=10000,
        )
        results.append(fit_result)
        print(fit_result)
        print(
            "Chi2",
            np.sum(((fluxes[mask2] - fit_result(waves[mask2])) / flux_unc[mask2]) ** 2),
        )

    return waves[mask1], fluxes[mask1], npts[mask1], results


def fit_features_ext(starpair, path):
    """
    Fit the extinction features separately with different profiles

    Parameters
    ----------
    starpair : string
        Name of the star pair for which to fit the extinction features, in the format "reddenedstarname_comparisonstarname" (no spaces)

    path : string
        Path to the data files

    Returns
    -------
    waves : np.ndarray
        Numpy array with wavelengths

    exts_sub : np.ndarray
        Numpy array with continuum subtracted extinctions

    results : list
        List with the fitted models for different profiles
    """
    # first, fit the continuum, excluding the region of the features
    fit_spex_ext(starpair, path, exclude=(2.8, 3.6))

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
    mask = (waves >= 2.8) & (waves <= 3.6)
    waves = waves[mask]
    exts_sub = exts_sub[mask]
    exts_unc = exts_unc[mask]

    # define different profiles
    # 2 Gaussians (stddev=FWHM/(2sqrt(2ln2)))
    gauss = Gaussian1D(mean=3, stddev=0.13) + Gaussian1D(mean=3.4, stddev=0.06)

    # 2 Drudes
    drude = Drude1D(x_0=3, fwhm=0.3) + Drude1D(x_0=3.4, fwhm=0.15)

    # 2 Lorentzians
    lorentz = Lorentz1D(x_0=3, fwhm=0.3) + Lorentz1D(x_0=3.4, fwhm=0.15)

    # 2 asymmetric Gaussians
    Gaussian_asym = custom_model(gauss_asymmetric)
    gauss_asym = Gaussian_asym(x_o=3, gamma_o=0.3) + Gaussian_asym(
        x_o=3.4, gamma_o=0.15
    )

    # 2 "asymmetric" Drudes
    Drude_asym = custom_model(drude_asymmetric)
    drude_asym = Drude_asym(x_o=3, gamma_o=0.3) + Drude_asym(x_o=3.4, gamma_o=0.15)

    # 2 asymmetric Lorentzians
    Lorentzian_asym = custom_model(lorentz_asymmetric)
    lorentz_asym = Lorentzian_asym(x_o=3, gamma_o=0.3) + Lorentzian_asym(
        x_o=3.4, gamma_o=0.15
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


def fit_spex_ext(
    starpair,
    path,
    functype="pow",
    dense=False,
    profile="drude_asym",
    exclude=None,
    bootstrap=False,
    fixed=False,
):
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

    dense : boolean [default=False]
        Whether or not to fit the features around 3 and 3.4 micron

    profile : string [default="drude_asym"]
        Profile to use for the features if dense = True (options are "gauss", "drude", "lorentz", "gauss_asym", "drude_asym", "lorentz_asym")

    exclude : list of tuples [default=None]
        list of tuples (min,max) with wavelength regions (in micron) that need to be excluded from the fitting, e.g. [(0.8,1.2),(2.2,5)]

    bootstrap : boolean [default=False]
        Whether or not to do a quick bootstrap fitting to get more realistic uncertainties on the fitting results

    fixed : boolean [default=False]
        Whether or not to add a fixed feature around 3 micron (for diffuse sightlines)

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

    # exclude wavelength regions if requested
    if exclude:
        mask = np.full_like(waves, False, dtype=bool)
        for region in exclude:
            mask += (waves > region[0]) & (waves < region[1])
        waves = waves[~mask]
        exts = exts[~mask]
        exts_unc = exts_unc[~mask]

    # get a quick estimate of A(V)
    if extdata.type == "elx":
        extdata.calc_AV()
        AV_guess = extdata.columns["AV"]
    else:
        AV_guess = None

    # convert to A(lambda)/A(1 micron)
    # ind1 = np.abs(waves - 1).argmin()
    # exts = exts / exts[ind1]
    # exts_unc = exts_unc / exts[ind1]

    # obtain the function to fit
    if "SpeX_LXD" not in extdata.waves.keys():
        dense = False
        fixed = False
    func = fit_function(
        dattype=extdata.type,
        functype=functype,
        dense=dense,
        profile=profile,
        AV_guess=AV_guess,
        fixed=fixed,
    )

    # for dense sightlines, add more weight to the feature region
    weights = 1 / exts_unc
    if dense:
        mask_ice = (waves > 2.88) & (waves < 3.19)
        mask_tail = (waves > 3.4) & (waves < 4)
        weights[mask_ice + mask_tail] *= 2

    # use the Levenberg-Marquardt algorithm to fit the data with the model
    fit = LevMarLSQFitter()
    fit_result_lev = fit(func, waves, exts, weights=weights, maxiter=10000)

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

    fit_result_mcmc = fit2(fit_result_lev, waves, exts, weights=weights)

    # create standard MCMC plots
    fit2.plot_emcee_results(
        fit_result_mcmc, filebase=path + "Fitting_results/" + starpair
    )

    # choose the fit result to save
    fit_result = fit_result_mcmc
    # fit_result = fit_result_lev
    print(fit_result)

    # determine the wavelengths at which to evaluate and save the fitted model curve: all SpeX wavelengths, sorted from short to long (to avoid problems with overlap between SXD and LXD), and shortest and longest wavelength should have data
    if "SpeX_LXD" not in extdata.waves.keys():
        full_waves = extdata.waves["SpeX_SXD"].value
        full_npts = extdata.npts["SpeX_SXD"]
    else:
        full_waves = np.concatenate(
            (extdata.waves["SpeX_SXD"].value, extdata.waves["SpeX_LXD"].value)
        )
        full_npts = np.concatenate((extdata.npts["SpeX_SXD"], extdata.npts["SpeX_LXD"]))
    # sort the wavelengths
    indxs_sort = np.argsort(full_waves)
    full_waves = full_waves[indxs_sort]
    full_npts = full_npts[indxs_sort]
    # cut the wavelength region
    indxs = np.logical_and(full_waves >= np.min(waves), full_waves <= np.max(waves))
    full_waves = full_waves[indxs]
    full_npts = full_npts[indxs]

    # calculate the residuals and put them in an array of the same length as "full_waves" for plotting
    residuals = exts - fit_result(waves)
    full_res = np.full_like(full_npts, np.nan)
    if exclude:
        mask = np.full_like(full_waves, False, dtype=bool)
        for region in exclude:
            mask += (full_waves > region[0]) & (full_waves < region[1])
        full_res[(full_npts > 0) * ~mask] = residuals

    else:
        full_res[(full_npts > 0)] = residuals

    # bootstrap to get more realistic uncertainties on the parameter results
    if bootstrap:
        red_star = StarData(extdata.red_file, path=path, use_corfac=True)
        comp_star = StarData(extdata.comp_file, path=path, use_corfac=True)
        red_V_unc = red_star.data["BAND"].get_band_mag("V")[1]
        comp_V_unc = comp_star.data["BAND"].get_band_mag("V")[1]
        unc_V = np.sqrt(red_V_unc ** 2 + comp_V_unc ** 2)
        fit_result_mcmc_low = fit2(fit_result_lev, waves, exts - unc_V, weights=weights)
        fit_result_mcmc_high = fit2(
            fit_result_lev, waves, exts + unc_V, weights=weights
        )

    # save the fitting results to the fits file
    if dense:
        functype += "_" + profile
    extdata.model["type"] = functype + "_" + extdata.type
    extdata.model["waves"] = full_waves
    extdata.model["exts"] = fit_result(full_waves)
    extdata.model["residuals"] = full_res
    extdata.model["chi2"] = np.sum((residuals / exts_unc) ** 2)
    print("Chi2", extdata.model["chi2"])
    extdata.model["params"] = []
    for param in fit_result.param_names:
        # update the uncertainties when bootstrapping
        if bootstrap:
            min_val = min(
                getattr(fit_result_mcmc, param).value,
                getattr(fit_result_mcmc_low, param).value,
                getattr(fit_result_mcmc_high, param).value,
            )
            max_val = max(
                getattr(fit_result_mcmc, param).value,
                getattr(fit_result_mcmc_low, param).value,
                getattr(fit_result_mcmc_high, param).value,
            )
            sys_unc = (max_val - min_val) / 2
            getattr(fit_result, param).unc_minus = np.sqrt(
                getattr(fit_result, param).unc_minus ** 2 + sys_unc ** 2
            )
            getattr(fit_result, param).unc_plus = np.sqrt(
                getattr(fit_result, param).unc_plus ** 2 + sys_unc ** 2
            )

        extdata.model["params"].append(getattr(fit_result, param))

        # save the column information (A(V), E(B-V) and R(V))
        if "Av" in param:
            extdata.columns["AV"] = (
                getattr(fit_result, param).value,
                getattr(fit_result, param).unc_minus,
                getattr(fit_result, param).unc_plus,
            )
            # calculate the distrubtion of R(V) from the distributions of A(V) and E(B-V)
            av_dist = getattr(fit_result, param).posterior
            b_indx = np.abs(extdata.waves["BAND"] - 0.438 * u.micron).argmin()
            ebv_dist = unc.normal(
                extdata.exts["BAND"][b_indx],
                std=extdata.uncs["BAND"][b_indx],
                n_samples=av_dist.n_samples,
            )
            ebv_per = ebv_dist.pdf_percentiles([16.0, 50.0, 84.0])
            extdata.columns["EBV"] = (
                ebv_per[1],
                ebv_per[1] - ebv_per[0],
                ebv_per[2] - ebv_per[1],
            )
            rv_dist = av_dist / ebv_dist
            rv_per = rv_dist.pdf_percentiles([16.0, 50.0, 84.0])
            extdata.columns["RV"] = (
                rv_per[1],
                rv_per[1] - rv_per[0],
                rv_per[2] - rv_per[1],
            )
            print(extdata.columns)

    if fixed:
        print(
            "Ice feature strength: ",
            extdata.model["params"][3].value,
            extdata.model["params"][3].unc_minus,
            extdata.model["params"][3].unc_plus,
        )
        extdata.save("%s%s_ext_ice.fits" % (path, starpair.lower()))
    else:
        extdata.save("%s%s_ext.fits" % (path, starpair.lower()))


if __name__ == "__main__":
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    fit_spex_ext("HD283809_HD003360", path)
