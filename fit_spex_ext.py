#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.models import Drude1D, custom_model, Gaussian1D
from astropy.modeling.fitting import LevMarLSQFitter

from measure_extinction.extdata import ExtData
from dust_extinction.conversions import AxAvToExv


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
        func = PowerLaw1D(
            fixed={"x_0": True},
        )
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

        def drude_modified(x, scale=1, x_o=1, gamma_o=1, asym=1):
            gamma = 2.0 * gamma_o / (1.0 + np.exp(asym * (x - x_o)))
            y = (
                scale
                * ((gamma / x_o) ** 2)
                / ((x / x_o - x_o / x) ** 2 + (gamma / x_o) ** 2)
            )
            return y

        Drude_modified_model = custom_model(drude_modified)
        # func += Drude_modified_model(x_o=3.05)
        func += Drude_modified_model(
            x_o=2.9461043067594406,
            gamma_o=0.4812388755514129,
            asym=-35.74637360250102,
            fixed={"x_o": True, "gamma_o": True, "asym": True},
        )

    # convert the function from A(lambda)/A(V) to E(lambda-V)
    if dattype == "elx":
        func = func | AxAvToExv()

    return func


def fit_spex_ext(
    starpair,
    path,
    functype="pow",
    ice=False,
):
    """
    Fit the observed SpeX NIR extinction curve

    Parameters
    ----------
    starpair : string
        Name of the star pair for which to fit the extinction curve, in the format "reddenedstarname_comparisonstarname" (no spaces)

    path : string
        Path to the data files

    functype : string [default="pow"]
        Fitting function type ("pow" for powerlaw or "pol" for polynomial)

    ice : boolean [default=False]
        Whether or not to fit the ice feature at 3.05 micron


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

    # obtain the function to fit
    if "SpeX_LXD" not in extdata.waves.keys():
        ice = False
    func = fit_function(dattype=extdata.type, functype=functype, ice=ice)

    # fit the data with the model
    fit = LevMarLSQFitter()
    fit_result = fit(func, waves, exts, weights=1 / exts_unc, maxiter=1000)
    print(fit_result)
    print(fit.fit_info["nfev"])

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
    full_res[full_npts > 0] = residuals

    # save the fitting results
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
            extdata.columns["AV"] = getattr(fit_result, param).value
            extdata.calc_RV()
            print("RV", extdata.columns["RV"])
    extdata.save("%s%s_ext.fits" % (path, starpair.lower()))


if __name__ == "__main__":
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    fit_spex_ext("HD283809_HD003360", path)
