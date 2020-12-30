#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling.powerlaws import PowerLaw1D
from astropy.modeling.polynomial import Polynomial1D
from astropy.modeling.models import Drude1D
from astropy.modeling.fitting import LevMarLSQFitter

from measure_extinction.extdata import ExtData
from dust_extinction.conversions import AxAvToExv


def fit_function(
    dattype="elx",
    functype="pow",
    ice=False,
    amp_bounds=(-1.5, 1.5),
    index_bounds=(0.0, 5.0),
    AV_bounds=(0.0, 6.0),
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

    amp_bounds : tuple [default=(-1.5,1.5)]
        Powerlaw amplitude bounds to be used in the fitting

    index_bounds : tuple [default=(0.0,5.0)]
        Powerlaw index bounds to be used in the fitting

    AV_bounds : tuple [default=(0.0,6.0)]
        A(V) bounds to be used in the fitting

    Returns
    -------
    func : Astropy CompoundModel
        The fitting function
    """
    # powerlaw model
    if functype == "pow":
        func = PowerLaw1D(
            fixed={"x_0": True},
            bounds={"amplitude": amp_bounds, "alpha": index_bounds},
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
        func += Drude1D(
            fwhm=0.7,
            fixed={"fwhm": True},
            bounds={
                "x_0": (3.04, 3.06),
            },
        )

    # convert the function from A(lambda)/A(V) to E(lambda-V)
    if dattype == "elx":
        func = func | AxAvToExv(bounds={"Av": AV_bounds})

    return func


def fit_spex_ext(
    starpair,
    path,
    functype="pow",
    ice=False,
    amp_bounds=(-1.5, 1.5),
    index_bounds=(0.0, 5.0),
    AV_bounds=(0.0, 6.0),
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

    amp_bounds : tuple [default=(-1.5,1.5)]
        Powerlaw amplitude bounds to be used in the fitting

    index_bounds : tuple [default=(0.0,5.0)]
        Powerlaw index bounds to be used in the fitting

    AV_bounds : tuple [default=(0.0,6.0)]
        A(V) bounds to be used in the fitting

    Returns
    -------
    Updates extdata.model["type", "waves", "exts", "residuals", "params"] and extdata.columns["AV"] with the fitting results:
        - type: string with the type of model ("pow_elx" or "pow_alav")
        - waves: np.ndarray with the SpeX wavelengths
        - exts: np.ndarray with the fitted model to the extinction curve at "waves" wavelengths
        - residuals: np.ndarray with the residuals, i.e. data-fit, at "waves" wavelengths
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
    func = fit_function(dattype=extdata.type, functype=functype, ice=ice)

    # fit the data with the model
    fit = LevMarLSQFitter()
    fit_result = fit(func, waves, exts, weights=1 / exts_unc)
    print(fit_result)
    # determine the wavelengths at which to evaluate and save the fitted model curve: all SpeX wavelengths, sorted from short to long (to avoid problems with overlap between SXD and LXD), and shortest and longest wavelength should have data
    full_waves = np.concatenate(
        (extdata.waves["SpeX_SXD"].value, extdata.waves["SpeX_LXD"].value)
    )
    indxs = np.argsort(full_waves)[
        np.logical_and(full_waves >= np.min(waves), full_waves <= np.max(waves))
    ]
    full_waves = full_waves[indxs]

    # calculate the residuals and put them in an array of the same length as "full_waves" for plotting
    residuals = exts - fit_result(waves)
    full_npts = np.concatenate((extdata.npts["SpeX_SXD"], extdata.npts["SpeX_LXD"]))[
        indxs
    ]
    full_res = np.full_like(full_npts, np.nan)
    full_res[full_npts > 0] = residuals

    # save the fitting results
    extdata.model["type"] = functype + "_" + extdata.type
    extdata.model["waves"] = full_waves
    extdata.model["exts"] = fit_result(full_waves)
    extdata.model["residuals"] = full_res
    extdata.model["params"] = []
    for param in fit_result.param_names:
        extdata.model["params"].append(getattr(fit_result, param))
        if "Av" in param:
            extdata.columns["AV"] = getattr(fit_result, param)
    extdata.save("%s%s_ext.fits" % (path, starpair.lower()))


if __name__ == "__main__":
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    fit_spex_ext("HD283809_HD003360", path)
