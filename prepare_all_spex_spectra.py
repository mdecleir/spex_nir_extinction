# This script is intended to automate the different steps to create SpeX spectra of all stars in the sample:
# 	- "merging" and rebinning the SpeX spectra
# 	- scaling the SpeX spectra
# 	- plotting the SpeX spectra

import glob
import os

from measure_extinction.utils.merge_spex_spec import merge_spex
from measure_extinction.utils.scale_spex_spec import calc_save_corfac_spex
from measure_extinction.plotting.plot_spec import plot_spectrum
from measure_extinction.stardata import StarData


def merge_scale_plot(star):
    print("Merging, scaling and plotting SpeX spectra for " + star.upper())
    merge_spex(star, inpath, spex_path)
    calc_save_corfac_spex(star, os.path.dirname(os.path.normpath(spex_path)) + "/")

    plot_spectrum(
        star,
        os.path.dirname(os.path.normpath(spex_path)) + "/",
        mlam4=True,
        range=[0.78, 5.55],
        norm_range=[1, 1.1],
        exclude=["IRS", "STIS_Opt"],
        pdf=True,
    )


def print_BV(star):
    path = "/Users/mdecleir/Documents/NIR_ext/Data/"
    star_data = StarData("%s.dat" % star.lower(), path=path, use_corfac=False)
    V = star_data.data["BAND"].get_band_mag("V")
    B = star_data.data["BAND"].get_band_mag("B")
    print(star, ", B:", B, ", V:", V, ", B-V:", B[0] - V[0])


if __name__ == "__main__":
    # collect the star names in the input directory
    inpath = "/Users/mdecleir/Documents/NIR_ext/Data/SpeX_Data/Reduced_Spectra"
    stars = []
    for filename in glob.glob(inpath + "/*.txt"):
        starname = os.path.basename(filename).split("_")[0]
        if starname not in stars:
            stars.append(starname)

    # do the different steps for all the stars
    spex_path = "/Users/mdecleir/Documents/NIR_ext/Data/SpeX_Data/"
    for star in stars:
        merge_scale_plot(star)
        print_BV(star)
