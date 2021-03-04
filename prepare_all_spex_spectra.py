# This script is intended to automate the different steps to create SpeX spectra of all stars in the sample:
# 	- "merging" and rebinning the SpeX spectra
# 	- scaling the SpeX spectra
# 	- plotting the SpeX spectra

import glob
import os

from measure_extinction.utils.merge_spex_spec import merge_spex
from measure_extinction.utils.scale_spex_spec import calc_save_corfac_spex
from measure_extinction.plotting.plot_spec import plot_multi_spectra, plot_spectrum


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

    # if args.onefig:  # plot all spectra in the same figure
    #     plot_multi_spectra(
    #         stars,
    #         os.path.dirname(os.path.normpath(args.spex_path)) + "/",
    #         args.mlam4,
    #         args.HI_lines,
    #         args.range,
    #         args.norm_range,
    #         args.spread,
    #         args.exclude,
    #         pdf=True,
    #     )
    # else:  # plot all spectra separately
    #     if args.spread:
    #         parser.error(
    #             "The flag --spread can only be used in combination with the flag --onefig. It only makes sense to spread out the spectra if there is more than one spectrum in the same plot."    #
