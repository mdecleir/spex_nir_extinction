# This script is intended to automate the different steps to create SpeX spectra for all stars in the sample:
# 	- "merging" and rebinning the SpeX spectra
# 	- scaling the SpeX spectra
# 	- plotting the spectra

import argparse
import pkg_resources
import glob
import os

from measure_extinction.utils.merge_spex_spec import merge_spex
from measure_extinction.utils.scale_spex_spec import calc_save_corfac_spex
from measure_extinction.plotting.plot_spec import plot_multi_spectra, plot_spectrum


if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inpath",
        help="path where original SpeX ASCII files are stored",
        default=pkg_resources.resource_filename("measure_extinction", "data/Orig/NIR"),
    )
    parser.add_argument(
        "--spex_path",
        help="path where merged SpeX spectra will be stored",
        default=pkg_resources.resource_filename("measure_extinction", "data/Spectra"),
    )
    parser.add_argument("--mlam4", help="plot lambda^4*F(lambda)", action="store_true")
    parser.add_argument("--HI_lines", help="indicate the HI-lines", action="store_true")
    parser.add_argument(
        "--onefig",
        help="whether or not to plot all spectra in the same figure",
        action="store_true",
    )
    parser.add_argument(
        "--range",
        nargs="+",
        help="wavelength range to be plotted (in micron)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--norm_range",
        nargs="+",
        help="wavelength range to use to normalize the spectrum (in micron)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--spread",
        help="spread the spectra out over the figure",
        action="store_true",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="data type(s) to be excluded from the plot",
        type=str,
        default=[],
    )

    args = parser.parse_args()

    # collect the star names in the input directory
    stars = []
    for filename in glob.glob(args.inpath + "/*.txt"):
        starname = os.path.basename(filename).split("_")[0]
        if starname not in stars:
            stars.append(starname)

    # do the different steps for all the stars
    stars = ["HD283809"]
    for star in stars:
        print(star.upper())
        merge_spex(star, args.inpath, args.spex_path, outname=None)
        calc_save_corfac_spex(
            star, os.path.dirname(os.path.normpath(args.spex_path)) + "/"
        )

    if args.onefig:  # plot all spectra in the same figure
        plot_multi_spectra(
            stars,
            os.path.dirname(os.path.normpath(args.spex_path)) + "/",
            args.mlam4,
            args.HI_lines,
            args.range,
            args.norm_range,
            args.spread,
            args.exclude,
            pdf=True,
        )
    else:  # plot all spectra separately
        if args.spread:
            parser.error(
                "The flag --spread can only be used in combination with the flag --onefig. It only makes sense to spread out the spectra if there is more than one spectrum in the same plot."
            )
        for star in stars:
            plot_spectrum(
                star,
                os.path.dirname(os.path.normpath(args.spex_path)) + "/",
                args.mlam4,
                args.HI_lines,
                args.range,
                args.norm_range,
                args.exclude,
                pdf=True,
            )
