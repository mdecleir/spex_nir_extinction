SpeX NIR Extinction Paper
=========================

This repository contains code to measure NIR extinction curves for a sample of stars, based on their SpeX spectra. The scripts in the plotting folder produce the figures in the SpeX NIR dust extinction paper by Decleir et al. (submitted to ApJ).

This work is still in development. Use at your own risk.


Contributors
------------

Marjorie Decleir


License
-------

This project is Copyright (c) Marjorie Decleir and licensed under
the terms of the BSD 3-Clause license (see the ``LICENSE`` file for more information).


Dependencies
------------

This project uses other repositories:

* `dust_extinction <https://github.com/karllark/dust_extinction>`_ provides a conversion function (AxAvToExv), average extinction curves and grain models from the literature.
* `measure_extinction <https://github.com/karllark/measure_extinction>`_ contains the routines to read the data and calculate extinction curves.


Use
---

* To create, scale and plot SpeX spectra of all stars in the sample, run: ::

    python prepare_all_spex_spectra.py
* To plot all spectra (Figs. 2-4), run: ::

    python plotting/plot_spex_spec.py
* To calculate, fit and plot extinction curves for all the reddened stars in the sample, run: ::

    python calc_fit_plot_ext.py

* To plot all extinction curves (Figs. 5, 6, 9, 10, 11), run: ::

    python plotting/plot_spex_ext.py

* To create tables and plot results (Figs. 7-8, Tables 2-3), run: ::

    python plotting/plot_results.py

* To measure, fit and plot the R(V) dependence (Figs. 12-14, Table 4), run: ::

    python RV_dependence.py
