# -*- coding: utf-8 -*-

import logging
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from matplotlib.ticker import ScalarFormatter
from scipy.stats import norm
import matplotlib
from input import *
import itertools

try:
    from scipy.ndimage import gaussian_filter
except ImportError:
    gaussian_filter = None

__all__ = ["corner", "hist2d", "quantile"]

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'


def spec(xs, xfull, xbin, yfull, ybin, ydata, x_err, y_err, range2, color_list, bins=20, range=None, weights=None, color="k",
           smooth=None, smooth1d=None,
           labels=None, label_kwargs=None,
           show_titles=False, title_fmt=".2f", title_kwargs=None,
           truths=None, truth_color="k",
           #ini_guess = [None]*(len(parameters)-2) + [(rstar,rstar_uncertainty), (g,g_uncertainty)],
           density=True,
           scale_hist=False, quantiles=None, verbose=False, fig=None,
           max_n_ticks=5, top_ticks=False, use_math_text=False, reverse=False,
           hist_kwargs=None, **hist2d_kwargs):
           

    pl.rc('text', usetex=True)
    # pl.rcParams['text.latex.preamble'] = [r'\boldmath']


    # Try filling in labels from pandas.DataFrame columns.
    if labels is None:
        try:
            labels = xs.columns
        except AttributeError:
            pass

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"


    # Some magic numbers for pretty axis layout.
    K = len(xs)
#    factor = 5.0           # size of one side of one panel
#    if reverse:
#        lbdim = 0.2 * factor   # size of left/bottom margin
#        trdim = 0.5 * factor   # size of top/right margin
#    else:
#        lbdim = 0.5 * factor   # size of left/bottom margin
#        trdim = 0.2 * factor   # size of top/right margin
#    whspace = 0.05         # w/hspace size
#    plotdim = factor * K + factor * (K - 1.) * whspace
#    dim = lbdim + plotdim + trdim


    # Create a new figure if one wasn't provided.
    if fig is None:
        fig2, ax = pl.subplots(figsize=(20, 20))   # fig, axes = pl.subplots(K, K, figsize=(dim, dim))
    else:
        try:
            axes = np.array(fig2.axes).reshape((K, K))
        except:
            raise ValueError("Provided figure has {0} axes, but data has "
                             "dimensions K={1}".format(len(fig2.axes), K))

    # Format the figure.
#    lb = lbdim / dim
#    tr = (lbdim + plotdim) / dim
#    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
#                        wspace=whspace, hspace=whspace)


    ax.set_frame_on(True)
    linethick = 1   # linethick = 0.5
    try:
        t_retrieved        
        line1 = ax.plot(xfull, yfull, linewidth=linethick, color='g', linestyle='-')
    except:
        line1 = ax.plot(xfull, yfull, linewidth=linethick, color='b', linestyle='-')
    symsize2 = 1
    mew1 = 6   # mew1 = 5*K/3
    msize = 6   # msize = 2.5*K/3
    elw = 3   # elw = 1.5*K/3
    ax.plot(xbin, ybin, 'ks', mew=mew1, markersize=msize)
    ax.errorbar(xbin, ybin, xerr=x_err, yerr=y_err, fmt='ks', elinewidth=elw)
    symsize = 4
    ax.plot(xbin, ydata, 'ro', mew=mew1, markersize=msize)
    ax.errorbar(xbin, ydata, xerr=x_err, yerr=y_err, fmt='ro', capthick=2, elinewidth=elw)
    text_size = 32   # text_size = 4*K +4
    wavelength_min = np.amin(wavelength_bins)
    wavelength_max = np.amax(wavelength_bins)
    transit_min = np.amin(transit_depth)
    transit_max = np.amax(transit_depth)
    ax.text(0.85*wavelength_min+0.1*wavelength_max, 2.6*transit_max - 1.6*transit_min, 'Circles: '+planet_name+' data', color='r', fontsize=text_size)   # ax.text(0.85*wavelength_min+0.1*wavelength_max, 2.5*transit_max - 1.5*transit_min, 'Circles: '+planet_name+' data', color='r', fontsize=text_size)
    ax.text(0.85*wavelength_min+0.1*wavelength_max, 2.75*transit_max-1.75*transit_min, 'Squares: Model (binned)', color='k', fontsize=text_size)
    ax.set_xlim([wavelength_min-0.03, wavelength_max+0.03])
    ax.set_ylim([1.5*transit_min-0.5*transit_max, 3*transit_max-2*transit_min])
    ax.xaxis.set_major_locator(MaxNLocator(5, prune="lower"))
    ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=use_math_text))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=use_math_text))
    tick_size = 36   # tick_size = 4*K + 10
    ax.tick_params(axis='both', which='major', labelsize=tick_size, pad=10)   # ax.tick_params(axis='both', which='major', labelsize=tick_size)
    label_size = 40   # label_size = 10*K/3 + 16
    ax.set_xlabel(r'\textbf{wavelength (}\boldmath $\mu$\textbf{m)}', fontsize=label_size, fontweight='bold', labelpad=20)   # ax.set_xlabel(r'\textbf{wavelength (}\boldmath $\mu$\textbf{m)}', fontsize=label_size, fontweight='bold')
    ax.set_ylabel(r'\boldmath $(R/R_\star)^2$ \textbf{(\%)}', fontsize=label_size, fontweight='bold', labelpad=30)   # ax.set_ylabel(r'\boldmath $(R/R_\star)^2$ \textbf{(\%)}', fontsize=label_size, fontweight='bold')
    #ax.xaxis.set_label_coords(0.5, -0.08)
    y_label_x = -0.1   # y_label_x = -0.25 + 0.06*K/3
    #ax.yaxis.set_label_coords(y_label_x, 0.5)

    return fig2
