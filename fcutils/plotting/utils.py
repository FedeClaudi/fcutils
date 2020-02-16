import os
import numpy as np
import seaborn as sns

from matplotlib import cm
from matplotlib import gridspec
from collections import namedtuple

import matplotlib.pyplot as plt


def set_figure_subplots_aspect(
    left=0.125, right=0.9, bottom=0.06, top=0.96, wspace=0.2, hspace=0.3
):
    plt.subplots_adjust(
        left=left,  # the left side of the subplots of the figure
        right=right,  # the right side of the subplots of the figure
        bottom=bottom,  # the bottom of the subplots of the figure
        top=top,  # the top of the subplots of the figure
        wspace=wspace,  # the amount of width reserved for blank space between subplots
        hspace=hspace,  # the amount of height reserved for white space between subplots
    )


def clean_axes(f):
    sns.despine(fig=f, offset=10, trim=False, left=False, right=True)


def save_all_open_figs(
    target_fld=False, name=False, format=False, exclude_number=False
):
    open_figs = plt.get_fignums()

    for fnum in open_figs:
        if name:
            if not exclude_number:
                ttl = "{}_{}".format(name, fnum)
            else:
                ttl = str(name)
        else:
            ttl = str(fnum)

        if target_fld:
            ttl = os.path.join(target_fld, ttl)
        if not format:
            ttl = "{}.{}".format(ttl, "svg")
        else:
            ttl = "{}.{}".format(ttl, format)

        plt.figure(fnum)
        plt.savefig(ttl)


def create_triplot(**kwargs):
    # Creates a figure with one main plot and two plots on the sides
    fig = plt.figure(**kwargs)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 0])
    plt.tight_layout()

    axes = namedtuple("axes", "main x y")
    return fig, axes(ax0, ax2, ax1)


def create_figure(subplots=True, **kwargs):
    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = (12, 8)
    if not subplots:
        f, ax = plt.subplots(**kwargs)
    else:
        f, ax = plt.subplots(**kwargs)
        ax = ax.flatten()
    return f, ax


def show():
    plt.show()


def ticksrange(start, stop, step):
    return np.arange(start, stop + step, step)


def save_figure(f, path, svg=False, verbose=True):
    """
		Paths should be the complete path to where the figure should be saved but without suffix
	"""
    if svg:
        f.savefig("{}.svg".format(path))

    f.savefig("{}.png".format(path))

    if verbose:
        print(" saved figure at: " + path)

def make_legend(ax):
    l = ax.legend()
    for text in l.get_texts():
        text.set_color([0.7, 0.7, 0.7])
