import os
import numpy as np
import seaborn as sns

from matplotlib import cm
from matplotlib import gridspec
from collections import namedtuple

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_axes_at_center(ax):
    """
        from https://stackoverflow.com/questions/31556446/how-to-draw-axis-in-the-middle-of-the-figure
    """
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')



def add_colorbar_to_img(img, ax, f, pos='right', orientation='vertical'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(pos, size='5%', pad=0.05)
    f.colorbar(img, cax=cax, orientation=orientation)


def forceAspect(ax,aspect):
    """
        Forces the aspect ratio of an axis onto which an image what plotted with imshow.
        From:
        https://www.science-emergence.com/Articles/How-to-change-imshow-aspect-ratio-in-matplotlib-/
    """
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def fig2data (fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll (buf, 3, axis=2)
    return buf

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
    ax_list = f.axes

    for ax in list(ax_list):
        try:
            sns.despine(ax=ax, offset=10, trim=False, left=False, right=True)
        except:
            pass


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


def save_figure(f, path, svg=False, verbose=True, close=False):
    """
		Paths should be the complete path to where the figure should be saved but without suffix
	"""
    if svg:
        f.savefig("{}.svg".format(path))

    f.savefig("{}.png".format(path))

    if verbose:
        print(" saved figure at: " + path)
    
    if close:
        plt.close(f)

def make_legend(ax):
    l = ax.legend()
    for text in l.get_texts():
        text.set_color([0.7, 0.7, 0.7])
