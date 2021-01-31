from pathlib import Path
from collections import namedtuple
import math
import seaborn as sns
from matplotlib import gridspec
import matplotlib.pyplot as plt

# ---------------------------- generating figures ---------------------------- #


def create_triplot(**kwargs):
    """
        Creates a figure with one main plot and two plots on the sides
    """
    fig = plt.figure(**kwargs)
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 0])
    plt.tight_layout()

    axes = namedtuple("axes", "main x y")
    return fig, axes(ax0, ax2, ax1)


def calc_nrows_ncols(N, aspect=(16, 9)):
    """
        Computs the number of rows and columns to fit
        a given number N of subplots in a figure with
        aspect `aspect`.
        from: https://stackoverflow.com/questions/36482328/how-to-use-a-python-produce-as-many-subplots-of-arbitrary-size-as-necessary-acco
    """
    width = aspect[0]
    height = aspect[1]
    area = width * height * 1.0
    factor = (N / area) ** (1 / 2.0)
    cols = math.floor(width * factor)
    rows = math.floor(height * factor)
    rowFirst = width < height
    while rows * cols < N:
        if rowFirst:
            rows += 1
        else:
            cols += 1
        rowFirst = not (rowFirst)
    return rows, cols


# ---------------------------------- editing --------------------------------- #


def clean_axes(f):
    """
        Makes axes look pretty

        Arguments:
            f: matplotlib figure
    """
    ax_list = f.axes

    for ax in list(ax_list):
        try:
            sns.despine(ax=ax, offset=10, trim=False, left=False, right=True)
        except Exception:
            pass


def set_figure_subplots_aspect(
    left=0.125, right=0.9, bottom=0.06, top=0.96, wspace=0.2, hspace=0.3
):
    """
        Adjust the aspect ratio of subplots in a figure

        Arguments:
            left: the left side of the subplots of the figure
            right: the right side of the subplots of the figure
            bottom: the bottom of the subplots of the figure
            top: the top of the subplots of the figure
            wspace: the amount of width reserved for blank space between subplots
            hspace: the amount of height reserved for white space between subplots        
    """
    plt.subplots_adjust(
        left=left,  # the left side of the subplots of the figure
        right=right,  # the right side of the subplots of the figure
        bottom=bottom,  # the bottom of the subplots of the figure
        top=top,  # the top of the subplots of the figure
        wspace=wspace,  # the amount of width reserved for blank space between subplots
        hspace=hspace,  # the amount of height reserved for white space between subplots
    )


# ---------------------------------- saving ---------------------------------- #


def save_figure(f, path, svg=False, close=False):
    """
        Saves figure to file

        Arguments:
            f: matplotlib figure
            path: str, Path. Where figure will be saved
            svg: bool If true a copy of the figure will be saved as svg
            close: bool. If true the figure is closed after saving
    """
    path = Path(path).with_suffix(".png")
    f.savefig(str(path))

    if svg:
        f.savefig(str(path.with_suffix(".svg")))

    if close:
        plt.close(f)
