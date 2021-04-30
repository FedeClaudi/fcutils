import numpy as np


from fcutils.maths.array import find_nearest, percentile_range


def plot_line_outlined(
    ax, x, y=None, lw=2, outline=1, outline_color="k", color="r", **kwargs
):
    if y is not None:
        ax.plot(x, y, lw=lw + outline, color=outline_color, zorder=-1)
        ax.plot(x, y, lw=lw, color=color, **kwargs)
    else:
        ax.plot(x, lw=lw + outline, color=outline_color, zorder=-1)
        ax.plot(x, lw=lw, color=color, **kwargs)


def plot_mean_and_error(y, yerr, ax, err_alpha=0.3, color="k", **kwargs):
    alpha = kwargs.pop("alpha", 1)
    lw = kwargs.pop("lw", 3)
    err_color = kwargs.pop("err_color", color)
    zorder = kwargs.pop("zorder", 90)
    x = kwargs.pop("x", np.arange(len(y)))

    ax.fill_between(
        x,
        y - yerr,
        y + yerr,
        alpha=err_alpha,
        zorder=zorder - 1,
        color=err_color,
    )
    ax.plot(x, y, alpha=alpha, lw=lw, zorder=zorder, color=color, **kwargs)


def ball_and_errorbar(
    x,
    y,
    ax,
    data=None,
    prange=None,
    orientation="horizontal",
    color="k",
    s=100,
    **kwargs,
):
    """
        Plots a line and a ball on top of it, used to plot
        stuff like mean and CI.
    """
    if prange is None:
        prange = percentile_range(data)

    if orientation == "horizontal":
        ax.plot([prange.low, prange.high], [y, y], color=color, **kwargs)
    else:
        ax.plot([x, x], [prange.low, prange.high], color=color, **kwargs)

    ax.scatter(x, y, color=color, s=s, zorder=99)


def ortholines(
    ax,
    orientations,
    values,
    color=[0.7, 0.7, 0.7],
    lw=3,
    alpha=0.5,
    ls="--",
    **kwargs,
):
    """[makes a set of vertical and horizzontal lines]
    
    Arguments:
        ax {[np.axarr]} -- [ax]
        orientations {[int]} -- [list of 0 and 1 with the orientation of each line. 0 = horizzontal and 1 = vertical]
        values {[float]} -- [list of x or y values at which the lines should be drawn. Should be the same length as orientations]

    """
    if not isinstance(orientations, list):
        orientations = [orientations]
    if not isinstance(values, (list, np.ndarray)):
        values = [values]

    for o, v in zip(orientations, values):
        if o == 0:
            func = ax.axhline
        else:
            func = ax.axvline

        func(v, color=color, lw=lw, alpha=alpha, ls=ls, **kwargs)


def vline_to_curve(
    ax, x, xdata, ydata, dot=False, line_kwargs={}, scatter_kwargs={}, **kwargs
):
    """[plots a vertical line from the x axis to the curve at location x]
    
    Arguments:
        ax {[axarray]} -- [ax to plot on]
        x {[float]} -- [x value to plot on ]
        curve {[np.array]} -- [array of data with the curve. The vertical line will go from 0 to curve[x]]
    """
    line = ax.plot(xdata, ydata, alpha=0)
    xline, yline = line[0].get_data()
    x = find_nearest(xline, x)
    yval = yline[np.where(xline == x)[0][0]]
    ax.plot([x, x], [0, yval], **line_kwargs)
    if dot:
        ax.scatter(x, yval, **scatter_kwargs, **kwargs)


def vline_to_point(ax, x, y, ymin=0, **kwargs):
    ax.plot([x, x], [ymin, y], **kwargs)


def hline_to_point(ax, x, y, xmin=0, **kwargs):
    ax.plot([xmin, x], [y, y], **kwargs)


def hline_to_curve(
    ax, y, xdata, ydata, dot=False, line_kwargs={}, scatter_kwargs={}, **kwargs
):
    """[plots a vertical line from the x axis to the curve at location x]
    
    Arguments:
        ax {[axarray]} -- [ax to plot on]
        x {[float]} -- [x value to plot on ]
        curve {[np.array]} -- [array of data with the curve. The vertical line will go from 0 to curve[x]]
    """
    line = ax.plot(xdata, ydata, alpha=0)
    xline, yline = line[0].get_data()
    y = find_nearest(yline, y)
    xval = xline[np.where(yline == y)[0][0]]
    ax.plot([0, xval], [y, y], **line_kwargs, **kwargs)
    if dot:
        ax.scatter(xval, y, **scatter_kwargs, **kwargs)


def plot_shaded_withline(
    ax, x, y, z=None, label=None, orientation="hor", alpha=0.15, **kwargs
):
    """[Plots a curve with shaded area and the line of the curve clearly visible]
    
    Arguments:
        ax {[type]} -- [matplotlib axis]
        x {[np.array, list]} -- [x data]
        y {[np.array, list]} -- [y data]
    
    Keyword Arguments:
        z {[type]} -- [description] (default: {None})
        label {[type]} -- [description] (default: {None})
        alpha {float} -- [description] (default: {.15})
    """

    if z == "min":  #
        z = np.nanmin(y)
    elif z == "max":
        z = np.nanmax(y)
    elif z == "mean":
        z = np.nanmean(y)

    if z is not None:
        if orientation == "hor":
            ax.fill_between(x, z, y, alpha=alpha, **kwargs)
        elif orientation == "ver":
            ax.fill_between(y, z, x, alpha=alpha, **kwargs)
        else:
            raise ValueError

    else:
        if orientation == "hor":
            ax.fill_between(x, y, alpha=alpha, **kwargs)
        elif orientation == "ver":
            ax.fill_between(y, x, alpha=alpha, **kwargs)
        else:
            raise ValueError

    ax.plot(x, y, alpha=1, label=label, **kwargs)


def rose_plot(
    ax,
    angles,
    nbins=16,
    theta_min=0,
    theta_max=360,
    density=None,
    offset=None,
    lab_unit="degrees",
    start_zero=False,
    as_hist=False,
    theta_0="N",
    **kwargs,
):
    """
    from https://stackoverflow.com/questions/22562364/circular-histogram-for-python
    Plot polar histogram of angles on ax. ax must have been created using
    subplot_kw=dict(projection='polar'). Angles are expected in radians.
    """
    if offset is None:
        ax.set_theta_offset(np.pi / 2)
    else:
        ax.set_theta_offset(offset)

    # parse kwargs
    edge_color = kwargs.pop("edge_color", "g")
    color = kwargs.pop("color", "g")
    fill = kwargs.pop("fill", False)
    linewidth = kwargs.pop("linewidth", 2)
    alpha = kwargs.pop("alpha", 1)
    xticks = kwargs.pop("xticks", True)

    # Wrap angles to [-pi, pi)
    # angles = (angles + np.pi) % (2 * np.pi) - np.pi

    # Set bins symetrically around zero
    if start_zero:
        # To have a bin edge at zero use an even number of bins
        if nbins % 2:
            nbins += 1
        bins = np.linspace(-np.pi, np.pi, num=nbins + 1)
    else:
        bins = np.linspace(
            np.radians(theta_min), np.radians(theta_max), num=nbins + 1
        )

    # Bin data and record counts
    count, bins = np.histogram(angles, bins=bins)

    # Compute width of each bins
    widths = np.diff(bins)

    # By default plot density (frequency potentially misleading)
    if density is None or density is True:
        # Area to assign each bins
        area = count / angles.size
        # Calculate corresponding bins radius
        radius = (area / np.pi) ** 0.5
    else:
        radius = count

    # Plot data on ax
    if as_hist:
        ax.bar(
            bins[:-1],
            radius,
            zorder=1,
            align="edge",
            width=widths,
            color=color,
            edgecolor=edge_color,
            fill=fill,
            linewidth=linewidth,
            alpha=alpha,
        )
    else:
        radius = np.concatenate([radius, [radius[0]]])
        if fill:
            ax.fill_between(
                bins, 0, radius, color=color, lw=linewidth, alpha=alpha
            )
        ax.plot(bins, radius, color=color, lw=linewidth, alpha=0.8)

    # Remove ylabels, they are mostly obstructive and not informative
    ax.set_yticks([])

    if not xticks:
        ax.set_xticks([])

    if lab_unit == "radians":
        label = [
            "$0$",
            r"$\pi/4$",
            r"$\pi/2$",
            r"$3\pi/4$",
            r"$\pi$",
            r"$5\pi/4$",
            r"$3\pi/2$",
            r"$7\pi/4$",
        ]
        ax.set_xticklabels(label)
