from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from fcutils.plot.elements import plot_shaded_withline
from fcutils.maths.distributions import fit_kde


def plot_distribution(
    *args,
    dist_type="logistic",
    comulative=False,
    ax=None,
    shaded=False,
    shade_alpha=0.5,
    line_alpha=1,
    vertical=False,
    flip_y=False,
    x_range=None,
    plot_kwargs={},
    ax_kwargs={},
    fill_offset=0,
    y_scale=1,
    **kwargs,
):
    # Get the distribution
    if dist_type == "logistic":
        dist = stats.logistic(*args, **kwargs)
    elif dist_type == "gamma":
        dist = stats.gamma(*args, **kwargs)
    elif dist_type == "beta":
        dist = stats.beta(*args, **kwargs)
    elif (
        dist_type == "normal" or dist_type == "gaussian" or dist_type == "norm"
    ):
        dist = stats.norm(*args, **kwargs)
    elif dist_type == "exponential":
        dist = np.exp
    else:
        raise NotImplementedError

    # Get the probability density function or comulative density function
    if comulative:
        func = dist.cdf
    else:
        try:
            func = dist.pdf
            if not flip_y:
                x = np.linspace(dist.ppf(0.0001), dist.ppf(0.99999), 100)
            else:
                x = np.linspace(dist.ppf(0.0001), -dist.ppf(0.99999), 100)

        except:
            func = dist
            if not flip_y:
                x = np.linspace(x_range[0], x_range[1], 100)
            else:
                x = np.linspace(x_range[0], -x_range[1], 100)

    # Plot
    if ax is None:
        f, ax = plt.subplots()

    if not shaded:
        if not vertical:
            if not flip_y:
                ax.plot(x, func(x) * y_scale, **plot_kwargs)
            else:
                ax.plot(x, -func(x) * y_scale, **plot_kwargs)

        else:
            if not flip_y:
                ax.plot(func(x), x * y_scale, **plot_kwargs)
            else:
                ax.plot(func(x), -x * y_scale, **plot_kwargs)

    else:
        if not vertical:
            if not flip_y:
                ax.fill_between(
                    x,
                    fill_offset,
                    (func(x) + fill_offset) * y_scale,
                    alpha=shade_alpha,
                    **plot_kwargs,
                )
                ax.plot(
                    x,
                    (func(x) + fill_offset) * y_scale,
                    alpha=line_alpha,
                    **plot_kwargs,
                )
            else:
                ax.fill_between(
                    x,
                    fill_offset,
                    -func(x) + fill_offset * y_scale,
                    alpha=shade_alpha,
                    **plot_kwargs,
                )
                ax.plot(
                    x,
                    -(func(x) + fill_offset) * y_scale,
                    alpha=line_alpha,
                    **plot_kwargs,
                )
        else:
            if not flip_y:
                ax.fill_between(
                    (func(x)) * y_scale + fill_offset,
                    fill_offset,
                    x,
                    alpha=shade_alpha,
                    **plot_kwargs,
                )
                ax.plot(
                    (func(x)) * y_scale + fill_offset,
                    x,
                    alpha=line_alpha,
                    **plot_kwargs,
                )
            else:
                ax.fill_between(
                    (func(x)) * y_scale + fill_offset,
                    fill_offset,
                    -x,
                    alpha=shade_alpha,
                    **plot_kwargs,
                )
                ax.plot(
                    (func(x)) * y_scale + fill_offset,
                    -x,
                    alpha=line_alpha,
                    **plot_kwargs,
                )

    ax.set(**ax_kwargs)

    return ax


def plot_fitted_curve(
    func,
    xdata,
    ydata,
    ax,
    *args,
    xrange=None,
    print_fit=False,
    numpy_polyfit=False,
    fit_kwargs={},
    scatter_kwargs={},
    line_kwargs={},
):
    if numpy_polyfit and not isinstance(numpy_polyfit, int):
        raise ValueError("numpy_polyfit should be an integer")
    # set numpy_polifit to an integer to fit a numpy polinomial with degree numpy polyfit

    if xrange is not None:
        x = np.linspace(xrange[0], xrange[1], 100)
    else:
        x = np.linspace(np.min(xdata), np.max(xdata), 100)

    if not numpy_polyfit:  # ? scipy curve fit instead
        popt, pcov = curve_fit(func, xdata, ydata, *args, **fit_kwargs)
        if print_fit:
            print(popt)
        y = func(x, *popt)
        to_return = popt

    else:
        func = np.poly1d(np.polyfit(xdata, ydata, numpy_polyfit))
        y = func(x)
        to_return = func
    ax.scatter(xdata, ydata, **scatter_kwargs)
    ax.plot(x, y, **line_kwargs)

    return to_return


def plot_kde(
    ax=None,
    z=0,
    kde=None,
    data=None,
    invert=False,
    vertical=False,
    normto=None,
    fig_kwargs={},
    kde_kwargs={},
    **kwargs,
):
    """[Plots a KDE distribution. Plots first the shaded area and then the outline. 
       KDE can be oriented vertically, inverted, normalised...]
    
    Arguments:
        ax {[plt.axis]} -- [ax onto which to plot]
        kde {[np.ndarray]} -- [KDE fitted with statsmodels, optional. Either KDE or data must be passed]
        data {[np.ndarray]} -- [1d numpy array with data (Y values of distribution), optional. Either KDE or data must be passed]
        z {[type]} -- [value used to shift the curve, for a horizontal KDE z=0 means the curve is on the x axis. ]
    
    Keyword Arguments:
        invert {bool} -- [mirror the KDE plot relative to the X or Y axis, depending on ortentation] (default: {False})
        vertical {bool} -- [plot KDE vertically] (default: {False})
        normto {[float]} -- [normalise the KDE so that the peak of the distribution is at a certain value] (default: {None})
        label {[string]} -- [label for the legend] (default: {None})
    
    Returns:
        ax, kde
    """
    if ax is None:
        f, ax = plt.subplots(**fig_kwargs)

    if kde is None:
        if data is None:
            raise ValueError("either KDE or data must be passed")
        kde = fit_kde(data, **kde_kwargs)

    if vertical:
        x = kde.density
        y = kde.support
    else:
        x, y = kde.support, kde.density

    if normto is not None:
        if not vertical:
            y = y / np.max(y) * normto
        else:
            x = x / np.max(x) * normto

    if invert:
        y = z - y
    else:
        if not vertical:
            y = y + z
        else:
            x = x + z

    plot_shaded_withline(ax, x, y, z, **kwargs)

    return ax, kde
