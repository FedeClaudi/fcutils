import matplotlib.pyplot as plt

from fcutils.plotting.utils import clean_axes

"""
    Bunch of functions for quickly plotting stuff, they create a figure and show the results, no fuss about it
"""


def quick_scatter(x, y, ax_kwargs={}, **kwargs):
    f, ax = plt.subplots()
    ax.scatter(x, y, **kwargs)
    clean_axes(f)
    ax.set(**ax_kwargs)
    return  f, ax

def quick_plot(*args, ax_kwargs={}, **kwargs):
    f, ax = plt.subplots()
    for var in args:
        ax.plot(var, **kwargs)
    clean_axes(f)
    ax.set(**ax_kwargs)
    return f, ax