import matplotlib.pyplot as plt

from fcutils.plotting.utils import clean_axes

"""
    Bunch of functions for quickly plotting stuff, they create a figure and show the results, no fuss about it
"""


def quick_scatter(x, y, **kwargs):
    f, ax = plt.subplots()
    ax.scatter(x, y, **kwargs)
    clean_axes(f)
    return  f, ax