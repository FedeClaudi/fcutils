import numpy as np
from scipy import stats
from collections import namedtuple


def unwrap(X, isdeg=True):
    """
        Unwraps an array of angular data.

        Arguments:
            X: np.array with data
            isdeg: bool. If true the data is in degrees
    """
    if isdeg:
        X = np.radians(X)

    X = np.unwrap(X)

    if isdeg:
        X = np.degrees(X)

    return X


def find_nearest(X, value):
    """
        Finds the index of the array value that is closest to a target value.

        Arguments:
            X: np.ndarray with data
            value: float, int. Value
    """

    idx = np.abs(X - value).argmin()
    return idx


def percentile_range(X, low=5, high=95):
    """ 
        Calculates the range between the low and high percentiles
        in an array of data X
        
        Arguments:
            X: data
            low, high: int. Low and high percentiles
    """

    lowp = np.nanpercentile(X, low)
    highp = np.nanpercentile(X, high)
    median = np.nanmedian(X)
    mean = np.nanmean(X)
    std = np.std(X)
    sem = stats.sem(X)

    res = namedtuple("percentile", "low median mean high std sem")
    return res(lowp, median, mean, highp, std, sem)
