import numpy as np


# ---------------------------------------------------------------------------- #
#                               ANGLES STATISTICS                              #
# ---------------------------------------------------------------------------- #
def average_angles(angles):
    """Average (mean) of angles

    Return the average of an input sequence of angles. The result is between
    ``0`` and ``2 * math.pi``.
    If the average is not defined (e.g. ``average_angles([0, math.pi]))``,
    a ``ValueError`` is raised.
    """

    x = sum(math.cos(a) for a in angles)
    y = sum(math.sin(a) for a in angles)

    if x == 0 and y == 0:
        raise ValueError("The angle average of the inputs is undefined: %r" % angles)

    # To get outputs from -pi to +pi, delete everything but math.atan2() here.
    return math.fmod(math.atan2(y, x) + 2 * math.pi, 2 * math.pi)


def moving_average(arr, window_size):
    cumsum_vec = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size


def mean_confidence_interval(data, confidence=0.95):
    mean, var, std = stats.bayes_mvs(data)
    res = namedtuple("confidenceinterval", "mean low high")
    return res(mean.statistic, mean.minmax[0], mean.minmax[1])


def percentile_range(data, low=5, high=95):
    """[Calculates the range between the low and high percentiles]
	"""

    lowp = np.percentile(data, low)
    highp = np.percentile(data, high)
    median = np.median(data)
    mean = np.mean(data)
    std = np.std(data)
    sem = stats.sem(data)

    res = namedtuple("percentile", "low median mean high std sem")
    return res(lowp, median, mean, highp, std, sem)
