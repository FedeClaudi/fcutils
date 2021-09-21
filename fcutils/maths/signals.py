import pandas as pd
import numpy as np
from scipy.signal import resample
from scipy import stats


def convolve_with_gaussian(
    data: np.ndarray, kernel_width: int = 21
) -> np.ndarray:
    """
        Convolves a 1D array with a gaussian kernel of given width
    """
    # create kernel and normalize area under curve
    norm = stats.norm(0, kernel_width)
    X = np.linspace(norm.ppf(0.0001), norm.ppf(0.9999), kernel_width)

    _kernnel = norm.pdf(X)
    kernel = _kernnel / np.sum(_kernnel)

    return np.convolve(data, kernel, mode="same")


def get_onset_offset(signal, th, clean=True):
    """
        Get onset/offset times when a signal goes below>above and
        above>below a given threshold
        Arguments:
            signal: 1d numpy array
            th: float, threshold
            clean: bool. If true ends before the first start and 
                starts after the last end are removed
    """
    above = np.zeros_like(signal)
    above[signal >= th] = 1

    der = derivative(above)
    starts = np.where(der > 0)[0]
    ends = np.where(der < 0)[0]

    if above[0] > 0:
        starts = np.concatenate([[0], starts])
    if above[-1] > 0:
        ends = np.concatenate([ends, [len(signal)]])

    if clean:
        ends = np.array([e for e in ends if e > starts[0]])

        if np.any(ends):
            starts = np.array([s for s in starts if s < ends[-1]])

    if not np.any(starts):
        starts = np.array([0])
    if not np.any(ends):
        ends = np.array([len(signal)])

    return starts, ends


def smooth_hanning(x, window_len=11):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Arguments:
        x: np.ndarray/ the input signal 
        window_len: int. the dimension of the smoothing window; should be an odd integer
    Returns:
        the smoothed signal
    
    """
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    w = eval("np." + window_len + "(window_len)")
    y = np.convolve(w / w.sum(), s, mode="valid")
    return y


def upsample_signal(X, original_sampling_rate, target_sampling_rate):
    """
        Updamples a signal to a target sampling rate

        Arguments:
            X: np.array with data
            original_sampling_rate: int. Sampling rate of input data
            target_sampling_rate: int. Sampling rate of output data
    """
    n_seconds = len(X) / original_sampling_rate
    goal_n_samples = np.int(n_seconds * target_sampling_rate)
    return resample(X, goal_n_samples)


def rolling_pearson_correlation(X, Y, window_size):
    """
        Computs the rolling (windowed) Pearson's
        correlation between two time series

        Arguments:
            X: 1d np.array with data
            Y: 1d np.array with data
            window_size: int. Size of rolling window
    """

    # Interpolate missing data.
    df = pd.DataFrame(dict(X=X, Y=Y))
    df_interpolated = df.interpolate()

    # Compute rolling window synchrony
    rolling_r = (
        df_interpolated["X"]
        .rolling(window=window_size, center=True)
        .corr(df_interpolated["Y"])
    )
    return rolling_r.values


def rolling_mean(X, window_size):
    """
        Computs the rolling mean of a signal

        Arguments:
            X: 1d np.array with data
            window_size: int. Size of rolling window
    """
    X = pd.Series(X)

    try:
        moving_avg = np.array(
            X.rolling(window=window_size, min_periods=1).mean(center=True)
        )
    except TypeError:  # compatible with pandas versions
        moving_avg = np.array(
            X.rolling(window=window_size, min_periods=1, center=True).mean()
        )
    return moving_avg


def derivative(X, axis=0, order=1):
    """"
        Takes the derivative of an array X along a given axis


        Arguments:
            X: np.array with data
            axis: int. Axis along which the derivative is to be computed
            order: int. Derivative order
    """

    return np.diff(X, n=order, axis=axis, prepend=0)
