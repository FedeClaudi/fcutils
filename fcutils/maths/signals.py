import pandas as pd
import numpy as np
from scipy.signal import resample


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
    moving_avg = np.array(
        X.rolling(window=window_size, min_periods=1).mean(center=True)
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
