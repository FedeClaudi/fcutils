import sys

sys.path.append("./")

import numpy as np
import pandas as pd

from scipy.signal import butter, lfilter, freqz, resample, wiener, gaussian
from scipy.ndimage import filters
from scipy.signal import medfilt as median_filter
from scipy.interpolate import interp1d
from math import factorial
# from neuroanalysis.filter import (
#     bessel_filter,
#     butterworth_filter,
#     savgol_filter,
#     downsample,
# )



def smooth_hanning(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def upsample_signal(start_fps, goal_fps, signal):
    n_seconds = len(signal) / start_fps
    goal_n_samples = np.int(n_seconds * goal_fps)
    return resample(signal, goal_n_samples)


def line_smoother(y, window_size=31, order=5, deriv=0, rate=1):
    # Apply a Savitzy-Golay filter to smooth traces
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat(
        [[k ** i for i in order_range] for k in range(-half_window, half_window + 1)]
    )
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with values taken from the signal itself
    try:
        firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode="valid")
    except:
        # print('ops smoothing')
        y = np.array(y)
        firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode="valid")


def line_smoother_convolve(y, window_size=31):
    box = np.ones(window_size) / window_size
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def median_filter_1d(x, pad=20, kernel=11):
    half_pad = int(pad / 2)
    x_pad = np.pad(x, pad, "edge")
    x_filtered = median_filter(x_pad, kernel_size=kernel)[pad:-pad]
    return x_filtered


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
