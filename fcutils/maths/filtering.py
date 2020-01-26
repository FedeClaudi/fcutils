import sys
sys.path.append('./')

import numpy as np
import pandas as pd

from scipy.signal import butter, lfilter, freqz, resample, wiener, gaussian
from scipy.ndimage import filters
from scipy.signal import medfilt as median_filter
from scipy.interpolate import interp1d


def upsample_signal(start_fps, goal_fps, signal):
    n_seconds = len(signal)/start_fps
    goal_n_samples = np.int(n_seconds * goal_fps)
    return (resample(signal, goal_n_samples))


def line_smoother(y, window_size=31, order=5, deriv=0, rate=1):
	# Apply a Savitzy-Golay filter to smooth traces
	order_range = range(order + 1)
	half_window = (window_size - 1) // 2
	# precompute coefficients
	b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
	m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
	# pad the signal at the extremes with values taken from the signal itself
	try:
		firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
		lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
		y = np.concatenate((firstvals, y, lastvals))
		return np.convolve(m[::-1], y, mode='valid')
	except:
		# print('ops smoothing')
		y = np.array(y)
		firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
		lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
		y = np.concatenate((firstvals, y, lastvals))
		return np.convolve(m[::-1], y, mode='valid')

def line_smoother_convolve(y, window_size=31):
	box = np.ones(window_size)/window_size
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth


def median_filter_1d(x, pad=20, kernel=11):
	half_pad = int(pad/2)
	x_pad = np.pad(x, pad, 'edge')
	x_filtered = median_filter(x_pad, kernel_size=kernel)[pad:-pad]
	return x_filtered

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

