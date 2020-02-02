import sys
sys.path.append('./')

import numpy as np
from scipy import misc, signal, stats
import pandas as pd
from scipy.spatial import distance
from math import factorial, atan2, degrees, acos, sqrt, pi
import math
from math import atan2,degrees
import matplotlib.pyplot as plt
from scipy.signal import medfilt as median_filter
from scipy.interpolate import interp1d
from collections import namedtuple
from scipy import stats


def binArray(data, axis, binstep, binsize, func=np.nanmean):
	"""
		Bins a numpy array along a specific axis
	"""
	data = np.array(data)
	dims = np.array(data.shape)
	argdims = np.arange(data.ndim)
	argdims[0], argdims[axis]= argdims[axis], argdims[0]
	data = data.transpose(argdims)
	data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
	data = np.array(data).transpose(argdims)
	return data

def log_transform(im):
	'''returns log(image) scaled to the interval [0,1]'''
	try:
		(min, max) = (im[im > 0].min(), im.max())
		if (max > min) and (max > 0):
			return (np.log(im.clip(min, max)) - np.log(min)) / (np.log(max) - np.log(min))
	except:
		pass
	return im

def find_nearest(a, a0):
	"Element in nd array `a` closest to the scalar value `a0`"
	idx = np.abs(a - a0).argmin()
	return a.flat[idx]
	
def interpolate_nans(A):
	nan = np.nan
	ok = ~np.isnan(A)
	xp = ok.ravel().nonzero()[0]
	fp = A[~np.isnan(A)]
	x  = np.isnan(A).ravel().nonzero()[0]

	A[np.isnan(A)] = np.interp(x, xp, fp)

	return A

def remove_nan_1d_arr(arr):
	nan_idxs = [i for i,x in enumerate(arr) if np.isnan(x)]
	return np.delete(arr, nan_idxs)
	
def normalise_to_val_at_idx(arr, idx):
	arr = remove_nan_1d_arr(arr)
	val = arr[idx]
	return arr / arr[idx]

def normalise_1d(arr):
	arr = np.nan_to_num(arr)
	normed = (arr - np.min(arr))/np.ptp(arr)
	return normed

def find_hist_peak(arr, bins=None, density=True):
	if bins is None: bins = np.linspace(np.min(arr), np.max(arr), 10)
	
	hist, binedges = np.histogram(arr, bins=bins, density=density)
	yi, y = np.argmax(hist), np.max(hist)
	x = np.mean(binedges[yi:yi+2])
	return yi, y, x

# ! MISC
def fill_nans_interpolate(y, pkind='linear'):
	"""
	Interpolates data to fill nan values

	Parameters:
		y : nd array 
			source data with np.NaN values

	Returns:
		nd array 
			resulting data with interpolated values instead of nans
	"""
	aindexes = np.arange(y.shape[0])
	agood_indexes, = np.where(np.isfinite(y))
	f = interp1d(agood_indexes
			, y[agood_indexes]
			, bounds_error=False
			, copy=False
			, fill_value="extrapolate"
			, kind=pkind)
	return f(aindexes)


# ! ERROR CORRECTION
def correct_speed(speed):
	speed = speed.copy()
	perc99 = np.nanpercentile(speed, 99.5)
	speed[speed>perc99] = perc99
	return median_filter(speed, 31)

def remove_tracking_errors(tracking, debug = False):
	"""
		Get timepoints in which the velocity of a bp tracking is too high and remove them
	"""
	filtered = np.zeros(tracking.shape)
	for i in np.arange(tracking.shape[1]):
		temp = tracking[:, i].copy()
		filtered[:, i] = signal.medfilt(temp, kernel_size  = 5)

		if debug:
			plt.figure()
			plt.plot(tracking[:, i], color='k', linewidth=2)
			plt.plot(temp, color='g', linewidth=1)
			plt.plot(filtered[:, i], 'o', color='r')
			plt.show()

	return filtered

