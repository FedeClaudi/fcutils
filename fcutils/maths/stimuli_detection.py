import sys

sys.path.append("./")


import numpy as np
import pandas as pd

from .filtering import *


# raise NotImplementedError("This code is old and should not be used, better functions are available in behaviour")

"""
    Collection of functions that analyse AI time series data to find the start and end of stimuli delivered through Mantis
"""


def find_peaks_in_signal(signal, time_limit, th, above=True):
    """[Function to find the start of square peaks in a time series. 
    Useful for example to find frame starts or stim starts in analog input data]
    
    Arguments:
        signal {[np.array]} -- [the time series to be analysd]
        time_limit {[float]} -- [min time inbetween peaks]
        th {[float]} -- [where to threshold the signal to identify the peaks]
    
    Returns:
        [np.ndarray] -- [peak starts times]
    """
    if above:
        above_th = np.where(signal > th)[0]
    else:
        above_th = np.where(signal < th)[0]
    if not np.any(above_th):
        return np.array([])

    peak_starts = [x for x, d in zip(above_th, np.diff(above_th)) if d > time_limit]

    # add the first and last above_th times to make sure all frames are included
    peak_starts.insert(0, above_th[0])
    peak_starts.append(above_th[-1])

    # we then remove the second item because it corresponds to the end of the first peak
    peak_starts.pop(1)

    return np.array(peak_starts)


def find_audio_stimuli(data, th, sampling_rate):
    above_th = np.where(data > th)[0]
    peak_starts = [x + 1 for x in np.where(np.diff(above_th) > sampling_rate)]
    stim_start_times = above_th[peak_starts]
    try:
        stim_start_times = np.insert(stim_start_times, 0, above_th[0])
    except:
        raise ValueError
    else:
        return stim_start_times


def find_visual_stimuli(data, th, sampling_rate):
    # Filter the data to remove high freq noise, then take the diff and thereshold to find changes
    filtered = butter_lowpass_filter(data, 75, int(sampling_rate / 2))
    d_filt = np.diff(filtered)

    starts = find_peaks_in_signal(d_filt, 10000, -0.0005, above=False)[1:]
    ends = find_peaks_in_signal(d_filt, 10000, 0.0003, above=True)[1:]

    if not len(starts) == len(ends):
        if abs(len(starts) - len(ends)) > 1:
            raise ValueError(
                "Too large error during detection: s:{} e{}".format(
                    len(starts), len(ends)
                )
            )
        print(
            "Something went wrong: {} - starts and {} - ends".format(
                len(starts), len(ends)
            )
        )

        # # ? Fo1r debugging
        # f, ax = plt.subplots()
        # ax.plot(filtered, color='r')
        # ax.plot(butter_lowpass_filter(np.diff(filtered), 75, int(sampling_rate/2)), color='g')
        # ax.scatter(starts, [0.25 for i in starts], c='r')
        # ax.scatter(ends, [0 for i in ends], c='k')

        # plt.show()

        # to_elim = int(input("Which one to delete "))
        to_elim = -1
        if len(starts) > len(ends):
            starts = np.delete(starts, to_elim)
        else:
            ends = np.delete(ends, to_elim)

    assert len(starts) == len(ends), "cacca"

    # Return as a list of named tuples
    stim = namedtuple("stim", "start end")
    stimuli = [stim(s, e) for s, e in zip(starts, ends)]

    for s, e in stimuli:  # check that the end is after the start
        return
        # if e < s: raise ValueError("Wrong stimuli detection")

    return stimuli
