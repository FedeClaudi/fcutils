from fcutils.maths import array
import numpy as np


def test_find_nearest():
    arr = np.random.uniform(-100, 100, 100 * 100).reshape(100, 100)
    arr[40, 40] = 1000

    idx = array.find_nearest(arr, 1000)
    assert idx == 4040


def test_find_percentile():
    arr = np.random.uniform(-100, 100, 100 * 100).reshape(100, 100)
    arr[40, 40] = 1000

    array.percentile_range(arr)
