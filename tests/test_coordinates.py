from fcutils.maths import coordinates

import numpy as np
from numpy.random import uniform


def test_coordinates_transformations():
    for i in range(5):
        x, y = uniform(-100, 100), uniform(-100, 100)

        r, p = coordinates.cart2pol(x, y)

        x2, y2 = coordinates.pol2cart(r, p)

        assert np.abs(x - x2) < 0.1
        assert np.abs(y - y2) < 0.1
