import numpy as np


def R(theta):
    '''
        Returns the rotation matrix necessary to remove the
        rotation of an object centered at the origin

        Arguments:
            theta: angle in degrees

        Returns:
            R: 2x2 np.ndarray with rotation matrix
    '''
    theta = np.radians(theta)
    return np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )


def cart2pol(x, y):
    """
        Cartesian to polar coordinates

        angles in degrees
    """
    rho = np.hypot(x, y)
    phi = np.degrees(np.arctan2(y, x))
    return rho, phi


def pol2cart(rho, phi):
    """
        Polar to cartesian coordinates

        angles in degrees
    """
    x = rho * np.cos(np.radians(phi))
    y = rho * np.sin(np.radians(phi))
    return x, y
