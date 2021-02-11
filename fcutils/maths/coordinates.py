import numpy as np


def R(theta):
    """
        Returns the rotation matrix for rotating an object
        centered around the origin with a given angle

        Arguments:
            theta: angle in degrees

        Returns:
            R: 2x2 np.ndarray with rotation matrix
    """
    theta = np.radians(theta)
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def M(axis="x"):
    """
        Returns a matrix to mirror an object against a given axis

        Arguments:
            axis: str. 'x', 'y', 'origin' or 'xy'

        Returns:
            M: mirror matrix
    """
    if axis == "x":
        return np.array([[1, 0], [0, -1]])
    elif axis == "y":
        return np.array([[-1, 0], [0, 1]])
    elif axis == "origin":
        return np.array([[-1, 0], [0, -1]])
    elif axis == "xy":
        return np.array([[0, 1], [1, 0]])
    else:
        raise NotImplementedError(
            f"Could not recognize axis of mirroring: {axis}"
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
