import sys

sys.path.append("./")

import numpy as np
from scipy.spatial import distance
import math

from fcutils.maths import derivative


def subtract_angles(lhs, rhs):
    """Return the signed difference between angles lhs and rhs

    Return ``(lhs - rhs)``, the value will be within ``[-math.pi, math.pi)``.
    Both ``lhs`` and ``rhs`` may either be zero-based (within
    ``[0, 2*math.pi]``), or ``-pi``-based (within ``[-math.pi, math.pi]``).
    """

    return math.fmod((lhs - rhs) + math.pi * 3, 2 * math.pi) - math.pi


def slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def intercept(x1, y1, x2, y2):
    return (x1 * y2 - x2 * y1) / (x1 - x2)


def get_random_point_on_line_between_two_points(x1, y1, x2, y2):
    slop = slope(x1, y1, x2, y2)
    interc = intercept(x1, y1, x2, y2)

    if np.isnan(interc):
        interc = 0

    # take a random X between the two values and compute y accordingly
    x = np.random.uniform(x1, x2)
    y = slop * x + interc
    return (x, y)


def calc_distance_between_point_and_line(line_points, p3):
    """[Calcs the perpendicular distance between a point and a line]
    
    Arguments:
        line_points {[list]} -- [list of two 2-by-1 np arrays with the two points that define the line]
        p3 {[np array]} -- [point to calculate the distance from]
    """
    p1, p2 = np.array(line_points[0]), np.array(line_points[1])
    return np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1)


def cals_distance_between_vector_and_line(line_points, v):
    dist = []
    if v.shape[1] > v.shape[0]:
        raise ValueError(
            "This function expects and NxM array with N being the number of frames and N>M, ideally M=2"
        )

    for i in range(v.shape[0]):
        p = [v[i, 0], v[i, 1]]
        dist.append(calc_distance_between_point_and_line(line_points, p))
    return dist


def calc_distance_between_points_2d(p1, p2):
    """calc_distance_between_points_2d [summary]
    
    Arguments:
        p1 {[list, array]} -- [X,Y for point one]
        p2 {[list, array]} -- [X,Y for point two]
    
    Returns:
        [float] -- [eucliden distance]

    Test: - to check : print(zero, oneh, negoneh)
    >>> zero = calc_distance_between_points_2d([0, 0], [0, 0])
    >>> oneh = calc_distance_between_points_2d([0, 0], [100, 0])
    >>> negoneh = calc_distance_between_points_2d([-100, 0], [0, 0])
    """

    return distance.euclidean(p1, p2)


def calc_distance_between_points_in_a_vector_2d(x, y):
    """
        Given a 2D array with eg X,Y tracking data it returns
        the distance between each (x,y) point
    """
    x_dot = np.abs(derivative(x))
    y_dot = np.abs(derivative(y))
    return np.sqrt(x_dot ** 2 + y_dot ** 2)


def calc_distance_between_points_two_vectors_2d(v1, v2):
    """calc_distance_between_points_two_vectors_2d [pairwise distance between vectors points]
    
    Arguments:
        v1 {[np.array]} -- [description]
        v2 {[type]} -- [description]
    
    Raises:
        ValueError -- [description]
        ValueError -- [description]
        ValueError -- [description]
    
    Returns:
        [type] -- [description]

    testing:
    >>> v1 = np.zeros((2, 5))
    >>> v2 = np.zeros((2, 5))
    >>> v2[1, :]  = [0, 10, 25, 50, 100]
    >>> d = calc_distance_between_points_two_vectors_2d(v1.T, v2.T)
    """
    # Check dataformats
    if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
        raise ValueError("Invalid argument data format")
    if not v1.shape[1] == 2 or not v2.shape[1] == 2:
        raise ValueError("Invalid shape for input arrays")
    if not v1.shape[0] == v2.shape[0]:
        raise ValueError("Error: input arrays should have the same length")

    # Calculate distance
    if v1.shape[1] < 20000 and v1.shape[0] < 20000:
        # For short vectors use cdist
        dist = distance.cdist(v1, v2, "euclidean")
        dist = dist[:, 0]
    else:
        dist = [
            calc_distance_between_points_2d(p1, p2) for p1, p2 in zip(v1, v2)
        ]
    return dist


def calc_distance_from_point(v, point):
    """[Calculates the euclidean distance from the point at each timepoint]
    
    Arguments:
        v {[np.ndarray]} -- [2D array with XY coordinates]
        point {[tuple]} -- [tuple of length 2 with X and Y coordinates of point]
    """
    assert isinstance(v, np.ndarray), "Input data needs to be a numpy array"

    if v.shape[0] == 2:
        pass  # good
    elif v.shape[1] == 2:
        v = v.T
    else:
        raise ValueError("Vector of weird shape: {}".format(v.shape))

    delta_x = v[0, :] - point[0]
    delta_y = v[1, :] - point[1]

    return np.sqrt(delta_x ** 2 + delta_y ** 2)


def calc_angles_with_arctan(x, y):
    theta = np.degrees(np.arctan2(x, y))

    if not isinstance(theta, np.ndarray):
        if theta < 0:
            theta += 360
        if theta < 0 or theta > 360:
            raise ValueError
        return theta

    theta[theta < 0] += 360
    if np.nanmax(theta) > 360 or np.nanmin(theta) < 0:
        raise ValueError("Something went wrong while computing angles")

    return theta


def calc_angle_between_points_of_vector_2d(x, y):
    """
        Given 2 1d arrays specifying for instance the X and Y coordinates at each frame,
        computes the angle between successive points (x,y)
    """
    return np.degrees(np.arctan2(derivative(x), derivative(y)))


def calc_angle_between_vectors_of_points_2d(x1, y1, x2, y2):
    """ 
        Given two sets of X,Y coordinates computes the angle
        between each pair of point in each set of coordinates.
    """
    # Calculate
    delta_x = np.array(x2 - x1)
    delta_y = np.array(y2 - y1)

    return calc_angles_with_arctan(delta_x, delta_y)


def calc_ang_velocity(angles):
    """calc_ang_velocity [calculates the angular velocity ]
    
    Arguments:
        angles {[np.ndarray]} -- [1d array with a timeseries of angles in degrees]
    
    Returns:
        [np.ndarray] -- [1d array with the angular velocity in degrees at each timepoint]
    
    testing:
    >>> v = calc_ang_velocity([0, 10, 100, 50, 10, 0])    
    """
    # Check input data
    if not isinstance(angles, np.ndarray) and not isinstance(angles, list):
        raise ValueError("Invalid input data format")

    if isinstance(angles, np.ndarray):
        if len(angles.shape) > 1:
            angles = angles.ravel()

    # Calculate
    angles_radis = np.unwrap(np.radians(np.nan_to_num(angles)))  # <- to unwrap
    ang_vel_rads = derivative(angles_radis)
    return np.degrees(ang_vel_rads)
