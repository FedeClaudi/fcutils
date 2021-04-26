import numpy as np
import math
from scipy import stats


# ------------------------ EXPONENTIALS AND LOGARITHMS ----------------------- #
def fexponential(x, a, b, c):
    return a * np.exp(-b * x) + c


def exponential(x, a, b, c, d):
    return a * np.exp(-c * (x - b)) + d


def logarithmic(x, a, b, c):
    return a * np.log(b * x) + c


# --------------------------------- SIGMOIDS --------------------------------- #
def logistic(x, L, x0, k, b):
    """
    L -> shirnks the function on the Y axis. 
    x0 -> x shift. 
    k  -> slope. the smaller the flatter. Vals > 5 are good
    b -> y shift. Shifts thw whole curve up and donw
    """
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def centered_logistic(x, L, x0, k):
    """
    L -> shirnks the function on the Y axis. 
    x0 -> x shift. 
    k  -> slope. the smaller the flatter. Vals > 5 are good
    """
    b = (1 - L) / 2
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def hill_function(x, n, L, b):  # ? no work
    return L / (1 + x ** -n) + b


def hyperbolic_tangent(x, L=1, b=0, x0=0):
    return np.tanh(x - x0) / L + b


def arctangent(x, L=2, b=0, x0=0):
    return np.arctan(x - x0) / L + b


def gudermannian(x, L=2, b=0, x0=0):
    return L * np.arctan(np.tanh((x - x0) / L)) + b


def generalised_logistic(x, a, x0):  # ? no work
    """
        a > 0
    """
    if a <= 0:
        raise ValueError("Paramter 'a' should be > 0")
    return (1 + np.exp(-(x - x0))) ** -a


def algebraic_sigmoid(x, L=1, b=0, x0=0):
    return (x - x0 / (math.sqrt(1 + (x - x0) ** 2))) / L + b


def error_function(x, x0=0, scale=1, L=1, b=0):
    norm = stats.norm(x0, scale)
    return norm.cdf(x) * L + b


def centered_error_function(x, x0=0, scale=1, L=1):
    b = (1 - L) / 2
    norm = stats.norm(x0, scale)
    return norm.cdf(x) * L + b


# ------------------------------ OTHER FUNCTIONS ----------------------------- #
def linear_func(x, a, b):
    return x * a + b


def step_function(x, a, b, c):
    # Step function
    """
    a: value at x = b
    f(x) = 0 if x<b, a if x=b and 2*a if  x > a
    """
    return a * (np.sign(x - b) + c)
