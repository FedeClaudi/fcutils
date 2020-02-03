import sys

sys.path.append("./")

import numpy as np
from scipy import misc, signal, stats
import statsmodels.api as sm
import math


# ---------------------------------------------------------------------------- #
#                      COLLECTION OF SOME MATH FUNCTIONS                       #
# ---------------------------------------------------------------------------- #

# ------------------------ EXPONENTIALS AND LOGARITHMS ----------------------- #
def fexponential(x, a, b, c):
    return a * np.exp(-b * x) + c


def exponential(x, a, b, c, d):
    return a * np.exp(-c * (x - b)) + d


def flogarithmic(x, a, b, c):
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


"""
	In these functions "L" is a scaling factor, 
	"b" is a Y-shift factor
	"x0" is a X-shift factor
"""


def hill_function(x, n, L, b):  # ? no work
    return L / (1 + x ** -n) + b


def hyperbolic_tangent(x, L=1, b=0, x0=0):
    return np.tanh(x - x0) / L + b


def arctangent(x, L=2, b=0, x0=0):
    return np.arctan(x - x0) / L + b


def gudermannian(x, L=2, b=0, x0=0):
    return L * np.arctan(np.tanh((x - x0) / L)) + b


def generalised_logistic(x, a):  # ? no work
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


# ---------------------------------------------------------------------------- #
#                            REGRESSION AND FITTING                            #
# ---------------------------------------------------------------------------- #
def polyfit(order, x, y):
    #  calculate polynomial
    z = np.polyfit(x, y, order)
    f = np.poly1d(z)
    return f


def linear_regression(X, Y, robust=False):
    import statsmodels.api as sm

    # ! sns.regplot much better
    # remove NANs
    remove_idx = [i for i, (x, y) in enumerate(zip(X, Y)) if np.isnan(x) or np.isnan(y)]

    X = np.delete(X, remove_idx)
    Y = np.delete(Y, remove_idx)

    # Regression with Robust Linear Model
    X = sm.add_constant(X)

    if robust:
        res = sm.RLM(Y, X, missing="drop").fit()
    else:
        res = sm.OLS(Y, X).fit()  # Ordinary least squares

    return X, res.params[0], res.params[1], res


def fit_kde(x, **kwargs):
    """ Fit a KDE using StatsModels. 
		kwargs is useful to pass stuff to the fit, e.g. the binwidth (bw)"""
    x = np.array(x).astype(np.float)
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit(**kwargs)  # Estimate the densities
    return kde


# ---------------------------------------------------------------------------- #
#                                 DISTRIBUTIONS                                #
# ---------------------------------------------------------------------------- #


def get_distribution(dist, *args, n_samples=10000):
    if dist == "uniform":
        return np.random.uniform(args[0], args[1], n_samples)
    elif dist == "normal":
        return np.random.normal(args[0], args[1], n_samples)
    elif dist == "beta":
        return np.random.beta(args[0], args[1], n_samples)
    elif dist == "gamma":
        return np.random.gamma(args[0], args[1], n_samples)


def get_parametric_distribution(dist, *args, x0=0.000001, x1=0.9999999, **kwargs):
    if dist == "beta":
        dist = stats.beta(*args, **kwargs)
    elif dist == "normal" or dist == "norm" or dist == "gaussian":
        dist = stats.norm(*args, **kwargs)
    else:
        raise NotImplementedError

    support = np.linspace(dist.ppf(x0), dist.ppf(x1), 100)
    density = dist.pdf(support)
    return dist, support, density


# --------------------- get parameters for distirbutions --------------------- #
def beta_distribution_params(
    a=None, b=None, mu=None, sigma=None, omega=None, kappa=None
):
    """[converts parameters of beta into different formulations]
	
	Keyword Arguments:
		a {[type]} -- [a param] (default: {None})
		b {[type]} -- [b param] (default: {None})
		mu {[type]} -- [mean] (default: {None})
		sigma {[type]} -- [standard var] (default: {None})
		omega {[type]} -- [mode] (default: {None})
		kappa {[type]} -- [concentration] (default: {None})
	
	Raises:
		NotImplementedError: [description]
	"""
    if kappa is not None and omega is not None:
        a = omega * (kappa - 2) + 1
        b = (1 - omega) * (kappa - 2) + 1
        return a, b
    elif a is not None and b is not None:
        mu = a / (a + b)
        omega = (a - 1) / (a + b - 2)
        kappa = a + b
        return mu, omega, kappa
    else:
        raise NotImplementedError


def gamma_distribution_params(mean=None, sd=None, mode=None, shape=None, rate=None):
    if mean is not None and sd is not None:
        if mean < 0:
            raise NotImplementedError

        shape = mean ** 2 / sd ** 2
        rate = mean / sd ** 2
    elif mode is not None and sd is not None:
        if mode < 0:
            raise NotImplementedError

        rate = (mode + math.sqrt(mode ** 2 + 4 * (sd ** 2))) / (2 * (sd ** 2))
        shape = 1 + mode * rate
    elif shape is not None and rate is not None:
        mu = shape / rate
        sd = math.sqrt(shape) / rate
        return mu, sd
    return shape, rate
