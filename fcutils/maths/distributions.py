import math
import numpy as np
from scipy import stats
import statsmodels.api as sm


def fit_kde(x, **kwargs):
    """ Fit a KDE using StatsModels. 
        kwargs is useful to pass stuff to the fit, e.g. the binwidth (bw)"""
    x = np.array(x).astype(np.float)
    kde = sm.nonparametric.KDEUnivariate(x)
    kde.fit(**kwargs)  # Estimate the densities
    return kde


def get_distribution(dist, *args, n_samples=10000):
    if dist == "uniform":
        return np.random.uniform(args[0], args[1], n_samples)
    elif dist == "normal":
        return np.random.normal(args[0], args[1], n_samples)
    elif dist == "beta":
        return np.random.beta(args[0], args[1], n_samples)
    elif dist == "gamma":
        return np.random.gamma(args[0], args[1], n_samples)


def get_parametric_distribution(
    dist, *args, x0=0.000001, x1=0.9999999, **kwargs
):
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


def gamma_distribution_params(
    mean=None, sd=None, mode=None, shape=None, rate=None
):
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
