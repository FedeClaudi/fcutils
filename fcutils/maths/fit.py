import statsmodels.api as sm
import numpy as np


def fit_polynomial(order, X, Y):
    """
        Fits a polinomia to data, with numpy

        Arguments:
            order: int. Order of the polinomial
            X, Y: 1d np.ndarray with data

        Reurns:
            f: np.array with fitted polynomial
    """
    #  calculate polynomial
    z = np.polyfit(X, Y, order)
    f = np.poly1d(z)
    return f


def linear_regression(X, Y, robust=False, rais_on_nan=False):
    """
        Fits a linear regression and returns the results and parameters

        Arguments:
            X, Y: np.ndarrays with data
            robust: bool. If trye robust RLM algorithm is fitted
            rais_on_nan: bool. If true an error is raised if there's nans
                in the input data

        Returns:
            X: np.array. With nans removed
            slope: float. SLope of linear regression
            intercept: float. Intercept of linear regressoin
            res: Results object with fitted model
    """
    # remove NANs
    remove_idx = [
        i for i, (x, y) in enumerate(zip(X, Y)) if np.isnan(x) or np.isnan(y)
    ]
    if rais_on_nan and remove_idx:
        raise ValueError(
            "Found invalid values in input data during linear regression"
        )

    X = np.delete(X, remove_idx)
    Y = np.delete(Y, remove_idx)

    X = sm.add_constant(X)
    if robust:
        res = sm.RLM(Y, X, missing="drop").fit()
    else:
        res = sm.OLS(Y, X).fit()  # Ordinary least squares

    return X, res.params[0], res.params[1], res


def fit_kde(X, **kwargs):
    """ 
        Fit a KDE using StatsModels. 

        Arguments:
            X: 1d np array with data to fit the KDE to.
            kwargs is useful to pass stuff to the fit, e.g. the binwidth (bw)

    """
    X = np.array(X).astype(np.float)
    kde = sm.nonparametric.KDEUnivariate(X)
    kde.fit(**kwargs)  # Estimate the densities

    return kde
