import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


def fit_svc_binary(X, y, test_size=.33, kernel='linear', max_iter=-1):
    """
        Given a 2d array/dataframe X and a 1d array/dataframe y, 
        fits a support vector macchine classifier

        ! It only works for binary classification
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    svc = svm.SVC(kernel=kernel, max_iter=max_iter).fit(X_train, y_train) # 10000000

    ypred = svc.predict(X_test)
    accuracy = accuracy_score(y_test, ypred)

    return accuracy, svc

def plot_svc_boundaries(ax, svc, color='k', **kwargs):
    """
        Given a matplotlib ax object and a fitted svm.SVC, it plots 
        the decision boundary. 

        To plot more boundaries you can set levels=[-1, 0, 1]
    """
    # plot the decision function
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svc.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    alpha = kwargs.pop('alpha', .8)
    lw = kwargs.pop('lw', 4)
    zorder = kwargs.pop('zorder', 120)
    levels = kwargs.pop('levels', [0])
    ax.contour(XX, YY, Z, colors=[color], levels=levels, alpha=alpha, lw=lw, zorder=zorder)


def plot_svc_boundaries_multiclass(ax, svc, h=1, cmap=None, filled=True, shadealpha=.2, linealpha=.8, **kwargs):
     # h is the step size in the mesh, lower valeus yield higher resolution

    # create a mesh to plot in
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x_min, x_max = xlim[0] - 1, xlim[1] + 1
    y_min, y_max = ylim[0] - 1, ylim[1] + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    if not filled:
        ax.contour(xx, yy, Z, cmap=cmap, alpha=linealpha, **kwargs)
    else:
        ax.contourf(xx, yy, Z, cmap=cmap, alpha=shadealpha,  zorder=-1)
        ax.contour(xx, yy, Z, colors=['k'], alpha=linealpha, zorder=99, lw=1)



