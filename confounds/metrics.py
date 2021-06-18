"""

Library of metrics for various purposes, including
quantifying the amount of association between confound and target,
degree of variability across different confound levels or groups,
degree of harmonization achieved (e.g. reduction in variance of means/medians)

"""

import numpy as np

from confounds import Residualize
from scipy import stats


def partial_correlation(X, y=None):
    """
    Calculates the partial correlation

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    C : {array-like, sparse matrix}, shape (n_samples, n_covariates)
            This does not refer to target as is typical in scikit-learn.

    Returns
    -------
    partial_correlations : ndarray
        Returns the pairwise partial correlations of each variable in X
    """
    resx = Residualize()
    resx.fit(X, y)
    deconfound_X = resx.transform(X, y)
    return np.corrcoef(deconfound_X, rowvar=False)


def prediction_partial_correlation(predictions, targets, confounds):
    """
    As seen in:
    Dinga R, Schmaal L, Penninx BW, Veltman DJ, Marquand AF. Controlling for effects of confounding variables on machine learning predictions. BioRxiv. 2020 Jan 1.

    Parameters
    ----------
    predictions : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    targets : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    confounds : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

    Returns
    -------
    corr_p : float
        The partial correlation of the predictions and targets with respect to the confounds
    t_statistic: float
        The t statistic for the statistical significance of the partial correlation
    statistical_significance: float
        The associated p value for the t statistic
    """
    n = predictions.shape[0]
    g = confounds.shape[1]
    corr_p = partial_correlation(np.hstack((predictions, targets)), confounds)[0, 1]
    t_statistic = corr_p * np.sqrt((n - 2 - g) / (1 - corr_p ** 2))
    statistical_significance = stats.t.sf(np.abs(t_statistic), df=n - 1)
    return corr_p, t_statistic, statistical_significance
