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
    Calculates the pairwise partial correlations between all of the variables in X with respect to some confounding variables y

    Can be used as a measure of the strength of the relationship between two variables of interest while controlling for some other variables.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    y : {array-like, sparse matrix}, shape (n_samples, n_covariates)
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


def partial_correlation_t_test(X, y=None):
    """
    Calculates the t-statistic and p-value for pairwise partial correlations between all of the variables in X with
    respect to some confounding variables y.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    y : {array-like, sparse matrix}, shape (n_samples, n_covariates)
            This does not refer to target as is typical in scikit-learn.

    Returns
    -------
    corr_p : ndarray
        Returns the pairwise partial correlations of each variable in X
    t_statistic : ndarray
        Returns the associated t-statistics for these pairwise partial correlations
    statistical_significance : ndarray
        Returns the associated p-values for these pairwise partial correlations
    """
    corr_p = partial_correlation(X, y=None)
    n = X.shape[0]
    g = y.shape[1]
    # Replace perfect correlations to ensure large but not infinite t statistic
    corr_p[corr_p == 1] = 1 - 1e-7
    # partial correlation degrees of freedom
    df = n - 2 - g
    t_statistic = corr_p * np.sqrt(df / (1 - corr_p ** 2))
    statistical_significance = stats.t.sf(np.abs(t_statistic), df=df)
    return corr_p, t_statistic, statistical_significance


def prediction_partial_correlation(predictions, targets, confounds):
    """
    Returns the partial correlation between predictions and targets after residualizing the effect of confounds.
    Also calculates the t-statistic and p-value for the statistical significance of this partial correlation which
    is a measure of the predictive power of the model controlling for the effect of confounds.

    References
    -----------
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
    p = predictions.shape[1]
    # Just extract the partials between predictions and associated targets
    corr_p, t_statistic, statistical_significance = np.diag(
        partial_correlation_t_test(np.hstack((predictions, targets)), confounds)[:p, p:])
    return corr_p, t_statistic, statistical_significance
