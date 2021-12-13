"""

Library of metrics for various purposes, including
quantifying the amount of association between confound and target,
degree of variability across different confound levels or groups,
degree of harmonization achieved (e.g. reduction in variance of means/medians)

"""

import numpy as np
from scipy import stats
from sklearn.utils.validation import check_array

from confounds import Residualize


def partial_correlation(X, C=None):
    """
    Calculates the pairwise partial correlations between all of the variables in X with respect to some confounding
    variables C

    Can be used as a measure of the strength of the relationship between two variables of interest while controlling
    for some other variables.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    C : {array-like, sparse matrix}, shape (n_samples, n_confounds)
            Array of confounding variables.

    Returns
    -------
    partial_correlations : ndarray
        Returns the pairwise partial correlations of each variable in X

    Examples
    ----------
    Observing the pairwise correlations of different MRI brain voxels in X in neuroimaging
    studies while controlling for the effects of the scanner type and age of participants in C.
    """
    resx = Residualize()
    resx.fit(X, C)
    deconfound_X = resx.transform(X, C)
    return np.corrcoef(deconfound_X, rowvar=False)


def partial_correlation_t_test(X, C=None):
    """
    Calculates the t-statistic and p-value for pairwise partial correlations between all of the variables in X with
    respect to some confounding variables C.

    References
    -----------
    Dinga R, Schmaal L, Penninx BW, Veltman DJ, Marquand AF. Controlling for effects of
    confounding variables on machine learning predictions. BioRxiv. 2020 Jan 1.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    C : {array-like, sparse matrix}, shape (n_samples, n_covariates)
            Array of confounding variables.

    Returns
    -------
    corr_p : ndarray
        Returns the pairwise partial correlations of each variable in X
    t_statistic : ndarray
        Returns the associated t-statistics for these pairwise partial correlations
    p_value : ndarray
        Returns the associated p-values for these pairwise partial correlations
    """
    X = check_array(X)
    C = check_array(C)
    g = C.shape[1]
    corr_p = partial_correlation(X, C=C)
    n = X.shape[0]
    # Replace perfect correlations to ensure large but not infinite t statistic
    corr_p[corr_p == 1] = 1 - 1e-7
    # partial correlation degrees of freedom
    df = n - 2 - g
    t_statistic = corr_p * np.sqrt(df / (1 - corr_p ** 2))
    p_value = stats.t.sf(np.abs(t_statistic), df=df)
    return corr_p, t_statistic, p_value


def prediction_partial_correlation(predictions, targets, confounds, t_test=False):
    """
    Returns the partial correlation between predictions and targets after residualizing the effect of confounds.
    Also calculates the t-statistic and p-value for the statistical significance of this partial correlation which
    is a measure of the predictive power of the model controlling for the effect of confounds.

    References
    -----------
    Dinga R, Schmaal L, Penninx BW, Veltman DJ, Marquand AF. Controlling for effects of
    confounding variables on machine learning predictions. BioRxiv. 2020 Jan 1.

    Parameters
    ----------
    predictions : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    targets : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    confounds : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    t_test : {bool} whether to return return t-test value and statistical significance

    Returns
    -------
    corr_p : float
        The partial correlation of the predictions and targets with respect to the confounds
    t_statistic: float
        The t statistic for the statistical significance of the partial correlation
    p_value: float
        The associated p value for the t statistic
    """
    if np.ndim(predictions) == 1:
        p = 1
    else:
        p = predictions.shape[1]
    corr_p = partial_correlation(np.hstack((predictions, targets)), confounds)
    if t_test:
        _, t_statistic, p_value = partial_correlation_t_test(
            np.stack((predictions, targets), axis=1),
            confounds)
        return np.diag(corr_p[:p, p:]), np.diag(t_statistic[:p, p:]), np.diag(p_value[:p, p:])
    # Just extract the partials between predictions and associated targets
    return np.diag(corr_p[:p, p:])
