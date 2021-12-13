"""

Library of metrics for various purposes, including
quantifying the amount of association between confound and target,
degree of variability across different confound levels or groups,
degree of harmonization achieved (e.g. reduction in variance of means/medians)

"""

import numpy as np
from scipy import stats

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
    resx.fit(X, y=C)
    deconfound_X = resx.transform(X, y=C)
    return np.corrcoef(deconfound_X, rowvar=False)


def partial_correlation_t_test(partial_correlations, n, g):
    """
    Calculates the t-statistic and p-value for pairwise partial correlations.

    The statistical significance of the partial correlation can be obtained
    parametrically using a Student‘s t distribution.

    This is equivalent to fitting and testing the significance of a linear
    regression model with  model predictions p and confounds C as covariates

    References
    -----------
    Dinga R, Schmaal L, Penninx BW, Veltman DJ, Marquand AF. Controlling for effects of
    confounding variables on machine learning predictions. BioRxiv. 2020 Jan 1.

    Parameters
    ----------
    partial_correlations : {array-like, sparse matrix}, shape (n_features, n_features)
            Correlations of variables with respect to some confounds
    n : {int}
            Number of samples
    g : {int}
            Number of confounding variables

    Returns
    -------
    t_statistic : ndarray
        Returns the associated t-statistics for these pairwise partial correlations
    p_value : ndarray
        Returns the associated p-values for these pairwise partial correlations
    """
    partial_correlations[partial_correlations == 1] = 1 - 1e-7
    #The degrees of freedom for the test are the number of samples minus the number of confounding variables minus 2
    df = n - 2 - g
    #We conduct a t-test and compute its significance
    t_statistic = partial_correlations * np.sqrt(df / (1 - partial_correlations ** 2))
    p_value = stats.t.sf(np.abs(t_statistic), df=df)
    return t_statistic, p_value


def prediction_partial_correlation(predictions, targets, confounds):
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

    Returns
    -------
    corr_p : float
        The partial correlation of the predictions and targets with respect to the confounds
    """
    if predictions.ndim == 1:
        predictions = predictions[:, np.newaxis]
    if targets.ndim == 1:
        targets = targets[:, np.newaxis]
    assert (predictions.shape == targets.shape), f"Dimensions of predictions and targets do not match. " \
                                                 f"predictions shape {predictions.shape}," \
                                                 f"targets shape {targets.shape}"
    p = predictions.shape[1]
    corr_p = partial_correlation(np.hstack((predictions, targets)), C=confounds)
    # Just extract the partials between predictions and associated targets
    return np.diag(corr_p[:p, p:])
