"""

Library of metrics for various purposes, including
quantifying the amount of association between confound and target,
degree of variability across different confound levels or groups,
degree of harmonization achieved (e.g. reduction in variance of means/medians)

"""

import numpy as np

from confounds import Residualize
from scipy import stats


def partial_correlation(predictions, targets, confounds, significance_test=True):
    """

    Parameters
    ----------
    predictions :
    targets :
    confounds :
    significance_test :

    Returns
    -------

    """
    res1=Residualize()
    res1.fit(predictions, confounds)
    deconfound_predictions = res1.transform(predictions, confounds)
    res2 = Residualize()
    res2.fit(targets, confounds)
    deconfound_targets = res2.transform(targets, confounds)
    partial_correlation = np.corrcoef(deconfound_predictions, deconfound_targets, rowvar=False)[0, 1]

    if significance_test:
        n = deconfound_predictions.shape[0]
        g = confounds.shape[1]
        t = partial_correlation * np.sqrt((n - 2 - g) / (1 - partial_correlation ** 2))
        statistical_significance = stats.t.sf(np.abs(t), df=n - 1)
        return partial_correlation, statistical_significance
    return partial_correlation
