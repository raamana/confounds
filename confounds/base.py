# -*- coding: utf-8 -*-

"""
Conquering confounds and covariates in machine learning


Definition of confound from Rao et al., 2017:
"For a given data sample D, a confound is a variable that affects the image data
and whose sample association with the target variable is not representative of the
population-of-interest. The sample D is then said to be biased (by the confound),
with respect to the population-of-interest.

Note that if a variable affects the image data but its association with the target
variable is representative of the population-of-interest, we would then consider
the sample to be unbiased, and the variable is not a true confound."


Other definitions used:

samplet: one row referring to single subject in the sample feature matrix X (size Nxp)

"""

from abc import ABC
from sklearn.base import BaseEstimator, TransformerMixin

class BaseDeconfound(BaseEstimator, TransformerMixin, ABC):
    """Base class for all deconfounding or covariate adjustment methods."""

    def __init__(self,
                 X, # variable names chosen to correspond to sklearn when possible
                 y,
                 confounds):
        """Constructor"""


class Augment(BaseDeconfound):
    """
    Deconfounding estimator class  that simply augments/concatenates the confounding
    variables to input features prior to prediction.
    """


class Residualize(BaseDeconfound):
    """
    Deconfounding estimator class that residualizes the input features by
    subtracting the contributions from the confound variables

    Example methods: Linear, Kernel Ridge, Gaussian Process Regression etc
    """


class Harmonize(BaseDeconfound):
    """
    Estimator to transform the input features to harmonize the input features
    across a given set of confound variables.

    Example methods include:
    Scaling (global etc)
    Normalization (Quantile, Functional etc)
    Surrogate variable analysis
    ComBat

    """


class ResidualizeTarget(BaseDeconfound):
    """
    Deconfounding estimator class that residualizes the input features by
    subtracting the contributions from the confound variables
    """


class StratifyByConfounds(BaseDeconfound):
    """
    Subsampling procedure to minimize the confound-to-target correlation.
    """


class DummyDeconfounding(BaseDeconfound):
    """
    A do-nothing dummy method, to serve as a reference for methodological comparisons
    """


