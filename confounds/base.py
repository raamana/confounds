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

from confounds.utils import get_model
from abc import ABC
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import (check_array, check_is_fitted,
                                      check_consistent_length)
import numpy as np


class ConfoundsException(BaseException):
    """Custom exception to indicate confounds-library specific issues."""


class BaseDeconfound(BaseEstimator, TransformerMixin, ABC):
    """Base class for all deconfounding or covariate adjustment methods."""

    _estimator_type = "deconfounder"


    def __init__(self, name='Deconfounder'):
        """Constructor"""

        self.name = name


    def fit(self,
            X,  # variable names chosen to correspond to sklearn when possible
            y,  # y is the confound variables here, not the target!
            ):
        """Fit method"""


    def transform(self,
                  X,  # variable names chosen to correspond to sklearn when possible
                  y,  # y is the confound variables here, not the target!
                  ):
        """Transform method"""


class Augment(BaseDeconfound):
    """
    Deconfounding estimator class  that simply augments/concatenates the confounding
    variables to input features prior to prediction.
    """


    def __init__(self):
        """Constructor"""

        super().__init__(name='Augment')

        raise NotImplementedError()


class Residualize(BaseDeconfound):
    """
    Deconfounding estimator class that residualizes the input features by
    subtracting the contributions from the confound variables

    Example methods: Linear, Kernel Ridge, Gaussian Process Regression etc
    """


    def __init__(self, model='linear'):
        """Constructor"""

        super().__init__(name='Residualize')

        self.model = model


    def fit(self,
            X,  # variable names chosen to correspond to sklearn when possible
            y=None,  # y is the confound variables here, not the target!
            ):
        """Placeholder to pass sklearn conventions"""

        return self._fit(X, y)  # which itself must return self


    def _fit(self, in_features, confounds=None):
        """Actual fit method"""

        in_features = check_array(in_features)
        confounds = check_array(confounds, ensure_2d=False)

        # turning it into 2D, in case if its just a column
        if confounds.ndim == 1:
            confounds = confounds[:, np.newaxis]

        try:
            check_consistent_length(in_features, confounds)
        except:
            raise ValueError('X (features) and y (confounds) must have the same '
                             'number rows/samplets!')

        self.n_features_ = in_features.shape[1]

        regr_model = clone(get_model(self.model))
        regr_model.fit(confounds, in_features)
        self.model_ = regr_model

        return self


    def transform(self, X, y=None):
        """Placeholder to pass sklearn conventions"""

        return self._transform(X, y)


    def _transform(self, test_features, test_confounds):
        """Actual deconfounding of the test features"""

        check_is_fitted(self, 'model_', 'n_features_')
        test_features = check_array(test_features, accept_sparse=True)

        if test_features.shape[1] != self.n_features_:
            raise ValueError('number of features must be {}. Given {}'
                             ''.format(self.n_features_, test_features.shape[1]))

        if test_confounds is None:  # during estimator checks
            return test_features  # do nothing

        test_confounds = check_array(test_confounds, ensure_2d=False)
        check_consistent_length(test_features, test_confounds)

        # test features as can be explained/predicted by their covariates
        test_feat_predicted = self.model_.predict(test_confounds)
        residuals = test_features - test_feat_predicted

        return residuals


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


    def __init__(self):
        """Constructor"""

        super().__init__(name='Harmonize')

        raise NotImplementedError()


class ResidualizeTarget(BaseDeconfound):
    """
    Deconfounding estimator class that residualizes the input features by
    subtracting the contributions from the confound variables
    """


    def __init__(self):
        """Constructor"""

        super().__init__(name='ResidualizeTarget')

        raise NotImplementedError()


class StratifyByConfounds(BaseDeconfound):
    """
    Subsampling procedure to minimize the confound-to-target correlation.
    """


    def __init__(self):
        """Constructor"""

        super().__init__(name='StratifyByConfounds')

        raise NotImplementedError()


class DummyDeconfounding(BaseDeconfound):
    """
    A do-nothing dummy method, to serve as a reference for methodological comparisons
    """


