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

samplet: one row referring to single subject in sample feature matrix X (size Nxp )

"""

from abc import ABC

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import (check_array, check_consistent_length,
                                      check_is_fitted)

from confounds.utils import get_model


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

        # this class has no parameters


    def fit(self,
            X,  # variable names chosen to correspond to sklearn when possible
            y=None,  # y is the confound variables here, not the target!
            ):
        """
        Learns the dimensionality of confounding variables to be augmented.

        Variable names X, y had to be used to pass sklearn conventions. y here
        refers to the confound variables, and NOT the target. See examples in docs!

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : ndarray
            Array of covariates, shape (n_samples, n_covariates)
            This does not refer to target as is typical in scikit-learn.

        Returns
        -------
        self : object
            Returns self
        """

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

        return self


    def transform(self, X, y=None):
        """
        Transforms the given feature set by augmenting the confounding variables.

        Variable names X, y had to be used to pass sklearn conventions. y here
        refers to the confound variables for the [test] to be transformed, and NOT
        their target values. See examples in docs!

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : ndarray
            Array of covariates, shape (n_samples, n_covariates)
            This does not refer to target as is typical in scikit-learn.

        Returns
        -------
        self : object
            Returns self
        """

        return self._transform(X, y)


    def _transform(self, test_features, test_confounds):
        """Actual deconfounding of the test features"""

        check_is_fitted(self, ('n_features_', ))
        test_features = check_array(test_features, accept_sparse=True)

        if test_features.shape[1] != self.n_features_:
            raise ValueError('number of features must be {}. Given {}'
                             ''.format(self.n_features_, test_features.shape[1]))

        if test_confounds is None:  # during estimator checks
            return test_features  # do nothing

        test_confounds = check_array(test_confounds, ensure_2d=False)
        check_consistent_length(test_features, test_confounds)

        return np.column_stack((test_features, test_confounds))


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
        """
        Fits the residualizing model (estimates the contributions of confounding
        variables (y) to the given [training] feature set X.  Variable names X,
        y had to be used to pass sklearn conventions. y here refers to the
        confound variables, and NOT the target. See examples in docs!

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : ndarray
            Array of covariates, shape (n_samples, n_covariates)
            This does not refer to target as is typical in scikit-learn.

        Returns
        -------
        self : object
            Returns self
        """

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
            raise ValueError('X (features) and y (confounds) '
                             'must have the same number of rows/samplets!')

        self.n_features_ = in_features.shape[1]

        regr_model = clone(get_model(self.model))
        regr_model.fit(confounds, in_features)
        self.model_ = regr_model

        return self


    def transform(self, X, y=None):
        """
        Transforms the given feature set by residualizing the [test] features
        by subtracting the contributions of their confounding variables.

        Variable names X, y had to be used to pass scikit-learn conventions. y here
        refers to the confound variables for the [test] to be transformed,
        and NOT their target values. See examples in docs!

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : ndarray
            Array of covariates, shape (n_samples, n_covariates)
            This does not refer to target as is typical in scikit-learn.

        Returns
        -------
        self : object
            Returns self
        """

        return self._transform(X, y)


    def _transform(self, test_features, test_confounds):
        """Actual deconfounding of the test features"""

        check_is_fitted(self, ('model_', 'n_features_'))
        test_features = check_array(test_features, accept_sparse=True)

        if test_features.shape[1] != self.n_features_:
            raise ValueError('number of features must be {}. Given {}'
                             ''.format(self.n_features_, test_features.shape[1]))

        if test_confounds is None:  # during estimator checks
            return test_features  # do nothing

        test_confounds = check_array(test_confounds, ensure_2d=False)
        if test_confounds.ndim == 1:
            test_confounds = test_confounds[:, np.newaxis]
        check_consistent_length(test_features, test_confounds)

        # test features as can be explained/predicted by their covariates
        test_feat_predicted = self.model_.predict(test_confounds)
        residuals = test_features - test_feat_predicted

        return residuals


class ResidualizeTarget(BaseDeconfound):
    """
    Deconfounding estimator class that residualizes the input features by
    subtracting the contributions from the confound variables
    """


    def __init__(self):
        """Constructor"""

        super().__init__(name='ResidualizeTarget')

        raise NotImplementedError()


class DummyDeconfounding(BaseDeconfound):
    """
    A do-nothing dummy method, to serve as a reference for methodological comparisons
    """


    def __init__(self):
        """Constructor"""

        super().__init__(name='DummyPassThrough')


    def fit(self, X, y=None):
        """
        A do-nothing fit method.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : ndarray
            Array of covariates, shape (n_samples, n_covariates)

        Returns
        -------
        self : object
            Returns self.
        """

        X = check_array(X, accept_sparse=True)
        self.n_features_ = X.shape[1]

        return self


    def transform(self, X, y=None):
        """
         A do-nothing transform method.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        y : ndarray
            Array of covariates, shape (n_samples, n_covariates)

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            Same as the input ``X``.

        """

        check_is_fitted(self, ('n_features_', ))
        X = check_array(X, accept_sparse=True)

        if X.shape[1] != self.n_features_:
            raise ValueError('num_features differ between fit and transform!')

        return X  # dummy pass-through, doing nothing except for shape checks.
