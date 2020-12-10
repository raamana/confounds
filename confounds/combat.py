
import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import (check_array, check_consistent_length,
                                      check_is_fitted)
from confounds.base import BaseDeconfound
from confounds.utils import get_model


class ComBat(BaseDeconfound):
    """ComBat method to remove batch effects

    """

    def __init__(self,
                 parametric=False,
                 adjust_variance=True):
        """Constructor"""

        super().__init__(name='ComBat')

    def fit(self,
            X,  # variable names chosen to correspond to sklearn when possible
            y=None,  # y is covariates, including batch variable, not the target!
            ):
        """
        Estimates the parameters for the ComBat model, based on the confounding
        variables (y) to the given [training] feature set X.  Variable names X,
        y had to be used to pass sklearn conventions. y here refers to the
        confound variables, and NOT the target. See examples in docs!

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : ndarray
            1D Array of batch identifiers, shape (n_samples, 1)
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

