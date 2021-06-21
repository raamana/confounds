#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `confounds` package."""

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_less
from sklearn.datasets import make_classification, make_sparse_uncorrelated
from sklearn.utils.estimator_checks import check_estimator

from confounds.base import Augment, DummyDeconfounding, Residualize
from confounds.metrics import partial_correlation, partial_correlation_t_test, prediction_partial_correlation
from sklearn.linear_model import LinearRegression


def test_estimator_API():
    for est in (Residualize, Augment, DummyDeconfounding):
        try:
            check_estimator(est)
            print('{} passes estimator checks'.format(est.__name__))
        except:
            raise


def splitter_X_confounds(X_whole, num_confounds):
    """Returns the last num_confounds columns as separate array"""
    X = X_whole[:, :-num_confounds]
    confounds = X_whole[:, -num_confounds:]
    return X, confounds


def test_augment():
    max_dim = 100
    for num_confounds in np.random.randint(1, max_dim, 3):
        X_all, y = make_classification(n_features=max_dim + 10)

        X = X_all[:, :-num_confounds]
        confounds = X_all[:, -num_confounds:]

        aug = Augment()
        aug.fit(X, confounds)
        X_aug = aug.transform(X, confounds)
        assert np.all(X_aug == X_all)


def test_residualize_linear():
    """sanity checks on implementation"""

    min_dim = 6  # atleast 4+ required for make_sparse_uncorrelated
    max_dim = 100
    for n_samples in np.random.randint(20, 500, 3):
        for num_confounds in np.random.randint(min_dim, max_dim, 3):
            train_all, train_y = make_sparse_uncorrelated(
                n_samples=n_samples, n_features=min_dim + num_confounds + 1)

            train_X, train_confounds = splitter_X_confounds(train_all, num_confounds)

            resid = Residualize(model='linear')
            resid.fit(train_X, train_confounds)

            residual_train_X = resid.transform(train_X, train_confounds)

            # residual_train_X and train_confounds must be orthogonal now!
            assert_almost_equal(residual_train_X.T.dot(train_confounds), 0)


def test_partial_correlation():
    """check that partial correlations are less than correlations"""
    n_samples = 100
    train_all, train_y = make_classification(n_features=5)
    train_X, train_confounds = splitter_X_confounds(train_all, 1)
    # check that partial correlation with no confounds is the same as correlation using np.corrcoef
    assert_almost_equal(np.corrcoef(train_X, rowvar=False),
                        partial_correlation(train_X, np.zeros((train_X.shape[0], 1))))
    # check that a linear regression fit using all variables has a lower r**2 partial correlation.
    lr = LinearRegression().fit(train_all, train_y)
    pred = lr.predict(train_all)
    corr_p, t_stat, p_val = prediction_partial_correlation(pred,train_y,train_confounds)
    assert_array_less(corr_p,lr.score(train_all,train_y))
    print()


def test_method_does_not_introduce_bias():
    """
    Test to ensure any deconfounding method does NOT introduce bias in a sample
    when confounds not have any relationship with the target!
    """
