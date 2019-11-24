#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `confounds` package."""

from confounds.base import Residualize, DummyDeconfounding, Augment
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import make_classification
import numpy as np

def test_estimator_API():


    for est in (Residualize, Augment, DummyDeconfounding):
        try:
            check_estimator(est)
            print('{} passes estimator checks'.format(est.__name__))
        except:
            raise

def test_augment():

    max_dim = 100
    for num_confounds in np.random.randint(1, max_dim, 3):

        X_all, y = make_classification(n_features=max_dim+10)

        X = X_all[:, :-num_confounds]
        confounds = X_all[:, -num_confounds:]

        aug = Augment()
        aug.fit(X, confounds)
        X_aug = aug.transform(X, confounds)
        assert np.all(X_aug==X_all)


def test_method_does_not_introduce_bias():
    """
    Test to ensure any deconfounding method does NOT introduce bias in a sample
    when confounds not have any relationship with the target!
    """

