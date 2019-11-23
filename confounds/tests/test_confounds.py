#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `confounds` package."""

from confounds.base import Residualize, DummyDeconfounding
from sklearn.utils.estimator_checks import check_estimator
from sklearn.datasets import make_classification

X, y = make_classification()


def test_estimator_API():


    for est in (Residualize, DummyDeconfounding):
        try:
            check_estimator(est)
            print('{} passes estimator checks'.format(est.__name__))
        except:
            raise


def test_method_does_not_introduce_bias():
    """
    Test to ensure any deconfounding method does NOT introduce bias in a sample
    when confounds not have any relationship with the target!
    """

