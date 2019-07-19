#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `confounds` package."""

import pytest


from confounds.base import Augment, Residualize, DummyDeconfounding
from sklearn.datasets import make_classification

X, y = make_classification()


def test_method_does_not_introduce_bias():
    """
    Test to ensure any deconfounding method does NOT introduce bias in a sample
    when confounds not have any relationship with the target!
    """
