#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `confounds` package."""

import os

import numpy as np

from confounds.combat import ComBat


def test_method_does_not_introduce_bias():
    """
    Test to ensure any deconfounding method does NOT introduce bias in a sample
    when confounds not have any relationship with the target!
    """


def test_combat():
    """Test to check that Combat effectively removes batchs effects."""

    from scipy.stats import f_oneway, bartlett, pearsonr

    rs = np.random.RandomState(0)

    n_subj_per_batch = 50
    n_features = 2

    # One effect of interest that we want to keep
    X = rs.normal(size=(n_subj_per_batch + n_subj_per_batch,))

    # Global mean and noise parameters
    grand_mean = np.array([0.3, 0.2])
    eps = rs.normal(scale=1e-5, size=(n_subj_per_batch * 2, n_features))

    # Batch parameters
    batch = [1] * n_subj_per_batch + [2] * n_subj_per_batch
    shift_1 = np.array([0.05, -0.1])
    shift_2 = np.array([-0.3, 0.15])
    eps_1 = rs.normal(scale=1e-1, size=(n_subj_per_batch, n_features))
    eps_2 = rs.normal(scale=0.4, size=(n_subj_per_batch, n_features))

    # Construct dataset
    Y = np.zeros(shape=(n_subj_per_batch + n_subj_per_batch, 2))

    # Add locations and and scales per batch
    Y[:n_subj_per_batch, :] = shift_1 + eps_1
    Y[n_subj_per_batch:, :] = shift_2 + eps_2
    # Add grand mean
    Y += grand_mean

    # Add dependence with the effect of interest
    Y[:, 0] += 0.2 * X
    Y[:, 1] += -0.16 * X

    # Add global noise
    Y += eps

    # Test that both batches have different means
    p_loc_before = np.array([f_oneway(y[:n_subj_per_batch],
                                      y[n_subj_per_batch:])[1]
                             for y in Y.T]
                            )
    assert np.all(p_loc_before < 0.05)

    # Test that both batches have different variances
    p_scale_before = np.array([bartlett(y[:n_subj_per_batch],
                                        y[n_subj_per_batch:])[1]
                               for y in Y.T]
                              )
    assert np.all(p_scale_before < 0.05)

    # Test that there's a significant effect with X
    p_effects_before = np.array([pearsonr(y, X)[1] for y in Y.T])
    assert np.all(p_effects_before < 0.05)

    combat = ComBat()
    Y_trans = combat.fit_transform(in_features=Y,
                                   batch=batch,
                                   effects_interest=X.reshape(-1, 1)
                                   )
    # Test that batches no longer have different means
    p_loc_after = np.array([f_oneway(y[:n_subj_per_batch],
                                     y[n_subj_per_batch:])[1]
                            for y in Y_trans.T]
                           )
    assert np.all(p_loc_after > 0.05)
    # Test that batches no longer have different variances
    p_scale_after = np.array([bartlett(y[:n_subj_per_batch],
                                       y[n_subj_per_batch:])[1]
                              for y in Y_trans.T]
                             )
    assert np.all(p_scale_after > 0.05)

    # Test that there is still a significant effect with X after harmonisation
    p_effects_after = np.array([pearsonr(y, X)[1] for y in Y_trans.T])
    assert np.all(p_effects_after < 0.05)


def test_combat_bladder():
    """Test to check that Combat effectively removes batchs effects using
    the bladder cancer data used in the R package "SVA"
    https://rdrr.io/bioc/sva/src/tests/testthat/test_combat_bladderbatch_parallel.R
    """

    from sklearn.preprocessing import OneHotEncoder

    bladder_file = os.path.join(os.path.dirname(__file__),
                                "data", "bladder_test.npz")
    bladder_test = np.load(bladder_file)

    in_features = bladder_test['Y']
    batch = bladder_test['batch']
    effects_interest = bladder_test['effects_interest']
    # Categorical features should one hot encoded, dropping the first column
    # I personally prefer to use pandas get dummies for this, but here
    # we used scikit-learn to avoid a new dependence on pandas.
    ohe = OneHotEncoder(drop="first", sparse=False)
    effects_interest = ohe.fit_transform(effects_interest.reshape(-1, 1))

    combat = ComBat()

    Y_combat = combat.fit_transform(in_features=in_features, batch=batch)
    assert np.allclose(Y_combat, bladder_test['Y_combat'])

    Y_combat_effects = combat.fit_transform(in_features=in_features,
                                            batch=batch,
                                            effects_interest=effects_interest)
    assert np.allclose(Y_combat_effects, bladder_test['Y_combat_effects'])
