#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `confounds` package."""

import os

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.stats import bartlett, f_oneway, pearsonr
from sklearn.datasets import make_classification, make_sparse_uncorrelated
from sklearn.utils.estimator_checks import check_estimator

from confounds.base import Augment, DummyDeconfounding, Residualize
from confounds.combat import ComBat


def test_estimator_API():
    estimators = [Residualize(), Augment(), DummyDeconfounding()]
    for est in estimators:
        try:
            check_estimator(est)
            print('{} passes estimator'
                  ' checks'.format(est.__getattribute__("name")))
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

            train_X, train_confounds = splitter_X_confounds(train_all,
                                                            num_confounds)

            resid = Residualize(model='linear')
            resid.fit(train_X, train_confounds)

            residual_train_X = resid.transform(train_X, train_confounds)

            # residual_train_X and train_confounds must be orthogonal now!
            assert_almost_equal(residual_train_X.T.dot(train_confounds), 0)


def test_method_does_not_introduce_bias():
    """
    Test to ensure any deconfounding method does NOT introduce bias in a sample
    when confounds not have any relationship with the target!
    """


def generate_dataset_with_confounding(n_subj_per_batch, n_features):
    """data generator with known relationships"""

    rs = np.random.RandomState(0)

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

    return Y, X, batch


def test_combat():
    """Test to check that Combat effectively removes batchs effects."""

    n_subj_per_batch = 50  # parametrize the tests over # subjects and # features
    n_features = 2

    Y, X, batch = generate_dataset_with_confounding(n_subj_per_batch, n_features)

    combat = ComBat()

    # testing on the same dataset
    Y_trans = combat.fit_transform(in_features=Y,
                                   batch=batch,
                                   effects_interest=X.reshape(-1, 1)
                                   )

    # TODO split the data into training and testing
    #   estimate ComBat on training
    #   apply it on the test set

    # Test that batches no longer have different means
    p_loc_after = np.array([f_oneway(y[:n_subj_per_batch],
                                     y[n_subj_per_batch:])[1]
                            for y in Y_trans.T]
                           )
    if not np.all(p_loc_after > 0.05):
        raise ValueError('batch means differ significantly after applying ComBat!')

    # Test that batches no longer have different variances
    p_scale_after = np.array([bartlett(y[:n_subj_per_batch],
                                       y[n_subj_per_batch:])[1]
                              for y in Y_trans.T]
                             )
    if not np.all(p_scale_after > 0.05):
        raise ValueError('batch variances differ significantly after applying '
                         'ComBat!')

    # Test that there is still a significant effect with X after harmonisation
    p_effects_after = np.array([pearsonr(y, X)[1] for y in Y_trans.T])
    assert np.all(p_effects_after < 0.05)


def test_equivalence_to_R_impl_SVA_on_bladder_dataset():
    """Test to check that Combat effectively removes batchs effects using
    the bladder cancer data used in the R package "SVA"
    https://rdrr.io/bioc/sva/src/tests/testthat/test_combat_bladderbatch_parallel.R
    """

    from sklearn.preprocessing import OneHotEncoder

    bladder_file = os.path.join(os.path.dirname(__file__),
                                "data", "bladder_test.npz")
    bladder_test = np.load(bladder_file)

    in_features = bladder_test['in_feat']
    batch = bladder_test['batch']
    effects_interest = bladder_test['effects_interest']
    # Categorical features should one hot encoded, dropping the first column
    # I personally prefer to use pandas get dummies for this, but here
    # we used scikit-learn to avoid a new dependence on pandas.
    ohe = OneHotEncoder(drop="first", sparse=False)
    effects_interest = ohe.fit_transform(effects_interest.reshape(-1, 1))

    combat = ComBat()

    in_feat_combat = combat.fit_transform(in_features=in_features, batch=batch)
    assert np.allclose(in_feat_combat, bladder_test['in_feat_combat'])

    in_feat_combat_effects = combat.fit_transform(in_features=in_features,
                                                  batch=batch,
                                                  effects_interest=effects_interest)
    assert np.allclose(in_feat_combat_effects,
                       bladder_test['in_feat_combat_effects'])


def test_combat_batch_data_types():
    """
    Ensure it works for right and expected data types;
    And also it generates an error for incorrect and unexpected data types!
    """

    # test1 re batch: it can be both numerical and categorical, but not floating


test_combat()
