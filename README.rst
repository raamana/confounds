

Conquering confounds and covariates in machine learning
------------------------------------------------------------

.. image:: https://img.shields.io/pypi/v/confounds.svg
        :target: https://pypi.python.org/pypi/confounds

.. image:: https://img.shields.io/travis/raamana/confounds.svg
        :target: https://travis-ci.org/raamana/confounds

.. image:: https://readthedocs.org/projects/confounds/badge/?version=latest
        :target: https://confounds.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status



Vision / Goals
~~~~~~~~~~~~~~~

The high-level goals of this package is to develop high-quality library to conquer confounds and covariates in ML applications. By conquering, we mean methods and tools to

 1. visualize and establish the presence of confounds (e.g. quantifying confound-to-target relationships),
 2. offer solutions to handle them appropriately via correction or removal etc, and
 3. analyze the effect of the deconfounding methods in the processed data (e.g. ability to check if they worked at all, or if they introduced new or unwanted biases etc).


Methods
~~~~~~~~

Available:

 - Residualize (e.g. via regression)
 - Augment (include confounds as predictors)

To be added:

 - Harmonize (correct batch effects via rescaling or normalization etc)
 - Stratify (sub- or resampling procedures to minimize confounding)
 - Utilities (Goals 1 and 3)


Example usage
~~~~~~~~~~~~~~

Let's say the features from your dataset are contained in ``X`` (N x p), and the confounds/covariates for those samples stored in ``C`` (of size N x c), with X and C having row-wise correspondence for each samplet.

X is often split into ``train_X`` and ``test_X`` inside the cross-validation loop, and the corresponding splits for C are ``train_C`` and ``test_C``. Then, the estimator classes (e.g. ``Residualize()``) from ``confounds`` library are used in the following **easy** manner:


.. code-block:: python

        resid = Residualize()          # instantiation
        # NOTE the second argument to deconfounding instance are confounds/covariate variables, not y (targets)
        resid.fit(train_X, train_C)    # training on X and C

        # NOTE the second argument to the instance are confounds/covariate variables, not y (targets)
        deconf_train_X = resid.transform(train_X, train_C)     # deconfounding X, N
        deconf_test_X  = resid.transform(test_X , test_C)


That's it.

You can also replace ``Residualize()`` with ``Augment()`` (which simply concatenates the covariates values to X horizontally) as well as ``DummyDeconfounding()`` which does nothing but return X back.



.. note::

    Only ``Residualize(model='linear')``, ``Augment()`` and ``DummyDeconfounding()`` are considered usable. The rest are yet to be developed, and subject to change without notice.


.. warning::

    Scikit-learn does not offer the ability to pass in covariates (or any other variables) besides ``X`` and ``y`` to their estimators. So, although classes from this ``confounds`` library act as scikit-learn estimators (passing estimator checks), **they should NOT be used in a pipeline** e.g. to pass it on ``GridSearchCV`` or similar convenience classes for optimization purposes.


Contributors are most welcome.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


``confounds`` package is beta and **under development**.
