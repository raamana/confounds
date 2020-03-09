=====
Usage
=====

To use the ``confounds`` library in a project, import the necessary classes or functions e.g.

.. code-block:: python

    from confounds import Residualize, Augment, DummyDeconfounding


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

Here is an broader example showing how the deconfouding classes can be used in a full cross-validation loop:

.. code-block:: python

    for iter in range(num_CV_iters):

        train_data, test_data, train_targets, original_test_targets = split_dataset(your_dataset, iter)
        # split_dataset() could be train_test_split from sklearn for simple flat numpy-array based feature set
        #   (which I discourage in favour of pyradigm)
        train_X, train_C, test_X , test_C = get_covariates(train_data, test_data)

        resid = Residualize()
        resid.fit(train_X, train_C)
        # NOTE 2nd argument here are covariates, not y (targets)
        deconf_train_X = resid.transform(train_X, train_C)
        deconf_test_X  = resid.transform(test_X , test_C)

        # wrapper containing calls to GridSearchCV() etc
        # pipeline here is the predictive model, sequence of sklearn estimators
        best_pipeline = optimize_pipeline(pipeline, deconf_train_X, train_targets)
        predicted_test_targets = best_pipeline.predict(deconf_test_X)
        results[iter] = evaluate_performance(predicted_test_targets, original_test_targets)


If the above example is confusing or you disagree with it, please `let me know <https://github.com/raamana/confounds/issues/new>`_. Appreciate your help in improving the methods and documentation. Thanks much!

.. note::

    Only ``Residualize(model='linear')``, ``Augment()`` and ``DummyDeconfounding()`` are considered usable. The rest are yet to be developed, and subject to change without notice.


.. warning::

    Scikit-learn does not offer the ability to pass in covariates (or any other variables) besides ``X`` and ``y`` to their estimators. So, although classes from this ``confounds`` library act as scikit-learn estimators (passing estimator checks), **they should NOT be used in a pipeline** e.g. to pass it on ``GridSearchCV`` or similar convenience classes for optimization purposes.

.. note::

  I highly recommend using the ``pyradigm`` data structure to manage the features and covariates of a given dataset, using the classes ``ClassificationDataset`` and ``RegressionDataset``. The development version of ``pyradigm`` provides the ``MultiDatasetClassify`` and ``MultiDatasetRegress`` classes that would make this data management even easier when engaging in comparisons across multiple modalities/feature-sets on the same sample (same subjects with same covariates).
