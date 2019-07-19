

Conquering confounds and covariates in machine learning
------------------------------------------------------------

This ``confounds`` package is **under development**. Contributors are most welcome.


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

 - Residualize (e.g. via regression)
 - Augment (include confounds as predictors)
 - Harmonize (correct batch effects via rescaling or normalization etc)
 - Stratify (sub- or resampling procedures to minimize confounding)
 - Utilities (Goals 1 and 3)

