#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

requirements = ['numpy', 'scikit-learn', 'seaborn']
setup_requirements = ['pytest-runner', 'setuptools']
test_requirements = ['pytest', ] + requirements


setup(
    author="Pradeep Reddy Raamana",
    author_email='raamana@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="""Conquering confounds and covariates in machine learning

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

""",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description="conquering confounds and covariates in machine learning",
    include_package_data=True,
    keywords='confounds',
    name='confounds',
    packages=find_packages(include=['confounds']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/raamana/confounds',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
