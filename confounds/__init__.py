# -*- coding: utf-8 -*-

"""Top-level package for confounds."""

__author__ = """Pradeep Reddy Raamana"""
__email__ = 'raamana@gmail.com'

from sys import version_info


if version_info.major >= 3:
    from confounds.base import Residualize, Augment, DummyDeconfounding, \
        ConfoundsException
else:
    raise NotImplementedError('confounds library requires Python 3 or higher! ')


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
