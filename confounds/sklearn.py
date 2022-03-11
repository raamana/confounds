#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:22:46 2022

@author: javi
"""

from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.utils.validation import (check_is_fitted, check_consistent_length)


class DeconfoundEstimate():

    def __init__(self,
                 decounfounder,
                 estimator):

        self.decounfounder = decounfounder
        self.estimator = estimator

    def fit(self,
            input_data,
            target_data,
            *,
            confounders,
            sample_weight=None):

        input_data, confounders, target_data = self._validate_inputs(
            input_data,
            confounders,
            target_data
            )

        # clone input arguments
        decounfounder = clone(self.decounfounder)
        estimator = clone(self.estimator)

        # Deconfound input data
        deconf_input = decounfounder.fit_transform(input_data, confounders)
        self.decounfounder_ = decounfounder

        # Fit deconfounded input data
        estimator.fit(deconf_input, target_data, sample_weight)
        self.estimator_ = estimator

        return self

    def predict(self,
                input_data,
                *,
                confounders):

        input_data, confounders = self._validate_inputs(input_data,
                                                        confounders)

        check_is_fitted(self, attributes=["decounfounder_", "estimator_"])

        deconf_input = self.decounfounder_.transform(input_data,
                                                     confounders)

        return self.estimator_.predict(deconf_input)

    def _validate_inputs(input_data,
                         confounders,
                         target_data=None,):

        input_data = check_array(input_data)
        confounders = check_array(confounders)

        check_consistent_length(input_data, target_data, confounders)

        if target_data:
            target_data = check_array(target_data)
            check_consistent_length(input_data, target_data)
            return input_data, confounders, target_data
        else:
            return input_data, confounders
