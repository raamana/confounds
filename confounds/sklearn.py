#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 16:22:46 2022

@author: javi
"""

from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.utils.validation import (check_is_fitted, check_consistent_length)
from sklearn.base import BaseEstimator


class BaseDeconfEstimator(BaseEstimator):

    def __init__(self,
                 deconfounder,
                 estimator):

        self.deconfounder = deconfounder
        self.estimator = estimator

    def fit(self,
            input_data,
            target_data,
            *,
            confounders,
            sample_weight=None):
        """
        Deconfound and fit.

        Parameters
        ----------
        input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,)
            Target data.
        confounders :  array-like of shape (n_samples, n_covariates)
            Array of covariates.
        sample_weight : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        self : object
            Deconfounded and fitted Estimator.

        """

        input_data, confounders, target_data = self._validate_inputs(
            input_data,
            confounders,
            target_data
            )

        # Validate input objects
        deconfounder, estimator = self._validate_objects(self.deconfounder,
                                                         self.estimator
                                                         )

        # Deconfound input data
        deconf_input, deconf_target = self._deconfound_data(
            deconfounder, input_data, target_data, confounders,
            method="fit_transfrom"
            )
        self.deconfounder_ = deconfounder

        # Fit deconfounded input data
        estimator.fit(deconf_input, deconf_target, sample_weight)
        self.estimator_ = estimator

        return self

    def predict(self,
                input_data,
                *,
                confounders):
        """
        Deconfound and predict.

        Parameters
        ----------
        input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        confounders :  array-like of shape (n_samples, n_covariates)
            Array of covariates.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,), or shape (n_samples, n_targets)
           Vector containing the predictions for each sample.

        """

        input_data, confounders = self._validate_inputs(input_data,
                                                        confounders)

        check_is_fitted(self, attributes=["deconfounder_", "estimator_"])

        # TODO: Add an if here, in case deconfounder is a ResidualizeTarget,
        # to not deconfound the data.
        deconf_input = self.decounfounder_.transform(input_data,
                                                     confounders)

        return self.estimator_.predict(deconf_input)

    def score(self,
              input_data,
              target_data,
              *,
              confounders):
        """
        Deconfound, predict and compare with observed target data.

        Parameters
        ----------
        input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,)
            Target data.
        confounders :  array-like of shape (n_samples, n_covariates)
            Array of covariates.

        Returns
        -------
        score : float
            Mean performance of decofounded wrt. target data.

        """

        input_data, confounders = self._validate_inputs(input_data,
                                                        confounders)

        check_is_fitted(self, attributes=["decounfounder_", "estimator_"])

        deconf_input, deconf_target = self._deconfound_data(
            self.decounfounder_, input_data, target_data, confounders,
            method="transform"
            )

        return self.estimator_.score(deconf_input, deconf_target)

    def _deconfound_data(deconfounder,
                         input_data,
                         target_data,
                         confounders,
                         method
                         ):
        """
        This auxiliary function is designed to deconfound the data.
        It will choose whether to performs this on the input or target data,
        according to the deconfounder object passed
        (e.g. Residualize vs Residualize Target).

        Parameters
        ----------
        deconfounder : Deconfounder object.
            DESCRIPTION.
        input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,)
            Target data.
        confounders :  array-like of shape (n_samples, n_covariates)
            Array of covariates.
        method : str
            The method to use by the deconfounder (fit_transform or transform).

        Returns
        -------
        deconf_input_data : {array-like, sparse matrix}, \
            shape (n_samples, n_features)
            Deconfounded input data.
        deconf_target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,)
            Deconfounded target data.
        """

        deconf_input_data = getattr(deconfounder, method)(input_data,
                                                          confounders)

        return deconf_input_data, target_data

    def _validate_inputs(input_data,
                         confounders,
                         target_data):
        """
        This functions validates input data. Target data is optional
        (e.g. when transforming the data only).

        Parameters
        ----------
        input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,). Optional
        confounders :  array-like of shape (n_samples, n_covariates)
                Array of covariates.
        Returns
        -------
        input_data: {array-like, sparse matrix}, shape (n_samples, n_features)
            Validated input data.
        target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,). Optional
            Validated target data.
        confounders :  array-like of shape (n_samples, n_covariates)
            Validated array of covariates.
        """
        input_data = check_array(input_data)
        confounders = check_array(confounders)
        target_data = check_array(target_data)

        check_consistent_length(input_data, target_data, confounders)

        return input_data, target_data, confounders

    def _validate_objects(deconfounder, estimator):

        if hasattr(deconfounder, "transform") is False:
            raise ValueError(f"{deconfounder} does not seem to have "
                             "a transform method."
                             )

        if hasattr(estimator, "predict") is False:
            return ValueError("f{estimator} does not seem to have "
                              "a predict method"
                              )

        return clone(deconfounder), clone(estimator)


class DeconfClassifier(BaseEstimator):


    def fit(self,
            input_data,
            target_data,
            *,
            confounders,
            sample_weight=None):
        """
        Deconfound and fit.

        Parameters
        ----------
        input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,)
            Target data.
        confounders :  array-like of shape (n_samples, n_covariates)
            Array of covariates.
        sample_weight : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        self : object
            Deconfounded and fitted Estimator.

        """

        input_data, confounders, target_data = self._validate_inputs(
            input_data,
            confounders,
            target_data
            )

        # Validate input objects
        deconfounder, estimator = self._validate_objects(self.deconfounder,
                                                         self.estimator
                                                         )

        # Deconfound input data
        deconf_input, deconf_target = self._deconfound_data(
            deconfounder, input_data, target_data, confounders,
            method="fit_transfrom"
            )
        self.deconfounder_ = deconfounder

        # Fit deconfounded input data
        estimator.fit(deconf_input, deconf_target, sample_weight)
        self.estimator_ = estimator

        return self

    def predict(self,
                input_data,
                *,
                confounders):
        """
        Deconfound and predict.

        Parameters
        ----------
        input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        confounders :  array-like of shape (n_samples, n_covariates)
            Array of covariates.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,), or shape (n_samples, n_targets)
           Vector containing the predictions for each sample.

        """

        input_data, confounders = self._validate_inputs(input_data,
                                                        confounders)

        check_is_fitted(self, attributes=["deconfounder_", "estimator_"])

        # TODO: Add an if here, in case deconfounder is a ResidualizeTarget,
        # to not deconfound the data.
        deconf_input = self.decounfounder_.transform(input_data,
                                                     confounders)

        return self.estimator_.predict(deconf_input)

    def score(self,
              input_data,
              target_data,
              *,
              confounders):
        """
        Deconfound, predict and compare with observed target data.

        Parameters
        ----------
        input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,)
            Target data.
        confounders :  array-like of shape (n_samples, n_covariates)
            Array of covariates.

        Returns
        -------
        score : float
            Mean performance of decofounded wrt. target data.

        """

        input_data, confounders = self._validate_inputs(input_data,
                                                        confounders)

        check_is_fitted(self, attributes=["decounfounder_", "estimator_"])

        deconf_input, deconf_target = self._deconfound_data(
            self.decounfounder_, input_data, target_data, confounders,
            method="transform"
            )

        return self.estimator_.score(deconf_input, deconf_target)

    def _deconfound_data(deconfounder,
                         input_data,
                         target_data,
                         confounders,
                         method
                         ):
        """
        This auxiliary function is designed to deconfound the data.
        It will choose whether to performs this on the input or target data,
        according to the deconfounder object passed
        (e.g. Residualize vs Residualize Target).

        Parameters
        ----------
        deconfounder : Deconfounder object.
            DESCRIPTION.
        input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,)
            Target data.
        confounders :  array-like of shape (n_samples, n_covariates)
            Array of covariates.
        method : str
            The method to use by the deconfounder (fit_transform or transform).

        Returns
        -------
        deconf_input_data : {array-like, sparse matrix}, \
            shape (n_samples, n_features)
            Deconfounded input data.
        deconf_target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,)
            Deconfounded target data.
        """

        deconf_input_data = getattr(deconfounder, method)(input_data,
                                                          confounders)

        return deconf_input_data, target_data

    def _validate_inputs(input_data,
                         confounders,
                         target_data):
        """
        This functions validates input data. Target data is optional
        (e.g. when transforming the data only).

        Parameters
        ----------
        input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,). Optional
        confounders :  array-like of shape (n_samples, n_covariates)
                Array of covariates.
        Returns
        -------
        input_data: {array-like, sparse matrix}, shape (n_samples, n_features)
            Validated input data.
        target_data : array-like of shape (n_samples, n_output) \
            or (n_samples,). Optional
            Validated target data.
        confounders :  array-like of shape (n_samples, n_covariates)
            Validated array of covariates.
        """
        input_data = check_array(input_data)
        confounders = check_array(confounders)
        target_data = check_array(target_data)

        check_consistent_length(input_data, target_data, confounders)

        return input_data, target_data, confounders

    def _validate_objects(deconfounder, estimator):

        if hasattr(deconfounder, "transform") is False:
            raise ValueError(f"{deconfounder} does not seem to have "
                             "a transform method."
                             )

        if hasattr(estimator, "predict") is False:
            return ValueError("f{estimator} does not seem to have "
                              "a predict method"
                              )

        return clone(deconfounder), clone(estimator)



class DeconfoundTransform():

    def __init__(self,
                 decounfounder,
                 transformer):

        self.decounfounder = decounfounder
        self.transformer = transformer

    def fit(self,
            input_data,
            *,
            confounders):
        return NotImplementedError()

    def fit_transform(self,
                      input_data,
                      *,
                      confounders):
        return NotImplementedError()

    def transform(self,
                  input_data,
                  *,
                  confounders):
        return NotImplementedError()

    def _validate_inputs(input_data,
                         confounders):
        """
        This functions validates input data. Target data is optional
        (e.g. when transforming the data only).

        Parameters
        ----------
        input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
            Input data.
        confounders :  array-like of shape (n_samples, n_covariates)
                Array of covariates.
        Returns
        -------
        input_data: {array-like, sparse matrix}, shape (n_samples, n_features)
            Validated input data.
        confounders :  array-like of shape (n_samples, n_covariates)
            Validated array of covariates.

        """

        input_data = check_array(input_data)
        confounders = check_array(confounders)

        check_consistent_length(input_data, confounders)

        return input_data, confounders
