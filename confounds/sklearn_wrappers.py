"module with some sklearn wrappers"
import numpy as np
from joblib import Parallel, delayed

from sklearn.base import clone, is_classifier
from sklearn.utils import check_array
from sklearn.utils.validation import (check_is_fitted, check_consistent_length)
from sklearn.base import BaseEstimator
from sklearn.model_selection._split import check_cv
from sklearn.metrics import check_scoring

from confounds.base import Augment, DummyDeconfounding, Residualize


class DeconfEstimator(BaseEstimator):
    """
    Estimator with a previous deconfounding of input data.

    Wrapper that first deconfounds the input data according to a supplied
    strategy and then run a passed estimator.

    Parameters
    ----------
    deconfounder : deconfounder object
        This is assumed to be one the deconfounders that can be found
        in this library. Right now only it works for input independent data
        deconfounding, that is Augment, DummyDeconfounding, Residualize.
    estimator : estimator object
        This is assumed to implement a scikit-learn estimator interface.

    """

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
        deconfounder.fit(input_data, confounders)
        self.deconfounder_ = deconfounder

        input_data = self.deconfounder_.transform(input_data, confounders)

        # Fit deconfounded input data
        estimator.fit(input_data, target_data, sample_weight)
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

        deconf_input = self.deconfounder_.transform(input_data, confounders)

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

        input_data, confounders, target_data = self._validate_inputs(
            input_data, confounders, target_data)

        check_is_fitted(self, attributes=["deconfounder_", "estimator_"])

        deconf_input = self.deconfounder_.transform(input_data, confounders)

        return self.estimator_.score(deconf_input, target_data)

    def _validate_inputs(self,
                         input_data,
                         confounders,
                         target_data="no_validation"):
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

        confounders = check_array(confounders)
        check_consistent_length(input_data, confounders)

        y = target_data
        no_val_y = y is None or isinstance(y, str) and y == "no_validation"

        if no_val_y:
            input_data = check_array(input_data)
            out = input_data, confounders
        else:
            input_data = check_array(input_data)
            target_data = check_array(target_data, ensure_2d=False)
            check_consistent_length(input_data, target_data)
            out = input_data, confounders, target_data

        return out

    def _validate_objects(self, deconfounder, estimator):

        valid_decfs = (Augment, DummyDeconfounding, Residualize)
        if isinstance(deconfounder, valid_decfs) is False:
            raise print(f"deconfounder should be one among {valid_decfs} "
                        f"but {deconfounder} was passed")

        if hasattr(deconfounder, "transform") is False:
            raise ValueError(f"{deconfounder} does not seem to have "
                             "a transform method."
                             )

        if hasattr(estimator, "predict") is False:
            return ValueError("f{estimator} does not seem to have "
                              "a predict method"
                              )

        return clone(deconfounder), clone(estimator)


# class DeconfoundTransform():

#     def __init__(self,
#                  decounfounder,
#                  transformer):

#         self.decounfounder = decounfounder
#         self.transformer = transformer

#     def fit(self,
#             input_data,
#             *,
#             confounders):
#         return NotImplementedError()

#     def fit_transform(self,
#                       input_data,
#                       *,
#                       confounders):
#         return NotImplementedError()

#     def transform(self,
#                   input_data,
#                   *,
#                   confounders):
#         return NotImplementedError()

#     def _validate_inputs(input_data,
#                          confounders):
#         """
#         This functions validates input data. Target data is optional
#         (e.g. when transforming the data only).

#         Parameters
#         ----------
#         input_data : {array-like, sparse matrix}, shape (n_samples, n_features)
#             Input data.
#         confounders :  array-like of shape (n_samples, n_covariates)
#                 Array of covariates.
#         Returns
#         -------
#         input_data: {array-like, sparse matrix}, shape (n_samples, n_features)
#             Validated input data.
#         confounders :  array-like of shape (n_samples, n_covariates)
#             Validated array of covariates.

#         """

#         input_data = check_array(input_data)
#         confounders = check_array(confounders)

#         check_consistent_length(input_data, confounders)

#         return input_data, confounders


def deconfounded_cv_predict(
        estimator,
        X,
        y=None,
        *,
        confounds,  # Mandatory, and better (IMHO) to be passed by key.
        groups=None,
        cv=None,
        n_jobs=None,
        verbose=0,
        fit_params=None,
        pre_dispatch="2*n_jobs",
        #  method="predict" TODO: implement more cases (prob, log_prob, etc)
        ):
    """Deconfound, fit estimator and predict values in a cross-validation

    Parameters
    ----------
    estimator : DeconfEstimator from this module.
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples, n_output) \
        or (n_samples,)
        Target data.
    C :  array-like of shape (n_samples, n_covariates)
        Array of covariates.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    Returns
    -------
    predictions : ndarray
        predictions
    """

    if isinstance(estimator, DeconfEstimator) is False:
        raise ValueError(f"estimator should be a DeconfEstimator instance, "
                         f"but {estimator} was passed instead"
                         )

    cv = check_cv(cv, y,
                  classifier=is_classifier(getattr(estimator, "estimator"))
                  )

    # Code borrowed from sklearn
    splits = list(cv.split(X, y, groups))
    test_indices = np.concatenate([test for _, test in splits])

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)

    # Here we call there our implementation of _fit_and_predict.
    predictions = parallel(
        delayed(_deconf_fit_and_predict)(
            clone(estimator), X, y, confounds,
            train, test, verbose, fit_params
        )
        for train, test in splits
    )

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    predictions = np.concatenate(predictions)

    return predictions[inv_test_indices]


def _deconf_fit_and_predict(
        estimator,
        X,
        y,
        C,  # Confounders
        train,
        test,
        verbose,
        fit_params,
        method="predict"):
    """Deconfound, fit estimator and predict values for a given dataset split.

    Parameters
    ----------
    estimator : DeconfEstimator from this module.
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
        .. versionchanged:: 0.20
            X is only required to be an object with finite length or shape now
    y : array-like of shape (n_samples, n_output) \
        or (n_samples,)
        Target data.
    C :  array-like of shape (n_samples, n_covariates)
        Array of covariates.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    method : str
        Invokes the passed method name of the passed estimator. For not it only
        predict the labels. Future inplementations will include other methods
        (e.g. prob, log_prob, etc)
    Returns
    -------
    predictions : ndarray
        predictions
    """

    # Split into training and test sets
    X_train, y_train, C_train = X[train], y[train], C[train]
    X_test, C_test = X[test], C[test]

    # N.B. estimator should be our sklearn wrapper. We should require this
    # during the initial checks of cross_val_predict
    estimator.fit(X_train, y_train, confounders=C_train)
    predictions = getattr(estimator, method)(X_test, confounders=C_test)

    return predictions


def deconfounded_cv_score(
        estimator,
        X,
        y=None,
        *,
        confounds,  # Mandatory, and better (IMHO) to be passed by key.
        groups=None,
        scoring=None,
        cv=None,
        n_jobs=None,
        verbose=0,
        fit_params=None,
        pre_dispatch="2*n_jobs"):
    """Deconfound, fit estimator and yield values for a cross-validation.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
        .. versionchanged:: 0.20
            X is only required to be an object with finite length or shape now
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    C :  array-like of shape (n_samples, n_covariates)
            Array of covariates.
    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.
        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.
        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

     Returns
    -------
    scores : ndarray of float of shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.

    """

    if isinstance(estimator, DeconfEstimator) is False:
        raise ValueError(f"estimator should be a DeconfEstimator instance, "
                         f"but {estimator} was passed instead"
                         )

    cv = check_cv(cv, y,
                  classifier=is_classifier(getattr(estimator, "estimator"))
                  )

    parallel = Parallel(n_jobs=n_jobs,
                        verbose=verbose,
                        pre_dispatch=pre_dispatch)

    scorer = check_scoring(getattr(estimator, "estimator"), scoring=scoring)

    scores = parallel(
        delayed(_deconf_fit_and_score)(
            clone(estimator), X, y, confounds, scorer,
            train, test, verbose, fit_params
            )
        for train, test in cv.split(X, y, groups)
        )

    return scores


def _deconf_fit_and_score(
        estimator,
        X,
        y,
        C,  # Confounders
        scorer,
        train,
        test,
        verbose,
        fit_params):
    """Deconfound, fit estimator and yield scores values for a given split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
        .. versionchanged:: 0.20
            X is only required to be an object with finite length or shape now
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    C :  array-like of shape (n_samples, n_covariates)
            Array of covariates.
    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.
        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.
        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

     Returns
    -------
    scores : ndarray of float of shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.

    """

    # Split into training and test sets
    X_train, y_train, C_train = X[train], y[train], C[train]
    X_test, y_test, C_test = X[test], y[test], C[test]

    # N.B. estimator should be our sklearn wrapper. We should require this
    # during the initial checks of cross_val_predict
    estimator.fit(X_train, y_train, confounders=C_train)

    X_deconf_test = estimator.deconfounder_.transform(X_test, C_test)
    score = scorer(estimator.estimator_, X_deconf_test, y_test)

    return score
