import numpy as np

from sklearn.utils.validation import (check_array, check_consistent_length,
                                      check_is_fitted, column_or_1d)
from confounds.base import BaseDeconfound


class ComBat(BaseDeconfound):
    """ComBat method to remove batch effects."""

    def __init__(self,
                 # parametric=True, # TODO: When implemented non-parametric
                 # adjust_variance=True, # TODO: When implemented only mean
                 tol=1e-4):
        """Initiate object."""
        super().__init__(name='ComBat')
        # self.parametric = True
        # self.adjust_variance = True
        self.tol = tol

    def fit(self,
            in_features,  # input features you want to harmonize
            batch,  # batch variable indicating site/scanner
            effects_interest=None  #
            ):
        """
        Fit Combat.

        Estimate parameters in the Combat model. This operation will estimate
        the scale and location effects in the batches supplied, and the
        coefficients for the effects to keep after harmonisation.

        Parameters
        ----------
        in_features : {array-like, sparse matrix},
                      shape (n_samples, n_features)
            The training input samples.
        batch : ndarray, shape (n_samples, )
            Array of batches.
        effects_interest: ndarray (n_samples, n_features_of_effects)
            Optional array of effects of interest to retain after harmonisation.

        Returns
        -------
        self

        """

        in_features = check_array(in_features, dtype="numeric",
                                  force_all_finite=True,
                                  ensure_2d=True, allow_nd=False,
                                  ensure_min_samples=1,
                                  ensure_min_features=1)
        batch = column_or_1d(batch)

        if effects_interest is not None:
            effects_interest = check_array(effects_interest)

        check_consistent_length([in_features,
                                 batch,
                                 effects_interest])

        return self._fit(in_feat=in_features,
                         batch=batch,
                         effects=effects_interest
                         )

    def _fit(self, in_feat, batch, effects):
        """Actual fit method."""
        # extract unique batch categories
        batches = np.unique(batch)
        self.batches_ = batches

        # Construct one-hot-encoding matrix for batches
        # TODO is there a reason this might fail? Should we OneHotEncoder()?
        B = np.column_stack([(batch == b_name).astype(int)
                             for b_name in self.batches_])

        n_samples, n_features = in_feat.shape
        n_batch = B.shape[1]

        if n_batch == 1:
            raise ValueError('The number of batches should be at least 2')

        sample_per_batch = B.sum(axis=0)

        if np.any(sample_per_batch == 1):
            raise ValueError('Each batch should have at least 2 observations'
                             'In the future, when this does not happens,'
                             'only mean adjustment will take place')

        # Construct design matrix
        M = B.copy()
        if isinstance(effects, np.ndarray):
            M = np.column_stack((M, effects))
            end_x = n_batch + effects.shape[1]
        else:
            end_x = n_batch

        # OLS estimation for standardization
        # X = self._standardize(in_feat, B, M, n_batch)
        beta_hat = np.matmul(np.linalg.inv(np.matmul(M.T, M)),
                             np.matmul(M.T, in_feat))

        # Find grand mean intercepts, from batch intercepts
        alpha_hat = np.matmul(sample_per_batch / float(n_samples),
                              beta_hat[:n_batch, :])
        self.intercept_ = alpha_hat

        # Find slopes for the  effects of interest
        coefs_x = beta_hat[n_batch:end_x, :]
        self.coefs_x_ = coefs_x

        # Compute error between predictions and observed values
        Y_hat = np.matmul(M, beta_hat)  # fitted observations
        variance = np.mean(((in_feat - Y_hat) ** 2), axis=0)
        self.variance_ = variance  # square of sigma i.e. variance

        # Standardise data
        Z = in_feat.copy()
        Z -= alpha_hat[np.newaxis, :]
        Z -= np.matmul(M[:, n_batch:end_x], coefs_x)
        Z /= np.sqrt(variance)

        # Find gamma fitted to Standardised data
        gamma_hat = np.matmul(np.linalg.inv(np.matmul(B.T, B)),
                              np.matmul(B.T, Z)
                              )
        # Mean gamma across input features
        gamma_bar = np.mean(gamma_hat, axis=1)

        # Variance across input features
        if n_features > 1:
            ddof_feat = 1
        else:
            raise print("Dataset with just one feature will give NaNs when "
                        "computing the variance across features. This will "
                        "be fixed in the feature")
            # ddof_feat = 0
        tau_bar_sq = np.var(gamma_hat, axis=1, ddof=ddof_feat)
        # tau_bar_sq += 1e-10

        # Variance per batch and feature
        delta_hat_sq = [np.var(Z[B[:, ii] == 1, :], axis=0, ddof=1)
                        for ii in range(B.shape[1])]
        delta_hat_sq = np.array(delta_hat_sq)

        # Compute inverse moments
        lambda_bar = np.apply_along_axis(self._compute_lambda,
                                         arr=delta_hat_sq,
                                         axis=1,
                                         ddof=ddof_feat)
        theta_bar = np.apply_along_axis(self._compute_theta,
                                        arr=delta_hat_sq,
                                        axis=1,
                                        ddof=ddof_feat)

        # TODO: Uncomment when implemented
        # if self.parametric:
        #     it_eb = self._it_eb_param
        # else:
        #     it_eb = self._it_eb_non_param

        it_eb = self._it_eb_param
        gamma_star, delta_sq_star = [], []
        for ii in range(B.shape[1]):
            g, d_sq = it_eb(Z[B[:, ii] == 1, :],
                            gamma_hat[ii, :],
                            delta_hat_sq[ii, :],
                            gamma_bar[ii],
                            tau_bar_sq[ii],
                            lambda_bar[ii],
                            theta_bar[ii],
                            self.tol
                            )

            gamma_star.append(g)
            delta_sq_star.append(d_sq)

        gamma_star = np.array(gamma_star)
        delta_sq_star = np.array(delta_sq_star)

        # TODO: estimates on the goodness of fit and run checks, issue warnings etc

        self.gamma_ = gamma_star
        self.delta_sq_ = delta_sq_star

        return self

    def transform(self,
                  in_features,
                  batch,
                  effects_interest=None):
        """
        Harmonise input features using an already estimated Combat model.

        Parameters
        ----------
        in_features : {array-like, sparse matrix},
                      shape (n_samples, n_features)
            The training input samples.
        batch : ndarray, shape (n_samples, )
            Array of batches.
        effects_interest: ndarray (n_samples, n_features_of_effects),
            optional array of effects of interest to keep after harmonisation.

        Returns
        -------
        in_features_transformed : harmonised in_features
        """

        in_features, batch, effects_interest = self._validate_for_transform(
            in_features, batch, effects_interest)

        return self._transform(in_features,
                               batch,
                               effects_interest)

    def _transform(self, in_feat, batch, effects):
        """Actual deconfounding of the test features."""
        test_batches = np.unique(batch)

        # test features must standardised before applying ComBat
        # TODO modularize this out for both training/testing
        #   to ensure it is applied on both using same estimates
        Y_trans = in_feat - self.intercept_[np.newaxis, :]

        if self.coefs_x_.size > 0:
            Y_trans -= np.matmul(effects, self.coefs_x_)

        Y_trans /= np.sqrt(self.variance_)

        # actual transformation:
        for tb in test_batches:
            # select corresponding batch estimation
            ix_batch = np.where(self.batches_ == tb)[0]
            # Apply to the observations within this particular test batch
            Y_trans[batch == tb, :] -= self.gamma_[ix_batch]
            Y_trans[batch == tb, :] /= np.sqrt(self.delta_sq_[ix_batch, :])
        Y_trans *= np.sqrt(self.variance_)

        # bringing test features into their original scale (sort of inv. standardize)
        # Add intercept
        Y_trans += self.intercept_[np.newaxis, :]

        # Add effects of interest, if there's any
        if self.coefs_x_.size > 0:
            Y_trans += np.matmul(effects, self.coefs_x_)

        return Y_trans

    def _validate_for_transform(self, Y, b, X):
        """
        Method to ensure data is appropriate for transform, such as
        - ensuring batches in test set exist in the training set also
        - consistency in variable dimensions tec

        Returns
        -------
        ValueError
            if any of the checks fail

        """

        # check if fitted
        attributes = ['intercept_',
                      'coefs_x_',
                      'variance_',
                      'gamma_',
                      'delta_sq_']

        # Check if Combat was previously fitted
        check_is_fitted(self, attributes=attributes)

        # Ensure that data are numpy array objects
        Y = check_array(Y)
        if X is not None:
            X = check_array(X)

        # Check that input arrays have the same observations
        check_consistent_length([Y, b, X])

        if Y.shape[1] != len(self.intercept_):
            raise ValueError("Wrong number of features for in_feat")

        # Check that supplied batches exist in the fitted object
        b_not_in_model = np.in1d(np.unique(b), self.batches_, invert=True)
        if np.any(b_not_in_model):
            raise ValueError("test batches categories not in "
                             "the trained model")

        if self.coefs_x_.size > 0:
            if X is None:
                raise ValueError("Effects of interest should be supplied, "
                                 "since Combat was fitted with them")
            if X.shape[1] != self.coefs_x_.shape[0]:
                raise ValueError("Dimensions of fitted beta "
                                 "and input X matrix do not match")

        return Y, b, X

    def fit_transform(self,
                      in_features,
                      batch,
                      effects_interest=None):
        """
        Concatenate fit and transform operations.

        Fit combat and then transform on the same data. You may want
        to use this function for training data harmonisation

       Parameters
        ----------
        in_features : {array-like, sparse matrix},
                      shape (n_samples, n_features)
            The training input samples.
        batch : ndarray, shape (n_samples, )
            Array of batches.
        effects_interest: ndarray, shape (n_samples, n_features_of_effects),
            optinal.
            Array of effects of interest to keep after harmonisation.

        Returns
        -------
        in_features_transformed : harmonised in_features

        """
        # Fit Combat
        self.fit(in_features=in_features,
                 batch=batch,
                 effects_interest=effects_interest
                 )
        # Use same data to harmonise it
        return self.transform(in_features=in_features,
                              batch=batch,
                              effects_interest=effects_interest)

    def _it_eb_param(self,
                     Z_batch,
                     gam_hat_batch,
                     del_hat_sq_batch,
                     gam_bar_batch,
                     tau_sq_batch,
                     lam_bar_batch,
                     the_bar_batch,
                     conv):
        """Parametric EB estimation of location and scale paramaters."""
        # Number of non nan samples within the batch for each variable
        n = np.sum(1 - np.isnan(Z_batch), axis=0)
        gam_prior = gam_hat_batch.copy()
        del_sq_prior = del_hat_sq_batch.copy()

        change = 1
        count = 0
        while change > conv:
            gam_post = self._post_gamma(del_sq_prior,
                                        gam_hat_batch,
                                        gam_bar_batch,
                                        tau_sq_batch,
                                        n)

            del_sq_post = self._post_delta(gam_post,
                                           Z_batch,
                                           lam_bar_batch,
                                           the_bar_batch,
                                           n)

            change = max((abs(gam_post - gam_prior) / gam_prior).max(),
                         (abs(del_sq_post - del_sq_prior) / del_sq_prior).max()
                         )
            gam_prior = gam_post
            del_sq_prior = del_sq_post
            count = count + 1

        # TODO: Make namedtuple?
        return (gam_post, del_sq_post)

    def _it_eb_non_param(self):
        """Non-parametric version"""

        # TODO
        return NotImplementedError()

    @staticmethod
    def _compute_lambda(del_hat_sq, ddof):
        """Estimation of hyper-parameter lambda."""
        v = np.mean(del_hat_sq)
        s2 = np.var(del_hat_sq, ddof=ddof)
        # s2 += 1e-10
        # In Johnson 2007  there's a typo
        # in the suppl. material as it
        # should be with v^2 and not v
        return (2 * s2 + v ** 2) / float(s2)

    @staticmethod
    def _compute_theta(del_hat_sq, ddof):
        """Estimation of hyper-parameter theta."""
        v = del_hat_sq.mean()
        s2 = np.var(del_hat_sq, ddof=ddof)
        # s2 += 1e-10
        return (v * s2 + v ** 3) / s2

    @staticmethod
    def _post_gamma(x, gam_hat, gam_bar, tau_bar_sq, n):
        # x is delta_star
        num = tau_bar_sq * n * gam_hat + x * gam_bar
        den = tau_bar_sq * n + x
        return num / den

    @staticmethod
    def _post_delta(x, Z, lam_bar, the_bar, n):
        num = the_bar + 0.5 * np.sum((Z - x[np.newaxis, :]) ** 2, axis=0)
        den = n / 2.0 + lam_bar - 1
        return num / den
