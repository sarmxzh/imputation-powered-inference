import logging
from typing import Literal

import numpy as np
from scipy.stats import norm

from ipi.utils.missing_data_utils import get_missing_patterns_and_ids
from ipi.utils.stats_utils import get_gradients, get_point_estimate, get_sample_hessian

logger = logging.getLogger(__name__)


class ClassOne_AIPW_CI:
    """
    ClassOne_AIPW_CI is a class for calculating the semiparametric confidence interval for the parameter of interest.
    Current implementation is for class one estimators (See Tsiatis (2006) Chapter 12)
    for *MCAR* data with fixed score function (not assuming correctly specified semi-parametric model)
    and basis functions defined to be monomials up to degree 2.
    """

    def __init__(
        self,
        regression_type: Literal["ols", "logistic"],
        fully_obs_data: np.ndarray,
        partially_obs_data: np.ndarray,
        regr_features_idxs: list[int],
        outcome_idx: int,
        target_idx: int,
        intercept_idx: int | None,
    ):
        self.regression_type = regression_type
        self.fully_obs_data = fully_obs_data
        self.partially_obs_data = partially_obs_data
        self.regr_features_idxs = regr_features_idxs
        self.outcome_idx = outcome_idx
        self.target_idx = target_idx
        self.intercept_idx = intercept_idx

        self._pattern_to_ids = get_missing_patterns_and_ids(
            data_array=self.partially_obs_data
        )

        self.theta_init = get_point_estimate(
            data_array=self.fully_obs_data,
            regr_feature_idxs=self.regr_features_idxs,
            outcome_idx=self.outcome_idx,
            regression_type=self.regression_type,
            return_se=False,
        )
        if self.theta_init.ndim == 1:
            self.theta_init = self.theta_init.reshape(-1, 1)

        self.n_fully_obs = self.fully_obs_data.shape[0]
        self.n_partially_obs = self.partially_obs_data.shape[0]
        self.n_total = self.n_fully_obs + self.n_partially_obs
        self.num_regr_features = len(self.regr_features_idxs)
        self.num_features = self.fully_obs_data.shape[1]
        self.num_patterns = len(self._pattern_to_ids)
        self.fully_obs_tuple = tuple(True for _ in range(self.num_features))

        augmentation_space_basis_dim = 0
        for pattern_k in self._pattern_to_ids:
            ## adding basis functions up to degree 2 for each pattern
            augmentation_space_basis_dim += self.get_basis_dim(pattern_k)
        self.augmentation_space_basis_dim = int(augmentation_space_basis_dim)

    def get_basis_dim(self, pattern: tuple[bool, ...]) -> int:
        num_obs_features_k = sum(pattern)
        if self.intercept_idx is not None:
            num_obs_features_k -= (
                1  # subtract 1 for intercept, since constant term is included
            )
        return (
            1 + num_obs_features_k + num_obs_features_k * (num_obs_features_k + 1) / 2
        )

    def get_quad_monomials(
        self, data_array: np.ndarray, pattern: tuple[bool, ...]
    ) -> np.ndarray:
        """
        Return all monomials up to degree 2 using observed features.

        For the subset of rows whose observed pattern equals `pattern`, construct a
        feature matrix consisting of:
          - constant term 1
          - all degree-1 terms for observed features
          - all unique degree-2 terms (including squares) for observed features

        The ordering is [1, x_i (i in regr features), x_i x_j for i<=j].

        Args:
            pattern: Tuple of booleans indicating observed (True) or missing (False)
                     for each column in the data array.

        Returns:
            A numpy array of shape (n_rows_in_pattern, 1 + k + k*(k+1)/2), where k is the
            number of observed regression features in `pattern`.
        """

        # Validate pattern exists and get row indices
        if pattern not in self._pattern_to_ids:
            raise KeyError("Pattern not found in the partially observed data")

        # Determine which regression features are observed under this pattern
        # ignore intercept if it exists since we already have a constant term
        observed_feature_idxs = [
            idx
            for idx in range(self.num_features)
            if pattern[idx] and idx != self.intercept_idx
        ]

        # Extract the observed feature matrix for the rows in this pattern
        X_basis_features = data_array[:, observed_feature_idxs]
        n_rows, n_features = (
            X_basis_features.shape
            if X_basis_features.ndim == 2
            else (X_basis_features.shape[0], 1)
        )

        # Start with constant term and degree-1 terms
        columns = [np.ones((n_rows, 1), dtype=float)]
        if n_features > 0:
            X_obs = X_basis_features.astype(float, copy=False)
            columns.append(X_obs)

            # Degree-2 unique terms including squares: take upper triangle i<=j
            # Compute row-wise outer products efficiently
            prod = np.einsum("ni,nj->nij", X_obs, X_obs)
            tri_i, tri_j = np.triu_indices(n_features)
            quad_terms = prod[:, tri_i, tri_j]
            columns.append(quad_terms)

        # Concatenate all columns to form the monomial basis
        return np.concatenate(columns, axis=1)

    def _calculate_Us_H1(
        self,
        grads_fully_obs: np.ndarray,
        basis_vectors_fully_obs: np.ndarray,
        hessian_fully_obs: np.ndarray,
    ):
        ## calculate the U_11, U_12, and U_22 matrices (as defined in Tsiatis (2006) Chapter 12)
        ## U_11 = E_{fully_obs}[(N+n)/n * grad l(D; \theta) grad l(D; \theta)^T]
        U_11 = (
            (self.n_total / self.n_fully_obs)
            * (grads_fully_obs.T @ grads_fully_obs)
            / self.n_fully_obs
        )
        assert U_11.shape[0] == self.num_regr_features
        assert U_11.shape[1] == self.num_regr_features

        ## U_12 = E_{fully_obs}[grad l(D; \theta) J_2(fully obs)^T]
        U_12 = (
            (self.n_total / self.n_fully_obs)
            * (grads_fully_obs.T @ basis_vectors_fully_obs)
            / self.n_fully_obs
        )
        assert U_12.shape[0] == self.num_regr_features
        assert U_12.shape[1] == self.augmentation_space_basis_dim

        ## U_22 = E_{overall}[ J_2(pattern)  J_2(pattern)^T]
        U_22 = (
            (self.n_total / self.n_fully_obs)
            * (basis_vectors_fully_obs.T @ basis_vectors_fully_obs)
            / self.n_fully_obs
        )
        starting_idx = 0
        for pattern_k in self._pattern_to_ids:
            n_pattern_k = len(self._pattern_to_ids[pattern_k])
            next_starting_idx = int(starting_idx + self.get_basis_dim(pattern_k))
            quad_monomials_k = -1 * self.get_quad_monomials(
                data_array=self.partially_obs_data[
                    list(self._pattern_to_ids[pattern_k])
                ],
                pattern=pattern_k,
            )
            assert quad_monomials_k.shape[0] == n_pattern_k
            assert quad_monomials_k.shape[1] == self.get_basis_dim(pattern_k)

            pattern_k_contribution = np.zeros(
                (self.augmentation_space_basis_dim, self.augmentation_space_basis_dim)
            )
            pattern_k_contribution[
                starting_idx:next_starting_idx, starting_idx:next_starting_idx
            ] = (
                (self.n_total / n_pattern_k)
                * (quad_monomials_k.T @ quad_monomials_k)
                / n_pattern_k
            )
            U_22 += pattern_k_contribution

            starting_idx = next_starting_idx
        assert U_22.shape[0] == self.augmentation_space_basis_dim
        assert U_22.shape[1] == self.augmentation_space_basis_dim

        ## H_1 = fully_obs_hessian.T
        H_1 = -1 * hessian_fully_obs.T

        ## calculate the inverse of U_22
        U_22_inv = np.linalg.inv(U_22)

        ## calculate the blockwise entries U^{11} and U^{12} of the inverted matrix U.
        ## where U = [U_{11} U_{12}; U_{12}^T U_{22}]
        U_11_schur = np.linalg.inv(U_11 - U_12 @ U_22_inv @ U_12.T)
        U_12_schur = -1 * U_11_schur @ U_12 @ U_22_inv

        return U_11_schur, U_12_schur, H_1

    def get_semiparam_ci(self, alpha: float = 0.1) -> tuple[float, float]:
        """
        This function calculates the semiparametric confidence interval for the
        parameter of interest. Current implementation is for class one estimators (See Tsiatis (2006) Chapter 12)
        with fixed score function (not assuming correctly specified semi-parametric model) and basis functions
        defined to be monomials up to degree 2.

        Args:
            alpha: The significance level for the confidence interval.

        Returns:
            A tuple containing the lower and upper bounds of the confidence interval.
        """
        ## following the Tsiatis (2006) notation, $J_2$ is the vector of basis functions for the augmentation space

        ## calculate the hessian and inverse hessian for the fully observed data
        ## this gives p x p array of hessians of the loss
        hessian_fully_obs = get_sample_hessian(
            data_array=self.fully_obs_data,
            regr_feature_idxs=self.regr_features_idxs,
            outcome_idx=self.outcome_idx,
            theta=self.theta_init.flatten(),
            regression_type=self.regression_type,
        )

        ## calculate the loss gradients for the fully observed data
        ## this gives n x p array of gradients of the loss
        grads_fully_obs = get_gradients(
            data_array=self.fully_obs_data,
            regr_feature_idxs=self.regr_features_idxs,
            outcome_idx=self.outcome_idx,
            theta=self.theta_init,
            regression_type=self.regression_type,
        )

        ## calculate the basis vectors for the augmentation space for fully observed data
        ## this gives a n x augmentation_space_basis_dim array of basis vectors
        ## following construction from Tsiatis (2006) Chapter 12 and Sun and Tchetgen (2018) paper
        ## slightly different in that it *does not* scale by n_total / n_fully_obs -- this is implemented
        ## in the _calculate_Us_H1 function
        basis_vectors_fully_obs = []
        for pattern_k in self._pattern_to_ids:
            quad_monomials_k = self.get_quad_monomials(
                data_array=self.fully_obs_data, pattern=pattern_k
            )
            assert quad_monomials_k.shape[0] == self.n_fully_obs
            assert quad_monomials_k.shape[1] == self.get_basis_dim(pattern_k)
            basis_vectors_fully_obs.append(quad_monomials_k)
        basis_vectors_fully_obs = np.concatenate(basis_vectors_fully_obs, axis=1)
        assert basis_vectors_fully_obs.shape[0] == self.n_fully_obs
        assert basis_vectors_fully_obs.shape[1] == self.augmentation_space_basis_dim

        U_11_schur, U_12_schur, H_1 = self._calculate_Us_H1(
            grads_fully_obs=grads_fully_obs,
            basis_vectors_fully_obs=basis_vectors_fully_obs,
            hessian_fully_obs=hessian_fully_obs,
        )

        ## calculate the scaling matrices for the augmentation space
        A_F_scaling = H_1 @ U_11_schur
        A_2_scaling = H_1 @ U_12_schur

        ## calculate the influence function for the parameter of interest
        IF = np.zeros((self.num_regr_features, 1))
        ## first term is the contribution from the fully observed data
        IF += np.mean(grads_fully_obs @ A_F_scaling.T, axis=0).reshape(-1, 1)
        IF += np.mean(basis_vectors_fully_obs @ A_2_scaling.T, axis=0).reshape(-1, 1)

        starting_idx_if = 0
        for pattern_k in self._pattern_to_ids:
            next_starting_idx_if = int(starting_idx_if + self.get_basis_dim(pattern_k))
            n_pattern_k = len(self._pattern_to_ids[pattern_k])
            quad_monomials_k_partially_obs = -1 * self.get_quad_monomials(
                data_array=self.partially_obs_data[
                    list(self._pattern_to_ids[pattern_k])
                ],
                pattern=pattern_k,
            )
            assert quad_monomials_k_partially_obs.shape[0] == n_pattern_k
            assert quad_monomials_k_partially_obs.shape[1] == self.get_basis_dim(
                pattern_k
            )

            pattern_k_contribution = np.mean(
                quad_monomials_k_partially_obs
                @ A_2_scaling[:, starting_idx_if:next_starting_idx_if].T,
                axis=0,
            ).reshape(-1, 1)

            IF += pattern_k_contribution

            starting_idx_if = next_starting_idx_if

        assert IF.shape[0] == self.num_regr_features
        assert IF.shape[1] == 1

        ## calculate the point estimate via one-step
        hessian_scale = -1 * A_F_scaling @ hessian_fully_obs
        theta_est = self.theta_init + (np.linalg.inv(hessian_scale) @ IF)

        hessian_fully_obs_updated = get_sample_hessian(
            data_array=self.fully_obs_data,
            regr_feature_idxs=self.regr_features_idxs,
            outcome_idx=self.outcome_idx,
            theta=theta_est.flatten(),
            regression_type=self.regression_type,
        )
        grads_fully_obs_updated = get_gradients(
            data_array=self.fully_obs_data,
            regr_feature_idxs=self.regr_features_idxs,
            outcome_idx=self.outcome_idx,
            theta=theta_est,
            regression_type=self.regression_type,
        )
        U_11_schur_updated, _, H_1_updated = self._calculate_Us_H1(
            grads_fully_obs_updated, basis_vectors_fully_obs, hessian_fully_obs_updated
        )

        var_est = (
            np.linalg.inv(H_1_updated @ U_11_schur_updated @ H_1_updated.T)
            / self.n_total
        )

        # # ## calculate the confidence interval for target_idx parameter
        theta_est_coord = theta_est[self.target_idx]
        se_theta_est = np.sqrt(var_est[self.target_idx, self.target_idx])
        halfwidth = norm.ppf(1 - alpha / 2) * se_theta_est
        return (theta_est_coord - halfwidth, theta_est_coord + halfwidth)
