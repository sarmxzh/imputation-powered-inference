# %%
import logging
import pickle
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from ipi.missforest.src.missforest import MissForest
from ipi.utils.factor_model_em_utils import fm_impute_X, get_pretrained_parameters_EM
from ipi.utils.missing_data_utils import get_missing_patterns_and_ids
from ipi.utils.stats_utils import get_gradients, get_point_estimate, get_sample_hessian

logger = logging.getLogger(__name__)


# %%
@dataclass
class IPI_Grad_Hessian:
    grads_fully_obs: np.ndarray
    hessians_fully_obs: np.ndarray
    grads_partially_obs: np.ndarray
    hessians_partially_obs: np.ndarray


@dataclass
class IPI_Point_Weights_and_Variance:
    theta_ipi: np.ndarray
    lambda_weights: np.ndarray
    ipi_variance: np.ndarray


class BaseIPI(ABC):
    """
    Base class for standard Improved Prediction Interval (IPI) methods.

    Standard IPI trains a single imputation model on independent data and reuses it
    for all imputations. This is computationally efficient but assumes the training
    data is truly independent from the inference data.

    Key Features:
    - train_model() trains and stores the model as an instance attribute AND returns it
    - impute_values() uses the stored model (no model parameter needed)
    - Persistent model storage for reuse
    - fit_and_setup() convenience method for one-step training and data preparation
    - Designed for scenarios with good independent training data

    For cross-fitting workflows (to reduce overfitting), see BaseCrossIPI instead.

    Workflow:
        1. fit_and_setup(train_data) or train_model() + setup_imputed_data()
        2. get_ipi_ci() for confidence intervals

    Example:
        ```python
        ipi = EM_IPI(...)
        ipi.fit_and_setup(train_data, **kwargs)
        result = ipi.get_ipi_ci(...)
        ```

    Args:
        regression_type: Type of regression ("ols" or "logistic")
        fully_obs_data: Fully observed data array
        partially_obs_data: Partially observed data array (with missing values)
        regr_features_idxs: Indices of regression features
        outcome_idx: Index of the outcome variable
        target_idx: Index of the target coefficient for inference
        pretrained_model: Optional pre-trained model to use
    """

    def __init__(
        self,
        regression_type: Literal["ols", "logistic"],
        fully_obs_data: np.ndarray,
        partially_obs_data: np.ndarray,
        regr_features_idxs: list[int],
        outcome_idx: int,
        target_idx: int,
        pretrained_model: object | None = None,
    ):
        # Checking validity of inputs
        if regression_type not in ["ols", "logistic"]:
            raise ValueError(
                f"regression_type must be 'ols' or 'logistic', got {regression_type}"
            )

        if not isinstance(regr_features_idxs, list) or not all(
            isinstance(i, int) for i in regr_features_idxs
        ):
            raise ValueError("regr_features_idxs must be a list of integers")

        if not isinstance(outcome_idx, int) or not isinstance(target_idx, int):
            raise ValueError("outcome_idx and target_idx must be integers")

        if not isinstance(fully_obs_data, np.ndarray) or not isinstance(
            partially_obs_data, np.ndarray
        ):
            raise ValueError("fully_obs_data and partially_obs_data must be np.ndarray")

        if fully_obs_data.shape[0] == 0 or partially_obs_data.shape[0] == 0:
            raise ValueError("fully_obs_data and partially_obs_data must have rows")

        if fully_obs_data.shape[1] != partially_obs_data.shape[1]:
            raise ValueError(
                "fully_obs_data and partially_obs_data must have the same number of columns"
            )

        if target_idx not in range(len(regr_features_idxs)):
            raise ValueError("target_idx outside length of regr_features_idxs")

        # Public attributes - user configuration and input data
        self.regression_type = regression_type
        self.fully_obs_data = fully_obs_data
        self.partially_obs_data = partially_obs_data
        self.regr_features_idxs = regr_features_idxs
        self.outcome_idx = outcome_idx
        self.target_idx = target_idx

        # Private attributes - cached computational results
        self._theta_ipi = None
        self._var_ipi = None
        self._lambda_weights = None
        self._ipi_computation_valid = False

        # Private attributes - internal state
        self._imputation_model = pretrained_model
        self._fully_obs_data_dict = None
        self._imputed_partially_obs_data = None

        # Cached derived properties (computed on first access)
        self._pattern_to_ids = get_missing_patterns_and_ids(
            data_array=self.partially_obs_data
        )

        # Compute theta_init once - always needed and depends only on immutable inputs
        self.theta_init = get_point_estimate(
            data_array=self.fully_obs_data,
            regr_feature_idxs=self.regr_features_idxs,
            outcome_idx=self.outcome_idx,
            regression_type=self.regression_type,
            return_se=False,
        )
        if self.theta_init.ndim == 1:
            self.theta_init = self.theta_init.reshape(-1, 1)

        # Computed derived values
        self.n_fully_obs = self.fully_obs_data.shape[0]
        self.n_partially_obs = self.partially_obs_data.shape[0]
        self.num_features = self.fully_obs_data.shape[1]
        self.num_patterns = len(self._pattern_to_ids)
        self.fully_obs_tuple = tuple(True for _ in range(self.num_features))

    # Properties for cached computational results
    @property
    def theta_ipi(self) -> np.ndarray | None:
        """Get the cached IPI point estimate, or None if not computed."""
        return self._theta_ipi

    @property
    def var_ipi(self) -> np.ndarray | None:
        """Get the cached IPI variance estimate, or None if not computed."""
        return self._var_ipi

    @property
    def lambda_weights(self) -> np.ndarray | None:
        """Get the cached lambda weights for IPI, or None if not computed."""
        return self._lambda_weights

    @property
    def fully_obs_data_dict(self) -> np.ndarray | None:
        """get fully observed data dictionary"""
        return self._fully_obs_data_dict

    @property
    def imputed_partially_obs_data(self) -> np.ndarray | None:
        """get imputed partially observed data"""
        return self._imputed_partially_obs_data

    # Properties for derived/computed values
    @property
    def pattern_to_ids(self) -> Mapping[tuple[bool, ...], set[int]]:
        """Mapping of missing patterns to observation IDs."""
        return self._pattern_to_ids

    @property
    def imputation_model(self) -> np.ndarray | None:
        """The trained imputation model."""
        return self._imputation_model

    @imputation_model.setter
    def imputation_model(self, value: np.ndarray | None) -> None:
        """Set imputation model and invalidate cache."""
        self._imputation_model = value
        if value is not None:
            self._invalidate_cache()

    @property
    def is_fitted(self) -> bool:
        """Check if imputation model has been trained."""
        return self._imputation_model is not None

    @property
    def is_computation_ready(self) -> bool:
        """Check if ready for IPI computation."""
        return (
            self._fully_obs_data_dict is not None
            and self._imputed_partially_obs_data is not None
        )

    def _invalidate_cache(self) -> None:
        """Invalidate cached computation results."""
        self._theta_ipi = None
        self._var_ipi = None
        self._lambda_weights = None
        self._ipi_computation_valid = False

    def _validate_computation_ready(self) -> None:
        """Validate that the object is ready for computation."""
        if (
            self._fully_obs_data_dict is None
            or self._imputed_partially_obs_data is None
        ):
            raise ValueError(
                "Data not set up for computation. Call setup_imputed_data() or use "
                "fit_and_setup() to prepare the data."
            )

    def get_ipi_grad_and_hessian(
        self,
        theta_est: np.ndarray | None = None,
    ) -> IPI_Grad_Hessian:
        """
        Get the IPI grad and hessian
        Returns:
            IPI_Grad_Hessian: IPI grad and hessian
        """
        self._validate_computation_ready()

        ## get grad and hessian for each pattern
        hessians_fully_obs = {}
        grads_fully_obs = {}

        ## can implement using IPI with theta_est = theta_hat but using theta_init is more efficient
        if theta_est is None:
            # logger.info("no theta_est provided, using theta_init")
            theta_est = self.theta_init

        # initialize with the fully observed data grads and sample hessian
        hessians_fully_obs[self.fully_obs_tuple] = get_sample_hessian(
            data_array=self._fully_obs_data_dict[self.fully_obs_tuple],
            regr_feature_idxs=self.regr_features_idxs,
            outcome_idx=self.outcome_idx,
            theta=theta_est.flatten(),
            regression_type=self.regression_type,
        )

        grads_fully_obs[self.fully_obs_tuple] = get_gradients(
            data_array=self._fully_obs_data_dict[self.fully_obs_tuple],
            regr_feature_idxs=self.regr_features_idxs,
            outcome_idx=self.outcome_idx,
            theta=theta_est,
            regression_type=self.regression_type,
        )

        hessians_partially_obs = {}
        grads_partially_obs = {}

        for pattern, ids_in_pattern in self.pattern_to_ids.items():
            # get the data for the partially observed pattern
            partially_obs_pattern_data = self._imputed_partially_obs_data[
                list(ids_in_pattern), :
            ]

            # get the grad and hessian for the partially observed pattern
            hessians_partially_obs[pattern] = get_sample_hessian(
                data_array=partially_obs_pattern_data,
                regr_feature_idxs=self.regr_features_idxs,
                outcome_idx=self.outcome_idx,
                theta=theta_est.flatten(),
                regression_type=self.regression_type,
            )
            grads_partially_obs[pattern] = get_gradients(
                data_array=partially_obs_pattern_data,
                regr_feature_idxs=self.regr_features_idxs,
                outcome_idx=self.outcome_idx,
                theta=theta_est,
                regression_type=self.regression_type,
            )

            # get the grad and hessian for the fully observed pattern

            hessians_fully_obs[pattern] = get_sample_hessian(
                data_array=self._fully_obs_data_dict[pattern],
                regr_feature_idxs=self.regr_features_idxs,
                outcome_idx=self.outcome_idx,
                theta=theta_est.flatten(),
                regression_type=self.regression_type,
            )
            grads_fully_obs[pattern] = get_gradients(
                data_array=self._fully_obs_data_dict[pattern],
                regr_feature_idxs=self.regr_features_idxs,
                outcome_idx=self.outcome_idx,
                theta=theta_est,
                regression_type=self.regression_type,
            )
        return IPI_Grad_Hessian(
            grads_fully_obs=grads_fully_obs,
            hessians_fully_obs=hessians_fully_obs,
            grads_partially_obs=grads_partially_obs,
            hessians_partially_obs=hessians_partially_obs,
        )

    def get_ipi_pointestimate_and_variance(
        self,
        lambda_weights: np.ndarray | None = None,
        track_variance: bool = True,
        use_theta_hat_for_variance: bool = False,
        *,
        lambda_reg_l2: float = 0.0,
        lambda_reg_l1: float = 0.0,
        cp_iter: int = 1000,
        cp_tol: float = 1e-6,
        shift_magnitudes: np.ndarray
        | None = None,  ## NOTE: should always be None for actual use
    ) -> IPI_Point_Weights_and_Variance:
        """
        Calculates point estimate for multiple missing entry patterns
        Args:
            lambda_weights: lambda weights for the linear combination of the masks
            track_variance: whether to track the variance
        Returns:
            IPI_Point_Weights_and_Variance: IPI point estimate for theta
        Optional regularization on lambda (weights over patterns):
            lambda_reg_l2: L2 strength; adds ridge to quadratic form over lambda
            lambda_reg_l1: L1 strength; solved via convex optimization when > 0
            cp_iter: Max iterations for convex solver (cp_iter)
            cp_tol: Convergence tolerance for convex solver (cp_tol)
            shift_magnitudes: shift magnitudes for the IPI point estimate and variance calculation
        """

        # Check if we can use cached results
        if (
            self._ipi_computation_valid
            and np.array_equal(self._lambda_weights, lambda_weights)
            and self._theta_ipi is not None
            and self._var_ipi is not None
        ):
            logger.info("Using cached IPI results")
            return IPI_Point_Weights_and_Variance(
                theta_ipi=self._theta_ipi,
                lambda_weights=self._lambda_weights,
                ipi_variance=self._var_ipi,
            )

        # set the lambda weights to the user-specified lambda_weights
        self._lambda_weights = lambda_weights

        # Compute fresh results
        ipi_grad_hessian = self.get_ipi_grad_and_hessian()
        grads_fully_obs = ipi_grad_hessian.grads_fully_obs
        hessians_fully_obs = ipi_grad_hessian.hessians_fully_obs
        grads_partially_obs = ipi_grad_hessian.grads_partially_obs
        hessians_partially_obs = ipi_grad_hessian.hessians_partially_obs

        if shift_magnitudes is not None:
            logger.info(
                "Using shift magnitudes for IPI point estimate and variance calculation"
            )
            for r, pattern in enumerate(self.pattern_to_ids.keys()):
                num_regr_features = grads_fully_obs[self.fully_obs_tuple].shape[1]
                grads_partially_obs[pattern] = grads_partially_obs[
                    pattern
                ] + shift_magnitudes[r] * np.ones((1, num_regr_features))

        # get the lambda weights
        if self._lambda_weights is None:
            # logger.info("No lambda weights provided, calculating lambda weights")
            # get the lambda weights
            self._lambda_weights = calc_lambda_weights(
                grad_patterns_fullyobs=grads_fully_obs,
                grad_patterns_partial=grads_partially_obs,
                hessian_patterns_fullyobs=hessians_fully_obs,
                hessian_patterns_partial=hessians_partially_obs,
                pattern_to_ids=self.pattern_to_ids,
                target_idx=self.target_idx,
                num_features=self.num_features,
                lambda_l2=float(lambda_reg_l2),
                lambda_l1=float(lambda_reg_l1),
                cp_iter=int(cp_iter),
                cp_tol=float(cp_tol),
            )

        # get the hessian and grad for the IPI
        hessian_ipi = hessians_fully_obs[self.fully_obs_tuple] + sum(
            self._lambda_weights[r]
            * (hessians_partially_obs[pattern] - hessians_fully_obs[pattern])
            for r, pattern in enumerate(self.pattern_to_ids.keys())
        )

        grad_ipi = np.mean(grads_fully_obs[self.fully_obs_tuple], axis=0).reshape(
            -1, 1
        ) + sum(
            self._lambda_weights[r]
            * (
                np.mean(grads_partially_obs[pattern], axis=0).reshape(-1, 1)
                - np.mean(grads_fully_obs[pattern], axis=0).reshape(-1, 1)
            )
            for r, pattern in enumerate(self.pattern_to_ids.keys())
        )

        # get the point estimate
        theta_hat = self.theta_init - np.linalg.solve(hessian_ipi, grad_ipi)

        # calculate the variance
        ipi_variance = None
        if track_variance:
            grads_fully_obs_for_variance = grads_fully_obs
            hessians_fully_obs_for_variance = hessians_fully_obs
            grads_partially_obs_for_variance = grads_partially_obs
            hessians_partially_obs_for_variance = hessians_partially_obs
            if use_theta_hat_for_variance:
                logger.info("Using theta_hat for variance calculation")
                ## NOTE: in all experiments, this is not used
                ## calculate grad and hessian for theta_hat
                ## should be more stable than using theta_init (if newton step does not overshoot)
                ipi_grad_hessian_for_variance = self.get_ipi_grad_and_hessian(
                    theta_est=theta_hat
                )
                grads_fully_obs_for_variance = (
                    ipi_grad_hessian_for_variance.grads_fully_obs
                )
                hessians_fully_obs_for_variance = (
                    ipi_grad_hessian_for_variance.hessians_fully_obs
                )
                grads_partially_obs_for_variance = (
                    ipi_grad_hessian_for_variance.grads_partially_obs
                )
                hessians_partially_obs_for_variance = (
                    ipi_grad_hessian_for_variance.hessians_partially_obs
                )
            delta_term = grads_fully_obs_for_variance[self.fully_obs_tuple].copy()

            # Subtract weighted masked gradients for each observation
            for r, pattern in enumerate(self.pattern_to_ids.keys()):
                delta_term -= (
                    self._lambda_weights[r] * grads_fully_obs_for_variance[pattern]
                )
            var_delta_term = np.cov(delta_term.T) / self.n_fully_obs
            var_partial_term = np.zeros_like(var_delta_term)
            for r, pattern in enumerate(self.pattern_to_ids.keys()):
                n_k = len(self.pattern_to_ids[pattern])

                threshold = 30  # TODO: make this a parameter
                if n_k >= threshold:
                    cov_k = np.cov(grads_partially_obs_for_variance[pattern].T)
                else:
                    raise ValueError(
                        f"number of samples per pattern {r} must be greater than {threshold}"
                    )
                var_partial_term += (self._lambda_weights[r] ** 2) * (cov_k / n_k)

            theta_hat = theta_hat.flatten()
            S = var_delta_term + var_partial_term

            inv_hessian_ipi_for_variance = np.linalg.inv(hessian_ipi)
            if use_theta_hat_for_variance:
                hessian_ipi_for_variance = hessians_fully_obs_for_variance[
                    self.fully_obs_tuple
                ] + sum(
                    self._lambda_weights[r]
                    * (
                        hessians_partially_obs_for_variance[pattern]
                        - hessians_fully_obs_for_variance[pattern]
                    )
                    for r, pattern in enumerate(self.pattern_to_ids.keys())
                )
                inv_hessian_ipi_for_variance = np.linalg.inv(hessian_ipi_for_variance)

            ipi_variance = (
                inv_hessian_ipi_for_variance @ S @ inv_hessian_ipi_for_variance.T
            )

        # Cache results
        self._theta_ipi = theta_hat
        self._var_ipi = ipi_variance
        self._ipi_computation_valid = True

        return IPI_Point_Weights_and_Variance(
            theta_ipi=theta_hat,
            lambda_weights=self._lambda_weights,
            ipi_variance=ipi_variance,
        )

    def get_ipi_ci(
        self,
        lambda_weights: np.ndarray | None,
        alpha: float = 0.1,
        lambda_reg_l2: float = 0.0,
        lambda_reg_l1: float = 0.0,
        cp_iter: int = 1000,
        cp_tol: float = 1e-6,
        shift_magnitudes: np.ndarray | None = None,
    ) -> tuple[float, float]:
        """
        Calculates the confidence interval for the IPI method
        Args:
            lambda_weights: lambda weights
            alpha: significance level
        Returns:
            ci: confidence interval
        """
        # get the IPI point estimate and variance
        ipi_vars = self.get_ipi_pointestimate_and_variance(
            lambda_weights=lambda_weights,
            track_variance=True,
            lambda_reg_l2=lambda_reg_l2,
            lambda_reg_l1=lambda_reg_l1,
            cp_iter=cp_iter,
            cp_tol=cp_tol,
            use_theta_hat_for_variance=False,
            shift_magnitudes=shift_magnitudes,
        )
        theta_ipi = ipi_vars.theta_ipi
        ipi_variance = ipi_vars.ipi_variance

        theta_ipi_coord = theta_ipi[self.target_idx]
        se_theta_ipi = np.sqrt(ipi_variance[self.target_idx, self.target_idx])
        halfwidth = norm.ppf(1 - alpha / 2) * se_theta_ipi

        return (theta_ipi_coord - halfwidth, theta_ipi_coord + halfwidth)

    def test1_mcar_moment_conditions(
        self,
        shift_magnitudes: np.ndarray
        | None = None,  # should always be None for actual use; shift_weights is for testing
        track_stats: bool = False,  # should be False for actual use; track_stats is for testing
    ) -> float | tuple[float, float]:
        """
        A second test for MCAR first moment condition for the current set up
        Just a chi-squared test on the debiasing term

        Args:
            None
        Returns:
            p-value: p-value for the single test
        Notes:
            This test is based on the debiasing term and variance calculation.
            The debiasing term is calculated as the difference between the fully observed data and the partially observed data.
            The variance is calculated as the covariance of the debiasing term.
            The standardized debiased term is calculated as the debiased term divided by the square root of the variance.
            If lambda weights were provided in a non-data-dependent way, this test is possibly conservative.
            The p-value is calculated as 2 * norm.cdf(-np.abs(standardized_debiased_term)).
        """
        assert self._theta_ipi is not None, "theta_ipi must be provided"
        assert self._var_ipi is not None, "var_ipi must be provided"
        assert (
            self._fully_obs_data_dict is not None
        ), "fully_obs_data_dict must be provided"
        assert (
            self._imputed_partially_obs_data is not None
        ), "imputed_partially_obs_data must be provided"
        assert self._pattern_to_ids is not None, "pattern_to_ids must be provided"
        assert self.target_idx is not None, "target_idx must be provided"
        assert self._theta_ipi is not None, "theta_ipi must be provided"

        ipi_grad_hessian = self.get_ipi_grad_and_hessian()
        num_regr_features = self._theta_ipi.shape[0]
        num_patterns = len(self.pattern_to_ids)

        grads_fully_obs = ipi_grad_hessian.grads_fully_obs
        grads_partially_obs = ipi_grad_hessian.grads_partially_obs
        hessians_fully_obs = ipi_grad_hessian.hessians_fully_obs
        hessians_partially_obs = ipi_grad_hessian.hessians_partially_obs

        inv_hessian = np.linalg.inv(hessians_fully_obs[self.fully_obs_tuple])

        debiasing_term_vec = np.zeros((num_patterns * num_regr_features, 1))
        fullyobs_vec = np.zeros((self.n_fully_obs, num_patterns * num_regr_features))
        debiasing_var = np.zeros(
            (num_patterns * num_regr_features, num_patterns * num_regr_features)
        )
        if shift_magnitudes is not None:
            logger.info("Using shift magnitudes for test2_mcar_moment_conditions")
        for k, (pattern, pattern_ids) in enumerate(self.pattern_to_ids.items()):
            pattern_shift_magnitude = (
                shift_magnitudes[k] * np.ones((1, num_regr_features))
                if shift_magnitudes is not None
                else np.zeros((1, num_regr_features))
            )
            grads_fully_obs_k = grads_fully_obs[pattern]
            hessians_fully_obs_k = hessians_fully_obs[pattern]
            hessians_partially_obs_k = hessians_partially_obs[pattern]
            grads_partially_obs_k = (
                grads_partially_obs[pattern] + pattern_shift_magnitude
            )

            debiasing_term_vec[k * num_regr_features : (k + 1) * num_regr_features] = (
                grads_fully_obs_k.mean(axis=0) - grads_partially_obs_k.mean(axis=0)
            ).reshape(-1, 1)
            fullyobs_vec[:, k * num_regr_features : (k + 1) * num_regr_features] = (
                grads_fully_obs_k
            ) + grads_fully_obs[self.fully_obs_tuple] @ inv_hessian.T @ (
                hessians_fully_obs_k - hessians_partially_obs_k
            ).T
            debiasing_var[
                k * num_regr_features : (k + 1) * num_regr_features,
                k * num_regr_features : (k + 1) * num_regr_features,
            ] = np.cov(grads_partially_obs_k.T) / len(pattern_ids)
        debiasing_var += np.cov(fullyobs_vec.T) / self.n_fully_obs

        eigenvalues, eigenvectors = np.linalg.eigh(debiasing_var)
        sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 1e-10))
        sqrt_var = eigenvectors @ np.diag(1.0 / sqrt_eigenvalues) @ eigenvectors.T
        standardized_debiased_term_sq = (
            np.linalg.norm(sqrt_var @ debiasing_term_vec) ** 2
        )
        if track_stats:
            return standardized_debiased_term_sq, chi2.sf(
                standardized_debiased_term_sq, num_patterns * num_regr_features
            )
        # chi-squared upper-tail p-value
        return chi2.sf(standardized_debiased_term_sq, num_patterns * num_regr_features)

    def test2_mcar_moment_conditions(
        self,
        shift_magnitudes: np.ndarray
        | None = None,  # should always be None for actual use; shift_weights is for testing
        track_stats: bool = False,  # should be False for actual use; track_stats is for testing
    ) -> float | tuple[float, float]:
        ## TODO: Check this is correct
        """
        Test for MCAR first moment condition for the current set up

        Args:
            None
        Returns:
            p-value: p-value for the single test
        Notes:
            This test is based on the debiasing term and variance calculation.
            The debiasing term is calculated as the difference between the fully observed data and the partially observed data.
            The variance is calculated as the covariance of the debiasing term.
            The standardized debiased term is calculated as the debiased term divided by the square root of the variance.
            If lambda weights were provided in a non-data-dependent way, this test is possibly conservative.
            The p-value is calculated as 2 * norm.cdf(-np.abs(standardized_debiased_term)).
        """

        ## check that pertinent values are computed
        assert (
            self._lambda_weights is not None
        ), "lambda_weights must be provided/calculated"
        assert self._theta_ipi is not None, "theta_ipi must be provided"
        assert self._var_ipi is not None, "var_ipi must be provided"
        assert (
            self._fully_obs_data_dict is not None
        ), "fully_obs_data_dict must be provided"
        assert (
            self._imputed_partially_obs_data is not None
        ), "imputed_partially_obs_data must be provided"
        assert self._pattern_to_ids is not None, "pattern_to_ids must be provided"
        assert self.target_idx is not None, "target_idx must be provided"
        assert self._theta_ipi is not None, "theta_ipi must be provided"

        ipi_grad_hessian = self.get_ipi_grad_and_hessian()

        grads_fully_obs = ipi_grad_hessian.grads_fully_obs
        hessians_fully_obs = ipi_grad_hessian.hessians_fully_obs
        grads_partially_obs = ipi_grad_hessian.grads_partially_obs
        hessians_partially_obs = ipi_grad_hessian.hessians_partially_obs

        num_regr_features = self._theta_ipi.shape[0]

        debiasing_term = np.zeros((num_regr_features, 1))
        debiasing_variance = np.zeros((num_regr_features, num_regr_features))
        weighted_grad_patterns_fully_obs = np.zeros(
            (self.n_fully_obs, num_regr_features)
        )
        inv_hessian_fully_obs = np.linalg.inv(hessians_fully_obs[self.fully_obs_tuple])

        if shift_magnitudes is not None:
            logger.info("Using shift magnitudes for test1_mcar_moment_conditions")

        for k, (pattern, pattern_ids) in enumerate(self.pattern_to_ids.items()):
            pattern_shift_magnitude = (
                shift_magnitudes[k] * np.ones((1, num_regr_features))
                if shift_magnitudes is not None
                else np.zeros((1, num_regr_features))
            )
            pattern_weight = self._lambda_weights[k]

            grad_fully_obs_k = grads_fully_obs[pattern]
            hessian_fully_obs_k = hessians_fully_obs[pattern]
            grad_partial_k = grads_partially_obs[pattern] + pattern_shift_magnitude
            hessian_partial_k = hessians_partially_obs[pattern]

            ## debiasing term per pattern calculation
            debias_term_k = (
                grad_fully_obs_k.mean(axis=0) - grad_partial_k.mean(axis=0)
            ).reshape(num_regr_features, 1)
            debiasing_term += pattern_weight * debias_term_k

            num_per_pattern_k = len(pattern_ids)

            weighted_grad_patterns_fully_obs += pattern_weight * grad_fully_obs_k
            weighted_grad_patterns_fully_obs += (
                pattern_weight
                * grads_fully_obs[self.fully_obs_tuple]
                @ inv_hessian_fully_obs.T
                @ (hessian_partial_k - hessian_fully_obs_k).T
            )

            debiasing_var_per_pattern = np.cov(grad_partial_k.T)
            debiasing_variance += (
                pattern_weight**2 / num_per_pattern_k * debiasing_var_per_pattern
            )

        debiasing_variance += (
            np.cov(weighted_grad_patterns_fully_obs.T) / self.n_fully_obs
        )

        # Calculate the standardized debiased term using matrix operations
        # For a multivariate normal distribution with covariance matrix Var,
        # the standardized statistic should be term^T * Var^(-1/2) * term
        # Which is equivalent to term / sqrt(Var) for univariate, but needs matrix operations for multivariate

        # Compute matrix square root (actually the inverse square root) of the covariance matrix
        # using eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(debiasing_variance)
        # Handle numerical stability for small eigenvalues
        sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 1e-10))
        sqrt_var = eigenvectors @ np.diag(1.0 / sqrt_eigenvalues) @ eigenvectors.T

        # Apply the transformation to the debiasing term
        standardized_debiased_term_sq = np.linalg.norm(sqrt_var @ debiasing_term) ** 2
        if track_stats:
            return standardized_debiased_term_sq, chi2.sf(
                standardized_debiased_term_sq, num_regr_features
            )
        return chi2.sf(standardized_debiased_term_sq, num_regr_features)

    @abstractmethod
    def train_model(self, train_data: np.ndarray, **kwargs) -> any:
        """
        Train the imputation model on the provided training data.

        The model is stored internally for use by impute_values() and other methods,
        but is also returned for direct access, debugging, or external storage.

        Args:
            train_data: Fully observed training data
            **kwargs: Additional keyword arguments specific to the imputation method

        Returns:
            The trained model object (also stored internally as self._imputation_model)
        """
        raise NotImplementedError("train_model must be implemented by subclasses")

    @abstractmethod
    def impute_values(self, test_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Impute missing values in the test data using the stored trained model.

        Args:
            test_data: Data with missing values to be imputed
            **kwargs: Additional keyword arguments specific to the imputation method

        Returns:
            Imputed data array

        Raises:
            ValueError: If no model has been trained yet (call train_model first)
        """
        raise NotImplementedError("impute_values must be implemented by subclasses")

    def setup_imputed_data(
        self,
        fully_obs_data_dict: Mapping[tuple[bool, ...], np.ndarray] | None = None,
        imputed_partially_obs_data: np.ndarray | None = None,
        pattern_to_ids: Mapping[tuple[bool, ...], np.ndarray] | None = None,
    ) -> None:
        """
        Set up the data dictionaries required for IPI computation.

        Args:
            fully_obs_data_dict: Dictionary mapping patterns to fully observed data arrays
            imputed_partially_obs_data: Imputed partially observed data array
        """
        if fully_obs_data_dict is not None:
            self._fully_obs_data_dict = fully_obs_data_dict
        if imputed_partially_obs_data is not None:
            self._imputed_partially_obs_data = imputed_partially_obs_data
        if pattern_to_ids is not None:
            self._pattern_to_ids = pattern_to_ids
        # Invalidate cached computations since data changed
        self._invalidate_cache()

    def fit_and_setup(
        self,
        train_data: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """
        Convenience method to train model and setup data in one step.

        Args:
            train_data: Training data for the imputation model
            **kwargs: Additional keyword arguments passed to train_model and impute_values
        """
        # Invalidate cache since we're retraining
        self._invalidate_cache()

        if self._imputation_model is None and train_data is None:
            raise ValueError(
                "train_data must be provided if pretrained model is not provided"
            )

        if self._imputation_model is None:
            # train the model (stores internally and returns model)
            self.train_model(train_data, **kwargs)
            if self._imputation_model is None:
                raise ValueError("Model must be trained before imputing values")

        # impute the partially observed data using stored model
        imputed_partially_obs_data = self.impute_values(
            self.partially_obs_data, **kwargs
        )

        # create the fully observed data dictionary
        fully_obs_data_dict = {self.fully_obs_tuple: self.fully_obs_data}
        # Add masked versions for each pattern
        for pattern in self.pattern_to_ids:
            masked_data = np.where(pattern, self.fully_obs_data, np.nan)
            fully_obs_data_dict[pattern] = self.impute_values(masked_data, **kwargs)

        self.setup_imputed_data(
            fully_obs_data_dict=fully_obs_data_dict,
            imputed_partially_obs_data=imputed_partially_obs_data,
        )

    def copy(self) -> "BaseIPI":
        """
        Create a copy of this IPI instance with the same configuration but fresh cache.

        Returns:
            A new instance of the same class with identical configuration
        """

        return self.__class__(
            regression_type=self.regression_type,
            fully_obs_data=self.fully_obs_data.copy(),
            partially_obs_data=self.partially_obs_data.copy(),
            regr_features_idxs=self.regr_features_idxs.copy(),
            outcome_idx=self.outcome_idx,
            target_idx=self.target_idx,
            pretrained_model=_safe_copy_model(self._imputation_model),
        )


class MF_IPI(BaseIPI):
    """IPI implementation using Random Forest for imputation."""

    def train_model(self, train_data: np.ndarray, **kwargs) -> any:
        categorical_features = kwargs["categorical_features"]
        num_cat_classes = kwargs["num_cat_classes"]
        column_names = kwargs["column_names"]
        do_compile = kwargs["do_compile"]

        trained_model_info = MissForest(
            categorical=categorical_features,
            num_cat_classes=num_cat_classes,
            verbose=-1,
            # with_wandb=True,
        )

        df_train = pd.DataFrame(train_data, columns=column_names)

        trained_model_info.fit(
            x=df_train,
            num_threads=-1,
            compile=do_compile,
        )

        self._imputation_model = trained_model_info

    def impute_values(self, test_data: np.ndarray, **kwargs) -> np.ndarray:
        column_names = kwargs["column_names"]

        if self._imputation_model is None:
            raise ValueError("Model must be trained before imputing values")
        if isinstance(self._imputation_model, bytes):
            self._imputation_model = MissForest.load_msgpack(self._imputation_model)

        test_df = pd.DataFrame(test_data, columns=column_names)

        # order columns by column_names order (because imputing is done by
        # ordering based on percentage of missingness and this is the default
        # order of columns)
        imputed_df = self._imputation_model.transform(test_df)[column_names]

        return imputed_df.to_numpy()


class Mean_IPI(BaseIPI):
    """IPI implementation using Mean imputation."""

    def train_model(self, train_data: np.ndarray, **_) -> None:
        mean = np.nanmean(train_data, axis=0)
        if mean is None:
            raise ValueError("Mean is None")
        self._imputation_model = mean

    def impute_values(self, test_data: np.ndarray, **_) -> np.ndarray:
        # fill in missing values with column means from self._imputation_model
        return np.where(np.isnan(test_data), self._imputation_model, test_data)


class Zero_IPI(BaseIPI):
    """IPI implementation using Zero imputation."""

    def train_model(self, train_data: np.ndarray, **_) -> None:
        del train_data
        # Store just the zero value, not an array with the shape of train_data
        self._imputation_model = 0.0

    def impute_values(self, test_data: np.ndarray, **_) -> np.ndarray:
        return np.where(np.isnan(test_data), self._imputation_model, test_data)


class Hot_Deck_IPI(BaseIPI):
    """IPI implementation using Hot Deck imputation."""

    def train_model(self, train_data: np.ndarray, **kwargs) -> None:
        n_neighbors = kwargs.get("n_neighbors", 5)
        distance_metric = kwargs.get("distance_metric", "nan_euclidean")
        weights = kwargs.get("weights", "uniform")

        # Validate inputs
        if train_data is None or train_data.size == 0:
            raise ValueError("Training data cannot be None or empty")

        # Check if we have enough samples for n_neighbors
        n_samples = train_data.shape[0]
        if n_samples < n_neighbors:
            logger.warning(
                f"Number of training samples ({n_samples}) is less than n_neighbors ({n_neighbors}). "
                f"Reducing n_neighbors to {n_samples - 1}"
            )
            n_neighbors = max(1, n_samples - 1)

        # Check for completely missing features in training data
        # n_features = train_data.shape[1]
        missing_per_feature = np.isnan(train_data).sum(axis=0)
        completely_missing_features = missing_per_feature == n_samples

        if completely_missing_features.any():
            missing_feature_indices = np.where(completely_missing_features)[0]
            logger.warning(
                f"Features {missing_feature_indices} are completely missing in training data. "
                "This may cause issues during imputation."
            )

        scaler = StandardScaler()
        scaled_train_data = scaler.fit_transform(train_data)

        self._imputation_model = {
            "knn": KNNImputer(
                n_neighbors=n_neighbors,
                metric=distance_metric,
                weights=weights,
            ).fit(scaled_train_data),
            "scaler": scaler,
        }

    def impute_values(self, test_data: np.ndarray, **_) -> np.ndarray:
        if self._imputation_model is None:
            raise ValueError("Model must be trained before imputing values")

        if test_data is None or test_data.size == 0:
            raise ValueError("Test data cannot be None or empty")

        # Check feature dimension consistency
        expected_n_features = len(self._imputation_model["scaler"].mean_)
        actual_n_features = test_data.shape[1]
        if actual_n_features != expected_n_features:
            raise ValueError(
                f"Test data has {actual_n_features} features, but model was trained with {expected_n_features} features"
            )

        scaled_test_data = self._imputation_model["scaler"].transform(test_data)
        imputed_data = self._imputation_model["knn"].transform(scaled_test_data)
        return self._imputation_model["scaler"].inverse_transform(imputed_data)


class EM_IPI(BaseIPI):
    """IPI implementation using Expectation Maximization for imputation."""

    def train_model(self, train_data: np.ndarray, **kwargs) -> np.ndarray:
        if self._imputation_model is not None:
            return self._imputation_model

        self._imputation_model = get_pretrained_parameters_EM(
            train_data=train_data,
            Q=kwargs.get("Q_est", 4),  # default to 4 factors
            tol=kwargs.get("tol", 1e-2),  # default to 1e-2
            max_iter=kwargs.get("max_iter", 100),  # default to 100
        )
        return self._imputation_model

    def impute_values(self, test_data: np.ndarray, **_) -> np.ndarray:
        """Impute using the stored EM model."""
        if self._imputation_model is None:
            raise ValueError("Model must be trained before imputing values")

        M_partial = (~np.isnan(test_data)).astype(int)  # 1 for observed, 0 for missing
        X_partial = test_data
        return fm_impute_X(X=X_partial, M=M_partial, cov_matrix=self._imputation_model)


# %%
def calc_lambda_weights(
    grad_patterns_fullyobs: Mapping[tuple[bool, ...], np.ndarray],
    grad_patterns_partial: Mapping[tuple[bool, ...], np.ndarray],
    hessian_patterns_fullyobs: Mapping[tuple[bool, ...], np.ndarray],
    hessian_patterns_partial: Mapping[tuple[bool, ...], np.ndarray],
    pattern_to_ids: Mapping[tuple[bool, ...], set[int]],
    target_idx: int,
    num_features: int,
    *,
    lambda_l2: float = 0.0,
    lambda_l1: float = 0.0,
    cp_iter: int = 1000,
    cp_tol: float = 1e-6,
) -> np.ndarray:
    """
    Calculate the lambda weights for the IPI-OLS method
    NOTE: currently only guarantees optimality for MCAR first and second moment settings

    Args:
        grad_patterns_fullyobs: gradient of the fully observed data
        grad_patterns_partial: gradient of the partially observed data
        hessian_patterns_fullyobs: hessian of the fully observed data
        hessian_patterns_partial: hessian of the partially observed data # NOTE: not used right now
        pattern_to_ids: pattern to ids
        num_regr_features: number of regression features
        target_idx: target index
    Returns:
        lambda_weights: lambda weights
    Additional Args (optional):
        lambda_l2: L2 (ridge) regularization strength on lambda. Adds lambda_l2 * I to the quadratic term.
        lambda_l1: L1 (lasso) regularization strength on lambda. If > 0, solved via convex optimization.
        cp_iter: Max iterations for the CVXPY solver when lambda_l1 > 0.
        cp_tol: Convergence tolerance for the CVXPY solver when lambda_l1 > 0.
    """
    if target_idx is None:
        raise Exception("target_idx must be specified")
    # Suppress linter warning for currently unused arg; kept for future extensions
    del hessian_patterns_partial

    # logger.info(
    #     "Calculating lambda weights for IPI-OLS -- not optimal if not MCAR first and second moment"
    # )
    num_missing_patterns = len(
        pattern_to_ids
    )  # Note that this excludes the fully-labeled pattern

    fully_obs_tuple = (True,) * num_features
    num_labeled, num_regr_features = grad_patterns_fullyobs[fully_obs_tuple].shape

    ## NOTE: We use hessian of just the fully observed points here as a proxy for the inverse Hessian
    ## This gives optimal lambda under MCAR first and second moment settings (including MCAR missing data)
    inv_hessian = np.linalg.inv(hessian_patterns_fullyobs[fully_obs_tuple])

    # vargrads_partial: num_masks x 1 containing the H^{-1}var(\nabla l^k(D))H^{-1}; k = 1, ..., num_masks
    # calculated using the partially observed dataset
    vargrads_partial = np.zeros(num_missing_patterns)

    # ndata_per_mask: (1 + num_masks) x 1 containing the number of data points in each missing pattern in entire dataset
    ndata_per_pattern = np.zeros(1 + num_missing_patterns)
    ndata_per_pattern[0] = num_labeled

    # vec: num_masks x 1 containing H^{-1}cov(\nabla l(D), \nabla l^k(D))H^{-1}; k = 1, ..., num_masks
    # calculated using fully observed dataset
    vec = np.zeros((num_missing_patterns, 1))

    # mat1: num_masks x num_masks containing H^{-1}cov(\nabla l^k(D), \nabla l^{k'}(D))H^{-1}; k, k' = 1, ..., num_masks
    # calculated using fully observed dataset
    mat1 = np.zeros((num_missing_patterns, num_missing_patterns))

    for (
        k1,
        (pattern1, idxs_in_pattern1),
    ) in enumerate(pattern_to_ids.items()):  # note pattern_k is 1 if **observed**
        grad_partial_mask_k = grad_patterns_partial[pattern1]

        # Check dimensions before computing covariance; if not enough data points, skip
        if grad_partial_mask_k.shape[0] < 30:
            raise ValueError(
                f"number of samples per pattern {k1} must be greater than 30"
            )

        cov_grad_k = np.cov(grad_partial_mask_k.T).reshape(
            num_regr_features, num_regr_features
        )

        # track the pertinent estimate of H^{-1}var(\nabla l^k(D))H^{-1}[target_idx, target_idx]
        vargrads_partial[k1] = (inv_hessian @ cov_grad_k @ inv_hessian)[
            target_idx, target_idx
        ]

        # track the number of data points in the missing pattern
        # Plus 1 because we include the fully-labeled pattern at index 0
        ndata_per_pattern[k1 + 1] = len(idxs_in_pattern1)

        # centered gradients
        grad_hat = grad_patterns_fullyobs[fully_obs_tuple] - grad_patterns_fullyobs[
            fully_obs_tuple
        ].mean(axis=0)
        grad_hat_k = grad_patterns_fullyobs[pattern1] - grad_patterns_fullyobs[
            pattern1
        ].mean(axis=0)

        cov_grad_k = (
            1 / (num_labeled - 1) * (grad_hat.T @ grad_hat_k + grad_hat_k.T @ grad_hat)
        )
        # calculate pertinent estimate of H^{-1}cov(\nabla l(D), \nabla l^k(D))H^{-1}[target_idx, target_idx]
        vec[k1, 0] = (inv_hessian @ cov_grad_k @ inv_hessian)[
            target_idx, target_idx
        ] / (2 * num_labeled)

        for k2, pattern2 in enumerate(pattern_to_ids.keys()):
            mat1[k1, k2] = (
                inv_hessian
                @ np.cov(
                    np.vstack(
                        (
                            grad_patterns_fullyobs[pattern1].T,
                            grad_patterns_fullyobs[pattern2].T,
                        )
                    )
                )[:num_regr_features, num_regr_features:]
                @ inv_hessian
            )[target_idx, target_idx] / num_labeled

    # Regularize the system for numerical stability in case of near-collinearity
    # among patterns; scale ridge by trace to be unitless and enforce SPD
    diag_term = np.diag(vargrads_partial / np.maximum(ndata_per_pattern[1:], 1))
    mat = mat1 + diag_term

    # If no regularization, return the original closed-form solution for backward compatibility
    if float(lambda_l1) == 0.0 and float(lambda_l2) == 0.0:
        # logger.info("No regularization, returning original closed-form solution")
        lambda_weights = np.linalg.solve(mat, vec)
        return lambda_weights.flatten()

    # If only L2 is requested, keep it lightweight and solve directly with ridge
    n_patterns = mat.shape[0]
    if float(lambda_l1) == 0.0 and float(lambda_l2) > 0.0:
        ridge_mat = mat + float(lambda_l2) * np.eye(n_patterns, dtype=mat.dtype)
        lambda_weights = np.linalg.solve(ridge_mat, vec)
        return lambda_weights.flatten()

    # Elastic Net (L1 and optional L2): use CVXPY
    b = vec.flatten()
    lam_var = cp.Variable(n_patterns)
    objective = (
        cp.quad_form(lam_var, mat)
        - b @ lam_var
        + float(lambda_l2) * cp.sum_squares(lam_var)
        + float(lambda_l1) * cp.norm1(lam_var)
    )
    prob = cp.Problem(cp.Minimize(objective))
    try:
        prob.solve(
            solver=cp.CLARABEL, max_iters=int(cp_iter), eps=float(cp_tol), verbose=False
        )
    except Exception:
        lam_val = None
    else:
        lam_val = lam_var.value
    if lam_val is None:
        # Fallback to ridge-only closed form if solver fails
        logger.info("CVXPY solver failed. Falling back to ridge-only closed form.")
        ridge_mat = mat + float(lambda_l2) * np.eye(n_patterns, dtype=mat.dtype)
        lambda_weights = np.linalg.solve(ridge_mat, vec)
        return lambda_weights.flatten()
    return np.asarray(lam_val, dtype=float).flatten()


def _safe_copy_model(model: object) -> object | None:
    """Attempt to copy arbitrary model objects robustly.

    Order of strategies:
    1) model.copy() if available
    2) Msgpack round-trip when save_msgpack/load_msgpack are available
    3) deepcopy
    4) pickle round-trip
    """
    if model is None:
        return None

    copy_method = getattr(model, "copy", None)
    if callable(copy_method):
        try:
            return copy_method()
        except Exception:
            pass

    save_msgpack = getattr(model, "save_msgpack", None)
    load_msgpack = getattr(model.__class__, "load_msgpack", None)
    if callable(save_msgpack) and callable(load_msgpack):
        try:
            return load_msgpack(save_msgpack())
        except Exception:
            pass

    try:
        return deepcopy(model)
    except Exception:
        pass

    try:
        return pickle.loads(pickle.dumps(model))
    except Exception as exc:
        raise TypeError(f"Unable to copy model of type {type(model)!r}") from exc
