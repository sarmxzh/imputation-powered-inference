# %%
import logging
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold

from ipi.ipi_methods import calc_lambda_weights
from ipi.missforest.src.missforest import MissForest
from ipi.utils.missing_data_utils import get_missing_patterns_and_ids
from ipi.utils.stats_utils import get_gradients, get_point_estimate, get_sample_hessian

logger = logging.getLogger(__name__)


# %%
def bootstrap_samples(
    n: int,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample from entire population with replacement

    Args:
        n: number of labeled samples
        N: number of partially labeled samples

    Returns:
        labeled_idxs: indices of labeled samples
        partial_idxs: indices of partially labeled samples
    """

    idxs = np.random.choice(a=n + N, size=n + N, replace=True)
    labeled_idxs = idxs[idxs < n]
    partial_idxs = idxs[idxs >= n] - n
    return labeled_idxs, partial_idxs


# %%
class BaseCrossIPI(ABC):
    """
    Base class for Cross-Fitting Improved Prediction Interval (Cross-IPI) methods.

    Cross-IPI differs from standard IPI by using cross-fitting to train multiple models
    instead of training a single model on independent data. This approach reduces
    overfitting risk but requires more computation.

    Key Differences from BaseIPI:
    - train_model() RETURNS a model object instead of storing it
    - impute_values() REQUIRES a model parameter instead of using stored model
    - No persistent model storage (stateless with respect to trained models)
    - Designed for cross-fitting workflows where models are trained per fold

    Workflow:
        1. For each cross-validation fold:
           a. Call train_model(fold_data) -> returns trained_model
           b. Call impute_values(test_data, trained_model) -> returns imputed_data
        2. Aggregate results across folds

    Args:
        regression_type: Type of regression ("ols" or "logistic")
        fully_obs_data: Fully observed data array
        partially_obs_data: Partially observed data array (with missing values)
        regr_features_idxs: Indices of regression features
        outcome_idx: Index of the outcome variable
        target_idx: Index of the target coefficient for inference
    """

    def __init__(
        self,
        regression_type: Literal["ols", "logistic"],
        fully_obs_data: np.ndarray,
        partially_obs_data: np.ndarray,
        regr_features_idxs: list[int],
        outcome_idx: int,
        target_idx: int,
    ):
        self.fully_obs_data = fully_obs_data
        self.partially_obs_data = partially_obs_data
        self.regr_features_idxs = regr_features_idxs
        self.outcome_idx = outcome_idx
        self.target_idx = target_idx

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
            raise ValueError("target_idx must be in range of regr_features_idxs")

        # Public attributes - user configuration and input data
        self.regression_type = regression_type
        self.fully_obs_data = fully_obs_data
        self.partially_obs_data = partially_obs_data
        self.regr_features_idxs = regr_features_idxs
        self.outcome_idx = outcome_idx
        self.target_idx = target_idx

        # Private attributes - cached computational results
        self._theta_cipi = None
        self._var_cipi = None
        self._lambda_weights = None
        self._cipi_computation_valid = False

        # Private attributes - internal state
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
        self.num_regr_features = len(self.regr_features_idxs)
        self.num_features = self.fully_obs_data.shape[1]
        self.num_patterns = len(self._pattern_to_ids)
        self.fully_obs_tuple = tuple(True for _ in range(self.num_features))

    @property
    def theta_cipi(self) -> np.ndarray | None:
        """Get the cached CIPI point estimate, or None if not computed."""
        return self._theta_cipi

    @property
    def var_cipi(self) -> np.ndarray | None:
        """Get the cached CIPI variance estimate, or None if not computed."""
        return self._var_cipi

    @property
    def lambda_weights(self) -> np.ndarray | None:
        """Get the cached lambda weights for IPI, or None if not computed."""
        return self._lambda_weights

    def _invalidate_cache(self) -> None:
        """Invalidate cached computation results."""
        logger.info("Invalidating cached computation results")
        self._theta_cipi = None
        self._var_cipi = None
        self._lambda_weights = None
        self._cipi_computation_valid = False

    def get_bootstrap_variance_estimate(
        self,
        num_folds: int = 5,
        num_bootstrap_trials: int = 50,
        lambda_weights: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Get the bootstrap variance estimate for the CIPI point estimate.

        Args:
            num_bootstrap_trials: Number of bootstrap trials to run
            kfold_seed: Seed for the k-fold cross-validation

        Returns:
            BootstrapVarianceEstimate: A dataclass containing the bootstrap variance estimate
        """
        # Numbers of datapoints in each fold
        fold_num_labeled = self.n_fully_obs // num_folds
        fold_num_partial = self.n_partially_obs // num_folds

        # Number of datapoints outside each fold
        train_num_labeled = self.n_fully_obs - fold_num_labeled
        train_num_partial = self.n_partially_obs - fold_num_partial

        ## collect average oob gradients over all bootstrap trials
        grad_partially_obs_avg = np.zeros(
            (self.n_partially_obs, self.num_regr_features)
        )
        grad_patterns_fully_obs_avg = np.zeros(
            (self.n_fully_obs, self.num_regr_features, self.num_patterns + 1)
        )
        num_oob_fully_obs = np.zeros(self.n_fully_obs)
        num_oob_partially_obs = np.zeros(self.n_partially_obs)

        for _ in range(num_bootstrap_trials):
            # logger.info("--- bootstrap trial {%d} ---", i)
            # --- Sample train-test split ---
            # Indices of samples in the bootstrap trial
            bs_labeled_indices, bs_partial_indices = bootstrap_samples(
                n=self.n_fully_obs,
                N=self.n_partially_obs,
            )

            train_fully_obs_idxs = bs_labeled_indices[:train_num_labeled]
            train_partially_obs_idxs = bs_partial_indices[:train_num_partial]

            train_data = np.vstack(
                [
                    self.fully_obs_data[train_fully_obs_idxs],
                    self.partially_obs_data[train_partially_obs_idxs],
                ]
            )

            trained_model = self.train_model(
                train_data=train_data,
                **kwargs,
            )

            fold_oob_labeled_idxs = np.setdiff1d(
                np.arange(self.n_fully_obs), bs_labeled_indices
            )
            fold_oob_partial_idxs = np.setdiff1d(
                np.arange(self.n_partially_obs), bs_partial_indices
            )

            # tick number of oob by 1
            num_oob_fully_obs[fold_oob_labeled_idxs] += 1
            num_oob_partially_obs[fold_oob_partial_idxs] += 1

            ## add partially observed oob data
            # impute the partially observed data
            imputed_partially_obs_oob_data = self.impute_values(
                test_data=self.partially_obs_data[fold_oob_partial_idxs],
                model=trained_model,
                **kwargs,
            )

            # add the gradient of the partially observed data to the average
            grad_partially_obs_avg[fold_oob_partial_idxs, :] += get_gradients(
                data_array=imputed_partially_obs_oob_data,
                regr_feature_idxs=self.regr_features_idxs,
                outcome_idx=self.outcome_idx,
                theta=self.theta_init,
                regression_type=self.regression_type,
            )

            ## deal with fully observed oob data
            # compute the gradient of the imputed partially observed data
            grad_patterns_fully_obs_avg[fold_oob_labeled_idxs, :, 0] += get_gradients(
                data_array=self.fully_obs_data[fold_oob_labeled_idxs],
                regr_feature_idxs=self.regr_features_idxs,
                outcome_idx=self.outcome_idx,
                theta=self.theta_init,
                regression_type=self.regression_type,
            )

            for k, pattern in enumerate(self._pattern_to_ids):
                # Mask out the pattern
                masked_fully_obs_oob_data_pattern_k = self.fully_obs_data[
                    fold_oob_labeled_idxs
                ].copy()
                masked_fully_obs_oob_data_pattern_k[:, ~np.array(pattern)] = np.nan

                dfhat_fully_obs_oob_data_pattern_k = self.impute_values(
                    test_data=masked_fully_obs_oob_data_pattern_k,
                    model=trained_model,
                    **kwargs,
                )

                # Adding to \nabla l^k(D), where we have +1 to the index to account for the fully observed pattern
                grad_patterns_fully_obs_avg[fold_oob_labeled_idxs, :, k + 1] += (
                    get_gradients(
                        data_array=dfhat_fully_obs_oob_data_pattern_k,
                        regr_feature_idxs=self.regr_features_idxs,
                        outcome_idx=self.outcome_idx,
                        theta=self.theta_init,
                        regression_type=self.regression_type,
                    )
                )
        # Scale all the gradients by the number of times each datapoint was OOB
        denominator = num_oob_fully_obs[:, None, None]
        scaling_factor = np.zeros_like(
            num_oob_fully_obs[:, None, None], dtype=grad_patterns_fully_obs_avg.dtype
        )
        # Set the scaling factor to 1 / num_oob_labeled, with 1/0 = 0
        np.divide(1, denominator, out=scaling_factor, where=(denominator != 0))
        grad_patterns_fully_obs_avg *= scaling_factor

        # Same for partially labeled data
        denominator = num_oob_partially_obs[:, None]
        scaling_factor = np.zeros_like(
            num_oob_partially_obs[:, None], dtype=grad_partially_obs_avg.dtype
        )
        np.divide(1, denominator, out=scaling_factor, where=(denominator != 0))
        grad_partially_obs_avg *= scaling_factor

        # calculates the Hessian of the fully observed data
        # consistent estimate of the Hessian of the PPI loss if assumptions are met
        hessian = get_sample_hessian(
            data_array=self.fully_obs_data,
            regr_feature_idxs=self.regr_features_idxs,
            outcome_idx=self.outcome_idx,
            theta=self.theta_init,
            regression_type=self.regression_type,
        ).reshape(self.num_regr_features, self.num_regr_features)
        inv_hessian = np.linalg.inv(hessian)

        self._lambda_weights = lambda_weights
        if self._lambda_weights is None:
            # Create the pattern-based dictionaries that calc_lambda_weights expects
            fully_obs_tuple = tuple(True for _ in range(self.num_features))

            # Create grad_patterns_fullyobs dictionary
            grad_patterns_fullyobs = {
                fully_obs_tuple: grad_patterns_fully_obs_avg[:, :, 0]
            }
            for k, pattern in enumerate(self._pattern_to_ids):
                grad_patterns_fullyobs[pattern] = grad_patterns_fully_obs_avg[
                    :, :, k + 1
                ]

            # Create grad_patterns_partial dictionary
            grad_patterns_partial = {}
            for _, pattern in enumerate(self._pattern_to_ids):
                pattern_idxs = list(self._pattern_to_ids[pattern])
                grad_patterns_partial[pattern] = grad_partially_obs_avg[pattern_idxs, :]

            # Create hessian dictionaries (use the same hessian for all patterns)
            hessian_patterns_fullyobs = {fully_obs_tuple: hessian}
            hessian_patterns_partial = {fully_obs_tuple: hessian}
            for pattern in self._pattern_to_ids:
                hessian_patterns_fullyobs[pattern] = hessian
                hessian_patterns_partial[pattern] = hessian

            self._lambda_weights = calc_lambda_weights(
                grad_patterns_fullyobs=grad_patterns_fullyobs,
                grad_patterns_partial=grad_patterns_partial,
                hessian_patterns_fullyobs=hessian_patterns_fullyobs,
                hessian_patterns_partial=hessian_patterns_partial,
                pattern_to_ids=self._pattern_to_ids,
                target_idx=self.target_idx,
                num_features=self.num_features,
            )

        # Calculate the asymptotic variance

        # delta term = \nabla l (D) - \sum_k \lambda_k \nabla l_k(D)
        delta_term = grad_patterns_fully_obs_avg[:, :, 0]

        # weighted_avg_covgrad = \sum_k \lambda_k^2 \frac{\Var(\nabla l_k(D))}{N_k}]
        # calculated with df_partial
        # N_k is number of observed missing patterns in df_partial
        weighted_avg_covgrad = np.zeros(
            (self.num_regr_features, self.num_regr_features)
        )
        # for k in range(len(missing_patterns_count)):
        for k, pattern in enumerate(self._pattern_to_ids):
            delta_term -= (
                self._lambda_weights[k] * grad_patterns_fully_obs_avg[:, :, k + 1]
            )

            # get the datapoint indices in df_partial corresponding to the missing pattern
            pattern_idxs = self._pattern_to_ids[pattern]

            # get the gradient of the partially labeled data for the missing pattern
            grad_partially_obs_mask_k = grad_partially_obs_avg[
                list(pattern_idxs)
            ].reshape(-1, self.num_regr_features)
            # Check dimensions before computing covariance; if not enough data points, skip
            if grad_partially_obs_mask_k.shape[0] < 2:
                logger.warning(
                    f"Not enough data points for pattern {pattern}. Skipping covariance computation."
                )
                continue

            # compute the covariance of the gradient of the partially observed data for the missing pattern
            cov_grad_k = np.cov(grad_partially_obs_mask_k.T).reshape(
                self.num_regr_features, self.num_regr_features
            )

            # add \lambda_k^2 \frac{\Var(\nabla l_k(D))}{N_k} to the weighted average
            weighted_avg_covgrad += (
                self._lambda_weights[k] ** 2 * cov_grad_k / len(pattern_idxs)
            )

        cov_delta_term = (
            np.cov(delta_term.T).reshape(self.num_regr_features, self.num_regr_features)
            / self.n_fully_obs
        )
        return inv_hessian @ (weighted_avg_covgrad + cov_delta_term) @ inv_hessian

    def get_ci_cipi(
        self,
        num_folds: int = 5,
        num_bootstrap_trials: int = 50,
        lambda_weights: np.ndarray | None = None,
        alpha: float = 0.05,
        cf_random_state: int = 42,
        **kwargs,
    ) -> tuple[float, float]:
        """
        Get the confidence interval for the CIPI point estimate.
        """
        # Calculate bootstrap variance and find optimal weights
        var_theta_hat = self.get_bootstrap_variance_estimate(
            num_folds=num_folds,
            num_bootstrap_trials=num_bootstrap_trials,
            lambda_weights=lambda_weights,
            **kwargs,
        )

        # Setup K-fold cross-validation
        kf = KFold(
            n_splits=num_folds,
            shuffle=True,
            random_state=cf_random_state,
        )

        # Get labeled and partial folds
        kfold_split_labeled = list(kf.split(self.fully_obs_data))
        kfold_split_partial = list(kf.split(self.partially_obs_data))

        # Get patterns and IDs
        hessian_L = get_sample_hessian(
            data_array=self.fully_obs_data,
            regr_feature_idxs=self.regr_features_idxs,
            outcome_idx=self.outcome_idx,
            theta=self.theta_init,
            regression_type=self.regression_type,
        )
        grad_L = np.mean(
            get_gradients(
                data_array=self.fully_obs_data,
                regr_feature_idxs=self.regr_features_idxs,
                outcome_idx=self.outcome_idx,
                theta=self.theta_init,
                regression_type=self.regression_type,
            ),
            axis=0,
        ).reshape(self.num_regr_features, 1)

        for j in range(num_folds):
            # Training step
            # get all but jth fold
            train_j_labeled_idxs = kfold_split_labeled[j][0]
            train_j_partial_idxs = kfold_split_partial[j][0]

            # Convert NumPy arrays to pandas DataFrames for model training
            train_data = np.vstack(
                [
                    self.fully_obs_data[train_j_labeled_idxs],
                    self.partially_obs_data[train_j_partial_idxs],
                ]
            )

            trained_model_info = self.train_model(
                train_data=train_data,
                **kwargs,
            )
            # Validation step
            # get jth fold
            fold_j_labeled_idxs = kfold_split_labeled[j][1]
            fold_j_partial_idxs = kfold_split_partial[j][1]

            df_labeled_fold = self.fully_obs_data[fold_j_labeled_idxs, :]
            df_partial_fold = self.partially_obs_data[fold_j_partial_idxs, :]

            # Convert NumPy array to pandas DataFrame for imputation
            dfhat_partial_fold = self.impute_values(
                test_data=df_partial_fold,
                model=trained_model_info,
                **kwargs,
            )

            # Ensure it's a numpy array (handle both numpy arrays and DataFrames)
            if hasattr(dfhat_partial_fold, "to_numpy"):
                dfhat_partial_fold = dfhat_partial_fold.to_numpy()
            else:
                dfhat_partial_fold = np.asarray(dfhat_partial_fold)

            for k, pattern in enumerate(self._pattern_to_ids):
                # Create masked version of labeled data
                dfmasked_labeled_fold = df_labeled_fold.copy()
                # Set columns to NaN where pattern is False
                mask_cols = ~np.array(pattern)
                dfmasked_labeled_fold[:, mask_cols] = np.nan

                # get the indices of the datapoints in the partially observed fold that have the missing pattern pattern
                mask_k_idxs = list(
                    self._pattern_to_ids[pattern].intersection(set(fold_j_partial_idxs))
                )

                # Skip if no datapoints with this pattern in this fold
                if not mask_k_idxs:
                    continue

                # Get partial data with this pattern
                dfhat_partial_fold_k = dfhat_partial_fold[
                    [fold_j_partial_idxs.tolist().index(idx) for idx in mask_k_idxs]
                ]

                # Impute missing values in masked labeled data
                dfhat_labeled_fold_k = self.impute_values(
                    test_data=dfmasked_labeled_fold,
                    model=trained_model_info,
                    **kwargs,
                )

                # Ensure it's a numpy array (handle both numpy arrays and DataFrames)
                if hasattr(dfhat_labeled_fold_k, "to_numpy"):
                    dfhat_labeled_fold_k = dfhat_labeled_fold_k.to_numpy()
                else:
                    dfhat_labeled_fold_k = np.asarray(dfhat_labeled_fold_k)

                # Update hessian and gradient
                hessian_L += (
                    self._lambda_weights[k]
                    / num_folds
                    * (
                        get_sample_hessian(
                            data_array=dfhat_partial_fold_k,
                            regr_feature_idxs=self.regr_features_idxs,
                            outcome_idx=self.outcome_idx,
                            theta=self.theta_init,
                            regression_type=self.regression_type,
                        ).reshape(self.num_regr_features, self.num_regr_features)
                        - get_sample_hessian(
                            data_array=dfhat_labeled_fold_k,
                            regr_feature_idxs=self.regr_features_idxs,
                            outcome_idx=self.outcome_idx,
                            theta=self.theta_init,
                            regression_type=self.regression_type,
                        ).reshape(self.num_regr_features, self.num_regr_features)
                    )
                )

                # adding \lambda_k * (\nabla l^k(\tilde{D}) - \nabla l^k(D)) for the jrth fold
                grad_L += (
                    self._lambda_weights[k]
                    / num_folds
                    * (
                        np.mean(
                            get_gradients(
                                data_array=dfhat_partial_fold_k,
                                regr_feature_idxs=self.regr_features_idxs,
                                outcome_idx=self.outcome_idx,
                                theta=self.theta_init,
                                regression_type=self.regression_type,
                            ),
                            axis=0,
                        ).reshape(self.num_regr_features, 1)
                        - np.mean(
                            get_gradients(
                                data_array=dfhat_labeled_fold_k,
                                regr_feature_idxs=self.regr_features_idxs,
                                outcome_idx=self.outcome_idx,
                                theta=self.theta_init,
                                regression_type=self.regression_type,
                            ),
                            axis=0,
                        ).reshape(self.num_regr_features, 1)
                    )
                )

        # Calculate final estimate and confidence interval
        inv_L = np.linalg.inv(hessian_L)

        theta_hat = (
            self.theta_init.reshape(self.num_regr_features, -1) - inv_L @ grad_L
        )[self.target_idx]
        se_theta_hat = np.sqrt(var_theta_hat[self.target_idx, self.target_idx])
        halfwidth = norm.ppf(1 - alpha / 2) * se_theta_hat

        return [float(theta_hat - halfwidth), float(theta_hat + halfwidth)]

    @abstractmethod
    def train_model(self, train_data: np.ndarray, **kwargs) -> any:
        """
        Train the imputation model on the provided training data.

        For cross-fitting, this method will be called multiple times (once per fold)
        with different training data. Each call returns a new trained model.

        Args:
            train_data: Fully observed training data for this fold
            **kwargs: Additional keyword arguments specific to the imputation method

        Returns:
            Trained model object (type depends on implementation)
        """
        raise NotImplementedError("train_model must be implemented by subclasses")

    @abstractmethod
    def impute_values(self, test_data: np.ndarray, model: any, **kwargs) -> np.ndarray:
        """
        Impute missing values in the test data using the provided trained model.

        Args:
            test_data: Data with missing values to be imputed
            model: Trained model object returned from train_model()
            **kwargs: Additional keyword arguments specific to the imputation method

        Returns:
            Imputed data array
        """
        raise NotImplementedError("impute_values must be implemented by subclasses")


class EM_CrossIPI(BaseCrossIPI):
    """Cross-fitting IPI implementation using Expectation Maximization for imputation."""

    def train_model(self, train_data: np.ndarray, **kwargs) -> np.ndarray:
        """Train EM model and return the trained model."""
        from ipi.utils.factor_model_em_utils import get_pretrained_parameters_EM

        return get_pretrained_parameters_EM(
            train_data=train_data,
            Q=kwargs.get("Q", 4),  # default to 4 factors
            tol=kwargs.get("tol", 1e-2),  # default to 1e-2
            max_iter=kwargs.get("max_iter", 100),  # default to 100
        )

    def impute_values(
        self, test_data: np.ndarray, model: np.ndarray, **_
    ) -> np.ndarray:
        """Impute using the provided EM model."""
        if model is None:
            raise ValueError("Model must be trained before imputing values")
        from ipi.utils.factor_model_em_utils import fm_impute_X

        M_partial = (~np.isnan(test_data)).astype(int)  # 1 for observed, 0 for missing
        X_partial = test_data
        return fm_impute_X(X=X_partial, M=M_partial, cov_matrix=model)


class MF_CrossIPI(BaseCrossIPI):
    """Cross-fitting IPI implementation using Random Forest for imputation."""

    def train_model(self, train_data: np.ndarray, **kwargs) -> any:
        categorical_features = kwargs["categorical_features"]
        num_cat_classes = kwargs["num_cat_classes"]
        column_names = kwargs["column_names"]
        do_compile = kwargs.get("do_compile", False)

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

        return trained_model_info

    def impute_values(self, test_data: np.ndarray, model: any, **kwargs) -> np.ndarray:
        column_names = kwargs["column_names"]

        if model is None:
            raise ValueError("Model must be trained before imputing values")
        if isinstance(model, bytes):
            model = MissForest.load_msgpack(model)

        test_df = pd.DataFrame(test_data, columns=column_names)

        # order columns by column_names order (because imputing is done by
        # ordering based on percentage of missingness and this is the default
        # order of columns)
        imputed_df = model.transform(test_df)[column_names]

        return imputed_df.to_numpy()
