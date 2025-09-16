import logging
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from ipi.missforest.src.missforest import MissForest
from ipi.utils.factor_model_em_utils import fm_impute_X, get_pretrained_parameters_EM
from ipi.utils.stats_utils import get_point_estimate

logger = logging.getLogger(__name__)


def classical_ci(
    fully_obs_data: np.ndarray,
    regr_features_idxs: list[int],
    outcome_idx: int,
    target_idx: int,
    regression_type: Literal["ols", "logistic"],
    alpha: float = 0.1,
) -> tuple[float, float]:
    theta_hat, se = get_point_estimate(
        data_array=fully_obs_data,
        regr_feature_idxs=regr_features_idxs,
        outcome_idx=outcome_idx,
        regression_type=regression_type,
        return_se=True,
    )

    theta_hat_coord = theta_hat[target_idx]
    se_theta_hat = se[target_idx]
    halfwidth = norm.ppf(1 - alpha / 2) * se_theta_hat

    return (theta_hat_coord - halfwidth, theta_hat_coord + halfwidth)


class Naive_CI(ABC):
    def __init__(
        self,
        regression_type: Literal["ols", "logistic"],
        fully_obs_data: np.ndarray,
        partially_obs_data: np.ndarray,
        regr_features_idxs: list[int],
        outcome_idx: int,
        target_idx: int,
        pretrained_model: object | None = None,
        inplace: bool = False,
    ):
        self.regression_type = regression_type
        self.fully_obs_data = fully_obs_data
        self.partially_obs_data = partially_obs_data
        self.regr_features_idxs = regr_features_idxs
        self.outcome_idx = outcome_idx
        self.target_idx = target_idx
        self._imputation_model = pretrained_model
        self.single_imputed_data = None
        self.inplace = inplace

    def get_naive_ci(self, alpha: float = 0.1) -> tuple[float, float]:
        if self.single_imputed_data is None:
            raise ValueError("single_imputed_data must be set before getting naive CI")

        ## compute classical interval as if imputed data was fully observed
        return classical_ci(
            fully_obs_data=self.single_imputed_data,
            regr_features_idxs=self.regr_features_idxs,
            outcome_idx=self.outcome_idx,
            target_idx=self.target_idx,
            regression_type=self.regression_type,
            alpha=alpha,
        )

    @abstractmethod
    def train_model(self) -> None:
        raise NotImplementedError("train_model must be implemented by subclasses")

    @abstractmethod
    def impute_values(self) -> None:
        raise NotImplementedError("train_model must be implemented by subclasses")

    def fit_and_setup(self, train_data: np.ndarray | None = None, **kwargs) -> None:
        if self.inplace:
            logger.info("Fitting model in place")
            train_data = np.vstack([self.fully_obs_data, self.partially_obs_data])
            self.train_model(train_data, **kwargs)
        else:
            if self._imputation_model is None:
                train_data = train_data
                if train_data is None:
                    raise ValueError(
                        "train_data must be provided if pretrained model is not provided"
                    )

                # Check for row overlap between train_data and (fully_obs_data and partially_obs_data)
                logger.info("Training model from scratch using train_data")
                self.train_model(train_data, **kwargs)
            else:
                logger.info("Using pretrained model")

        imputed_partially_obs_data = self.impute_values(
            self.partially_obs_data, **kwargs
        )
        self.single_imputed_data = np.vstack(
            [self.fully_obs_data, imputed_partially_obs_data]
        )


class EM_Naive_CI(Naive_CI):
    def train_model(self, train_data: np.ndarray, **kwargs) -> np.ndarray:
        if self._imputation_model is not None:
            return self._imputation_model

        logger.info(
            "Covariance matrix not found. Training EM model with Q_est=%s",
            kwargs.get("Q_est", 4),
        )
        self._imputation_model = get_pretrained_parameters_EM(
            train_data=train_data,
            Q=kwargs.get("Q_est", 4),  # default to 4 factors
            tol=kwargs.get("tol", 1e-2),  # default to 1e-2
            max_iter=kwargs.get("max_iter", 100),  # default to 100
        )
        logger.info("EM model trained successfully")
        return self._imputation_model

    def impute_values(self, test_data: np.ndarray, **_) -> np.ndarray:
        """Impute using the stored EM model."""
        if self._imputation_model is None:
            raise ValueError("Model must be trained before imputing values")

        M_partial = (~np.isnan(test_data)).astype(int)  # 1 for observed, 0 for missing
        X_partial = test_data
        return fm_impute_X(X=X_partial, M=M_partial, cov_matrix=self._imputation_model)


class MF_Naive_CI(Naive_CI):
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


class Hot_Deck_Naive_CI(Naive_CI):
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

        logger.info(
            f"Training Hot Deck model with n_neighbors={n_neighbors}, distance_metric={distance_metric}, weights={weights}"
        )

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


class Mean_Naive_CI(Naive_CI):
    def train_model(self, train_data: np.ndarray, **_) -> None:
        mean = np.nanmean(train_data, axis=0)
        if mean is None:
            raise ValueError("Mean is None")
        self._imputation_model = mean

    def impute_values(self, test_data: np.ndarray, **_) -> np.ndarray:
        # fill in missing values with column means from self._imputation_model
        return np.where(np.isnan(test_data), self._imputation_model, test_data)


class Zero_Naive_CI(Naive_CI):
    def train_model(self, train_data: np.ndarray, **_) -> None:
        del train_data
        # Store just the zero value, not an array with the shape of train_data
        self._imputation_model = 0.0

    def impute_values(self, test_data: np.ndarray, **_) -> np.ndarray:
        return np.where(np.isnan(test_data), self._imputation_model, test_data)
