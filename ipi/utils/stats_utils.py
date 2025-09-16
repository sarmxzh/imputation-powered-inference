from typing import Literal

import numpy as np
from numba import njit
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS


def get_features_and_outcome(
    data_array: np.ndarray,
    regr_feature_idxs: list[int],
    outcome_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the features and outcome from the data array
    """
    X = data_array[:, regr_feature_idxs]
    y = data_array[:, outcome_idx]

    # check for nan in regression features
    if np.isnan(X).any():
        raise ValueError("Missing values found in regression features")

    # check for nan in outcome
    if np.isnan(y).any():
        raise ValueError("Missing values found in outcome variable")

    return X, y


@njit
def safe_expit(x: np.ndarray) -> np.ndarray:
    ## from https://github.com/aangelopoulos/ppi_py/blob/main/ppi_py/utils/statistics_utils.py#L63
    """Computes the sigmoid function in a numerically stable way."""
    return np.exp(-np.logaddexp(0, -x))


def get_point_estimate(
    data_array: np.ndarray,
    regr_feature_idxs: list[int],
    outcome_idx: int,
    regression_type: Literal["ols", "logistic"],
    return_se: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Get the point estimate for the OLS model
    """
    ## get the features and outcome
    X, y = get_features_and_outcome(
        data_array=data_array,
        regr_feature_idxs=regr_feature_idxs,
        outcome_idx=outcome_idx,
    )
    # convert to lowercase
    regression_type = regression_type.lower()

    ## calculate the point estimate
    ## adapted from https://github.com/aangelopoulos/ppi_py/blob/main/ppi_py/ppi.py
    if regression_type == "ols":
        regression = OLS(y, exog=X).fit()
    elif regression_type == "logistic":
        regression = Logit(y, exog=X).fit()
    else:
        raise ValueError(f"Invalid regression type: {regression_type}")

    theta = regression.params
    if return_se:
        return theta, regression.bse
        ## unlike ppi_py, not computing HC0_se
    return theta


def get_gradients(
    data_array: np.ndarray,
    regr_feature_idxs: list[int],
    outcome_idx: int,
    theta: np.ndarray,
    regression_type: Literal["ols", "logistic"],
) -> np.ndarray:
    """
    Calculate the gradient of the OLS objective function
    Args:
        data_array: numpy array (n x d)
        regr_features_idxs: list of feature indices (p dimensional)
        outcome_idx: outcome index (1 dimensional)
        theta: parameter estimates (p x 1)
    Returns:
        grad: gradient of the OLS objective function for each observation (n x p array)
    """
    ## get the features and outcome
    X, y = get_features_and_outcome(
        data_array=data_array,
        regr_feature_idxs=regr_feature_idxs,
        outcome_idx=outcome_idx,
    )

    regression_type = regression_type.lower()

    n = y.shape[0]

    grads = np.zeros(X.shape)
    for i in range(n):
        if regression_type == "ols":
            grads[i, :] = (np.dot(X[i, :], theta) - y[i]) * X[i, :]
        elif regression_type == "logistic":
            grads[i, :] = (safe_expit(X[i, :] @ theta) - y[i]) * X[i, :]
        else:
            raise ValueError(f"Invalid regression type: {regression_type}")
    return grads


def get_sample_hessian(
    data_array: np.ndarray,
    regr_feature_idxs: list[int],
    outcome_idx: int,
    theta: np.ndarray,
    regression_type: Literal["ols", "logistic"],
) -> np.ndarray:
    """
    Calculate the Hessian of the OLS objective function
    Args:
        data_array: numpy array (n x d)
        regr_features_idxs: list of feature indices (p dimensional)
    Returns:
        hessian: sample mean of the Hessian of the OLS objective function (p x p)
    """
    X, y = get_features_and_outcome(
        data_array=data_array,
        regr_feature_idxs=regr_feature_idxs,
        outcome_idx=outcome_idx,
    )

    regression_type = regression_type.lower()

    if regression_type == "ols":
        return X.T @ X / X.shape[0]
    if regression_type == "logistic":
        probs = safe_expit(X @ theta)
        weights = probs * (1 - probs)
        return X.T @ (weights[:, None] * X) / X.shape[0]
    raise ValueError(f"Invalid regression type: {regression_type}")


def bootstrap_samples(
    n: int, N: int, m_out_of_n_exp: float | None = None
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
    boot_size = int((n + N) ** (m_out_of_n_exp if m_out_of_n_exp is not None else 1))

    idxs = np.random.choice(a=n + N, size=boot_size, replace=True)
    labeled_idxs = idxs[idxs < n]
    partial_idxs = idxs[idxs >= n] - n
    return labeled_idxs, partial_idxs
