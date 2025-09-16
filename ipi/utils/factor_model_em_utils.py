import numpy as np
import polars as pl
from scipy.stats import multivariate_normal


def calculate_log_likelihood_X_only(
    X: np.ndarray,
    # C_ff: np.ndarray,
    Lamda_hat: np.ndarray,
    Psi_hat: np.ndarray,
    index_groups: pl.DataFrame,
    M: np.ndarray[np.bool],
) -> float:
    """Calculated marginal log likelihood of the data given the observed data for FA model"""
    n, d = X.shape

    # Ensure Psi_hat is positive to maintain positive definiteness
    eps = 1e-6
    Psi_hat_clipped = np.maximum(Psi_hat, eps)

    Sigma = Lamda_hat @ Lamda_hat.T + np.diag(Psi_hat_clipped)

    # Add small regularization to diagonal for numerical stability
    Sigma += eps * np.eye(Sigma.shape[0])

    log_likelihood = 0
    for row in index_groups.iter_rows(named=True):
        indices = row["idx"]
        idx = indices[0]

        obs_idxs = np.where(M[idx] == 1)[0]

        # list of observed and unobserved idxs
        X_obs = X[np.ix_(indices, obs_idxs)]
        Sigma_obs = Sigma[np.ix_(obs_idxs, obs_idxs)]

        # Additional safeguard: add regularization if the submatrix is still not positive definite
        try:
            # Try to compute Cholesky decomposition to check positive definiteness
            np.linalg.cholesky(Sigma_obs)
        except np.linalg.LinAlgError:
            # If not positive definite, add more regularization
            Sigma_obs += eps * np.eye(Sigma_obs.shape[0])

        log_likelihood += (
            multivariate_normal.logpdf(
                X_obs, mean=np.zeros(obs_idxs.shape[0]), cov=Sigma_obs
            ).sum()
            / n
        )

    # return elbo  # log_likelihood
    return log_likelihood


def EM_update_step_X_only(
    X: np.ndarray[np.float64],
    M: np.ndarray[np.bool],
    Lamda: np.ndarray[np.float64],
    Psi: np.ndarray[np.float64],
    index_groups: pl.DataFrame,
    find_ll: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    # Mean imputation to initialize to 0
    n = X.shape[0]
    d = X.shape[1]
    q = Lamda.shape[1]

    C_xx_diag = np.zeros(d, dtype=np.float64)
    C_xf = np.zeros((d, q), dtype=np.float64)
    C_ff = np.zeros((q, q), dtype=np.float64)

    for row in index_groups.iter_rows(named=True):
        indices = row["idx"]

        # these have the same missingness pattern, so we can select the first one
        idx = indices[0]

        # list of observed and unobserved idxs
        obs_idxs = np.where(M[idx] == 1)[0]
        unobs_idxs = np.where(M[idx] == 0)[0]

        X_obs = X[np.ix_(indices, obs_idxs)]
        Lamda_obs = Lamda[obs_idxs, :]
        Lamda_miss = Lamda[unobs_idxs, :]
        Psi_obs = Psi[obs_idxs]
        Psi_miss = Psi[unobs_idxs]

        n_data_pattern = len(indices)

        ## posterior distribution of F_i given X_i_obs
        # Ensure Psi_obs is positive for numerical stability
        eps = 1e-8
        Psi_obs = np.maximum(Psi_obs, eps)

        psi_inv_diag = np.diag(1 / Psi_obs)
        G_pattern = np.linalg.solve(
            Lamda_obs.T @ psi_inv_diag @ Lamda_obs + np.eye(q), np.eye(q)
        ).astype(np.float64)  # posterior covariance of F_i given X_i_obs
        F_hat = (
            X_obs @ psi_inv_diag @ Lamda_obs @ G_pattern.T
        )  # posterior mean of F_i given X_i_obs

        C_ff_pattern = F_hat.T @ F_hat + n_data_pattern * G_pattern

        ## posterior distribution of X_i_unobs given X_i_obs
        X_hat = np.zeros((n_data_pattern, d))
        X_hat[:, obs_idxs] = X_obs
        X_hat[:, unobs_idxs] = F_hat @ Lamda_miss.T

        H_pattern = np.zeros((d, q), dtype=np.float64)
        H_pattern[unobs_idxs, :] = Lamda_miss @ G_pattern

        C_xf_pattern = X_hat.T @ F_hat + n_data_pattern * H_pattern

        ## diagonal of C_xx
        J_pattern = np.zeros(d)
        J_pattern[unobs_idxs] = (
            np.diag(Lamda_miss @ G_pattern @ Lamda_miss.T) + Psi_miss
        )
        diag_C_xx_pattern = np.sum(X_hat**2, axis=0) + n_data_pattern * J_pattern

        C_xx_diag += diag_C_xx_pattern / n
        C_xf += C_xf_pattern / n
        C_ff += C_ff_pattern / n

    Lamda_hat = C_xf @ np.linalg.solve(C_ff, np.eye(q)).astype(np.float64)

    Psi_hat = (
        C_xx_diag
        - 2 * np.diag(C_xf @ Lamda_hat.T)
        + np.diag(Lamda_hat @ C_ff @ Lamda_hat.T)
    )

    # Ensure Psi_hat remains positive for numerical stability
    eps = 1e-6
    Psi_hat = np.maximum(Psi_hat, eps)

    log_lik_per_sample = None
    if find_ll:
        log_lik_per_sample = calculate_log_likelihood_X_only(
            X=X,
            Lamda_hat=Lamda_hat,
            Psi_hat=Psi_hat,
            index_groups=index_groups,
            M=M,
        )

    return Lamda_hat, Psi_hat, log_lik_per_sample


def EM_algorithm_X_only(
    X: np.ndarray,
    M: np.ndarray,
    Q: int,
    tol: float = 1e-2,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    n, d = X.shape

    L = np.random.randn(d, Q).astype(np.float64)
    Psi = np.ones(d, dtype=np.float64)
    log_likelihoods = []

    X_imputed = X * M  # impute missing values with 0
    find_ll = False

    pl_M = pl.from_numpy(M).with_row_count("idx")

    index_groups = (
        pl_M.group_by(pl.exclude("idx"))
        .agg(pl.col("idx"))
        .select(
            pl.col("idx"),
        )
    )

    for iteration in range(max_iter):
        # E-step and M-step: Estimate latent factors

        find_ll = iteration % 10 == 0

        L, Psi, ll_per_point = EM_update_step_X_only(
            X_imputed, M, L, Psi, index_groups, find_ll=find_ll
        )

        if (Psi == 0).any():
            print(f"Log likelihood per point: {ll_per_point}")
            print(f"Latent factors: {L}")
            print(f"Noise covariance: {Psi}")

        if find_ll:
            log_likelihoods.append(ll_per_point)

            # Check convergence
            if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                # print(f"Converged at iteration {iteration}")
                break

    return L, Psi, log_likelihoods


def get_pretrained_parameters_EM(
    train_data: np.ndarray,
    Q: int,
    tol: float = 1e-2,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Get pretrained parameters for the Factor Model Helper
    Args:

    Returns:
        est_cov_matrix: estimated covariance metrix
    """
    # TODO: handle mean imputation setting for Y

    X_train = train_data
    M_train = (~np.isnan(train_data)).astype(int)  # 1 for observed, 0 for missing
    L_hat, Psi_hat, _ = EM_algorithm_X_only(
        X=X_train,
        M=M_train,
        Q=Q,
        tol=tol,
        max_iter=max_iter,
    )
    return L_hat @ L_hat.T + np.diag(Psi_hat)


def fm_impute_X(
    X: np.ndarray,
    M: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    """X is the data matrix, M is the mask matrix,
    cov_matrix is the covariance matrix (or estimate) of X."""
    n, d = X.shape
    Sigma = cov_matrix
    X_imputed = X.copy()

    # Identify unique missingness patterns
    patterns = {}
    for i in range(n):
        pattern = tuple(M[i, :])
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(i)

    # Process each pattern
    for pattern, indices in patterns.items():
        obs_idxs = np.where(np.array(pattern) == 1)[0]
        unobs_idxs = np.where(np.array(pattern) == 0)[0]

        if len(unobs_idxs) == 0:
            continue
        if len(obs_idxs) == 0:
            raise ValueError("No observed values in row")

        # Extract submatrices
        Sigma_11 = Sigma[np.ix_(obs_idxs, obs_idxs)]
        Sigma_12 = Sigma[np.ix_(obs_idxs, unobs_idxs)]

        # Impute missing values for all rows with this pattern
        X_obs = X_imputed[np.ix_(indices, obs_idxs)]

        # Add regularization for numerical stability
        eps = 1e-8
        try:
            X_unobs = X_obs @ np.linalg.solve(Sigma_11, Sigma_12)
        except np.linalg.LinAlgError:
            # If singular, add regularization to diagonal
            Sigma_11_reg = Sigma_11 + eps * np.eye(Sigma_11.shape[0])
            X_unobs = X_obs @ np.linalg.solve(Sigma_11_reg, Sigma_12)

        X_imputed[np.ix_(indices, unobs_idxs)] = X_unobs

    return X_imputed
