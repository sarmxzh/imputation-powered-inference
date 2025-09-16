# %%
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from numpy.random import PCG64, Generator
from rich.logging import RichHandler
from safetensors.numpy import load_file, save
from sklearn.model_selection import train_test_split

from ipi.aipw_baselines import ClassOne_AIPW_CI
from ipi.baselines import (
    EM_Naive_CI,
    Hot_Deck_Naive_CI,
    Mean_Naive_CI,
    MF_Naive_CI,
    Zero_Naive_CI,
    classical_ci,
)
from ipi.config import DATA_DIR, INTERCEPT_COL
from ipi.cross_ipi_methods import EM_CrossIPI
from ipi.ipi_methods import EM_IPI, MF_IPI, Hot_Deck_IPI, Mean_IPI, Zero_IPI
from ipi.utils.missing_data_utils import (
    get_missing_patterns_and_ids,
)

logger = logging.getLogger(__name__)


# %%
@dataclass
class CovData:
    L_TRUE: np.ndarray
    D_TRUE: np.ndarray
    TRUE_COV_MATRIX: np.ndarray


def make_cov(
    num_features: int,
    num_factors: int,
) -> bytes:
    ## Random number generator - set seed for reproducibility ##
    RG_EXPTS = Generator(PCG64(1234))

    L_TRUE = RG_EXPTS.multivariate_normal(
        np.zeros(num_factors), np.eye(num_factors) * 1 / 4, num_features
    )

    sigma_sq = np.trace(L_TRUE @ L_TRUE.T) * (1 - 0.5) / (0.5) / num_features
    # sigma_sq = 1
    D_TRUE = sigma_sq * np.ones(
        num_features
    )  # np.abs(RG_EXPTS.standard_normal(num_features))
    TRUE_COV_MATRIX = L_TRUE @ L_TRUE.T + np.diag(D_TRUE)

    # Save numpy arrays directly without converting to torch tensors
    return save(
        {
            "L_TRUE": L_TRUE,
            "D_TRUE": D_TRUE,
            "TRUE_COV_MATRIX": TRUE_COV_MATRIX,
        }
    )


def make_and_save_cov(
    num_features: int,
    num_factors: int,
    data_dir: Path | None = None,
) -> None:
    cov_data_to_save = make_cov(num_features, num_factors)

    if data_dir is None:
        data_dir = DATA_DIR / "factor_model"

    cov_matrix_name = f"covq{num_factors}d{num_features}.safetensors"
    with open(data_dir / cov_matrix_name, "wb") as f:
        f.write(cov_data_to_save)
    logger.info(f"Saved covariance matrix to {data_dir / cov_matrix_name}")


def load_cov(cov_matrix_name: str, data_dir: Path | None = None) -> CovData:
    if data_dir is None:
        data_dir = DATA_DIR / "factor_model"
    data = load_file(data_dir / cov_matrix_name)
    return CovData(
        L_TRUE=data["L_TRUE"],
        D_TRUE=data["D_TRUE"],
        TRUE_COV_MATRIX=data["TRUE_COV_MATRIX"],
    )


@dataclass
class FMDataset:
    fully_observed_data: np.ndarray
    partially_observed_data: np.ndarray
    column_names: list[str]
    patterns: np.ndarray
    probabilities: np.ndarray


def generate_dataset_ols(
    num_fully_observed: int,
    num_partially_observed: int,
    num_unique_patterns: int,
    true_cov_matrix: np.ndarray,
    prob_missing: float,
    mask_seed: int,
    column_names: int,
) -> FMDataset:
    if true_cov_matrix is None:
        raise ValueError("true_cov_matrix must be provided")

    num_features = len(column_names)
    # columns = [f"X{i}" for i in range(num_features)]

    # --- Dataset 0: Fully Observed ---
    fully_observed_data = np.random.multivariate_normal(
        np.zeros(num_features), true_cov_matrix, size=num_fully_observed
    )

    # --- Dataset 1: Partially Observed ---
    patterns, probabilities = generate_patterns(
        prob_missing,
        total_num_masks=num_unique_patterns,
        seed=mask_seed,
        num_features=num_features,
    )

    partially_observed_data = np.random.multivariate_normal(
        np.zeros(num_features), true_cov_matrix, size=num_partially_observed
    )

    # Apply masks
    masking = np.array(
        random.choices(patterns, k=num_partially_observed, weights=probabilities)
    )
    partially_observed_data = np.where(masking, partially_observed_data, np.nan)

    return FMDataset(
        fully_observed_data=fully_observed_data,
        partially_observed_data=partially_observed_data,
        column_names=column_names,
        patterns=patterns,
        probabilities=probabilities,
    )


@dataclass
class ShiftedCovMatrices:
    obs_cov_matrix: np.ndarray
    part_cov_matrix: np.ndarray


def generate_shifted_cov_matrices(
    true_cov_matrix: np.ndarray,
    num_partially_observed: int,
    num_fully_observed: int,
    shift_magnitude: float,
) -> ShiftedCovMatrices:
    if not np.all(np.linalg.eigvals(true_cov_matrix) > 0):
        raise ValueError("true_cov_matrix is not positive definite")
    # Work on copies to avoid mutating the input matrix and aliasing between outputs
    obs_cov_matrix = true_cov_matrix.copy()
    obs_cov_matrix[:2, 2] = true_cov_matrix[:2, 2] * (1 + shift_magnitude)
    obs_cov_matrix[2, :2] = true_cov_matrix[2, :2] * (1 + shift_magnitude)

    part_cov_matrix = true_cov_matrix.copy()
    part_cov_matrix[:2, 2] = true_cov_matrix[:2, 2] * (
        1 - num_fully_observed / num_partially_observed * shift_magnitude
    )
    part_cov_matrix[2, :2] = true_cov_matrix[2, :2] * (
        1 - num_fully_observed / num_partially_observed * shift_magnitude
    )
    # check if part_cov_matrix is positive definite
    if not np.all(np.linalg.eigvals(obs_cov_matrix) > 0):
        print(np.linalg.eigvals(obs_cov_matrix))
        raise ValueError("obs_cov_matrix is not positive definite")
    if not np.all(np.linalg.eigvals(part_cov_matrix) > 0):
        raise ValueError("part_cov_matrix is not positive definite")
    return ShiftedCovMatrices(
        obs_cov_matrix=obs_cov_matrix, part_cov_matrix=part_cov_matrix
    )


def generate_dataset_shift(
    num_fully_observed: int,
    num_partially_observed: int,
    num_unique_patterns: int,
    true_cov_matrix: np.ndarray,
    prob_missing: float,  # 0.2
    mask_seed: int,  # 110
    shift_magnitude: float,
    column_names: list[str],
) -> FMDataset:
    """
    Generate a dataset with covariate shift.
    Fully observed data comes from one MVN, partially observed from another MVN.
    Both covariance matrices are saved for consistency across runs.
    Designed so that the overall theta_* is the same as the original true theta

    Args:
        shift_magnitude: Controls how different the shifted distribution is (0=no shift)
    """
    # Load or create the original covariance matrix
    if true_cov_matrix is None:
        raise ValueError("true_cov_matrix must be provided")

    num_features = len(column_names)
    ## generate shifted covariance matrices
    shifted_cov_matrices = generate_shifted_cov_matrices(
        true_cov_matrix=true_cov_matrix,
        num_partially_observed=num_partially_observed,
        num_fully_observed=num_fully_observed,
        shift_magnitude=shift_magnitude,
    )
    obs_cov_matrix = shifted_cov_matrices.obs_cov_matrix
    part_cov_matrix = shifted_cov_matrices.part_cov_matrix

    num_features = len(column_names)
    # --- Dataset 0: Fully Observed (has shifted covariance matrix) ---
    fully_observed_data = np.random.multivariate_normal(
        np.zeros(num_features), obs_cov_matrix, size=num_fully_observed
    )

    # --- Dataset 1: Partially Observed (has covariance matrix compensating for shift in observed) ---
    patterns, probabilities = generate_patterns(
        prob_missing,
        total_num_masks=num_unique_patterns,
        seed=mask_seed,
        num_features=num_features,
    )

    partially_observed_data = np.random.multivariate_normal(
        np.zeros(num_features), part_cov_matrix, size=num_partially_observed
    )

    # Apply masks
    masking = np.array(
        random.choices(patterns, k=num_partially_observed, weights=probabilities)
    )
    partially_observed_data = np.where(masking, partially_observed_data, np.nan)

    return FMDataset(
        fully_observed_data=fully_observed_data,
        partially_observed_data=partially_observed_data,
        column_names=column_names,
        patterns=patterns,
        probabilities=probabilities,
    )


def generate_patterns(
    prob_missing: float,
    total_num_masks: int,
    seed: int,  # 110
    num_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a set of masks with the given probability of missing
    Masks here are binary vectors of length NUM_FEATURES
    where each element is 1 if the feature is observed and 0 if it is missing
    Ensures that at least one of the first three features is missing per pattern
    while maintaining overall missingness rate of approximately prob_missing
    Args:
        prob_missing: probability of missing
        total_num_masks: total number of masks to generate
        seed: seed for the random number generator
        num_features: number of features
    Returns:
        masks: numpy array with boolean masks
        probabilities: numpy array with probabilities of each mask
    """
    rg = np.random.default_rng(seed)
    all_masks = []
    unique_masks = set()

    # Generate masks until we collect enough unique patterns
    while len(unique_masks) < total_num_masks:
        mask = rg.binomial(1, 1 - prob_missing, num_features).astype(bool)

        # Ensure at least one of the first three features is missing
        if mask[0] and mask[1] and mask[2]:  # If all first 3 are observed
            # Randomly choose one of the first 3 to be missing
            missing_idx = rg.choice(3)
            mask[missing_idx] = False

        # Exclude all-True/all-False patterns
        if 0 < np.sum(mask) < num_features:
            mask_tuple = tuple(mask)
            if mask_tuple not in unique_masks:
                all_masks.append(mask)
                unique_masks.add(mask_tuple)

    # Convert to numpy array with boolean dtype
    masks = np.array(all_masks).astype(bool)

    # Find unique patterns and their counts using numpy
    # Convert boolean masks to strings for unique identification
    mask_strings = np.array(["".join(mask.astype(int).astype(str)) for mask in masks])
    unique_strings, counts = np.unique(mask_strings, return_counts=True)

    # Convert unique strings back to boolean arrays
    patterns = np.array(
        [np.array([bool(int(char)) for char in string]) for string in unique_strings]
    )

    # Calculate probabilities
    probabilities = counts / counts.sum()

    return patterns, probabilities


# %%
@dataclass
class FM_OlsResult:
    lb: float
    ub: float
    coverage: bool
    estimator: str
    target_idx: int
    N_to_n0_ratio: float
    num_fully_observed: int
    prob_missing: float
    n_eff: float
    n_neighbors: int | None
    Q_est: int | None
    trial_seed: int
    num_masks: int
    mask_seed: int
    nmcar_shift_magnitude: float


def get_tracking_input_fm_ols(
    ci_method_name: str,  # string
    ci: tuple[float, float],  # confidence interval
    baseline_classical_interval: tuple[float, float],  # classical confidence interval
    true_theta: float,  # true theta value (float)
    target_idx: int,  # index of the coefficient
    ratio: float,  # num_partially_obserd = ratio * num_fully_observed
    num_fully_observed: int,  # num_fully_observed
    prob_missing: float,  # probability of missingness (how much is missing)
    trial_seed: int,  # seed for the trial
    num_masks: int,  # number of masks
    mask_seed: int,  # seed for the mask
    Q_est: int | None = None,  # q used to construct EM,
    n_neighbors: int | None = None,  # Number of neighbors for Hot Deck imputation
    nmcar_shift_magnitude: float | None = None,  # NM-CAR shift magnitude
) -> FM_OlsResult:
    return FM_OlsResult(
        lb=ci[0],
        ub=ci[1],
        coverage=(ci[0] <= true_theta) and (true_theta <= ci[1]),
        target_idx=target_idx,
        estimator=ci_method_name,
        N_to_n0_ratio=ratio,
        num_fully_observed=num_fully_observed,
        prob_missing=prob_missing,
        n_eff=(
            (baseline_classical_interval[1] - baseline_classical_interval[0])
            / (ci[1] - ci[0])
        )
        ** 2
        * num_fully_observed,
        n_neighbors=n_neighbors,
        Q_est=Q_est,
        trial_seed=trial_seed,
        num_masks=num_masks,
        mask_seed=mask_seed,
        nmcar_shift_magnitude=nmcar_shift_magnitude,
    )


@dataclass
class ComputeTrialFMArgs:
    # Data parameters
    num_fully_observed: int  # Number of fully observed data points
    ratio: int  # Ratio for partially observed data
    column_names: list[str]  # All data columns
    true_cov_matrix: np.ndarray  # true covariance matrix
    prob_missing: float  # Probability of missingness
    num_masks: int  # Number of unique masks
    mask_seed: int  # Seed for the mask
    trial_seed: int | None  # Seed for the trial
    nmcar_shift_magnitude: float | None  # NMCAR shift magnitude

    # Experiment parameters
    regr_features: list[str]  # Features for regression
    outcome: str  # Outcome variable name
    target_idx: int  # Index of the coefficient
    true_theta: float  # True parameter value
    alpha: float  # Significance level
    train_percent: float  # Percentage of data to hold out for training
    num_folds: int  # Number of folds for cross-validation
    num_bootstrap_trials: int  # Number of bootstrap trials for cross-validation

    # Model parameters
    est_latent_dim: int  # q used to construct EM
    n_neighbors: int  # Number of neighbors for Hot Deck imputation

    # Method parameters
    methods: set[
        Literal[
            "aipw_baseline",
            "naive_em_ci",
            "naive_inplace_em_ci",
            "naive_hotdeck_ci",
            "naive_inplace_hotdeck_ci",
            "ipi_untuned_em",
            "ipi_untuned_mf",
            "ipi_untuned_mean",
            "ipi_untuned_zero",
            "ipi_untuned_hotdeck",
            "ipi_tuned_em",
            "ipi_tuned_mf",
            "ipi_tuned_mean",
            "ipi_tuned_zero",
            "ipi_tuned_hotdeck",
            "ipi_per_mask_untuned_em",
            "ipi_per_mask_tuned_em",
            "cipi_untuned_em",
            "cipi_tuned_em",
        ]
    ]


def fm_compute_trial_oop(args: ComputeTrialFMArgs) -> list[FM_OlsResult]:
    # Configure basic logging with both handlers
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])

    if args.trial_seed is not None:
        np.random.seed(args.trial_seed)
        random.seed(args.trial_seed)
    else:
        # Get and log the current numpy random seed
        # Unix child processes inherit the parent process's random seed, so we need to set a new seed
        # See https://stackoverflow.com/questions/9209078/using-python-multiprocessing-with-different-random-seed-for-each-process
        pid_timestamp_seed = (os.getpid() * int(time.time_ns())) % 123456789
        np.random.seed(pid_timestamp_seed)
        random.seed(pid_timestamp_seed)

    # Data parameters
    num_fully_observed = args.num_fully_observed  # Number of fully observed data points
    ratio = args.ratio  # Ratio for partially observed data
    prob_missing = args.prob_missing  # Probability of missingness
    num_masks = args.num_masks  # Number of unique masks
    regr_features = args.regr_features  # Features for regression
    column_names = args.column_names  # All data columns
    true_cov_matrix = args.true_cov_matrix  # True covariance matrix
    mask_seed = args.mask_seed  # Seed for the mask

    # Experiment parameters
    est_latent_dim = args.est_latent_dim  # q used to construct EM
    alpha = args.alpha  # Significance level
    train_percent = args.train_percent  # Percentage of data to hold out for training
    num_folds = args.num_folds  # Number of folds for cross-validation
    num_bootstrap_trials = (
        args.num_bootstrap_trials
    )  # Number of bootstrap trials for cross-validation

    # Model parameters
    target_idx = args.target_idx  # Index of the coefficient
    true_theta = args.true_theta  # True parameter value
    outcome = args.outcome  # Outcome variable name
    n_neighbors = args.n_neighbors  # Number of neighbors for Hot Deck imputation

    if num_masks <= 0:
        raise ValueError(f"num_masks must be greater than 0, got {num_masks}")

    if args.nmcar_shift_magnitude is not None:
        logger.info(
            f"Generating dataset with shift magnitude: {args.nmcar_shift_magnitude}"
        )
        fm_dataset = generate_dataset_shift(
            num_fully_observed=num_fully_observed,
            num_partially_observed=int(ratio * num_fully_observed),
            num_unique_patterns=num_masks,
            true_cov_matrix=true_cov_matrix,
            prob_missing=prob_missing,
            mask_seed=mask_seed,
            shift_magnitude=args.nmcar_shift_magnitude,
            column_names=column_names,
        )
    else:
        fm_dataset = generate_dataset_ols(
            num_fully_observed=num_fully_observed,
            num_partially_observed=int(ratio * num_fully_observed),
            num_unique_patterns=num_masks,
            true_cov_matrix=true_cov_matrix,
            prob_missing=prob_missing,
            mask_seed=mask_seed,
            column_names=column_names,
        )

    fullyobs_data = fm_dataset.fully_observed_data
    partiallyobs_data = fm_dataset.partially_observed_data
    column_names = fm_dataset.column_names

    ## add intercept column to both datasets
    fullyobs_data = np.hstack([np.ones((fullyobs_data.shape[0], 1)), fullyobs_data])
    partiallyobs_data = np.hstack(
        [np.ones((partiallyobs_data.shape[0], 1)), partiallyobs_data]
    )
    column_names = [INTERCEPT_COL, *column_names]

    ## get indices of regr_features and outcome
    regr_features_idxs = [column_names.index(feature) for feature in regr_features]
    outcome_idx = column_names.index(outcome)
    intercept_idx = column_names.index(INTERCEPT_COL)

    pattern_to_ids = get_missing_patterns_and_ids(data_array=partiallyobs_data)

    # # note: for debugging, save a visual of missingness patterns
    # plot_missing_patterns(
    #     pattern_to_ids=pattern_to_ids,
    #     column_names=column_names,
    #     png_dir=RES_DIR / "factor_model_ols" / "fm_ols_expt1",
    #     png_filename=f"missingness_patterns_seed{args.trial_seed}.png",
    # )

    # Calculate counts for each pattern
    pattern_counts = {
        pattern: len(indices) for pattern, indices in pattern_to_ids.items()
    }

    ols_results: list[FM_OlsResult] = []

    ## calculate classical CI
    ci_tuple = classical_ci(
        fully_obs_data=fullyobs_data,
        regr_features_idxs=regr_features_idxs,
        outcome_idx=outcome_idx,
        target_idx=target_idx,
        regression_type="ols",
        alpha=alpha,
    )

    ols_results.append(
        get_tracking_input_fm_ols(
            ci_method_name="classical",
            ci=ci_tuple,
            baseline_classical_interval=ci_tuple,
            true_theta=true_theta,
            target_idx=target_idx,
            ratio=ratio,
            num_fully_observed=num_fully_observed,
            prob_missing=prob_missing,
            trial_seed=args.trial_seed,
            Q_est=None,
            n_neighbors=None,
            num_masks=num_masks,
            mask_seed=mask_seed,
            nmcar_shift_magnitude=args.nmcar_shift_magnitude,
        )
    )

    ## calculate AIPW baseline CI when required
    if "aipw_baseline" in args.methods:
        aipw_method = ClassOne_AIPW_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_data,
            partially_obs_data=partiallyobs_data,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            intercept_idx=intercept_idx,
        )
        aipw_ci_tuple = aipw_method.get_semiparam_ci(alpha=alpha)
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="aipw_baseline",
                ci=aipw_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    ### INITIALIZE MODELS ###

    # Initialize IPI and naive models when required
    # train_test_split based on train_percent
    fullyobs_data_train, fullyobs_data_test = train_test_split(
        fullyobs_data, train_size=train_percent, random_state=42
    )
    partiallyobs_data_train, partiallyobs_data_test = train_test_split(
        partiallyobs_data, train_size=train_percent, random_state=42
    )
    train_data = np.vstack([fullyobs_data_train, partiallyobs_data_train])
    pattern_to_ids_for_ipi = get_missing_patterns_and_ids(
        data_array=partiallyobs_data_test
    )
    pattern_counts_for_ipi = {
        pattern: len(indices) for pattern, indices in pattern_to_ids_for_ipi.items()
    }

    ## fit IPI model when required
    if any(m.startswith("ipi_") and (m.endswith("tuned_em")) for m in args.methods):
        # fit IPI model
        ipi_method_em = EM_IPI(
            regression_type="ols",
            fully_obs_data=fullyobs_data_test,
            partially_obs_data=partiallyobs_data_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
        )

        # fit IPI model
        ipi_method_em.fit_and_setup(
            train_data=train_data,
            Q_est=est_latent_dim,
        )

    if any(m.startswith("ipi_") and (m.endswith("tuned_mf")) for m in args.methods):
        ipi_method_mf = MF_IPI(
            regression_type="ols",
            fully_obs_data=fullyobs_data_test,
            partially_obs_data=partiallyobs_data_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
        )
        ## NOTE: currently tailored for the factor model dataset
        ipi_method_mf.fit_and_setup(
            train_data=train_data,
            column_names=column_names,
            num_cat_classes=None,
            do_compile=False,
            categorical_features=None,
        )

    if any(m.startswith("ipi_") and (m.endswith("tuned_mean")) for m in args.methods):
        ipi_method_mean = Mean_IPI(
            regression_type="ols",
            fully_obs_data=fullyobs_data_test,
            partially_obs_data=partiallyobs_data_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
        )
        ipi_method_mean.fit_and_setup(train_data=train_data)

    if any(m.startswith("ipi_") and (m.endswith("tuned_zero")) for m in args.methods):
        ipi_method_zero = Zero_IPI(
            regression_type="ols",
            fully_obs_data=fullyobs_data_test,
            partially_obs_data=partiallyobs_data_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
        )
        ipi_method_zero.fit_and_setup(train_data=train_data)

    if any(
        m.startswith("ipi_") and (m.endswith("tuned_hotdeck")) for m in args.methods
    ):
        ipi_method_hotdeck = Hot_Deck_IPI(
            regression_type="ols",
            fully_obs_data=fullyobs_data_test,
            partially_obs_data=partiallyobs_data_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
        )
        ipi_method_hotdeck.fit_and_setup(train_data=train_data, n_neighbors=n_neighbors)

    if "naive_em_ci" in args.methods:
        naive_em_method = EM_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_data_test,
            partially_obs_data=partiallyobs_data_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=False,
        )
        naive_em_method.fit_and_setup(
            train_data=train_data,
            Q_est=est_latent_dim,
        )

    if "naive_hotdeck_ci" in args.methods:
        naive_hotdeck_method = Hot_Deck_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_data_test,
            partially_obs_data=partiallyobs_data_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=False,
        )
        naive_hotdeck_method.fit_and_setup(
            train_data=train_data, n_neighbors=n_neighbors
        )
    if "naive_mf_ci" in args.methods:
        naive_mf_method = MF_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_data,
            partially_obs_data=partiallyobs_data,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=False,
        )
        naive_mf_method.fit_and_setup(
            train_data=train_data,
            column_names=column_names,
            num_cat_classes=None,
            do_compile=False,
            categorical_features=None,
        )

    if "naive_mean_ci" in args.methods:
        naive_mean_method = Mean_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_data_test,
            partially_obs_data=partiallyobs_data_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=False,
        )
        naive_mean_method.fit_and_setup(train_data=train_data)

    if "naive_zero_ci" in args.methods:
        naive_zero_method = Zero_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_data_test,
            partially_obs_data=partiallyobs_data_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=False,
        )
        naive_zero_method.fit_and_setup(train_data=train_data)

    ## Initialize CIPI model
    cipi_method_em = EM_CrossIPI(
        regression_type="ols",
        fully_obs_data=fullyobs_data,
        partially_obs_data=partiallyobs_data,
        regr_features_idxs=regr_features_idxs,
        outcome_idx=outcome_idx,
        target_idx=target_idx,
    )

    ##### CALCULATE CIs #####

    # untuned IPI methods
    untuned_lambda_weights_for_ipi = np.array(
        [pattern_counts_for_ipi[pattern] for pattern in pattern_to_ids_for_ipi]
    ) / sum(pattern_counts_for_ipi.values())

    if "ipi_untuned_em" in args.methods:
        # untuned IPI for all X
        ipi_untuned_em_ci_tuple = ipi_method_em.get_ipi_ci(
            lambda_weights=untuned_lambda_weights_for_ipi,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="ipi_untuned_em",
                ci=ipi_untuned_em_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=est_latent_dim,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    if "ipi_untuned_mf" in args.methods:
        ipi_untuned_mf_ci_tuple = ipi_method_mf.get_ipi_ci(
            lambda_weights=untuned_lambda_weights_for_ipi,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="ipi_untuned_mf",
                ci=ipi_untuned_mf_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    if "ipi_untuned_mean" in args.methods:
        ipi_untuned_mean_ci_tuple = ipi_method_mean.get_ipi_ci(
            lambda_weights=untuned_lambda_weights_for_ipi,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="ipi_untuned_mean",
                ci=ipi_untuned_mean_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )
    if "ipi_untuned_zero" in args.methods:
        ipi_untuned_zero_ci_tuple = ipi_method_zero.get_ipi_ci(
            lambda_weights=untuned_lambda_weights_for_ipi,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="ipi_untuned_zero",
                ci=ipi_untuned_zero_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )
    if "ipi_untuned_hotdeck" in args.methods:
        ipi_untuned_hotdeck_ci_tuple = ipi_method_hotdeck.get_ipi_ci(
            lambda_weights=untuned_lambda_weights_for_ipi,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="ipi_untuned_hotdeck",
                ci=ipi_untuned_hotdeck_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=n_neighbors,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    # tuned IPI methods
    if "ipi_tuned_em" in args.methods:
        # tuned IPI for all X
        ipi_tuned_em_ci_tuple = ipi_method_em.get_ipi_ci(
            lambda_weights=None,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="ipi_tuned_em",
                ci=ipi_tuned_em_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=est_latent_dim,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    if "ipi_tuned_mf" in args.methods:
        ipi_tuned_mf_ci_tuple = ipi_method_mf.get_ipi_ci(
            lambda_weights=None,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="ipi_tuned_mf",
                ci=ipi_tuned_mf_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    if "ipi_tuned_mean" in args.methods:
        ipi_tuned_mean_ci_tuple = ipi_method_mean.get_ipi_ci(
            lambda_weights=None,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="ipi_tuned_mean",
                ci=ipi_tuned_mean_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    if "ipi_tuned_zero" in args.methods:
        ipi_tuned_zero_ci_tuple = ipi_method_zero.get_ipi_ci(
            lambda_weights=None,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="ipi_tuned_zero",
                ci=ipi_tuned_zero_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    if "ipi_tuned_hotdeck" in args.methods:
        ipi_tuned_hotdeck_ci_tuple = ipi_method_hotdeck.get_ipi_ci(
            lambda_weights=None,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="ipi_tuned_hotdeck",
                ci=ipi_tuned_hotdeck_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=n_neighbors,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    # naive methods
    if "naive_em_ci" in args.methods:
        # trains on held out train data, imputes on test data
        naive_em_ci_tuple = naive_em_method.get_naive_ci(alpha=alpha)
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="naive_em_ci",
                ci=naive_em_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=est_latent_dim,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    if "naive_hotdeck_ci" in args.methods:
        # trains on held out train data, imputes on test data
        naive_hotdeck_ci_tuple = naive_hotdeck_method.get_naive_ci(alpha=alpha)
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="naive_hotdeck_ci",
                ci=naive_hotdeck_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=n_neighbors,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )
    if "naive_mf_ci" in args.methods:
        naive_mf_ci_tuple = naive_mf_method.get_naive_ci(alpha=alpha)
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="naive_mf_ci",
                ci=naive_mf_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    if "naive_mean_ci" in args.methods:
        naive_mean_ci_tuple = naive_mean_method.get_naive_ci(alpha=alpha)
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="naive_mean_ci",
                ci=naive_mean_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    if "naive_zero_ci" in args.methods:
        naive_zero_ci_tuple = naive_zero_method.get_naive_ci(alpha=alpha)
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="naive_zero_ci",
                ci=naive_zero_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    # naive inplace methods
    if "naive_inplace_em_ci" in args.methods:
        ## uses full data, fit in place
        naive_inplace_em_method = EM_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_data,
            partially_obs_data=partiallyobs_data,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=True,
        )
        naive_inplace_em_method.fit_and_setup(
            train_data=None,
            Q_est=est_latent_dim,
        )
        naive_inplace_em_ci_tuple = naive_inplace_em_method.get_naive_ci(alpha=alpha)
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="naive_inplace_em_ci",
                ci=naive_inplace_em_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=est_latent_dim,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    if "naive_inplace_hotdeck_ci" in args.methods:
        naive_inplace_hotdeck_method = Hot_Deck_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_data,
            partially_obs_data=partiallyobs_data,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=True,
        )
        naive_inplace_hotdeck_method.fit_and_setup(
            train_data=None, n_neighbors=n_neighbors
        )
        naive_inplace_hotdeck_ci_tuple = naive_inplace_hotdeck_method.get_naive_ci(
            alpha=alpha
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="naive_inplace_hotdeck_ci",
                ci=naive_inplace_hotdeck_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=n_neighbors,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )
    if "naive_inplace_mf_ci" in args.methods:
        naive_inplace_mf_method = MF_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_data,
            partially_obs_data=partiallyobs_data,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=True,
        )
        naive_inplace_mf_method.fit_and_setup(train_data=None)
        naive_inplace_mf_ci_tuple = naive_inplace_mf_method.get_naive_ci(alpha=alpha)
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="naive_inplace_mf_ci",
                ci=naive_inplace_mf_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )
    if "naive_inplace_mean_ci" in args.methods:
        naive_inplace_mean_method = Mean_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_data,
            partially_obs_data=partiallyobs_data,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=True,
        )
        naive_inplace_mean_method.fit_and_setup(train_data=None)
        naive_inplace_mean_ci_tuple = naive_inplace_mean_method.get_naive_ci(
            alpha=alpha
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="naive_inplace_mean_ci",
                ci=naive_inplace_mean_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )
    if "naive_inplace_zero_ci" in args.methods:
        naive_inplace_zero_method = Zero_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_data,
            partially_obs_data=partiallyobs_data,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=True,
        )
        naive_inplace_zero_method.fit_and_setup(train_data=None)
        naive_inplace_zero_ci_tuple = naive_inplace_zero_method.get_naive_ci(
            alpha=alpha
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="naive_inplace_zero_ci",
                ci=naive_inplace_zero_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=None,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    # CIPI methods
    untuned_lambda_weights_for_cipi = np.array(
        [pattern_counts[pattern] for pattern in pattern_to_ids]
    ) / sum(pattern_counts.values())

    if "cipi_untuned_em" in args.methods:
        cipi_untuned_ci_tuple = cipi_method_em.get_ci_cipi(
            num_folds=num_folds,
            num_bootstrap_trials=num_bootstrap_trials,
            alpha=alpha,
            lambda_weights=untuned_lambda_weights_for_cipi,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="cipi_untuned_em",
                ci=cipi_untuned_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=est_latent_dim,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    if "cipi_tuned_em" in args.methods:
        cipi_tuned_ci_tuple = cipi_method_em.get_ci_cipi(
            num_folds=num_folds,
            num_bootstrap_trials=num_bootstrap_trials,
            alpha=alpha,
            lambda_weights=None,
        )
        ols_results.append(
            get_tracking_input_fm_ols(
                ci_method_name="cipi_tuned_em",
                ci=cipi_tuned_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                prob_missing=prob_missing,
                trial_seed=args.trial_seed,
                Q_est=est_latent_dim,
                n_neighbors=None,
                num_masks=num_masks,
                mask_seed=mask_seed,
                nmcar_shift_magnitude=args.nmcar_shift_magnitude,
            )
        )

    # IPI per mask methods
    use_methods_with_one_mask = (
        "ipi_per_mask_untuned_em" in args.methods
        or "ipi_per_mask_tuned_em" in args.methods
    )
    if use_methods_with_one_mask:
        for k, pattern in enumerate(pattern_to_ids.keys()):
            ## processing for ipi for pattern_k
            ipi_method_pattern_k = ipi_method_em.copy()
            ipi_method_pattern_k.setup_imputed_data(
                fully_obs_data_dict={
                    ipi_method_pattern_k.fully_obs_tuple: ipi_method_em.fully_obs_data,
                    pattern: ipi_method_em.fully_obs_data_dict[pattern],
                },
                imputed_partially_obs_data=ipi_method_em.imputed_partially_obs_data[
                    list(pattern_to_ids_for_ipi[pattern]), :
                ],
                pattern_to_ids={
                    pattern: list(range(len(pattern_to_ids_for_ipi[pattern])))
                },
            )

            if "ipi_per_mask_untuned_em" in args.methods:
                ipi_k_untuned_ci_tuple = ipi_method_pattern_k.get_ipi_ci(
                    lambda_weights=np.ones(1),
                    alpha=alpha,
                )
                ols_results.append(
                    get_tracking_input_fm_ols(
                        ci_method_name=f"ipi_mask_{k}_untuned",
                        ci=ipi_k_untuned_ci_tuple,
                        baseline_classical_interval=ci_tuple,
                        true_theta=true_theta,
                        target_idx=target_idx,
                        ratio=ratio,
                        num_fully_observed=num_fully_observed,
                        prob_missing=prob_missing,
                        trial_seed=args.trial_seed,
                        Q_est=est_latent_dim,
                        n_neighbors=None,
                        num_masks=num_masks,
                        mask_seed=mask_seed,
                        nmcar_shift_magnitude=args.nmcar_shift_magnitude,
                    )
                )
            if "ipi_per_mask_tuned_em" in args.methods:
                ipi_k_tuned_ci_tuple = ipi_method_pattern_k.get_ipi_ci(
                    lambda_weights=None,
                    alpha=alpha,
                )
                ols_results.append(
                    get_tracking_input_fm_ols(
                        ci_method_name=f"ipi_mask_{k}_tuned",
                        ci=ipi_k_tuned_ci_tuple,
                        baseline_classical_interval=ci_tuple,
                        true_theta=true_theta,
                        target_idx=target_idx,
                        ratio=ratio,
                        num_fully_observed=num_fully_observed,
                        prob_missing=prob_missing,
                        trial_seed=args.trial_seed,
                        Q_est=est_latent_dim,
                        n_neighbors=None,
                        num_masks=num_masks,
                        mask_seed=mask_seed,
                        nmcar_shift_magnitude=args.nmcar_shift_magnitude,
                    )
                )

    return ols_results


# %%
def run_fm_expt(
    init_seed: int,
    num_trials: int,
    num_features: int,
    num_masks: int,
    num_fully_observed: int,
    prob_missing: float,
    ratio: int,
    num_factors: int,
    header_name: str,
    methods: list[str],
    data_dir: Path | None,  # data_dir default None -> DATA_DIR / "factor_model"
    nmcar_shift_magnitude: float | None,  # None default
    mask_seed: int,
) -> str:
    from tqdm import tqdm

    from ipi.config import INTERCEPT_COL, RES_DIR
    from ipi.utils.factor_model_utils import fm_compute_trial_oop

    ## set of parameters that should be the same for all trials
    ALPHA = 0.1
    OUTCOME = "X2"
    COLUMNS = [f"X{i}" for i in range(num_features)]
    REGR_FEATURES = [INTERCEPT_COL, "X0", "X1"]
    TARGET_IDX = 1
    NUM_FOLDS = 10
    TRAIN_PERCENT = 1 / NUM_FOLDS
    EST_LATENT_DIM = 4
    NUM_BOOTSTRAP_TRIALS = 50
    N_NEIGHBORS = 1
    MASK_SEED = mask_seed if mask_seed is not None else 110

    if data_dir is None:
        data_dir = DATA_DIR / "factor_model"
    cov_matrix_name = f"covq{num_factors}d{num_features}.safetensors"
    try:
        cov_data = load_cov(cov_matrix_name=cov_matrix_name, data_dir=data_dir)
        TRUE_COV_MATRIX = cov_data.TRUE_COV_MATRIX
    except FileNotFoundError:
        make_and_save_cov(
            num_features=num_features,
            num_factors=num_factors,
            data_dir=data_dir,
        )
        cov_data = load_cov(cov_matrix_name=cov_matrix_name, data_dir=data_dir)
        TRUE_COV_MATRIX = cov_data.TRUE_COV_MATRIX

    TRUE_THETA = (np.linalg.inv(TRUE_COV_MATRIX[:2, :2]) @ TRUE_COV_MATRIX[:2, 2])[0]

    results_list = []
    for j in tqdm(range(num_trials), desc="Running trials"):
        results = fm_compute_trial_oop(
            ComputeTrialFMArgs(
                num_fully_observed=num_fully_observed,
                ratio=ratio,
                column_names=COLUMNS,
                true_cov_matrix=TRUE_COV_MATRIX,
                prob_missing=prob_missing,
                num_masks=num_masks,
                mask_seed=MASK_SEED,
                regr_features=REGR_FEATURES,
                outcome=OUTCOME,
                target_idx=TARGET_IDX,
                true_theta=TRUE_THETA,
                alpha=ALPHA,
                train_percent=TRAIN_PERCENT,
                num_folds=NUM_FOLDS,
                num_bootstrap_trials=NUM_BOOTSTRAP_TRIALS,
                est_latent_dim=EST_LATENT_DIM,
                n_neighbors=N_NEIGHBORS,
                methods=methods,
                trial_seed=init_seed + j,
                nmcar_shift_magnitude=nmcar_shift_magnitude,
            )
        )
        results_list.extend(results)
    results_df = pl.DataFrame(results_list)
    if nmcar_shift_magnitude is not None:
        experiment_name = f"factor_model_ols/{header_name}/nsm{nmcar_shift_magnitude}_n0_{num_fully_observed}_ratio{ratio}_d{num_features}_q{num_factors}_p{prob_missing}_{num_masks}masks_Qest{EST_LATENT_DIM}_numtrials{num_trials}_initseed{init_seed}"
    else:
        experiment_name = f"factor_model_ols/{header_name}/n0_{num_fully_observed}_ratio{ratio}_d{num_features}_q{num_factors}_p{prob_missing}_{num_masks}masks_Qest{EST_LATENT_DIM}_numtrials{num_trials}_initseed{init_seed}"

    os.makedirs(RES_DIR / experiment_name, exist_ok=True)
    results_df.write_csv(RES_DIR / experiment_name / "results.csv")
    return str(RES_DIR / experiment_name / "results.csv")
