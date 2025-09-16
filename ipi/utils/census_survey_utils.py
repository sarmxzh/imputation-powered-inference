import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import folktables
import numpy as np
import pandas as pd
import polars as pl
from rich.logging import RichHandler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from ipi.baselines import Hot_Deck_Naive_CI, MF_Naive_CI, classical_ci
from ipi.config import DATA_DIR, INTERCEPT_COL, RES_DIR
from ipi.ipi_methods import MF_IPI, Mean_IPI, Zero_IPI
from ipi.utils.factor_model_utils import get_missing_patterns_and_ids
from ipi.utils.stats_utils import get_point_estimate

logger = logging.getLogger(__name__)

FEATURES = [
    "AGEP",
    "SCHL",
    "MAR",
    "DIS",
    "ESP",
    "CIT",
    "MIG",
    "MIL",
    "ANC1P",
    "NATIVITY",
    "DEAR",
    "DEYE",
    "DREM",
    "SEX",
    "RAC1P",
    "COW",
    "PINCP",
]
FEATURE_TYPES = np.array(
    [
        "q",
        "q",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        "q",
    ]
)
ALLOCATION_FLAGS = [
    "FAGEP",
    "FSCHLP",
    "FMARP",
    "FDISP",
    "FCITP",
    "FMIGSP",
    "FMILSP",
    "FANCP",
    "FDEARP",
    "FDEYEP",
    "FDREMP",
    "FSEXP",
    "FRACP",
    "FCOWP",
    "FPINCP",
]


def get_census_survey_data(
    year: int,
    regr_features: list[str],
    regr_ft_alloc: list[str],
    outcome: str,
    outcome_alloc: str,
    topk: int,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Check if cached data exists
    data_dir = DATA_DIR / "census_survey" if data_dir is None else data_dir
    os.makedirs(data_dir, exist_ok=True)

    try:
        df_filtered = pd.read_parquet(data_dir / f"acs_{year}_CA_filtered.parquet")
        patterns_df = pd.read_parquet(
            data_dir / f"missingness_patterns_{year}_CA.parquet"
        )
        probs_df = pd.read_parquet(data_dir / f"missingness_probs_{year}_CA.parquet")
        # Convert cached DataFrames back to NumPy arrays for return type consistency
        missingness_patterns = patterns_df.to_numpy()
        # Ensure 1-D shape for probabilities
        missingness_probs = probs_df.squeeze(axis=1).to_numpy()
    except FileNotFoundError:
        logger.info(
            f"No processed census info found for year {year} in CA, downloading..."
        )

        data_file = data_dir / f"acs_{year}_CA.parquet"

        if os.path.exists(data_file):
            logger.info(f"Loading cached census data from {data_file}")
            acs_data = pd.read_parquet(data_file)
        else:
            # get data
            logger.info(f"Downloading census data for year {year}")
            data_source = folktables.ACSDataSource(
                survey_year=year, horizon="1-Year", survey="person", root_dir=data_dir
            )
            acs_data: pd.DataFrame = data_source.get_data(states=["CA"], download=True)
            # Cache the downloaded data
            logger.info(f"Caching census data to {data_file}")
            acs_data.to_parquet(data_file)

        # define variables/columsn
        ols_vars = [*regr_features, outcome]

        columns = []
        columns.extend(FEATURES)
        columns.extend(ALLOCATION_FLAGS)  # allocation flags

        df = acs_data[columns]
        logger.info("Loaded %d rows of census survey data, adding intercept", len(df))
        df.loc[:, INTERCEPT_COL] = 1
        features = [*FEATURES, INTERCEPT_COL]

        # drop all rows where any of the ols features are N/A (not applicable)
        df_filtered = df[df[ols_vars].notna().all(axis=1)]

        missingness_patterns_dict = get_top_k_masks(
            df=df,
            topk=topk,
            ols_vars=[*regr_ft_alloc, outcome_alloc],
        )
        missingness_patterns = missingness_patterns_dict["topk_patterns"]
        missingness_probs = missingness_patterns_dict["topk_probabilities"]

        # drop all rows where any of the allocation flags are 1 (i.e. imputed values instead of real values)
        df_filtered = df_filtered[(df_filtered[ALLOCATION_FLAGS] == 0).all(axis=1)]

        ## Drop the allocation flags and fill any remaining missing values with -1
        df_filtered = df_filtered[features].reset_index(drop=True)
        df_filtered = df_filtered.fillna(-1)

        assert (
            not df_filtered[ols_vars].isna().any().any()
        ), "There are still missing values in regression features in the dataframe"

        for idx, col in enumerate(features):
            if col != INTERCEPT_COL and FEATURE_TYPES[idx] == "c":
                le = LabelEncoder()
                df_filtered[col] = le.fit_transform(df_filtered[col])

        df_filtered.to_parquet(data_dir / f"acs_{year}_CA_filtered.parquet")
        # Wrap NumPy arrays in DataFrames before saving to Parquet
        pd.DataFrame(missingness_patterns).to_parquet(
            data_dir / f"missingness_patterns_{year}_CA.parquet"
        )
        pd.DataFrame({"probability": missingness_probs}).to_parquet(
            data_dir / f"missingness_probs_{year}_CA.parquet"
        )
    return df_filtered, missingness_patterns, missingness_probs


def get_top_k_masks(
    df: pd.DataFrame,
    topk: int,
    ols_vars: list[str],
) -> dict[np.ndarray, np.ndarray]:
    """
    Get the top k masks for the census survey data.
    Args:
        df: the census survey data
        topk: the number of top k masks to get
        ols_vars: the variables to use for the ols regression, includes outcome
    Returns:
        dictionary with:
            topk_patterns: the top k patterns
            topk_probabilities_scaled: the probabilities of the top k masks
            scaled by the sum of the probabilities
    """
    ## hard coded in right now -- TODO: fix
    df["FESPP"] = np.zeros(len(df)).astype(int)
    df["FNATP"] = np.zeros(len(df)).astype(int)

    added_allocation_flags = [
        "FAGEP",
        "FSCHLP",
        "FMARP",
        "FDISP",
        "FESPP",
        "FCITP",
        "FMIGSP",
        "FMILSP",
        "FANCP",
        "FNATP",
        "FDEARP",
        "FDEYEP",
        "FDREMP",
        "FSEXP",
        "FRACP",
        "FCOWP",
        "FPINCP",
    ]

    # get missingness patterns
    # note that True here means the value is imputed!
    observed_masks = df[added_allocation_flags].astype(bool)  # Includes outcome column
    unique_col_patterns = observed_masks.drop_duplicates()

    # group by unique patterns + count number of samples with each missingness pattern
    observed_masks_df = (
        observed_masks.groupby(list(unique_col_patterns.columns))
        .size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )
    all_mask_patterns = observed_masks_df.drop(columns="count").astype(bool)

    # filter so that missingness patterns that have at least one of the ols variables missing
    # does not include flag for the intercept
    observed_masks_df_filtered = observed_masks_df[
        (all_mask_patterns[ols_vars]).any(axis=1)
    ].reset_index(drop=True)

    # drop all rows where every feature is imputed
    observed_masks_df_filtered = observed_masks_df_filtered[
        ~(observed_masks_df_filtered[added_allocation_flags] == 1).all(axis=1)
    ].reset_index(drop=True)

    # get topk missingness patterns and probabilities
    topk_missingness_patterns = observed_masks_df_filtered.head(topk)
    topk_counts = topk_missingness_patterns["count"].values
    topk_probabilities = topk_counts / topk_counts.sum()
    topk_patterns = topk_missingness_patterns.drop(columns="count").values

    return {"topk_patterns": topk_patterns, "topk_probabilities": topk_probabilities}


def generate_mask_census_df(
    df: pd.DataFrame,
    labeled_size: float,
    missingness_patterns: np.ndarray,
    missingness_probs: np.ndarray,
    ratio: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_labeled, df_partial = train_test_split(df, train_size=labeled_size)
    sampled_idxs = np.random.choice(
        len(missingness_patterns), size=len(df_partial), p=missingness_probs
    )
    sampled_masks = missingness_patterns[sampled_idxs]
    intercept_masks = np.zeros((len(df_partial), 1)).astype(bool)
    sampled_masks = np.hstack((sampled_masks, intercept_masks))
    df_partial = df_partial.mask(sampled_masks)

    sampled_partial_idxs = np.random.choice(
        len(df_partial), size=int(labeled_size * ratio), replace=False
    )
    df_partial = df_partial.iloc[sampled_partial_idxs]
    return df_labeled.reset_index(drop=True), df_partial.reset_index(drop=True)


@dataclass
class Census_OlsResult:
    lb: float
    ub: float
    coverage: bool
    estimator: str
    target_idx: int
    N_to_n0_ratio: float
    num_fully_observed: int
    n_eff: float
    trial_seed: int
    num_masks: int
    n_neighbors: int | None


def get_tracking_input_census_ols(
    ci_method_name: str,  # string
    ci: tuple[float, float],  # confidence interval
    baseline_classical_interval: tuple[float, float],  # classical confidence interval
    true_theta: float,  # true theta value (float)
    target_idx: int,  # index of the coefficient
    ratio: float,  # num_partially_obserd = ratio * num_fully_observed
    num_fully_observed: int,  # num_fully_observed
    trial_seed: int,  # seed for the trial
    num_masks: int,  # number of masks
    n_neighbors: int | None,  # number of neighbors for Hot Deck imputation
) -> Census_OlsResult:
    return Census_OlsResult(
        lb=ci[0],
        ub=ci[1],
        coverage=(ci[0] <= true_theta) and (true_theta <= ci[1]),
        target_idx=target_idx,
        estimator=ci_method_name,
        N_to_n0_ratio=ratio,
        num_fully_observed=num_fully_observed,
        n_eff=(
            (baseline_classical_interval[1] - baseline_classical_interval[0])
            / (ci[1] - ci[0])
        )
        ** 2
        * num_fully_observed,
        trial_seed=trial_seed,
        num_masks=num_masks,
        n_neighbors=n_neighbors,
    )


@dataclass
class ComputeTrialCensusArgs:
    # Data parameters
    num_fully_observed: int  # Number of fully observed data points
    ratio: int  # Ratio for partially observed data
    num_masks: int  # Number of unique masks
    trial_seed: int | None  # Seed for the trial

    # Experiment parameters
    regr_features: list[str]  # Features for regression
    regr_features_alloc: list[str]  # Features for regression allocation
    outcome: str  # Outcome variable name
    outcome_alloc: str  # Outcome variable name allocation
    target_idx: int  # Index of the coefficient
    true_theta: float  # True parameter value
    alpha: float  # Significance level
    train_percent: float  # Percentage of data to hold out for training
    num_folds: int  # Number of folds for cross-validation
    num_bootstrap_trials: int  # Number of bootstrap trials for cross-validation

    # Model parameters
    n_neighbors: int  # Number of neighbors for Hot Deck imputation

    # Method parameters
    methods: set[
        Literal[
            "classical",
            "ipi_untuned_mf",
            "ipi_tuned_mf",
            "ipi_untuned_mean",
            "ipi_tuned_mean",
            "ipi_untuned_zero",
            "ipi_tuned_zero",
            "ipi_per_mask_untuned_mf",
            "ipi_per_mask_tuned_mf",
            "cipi_untuned_mf",
            "cipi_tuned_mf",
        ]
    ]


def census_compute_trial_oop(args: ComputeTrialCensusArgs) -> list[Census_OlsResult]:
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
    num_masks = args.num_masks  # Number of unique masks

    # Experiment parameters
    alpha = args.alpha  # Significance level
    train_percent = args.train_percent  # Percentage of data to hold out for training

    # Model parameters
    target_idx = args.target_idx  # Index of the coefficient
    true_theta = args.true_theta  # True parameter value
    regr_features = args.regr_features  # Features for regression
    regr_features_alloc = args.regr_features_alloc  # Features for regression allocation
    outcome = args.outcome  # Outcome variable name
    outcome_alloc = args.outcome_alloc  # Outcome variable name allocation

    if num_masks <= 0:
        raise ValueError(f"num_masks must be greater than 0, got {num_masks}")

    df_census_full, missingness_patterns, missingness_probs = get_census_survey_data(
        year=2019,
        regr_features=regr_features,
        regr_ft_alloc=regr_features_alloc,
        outcome=outcome,
        outcome_alloc=outcome_alloc,
        topk=num_masks,
    )

    df_fullyobs, df_partiallyobs = generate_mask_census_df(
        df=df_census_full,
        labeled_size=num_fully_observed,
        missingness_patterns=missingness_patterns,
        missingness_probs=missingness_probs,
        ratio=ratio,
    )

    cols_wout_intercept = [f for f in df_census_full.columns if f != INTERCEPT_COL]
    feature_idxs = {f: FEATURES.index(f) for f in cols_wout_intercept}
    categorical_features = [
        f for f in cols_wout_intercept if FEATURE_TYPES[feature_idxs[f]] == "c"
    ]
    num_cat_classes = {}
    for f in categorical_features:
        num_cat_classes[f] = len(np.unique(df_census_full[f]))
    ## cast to numpy arrays
    ## get indices of regr_features and outcome
    fullyobs_data = df_fullyobs.to_numpy()
    partiallyobs_data = df_partiallyobs.to_numpy()
    column_names = df_census_full.columns.tolist()
    regr_features_idxs = [column_names.index(feature) for feature in regr_features]
    outcome_idx = column_names.index(outcome)

    pattern_to_ids = get_missing_patterns_and_ids(data_array=partiallyobs_data)

    # # note: for debugging, save a visual of missingness patterns
    # plot_missing_patterns(
    #     pattern_to_ids=pattern_to_ids,
    #     column_names=column_names,
    #     png_dir=RES_DIR / "factor_model_ols" / "fm_ols_expt1",
    #     png_filename=f"missingness_patterns_seed{args.trial_seed}.png",
    # )
    ols_results: list[Census_OlsResult] = []

    ## calculate classical CI
    ci_tuple = classical_ci(
        fully_obs_data=fullyobs_data,
        regr_features_idxs=regr_features_idxs,
        outcome_idx=outcome_idx,
        target_idx=target_idx,
        regression_type="ols",
        alpha=alpha,
    )
    # logger.info(f"Classical tuple: {ci_tuple}")

    ols_results.append(
        get_tracking_input_census_ols(
            ci_method_name="classical",
            ci=ci_tuple,
            baseline_classical_interval=ci_tuple,
            true_theta=true_theta,
            target_idx=target_idx,
            ratio=ratio,
            num_fully_observed=num_fully_observed,
            trial_seed=args.trial_seed,
            num_masks=num_masks,
            n_neighbors=None,
        )
    )

    ### INITIALIZE MODELS ###
    # Initialize IPI and naive models when required
    # train_test_split based on train_percent
    fullyobs_train, fullyobs_test = train_test_split(
        fullyobs_data, train_size=train_percent, random_state=42
    )
    partiallyobs_train, partiallyobs_test = train_test_split(
        partiallyobs_data, train_size=train_percent, random_state=42
    )
    train_data = np.vstack([fullyobs_train, partiallyobs_train])
    pattern_to_ids_for_ipi = get_missing_patterns_and_ids(data_array=partiallyobs_test)
    pattern_counts_for_ipi = {
        pattern: len(indices) for pattern, indices in pattern_to_ids_for_ipi.items()
    }

    ## fit IPI model when required
    if any(m.startswith("ipi_") and (m.endswith("tuned_mf")) for m in args.methods):
        ipi_method_mf = MF_IPI(
            regression_type="ols",
            fully_obs_data=fullyobs_test,
            partially_obs_data=partiallyobs_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
        )
        ipi_method_mf.fit_and_setup(
            train_data=train_data,
            column_names=column_names,
            num_cat_classes=num_cat_classes,
            do_compile=False,
            categorical_features=categorical_features,
        )

    if any(m.startswith("ipi_") and (m.endswith("tuned_mean")) for m in args.methods):
        ipi_method_mean = Mean_IPI(
            regression_type="ols",
            fully_obs_data=fullyobs_test,
            partially_obs_data=partiallyobs_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
        )
        ipi_method_mean.fit_and_setup(train_data=train_data)

    if any(m.startswith("ipi_") and (m.endswith("tuned_zero")) for m in args.methods):
        ipi_method_zero = Zero_IPI(
            regression_type="ols",
            fully_obs_data=fullyobs_test,
            partially_obs_data=partiallyobs_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
        )
        ipi_method_zero.fit_and_setup(train_data=train_data)

    if "naive_mf_ci" in args.methods:
        naive_mf_method = MF_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_test,
            partially_obs_data=partiallyobs_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=False,
        )
        naive_mf_method.fit_and_setup(
            train_data=train_data,
            column_names=column_names,
            num_cat_classes=num_cat_classes,
            do_compile=False,
            categorical_features=categorical_features,
        )

    if "naive_hotdeck_ci" in args.methods:
        naive_hotdeck_method = Hot_Deck_Naive_CI(
            regression_type="ols",
            fully_obs_data=fullyobs_test,
            partially_obs_data=partiallyobs_test,
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
            pretrained_model=None,
            inplace=False,
        )
        naive_hotdeck_method.fit_and_setup(
            train_data=train_data, n_neighbors=args.n_neighbors
        )

    ##### CALCULATE CIs #####
    # untuned IPI methods
    untuned_lambda_weights_for_ipi = np.array(
        [pattern_counts_for_ipi[pattern] for pattern in pattern_to_ids_for_ipi]
    ) / sum(pattern_counts_for_ipi.values())

    if "ipi_untuned_mf" in args.methods:
        ipi_untuned_mf_ci_tuple = ipi_method_mf.get_ipi_ci(
            lambda_weights=untuned_lambda_weights_for_ipi,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_census_ols(
                ci_method_name="ipi_untuned_mf",
                ci=ipi_untuned_mf_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                trial_seed=args.trial_seed,
                num_masks=num_masks,
                n_neighbors=None,
            )
        )

    if "ipi_untuned_mean" in args.methods:
        ipi_untuned_mean_ci_tuple = ipi_method_mean.get_ipi_ci(
            lambda_weights=untuned_lambda_weights_for_ipi,
            alpha=alpha,
        )
        # logger.info(f"IPI untuned mean tuple: {ipi_untuned_mean_ci_tuple}")
        ols_results.append(
            get_tracking_input_census_ols(
                ci_method_name="ipi_untuned_mean",
                ci=ipi_untuned_mean_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                trial_seed=args.trial_seed,
                num_masks=num_masks,
                n_neighbors=None,
            )
        )
    if "ipi_untuned_zero" in args.methods:
        ipi_untuned_zero_ci_tuple = ipi_method_zero.get_ipi_ci(
            lambda_weights=untuned_lambda_weights_for_ipi,
            alpha=alpha,
        )
        # logger.info(f"IPI untuned zero tuple: {ipi_untuned_zero_ci_tuple}")
        ols_results.append(
            get_tracking_input_census_ols(
                ci_method_name="ipi_untuned_zero",
                ci=ipi_untuned_zero_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                trial_seed=args.trial_seed,
                num_masks=num_masks,
                n_neighbors=None,
            )
        )

    # tuned IPI methods
    if "ipi_tuned_mf" in args.methods:
        ipi_tuned_mf_ci_tuple = ipi_method_mf.get_ipi_ci(
            lambda_weights=None,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_census_ols(
                ci_method_name="ipi_tuned_mf",
                ci=ipi_tuned_mf_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                trial_seed=args.trial_seed,
                num_masks=num_masks,
                n_neighbors=None,
            )
        )

    if "ipi_tuned_mean" in args.methods:
        ipi_tuned_mean_ci_tuple = ipi_method_mean.get_ipi_ci(
            lambda_weights=None,
            alpha=alpha,
        )
        # logger.info(f"IPI tuned mean tuple: {ipi_tuned_mean_ci_tuple}")
        ols_results.append(
            get_tracking_input_census_ols(
                ci_method_name="ipi_tuned_mean",
                ci=ipi_tuned_mean_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                trial_seed=args.trial_seed,
                num_masks=num_masks,
                n_neighbors=None,
            )
        )

    if "ipi_tuned_zero" in args.methods:
        ipi_tuned_zero_ci_tuple = ipi_method_zero.get_ipi_ci(
            lambda_weights=None,
            alpha=alpha,
        )
        ols_results.append(
            get_tracking_input_census_ols(
                ci_method_name="ipi_tuned_zero",
                ci=ipi_tuned_zero_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                trial_seed=args.trial_seed,
                num_masks=num_masks,
                n_neighbors=None,
            )
        )

    # naive methods
    if "naive_mf_ci" in args.methods:
        # trains on held out train data, imputes on test data
        naive_mf_ci_tuple = naive_mf_method.get_naive_ci(alpha=alpha)
        ols_results.append(
            get_tracking_input_census_ols(
                ci_method_name="naive_mf_ci",
                ci=naive_mf_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                trial_seed=args.trial_seed,
                num_masks=num_masks,
                n_neighbors=None,
            )
        )

    if "naive_hotdeck_ci" in args.methods:
        # trains on held out train data, imputes on test data
        naive_hotdeck_ci_tuple = naive_hotdeck_method.get_naive_ci(alpha=alpha)
        ols_results.append(
            get_tracking_input_census_ols(
                ci_method_name="naive_hotdeck_ci",
                ci=naive_hotdeck_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                trial_seed=args.trial_seed,
                num_masks=num_masks,
                n_neighbors=args.n_neighbors,
            )
        )

    # naive inplace methods
    if "naive_inplace_mf_ci" in args.methods:
        ## uses full data, fit in place
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
        naive_inplace_mf_method.fit_and_setup(
            train_data=None,
            column_names=column_names,
            num_cat_classes=num_cat_classes,
            do_compile=False,
            categorical_features=categorical_features,
        )
        naive_inplace_em_ci_tuple = naive_inplace_mf_method.get_naive_ci(alpha=alpha)
        ols_results.append(
            get_tracking_input_census_ols(
                ci_method_name="naive_inplace_em_ci",
                ci=naive_inplace_em_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                trial_seed=args.trial_seed,
                num_masks=num_masks,
                n_neighbors=None,
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
            train_data=None, n_neighbors=args.n_neighbors
        )
        naive_inplace_hotdeck_ci_tuple = naive_inplace_hotdeck_method.get_naive_ci(
            alpha=alpha
        )
        ols_results.append(
            get_tracking_input_census_ols(
                ci_method_name="naive_inplace_hotdeck_ci",
                ci=naive_inplace_hotdeck_ci_tuple,
                baseline_classical_interval=ci_tuple,
                true_theta=true_theta,
                target_idx=target_idx,
                ratio=ratio,
                num_fully_observed=num_fully_observed,
                trial_seed=args.trial_seed,
                n_neighbors=args.n_neighbors,
                num_masks=num_masks,
            )
        )

    # IPI per mask methods
    use_methods_with_one_mask = (
        "ipi_per_mask_untuned_mf" in args.methods
        or "ipi_per_mask_tuned_mf" in args.methods
    )
    if use_methods_with_one_mask:
        for k, pattern in enumerate(pattern_to_ids.keys()):
            ## processing for ipi for pattern_k
            ipi_method_pattern_k = ipi_method_mf.copy()
            ipi_method_pattern_k.setup_imputed_data(
                fully_obs_data_dict={
                    ipi_method_pattern_k.fully_obs_tuple: ipi_method_mf.fully_obs_data,
                    pattern: ipi_method_mf.fully_obs_data_dict[pattern],
                },
                imputed_partially_obs_data=ipi_method_mf.imputed_partially_obs_data[
                    list(pattern_to_ids_for_ipi[pattern]), :
                ],
                pattern_to_ids={
                    pattern: list(range(len(pattern_to_ids_for_ipi[pattern])))
                },
            )

            if "ipi_per_mask_untuned_mf" in args.methods:
                ipi_k_untuned_ci_tuple = ipi_method_pattern_k.get_ipi_ci(
                    lambda_weights=np.ones(1),
                    alpha=alpha,
                )
                ols_results.append(
                    get_tracking_input_census_ols(
                        ci_method_name=f"ipi_mask_{k}_untuned",
                        ci=ipi_k_untuned_ci_tuple,
                        baseline_classical_interval=ci_tuple,
                        true_theta=true_theta,
                        target_idx=target_idx,
                        ratio=ratio,
                        num_fully_observed=num_fully_observed,
                        trial_seed=args.trial_seed,
                        num_masks=num_masks,
                        n_neighbors=None,
                    )
                )
            if "ipi_per_mask_tuned_mf" in args.methods:
                ipi_k_tuned_ci_tuple = ipi_method_pattern_k.get_ipi_ci(
                    lambda_weights=None,
                    alpha=alpha,
                )
                ols_results.append(
                    get_tracking_input_census_ols(
                        ci_method_name=f"ipi_mask_{k}_tuned",
                        ci=ipi_k_tuned_ci_tuple,
                        baseline_classical_interval=ci_tuple,
                        true_theta=true_theta,
                        target_idx=target_idx,
                        ratio=ratio,
                        num_fully_observed=num_fully_observed,
                        trial_seed=args.trial_seed,
                        num_masks=num_masks,
                        n_neighbors=None,
                    )
                )
    return ols_results


## run census survey experiment function
def run_census_survey_expt(
    init_seed: int,
    num_fully_observed: int,
    ratio: int,
    num_masks: int,
    num_trials: int,
    methods: list[str],
    header_name: str,
    data_dir: Path | None,
) -> str:
    REGR_FEATURES = [INTERCEPT_COL, "AGEP", "SCHL"]
    REGR_FEATURES_ALLOC = ["FAGEP", "FSCHLP"]
    OUTCOME = "PINCP"
    OUTCOME_ALLOC = "FPINCP"
    TARGET_IDX = 2
    ALPHA = 0.1  # significance level
    NUM_FOLDS = 10  # number of folds
    TRAIN_PERCENT = 1 / NUM_FOLDS
    NUM_BOOTSTRAP_TRIALS = 50
    N_NEIGHBORS = 1

    df_census_full, _, _ = get_census_survey_data(
        year=2019,
        regr_features=REGR_FEATURES,
        regr_ft_alloc=REGR_FEATURES_ALLOC,
        outcome=OUTCOME,
        outcome_alloc=OUTCOME_ALLOC,
        topk=num_masks,
        data_dir=data_dir,
    )
    column_names = df_census_full.columns.tolist()
    regr_feature_idxs = [column_names.index(feature) for feature in REGR_FEATURES]
    outcome_idx = column_names.index(OUTCOME)
    regression_type = "ols"

    TRUE_THETA = get_point_estimate(
        data_array=df_census_full.to_numpy(),
        regr_feature_idxs=regr_feature_idxs,
        outcome_idx=outcome_idx,
        regression_type=regression_type,
        return_se=False,
    )

    results_list = []
    for j in tqdm(range(num_trials), desc="Running trials"):
        results = census_compute_trial_oop(
            ComputeTrialCensusArgs(
                num_fully_observed=num_fully_observed,
                ratio=ratio,
                num_masks=num_masks,
                regr_features=REGR_FEATURES,
                regr_features_alloc=REGR_FEATURES_ALLOC,
                outcome=OUTCOME,
                outcome_alloc=OUTCOME_ALLOC,
                target_idx=TARGET_IDX,
                true_theta=TRUE_THETA[TARGET_IDX],
                alpha=ALPHA,
                train_percent=TRAIN_PERCENT,
                num_folds=NUM_FOLDS,
                num_bootstrap_trials=NUM_BOOTSTRAP_TRIALS,
                methods=methods,
                trial_seed=init_seed + j,
                n_neighbors=N_NEIGHBORS,
            )
        )
        results_list.extend(results)
    results_df = pl.DataFrame(results_list)
    experiment_name = f"census_survey/{header_name}/n0_{num_fully_observed}_ratio{ratio}_{num_masks}masks_numtrials{num_trials}_initseed{init_seed}"

    os.makedirs(RES_DIR / experiment_name, exist_ok=True)
    results_df.write_csv(RES_DIR / experiment_name / "results.csv")
    return str(RES_DIR / experiment_name / "results.csv")
