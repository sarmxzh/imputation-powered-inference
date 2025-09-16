## code for testing sensitivit of mcar first moment tests
## leverages factor model setting

import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from ipi.baselines import classical_ci
from ipi.config import INTERCEPT_COL, RES_DIR
from ipi.ipi_methods import EM_IPI
from ipi.utils.factor_model_utils import generate_dataset_ols

logger = logging.getLogger(__name__)


# %%
@dataclass
class RunDiagnosticExptArgs:
    shift_magnitudes: np.ndarray | None
    num_fully_observed: int
    ratio: int
    num_masks: int
    true_cov_matrix: np.ndarray
    prob_missing: float
    mask_seed: int
    column_names: list[str]
    train_percent: float
    est_latent_dim: int
    alpha: float
    num_folds: int
    regr_features: list[str]
    outcome: str
    target_idx: int
    true_theta: float
    trial_seed: int | None


@dataclass
class DiagnosticExptResult:
    lb: float
    ub: float
    coverage: bool
    estimator: str
    test1_stat: float
    test2_stat: float
    test1_pvalue: float
    test2_pvalue: float
    target_idx: int
    N_to_n0_ratio: float
    num_fully_observed: int
    prob_missing: float
    n_eff: float
    Q_est: int | None
    trial_seed: int
    num_masks: int
    mask_seed: int
    shift_magnitude: np.ndarray | None


def run_one_diagnostics_trial(args: RunDiagnosticExptArgs) -> DiagnosticExptResult:
    if args.trial_seed is not None:
        np.random.seed(args.trial_seed)
        random.seed(args.trial_seed)
    else:
        logger.info(
            "No trial seed provided, using pid and timestamp for numpy random seed"
        )
        pid_timestamp_seed = (os.getpid() * int(time.time_ns())) % 123456789
        np.random.seed(pid_timestamp_seed)
        random.seed(pid_timestamp_seed)

    ## Load in dataset
    fm_dataset = generate_dataset_ols(
        num_fully_observed=args.num_fully_observed,
        num_partially_observed=int(args.ratio * args.num_fully_observed),
        num_unique_patterns=args.num_masks,
        true_cov_matrix=args.true_cov_matrix,
        prob_missing=args.prob_missing,
        mask_seed=args.mask_seed,
        column_names=args.column_names,
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
    regr_features_idxs = [column_names.index(feature) for feature in args.regr_features]
    outcome_idx = column_names.index(args.outcome)

    fullyobs_data_train, fullyobs_data_test = train_test_split(
        fullyobs_data, train_size=args.train_percent, random_state=42
    )
    partiallyobs_data_train, partiallyobs_data_test = train_test_split(
        partiallyobs_data, train_size=args.train_percent, random_state=42
    )
    train_data = np.vstack([fullyobs_data_train, partiallyobs_data_train])

    classical_ci_tuple = classical_ci(
        fully_obs_data=fullyobs_data_test,
        regr_features_idxs=regr_features_idxs,
        outcome_idx=outcome_idx,
        target_idx=args.target_idx,
        regression_type="ols",
        alpha=args.alpha,
    )

    ipi_method_em = EM_IPI(
        regression_type="ols",
        fully_obs_data=fullyobs_data_test,
        partially_obs_data=partiallyobs_data_test,
        regr_features_idxs=regr_features_idxs,
        outcome_idx=outcome_idx,
        target_idx=args.target_idx,
    )

    # fit IPI model
    ipi_method_em.fit_and_setup(
        train_data=train_data,
        Q_est=args.est_latent_dim,
    )

    ipi_tuned_em_ci_tuple = ipi_method_em.get_ipi_ci(
        lambda_weights=None,
        alpha=args.alpha,
        shift_magnitudes=args.shift_magnitudes,
    )

    n_eff = (
        (ipi_tuned_em_ci_tuple[1] - ipi_tuned_em_ci_tuple[0])
        / (classical_ci_tuple[1] - classical_ci_tuple[0])
    ) ** 2 * args.num_fully_observed

    # diagnostics
    test1_res = ipi_method_em.test1_mcar_moment_conditions(
        shift_magnitudes=args.shift_magnitudes,
        track_stats=True,
    )
    test2_res = ipi_method_em.test2_mcar_moment_conditions(
        shift_magnitudes=args.shift_magnitudes,
        track_stats=True,
    )

    # return results
    return DiagnosticExptResult(
        lb=ipi_tuned_em_ci_tuple[0],
        ub=ipi_tuned_em_ci_tuple[1],
        coverage=ipi_tuned_em_ci_tuple[0] <= args.true_theta
        and ipi_tuned_em_ci_tuple[1] >= args.true_theta,
        estimator="ipi_tuned_em",
        test1_stat=test1_res[0],
        test2_stat=test2_res[0],
        test1_pvalue=test1_res[1],
        test2_pvalue=test2_res[1],
        target_idx=args.target_idx,
        N_to_n0_ratio=args.ratio,
        num_fully_observed=args.num_fully_observed,
        n_eff=n_eff,
        Q_est=args.est_latent_dim,
        trial_seed=args.trial_seed,
        num_masks=args.num_masks,
        prob_missing=args.prob_missing,
        mask_seed=args.mask_seed,
        shift_magnitude=args.shift_magnitudes,
    )


# %%
def run_diagnostic_expt(
    shift_magnitudes: list[float] | None,
    ratio: int,
    init_seed: int,
    num_trials: int,
    num_fully_observed: int,
    num_features: int,
    num_factors: int,
    prob_missing: float,
    data_dir: Path | None,
    header_dir: str,
) -> str:
    from tqdm import tqdm

    from ipi.config import DATA_DIR
    from ipi.utils.factor_model_utils import load_cov, make_and_save_cov

    ## set of parameters that should be the same for all trials
    OUTCOME = "X2"
    REGR_FEATURES = [INTERCEPT_COL, "X0", "X1"]
    TARGET_IDX = 1
    NUM_MASKS = len(shift_magnitudes) if shift_magnitudes is not None else 10
    ALPHA = 0.1
    COLUMNS = [f"X{i}" for i in range(num_features)]
    NUM_FOLDS = 10
    TRAIN_PERCENT = 1 / NUM_FOLDS
    EST_LATENT_DIM = 4
    MASK_SEED = 110

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
        results = run_one_diagnostics_trial(
            RunDiagnosticExptArgs(
                shift_magnitudes=shift_magnitudes,
                num_fully_observed=num_fully_observed,
                ratio=ratio,
                num_masks=NUM_MASKS,
                true_cov_matrix=TRUE_COV_MATRIX,
                prob_missing=prob_missing,
                mask_seed=MASK_SEED,
                column_names=COLUMNS,
                train_percent=TRAIN_PERCENT,
                est_latent_dim=EST_LATENT_DIM,
                alpha=ALPHA,
                num_folds=NUM_FOLDS,
                regr_features=REGR_FEATURES,
                outcome=OUTCOME,
                target_idx=TARGET_IDX,
                true_theta=TRUE_THETA,
                trial_seed=init_seed + j,
            )
        )
        results_list.append(results)
    results_df = pl.DataFrame(results_list, strict=False)
    if shift_magnitudes is not None:
        # tranform shift_magnitudes to string
        shift_magnitudes_str = "_".join(
            [str(magnitude) for magnitude in shift_magnitudes]
        )
        experiment_name = f"factor_model_ols/{header_dir}/sm{shift_magnitudes_str}_n0_{num_fully_observed}_ratio{ratio}_d{num_features}_q{num_factors}_p{prob_missing}_{NUM_MASKS}masks_Qest{EST_LATENT_DIM}_numtrials{num_trials}_initseed{init_seed}"
    else:
        experiment_name = f"factor_model_ols/{header_dir}/n0_{num_fully_observed}_ratio{ratio}_d{num_features}_q{num_factors}_p{prob_missing}_{NUM_MASKS}masks_Qest{EST_LATENT_DIM}_numtrials{num_trials}_initseed{init_seed}"

    os.makedirs(RES_DIR / experiment_name, exist_ok=True)
    results_df.write_parquet(RES_DIR / experiment_name / "results.parquet")
    return str(RES_DIR / experiment_name / "results.parquet")
