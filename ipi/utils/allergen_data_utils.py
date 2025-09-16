## contains main functions for allergen data experiments
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ipi.baselines import classical_ci
from ipi.config import DATA_DIR, INTERCEPT_COL, RES_DIR
from ipi.cross_ipi_methods import MF_CrossIPI
from ipi.ipi_methods import MF_IPI
from ipi.utils.general_utils import _json_default
from ipi.utils.missing_data_utils import get_missing_patterns_and_ids
from ipi.utils.plotting_utils import plot_missing_patterns

logger = logging.getLogger(__name__)

## hard coded global variables for allergen data
IGE_FEATURES = [
    "Act d 1",
    "Act d 2",
    "Act d 5",
    "Act d 8",
    "Aln g 1",
    "Alt a 1",
    "Alt a 6",
    "Amb a 1",
    "Ana o 2",
    "Ani s 1",
    "Ani s 3",
    "Api g 1",
    "Api m 1",
    "Api m 4",
    "Ara h 1",
    "Ara h 2",
    "Ara h 3",
    "Ara h 6",
    "Ara h 8",
    "Ara h 9",
    "Art v 1",
    "Art v 3",
    "Asp f 1",
    "Asp f 3",
    "Asp f 6",
    "Ber e 1",
    "Bet v 1",
    "Bet v 2",
    "Bet v 4",
    "Bla g 1",
    "Bla g 2",
    "Bla g 5",
    "Bla g 7",
    "Blo t 5",
    "Bos d 4",
    "Bos d 5",
    "Bos d 6",
    "Bos d 8",
    "Bos d Lactoferrin",
    "Can f 1",
    "Can f 2",
    "Can f 3",
    "Can f 5",
    "Che a 1",
    "Cla h 8",
    "Cor a 1.0101",
    "Cor a 1.0401",
    "Cor a 8",
    "Cor a 9",
    "Cry j 1",
    "Cup a 1",
    "Cyn d 1",
    "Der f 1",
    "Der f 2",
    "Der p 1",
    "Der p 10",
    "Der p 2",
    "Equ c 1",
    "Equ c 3",
    "Fag e 2",
    "Fel d 1",
    "Fel d 2",
    "Fel d 4",
    "Gad c 1",
    "Gal d 1",
    "Gal d 2",
    "Gal d 3",
    "Gal d 5",
    "Gly m 4",
    "Gly m 5",
    "Gly m 6",
    "Hev b 1",
    "Hev b 3",
    "Hev b 5",
    "Hev b 6.01",
    "Hev b 8",
    "Jug r 1",
    "Jug r 2",
    "Jug r 3",
    "Lep d 2",
    "Mal d 1",
    "Mer a 1",
    "Mus m 1",
    "MUXF3",
    "Ole e 1",
    "Ole e 7",
    "Ole e 9",
    "Par j 2",
    "Pen m 1",
    "Pen m 2",
    "Pen m 4",
    "Phl p 1",
    "Phl p 11",
    "Phl p 12",
    "Phl p 2",
    "Phl p 4",
    "Phl p 5",
    "Phl p 6",
    "Phl p 7",
    "Pla a 1",
    "Pla a 2",
    "Pla a 3",
    "Pla l 1",
    "Pol d 5",
    "Pru p 1",
    "Pru p 3",
    "Sal k 1",
    "Ses i 1",
    "Tri a 14",
    "Tri a 19.0101",
    "Tri a aA_TI",
    "Ves v 5",
]
AIRWAY_IGES = [
    "Aln g 1",
    "Amb a 1",
    "Art v 1",
    "Art v 3",
    "Art v 1",
    "Art v 3",
    "Asp f 1",
    "Asp f 3",
    "Asp f 6",
    "Bet v 1",
    "Bet v 2",
    "Bet v 4",
    "Bla g 1",
    "Bla g 2",
    "Bla g 5",
    "Bla g 7",
    "Blo t 5",
    "Can f 1",
    "Can f 2",
    "Can f 3",
    "Can f 5",
    "Che a 1",
    "Cla h 8",
    "Cor a 1.0101",
    "Cor a 1.0401",
    "Cry j 1",
    "Cup a 1",
    "Cyn d 1",
    "Der f 1",
    "Der f 2",
    "Der p 1",
    "Der p 10",
    "Der p 2",
    "Equ c 1",
    "Equ c 3",
    "Fel d 1",
    "Fel d 2",
    "Fel d 4",
    "Mer a 1",
    "Mus m 1",
    "Ole e 1",
    "Ole e 7",
    "Ole e 9",
    "Par j 2",
    "Phl p 1",
    "Phl p 11",
    "Phl p 12",
    "Phl p 2",
    "Phl p 4",
    "Phl p 5",
    "Phl p 6",
    "Phl p 7",
    "Pla a 1",
    "Pla a 2",
    "Pla a 3",
    "Pla l 1",
]

ARA_H_IGES = [
    "Ara h 1",
    "Ara h 2",
    "Ara h 3",
    "Ara h 6",
    "Ara h 8",
    "Ara h 9",
]

CAT_FEATURES = [
    "Code local",
    "CHECK",
    "CODE puce",
    "Sexe",
    "Département de résidence",
    "Région de France",
    "Type d'habitat",
    "Atopie familliale (parents et fratrie)",
    "ATCD ou actuel Dermatite Atopique",
    "Traitement actuel DA",
    "Conjonctivite",
    "Syndrome oral",
    "ATCD de pathologie mastocytaire",
    "ATCD d'anaphylaxie aux venins",
    "Peau/urticaire",
    "Digestif",
    "respiratoire",
    "CV",
    "Polyallergie alimentaire",
    "Viande de mammifère",
    "Puce réalisée",
    "Cofacteur_0",
    "Cofacteur_1",
    "Cofacteur_2",
    "Cofacteur_3",
    "Cofacteur_4",
    "Cofacteur_5",
    "Cofacteur_6",
    "Cofacteur_7",
    "Cofacteur_8",
    "Cofacteur_9",
    "Anaphylaxie_0",
    "Anaphylaxie_1",
    "Anaphylaxie_2",
    "Anaphylaxie_3",
    "Anaphylaxie_4",
    "Anaphylaxie_5",
    "Anaphylaxie_6",
    "Anaphylaxie_7",
    "Anaphylaxie_8",
    "Clinique_severite",
    "Clinique_arachide",
    "Clinique_oeuf",
    "Clinique_noisette",
    "Clinique_noix",
    "Clinique_cajou",
    "Clinique_pistache",
    "Clinique_amande",
    "Clinique_fruits à coque",
    "Clinique_soja",
    "Clinique_crevette",
    "Clinique_poisson",
    "Clinique_blé",
    "Clinique_sésame",
    "Clinique_sarrasin",
    "Clinique_lait",
    "Clinique_autres",
    "NSRegion",
]


def process_and_save_allergen_data(data_dir: str | None = None) -> None:
    """
    Process and save allergen data for allergen experiments.
    Requires the ACC_2023_Chip1_cleaned.csv file to be in the DATA_DIR / "allergen_data" directory.
    """
    if data_dir is None:
        data_dir = DATA_DIR / "allergen_data"
    data_dir = Path(data_dir)
    allergen_chip1_df = pd.read_csv(
        data_dir / "ACC_2023_Chip1_cleaned.csv", index_col=0
    )

    # compute average IGE readings for each row
    allergen_chip1_df["avg_Ige_readings"] = allergen_chip1_df[IGE_FEATURES].mean(axis=1)
    allergen_chip1_df["avg_airwayIge_readings"] = allergen_chip1_df[AIRWAY_IGES].mean(
        axis=1
    )

    # only sample rows with positive IGE readings (some allergens detected)
    allergen_chip1_df = allergen_chip1_df[
        allergen_chip1_df["avg_Ige_readings"] > 0
    ].reset_index(drop=True)

    # features in experiment
    FEATURES_IN_EXPT = [
        INTERCEPT_COL,
        "Age",
        "Sexe",
        "Région de France",  # region of france
        "ARIA (rhinite)",  # rhinitis
        "Ara h 1",  # ara h IgE readings
        "Ara h 2",
        "Ara h 3",
        "Ara h 6",
        "Ara h 8",
        "Polyallergie alimentaire",  # polyallergy
        "Conjonctivite",  # pink eye
        "GINA (ancien)",  # asthma severity
        "Mois du prélèvement",  # month of sampling
        "avg_airwayIge_readings",  # avg airway ige readings, does *not* involve peanut ara h proteins
        "Cofacteur_0",  # cofactors
        "Cofacteur_1",
        "Cofacteur_2",
        "Cofacteur_3",
        "Cofacteur_4",
        "Cofacteur_5",
        "Cofacteur_6",
        "Cofacteur_7",
        "Cofacteur_8",
        "Cofacteur_9",
        "Anaphylaxie_0",  # anaphylaxis categories
        "Anaphylaxie_1",
        "Anaphylaxie_2",
        "Anaphylaxie_3",
        "Anaphylaxie_4",
        "Anaphylaxie_5",
        "Anaphylaxie_6",
        "Anaphylaxie_7",
        "Anaphylaxie_8",
    ]

    # add intercept
    allergen_chip1_df[INTERCEPT_COL] = 1
    df = allergen_chip1_df[FEATURES_IN_EXPT]
    # save dataframe
    df.to_csv(data_dir / "allergen_chip1_processed_df.csv", index=False)


@dataclass
class FilterDatasetForExptVars:
    filtered_df: pd.DataFrame
    column_names: list[str]
    outcome: str
    regr_features: list[str]
    feature_of_inference: str
    regr_features_idxs: list[int]
    outcome_idx: int
    target_idx: int
    num_cat_classes: dict[str, int]
    categorical_features: list[str]


def filter_dataset_for_expt(
    outcome: str,
    regr_features: list[str],
    feature_of_inference: str,
    data_dir: str | Path | None,
    do_plot_missing_patterns: bool,
) -> FilterDatasetForExptVars:
    """
    Filter dataset for experiment.
    """
    if data_dir is None:
        data_dir = DATA_DIR / "allergen_data"

    # load in dataframe
    try:
        df = pd.read_csv(data_dir / "allergen_chip1_processed_df.csv")
    except FileNotFoundError:
        process_and_save_allergen_data(data_dir=data_dir)
        df = pd.read_csv(data_dir / "allergen_chip1_processed_df.csv")

    try:
        target_idx = regr_features.index(feature_of_inference)
    except ValueError as err:
        raise ValueError(
            f"Feature of inference {feature_of_inference} not found in regr_features"
        ) from err

    # process data
    df = df[df["Région de France"] != 14]

    northern_regions = [2, 3, 4, 6, 7, 8, 9, 12]
    # southern_regions = [1, 5, 10, 11, 13]

    df["NSRegion"] = np.where(df["Région de France"].isin(northern_regions), 0, 1)
    df.drop(columns=["Région de France"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # get missing patterns and ids
    pattern_to_ids = get_missing_patterns_and_ids(df.to_numpy())
    # plot missing patterns
    if do_plot_missing_patterns:
        plot_missing_patterns(
            pattern_to_ids=pattern_to_ids,
            column_names=df.columns.tolist(),
            png_dir=data_dir,
            png_filename="original_missing_patterns.png",
            save_fig=True,
        )
    # check for number of unique patterns in pattern_to_ids
    print(len(pattern_to_ids))
    MASK_MIN_COUNT = 50
    patterns_count = {pattern: len(ids) for pattern, ids in pattern_to_ids.items()}
    populated_patterns = [
        pattern for pattern, count in patterns_count.items() if count >= MASK_MIN_COUNT
    ]
    filtered_patterns_to_ids = {
        pattern: ids
        for pattern, ids in pattern_to_ids.items()
        if pattern in populated_patterns
    }
    print(len(filtered_patterns_to_ids))

    filtered_ids = {
        idx for sublist in filtered_patterns_to_ids.values() for idx in sublist
    }
    print(len(filtered_ids))

    filtered_df = df.loc[list(filtered_ids)].reset_index(drop=True)
    if do_plot_missing_patterns:
        plot_missing_patterns(
            pattern_to_ids=filtered_patterns_to_ids,
            column_names=df.columns.tolist(),
            png_dir=data_dir,
            png_filename="filtered_missing_patterns.png",
            save_fig=True,
        )

    categorical_features = [col for col in CAT_FEATURES if col in filtered_df.columns]
    for _idx, col in enumerate(filtered_df.columns):
        if col != INTERCEPT_COL and col in categorical_features:
            le = LabelEncoder()

            # Create a mask for non-NaN values in the current column
            non_nan_mask = filtered_df[col].notna()

            # If there are any non-NaN values to encode
            if non_nan_mask.any():
                # Fit the LabelEncoder and transform only the non-NaN values
                # NaNs will remain NaNs in the column
                filtered_df.loc[non_nan_mask, col] = le.fit_transform(
                    filtered_df.loc[non_nan_mask, col]
                )
                # If a column was all NaNs, it remains all NaNs.
                # If a column had mixed NaNs and values, NaNs are preserved, and values are encoded.
                # The column dtype might change to float to accommodate NaNs alongside integer codes.

    num_cat_classes = {col: filtered_df[col].nunique() for col in categorical_features}

    column_names = filtered_df.columns.tolist()
    regr_features_idxs = [column_names.index(feature) for feature in regr_features]
    outcome_idx = column_names.index(outcome)

    return FilterDatasetForExptVars(
        filtered_df=filtered_df.reset_index(drop=True),
        column_names=column_names,
        outcome=outcome,
        regr_features=regr_features,
        feature_of_inference=feature_of_inference,
        regr_features_idxs=regr_features_idxs,
        outcome_idx=outcome_idx,
        target_idx=target_idx,
        num_cat_classes=num_cat_classes,
        categorical_features=categorical_features,
    )


def run_allergen_expt(
    outcome: str,
    regr_features: list[str],
    feature_of_inference: str,
    num_folds: int,
    num_bootstrap_trials: int,
    train_percent: float,
    train_test_split_seed: int,
    track_cipi: bool,
    header_dir: str | Path | None,
    data_dir: str | Path | None,
) -> str:
    """
    Replicate allergen data experiment.

    Args:
        outcome: outcome variable
        regr_features: regression features
        feature_of_inference: feature of inference in regr_features list
        num_folds: number of folds
        num_bootstrap_trials: number of bootstrap trials
        train_percent: train percent
        train_test_split_seed: train test split seed
        cf_random_state: cross-validation random state
        track_cipi: track cipi
        header_dir: header directory
        data_dir: data directory
    Returns:
        file path string
    """
    filter_expt_vars = filter_dataset_for_expt(
        outcome=outcome,
        regr_features=regr_features,
        feature_of_inference=feature_of_inference,
        data_dir=data_dir,
        do_plot_missing_patterns=False,
    )
    filtered_df = filter_expt_vars.filtered_df
    column_names = filter_expt_vars.column_names
    regr_features_idxs = filter_expt_vars.regr_features_idxs
    outcome_idx = filter_expt_vars.outcome_idx
    target_idx = filter_expt_vars.target_idx
    num_cat_classes = filter_expt_vars.num_cat_classes
    categorical_features = filter_expt_vars.categorical_features

    # classical ols for fully observed X, U data
    fully_observed_df = filtered_df[filtered_df.T.notna().all()]
    classical_ci_tuple = classical_ci(
        fully_obs_data=fully_observed_df.to_numpy(),
        regr_features_idxs=regr_features_idxs,
        outcome_idx=outcome_idx,
        target_idx=target_idx,
        regression_type="ols",
        alpha=0.1,
    )

    # stuff for IPI and CIPI
    partially_observed_df = filtered_df[filtered_df.T.isna().any()]

    # for IPI specifically
    # train-test split
    fullyobs_data_train, fullyobs_data_test = train_test_split(
        fully_observed_df.to_numpy(),
        train_size=train_percent,
        random_state=train_test_split_seed,
    )
    partiallyobs_data_train, partiallyobs_data_test = train_test_split(
        partially_observed_df.to_numpy(),
        train_size=train_percent,
        random_state=train_test_split_seed,
    )

    train_data = np.vstack([fullyobs_data_train, partiallyobs_data_train])

    ipi_method_mf = MF_IPI(
        fully_obs_data=fullyobs_data_test,
        partially_obs_data=partiallyobs_data_test,
        regr_features_idxs=regr_features_idxs,
        outcome_idx=outcome_idx,
        target_idx=target_idx,
        regression_type="ols",
    )
    ipi_method_mf.fit_and_setup(
        train_data=train_data,
        column_names=column_names,
        num_cat_classes=num_cat_classes,
        categorical_features=categorical_features,
        do_compile=False,
    )

    ipi_tuned_mf_ci_tuple = ipi_method_mf.get_ipi_ci(
        lambda_weights=None,
        alpha=0.1,
    )
    logger.info(ipi_tuned_mf_ci_tuple)
    ipi_p_value_test1 = ipi_method_mf.test1_mcar_moment_conditions()
    ipi_p_value_test2 = ipi_method_mf.test2_mcar_moment_conditions()
    logger.info(ipi_p_value_test1)
    logger.info(ipi_p_value_test2)

    if track_cipi:
        cipi_method_mf = MF_CrossIPI(
            regression_type="ols",
            fully_obs_data=fully_observed_df.to_numpy(),
            partially_obs_data=partially_observed_df.to_numpy(),
            regr_features_idxs=regr_features_idxs,
            outcome_idx=outcome_idx,
            target_idx=target_idx,
        )
        cipi_tuned_mf_ci_tuple = cipi_method_mf.get_ci_cipi(
            num_folds=num_folds,
            num_bootstrap_trials=num_bootstrap_trials,
            lambda_weights=None,
            alpha=0.1,
            cf_random_state=train_test_split_seed,
            categorical_features=categorical_features,
            num_cat_classes=num_cat_classes,
            column_names=column_names,
            do_compile=False,
        )
    else:
        logger.info("WARNING: CIPI not tracked for allergen data expt")
        cipi_tuned_mf_ci_tuple = None

    results_dict = {
        "num_folds": num_folds,
        "train_percent": train_percent,
        "num_bootstrap_trials": num_bootstrap_trials,
        "classical": classical_ci_tuple,
        "ipi_tuned": ipi_tuned_mf_ci_tuple,
        "cipi_tuned": cipi_tuned_mf_ci_tuple,
        "ipi_tuned_p_value_test1": ipi_p_value_test1,
        "ipi_tuned_p_value_test2": ipi_p_value_test2,
        "full_features_list": column_names,
        "regr_features_list": regr_features,
        "outcome": outcome,
        "target_idx": target_idx,
        "train_test_split_seed": train_test_split_seed,
        "feature_of_inference": feature_of_inference,
        "regr_features_idxs": regr_features_idxs,
        "outcome_idx": outcome_idx,
    }
    file_local_dir = Path("allergen_data")

    if header_dir is not None:
        file_local_dir = file_local_dir / header_dir
    file_local_dir = (
        file_local_dir
        / f"splitseed{train_test_split_seed}_numfolds{num_folds}_numbootstraptrials{num_bootstrap_trials}_trainpercent{train_percent}"
    )
    if track_cipi:
        file_local_dir = file_local_dir / "cipi"
    else:
        file_local_dir = file_local_dir / "no_cipi"
    file_name = (
        f"outcome{outcome}_featureofinference{feature_of_inference}_results.json"
    )

    # else, save file locally and return file path str
    os.makedirs(RES_DIR / file_local_dir, exist_ok=True)
    with open(RES_DIR / file_local_dir / file_name, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, default=_json_default, ensure_ascii=False, indent=2)
    return str(RES_DIR / file_local_dir / file_name)
