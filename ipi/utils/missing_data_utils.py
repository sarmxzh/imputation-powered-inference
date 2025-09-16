import logging
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ipi.utils.factor_model_em_utils import fm_impute_X

logger = logging.getLogger(__name__)


def get_missing_patterns_and_ids(
    data_array: np.ndarray,
    *,
    sort: bool = True,
    sort_key: str = "lex",
) -> Mapping[tuple[bool, ...], set[int]]:
    ## NOTE: does NOT change the input data_array
    """
    Get missing patterns and their corresponding row indices
    Args:
        data_array: numpy array with missing values (denoted by NaNs)
    Returns:
        pattern_to_indices: Dictionary with missing patterns as keys and row indices in df
        pattern keys are tuples of bool with True for observed and False for missing.
        If sort=True, patterns are inserted in deterministic order based on sort_key.
    Args (sorting):
        sort: whether to deterministically sort the returned mapping's keys.
        sort_key: sorting strategy when sort=True.
            - "lex": lexicographic over the boolean tuples (default).
            - "missing_then_lex": ascending by number of missing (False) values, then lex.
    """
    # It's observed if it's not null and not NaN
    observed_patterns = pd.notna(data_array)

    # Use np.unique to find all distinct boolean rows, and also get 'inverse'
    # telling us which unique pattern each row belongs to
    # inverse is a 1d array mapping each row to the observed pattern
    unique_patterns, inverse = np.unique(observed_patterns, axis=0, return_inverse=True)

    # Build the dictionary mapping pattern -> row indices
    pattern_to_indices = defaultdict(set)

    for row_idx, pattern_idx in enumerate(inverse):
        # Convert the pattern row to a tuple of bool
        pattern_tuple = tuple(unique_patterns[pattern_idx])
        pattern_to_indices[pattern_tuple].add(row_idx)

    # Optionally return a deterministically ordered dict (insertion-ordered)
    if not sort:
        return dict(pattern_to_indices)

    def _key_func(pattern: tuple[bool, ...]):
        if sort_key == "lex":
            return pattern
        if sort_key == "missing_then_lex":
            num_missing = sum(not v for v in pattern)
            return (num_missing, pattern)
        raise ValueError(
            f"Unknown sort_key '{sort_key}'. Expected 'lex' or 'missing_then_lex'."
        )

    return {
        pat: pattern_to_indices[pat]
        for pat in sorted(pattern_to_indices.keys(), key=_key_func)
    }


def impute_missing_values(
    data_array: np.ndarray,
    column_names: list[str],
    model: str,
    trained_model: object,
) -> np.ndarray:
    """
    Impute missing values in df_partial
    Args:
        df_partial: pandas DataFrame with missing values (denoted by NaNs)
        mask: pandas Series with boolean values (True for observed, False for missing)
    Returns:
        dfhat_partial: pandas DataFrame with imputed missing values
    """
    from ipi.missforest.src.missforest import MissForest

    dfhat_partial = data_array.copy()
    if model == "EM":
        M_partial = (~np.isnan(data_array)).astype(int)  # 1 for observed, 0 for missing
        X_partial = data_array
        cov_matrix = trained_model
        dfhat_partial = fm_impute_X(X_partial, M_partial, cov_matrix)
        ## TODO: add functionality for adding a linear regression predictor for Y for mean imputation
    elif model == "missforest":
        df_partial = pd.DataFrame(data_array, columns=column_names)
        if isinstance(trained_model, bytes):
            trained_model = MissForest.load_msgpack(trained_model)

        dfhat_partial = trained_model.transform(df_partial)
        dfhat_partial = dfhat_partial[
            df_partial.columns  ## this makes sure that the columns are ordered the same way as the input dataframe
        ]
        dfhat_partial = dfhat_partial.to_numpy()

        ## NOTE: missforest rearranges the columns by percentage of missingness
        ## NOTE: we need to change the columns back (especially if we want to use numpy array indexing)
        # This is done in the return statement below to catch any issues
    elif model == "mean_imputation":
        for col in range(data_array.shape[1]):
            idxs_bool = np.isnan(data_array[:, col])
            if len(idxs_bool) == 0:
                logger.info(f"No missing values for {col} -- skipping imputing...")
                continue  # skip if no missing values
            # impute missing values
            dfhat_partial[idxs_bool, col] = trained_model[col]
    else:
        raise ValueError(f"Model {model} not recognized.")

    return dfhat_partial
