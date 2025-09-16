# %%
## plotting script for factor model ols experiments

import numpy as np
import polars as pl

from ipi.config import RES_DIR
from ipi.utils.factor_model_utils import load_cov
from ipi.utils.plotting_utils import (
    find_best_ipi_pattern,
    generate_composite_facet_plot,
    get_base_estimator,
    report_average_metrics,
)

# %%
n0s = [200]
ratios = [10, 30, 50]
d = 20
q = 2
p = 0.2
q_est = 4
num_masks = [10]
total_trials = 100
num_trial_per_expt = 10
init_seed = 0
nmcar_shift_magnitudes = [None]

header_name = "test123"


# %%
def load_results(
    nmcar_shift_magnitude: float | None,
    n0: int,
    ratio: int,
    d: int,
    q: int,
    p: float,
    num_mask: int,
    q_est: int,
    num_trial_per_expt: int,
    init_seed: int,
    header_name: str,
    seed_add: int,
) -> pl.DataFrame:
    if nmcar_shift_magnitude is not None:
        experiment_name = f"nsm{nmcar_shift_magnitude}_n0_{n0}_ratio{ratio}_d{d}_q{q}_p{p}_{num_mask}masks_Qest{q_est}_numtrials{num_trial_per_expt}_initseed{init_seed + seed_add}"
    else:
        experiment_name = f"n0_{n0}_ratio{ratio}_d{d}_q{q}_p{p}_{num_mask}masks_Qest{q_est}_numtrials{num_trial_per_expt}_initseed{init_seed + seed_add}"
    print(f"{header_name} / {experiment_name}")
    return pl.read_csv(
        RES_DIR / "factor_model_ols" / header_name / experiment_name / "results.csv"
    )


results_list = []
print("Loading data")
for ratio in ratios:
    for n0 in n0s:
        for num_mask in num_masks:
            for nmcar_shift_magnitude in nmcar_shift_magnitudes:
                for seed_add in range(0, total_trials, num_trial_per_expt):
                    # main one: classical, ipi, ipi single pattern, and cipi
                    results = load_results(
                        nmcar_shift_magnitude=nmcar_shift_magnitude,
                        n0=n0,
                        ratio=ratio,
                        d=d,
                        q=q,
                        p=p,
                        num_mask=num_mask,
                        q_est=q_est,
                        num_trial_per_expt=num_trial_per_expt,
                        init_seed=init_seed,
                        header_name=header_name,
                        seed_add=seed_add,
                    )
                    results = results.with_columns(pl.col("n_neighbors").cast(pl.Int64))
                    results_list.append(results)
                    print(f"Loaded {len(results)/(2 * (10 +1 + 1) + 1)} trials")

results_df = pl.concat(results_list)
results_df.head()

# %%
COV_MATRIX_NAME = f"covq{q}d{d}.safetensors"
cov_data = load_cov(cov_matrix_name=COV_MATRIX_NAME)
true_cov_matrix = cov_data.TRUE_COV_MATRIX
TRUE_THETA = (np.linalg.inv(true_cov_matrix[:2, :2]) @ true_cov_matrix[:2, 2])[0]

# Ensure numeric types for arithmetic
results_df = results_df.with_columns(
    [
        pl.col("lb").cast(pl.Float64, strict=False),
        pl.col("ub").cast(pl.Float64, strict=False),
        pl.col("nmcar_shift_magnitude").cast(pl.Float64, strict=False).fill_null(0.0),
    ]
)

# Add coverage for shifted estimand: theta_shifted = TRUE_THETA * (1 + nmcar_shift_magnitude)
results_df = results_df.with_columns(
    (
        (pl.col("lb") <= (pl.lit(TRUE_THETA) * (1 + pl.col("nmcar_shift_magnitude"))))
        & ((pl.lit(TRUE_THETA) * (1 + pl.col("nmcar_shift_magnitude"))) <= pl.col("ub"))
    ).alias("coverage_shifted")
)
# %%
# Add base_estimator column
results_df = results_df.with_columns(
    pl.col("estimator")
    .map_elements(get_base_estimator, return_dtype=pl.Utf8)
    .alias("base_estimator")
)
# Cast coverage to float
results_df = results_df.with_columns(
    [
        pl.col("coverage").cast(pl.Float64),
        pl.col("coverage_shifted").cast(pl.Float64),
    ]
)
# Convert N_to_n0_ratio to integer for proper numerical ordering
results_df = results_df.with_columns(pl.col("N_to_n0_ratio").cast(pl.Int64))
results_df = results_df.with_columns(
    pl.col("N_to_n0_ratio").cast(pl.Utf8).alias("N_to_n0_factor")
)
# add a num_masks column with 10 as value if num_masks not present
if "num_masks" not in results_df.columns:
    print("Adding num_masks column with 10 as value")
    results_df = results_df.with_columns(pl.lit(10).alias("num_masks"))
# %%
# filter tuned vs untuned (now using base_estimator for clarity if preferred, or stick to raw names)
# For this plot, we focus on untuned results as per original df_untuned
results_df_untuned = results_df.filter(
    pl.col("estimator").str.contains("_untuned")
    | (pl.col("estimator") == "classical")
    | pl.col("estimator").str.contains("naive")
    | (pl.col("estimator") == "aipw_baseline")
    # classical is considered untuned for this purpose
)
results_df_tuned = results_df.filter(
    pl.col("estimator").str.contains("_tuned")
    | (
        pl.col("estimator") == "classical"
    )  # classical is considered untuned for this purpose
    | pl.col("estimator").str.contains("naive")
    | (pl.col("estimator") == "aipw_baseline")
)

# %%
## Find IPI best pattern
ipi_best_pattern_name = find_best_ipi_pattern(results_df_untuned, metric="n_eff")
print(f"IPI best pattern name: {ipi_best_pattern_name}")
# Define target estimators for the plot based on the requested order
target_estimators_for_plot = [
    "Complete case",
    # "AIPW baseline",
    ipi_best_pattern_name,
    "IPI (EM)",
    # "IPI (zero)",
    # "IPI (mean)",
    # "IPI (hotdeck)",
    # "IPI (MissForest)",
    "CIPI (EM)",
    # "Single Imputation (EM)",
    # "Single Imputation (Hotdeck)",
    # "Single Imputation (zero)",
    # "Single Imputation (mean)",
    # "Single Imputation (MissForest)",
]
# %%

# Filter out None values from target_estimators_for_plot
target_estimators_for_plot = [
    est for est in target_estimators_for_plot if est is not None
]

# Verify which of these are actually available after get_base_estimator mapping
all_available_estimators = results_df_tuned["base_estimator"].unique().to_list()
final_target_estimators = [
    est for est in target_estimators_for_plot if est in all_available_estimators
]
missing_estimators = [
    est for est in target_estimators_for_plot if est not in all_available_estimators
]
if missing_estimators:
    print(
        f"Warning: The following requested estimators are not available in the data and will be skipped: {missing_estimators}"
    )

# print(f"Filtering df_untuned for these estimators: {final_target_estimators}")
df_untuned_filtered = results_df_untuned.filter(
    pl.col("base_estimator").is_in(final_target_estimators)
)
df_tuned_filtered = results_df_tuned.filter(
    pl.col("base_estimator").is_in(final_target_estimators)
)


# Report metrics for df_untuned_filtered
avg_vals_untuned = report_average_metrics(
    df_untuned_filtered,
    ratio_col_name="N_to_n0_ratio",
    eff_sample_col_name="n_eff",
    coverage_col_name="coverage",  # shifted",  # _shifted",
    num_mask_col_name="num_masks",
    num_fully_observed_col_name="num_fully_observed",
    nmcar_shift_magnitude_col_name="nmcar_shift_magnitude",  # "nmcar_shift_magnitude",
)

avg_vals_tuned = report_average_metrics(
    df_tuned_filtered,
    ratio_col_name="N_to_n0_ratio",
    eff_sample_col_name="n_eff",
    coverage_col_name="coverage",  # _shifted",  # _shifted",
    num_mask_col_name="num_masks",
    num_fully_observed_col_name="num_fully_observed",
    nmcar_shift_magnitude_col_name="nmcar_shift_magnitude",  # "nmcar_shift_magnitude",
)

color_values_for_including_all_methods = {
    "Complete case": "#0072B2",
    "AIPW baseline": "#009E73",
    "IPI (EM)": "#FD8D3C",
    "Single Imputation (EM)": "#FD8D3C",
    "IPI (MissForest)": "#A50F15",
    "Single Imputation (MissForest)": "#A50F15",
    "IPI (Hotdeck)": "#AA4499",
    "Single Imputation (Hotdeck)": "#AA4499",
    "IPI (mean)": "#56B4E9",
    "Single Imputation (mean)": "#56B4E9",
    "IPI (zero)": "#999933",
    "Single Imputation (zero)": "#999933",
}

shapes_dict = {
    "Complete case": "o",
    ipi_best_pattern_name: "s",
    "IPI (EM)": "s",
    "CIPI (EM)": "s",
    "AIPW baseline": "v",
    "Single Imputation (EM)": "D",
    "Single Imputation (Hotdeck)": "D",
    "Single Imputation (zero)": "D",
    "Single Imputation (mean)": "D",
    "Single Imputation (MissForest)": "D",
    "IPI (MissForest)": "s",
    "IPI (zero)": "s",
    "IPI (mean)": "s",
    "IPI (hotdeck)": "s",
}

colors_dict = {
    "Complete case": "#0072B2",
    "AIPW baseline": "#009E73",
    ipi_best_pattern_name: "#DF65B0",
    "Single Imputation (MissForest)": "#084594",
    "Single Imputation (Hotdeck)": "#2171B5",
    "Single Imputation (EM)": "#4292C6",
    "Single Imputation (zero)": "#6BAED6",
    "Single Imputation (mean)": "#9ECAE1",
    "IPI (MissForest)": "#A63603",
    "IPI (hotdeck)": "#E6550D",
    "IPI (EM)": "#FD8D3C",
    "IPI (mean)": "#FDD0A2",
    "CIPI (EM)": "#EF3B2C",
}

# %%
# Generate and save the composite plot
composite_plot = generate_composite_facet_plot(
    avg_untuned_data=avg_vals_untuned,
    avg_tuned_data=avg_vals_tuned,
    estimator_order=target_estimators_for_plot,  # Defined earlier: ["Complete case", ipi_best_pattern_name, "IPI", "CIPI"]
    file_path=str(
        RES_DIR / "factor_model_ols" / header_name / "expt1_results_plot.png",
    ),  # Base path for saving
    x_by_feature="N_to_n0_ratio",
    x_axis_label="Partially observed to fully observed sample ratio",
    shape_values=shapes_dict,
    legend_marker_size=20,
    color_values=colors_dict,
)
composite_plot  # To display in notebooks if needed


# %%
# Display table with values represented in the plot
print("\n" + "=" * 80)
print("TABLE: Values represented in the plot")
print("=" * 80)

print("\n--- POOLED ESTIMATORS (Untuned) ---")
print(
    "Columns: base_estimator, N_to_n0_ratio, avg_n_eff, avg_coverage, std_n_eff, std_coverage, se_n_eff, se_coverage, count_n_eff"
)
with pl.Config(tbl_rows=-1, tbl_width_chars=120):
    print(
        avg_vals_untuned.select(
            [
                "base_estimator",
                "N_to_n0_ratio",
                "avg_n_eff",
                "avg_coverage",
                "std_n_eff",
                "std_coverage",
                "se_n_eff",
                "se_coverage",
            ]
        ).sort(["base_estimator", "N_to_n0_ratio"])
    )

print("\n--- POWER-TUNED ESTIMATORS (Tuned) ---")
print(
    "Columns: base_estimator, N_to_n0_ratio, avg_n_eff, avg_coverage, std_n_eff, std_coverage, se_n_eff, se_coverage, count_n_eff"
)
with pl.Config(tbl_rows=-1, tbl_width_chars=120):
    print(
        avg_vals_tuned.select(
            [
                "base_estimator",
                "N_to_n0_ratio",
                "avg_n_eff",
                "avg_coverage",
                "std_n_eff",
                "std_coverage",
                "se_n_eff",
                "se_coverage",
            ]
        ).sort(["base_estimator", "N_to_n0_ratio"])
    )

print("\n" + "=" * 80)
print("Note: avg_n_eff = Average effective sample size")
print("      avg_coverage = Average coverage")
print("      std_n_eff = Standard deviation of effective sample size")
print("      std_coverage = Standard deviation of coverage")
print("      se_n_eff = Standard error of effective sample size")
print("      se_coverage = Standard error of coverage")
print("      count_n_eff = Number of trials")
print("=" * 80)

# %%
