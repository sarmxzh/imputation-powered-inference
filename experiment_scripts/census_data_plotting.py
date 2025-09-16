# %%
import polars as pl

from ipi.config import RES_DIR
from ipi.utils.plotting_utils import (
    display_results_table,
    find_best_ipi_pattern,
    generate_composite_facet_plot,
    get_base_estimator,
    report_average_metrics,
)

# %%
census_expts_dir = RES_DIR / "census_survey"
experiment_name_ipi = "ipi_w_schooling"
# %%
n0 = 2000
ratios = [10, 30, 50]
num_masks = 10
total_trials = 100
num_trial_per_expt = 5
init_seed = 0
# %%
results_list = []
print("Loading data")
for ratio in ratios:
    for seed_add in range(0, total_trials, num_trial_per_expt):
        results = pl.read_csv(
            census_expts_dir
            / experiment_name_ipi
            / f"n0_{n0}_ratio{ratio}_{num_masks}masks_numtrials{num_trial_per_expt}_initseed{init_seed + seed_add}"
            / "results.csv"
        )
        print(f"Loaded {len(results)/(2 * (10 +1) + 1)} trials")
        results_list.append(results)

results_df = pl.concat(results_list)
results_df.head()
# %%
# Add base_estimator column
results_df = results_df.with_columns(
    pl.col("estimator")
    .map_elements(get_base_estimator, return_dtype=pl.Utf8)
    .alias("base_estimator")
)
# Cast coverage to float
results_df = results_df.with_columns(pl.col("coverage").cast(pl.Float64))
# Convert N_to_n0_ratio to integer for proper numerical ordering
results_df = results_df.with_columns(pl.col("N_to_n0_ratio").cast(pl.Int64))
results_df = results_df.with_columns(
    pl.col("N_to_n0_ratio").cast(pl.Utf8).alias("N_to_n0_factor")
)

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
# Define target estimators for the plot based on the requested order
target_estimators_for_plot = [
    "Complete case",
    ipi_best_pattern_name,
    "IPI (MissForest)",
    "Single Imputation (MissForest)",
]
# %%

# Filter out None values from target_estimators_for_plot
target_estimators_for_plot = [
    est for est in target_estimators_for_plot if est is not None
]

# Verify which of these are actually available after get_base_estimator mapping
all_available_estimators = results_df_untuned["base_estimator"].unique().to_list()
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
    coverage_col_name="coverage",
    num_mask_col_name="num_masks",
    num_fully_observed_col_name="num_fully_observed",
    nmcar_shift_magnitude_col_name=None,
)

avg_vals_tuned = report_average_metrics(
    df_tuned_filtered,
    ratio_col_name="N_to_n0_ratio",
    eff_sample_col_name="n_eff",
    coverage_col_name="coverage",
    num_mask_col_name="num_masks",
    num_fully_observed_col_name="num_fully_observed",
    nmcar_shift_magnitude_col_name=None,
)

# %%

# Generate and save the composite plot
composite_plot = generate_composite_facet_plot(
    avg_untuned_data=avg_vals_untuned,
    avg_tuned_data=avg_vals_tuned,
    estimator_order=target_estimators_for_plot,  # Defined earlier: ["Complete case", ipi_best_pattern_name, "IPI", "CIPI"]
    file_path=str(
        RES_DIR / "census_survey" / "ipi_w_schooling" / "expt2_results_plot.png"
    ),  # Base path for saving
    x_by_feature="N_to_n0_ratio",
    x_axis_label="Partially observed to fully observed sample ratio",
    hide_n_eff_for_baselines=True,
    color_values={
        "Complete case": "#0072B2",
        "AIPW baseline": "#009E73",
        ipi_best_pattern_name: "#DF65B0",
        "Single Imputation (MissForest)": "#4292C6",
        "IPI (MissForest)": "#FD8D3C",
    },
    shape_values={
        "Complete case": "o",
        ipi_best_pattern_name: "s",
        "Single Imputation (MissForest)": "D",
        "IPI (MissForest)": "s",
    },
    legend_marker_size=20,
)
# To display in notebooks if needed
composite_plot


# %%
# Display table with values represented in the plot
display_results_table(avg_vals_untuned, avg_vals_tuned)

# %%
