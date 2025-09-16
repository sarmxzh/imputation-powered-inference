from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex
from matplotlib.ticker import AutoLocator, MaxNLocator
from plotnine import (
    aes,
    element_rect,  # Added for strip background
    element_text,
    facet_grid,  # Added for faceting
    geom_blank,
    geom_errorbar,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_brewer,
    scale_color_manual,
    scale_shape_manual,  # Added for custom point shapes
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)
from scipy.stats import norm


# %%
# Define get_base_estimator function (consistent with previous versions)
def get_base_estimator(estimator_name: str) -> str:
    if estimator_name is None:
        return "Unknown"
    if "ipi_mask" in estimator_name and "_untuned" in estimator_name:
        pattern_part = estimator_name.replace("ipi_mask_", "").replace("_untuned", "")
        try:
            return f"IPI pattern {int(pattern_part) + 1}"
        except ValueError:
            return f"IPI pattern {pattern_part}"  # If pattern_part is not int-castable
    if "ipi_mask" in estimator_name and "_tuned" in estimator_name:
        pattern_part = estimator_name.replace("ipi_mask_", "").replace("_tuned", "")
        try:
            return f"IPI pattern {int(pattern_part) + 1}"
        except ValueError:
            return f"IPI pattern {pattern_part}"
    if estimator_name == "ipi_untuned" or estimator_name == "ipi_tuned":
        return "IPI"
    if estimator_name == "ipi_untuned_em" or estimator_name == "ipi_tuned_em":
        return "IPI (EM)"
    if estimator_name == "ipi_untuned_mf" or estimator_name == "ipi_tuned_mf":
        return "IPI (MissForest)"
    if estimator_name == "ipi_untuned_mean" or estimator_name == "ipi_tuned_mean":
        return "IPI (mean)"
    if estimator_name == "ipi_untuned_zero" or estimator_name == "ipi_tuned_zero":
        return "IPI (zero)"
    if estimator_name == "ipi_untuned_hotdeck" or estimator_name == "ipi_tuned_hotdeck":
        return "IPI (hotdeck)"
    if estimator_name == "cipi_untuned_em" or estimator_name == "cipi_tuned_em":
        return "CIPI (EM)"
    if estimator_name == "classical":
        return "Complete case"

    ## aipw baseline
    if estimator_name == "aipw_baseline":
        return "AIPW baseline"

    ## naive baselines
    if estimator_name == "naive_em_ci":
        return "Naive EM"
    if estimator_name == "naive_mf_ci":
        return "Single Imputation (MissForest)"
    if estimator_name == "naive_inplace_em_ci":
        return "Single Imputation (EM)"
    if estimator_name == "naive_inplace_mf_ci":
        return "Naive inplace MissForest"
    if estimator_name == "naive_hotdeck_ci":
        return "Naive Hotdeck"
    if estimator_name == "naive_inplace_hotdeck_ci":
        return "Single Imputation (Hotdeck)"
    if estimator_name == "naive_zero_ci":
        return "Naive zero"
    if estimator_name == "naive_inplace_zero_ci":
        return "Single Imputation (zero)"
    if estimator_name == "naive_mean_ci":
        return "Naive mean"
    if estimator_name == "naive_inplace_mean_ci":
        return "Single Imputation (mean)"

    # Fallback for names that might just have _tuned or _untuned suffix
    if estimator_name.endswith("_untuned"):
        return estimator_name.replace("_untuned", " (untuned)")
    if estimator_name.endswith("_tuned"):
        return estimator_name.replace("_tuned", " (tuned)")

    return estimator_name  # Default if no specific mapping


# Identify the best performing IPI pattern from df_untuned
def find_best_ipi_pattern(
    df: pl.DataFrame,
    metric: str = "n_eff",
) -> str:
    best_ipi_pattern_name = None
    if metric in df.columns:
        ipi_patterns_df = df.filter(
            pl.col("base_estimator").str.starts_with("IPI pattern")
        )
        if not ipi_patterns_df.is_empty():
            mean_n_eff_per_ipi_pattern = (
                ipi_patterns_df.group_by("base_estimator")
                .agg(pl.mean(metric).alias(f"mean_{metric}"))
                .sort(f"mean_{metric}", descending=True)
            )
            # pdb.set_trace()
            if (
                not mean_n_eff_per_ipi_pattern.is_empty()
                and mean_n_eff_per_ipi_pattern.row(0, named=True)[f"mean_{metric}"]
                is not None
            ):
                best_ipi_pattern_name = mean_n_eff_per_ipi_pattern.row(0, named=True)[
                    "base_estimator"
                ]
                print(
                    f"Best performing IPI pattern from untuned data: {best_ipi_pattern_name}"
                )
            else:
                print(
                    "Could not determine best IPI pattern from untuned data (empty or null mean_n_eff)."
                )
        else:
            print("No IPI patterns found in untuned data to determine the best one.")
    else:  # This else corresponds to `if metric in df.columns:`
        print(
            f"Metric column '{metric}' not found in DataFrame for finding best IPI pattern."
        )
    return best_ipi_pattern_name


# Helper function to report average metrics
def report_average_metrics(
    df: pl.DataFrame,
    ratio_col_name: str,
    eff_sample_col_name: str,
    coverage_col_name: str,  # df_name: str
    num_mask_col_name: str,
    num_fully_observed_col_name: str,
    nmcar_shift_magnitude_col_name: str,
) -> pl.DataFrame:
    """
    Report average metrics for a given dataframe.

    Args:
        df (pl.DataFrame): The dataframe containing the metrics.
        ratio (float): The ratio of partially observed to fully observed samples.
        eff_sample (str): The name of the effective sample size column.
        coverage (str): The name of the coverage column.
        num_mask (int): The number of masks.
        num_fully_observed (int): The number of fully observed samples.

    Returns:
        pl.DataFrame: The dataframe with the average metrics.
    """
    # print(f"\n--- Average Metrics for {df_name} ---")
    if df.is_empty():
        print("DataFrame is empty, cannot report metrics.")
        return None

    # Calculate average n_eff and coverage, grouped by base_estimator and N_to_n0_factor
    # Ensure 'coverage' and 'n_eff' are present and numeric
    required_cols = [
        "base_estimator",
        ratio_col_name,
        coverage_col_name,
        eff_sample_col_name,
        num_mask_col_name,
        num_fully_observed_col_name,
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing required columns: {missing_cols}. Cannot report metrics.")
        return None

    # Attempt to cast to float, handling potential errors if already float or not castable
    df = df.with_columns(
        pl.col(coverage_col_name).cast(pl.Float64, strict=False),
        pl.col(eff_sample_col_name).cast(pl.Float64, strict=False),
    )

    # Calculate number of trials per base_estimator and ratio combination beforehand
    trial_counts = df.group_by(
        [
            "base_estimator",
            ratio_col_name,
            num_mask_col_name,
            num_fully_observed_col_name,
            nmcar_shift_magnitude_col_name,
        ]
    ).agg(
        pl.count(eff_sample_col_name).alias("count_n_eff"),
        pl.count(coverage_col_name).alias("count_coverage"),
    )

    # Check that trial counts are consistent between eff_sample and coverage for each combination
    inconsistent_counts = trial_counts.filter(
        pl.col("count_n_eff") != pl.col("count_coverage")
    )

    if not inconsistent_counts.is_empty():
        print("Warning: Inconsistent trial counts between eff_sample and coverage:")
        print(inconsistent_counts)

    # Check that trial counts are the same within each base_estimator and ratio combination
    # (this should always be true given our grouping, but good to verify)
    trial_count_check = (
        trial_counts.group_by(
            [
                "base_estimator",
                ratio_col_name,
                num_mask_col_name,
                num_fully_observed_col_name,
                nmcar_shift_magnitude_col_name,
            ]
        )
        .agg(pl.col("count_n_eff").n_unique().alias("unique_counts_eff"))
        .filter(pl.col("unique_counts_eff") > 1)
    )

    if not trial_count_check.is_empty():
        raise ValueError(
            "Warning: Multiple different trial counts found for same base_estimator and ratio:"
        )
        print(trial_count_check)

    avg_metrics = (
        df.group_by(
            [
                "base_estimator",
                ratio_col_name,
                num_mask_col_name,
                num_fully_observed_col_name,
                nmcar_shift_magnitude_col_name,
            ]
        )
        .agg(
            pl.mean(eff_sample_col_name).alias("avg_n_eff"),
            pl.mean(coverage_col_name).alias("avg_coverage"),
            pl.std(eff_sample_col_name).alias("std_n_eff"),
            pl.std(coverage_col_name).alias("std_coverage"),
            # Quantiles for n_eff and coverage
            pl.col(eff_sample_col_name).quantile(0.95).alias("q95_n_eff"),
            pl.col(eff_sample_col_name).quantile(0.05).alias("q05_n_eff"),
            pl.col(coverage_col_name).quantile(0.95).alias("q95_coverage"),
            pl.col(coverage_col_name).quantile(0.05).alias("q05_coverage"),
            pl.count(eff_sample_col_name).alias("count_n_eff"),
        )
        .join(
            trial_counts.select(
                [
                    "base_estimator",
                    ratio_col_name,
                    num_mask_col_name,
                    num_fully_observed_col_name,
                    nmcar_shift_magnitude_col_name,
                    "count_n_eff",
                ]
            ),
            on=[
                "base_estimator",
                ratio_col_name,
                num_mask_col_name,
                num_fully_observed_col_name,
                nmcar_shift_magnitude_col_name,
            ],
            how="left",
        )
        .with_columns(
            # Calculate standard error using pre-calculated trial counts (standard deviation / sqrt(n_trials))
            (pl.col("std_n_eff") / pl.col("count_n_eff").sqrt()).alias("se_n_eff"),
            (pl.col("std_coverage") / pl.col("count_n_eff").sqrt()).alias(
                "se_coverage"
            ),
            # # Verify that count_n_eff matches our pre-calculated n_trials_eff
            # (pl.col("count_n_eff") == pl.col("count_n_eff")).alias(
            #     "count_verification"
            # ),
        )
        .sort(
            [
                "base_estimator",
                ratio_col_name,
                num_mask_col_name,
                num_fully_observed_col_name,
                nmcar_shift_magnitude_col_name,
            ]
        )
    )

    ## TODO: add this back for checking
    # # Check if there are any mismatches in our count verification
    # count_mismatches = avg_metrics.filter(pl.col("count_n_eff").not_())
    # if not count_mismatches.is_empty():
    #     print(
    #         "Warning: Mismatch between count_n_eff and pre-calculated num_trials_eff:"
    #     )
    #     print(
    #         count_mismatches.select(
    #             [
    #                 "base_estimator",
    #                 ratio,
    #                 num_mask,
    #                 num_fully_observed,
    #                 "count_n_eff",
    #             ]
    #         )
    #     )

    # Remove verification columns for final output
    avg_metrics = avg_metrics.drop(
        ["count_n_eff"]  # ,"count_verification",  "count_coverage"]
    )

    with pl.Config(
        tbl_rows=-1, tbl_width_chars=120
    ):  # Ensure all rows and wider columns are printed
        print(avg_metrics)
    return avg_metrics


# %%
def generate_composite_facet_plot(
    avg_untuned_data: pl.DataFrame,
    avg_tuned_data: pl.DataFrame,
    estimator_order: list[str],
    file_path: str,
    x_axis_label: str = "Partially observed to fully observed sample ratio",
    x_by_feature: str = "N_to_n0_ratio",
    color_values: Mapping[str, str] | list[str] | None = None,
    legend_title: str = "CI Method",
    legend_ncol: int | None = None,
    legend_marker_size: float | None = None,
    hide_n_eff_for_baselines: bool = False,
    shape_values: Mapping[str, str] | list[str] | None = None,
    point_size: float = 6.0,
) -> ggplot:
    """Generates and saves a 2x2 composite facet plot.
    Args:
        avg_untuned_data (pl.DataFrame): The dataframe containing the average metrics for the untuned data.
        avg_tuned_data (pl.DataFrame): The dataframe containing the average metrics for the tuned data.
        estimator_order (list[str]): The order of the estimators to plot.
        file_path (str): The path to save the plot.
        x_by_feature (str): The feature to plot on the x-axis.
        color_values: Optional mapping or list of exact colors for each
            `base_estimator`. If a mapping is provided, keys should be the
            display names (e.g., "Complete case"). If a list is provided, it
            will be applied in the order of `estimator_order`.
        shape_values: Optional mapping or list of exact shapes for each
            `base_estimator`. Accepts matplotlib marker strings (e.g., "o",
            "^", "s", "D", "v", "<", ">", "p", "*", "h", "+", "x"). If a
            mapping is provided, keys should be the display names. If a list is
            provided, it will be applied in the order of `estimator_order`.
        point_size: Size of the point markers in the plot.
        legend_title: Title to display for the legend.
        legend_ncol: Number of columns in the legend (for readability).
        legend_marker_size: Size override for legend markers.
    Returns:
        ggplot: The composite facet plot.
    """

    if x_by_feature not in avg_untuned_data.columns:
        raise ValueError(f"x_by_feature {x_by_feature} not found in avg_untuned_data")
    if x_by_feature not in avg_tuned_data.columns:
        raise ValueError(f"x_by_feature {x_by_feature} not found in avg_tuned_data")

    df_u = avg_untuned_data.with_columns(
        pl.lit("Pooled estimators").alias("tuning_type")
        # pl.lit("Pooled estimators").alias("tuning_type")
    )
    df_t = avg_tuned_data.with_columns(
        pl.lit("Power-tuned estimators").alias("tuning_type")
        # pl.lit("Power-tuned estimators").alias("tuning_type")
    )

    # Melt both the means and standard errors
    df_u_melted = df_u.melt(
        id_vars=["base_estimator", x_by_feature, "tuning_type"],
        value_vars=["avg_n_eff", "avg_coverage"],
        variable_name="metric_variable",
        value_name="metric_value",
    )
    df_u_se_melted = df_u.melt(
        id_vars=["base_estimator", x_by_feature, "tuning_type"],
        value_vars=["se_n_eff", "se_coverage"],
        variable_name="se_variable",
        value_name="se_value",
    )
    # Map se variable names to match metric variable names
    df_u_se_melted = df_u_se_melted.with_columns(
        pl.col("se_variable").str.replace("se_", "avg_").alias("metric_variable")
    )
    # Join the mean and se dataframes
    df_u_melted = df_u_melted.join(
        df_u_se_melted.select(
            [
                "base_estimator",
                x_by_feature,
                "tuning_type",
                "metric_variable",
                "se_value",
            ]
        ),
        on=["base_estimator", x_by_feature, "tuning_type", "metric_variable"],
        how="left",
    )

    df_t_melted = df_t.melt(
        id_vars=["base_estimator", x_by_feature, "tuning_type"],
        value_vars=["avg_n_eff", "avg_coverage"],
        variable_name="metric_variable",
        value_name="metric_value",
    )
    df_t_se_melted = df_t.melt(
        id_vars=["base_estimator", x_by_feature, "tuning_type"],
        value_vars=["se_n_eff", "se_coverage"],
        variable_name="se_variable",
        value_name="se_value",
    )
    # Map se variable names to match metric variable names
    df_t_se_melted = df_t_se_melted.with_columns(
        pl.col("se_variable").str.replace("se_", "avg_").alias("metric_variable")
    )
    # Join the mean and se dataframes
    df_t_melted = df_t_melted.join(
        df_t_se_melted.select(
            [
                "base_estimator",
                x_by_feature,
                "tuning_type",
                "metric_variable",
                "se_value",
            ]
        ),
        on=["base_estimator", x_by_feature, "tuning_type", "metric_variable"],
        how="left",
    )

    combined_df_pl = pl.concat([df_u_melted, df_t_melted])

    # Optionally hide Effective sample size for AIPW and Naive methods
    if hide_n_eff_for_baselines:
        combined_df_pl = combined_df_pl.filter(
            ~(
                (pl.col("metric_variable") == "avg_n_eff")
                & (
                    (pl.col("base_estimator") == "AIPW baseline")
                    | pl.col("base_estimator").str.contains("Single Imputation")
                )
            )
        )

    metric_mapping = {
        "avg_n_eff": "Effective sample size",
        "avg_coverage": "Average coverage",
    }
    combined_df_pl = combined_df_pl.with_columns(
        pl.col("metric_variable")
        .map_elements(lambda val: metric_mapping[val], return_dtype=pl.Utf8)
        .alias("metric_name")
    )

    combined_df_pd = combined_df_pl.to_pandas()

    # Coerce x-axis feature to numeric (handles int/float or numeric strings)
    combined_df_pd["x_numeric"] = pd.to_numeric(
        combined_df_pd[x_by_feature], errors="coerce"
    )
    # Drop rows that could not be coerced
    num_dropped = combined_df_pd["x_numeric"].isna().sum()
    if num_dropped > 0:
        print(
            f"Warning: Dropping {num_dropped} rows with non-numeric '{x_by_feature}' values."
        )
    combined_df_pd = combined_df_pd.dropna(subset=["x_numeric"])  # keep only numeric

    # Sort for proper line drawing order
    combined_df_pd = combined_df_pd.sort_values(
        by=["metric_name", "tuning_type", "base_estimator", "x_numeric"]
    )

    # Compute x-axis breaks and labels
    unique_x = sorted(combined_df_pd["x_numeric"].unique())
    x_tick_labels = [
        (str(int(x)) if float(x).is_integer() else f"{x:g}") for x in unique_x
    ]

    x_range = combined_df_pd["x_numeric"].max() - combined_df_pd["x_numeric"].min()

    # Calculate error bar limits
    combined_df_pd["ymin"] = (
        combined_df_pd["metric_value"] - norm.ppf(0.95) * combined_df_pd["se_value"]
    )
    combined_df_pd["ymax"] = (
        combined_df_pd["metric_value"] + norm.ppf(0.95) * combined_df_pd["se_value"]
    )

    # Order for hue (base_estimator)
    # Filter out None values from estimator_order to avoid issues
    valid_estimator_order = [
        est
        for est in estimator_order
        if est is not None and est in combined_df_pd["base_estimator"].unique()
    ]

    # If there are any null values in the actual data, handle them separately
    if combined_df_pd["base_estimator"].isnull().any():
        # Replace null values with a placeholder
        combined_df_pd["base_estimator"] = combined_df_pd["base_estimator"].fillna(
            "Missing_estimator"
        )
        if "Missing_estimator" not in valid_estimator_order:
            valid_estimator_order.append("Missing_estimator")

    combined_df_pd["base_estimator"] = pd.Categorical(
        combined_df_pd["base_estimator"],
        categories=valid_estimator_order,
        ordered=True,
    )

    # If we used placeholder, map it back for legend, or ensure plotnine handles it. For simplicity, ensure names are actual strings.
    # Assuming ipi_best_pattern_name, if None, means that category might not be plotted or is handled if 'None' string exists.
    # The original target_estimators_for_plot is best if all names are strings.
    # Reverting to simpler categorical conversion if find_best_ipi_pattern guarantees string or data doesn't have actual None estimators.
    current_plot_estimators = [
        e for e in estimator_order if e is not None
    ]  # pd.Categorical categories can't be None
    combined_df_pd["base_estimator"] = pd.Categorical(
        combined_df_pd["base_estimator"],
        categories=current_plot_estimators,  # Use the original list passed, which might include non-existent ones for consistent ordering
        ordered=True,
    )
    combined_df_pd.dropna(
        subset=["base_estimator"], inplace=True
    )  # Drop rows where base_estimator became NaN due to not being in categories

    # Order for facets
    combined_df_pd["metric_name"] = pd.Categorical(
        combined_df_pd["metric_name"],
        categories=["Effective sample size", "Average coverage"],
        ordered=True,
    )
    combined_df_pd["tuning_type"] = pd.Categorical(
        combined_df_pd["tuning_type"],
        categories=["Pooled estimators", "Power-tuned estimators"],
        ordered=True,
    )

    # Data for horizontal line in coverage plots
    # Ensure 'metric_name' in hline_data has same categories as in main df for proper faceting
    hline_data = pd.DataFrame(
        {
            "metric_name": pd.Categorical(
                ["Average coverage"],
                categories=combined_df_pd["metric_name"].cat.categories,
                ordered=True,
            ),
            "hline_val": [0.9],
        }
    )

    # Data for geom_blank to set y-limits for "Average coverage" facets
    blank_data_coverage = pd.DataFrame(
        {
            "metric_name": pd.Categorical(
                ["Average coverage"] * 2,
                categories=combined_df_pd["metric_name"].cat.categories,
                ordered=True,
            ),
            "metric_value": [0.75, 1.0],
        }
    )

    # Custom y-axis breaks: ensure a 0.9 tick for coverage facets only
    def _custom_y_breaks(limits: tuple[float, float]):
        lower, upper = limits
        auto = AutoLocator()
        base_ticks = auto.tick_values(lower, upper)
        if upper <= 1.25:
            return sorted(set([t for t in base_ticks if lower <= t <= upper] + [0.9]))
        # For effective sample size facets, reduce the number of ticks for readability
        eff_locator = MaxNLocator(nbins=5)
        eff_ticks = eff_locator.tick_values(lower, upper)
        return [t for t in eff_ticks if lower <= t <= upper]

    plot = (
        ggplot(
            combined_df_pd,
            aes(
                x="x_numeric",
                y="metric_value",
                color="base_estimator",
                group="base_estimator",
            ),
        )
        + geom_line(linetype="dashed", size=1.2, show_legend=False)
        + geom_point(aes(shape="base_estimator"), size=point_size)
        + geom_errorbar(
            aes(ymin="ymin", ymax="ymax", color="base_estimator"),
            width=x_range * 0.001,
            size=1,
            show_legend=False,
        )
        + facet_grid("metric_name ~ tuning_type", scales="free_y")
        + scale_x_continuous(breaks=unique_x, labels=x_tick_labels, expand=(0, 0))
        + scale_y_continuous(breaks=_custom_y_breaks)
        + geom_hline(
            data=hline_data,
            mapping=aes(yintercept="hline_val"),
            linetype="dotted",
            colour="grey",
            inherit_aes=False,
        )
        + geom_blank(
            data=blank_data_coverage, mapping=aes(y="metric_value"), inherit_aes=False
        )
        + theme_bw()
        + theme(
            figure_size=(12, 8),
            text=element_text(size=24),
            axis_text=element_text(size=24),
            axis_text_x=element_text(angle=45, hjust=1, size=24),
            strip_text=element_text(size=18),  # face="bold",
            strip_background=element_rect(
                fill="#E0E0E0", color="grey", size=0.5
            ),  # Light grey fill
            legend_position="bottom",
            legend_title=element_text(size=24),  # , face="bold"),
            legend_text=element_text(size=24),
        )
        + labs(x=x_axis_label, y="")
    )

    # Color scale: if exact colors are provided, use them, otherwise default to brewer
    if color_values is not None:
        # Build a list of colors aligned with the category order
        # Accept both mapping and list-like inputs
        if isinstance(color_values, Mapping):
            # Fallback palette for any missing keys
            fallback = sns.color_palette(
                "Dark2", n_colors=len(current_plot_estimators)
            ).as_hex()
            color_list = [
                color_values.get(est, fallback[i])
                for i, est in enumerate(current_plot_estimators)
            ]
        else:
            # List/tuple provided
            fallback = sns.color_palette(
                "Dark2", n_colors=len(current_plot_estimators)
            ).as_hex()
            color_list = list(color_values) + list(fallback)[len(color_values or []) :]
            color_list = color_list[: len(current_plot_estimators)]

        # Normalize any non-string colors (e.g., RGB tuples) to hex strings
        color_list = [c if isinstance(c, str) else to_hex(c) for c in color_list]

        plot = plot + scale_color_manual(
            values=color_list,
            limits=current_plot_estimators,
            name=legend_title,
        )
    else:
        plot = plot + scale_color_brewer(
            type="qual", palette="Dark2", name=legend_title
        )

    # Shape scale: if exact shapes are provided, use them; otherwise use defaults
    default_shapes = ["o", "^", "s", "D", "v", "<", ">", "p", "*", "h", "+", "x"]
    if shape_values is not None:
        if isinstance(shape_values, Mapping):
            # Build a list of shapes aligned with the category order
            # Fallback to cycling through default shapes for any missing keys
            shape_list = [
                shape_values.get(est, default_shapes[i % len(default_shapes)])
                for i, est in enumerate(current_plot_estimators)
            ]
        else:
            # List/tuple provided
            provided = list(shape_values)
            # Extend with defaults if needed and truncate to length
            needed = len(current_plot_estimators) - len(provided)
            if needed > 0:
                provided += [
                    default_shapes[i % len(default_shapes)] for i in range(needed)
                ]
            shape_list = provided[: len(current_plot_estimators)]
    else:
        # No shapes provided: assign defaults in order, cycling if necessary
        shape_list = [
            default_shapes[i % len(default_shapes)]
            for i in range(len(current_plot_estimators))
        ]

    plot = plot + scale_shape_manual(
        values=shape_list,
        limits=current_plot_estimators,
        name=legend_title,
    )

    # Legend readability: allow multi-column and larger markers
    guide_kwargs = {}
    if legend_ncol is not None:
        guide_kwargs["ncol"] = legend_ncol
    marker_size_for_legend = (
        legend_marker_size if legend_marker_size is not None else point_size
    )
    guide_kwargs["override_aes"] = {"size": marker_size_for_legend}
    plot = plot + guides(
        color=guide_legend(**guide_kwargs),
        shape=guide_legend(**guide_kwargs),
    )

    # Conditional y-limits and breaks (applied after main plot structure)
    # This part is tricky with facets in a single call.
    # For now, relying on scales='free_y'. If exact limits/breaks from image are crucial:
    # One might need to draw the plot and then iterate through g.axes to set them,
    # or construct the plot in a more complex way (e.g. combining 4 plots).
    # The example solution will rely on `scales='free_y'` for simplicity.

    plot_path = file_path
    plot.save(plot_path, dpi=300, transparent=True)
    print(f"Composite plot saved to: {plot_path}")
    return plot


# %%
def plot_missing_patterns(
    pattern_to_ids: Mapping[tuple[bool, ...], set[int]],
    column_names: list[str],
    png_dir: Path,
    png_filename: str,
    save_fig: bool = True,
) -> plt.Figure:
    binary_data = np.zeros((len(pattern_to_ids), len(column_names)))
    counts = np.zeros(len(pattern_to_ids))
    for i, pattern in enumerate(pattern_to_ids):
        binary_data[i, pattern] = 1
        counts[i] = len(pattern_to_ids[pattern])

    n, m = binary_data.shape

    # X boundaries for columns
    x = np.arange(m + 1)
    # Y boundaries for rows (cumulative counts for variable row height)
    y = np.concatenate(([0], np.cumsum(counts)))

    # Make the figure bigger: (width=10 inches, height=6 inches, for example)
    ig, ax = plt.subplots(figsize=(24, 20))

    # Two-color map for (not missing=1, missing=0)
    color1_rgb = sns.color_palette("Set2")[2]
    color2_rgb = sns.color_palette("Set2")[2]
    # Add alpha channel to the first color (e.g., 0.5 for 50% opacity)
    color1_rgba = (*color1_rgb, 0.1)  # Change 0.5 to desired opacity
    cmap = ListedColormap([color1_rgba, color2_rgb])

    bounds = [-0.5, 0.5, 1.5]

    bound_norm = BoundaryNorm(bounds, cmap.N)

    # Plot the "heatmap"
    pc = ax.pcolormesh(x, y, binary_data, cmap=cmap, norm=bound_norm, shading="auto")

    # Add thin horizontal lines between patterns
    for y_boundary in y[1:-1]:  # Exclude the outermost boundaries
        ax.axhline(y_boundary, color="lightgray", linewidth=2, linestyle="-")

    # Axis labels & title (bigger fonts)
    ax.set_xlabel("Features", fontsize=42)
    ax.set_ylabel("Samples", fontsize=42)

    # Feature names at column midpoints (with bigger font)
    ax.set_xticks(np.arange(m) + 0.5)
    ax.set_xticklabels(
        column_names,
        rotation=45,
        ha="right",
        fontsize=42,
    )

    # Create a discrete colorbar at the side
    # fraction: how much of the axis the colorbar should occupy
    # pad: space between the colorbar and main axes
    cbar = ig.colorbar(
        pc,
        ax=ax,
        ticks=[0, 1],
        orientation="horizontal",
        location="top",
        fraction=0.02,  # Adjusted fraction for horizontal bar
        pad=0.05,  # Adjusted pad for top location, title removed
        aspect=40,  # Make the colorbar thinner
    )
    cbar.ax.set_xticklabels(
        ["missing", "observed"], fontsize=60
    )  # Changed to set_xticklabels

    plt.tight_layout()
    if save_fig:
        plt.savefig(png_dir / png_filename)

    return plt


def display_results_table(
    avg_vals_untuned: pl.DataFrame, avg_vals_tuned: pl.DataFrame
) -> None:
    """
    Display table with values represented in the plot.

    Args:
        avg_vals_untuned (pl.DataFrame): The dataframe containing the average metrics for the untuned data.
        avg_vals_tuned (pl.DataFrame): The dataframe containing the average metrics for the tuned data.
    """
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
