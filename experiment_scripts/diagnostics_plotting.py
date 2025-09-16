# %%

# %%
import numpy as np
import pandas as pd
import polars as pl
from plotnine import (
    aes,
    element_rect,
    element_text,
    facet_wrap,
    geom_abline,
    geom_errorbar,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_manual,
    scale_x_continuous,
    theme,
    theme_bw,
)
from scipy.stats import norm, uniform

from ipi.config import RES_DIR

# %%
num_fully_observed = 200
num_masks = 10
num_trials = 1000
ratio = 10
header_dir = "fm_ols_diagnostics"
init_seed = 0
prob_missing = 0.2
num_features = 20
num_factors = 2

num_regr_features = 3

expt_name = "single pattern shift"

ratios = [10]
indices = [*list(range(10))]
hyperparams = indices
shift_magnitudes_list = []
for idx in hyperparams:
    listed_shift = [0.0] * 10
    listed_shift[idx] = 0.1
    shift_magnitudes_list.append(listed_shift)
# %%
loaded_data_list = []
for _, shift_magnitudes in enumerate(shift_magnitudes_list):
    ratio = 10
    if shift_magnitudes is not None:
        num_masks = len(shift_magnitudes)
        shift_magnitudes_str = "_".join([str(magn) for magn in shift_magnitudes])
        experiment_name = f"sm{shift_magnitudes_str}_n0_{num_fully_observed}_ratio{ratio}_d{num_features}_q{num_factors}_p{prob_missing}_{num_masks}masks_Qest4_numtrials{num_trials}_initseed{init_seed}"  # f"nsm{shift_magnitudes}_n0_{num_fully_observed}_ratio{ratio}_d{num_features}_q{num_factors}_p{prob_missing}_{num_masks}masks_Qest4_numtrials{num_trials}_initseed{init_seed}"
    else:
        num_masks = 10
        experiment_name = f"n0_{num_fully_observed}_ratio{ratio}_d{num_features}_q{num_factors}_p{prob_missing}_{num_masks}masks_Qest4_numtrials{num_trials}_initseed{init_seed}"

    print(experiment_name)
    loaded_data = pl.read_parquet(
        RES_DIR / "factor_model_ols" / header_dir / experiment_name / "results.parquet"
    )
    loaded_data_list.append(loaded_data)

# %%
# QQ plot for the p-values wrt uniform(0,1) using plotnine (ggplot)
tests = ["test1", "test2"]

# Build a tidy DataFrame of theoretical and empirical quantiles for each test and num_patterns
plot_rows = []
for i, loaded_data in enumerate(loaded_data_list):
    for test in tests:
        pvalues = loaded_data[f"{test}_pvalue"].to_numpy()
        pvalues_sorted = np.sort(pvalues)
        n = len(pvalues_sorted)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical_q = uniform.ppf(probs, loc=0, scale=1)
        for tq, ev in zip(theoretical_q, pvalues_sorted, strict=False):
            # numeric for continuous color, separate label for discrete shapes
            num_val = -1 if hyperparams[i] is None else int(hyperparams[i])
            num_lab = "None" if hyperparams[i] is None else str(hyperparams[i])
            plot_rows.append(
                {
                    "test": test,
                    "theoretical_q": tq,
                    "empirical_q": ev,
                    "num_patterns": num_val,
                    "num_patterns_label": num_lab,
                }
            )

plot_df = pd.DataFrame(plot_rows)

# %%

# Build a dark, high-contrast 12-color palette (Dark2-inspired)
dark_colors_12 = [
    "#1B9E77",  # teal (Dark2)
    "#D95F02",  # orange (Dark2)
    "#7570B3",  # purple (Dark2)
    "#E7298A",  # magenta (Dark2)
    "#66A61E",  # green (Dark2)
    "#E6AB02",  # mustard (Dark2)
    "#A6761D",  # brown (Dark2)
    "#666666",  # gray (Dark2)
    "#1f77b4",  # blue (tab10)
    "#d62728",  # red (tab10)
    "#17becf",  # cyan (tab10)
    "#2ca02c",  # green (tab10)
]

# Map labels to colors to ensure stable, explicit assignment
labels = sorted(
    plot_df["num_patterns_label"].unique(),
    key=lambda x: (-1 if x == "None" else x),
)
label_to_color = {
    lab: dark_colors_12[i % len(dark_colors_12)] for i, lab in enumerate(labels)
}

qq_plot = (
    ggplot(
        plot_df,
        aes(
            x="theoretical_q",
            y="empirical_q",
            color="num_patterns_label",
            shape="num_patterns_label",
        ),
    )
    + geom_point(alpha=0.7, size=0.7, stroke=0.4)
    + geom_abline(intercept=0, slope=1, linetype="dashed", color="red", alpha=0.7)
    + facet_wrap("~test", nrow=1)
    + scale_color_manual(values=label_to_color)
    + guides(
        color=guide_legend(override_aes={"size": 4.0, "alpha": 0.9}),
        shape=guide_legend(override_aes={"size": 4.0, "alpha": 0.9}),
    )
    + labs(
        title=f"QQ Plot of p-values under {expt_name}",
        x="Theoretical Quantiles",
        y="Empirical Quantiles",
        color="num_patterns",
        shape="num_patterns",
    )
    + theme_bw()
    + theme(
        figure_size=(8, 4),
        legend_key_size=12,
        legend_title=element_text(size=12, weight="bold"),
        legend_text=element_text(size=11),
        legend_key=element_rect(fill="#f7f7f7", color="#bdbdbd"),
        legend_background=element_rect(fill="white", alpha=0.8),
    )
)

qq_plot.save(
    RES_DIR / "factor_model_ols" / header_dir / f"qq_plot_{expt_name}.png",
    dpi=300,
    transparent=True,
)

# %%

# Compute and plot average coverage for IPI tuned/untuned vs num_val
coverage_rows = []
for i, loaded_data in enumerate(loaded_data_list):
    num_val = -1 if hyperparams[i] is None else (hyperparams[i])
    num_lab = "None" if hyperparams[i] is None else str(hyperparams[i])
    print(loaded_data.shape)

    tmp = (
        loaded_data.select(pl.col("estimator"), pl.col("coverage").cast(pl.Float64))
        .group_by("estimator")
        .agg(
            pl.col("coverage").mean().alias("avg_coverage"),
            pl.col("coverage").std().alias("std_coverage"),
            pl.count("coverage").alias("n_trials"),
        )
    )
    print(tmp)
    print(num_val)

    print(loaded_data["coverage"].mean())

    for row in tmp.iter_rows(named=True):
        est = row["estimator"]
        avg_cov = (
            float(row["avg_coverage"])
            if row["avg_coverage"] is not None
            else float("nan")
        )
        std_cov = row["std_coverage"]
        n_trials = int(row["n_trials"]) if row["n_trials"] is not None else 0
        se_cov = (
            float(std_cov) / np.sqrt(n_trials)
            if std_cov is not None and n_trials > 0
            else float("nan")
        )
        coverage_rows.append(
            {
                "num_patterns": num_val,
                "num_patterns_label": num_lab,
                "estimator": est,
                "avg_coverage": avg_cov,
                "se_coverage": se_cov,
            }
        )

cov_df = pd.DataFrame(coverage_rows)

# Keep only IPI tuned/untuned if present and compute error bar bounds
if not cov_df.empty:
    ipi_mask = cov_df["estimator"].isin(["ipi_tuned_em", "ipi_untuned_em"])
    cov_df = cov_df[ipi_mask] if ipi_mask.any() else cov_df
    cov_df["ymin"] = cov_df["avg_coverage"] - cov_df["se_coverage"] * norm.ppf(0.95)
    cov_df["ymax"] = cov_df["avg_coverage"] + cov_df["se_coverage"] * norm.ppf(0.95)

# Derive x-axis breaks to show ticks for all observed x values
x_breaks = sorted(cov_df["num_patterns"].unique()) if not cov_df.empty else None
coverage_plot = (
    ggplot(
        cov_df,
        aes(
            x="num_patterns",
            y="avg_coverage",
            color="estimator",
            shape="estimator",
            group="estimator",
        ),
    )
    + geom_errorbar(aes(ymin="ymin", ymax="ymax"), size=1.2, width=0.001, alpha=0.7)
    + geom_point(size=3, alpha=0.9, shape="s")
    + geom_line(alpha=0.8, linetype="dotted", size=1.2)
    + geom_hline(yintercept=0.90, linetype="dotted", color="black", alpha=0.6)
    + scale_color_manual(
        values={"ipi_tuned_em": "#1f77b4", "ipi_untuned_em": "#d62728"}
    )
    + scale_x_continuous(breaks=x_breaks)
    + labs(
        title="Average coverage by hyperparameter (num_val)",
        x="num_val (index of shifted pattern; -1 means None)",
        y="Average coverage",
        color="Estimator",
        shape="Estimator",
    )
    + theme_bw()
    + theme(
        figure_size=(8, 4),
        legend_key_size=12,
        legend_title=element_text(size=12, weight="bold"),
        legend_text=element_text(size=11),
        legend_key=element_rect(fill="#f7f7f7", color="#bdbdbd"),
        legend_background=element_rect(fill="white", alpha=0.8),
    )
)

coverage_plot.save(
    RES_DIR / "factor_model_ols" / header_dir / f"coverage_plot_{expt_name}.png",
    dpi=300,
    transparent=True,
)

# %%
