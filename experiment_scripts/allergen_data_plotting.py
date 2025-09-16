# %%

import json

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_rect,
    element_text,
    geom_linerange,
    geom_point,
    ggplot,
    labs,
    position_dodge,
    scale_color_manual,
    scale_fill_manual,
    scale_shape_manual,
    scale_x_discrete,
    theme,
    theme_classic,
)

from ipi.config import RES_DIR

# %%

# %%
file_dir = RES_DIR / "allergen_data"
header_dir = "rhinitis_control_age_sex"
numfolds = 10
numbootstraptrials = 50
splitseed = 0
trainpercent = 0.1
featureofinference = "ARIA (rhinite)"
outcomes = ["Ara h 1", "Ara h 2", "Ara h 3", "Ara h 6", "Ara h 8"]
track_cipi = False

dict_list = []
for outcome in outcomes:
    file_location = (
        file_dir
        / header_dir
        / f"splitseed{splitseed}_numfolds{numfolds}_numbootstraptrials{numbootstraptrials}_trainpercent{trainpercent}"
    )
    file_location = file_location / "cipi" if track_cipi else file_location / "no_cipi"
    file_location = (
        file_location
        / f"outcome{outcome}_featureofinference{featureofinference}_results.json"
    )
    with open(file_location) as f:
        loaded_dict = json.load(f)
        print(loaded_dict)
        dict_list.append(loaded_dict)

# %%
# Prepare data for plotnine
plot_data_list = []
for loaded_dict in dict_list:
    # Classical CI
    classical_ci_bounds = loaded_dict["classical"]
    classical_mid = np.mean(classical_ci_bounds)
    plot_data_list.append(
        {
            "outcome": loaded_dict["outcome"],
            "ci_type": "Complete case",
            "mid": classical_mid,
            "lower": classical_ci_bounds[0],
            "upper": classical_ci_bounds[1],
        }
    )

    # IPI CI
    ipi_ci_bounds = loaded_dict["ipi_tuned"]
    ipi_mid = np.mean(ipi_ci_bounds)
    plot_data_list.append(
        {
            "outcome": loaded_dict["outcome"],
            "ci_type": "IPI (MissForest)",
            "mid": ipi_mid,
            "lower": ipi_ci_bounds[0],
            "upper": ipi_ci_bounds[1],
        }
    )

    # CIPI CI
    cipi_ci_bounds = None if not track_cipi else loaded_dict["cipi_tuned"]
    cipi_ci_bounds = (0, 0) if cipi_ci_bounds is None else cipi_ci_bounds
    cipi_mid = np.mean(cipi_ci_bounds)
    plot_data_list.append(
        {
            "outcome": loaded_dict["outcome"],
            "ci_type": "CIPI (MissForest)",
            "mid": cipi_mid,
            "lower": cipi_ci_bounds[0],
            "upper": cipi_ci_bounds[1],
        }
    )

plot_df = pd.DataFrame(plot_data_list)

# Ensure 'outcome' is categorical and ordered as in the 'outcomes' list
plot_df["outcome"] = pd.Categorical(
    plot_df["outcome"], categories=outcomes, ordered=True
)
# Ensure 'ci_type' is categorical and ordered for consistent legend and dodging
plot_df["ci_type"] = pd.Categorical(
    plot_df["ci_type"],
    categories=["Complete case", "IPI (MissForest)", "CIPI (MissForest)"],
    ordered=True,
)

# Define colors
plot_colors = {
    "Complete case": "#0072B2",
    "IPI (MissForest)": "#FD8D3C",
    "CIPI (MissForest)": "#EF3B2C",
}

plot_shapes = {
    "Complete case": "o",
    "IPI (MissForest)": "s",
    "CIPI (MissForest)": "s",
}

dodge_width = 0.3  # Adjust for desired spacing of dodged points/errorbars

# Create the plotnine plot
p = (
    ggplot(plot_df, aes(x="outcome", y="mid"))
    + geom_linerange(
        aes(ymin="lower", ymax="upper", color="ci_type"),
        size=10,  # User set this to 10
        alpha=0.8,  # Added alpha for lighter opacity
        position=position_dodge(width=dodge_width),
    )
    + geom_point(
        aes(y="mid", fill="ci_type", shape="ci_type"),
        color="black",
        stroke=0.75,
        size=5,
        position=position_dodge(width=dodge_width),
    )
    + scale_fill_manual(values=plot_colors, name="CI Type")
    + scale_color_manual(values=plot_colors, name="CI Type")
    + scale_shape_manual(values=plot_shapes, name="CI Type")
    + scale_x_discrete(name=None)
    + labs(
        title="Confidence Intervals by Method Controlling for Age and Sex",  # , and Region",  # , and Region)",
        y="CI for Regression Coefficient",  # Shortened y-axis label
    )
    + theme_classic()
    + theme(
        figure_size=(20, 10),
        plot_title=element_text(size=42, face="bold", margin={"b": 15}),  # Increased
        axis_title_y=element_text(size=42, margin={"r": 10}),  # Increased
        axis_text_x=element_text(size=42),  # Kept very large as per image
        axis_text_y=element_text(size=42),  # Increased
        axis_line_x=element_line(
            color="grey", size=0.75
        ),  # Increased axis line thickness
        axis_line_y=element_line(
            color="grey", size=0.75
        ),  # Increased axis line thickness
        legend_title=element_text(size=42),  # Increased
        legend_text=element_text(size=42),  # Increased
        legend_background=element_rect(fill="whitesmoke", color="lightgrey"),
        legend_position="top",
        legend_key_spacing_x=40,  # Add space between legend keys
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
    )
)
p
# %%

# Save and show the plotnine plot
plotnine_filename = "allergen_data_plotnine.png"
p.save(
    file_dir
    / header_dir
    / f"splitseed{splitseed}_numfolds{numfolds}_numbootstraptrials{numbootstraptrials}_trainpercent{trainpercent}"
    / plotnine_filename,
    dpi=300,
    bbox_inches="tight",
)
print(f"Plotnine figure saved to: {file_dir / header_dir / plotnine_filename}")

# %%
