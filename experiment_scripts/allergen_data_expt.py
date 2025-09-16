## wrapper for allergen data experiments
import logging

import click
from rich.logging import RichHandler

from ipi.config import INTERCEPT_COL
from ipi.utils.allergen_data_utils import filter_dataset_for_expt, run_allergen_expt

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--outcomes",
    "-o",
    default=("Ara h 1", "Ara h 2", "Ara h 3", "Ara h 6", "Ara h 8"),
    type=click.Choice(["Ara h 1", "Ara h 2", "Ara h 3", "Ara h 6", "Ara h 8"]),
    multiple=True,
    help="Outcomes",
)
@click.option(
    "--regr_features",
    "-rf",
    default=(INTERCEPT_COL, "Age", "Sexe", "ARIA (rhinite)"),
    type=str,
    multiple=True,
    help="Regression features",
)
@click.option(
    "--feature_of_inference",
    "-foi",
    default="ARIA (rhinite)",
    type=str,
    help="Feature of inference",
)
@click.option(
    "--split_seed",
    "-s",
    default=0,
    type=int,
    help="Train test split seed",
)
@click.option("--num_folds", "-nf", default=10, type=int, help="Number of folds")
@click.option(
    "--num_bootstrap_trials",
    "-nb",
    default=50,
    type=int,
    help="Number of bootstrap trials",
)
@click.option("--train_percent", "-tp", default=0.1, type=float, help="Train percent")
@click.option(
    "--header_dir",
    "-hd",
    default="rhinitis_control_age_sex",
    type=str,
    help="Header directory",
)
@click.option("--data_dir", "-dd", default=None, type=str, help="Data directory")
@click.option("--track_cipi", "-tc", default=False, type=bool, help="Track CIPI")
def main(
    outcomes: list[str],
    regr_features: list[str],
    feature_of_inference: str,
    split_seed: int,
    num_folds: int,
    num_bootstrap_trials: int,
    train_percent: float,
    header_dir: str | None,
    data_dir: str | None,
    track_cipi: bool,
) -> None:
    regr_features_list = list(regr_features)
    for outcome in outcomes:
        path = run_allergen_expt(
            outcome=outcome,
            regr_features=regr_features_list,
            feature_of_inference=feature_of_inference,
            train_test_split_seed=split_seed,
            num_folds=num_folds,
            num_bootstrap_trials=num_bootstrap_trials,
            train_percent=train_percent,
            header_dir=header_dir,
            data_dir=data_dir,
            track_cipi=track_cipi,
        )
        logger.info(f"Wrote results to: {path}")


@click.command()
@click.option(
    "--outcome",
    "-o",
    default="Ara h 1",
    type=click.Choice(["Ara h 1", "Ara h 2", "Ara h 3", "Ara h 6", "Ara h 8"]),
    multiple=False,
    help="Outcome",
)
@click.option(
    "--regr_features",
    "-rf",
    default=(INTERCEPT_COL, "Age", "Sexe", "ARIA (rhinite)"),
    type=str,
    multiple=True,
    help="Regression features",
)
@click.option(
    "--feature_of_inference",
    "-foi",
    default="ARIA (rhinite)",
    type=str,
    help="Feature of inference",
)
@click.option("--data_dir", "-dd", default=None, type=str, help="Data directory")
def run_allergen_expt_plot_missing_patterns(
    outcome: str,
    regr_features: list[str],
    feature_of_inference: str,
    data_dir: str | None,
) -> None:
    """
    Quick function for plotting missing patterns for allergen data.
    """
    filter_dataset_for_expt(
        outcome=outcome,
        regr_features=regr_features,
        feature_of_inference=feature_of_inference,
        data_dir=data_dir,
        do_plot_missing_patterns=True,
    )


# %%
if __name__ == "__main__":
    main()


## example command line runs:
# uv run experiment_scripts/allergen_data_expt.py
# uv run experiment_scripts/allergen_data_expt.py -rf "INTERCEPT" -rf "Age" -rf "Sexe" -rf "GINA (ancien)" -foi "GINA (ancien)" -hd "asthma_control_age_sex"
