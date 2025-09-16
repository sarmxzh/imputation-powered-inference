# %%
import click

from ipi.config import LIST_OF_AVAIL_METHODS
from ipi.utils.factor_model_utils import run_fm_expt


# %%
@click.command()
## data parameters
@click.option("--num_trials", "-nt", default=100, type=int, help="Number of trials")
@click.option(
    "--num_fully_observed",
    "-n",
    default=200,
    type=int,
    help="Number of fully observed data points",
)
@click.option("--num_features", "-d", default=20, type=int, help="Number of features")
@click.option("--num_factors", "-q", default=2, type=int, help="Number of factors")
@click.option(
    "--prob_missing", "-p", default=0.2, type=float, help="Probability of missingness"
)
@click.option("--num_masks", "-R", default=10, type=int, help="Number of masks")
@click.option("--init_seed", "-is", default=0, type=int, help="Initial trial seed")
@click.option(
    "--methods",
    "-m",
    default=[
        "classical",
        "ipi_untuned_em",
        "ipi_tuned_em",
        "ipi_per_mask_untuned_em",
        "ipi_per_mask_tuned_em",
        "cipi_untuned_em",
        "cipi_tuned_em",
    ],
    type=click.Choice(LIST_OF_AVAIL_METHODS),
    multiple=True,
    help="Confidence interval methods to run",
)
@click.option("--header_name", "-hn", default="test123", type=str, help="Header name")
@click.option(
    "--ratio", "-r", default=10, type=int, help="Ratio for partially observed data"
)
@click.option(
    "--nmcar_shift_magnitude",
    "-nsm",
    default=None,
    type=float,
    help="NMCAR shift magnitude",
)
@click.option(
    "--mask_seed",
    "-ms",
    default=110,
    type=int,
    help="Mask seed",
)
def main(
    num_trials: int,
    num_fully_observed: int,
    prob_missing: float,
    num_features: int,
    num_factors: int,
    num_masks: int,
    init_seed: int,
    ratio: int,
    methods: list[str],
    header_name: str,
    nmcar_shift_magnitude: float | None,
    mask_seed: int,
) -> None:
    run_fm_expt(
        init_seed=init_seed,
        num_trials=num_trials,
        num_features=num_features,
        num_masks=num_masks,
        num_fully_observed=num_fully_observed,
        prob_missing=prob_missing,
        ratio=ratio,
        num_factors=num_factors,
        header_name=header_name,
        methods=methods,
        nmcar_shift_magnitude=nmcar_shift_magnitude,  # None default
        data_dir=None,  # data_dir default None -> DATA_DIR / "factor_model"
        mask_seed=mask_seed,
    )


if __name__ == "__main__":
    main()

## example command line run:
# uv run experiment_scripts/factor_model_expt.py

# %%
