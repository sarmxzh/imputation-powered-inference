## wrapper class for census survey experiment local runs

import click

from ipi.config import LIST_OF_AVAIL_METHODS
from ipi.utils.census_survey_utils import run_census_survey_expt


@click.command()
@click.option("--init_seed", "-is", default=0, type=int, help="Initial seed")
@click.option(
    "--num_fully_observed",
    "-n",
    default=2000,
    type=int,
    help="Number of fully observed data points",
)
@click.option(
    "--ratio", "-r", default=10, type=int, help="Ratio for partially observed data"
)
@click.option("--num_masks", "-R", default=10, type=int, help="Number of masks")
@click.option(
    "--total_num_trials", "-nt", default=100, type=int, help="Total number of trials"
)
@click.option(
    "--methods",
    "-m",
    default=[
        "classical",
        "ipi_untuned_mf",
        "ipi_tuned_mf",
        "ipi_per_mask_untuned_mf",
        "ipi_per_mask_tuned_mf",
    ],
    type=click.Choice(LIST_OF_AVAIL_METHODS),
    multiple=True,
    help="Confidence interval methods to run",
)
@click.option(
    "--header_name", "-hn", default="ipi_w_schooling", type=str, help="Header name"
)
@click.option("--data_dir", "-dd", default=None, type=str, help="Data directory")
def main(
    init_seed: int,
    num_fully_observed: int,
    ratio: int,
    num_masks: int,
    total_num_trials: int,
    methods: list[str],
    header_name: str,
    data_dir: str | None,
) -> None:
    run_census_survey_expt(
        init_seed=init_seed,
        num_fully_observed=num_fully_observed,
        ratio=ratio,
        num_masks=num_masks,
        num_trials=total_num_trials,
        methods=methods,
        header_name=header_name,
        data_dir=data_dir,
    )


if __name__ == "__main__":
    main()
