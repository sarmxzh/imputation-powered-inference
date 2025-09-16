import click

from ipi.utils.diagnostic_expt_utils import run_diagnostic_expt


@click.command()
## data parameters
@click.option(
    "--ratio", "-r", default=10, type=int, help="Ratio for partially observed data"
)
@click.option("--init_seed", "-is", default=0, type=int, help="Initial trial seed")
@click.option("--num_trials", "-nt", default=1000, type=int, help="Number of trials")
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
@click.option(
    "--header_dir", "-hd", default="fm_ols_diagnostics", type=str, help="Header name"
)
def main(
    ratio: int,
    num_trials: int,
    num_fully_observed: int,
    num_features: int,
    num_factors: int,
    prob_missing: float,
    init_seed: int,
    header_dir: str,
) -> None:
    # currently setup to run single pattern shift expt
    num_masks = 10
    shift = 0.1
    shift_magnitudes_list = []
    for idx in range(num_masks):
        shift_magnitudes = [0.0] * num_masks
        shift_magnitudes[idx] = shift
        shift_magnitudes_list.append(shift_magnitudes)

    for shift_magnitudes in shift_magnitudes_list:
        run_diagnostic_expt(
            shift_magnitudes=shift_magnitudes,
            ratio=ratio,
            init_seed=init_seed,
            num_trials=num_trials,
            num_fully_observed=num_fully_observed,
            num_features=num_features,
            num_factors=num_factors,
            prob_missing=prob_missing,
            header_dir=header_dir,
            data_dir=None,
        )


if __name__ == "__main__":
    main()
    # uv run experiment_scripts/diagnostics_local.py
