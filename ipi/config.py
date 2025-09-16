# %%
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RES_DIR = DATA_DIR / "experimental_results"
LOG_DIR = ROOT_DIR / "logs"

INTERCEPT_COL = "INTERCEPT"

LIST_OF_AVAIL_METHODS = [
    "classical",
    "aipw_baseline",
    # naive methods
    "naive_em_ci",
    "naive_inplace_em_ci",
    "naive_hotdeck_ci",
    "naive_inplace_hotdeck_ci",
    "naive_mf_ci",
    "naive_inplace_mf_ci",
    "naive_mean_ci",
    "naive_inplace_mean_ci",
    "naive_zero_ci",
    "naive_inplace_zero_ci",
    # ipi methods
    "ipi_untuned_em",
    "ipi_untuned_mf",
    "ipi_untuned_mean",
    "ipi_untuned_zero",
    "ipi_untuned_hotdeck",
    "ipi_tuned_em",
    "ipi_tuned_mf",
    "ipi_tuned_mean",
    "ipi_tuned_zero",
    "ipi_tuned_hotdeck",
    # ipi per mask methods
    "ipi_per_mask_untuned_mf",
    "ipi_per_mask_tuned_mf",
    "ipi_per_mask_untuned_em",
    "ipi_per_mask_tuned_em",
    # cipi methods
    "cipi_untuned_em",
    "cipi_tuned_em",
]
