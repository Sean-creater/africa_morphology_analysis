"""
06_regional_effect_by_distance.py
==================================
Q4 — Experiment: Connecting Part 2 regional context to intra-urban position.

For each radial bin, partial R^2 of climate zone, colonial legacy, and
log(GDP per capita) on the 17-type composition PCoA space, net of city
population. See _regional_effect_shared.py for the full model
specification; this script and 10_figure_main.py's Panel D both call the
same `regional_effect_by_bin()` function, so the reported partial R^2
values are identical between the table and the figure.

IMPORTANT
---------
External predictors must be EXOGENOUS — do NOT use any morphology-derived
variable (community label, H17, r17). Only climate / economic / historical
/ population.

Outputs
-------
  regional_effect_profile.csv
      radial_bin | n_cities | n_countries | n_pcoa_axes |
      pcoa_variance_explained | design_rank_full |
      n_boot_requested | n_boot_valid |
      partial_r2_climate | partial_r2_colonial | partial_r2_gdp |
      partial_r2_population | r2_full  (+ 95% CI columns per block)
"""

import logging
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _regional_effect_shared import regional_effect_by_bin, DEFAULT_N_BOOT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).parent
PART3_DIR   = ROOT.parent.parent / "03_urban_hierarchy" / "outputs" / "tables"
AI_PATH     = ROOT.parent.parent / "02_regional_structure" / "regression" / "outputs" / "tables" / "analysis_input.csv"
OUT_TABLES  = ROOT / "outputs" / "tables"

PROFILES_PQ = OUT_TABLES / "city_bin_profiles.parquet"

N_TYPES    = 17
TYPE_COLS  = [f"p_{k:02d}" for k in range(1, N_TYPES + 1)]


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    log.info("Loading city_bin_profiles.parquet ...")
    prof = pd.read_parquet(PROFILES_PQ)

    if not AI_PATH.exists():
        log.warning(f"{AI_PATH} not found — skipping regional effect analysis.")
        pd.DataFrame(columns=[
            "radial_bin", "n_cities", "n_countries", "n_pcoa_axes",
            "pcoa_variance_explained", "design_rank_full",
            "n_boot_requested", "n_boot_valid",
            "partial_r2_climate", "partial_r2_colonial",
            "partial_r2_gdp", "partial_r2_population", "r2_full",
        ]).to_csv(OUT_TABLES / "regional_effect_profile.csv", index=False)
        log.info("Saved: regional_effect_profile.csv (empty — analysis_input.csv not available)")
        return

    covariates = pd.read_csv(
        AI_PATH,
        usecols=["city", "country", "climate_zone", "colonial", "log_gdp",
                 "log_pop_worldpop_2022"],
    ).rename(columns={"log_pop_worldpop_2022": "log_pop"})

    log.info(f"Running partial R^2 analysis ({DEFAULT_N_BOOT} bootstrap resamples per bin) ...")
    result = regional_effect_by_bin(
        profiles=prof,
        covariates=covariates,
        composition_cols=TYPE_COLS,
    )

    result.to_csv(OUT_TABLES / "regional_effect_profile.csv", index=False, float_format="%.4f")
    log.info("Saved: regional_effect_profile.csv")

    print("\n=== Regional effect by radial position (partial R^2) ===")
    print(result.to_string(index=False))

    log.info("Done.")


if __name__ == "__main__":
    main()
