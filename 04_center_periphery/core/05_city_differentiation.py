"""
05_city_differentiation.py
==========================
Compute city-level centre-periphery differentiation metrics from ring profiles,
for THREE distance frameworks (abs / scl / rel).

Metrics (computed per distance framework)
------------------------------------------
CPC (Centre-Periphery Contrast):
    CPC = p2(inner_ring) - p2(outer_ring)
    Measures concentration of high-density compact morphology (Group 2) at core.
    CPC > 0: core denser than periphery. Range [-1, 1].

    Ring definitions per framework:
      rel: inner = ring 1, outer = ring 5  (within-city relative radius)
      abs: inner = ring 1 (0-1 km), outer = ring 4 (5-10 km)
           (ring 5 >10km only exists for ~8% cities; use ring 4 for consistency)
      scl: inner = ring 1, outer = ring 5  (population-scaled quintile bands)

D_IO (Inner-Outer Jensen-Shannon Divergence):
    D_IO = JSD(P_inner, P_outer)^2   where P = (p1, p2, p3, p4), base=2
    Total compositional dissimilarity between inner and outer ring. [0, 1].

QC flags
--------
qc_cpc_main   -- both inner and outer rings present and n_cells_ring >= 5
qc_cpc_strict -- both present and n_cells_ring >= 10 (conservative)

Output
------
    outputs/tables/city_radial_metrics.csv

    Columns (for each ring_type {rel, abs, scl}):
        CPC_{rt}, D_IO_{rt}
        qc_cpc_main_{rt}, qc_cpc_strict_{rt}
    Plus city covariates: city, country, community, log_pop, n_cells

Usage
-----
    python 05_city_differentiation.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "tables"

PROFILES_PATH = OUT_DIR / "radial_profiles.parquet"
CITY_METRICS  = (ROOT.parent.parent / "03_urban_hierarchy" /
                 "outputs" / "tables" / "city_metrics.parquet")
OUT_PATH      = OUT_DIR / "city_radial_metrics.csv"

# Inner / outer ring for each framework
# abs: inner=1 (0-1 km), outer=4 (5-10 km)  [ring 5 too sparse]
RING_PAIRS = {
    "rel": (1, 5),
    "abs": (1, 4),
    "scl": (1, 5),
}

MIN_CELLS_MAIN   = 5
MIN_CELLS_STRICT = 10


# ---------------------------------------------------------------------------
# JSD helper
# ---------------------------------------------------------------------------
def _jsd(p_in: np.ndarray, p_out: np.ndarray) -> float:
    """Jensen-Shannon divergence in [0, 1] (base 2, squared)."""
    p_in  = np.clip(p_in,  0.0, 1.0)
    p_out = np.clip(p_out, 0.0, 1.0)
    s_in  = p_in.sum()
    s_out = p_out.sum()
    if s_in  > 0: p_in  /= s_in
    if s_out > 0: p_out /= s_out
    return float(jensenshannon(p_in, p_out, base=2) ** 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    logging.info("Loading ring profiles (%s) ...", PROFILES_PATH.name)
    prof_all = pd.read_parquet(PROFILES_PATH)
    logging.info("  %d rows (city x ring_type x ring).", len(prof_all))

    city_meta = pd.read_parquet(CITY_METRICS,
                                columns=["city", "country", "community",
                                         "log_pop", "n_cells"])

    results_rt = []

    for rt, (r_inner, r_outer) in RING_PAIRS.items():
        logging.info("  [%s] inner=ring%d  outer=ring%d", rt, r_inner, r_outer)
        prof = prof_all[prof_all["ring_type"] == rt].copy()

        # Extract inner and outer ring rows
        inner = (prof[prof["ring"] == r_inner]
                 .set_index(["city", "country"])
                 [["p1", "p2", "p3", "p4", "n_cells_ring", "qc_comp", "qc_div"]].copy())
        outer = (prof[prof["ring"] == r_outer]
                 .set_index(["city", "country"])
                 [["p1", "p2", "p3", "p4", "n_cells_ring", "qc_comp"]].copy())

        inner.columns = [f"{c}_in"  for c in inner.columns]
        outer.columns = [f"{c}_out" for c in outer.columns]

        both = inner.join(outer, how="inner").reset_index()

        # CPC
        both[f"CPC_{rt}"] = both["p2_in"] - both["p2_out"]

        # D_IO
        def _row_jsd(row):
            p_in  = np.array([row["p1_in"],  row["p2_in"],
                               row["p3_in"],  row["p4_in"]],  dtype=float)
            p_out = np.array([row["p1_out"], row["p2_out"],
                               row["p3_out"], row["p4_out"]], dtype=float)
            return _jsd(p_in, p_out)

        both[f"D_IO_{rt}"] = both.apply(_row_jsd, axis=1)

        # QC flags: both rings must pass qc_comp (n_cells >= 5)
        both[f"qc_cpc_main_{rt}"]   = both["qc_comp_in"] & both["qc_comp_out"]
        # Strict: n_cells_ring >= 10 (approximate using qc_comp threshold)
        both[f"qc_cpc_strict_{rt}"] = (
            (both["n_cells_ring_in"]  >= MIN_CELLS_STRICT) &
            (both["n_cells_ring_out"] >= MIN_CELLS_STRICT)
        )

        keep = (["city", "country"] +
                [f"CPC_{rt}", f"D_IO_{rt}",
                 f"qc_cpc_main_{rt}", f"qc_cpc_strict_{rt}"])
        results_rt.append(both[keep].set_index(["city", "country"]))
        logging.info("    %d cities with both rings", len(both))

    # Merge all frameworks on city x country
    merged = results_rt[0]
    for df in results_rt[1:]:
        merged = merged.join(df, how="outer")
    merged = merged.reset_index()

    # Merge city covariates
    merged = merged.merge(city_meta, on=["city", "country"], how="left")

    # Save
    merged.to_csv(OUT_PATH, index=False, float_format="%.6f")
    logging.info("Saved: %s  (%d cities, %d cols)",
                 OUT_PATH, len(merged), merged.shape[1])

    # Validation
    for rt in ["rel", "abs", "scl"]:
        qc_col = f"qc_cpc_main_{rt}"
        if qc_col not in merged.columns:
            continue
        qc = merged[merged[qc_col].fillna(False)]
        cpc_col = f"CPC_{rt}"
        logging.info(
            "  [%s] n=%d  CPC mean=%.4f [%.4f, %.4f]  D_IO mean=%.4f",
            rt, len(qc),
            qc[cpc_col].mean(), qc[cpc_col].min(), qc[cpc_col].max(),
            qc[f"D_IO_{rt}"].mean(),
        )

    # CPC cross-framework correlation
    common = merged.dropna(subset=["CPC_rel", "CPC_abs", "CPC_scl"])
    logging.info("CPC cross-framework correlations (n=%d cities with all three):",
                 len(common))
    for a, b in [("rel", "abs"), ("rel", "scl"), ("abs", "scl")]:
        r = common[f"CPC_{a}"].corr(common[f"CPC_{b}"])
        logging.info("  CPC_%s vs CPC_%s: r=%.4f", a, b, r)

    # Sample audit
    logging.info("\n=== Sample audit (rel framework, qc_cpc_main_rel) ===")
    full_meta = pd.read_parquet(CITY_METRICS,
                                columns=["city", "country", "log_pop",
                                         "n_cells", "H17_star"])
    total = len(full_meta)
    in_set = set(zip(merged["city"], merged["country"]))
    full_meta["in_part4"] = full_meta.apply(
        lambda r: (r["city"], r["country"]) in in_set, axis=1)
    inc = full_meta[full_meta["in_part4"]]
    exc = full_meta[~full_meta["in_part4"]]
    logging.info("  All cities: %d  |  In Part4: %d  |  Excluded: %d (%.1f%%)",
                 total, len(inc), len(exc), 100 * len(exc) / total)
    if len(exc) > 0:
        logging.info("  Excl median log_pop=%.2f  vs  Incl median=%.2f",
                     exc["log_pop"].median(), inc["log_pop"].median())
        logging.info("  Excl median n_cells=%.0f  vs  Incl median=%.0f",
                     exc["n_cells"].median(), inc["n_cells"].median())

    print("\n=== CPC summary across distance frameworks ===")
    for rt in ["rel", "abs", "scl"]:
        qc_col = f"qc_cpc_main_{rt}"
        if qc_col not in merged.columns:
            continue
        qc = merged[merged[qc_col].fillna(False)]
        print(f"  [{rt}] n={len(qc):4d}  "
              f"mean={qc[f'CPC_{rt}'].mean():.4f}  "
              f"β_logpop={qc[f'CPC_{rt}'].corr(qc['log_pop']):.4f}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
