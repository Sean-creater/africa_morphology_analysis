"""
02_radial_type_profiles.py
==========================
Q4 — Experiment A: 17-type radial morphology spectrum.

For each city × radial_bin, compute the composition vector
  P_i(b) = [p_i1(b), ..., p_i17(b)]
where p_ik(b) = fraction of cells in bin b with label_k17 = k.

Two outputs
-----------
1. city_bin_profiles.parquet  (wide)
   city | country | tier | radial_bin | n_cells | p_01 … p_17
   Used for JSD calculations in 05_between_city_divergence.py.

2. tier_bin_type_shares.csv  (long, Tier × bin × type, tier-averaged)
   tier | radial_bin | type17 | share_mean | share_se
   Used for Figure 4A heatmaps.

3. size_effect_profiles.csv  (β_k(u): population effect by type × bin)
   radial_bin | type17 | beta | beta_se | p_value
   OLS:  p_ik(b) ~ log(Pop_i)  per type per bin (country-clustered SE).
   Used for Figure 4A size-effect heatmap.

QC: city excluded from a bin if n_cells_in_bin < 5.
"""

import logging
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent
PART3_DIR  = ROOT.parent.parent / "03_urban_hierarchy" / "outputs" / "tables"
OUT_TABLES = ROOT / "outputs" / "tables"

CELLS_PQ   = OUT_TABLES / "radial_cells.parquet"
TIERS_PQ   = OUT_TABLES / "city_tiers.parquet"

N_TYPES    = 17
MIN_CELLS_BIN = 5   # minimum cells in a bin to include city–bin observation


def _country_clustered_se(model_res, groups) -> np.ndarray:
    """Return cluster-robust BSE array [const, log_pop] using country as cluster."""
    try:
        cov = model_res.get_robustcov_results(cov_type="cluster", groups=groups)
        return np.asarray(cov.bse)
    except Exception:
        return np.asarray(model_res.bse)


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────
    log.info("Loading radial_cells.parquet ...")
    cells = pd.read_parquet(CELLS_PQ,
                            columns=["city", "country", "label_k17",
                                     "Pop2022", "radial_bin", "tier"])
    cells = cells[cells["radial_bin"].notna()].copy()
    cells["radial_bin"] = cells["radial_bin"].astype(int)
    log.info(f"  {len(cells):,} cells with valid radial_bin.")

    log.info("Loading city_tiers.parquet ...")
    tiers = pd.read_parquet(TIERS_PQ, columns=["city", "country", "pop", "log_pop"])
    pop_map = tiers.set_index(["city", "country"])["log_pop"].to_dict()

    log.info("Loading bin_grid.json ...")
    with open(OUT_TABLES / "bin_grid.json") as f:
        grid = json.load(f)
    n_bins = int(grid["n_bins"])

    # ── per-city × per-bin composition vectors ───────────────────────────
    log.info("Computing city × bin composition vectors ...")

    type_cols = [f"p_{k:02d}" for k in range(1, N_TYPES + 1)]
    records   = []

    grouped = cells.groupby(["city", "country", "tier", "radial_bin"])
    for (city, country, tier, rbin), grp in grouped:
        n = len(grp)
        if n < MIN_CELLS_BIN:
            continue
        counts = np.bincount(grp["label_k17"].values.astype(int),
                             minlength=N_TYPES + 1)[1:N_TYPES + 1].astype(float)
        counts /= counts.sum()
        row = {"city": city, "country": country, "tier": tier,
               "radial_bin": int(rbin), "n_cells": n}
        for k, v in enumerate(counts, start=1):
            row[f"p_{k:02d}"] = float(v)
        records.append(row)

    prof = pd.DataFrame(records)
    prof["log_pop"] = prof.apply(
        lambda r: pop_map.get((r["city"], r["country"]), np.nan), axis=1)
    log.info(f"  {len(prof):,} city × bin observations.")

    _path_prof = OUT_TABLES / "city_bin_profiles.parquet"
    prof.to_parquet(_path_prof, index=False)
    _chk = pd.read_parquet(_path_prof)
    assert len(_chk) == len(prof), \
        f"Parquet read-back mismatch: wrote {len(prof)}, read {len(_chk)}"
    log.info(f"Saved + verified: city_bin_profiles.parquet ({len(prof):,} rows)")

    # ── tier × bin × type mean shares ────────────────────────────────────
    log.info("Computing tier × bin × type tier-averaged shares ...")
    tier_rows = []
    for tier in ["T1", "T2", "T3", "T4"]:
        sub = prof[prof["tier"] == tier]
        for rbin in range(n_bins):
            sbin = sub[sub["radial_bin"] == rbin]
            if len(sbin) == 0:
                continue
            for k in range(1, N_TYPES + 1):
                col = f"p_{k:02d}"
                vals = sbin[col].values
                tier_rows.append({
                    "tier":       tier,
                    "radial_bin": rbin,
                    "type17":     k,
                    "n_cities":   len(sbin),
                    "share_mean": float(np.nanmean(vals)),
                    "share_se":   float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
                                  if len(vals) > 1 else np.nan,
                })
    tier_df = pd.DataFrame(tier_rows)
    tier_df.to_csv(OUT_TABLES / "tier_bin_type_shares.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: tier_bin_type_shares.csv")

    # ── β_k(u): population-scaling effect per type per bin ───────────────
    log.info("Estimating β_k(u): OLS p_k(b) ~ log(Pop) per type × bin ...")
    size_rows = []
    df_reg = prof.dropna(subset=["log_pop"]).copy()

    for rbin in range(n_bins):
        sbin = df_reg[df_reg["radial_bin"] == rbin]
        if len(sbin) < 20:
            continue
        X_df = sm.add_constant(pd.DataFrame({"log_pop": sbin["log_pop"].values}))
        groups = sbin["country"].values
        for k in range(1, N_TYPES + 1):
            y = sbin[f"p_{k:02d}"].values
            try:
                res  = sm.OLS(y, X_df).fit()
                bse  = _country_clustered_se(res, groups)
                beta = float(res.params["log_pop"])
                se   = float(bse[1])   # index 1 = log_pop slope
                t    = beta / se if se > 0 else np.nan
                pval = float(2 * sp_stats.t.sf(abs(t), df=sbin["country"].nunique() - 1))
                size_rows.append({
                    "radial_bin": rbin,
                    "type17":     k,
                    "n_cities":   len(sbin),
                    "beta":       beta,
                    "beta_se":    se,
                    "p_value":    pval,
                })
            except Exception:
                continue

    size_df = pd.DataFrame(size_rows)

    # Benjamini–Hochberg FDR correction across all 170 tests (10 bins × 17 types)
    if len(size_df) > 0:
        _, q_vals, _, _ = multipletests(size_df["p_value"].values,
                                        alpha=0.05, method="fdr_bh")
        size_df["q_value_bh"] = q_vals
    else:
        size_df["q_value_bh"] = pd.Series(dtype=float)

    size_df.to_csv(OUT_TABLES / "size_effect_profiles.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: size_effect_profiles.csv")

    # ── brief summary ─────────────────────────────────────────────────────
    print("\n=== Tier sample counts per radial bin (n cities) ===")
    piv = prof.groupby(["tier", "radial_bin"]).size().unstack("radial_bin", fill_value=0)
    print(piv.to_string())

    if len(size_df) > 0:
        top_pos = size_df.nlargest(5, "beta")[["radial_bin","type17","beta","p_value"]]
        top_neg = size_df.nsmallest(5, "beta")[["radial_bin","type17","beta","p_value"]]
        print("\n--- Top 5 positive β_k(u) (large cities have MORE of this type here) ---")
        print(top_pos.to_string(index=False))
        print("\n--- Top 5 negative β_k(u) (large cities have LESS) ---")
        print(top_neg.to_string(index=False))

    log.info("Done.")


if __name__ == "__main__":
    main()
