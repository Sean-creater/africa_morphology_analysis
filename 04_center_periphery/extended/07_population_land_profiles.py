"""
07_population_land_profiles.py
===============================
Q4 — Experiment E: Population–land allocation profile (SDG 11.3).

For each Tier × radial_bin × morphology_type compute:

  Land share   A_kb = N_cells(k, b) / N_cells(city)          [hex cells → equal area]
  Pop share    P_kb = sum(Pop_cells(k, b)) / sum(Pop(city))
  RPC_kb       = P_kb / A_kb                                  (> 1: pop-dense relative to land)

Then aggregate over cities within each Tier × bin, reporting:
  - city-equally-weighted mean RPC per (tier, bin, type)
  - radial RPC profile: averaging over all types → RPC(u) per tier

SDG 11.3 read-out:
  The spatial coincidence of (β_H > 0, β_r > 0) with (RPC < 1) identifies
  where morphological scaling occurs in land-abundant but population-sparse
  peripheral zones.

Outputs
-------
  city_bin_rpc.parquet
      city | country | tier | radial_bin | land_share | pop_share | rpc

  tier_rpc_profile.csv
      tier | radial_bin | n_cities | rpc_mean | rpc_ci_lo | rpc_ci_hi

  tier_type_rpc.csv
      tier | radial_bin | type17 | rpc_mean | n_cities
      (useful for identifying which morphology types are land-abundant at periphery)
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent
OUT_TABLES = ROOT / "outputs" / "tables"
CELLS_PQ   = OUT_TABLES / "radial_cells.parquet"
TIERS_PQ   = OUT_TABLES / "city_tiers.parquet"

N_TYPES    = 17
N_BOOT     = 300
MIN_CITIES = 3


def _cluster_boot_ci(values: np.ndarray, groups: np.ndarray,
                     B: int = N_BOOT, alpha: float = 0.10) -> tuple:
    uniq_g = np.unique(groups)
    means  = []
    rng    = np.random.default_rng(42)
    for _ in range(B):
        g_samp    = rng.choice(uniq_g, size=len(uniq_g), replace=True)
        boot_vals = np.concatenate([values[groups == g] for g in g_samp])
        means.append(float(np.nanmean(boot_vals)))
    lo = float(np.nanpercentile(means, 100 * alpha / 2))
    hi = float(np.nanpercentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    log.info("Loading radial_cells.parquet ...")
    cells_all = pd.read_parquet(CELLS_PQ,
                                columns=["city", "country", "label_k17",
                                         "Pop2022", "radial_bin", "tier"])
    log.info(f"  {len(cells_all):,} cells total (before P95 filter).")

    # Per-city totals computed from ALL cells (including those outside P95),
    # so that land_share and pop_share are relative to the full city extent.
    city_totals = cells_all.groupby(["city", "country"]).agg(
        total_cells=("label_k17", "count"),
        total_pop=("Pop2022", "sum")
    ).reset_index()

    # Working set: only cells with a valid radial_bin (within common support)
    cells = cells_all[cells_all["radial_bin"].notna()].copy()
    cells["radial_bin"] = cells["radial_bin"].astype(int)
    log.info(f"  {len(cells):,} cells with valid radial_bin (within P95 support).")

    log.info("Loading bin_grid.json ...")
    with open(OUT_TABLES / "bin_grid.json") as f:
        grid = json.load(f)
    n_bins = int(grid["n_bins"])

    # ── per-city × bin RPC (radial aggregate, all types combined) ─────────
    log.info("Computing city × bin land/pop shares and RPC ...")
    bin_agg = cells.groupby(["city", "country", "tier", "radial_bin"]).agg(
        n_cells=("label_k17", "count"),
        pop_sum=("Pop2022", "sum")
    ).reset_index()
    bin_agg = bin_agg.merge(city_totals, on=["city", "country"])

    bin_agg["land_share"] = bin_agg["n_cells"] / bin_agg["total_cells"]
    bin_agg["pop_share"]  = bin_agg["pop_sum"] / bin_agg["total_pop"].clip(1e-6)
    bin_agg["rpc"]        = bin_agg["pop_share"] / bin_agg["land_share"].clip(1e-6)

    # clip extreme RPC (>10 likely data artefacts in tiny bins)
    # Report truncation count before clipping so auditors can verify magnitude
    n_clipped = int((bin_agg["rpc"] > 10.0).sum())
    pct_clipped = 100 * n_clipped / max(len(bin_agg), 1)
    log.info(f"  RPC clipping: {n_clipped} rows ({pct_clipped:.1f}%) have rpc > 10 "
             f"and are capped at 10 (raw rpc_raw column retained)")
    bin_agg["rpc_raw"] = bin_agg["rpc"].copy()
    bin_agg["rpc"]     = bin_agg["rpc"].clip(upper=10.0)

    keep = ["city", "country", "tier", "radial_bin",
            "n_cells", "land_share", "pop_share", "rpc", "rpc_raw"]
    _path_rpc = OUT_TABLES / "city_bin_rpc.parquet"
    bin_agg[keep].to_parquet(_path_rpc, index=False)
    _chk = pd.read_parquet(_path_rpc)
    assert len(_chk) == len(bin_agg), \
        f"Parquet read-back mismatch: wrote {len(bin_agg)}, read {len(_chk)}"
    log.info(f"Saved + verified: city_bin_rpc.parquet ({len(bin_agg):,} rows)")

    # ── tier × bin RPC profile ────────────────────────────────────────────
    log.info("Computing tier × bin RPC profiles ...")
    tier_rows = []
    for tier in ["T1", "T2", "T3", "T4"]:
        sub = bin_agg[bin_agg["tier"] == tier].dropna(subset=["rpc"])
        for rbin in range(n_bins):
            sbin = sub[sub["radial_bin"] == rbin]
            if len(sbin) < MIN_CITIES:
                continue
            vals   = sbin["rpc"].values
            groups = sbin["country"].values
            lo, hi = _cluster_boot_ci(vals, groups)
            tier_rows.append({
                "tier":       tier,
                "radial_bin": rbin,
                "n_cities":   len(sbin),
                "rpc_mean":   float(np.nanmean(vals)),
                "rpc_median": float(np.nanmedian(vals)),
                "rpc_se":     float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))),
                "rpc_ci_lo":  lo,
                "rpc_ci_hi":  hi,
            })
    tier_rpc = pd.DataFrame(tier_rows)
    tier_rpc.to_csv(OUT_TABLES / "tier_rpc_profile.csv",
                    index=False, float_format="%.4f")
    log.info("Saved: tier_rpc_profile.csv")

    # ── tier × bin × type RPC ─────────────────────────────────────────────
    log.info("Computing tier × bin × type RPC ...")
    type_agg = cells.groupby(
        ["city", "country", "tier", "radial_bin", "label_k17"]
    ).agg(n_cells_type=("Pop2022", "count"),
          pop_sum_type=("Pop2022", "sum")).reset_index()
    type_agg = type_agg.merge(city_totals, on=["city", "country"])
    type_agg["land_share_type"] = type_agg["n_cells_type"] / type_agg["total_cells"]
    type_agg["pop_share_type"]  = type_agg["pop_sum_type"] / type_agg["total_pop"].clip(1e-6)
    type_agg["rpc_type"] = (type_agg["pop_share_type"] /
                            type_agg["land_share_type"].clip(1e-6)).clip(upper=10.0)

    tier_type_rows = []
    for (tier, rbin, t17), grp in type_agg.groupby(
            ["tier", "radial_bin", "label_k17"]):
        vals = grp["rpc_type"].dropna().values
        if len(vals) < MIN_CITIES:
            continue
        tier_type_rows.append({
            "tier":       tier,
            "radial_bin": int(rbin),
            "type17":     int(t17),
            "n_cities":   len(vals),
            "rpc_mean":   float(np.nanmean(vals)),
            "rpc_median": float(np.nanmedian(vals)),
        })
    tier_type_rpc = pd.DataFrame(tier_type_rows)
    tier_type_rpc.to_csv(OUT_TABLES / "tier_type_rpc.csv",
                         index=False, float_format="%.4f")
    log.info("Saved: tier_type_rpc.csv")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n=== RPC tier profiles (radially aggregated) ===")
    piv = tier_rpc.pivot(index="tier", columns="radial_bin", values="rpc_mean")
    piv.columns = [f"bin{c}" for c in piv.columns]
    print(piv.round(3).to_string())

    print("\n  RPC > 1 → population denser than land share (core)")
    print("  RPC < 1 → land abundant relative to population (periphery)")

    log.info("Done.")


if __name__ == "__main__":
    main()
