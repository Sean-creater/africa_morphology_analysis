"""
03_radial_diversity.py
======================
Q4 — Experiment B: 17-type Shannon diversity profile H_17(u).

For each city × radial_bin compute rarefied normalised Shannon entropy:
  H_17 = -sum_k p_k ln(p_k) / ln(17)

Rarefaction (n_rare = 20 cells drawn without replacement, 100 repeats):
  - Removes bias from unequal bin sizes across the radial gradient.
  - Bins with fewer than n_rare cells → NaN (excluded from tier average).

Two outputs
-----------
1. city_bin_diversity.parquet
   city | country | tier | radial_bin | n_cells | H17_raw | H17_rarefied

2. tier_diversity_profile.csv
   tier | radial_bin | n_cities | H17_mean | H17_se | H17_ci_lo | H17_ci_hi
   (country-cluster bootstrap CI, B=300)

3. beta_H_profile.csv
   H_17,i(u) ~ log(Pop_i)  per bin  →  β_H(u)
   radial_bin | n_cities | beta_H | beta_H_se | p_value

Key parameter: n_rare = 20 cells per rarefaction draw.
"""

import logging
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
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
N_RARE     = 20    # rarefaction sample size
N_RARE_REP = 100   # rarefaction repeats
N_BOOT     = 300   # bootstrap for CI
MIN_CITIES = 5     # minimum cities to estimate tier average
MIN_CELLS_REG = 15 # minimum cities per bin for regression


def _H17_rarefied(labels: np.ndarray, n: int = N_RARE,
                  reps: int = N_RARE_REP, rng=None) -> float:
    """Rarefied normalised Shannon entropy H_17 ∈ [0, 1]."""
    if rng is None:
        rng = np.random.default_rng()
    if len(labels) < n:
        return np.nan
    h_vals = []
    for _ in range(reps):
        samp = rng.choice(labels, size=n, replace=False)
        counts = np.bincount(samp, minlength=N_TYPES + 1)[1:N_TYPES + 1].astype(float)
        counts /= counts.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.nansum(counts * np.log(counts + 1e-15)) / np.log(N_TYPES)
        h_vals.append(float(h))
    return float(np.mean(h_vals))


def _H17_raw(labels: np.ndarray) -> float:
    counts = np.bincount(labels, minlength=N_TYPES + 1)[1:N_TYPES + 1].astype(float)
    counts /= counts.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.nansum(counts * np.log(counts + 1e-15)) / np.log(N_TYPES))


def _cluster_boot_ci(values: np.ndarray, groups: np.ndarray,
                     B: int = N_BOOT, alpha: float = 0.05) -> tuple:
    """Country-cluster bootstrap CI for the mean (default 95% — was
    previously alpha=0.10 i.e. a 90% interval, inconsistent with every
    other CI in this project)."""
    uniq_g = np.unique(groups)
    means  = []
    rng    = np.random.default_rng(42)
    for _ in range(B):
        g_samp = rng.choice(uniq_g, size=len(uniq_g), replace=True)
        boot_vals = np.concatenate([values[groups == g] for g in g_samp])
        means.append(float(np.nanmean(boot_vals)))
    lo = float(np.nanpercentile(means, 100 * alpha / 2))
    hi = float(np.nanpercentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    log.info("Loading radial_cells.parquet ...")
    cells = pd.read_parquet(CELLS_PQ,
                            columns=["city", "country", "label_k17",
                                     "radial_bin", "tier"])
    cells = cells[cells["radial_bin"].notna()].copy()
    cells["radial_bin"] = cells["radial_bin"].astype(int)
    log.info(f"  {len(cells):,} cells.")

    log.info("Loading city_tiers.parquet (log_pop) ...")
    tiers = pd.read_parquet(TIERS_PQ, columns=["city", "country", "log_pop", "tier"])
    pop_map = tiers.set_index(["city", "country"])["log_pop"].to_dict()

    log.info("Loading bin_grid.json ...")
    with open(OUT_TABLES / "bin_grid.json") as f:
        grid = json.load(f)
    n_bins = int(grid["n_bins"])

    # ── per-city × per-bin diversity ─────────────────────────────────────
    log.info("Computing H17 rarefied per city × bin ...")
    rng_global = np.random.default_rng(2024)

    records = []
    grouped = cells.groupby(["city", "country", "tier", "radial_bin"])
    for (city, country, tier, rbin), grp in grouped:
        labels = grp["label_k17"].values.astype(int)
        n      = len(labels)
        h_raw  = _H17_raw(labels)
        h_rare = _H17_rarefied(labels, n=N_RARE, reps=N_RARE_REP, rng=rng_global)
        records.append({
            "city":         city,
            "country":      country,
            "tier":         tier,
            "radial_bin":   int(rbin),
            "n_cells":      n,
            "H17_raw":      h_raw,
            "H17_rarefied": h_rare,
        })

    div = pd.DataFrame(records)
    div["log_pop"] = div.apply(
        lambda r: pop_map.get((r["city"], r["country"]), np.nan), axis=1)
    _path = OUT_TABLES / "city_bin_diversity.parquet"
    div.to_parquet(_path, index=False)
    _chk = pd.read_parquet(_path, columns=["city"])
    assert len(_chk) == len(div), \
        f"Parquet read-back mismatch: wrote {len(div)}, read {len(_chk)}"
    log.info(f"Saved + verified: city_bin_diversity.parquet  ({len(div):,} rows)")

    # ── tier × bin profiles ───────────────────────────────────────────────
    log.info("Computing tier × bin diversity profiles (bootstrap CI) ...")
    tier_rows = []
    for tier in ["T1", "T2", "T3", "T4"]:
        sub = div[div["tier"] == tier].dropna(subset=["H17_rarefied"])
        for rbin in range(n_bins):
            sbin = sub[sub["radial_bin"] == rbin].dropna(subset=["H17_rarefied"])
            if len(sbin) < MIN_CITIES:
                continue
            vals   = sbin["H17_rarefied"].values
            groups = sbin["country"].values
            mean_h = float(np.nanmean(vals))
            se_h   = float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals)))
            lo, hi = _cluster_boot_ci(vals, groups, B=N_BOOT)
            tier_rows.append({
                "tier":       tier,
                "radial_bin": rbin,
                "n_cities":   len(sbin),
                "H17_mean":   mean_h,
                "H17_se":     se_h,
                "H17_ci_lo":  lo,
                "H17_ci_hi":  hi,
            })
    tier_df = pd.DataFrame(tier_rows)
    tier_df.to_csv(OUT_TABLES / "tier_diversity_profile.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: tier_diversity_profile.csv")

    # ── β_H(u): population effect on H17 per bin ─────────────────────────
    log.info("Estimating β_H(u): OLS H17_rarefied ~ log(Pop) per bin ...")
    beta_rows = []
    df_reg = div.dropna(subset=["log_pop", "H17_rarefied"]).copy()

    for rbin in range(n_bins):
        sbin = df_reg[df_reg["radial_bin"] == rbin]
        if len(sbin) < MIN_CELLS_REG:
            continue
        X = sm.add_constant(sbin["log_pop"].values)
        y = sbin["H17_rarefied"].values
        try:
            res  = sm.OLS(y, X).fit()
            cov  = res.get_robustcov_results(cov_type="cluster",
                                             groups=sbin["country"].values)
            beta = float(cov.params[1])
            se   = float(cov.bse[1])
            t    = beta / se if se > 0 else np.nan
            pval = float(2 * sp_stats.t.sf(abs(t),
                                           df=sbin["country"].nunique() - 1))
            beta_rows.append({
                "radial_bin": rbin,
                "n_cities":   len(sbin),
                "beta_H":     beta,
                "beta_H_se":  se,
                "p_value":    pval,
            })
        except Exception:
            continue

    beta_df = pd.DataFrame(beta_rows)
    beta_df.to_csv(OUT_TABLES / "beta_H_profile.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: beta_H_profile.csv")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n=== H17 rarefied tier profiles ===")
    print(tier_df[["tier","radial_bin","n_cities","H17_mean"]].to_string(index=False))
    print("\n=== β_H(u) — population effect on H17 ===")
    if len(beta_df):
        print(beta_df[["radial_bin","n_cities","beta_H","beta_H_se","p_value"]].to_string(index=False))

    log.info("Done.")


if __name__ == "__main__":
    main()
