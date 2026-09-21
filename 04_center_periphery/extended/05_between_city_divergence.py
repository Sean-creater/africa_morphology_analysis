"""
05_between_city_divergence.py
=============================
Q4 — Experiment D: Between-city morphological divergence D_between(u).

For each radial_bin, compute the JS divergence of each city's 17-type
composition vector to the GLOBAL bin centroid (grand mean across ALL
cities in that bin, pooled across tiers):

  JS_i(u) = JS( P_i(u), P̄_global(u) )     base=2, ∈ [0, 1]

A single global reference per bin ensures JS values are on the same
scale across tiers and can be entered into a continuous log(Pop)
regression (β_D regression) without mixing incomparable reference points.

Tier-stratified averages of JS_i(u) are reported for visualisation only.

Outputs
-------
  tier_divergence_profile.csv
      tier | radial_bin | n_cities | D_between_mean | D_between_ci_lo | D_between_ci_hi
      (descriptive, uses common global-bin reference)

  city_bin_divergence.parquet
      city | country | tier | radial_bin | JS_to_centroid | log_pop
      (for regression of divergence ~ log(Pop))

  beta_D_profile.csv
      JS_i(u) ~ log(Pop_i) per bin  →  β_D(u)  (pooled OLS + country-clustered SE)
"""

import logging
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.spatial.distance import jensenshannon
from scipy import stats as sp_stats
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent
OUT_TABLES = ROOT / "outputs" / "tables"

PROFILES_PQ = OUT_TABLES / "city_bin_profiles.parquet"
TIERS_PQ    = OUT_TABLES / "city_tiers.parquet"

N_TYPES    = 17
N_BOOT     = 300
MIN_CITIES = 5
MIN_CITIES_REG = 10

TYPE_COLS = [f"p_{k:02d}" for k in range(1, N_TYPES + 1)]


def _js_to_centroid(mat: np.ndarray) -> np.ndarray:
    """
    JS divergence of each row vector to the row-mean (centroid).
    mat: (n, 17) array of composition vectors (sum-to-1 rows).
    Returns (n,) array of JS distances.
    """
    centroid = mat.mean(axis=0)
    centroid = centroid / centroid.sum() if centroid.sum() > 0 else centroid
    dists = []
    for row in mat:
        s = row.sum()
        p = row / s if s > 0 else row
        dists.append(float(jensenshannon(p, centroid, base=2)))
    return np.array(dists)


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

    log.info("Loading city_bin_profiles.parquet ...")
    prof = pd.read_parquet(PROFILES_PQ)
    log.info(f"  {len(prof):,} city × bin records.")

    log.info("Loading city_tiers.parquet (log_pop) ...")
    tiers = pd.read_parquet(TIERS_PQ, columns=["city", "country", "log_pop"])
    pop_map = tiers.set_index(["city", "country"])["log_pop"].to_dict()

    log.info("Loading bin_grid.json ...")
    with open(OUT_TABLES / "bin_grid.json") as f:
        grid = json.load(f)
    n_bins = int(grid["n_bins"])

    # ── JS to centroid per city × bin ─────────────────────────────────────
    # Centroid is computed across ALL cities in a bin (not tier-specific) so
    # that JS distances are comparable across tiers and can be pooled in a
    # single regression.  Tier-stratified averages are reported separately.
    log.info("Computing JS-to-centroid per city × bin (global centroid per bin) ...")
    div_records = []
    tier_div_rows = []

    # First pass: global centroid per bin
    bin_centroids = {}
    for rbin, grp_all in prof.groupby("radial_bin"):
        if len(grp_all) < MIN_CITIES:
            continue
        mat = grp_all[TYPE_COLS].values.astype(float)
        centroid = mat.mean(axis=0)
        s = centroid.sum()
        bin_centroids[int(rbin)] = centroid / s if s > 0 else centroid

    # Second pass: JS distance of every city to its bin's global centroid
    for rbin, grp_all in prof.groupby("radial_bin"):
        if int(rbin) not in bin_centroids:
            continue
        centroid = bin_centroids[int(rbin)]
        for _, row in grp_all.iterrows():
            p = np.array([row[c] for c in TYPE_COLS], dtype=float)
            s = p.sum()
            p = p / s if s > 0 else p
            d = float(jensenshannon(p, centroid, base=2))
            div_records.append({
                "city":           row["city"],
                "country":        row["country"],
                "tier":           row["tier"],
                "radial_bin":     int(rbin),
                "JS_to_centroid": d,
            })

    div_df = pd.DataFrame(div_records)

    # Tier-stratified averages (for visualisation only)
    for (tier, rbin), grp in div_df.groupby(["tier", "radial_bin"]):
        if len(grp) < MIN_CITIES:
            continue
        vals   = grp["JS_to_centroid"].values
        groups = grp["country"].values
        lo, hi = _cluster_boot_ci(vals, groups)
        tier_div_rows.append({
            "tier":              tier,
            "radial_bin":        int(rbin),
            "n_cities":          len(grp),
            "D_between_mean":    float(np.nanmean(vals)),
            "D_between_se":      float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))),
            "D_between_ci_lo":   lo,
            "D_between_ci_hi":   hi,
        })

    div_df  = pd.DataFrame(div_records)
    tier_df = pd.DataFrame(tier_div_rows)

    div_df["log_pop"] = div_df.apply(
        lambda r: pop_map.get((r["city"], r["country"]), np.nan), axis=1)

    _path_div = OUT_TABLES / "city_bin_divergence.parquet"
    div_df.to_parquet(_path_div, index=False)
    _chk = pd.read_parquet(_path_div)
    assert len(_chk) == len(div_df), \
        f"Parquet read-back mismatch: wrote {len(div_df)}, read {len(_chk)}"
    log.info(f"Saved + verified: city_bin_divergence.parquet ({len(div_df):,} rows)")
    tier_df.to_csv(OUT_TABLES / "tier_divergence_profile.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: tier_divergence_profile.csv")

    # ── β_D(u): regression JS ~ log(Pop) per bin ─────────────────────────
    log.info("Estimating β_D(u): OLS JS ~ log(Pop) per bin ...")
    df_reg   = div_df.dropna(subset=["log_pop", "JS_to_centroid"])
    beta_rows = []
    for rbin in range(n_bins):
        sbin = df_reg[df_reg["radial_bin"] == rbin]
        if len(sbin) < MIN_CITIES_REG:
            continue
        X = sm.add_constant(sbin["log_pop"].values)
        y = sbin["JS_to_centroid"].values
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
                "beta_D":     beta,
                "beta_D_se":  se,
                "p_value":    pval,
            })
        except Exception:
            continue

    beta_df = pd.DataFrame(beta_rows)
    beta_df.to_csv(OUT_TABLES / "beta_D_profile.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: beta_D_profile.csv")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n=== D_between (JS-to-centroid) tier profiles ===")
    print(tier_df[["tier","radial_bin","n_cities","D_between_mean"]].to_string(index=False))
    print("\n=== β_D(u) — does divergence scale with city size? ===")
    if len(beta_df):
        print(beta_df.to_string(index=False))

    log.info("Done.")


if __name__ == "__main__":
    main()
