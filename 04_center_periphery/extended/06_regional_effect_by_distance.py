"""
06_regional_effect_by_distance.py
==================================
Q4 — Experiment: Connecting Part 2 regional context to intra-urban position.

For each radial bin b, measure how much of the cross-city variation in
17-type morphology composition is explained by:
  - Climate zone        (Köppen-based; from city-level context)
  - Colonial legacy     (binary categories)
  - log(GDP per capita) (country-level, if available)

Method
------
For each radial bin:
  1. Compute PCoA scores from the inter-city JS-distance matrix
     (first N_PCA_AXES positive-eigenvalue axes).
  2. For each predictor block, fit OLS of PCoA scores on the block alone
     and record R² — reported as R2_climate_marginal, R2_colonial_marginal.
  3. Fit all predictors jointly and record R2_total.

INTERPRETATION NOTE
-------------------
R2_climate_marginal and R2_colonial_marginal are MARGINAL (unadjusted) R²
values, NOT partial R². Because climate and colonial legacy are correlated,
they cannot be summed (their sum exceeds R2_total). The difference
  R2_total − max(R2_climate_marginal, R2_colonial_marginal)
provides a conservative lower bound on the joint unique contribution.
True variation partitioning (Borcard 1992) requires a 2³ framing and is
not implemented here.

IMPORTANT
---------
External predictors must be EXOGENOUS — do NOT use any morphology-derived
variable (community label, H17, r17). Only climate / economic / historical.

This analysis is SUPPLEMENTARY; its outputs are reported as exploratory
(labeled _marginal to avoid confusion with partial R²).

Outputs
-------
  regional_effect_profile.csv
      radial_bin | n_cities | R2_climate | R2_gdp | R2_colonial | R2_total
  (saved to outputs/tables/ and figures/supplementary/)
"""

import logging
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).parent
PART3_DIR   = ROOT.parent.parent / "03_urban_hierarchy" / "outputs" / "tables"
OUT_TABLES  = ROOT / "outputs" / "tables"

PROFILES_PQ = OUT_TABLES / "city_bin_profiles.parquet"
TIERS_PQ    = OUT_TABLES / "city_tiers.parquet"

N_TYPES    = 17
TYPE_COLS  = [f"p_{k:02d}" for k in range(1, N_TYPES + 1)]
MIN_CITIES = 20
N_PCA_AXES = 5    # PCoA axes used as response in variation partitioning


def _pcoa_axes(mat: np.ndarray, n_axes: int = N_PCA_AXES) -> np.ndarray:
    """
    Principal Coordinate Analysis on JS-distance matrix.
    Returns (n_cities, n_axes) array of PCoA scores.
    JS-distance matrix computed via vectorised cdist (O(n²) memory but
    avoids Python-loop overhead; acceptable for n ≤ few thousand per bin).
    """
    from scipy.spatial.distance import cdist
    n = mat.shape[0]
    # clip to avoid log(0); rows should already sum to 1
    mat_safe = np.clip(mat, 1e-15, None)
    mat_safe /= mat_safe.sum(axis=1, keepdims=True)
    # Jensen-Shannon distance via scipy cdist (vectorised C implementation)
    D = cdist(mat_safe, mat_safe, metric="jensenshannon")
    # double centring
    D2 = D ** 2
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand    = D2.mean()
    B        = -0.5 * (D2 - row_mean - col_mean + grand)
    # eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    idx     = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    pos     = eigvals > 0
    k       = min(n_axes, int(pos.sum()))
    if k == 0:
        return np.zeros((n, n_axes))
    scores  = eigvecs[:, :k] * np.sqrt(eigvals[:k])
    if k < n_axes:
        scores = np.hstack([scores, np.zeros((n, n_axes - k))])
    return scores


def _partial_r2(Y: np.ndarray, X_full: np.ndarray, X_part: np.ndarray) -> float:
    r"""
    Marginal (unadjusted) R² of X_part vs intercept-only baseline.
    Named _partial_r2 for API compatibility but returns the unadjusted
    R² of X_part alone (not X_full minus X_part).
    See INTERPRETATION NOTE in module docstring.
    """
    from numpy.linalg import lstsq
    n = Y.shape[0]
    # intercept-only baseline
    Xb = np.ones((n, 1))
    # full model
    Xf  = np.column_stack([Xb, X_full])
    coef_f, _, _, _ = lstsq(Xf, Y, rcond=None)
    ss_res_f = float(np.sum((Y - Xf @ coef_f) ** 2))
    ss_tot   = float(np.sum((Y - Y.mean(axis=0)) ** 2))
    r2_full  = 1 - ss_res_f / ss_tot if ss_tot > 0 else 0.0
    # reduced model (full without X_part)
    if X_full.shape[1] == X_part.shape[1]:
        # only this predictor block
        Xr  = Xb
    else:
        # remove columns of X_part from X_full
        Xr = Xb   # simplified: just compute R²(part alone) as rough proxy
    coef_r, _, _, _ = lstsq(np.column_stack([Xb, X_part]), Y, rcond=None)
    ss_res_r = float(np.sum((Y - np.column_stack([Xb, X_part]) @ coef_r) ** 2))
    r2_part  = 1 - ss_res_r / ss_tot if ss_tot > 0 else 0.0
    return float(r2_part)


def _try_load_regional_vars() -> pd.DataFrame | None:
    """
    Attempt to load climate / colonial / GDP covariates from Part 2 or Part 3.
    Returns a DataFrame with columns [city, country, climate, colonial, log_gdp]
    or None if not available.
    """
    # Try Part 3 city_metrics for country-level GDP and climate proxy
    city_pq = PART3_DIR / "city_metrics.parquet"
    if not city_pq.exists():
        log.warning("city_metrics.parquet not found — skipping regional analysis.")
        return None

    city = pd.read_parquet(city_pq)
    needed = {"city", "country"}
    extra  = {"climate_zone", "colonial_legacy", "log_gdp_pc",
               "gdp_pc", "climate", "colonial", "gdp"}
    avail  = set(city.columns) & extra
    if len(avail) == 0:
        log.warning("No regional covariate columns found in city_metrics.parquet. "
                    "Columns present: " + str(list(city.columns)[:20]))
        log.warning("Skipping 06_regional_effect_by_distance.py — "
                    "add climate/GDP/colonial columns to city_metrics to enable this analysis.")
        return None

    log.info(f"  Regional covariate columns available: {sorted(avail)}")
    return city[["city", "country"] + list(avail)]


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    log.info("Loading city_bin_profiles.parquet ...")
    prof = pd.read_parquet(PROFILES_PQ)

    log.info("Loading bin_grid.json ...")
    with open(OUT_TABLES / "bin_grid.json") as f:
        grid = json.load(f)
    n_bins = int(grid["n_bins"])

    # ── load regional covariates ─────────────────────────────────────────
    reg = _try_load_regional_vars()
    if reg is None:
        # Write empty placeholder so downstream scripts can check
        pd.DataFrame(columns=["radial_bin", "n_cities",
                               "R2_climate_marginal", "R2_gdp_marginal",
                               "R2_colonial_marginal", "R2_total", "note"]) \
          .assign(note="Regional covariates not available in city_metrics.parquet") \
          .to_csv(OUT_TABLES / "regional_effect_profile.csv", index=False)
        log.info("Saved: regional_effect_profile.csv  (empty — covariates not available)")
        log.info("Done (skipped — add climate/GDP/colonial to city_metrics to run).")
        return

    # ── merge profiles with regional vars ───────────────────────────────
    merged = prof.merge(reg, on=["city", "country"], how="inner")
    log.info(f"  {len(merged):,} city × bin records with regional covariates.")

    # detect column names dynamically
    clim_col     = next((c for c in ["climate_zone", "climate"] if c in merged), None)
    colonial_col = next((c for c in ["colonial_legacy", "colonial"] if c in merged), None)
    gdp_col      = next((c for c in ["log_gdp_pc", "gdp_pc", "log_gdp", "gdp"]
                         if c in merged), None)

    if clim_col:
        merged = pd.get_dummies(merged, columns=[clim_col], drop_first=True)
        clim_dummies = [c for c in merged.columns if c.startswith(clim_col + "_")]
    else:
        clim_dummies = []

    if colonial_col:
        merged = pd.get_dummies(merged, columns=[colonial_col], drop_first=True)
        col_dummies = [c for c in merged.columns if c.startswith(colonial_col + "_")]
    else:
        col_dummies = []

    if gdp_col:
        if "log" not in gdp_col:
            merged[gdp_col] = np.log(merged[gdp_col].clip(1))
        gdp_cols = [gdp_col]
    else:
        gdp_cols = []

    # ── variation partitioning per radial bin ───────────────────────────
    log.info("Variation partitioning per radial bin ...")
    result_rows = []

    for rbin in range(n_bins):
        sbin = merged[merged["radial_bin"] == rbin].dropna(
            subset=TYPE_COLS + clim_dummies + col_dummies + gdp_cols)
        if len(sbin) < MIN_CITIES:
            continue

        mat = sbin[TYPE_COLS].values.astype(float)
        # normalise rows
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = np.where(row_sums > 0, mat / row_sums, mat)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y = _pcoa_axes(mat, n_axes=N_PCA_AXES)

        r2_clim = _partial_r2(Y, np.zeros((len(sbin), 0)),
                               sbin[clim_dummies].values.astype(float)) \
                  if clim_dummies else np.nan
        r2_gdp  = _partial_r2(Y, np.zeros((len(sbin), 0)),
                               sbin[gdp_cols].values.astype(float)) \
                  if gdp_cols else np.nan
        r2_col  = _partial_r2(Y, np.zeros((len(sbin), 0)),
                               sbin[col_dummies].values.astype(float)) \
                  if col_dummies else np.nan

        # total R² (all predictors)
        all_preds = clim_dummies + col_dummies + gdp_cols
        r2_total  = _partial_r2(Y, np.zeros((len(sbin), 0)),
                                 sbin[all_preds].values.astype(float)) \
                    if all_preds else np.nan

        result_rows.append({
            "radial_bin":          rbin,
            "n_cities":            len(sbin),
            "R2_climate_marginal": r2_clim,
            "R2_gdp_marginal":     r2_gdp,
            "R2_colonial_marginal":r2_col,
            "R2_total":            r2_total,
        })
        log.info(f"  bin {rbin}: n={len(sbin)}  R²_clim={r2_clim:.3f}  "
                 f"R²_gdp={r2_gdp:.3f}  R²_col={r2_col:.3f}")

    result = pd.DataFrame(result_rows)
    result.to_csv(OUT_TABLES / "regional_effect_profile.csv",
                  index=False, float_format="%.4f")
    log.info("Saved: regional_effect_profile.csv")

    print("\n=== Regional effect by radial position ===")
    print(result.to_string(index=False))

    log.info("Done.")


if __name__ == "__main__":
    main()
