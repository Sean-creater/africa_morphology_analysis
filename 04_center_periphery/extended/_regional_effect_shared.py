"""
_regional_effect_shared.py
===========================
Shared implementation for the regional-context partial R² analysis, used
by both 06_regional_effect_by_distance.py and 10_figure_main.py so the two
report the same numbers from the same model rather than two independent
approximations.

Method
------
For each radial bin:
  1. PCoA on the inter-city Jensen-Shannon distance matrix of 17-type
     composition (first N_PCA_AXES positive-eigenvalue axes), giving a
     multivariate response Y (n_cities x N_PCA_AXES).
  2. Fit Y ~ climate + colonial + z(log GDP) + z(log population) by
     multivariate OLS (one model per PCoA axis, sharing the design
     matrix), on the complete-case sample for that bin.
  3. For each predictor block g, partial R^2_g = 1 - SSE_full / SSE_reduced(-g),
     where SSE is summed over cities and PCoA axes, and the reduced model
     drops only block g, keeping every other block (population included,
     so the climate/colonial contribution is not confounded with city
     size).
  4. Confidence intervals via country-level cluster bootstrap (resample
     countries with replacement, keep all cities in each resampled
     country), since observations within a country are not independent.

Population is retained as an adjustment variable in every model
(full and every reduced model except the population-drop model), so the
climate/colonial contributions reported here are net of city size.
"""
import logging
import os
from multiprocessing import get_context
from typing import Optional

import numpy as np
import pandas as pd
import threadpoolctl
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)

N_PCA_AXES = 5
MIN_CITIES = 20
DEFAULT_N_BOOT = 2000
RANDOM_SEED = 42
DEFAULT_N_WORKERS = min(14, os.cpu_count() or 1)

# Populated in each worker process by _init_bootstrap_worker(); avoids
# re-pickling the (unchanging) per-bin dataframe for every one of n_boot
# tasks, since only the small sampled-country list varies per task.
_WORKER_STATE: dict = {}


def pcoa_axes(composition: np.ndarray, n_axes: int = N_PCA_AXES) -> np.ndarray:
    """Principal Coordinate Analysis on the Jensen-Shannon distance matrix
    of row-normalised composition vectors.

    Args:
        composition: (n_cities, n_types) array; rows need not already sum
            to 1 (renormalised internally).
        n_axes: number of positive-eigenvalue axes to return.

    Returns:
        (n_cities, n_axes) array of PCoA scores. Axes beyond the number of
        positive eigenvalues are zero-filled.
    """
    mat = np.clip(composition, 1e-15, None)
    mat = mat / mat.sum(axis=1, keepdims=True)
    D = cdist(mat, mat, metric="jensenshannon")

    D2 = D ** 2
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + grand)

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    pos = eigvals > 0
    k = min(n_axes, int(pos.sum()))
    n = mat.shape[0]
    if k == 0:
        return np.zeros((n, n_axes))

    scores = eigvecs[:, :k] * np.sqrt(eigvals[:k])
    if k < n_axes:
        scores = np.hstack([scores, np.zeros((n, n_axes - k))])
    return scores


def _ols_sse(X: np.ndarray, Y: np.ndarray) -> float:
    """Residual sum of squares of a multivariate OLS fit, summed over
    all response columns."""
    coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    return float(np.sum(resid ** 2))


def build_design_blocks(
    df: pd.DataFrame,
    climate_col: str,
    colonial_col: str,
    log_gdp_col: str,
    log_pop_col: str,
) -> dict[str, np.ndarray]:
    """One-hot encode categorical blocks and standardise continuous blocks.

    Reference category for each categorical block is its most frequent
    level in `df`; ties are broken alphabetically. `value_counts().idxmax()`
    alone breaks ties by first-seen row order, which would make the
    reference category — and therefore the fitted dummy coefficients —
    depend on row order, so counts are sorted by (count desc, label asc)
    instead.
    """
    blocks: dict[str, np.ndarray] = {}

    for block_name, col in [("climate", climate_col), ("colonial", colonial_col)]:
        counts = df[col].value_counts()
        ref = sorted(counts.index, key=lambda lbl: (-counts[lbl], str(lbl)))[0]
        dummies = pd.get_dummies(df[col], dtype=float)
        dummies = dummies.drop(columns=[ref])
        blocks[block_name] = dummies.to_numpy()

    for block_name, col in [("gdp", log_gdp_col), ("population", log_pop_col)]:
        x = df[col].to_numpy(dtype=float)
        x_std = (x - x.mean()) / x.std(ddof=0)
        blocks[block_name] = x_std.reshape(-1, 1)

    return blocks


def partial_r2_by_block(
    Y: np.ndarray,
    blocks: dict[str, np.ndarray],
) -> dict[str, float]:
    """Partial R^2 for each block, holding all other blocks in the model.

    Args:
        Y: (n_cities, n_axes) response matrix.
        blocks: {block_name: (n_cities, n_cols) design matrix}, one entry
            per predictor block (e.g. climate, colonial, gdp, population).

    Returns:
        {block_name: partial_r2, "r2_full": r2_full}
    """
    n = Y.shape[0]
    intercept = np.ones((n, 1))
    all_names = list(blocks.keys())

    X_full = np.column_stack([intercept] + [blocks[b] for b in all_names])
    sse_full = _ols_sse(X_full, Y)

    sst = float(np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2))
    r2_full = 1 - sse_full / sst if sst > 0 else np.nan

    result = {"r2_full": r2_full}
    for drop_name in all_names:
        keep = [b for b in all_names if b != drop_name]
        X_reduced = np.column_stack([intercept] + [blocks[b] for b in keep])
        sse_reduced = _ols_sse(X_reduced, Y)
        result[drop_name] = (1 - sse_full / sse_reduced) if sse_reduced > 0 else np.nan

    return result


def _init_bootstrap_worker(
    df: pd.DataFrame,
    composition_cols: list[str],
    climate_col: str,
    colonial_col: str,
    log_gdp_col: str,
    log_pop_col: str,
) -> None:
    """Pool initializer: stash the (fixed, per-bin) data once per worker
    process so each task only has to ship its small sampled-country list.

    Also pins this worker to single-threaded BLAS: the parallelism here
    comes from one resample per worker process, so each worker's own
    internal OpenBLAS threading (used by eigh/lstsq) would otherwise
    oversubscribe the machine's cores n_workers-fold and make the pool
    slower than running serially.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    threadpoolctl.threadpool_limits(1)
    _WORKER_STATE["df"] = df
    _WORKER_STATE["composition_cols"] = composition_cols
    _WORKER_STATE["climate_col"] = climate_col
    _WORKER_STATE["colonial_col"] = colonial_col
    _WORKER_STATE["log_gdp_col"] = log_gdp_col
    _WORKER_STATE["log_pop_col"] = log_pop_col


def _run_one_bootstrap(sampled_countries: np.ndarray) -> Optional[dict[str, float]]:
    """Single bootstrap resample: rebuild the resampled frame from the
    worker-local df and return this iteration's per-block partial R^2, or
    None if the resample is degenerate / the fit fails."""
    df = _WORKER_STATE["df"]
    boot_df = pd.concat(
        [df[df["country"] == c] for c in sampled_countries], ignore_index=True
    )
    if boot_df["country"].nunique() < 2 or len(boot_df) < MIN_CITIES:
        return None
    try:
        mat = boot_df[_WORKER_STATE["composition_cols"]].to_numpy(dtype=float)
        Y = pcoa_axes(mat, n_axes=N_PCA_AXES)
        blocks = build_design_blocks(
            boot_df,
            _WORKER_STATE["climate_col"],
            _WORKER_STATE["colonial_col"],
            _WORKER_STATE["log_gdp_col"],
            _WORKER_STATE["log_pop_col"],
        )
        return partial_r2_by_block(Y, blocks)
    except Exception:
        return None


def _cluster_bootstrap_ci(
    df: pd.DataFrame,
    composition_cols: list[str],
    climate_col: str,
    colonial_col: str,
    log_gdp_col: str,
    log_pop_col: str,
    n_boot: int,
    rng: np.random.Generator,
    n_workers: int = DEFAULT_N_WORKERS,
) -> dict[str, tuple[float, float]]:
    """Country-level cluster bootstrap 95% CI for each block's partial R^2.

    Each resample refits PCoA + the 4-block OLS from scratch, which is
    expensive at city counts in the thousands (dense eigendecomposition
    dominates), so the n_boot resamples are dispatched across a process
    pool. The country samples are pre-drawn sequentially from `rng` (not
    inside the workers) so the realised bootstrap distribution does not
    depend on the number of workers or task scheduling order.
    """
    countries = df["country"].unique()
    sampled_country_sets = [
        rng.choice(countries, size=len(countries), replace=True) for _ in range(n_boot)
    ]

    boot_results: dict[str, list[float]] = {}
    n_workers = max(1, min(n_workers, n_boot))

    ctx = get_context("spawn")
    with ctx.Pool(
        n_workers,
        initializer=_init_bootstrap_worker,
        initargs=(df, composition_cols, climate_col, colonial_col, log_gdp_col, log_pop_col),
    ) as pool:
        for r2 in pool.imap_unordered(_run_one_bootstrap, sampled_country_sets, chunksize=8):
            if r2 is None:
                continue
            for k, v in r2.items():
                boot_results.setdefault(k, []).append(v)

    ci = {}
    for k, vals in boot_results.items():
        vals = np.array([v for v in vals if np.isfinite(v)])
        if len(vals) < 10:
            ci[k] = (np.nan, np.nan)
        else:
            ci[k] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return ci


def regional_effect_by_bin(
    profiles: pd.DataFrame,
    covariates: pd.DataFrame,
    composition_cols: list[str],
    climate_col: str = "climate_zone",
    colonial_col: str = "colonial",
    log_gdp_col: str = "log_gdp",
    log_pop_col: str = "log_pop",
    n_boot: int = DEFAULT_N_BOOT,
    random_seed: int = RANDOM_SEED,
    compute_ci: bool = True,
) -> pd.DataFrame:
    """Partial R^2 of climate / colonial / GDP / population on the 17-type
    composition PCoA space, per radial bin.

    Args:
        profiles: one row per city x radial_bin, with `composition_cols`
            (17-type composition, need not be row-normalised) and a
            `radial_bin` column.
        covariates: one row per city, with `city`, `country`, and the
            climate/colonial/gdp/population columns.
        composition_cols: the 17 composition columns in `profiles`.
        n_boot: country-level cluster bootstrap resamples for the 95% CI.
        compute_ci: set False to skip the bootstrap (fast path for
            exploratory use; the shipped figures/tables use the default).

    Returns:
        One row per radial bin:
        radial_bin | n_cities | n_countries | n_pcoa_axes |
        partial_r2_climate | partial_r2_colonial | partial_r2_gdp |
        partial_r2_population | r2_full
        (+ *_ci_lo / *_ci_hi columns if compute_ci).
    """
    covariate_cols = {climate_col, colonial_col, log_gdp_col, log_pop_col}
    overlap = covariate_cols & set(profiles.columns) - {"city", "country"}
    if overlap:
        profiles = profiles.drop(columns=list(overlap))

    merged = profiles.merge(covariates, on=["city", "country"], how="inner")
    required = composition_cols + [climate_col, colonial_col, log_gdp_col, log_pop_col]
    merged = merged.dropna(subset=required)

    rng = np.random.default_rng(random_seed)
    rows = []

    for rbin in sorted(merged["radial_bin"].unique()):
        sub = merged[merged["radial_bin"] == rbin]
        if len(sub) < MIN_CITIES:
            continue

        mat = sub[composition_cols].to_numpy(dtype=float)
        Y = pcoa_axes(mat, n_axes=N_PCA_AXES)
        blocks = build_design_blocks(sub, climate_col, colonial_col, log_gdp_col, log_pop_col)
        r2 = partial_r2_by_block(Y, blocks)

        row = {
            "radial_bin": rbin,
            "n_cities": len(sub),
            "n_countries": sub["country"].nunique(),
            "n_pcoa_axes": N_PCA_AXES,
            "partial_r2_climate": r2["climate"],
            "partial_r2_colonial": r2["colonial"],
            "partial_r2_gdp": r2["gdp"],
            "partial_r2_population": r2["population"],
            "r2_full": r2["r2_full"],
        }

        if compute_ci:
            ci = _cluster_bootstrap_ci(
                sub, composition_cols, climate_col, colonial_col,
                log_gdp_col, log_pop_col, n_boot, rng,
            )
            for block in ["climate", "colonial", "gdp", "population", "r2_full"]:
                lo, hi = ci.get(block, (np.nan, np.nan))
                key = "r2_full" if block == "r2_full" else f"partial_r2_{block}"
                row[f"{key}_ci_lo"] = lo
                row[f"{key}_ci_hi"] = hi

        rows.append(row)
        log.info(
            f"  bin {rbin}: n={len(sub)}  "
            f"R2_climate={r2['climate']:.3f}  R2_colonial={r2['colonial']:.3f}  "
            f"R2_gdp={r2['gdp']:.3f}  R2_pop={r2['population']:.3f}  "
            f"R2_full={r2['r2_full']:.3f}"
        )

    return pd.DataFrame(rows)
