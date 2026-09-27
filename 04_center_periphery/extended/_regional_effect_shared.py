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
     multivariate response Y (n_cities x N_PCA_AXES). Computed ONCE per
     bin, on the complete-case sample — not recomputed inside the
     bootstrap (see below).
  2. Fit Y ~ climate + colonial + z(log GDP) + z(log population) by
     multivariate OLS (one model per PCoA axis, sharing the design
     matrix).
  3. For each predictor block g, partial R^2_g = 1 - SSE_full / SSE_reduced(-g),
     where SSE is summed over cities and PCoA axes, and the reduced model
     drops only block g, keeping every other block (population included,
     so the climate/colonial contribution is not confounded with city
     size).
  4. Confidence intervals via country-level cluster bootstrap: resample
     countries with replacement, then take the corresponding ROWS of the
     already-computed Y (PCoA scores) and covariates, and refit only the
     full/reduced OLS models on that resampled design. The PCoA space
     itself — the eigendecomposition of the full-sample Jensen-Shannon
     distance matrix — is NOT recomputed per resample.

     This treats the bootstrap as answering "how uncertain is the
     regression coefficient given this composition space", not "how
     uncertain is the composition space itself" — the latter would need
     the PCoA redone per resample, which is both an O(n^3) dense
     eigendecomposition (the dominant cost at bin sizes in the thousands,
     making the naive per-resample-PCoA design impractical to run
     thousands of times) and a different, broader question than what the
     regression CI is meant to convey here. Conditioning the bootstrap on
     the full-sample PCoA is the standard simplification for this case
     and is what the CI reported below quantifies.

Population is retained as an adjustment variable in every model
(full and every reduced model except the population-drop model), so the
climate/colonial contributions reported here are net of city size.
"""
import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)

N_PCA_AXES = 5
MIN_CITIES = 20
DEFAULT_N_BOOT = 2000
RANDOM_SEED = 42
MIN_VALID_BOOT_FRAC = 0.95  # fail loudly if fewer than this fraction of
                            # bootstrap replicates produce a usable fit


def pcoa_axes(composition: np.ndarray, n_axes: int = N_PCA_AXES,
              return_diagnostics: bool = False):
    """Principal Coordinate Analysis on the Jensen-Shannon distance matrix
    of row-normalised composition vectors.

    Args:
        composition: (n_cities, n_types) array; rows need not already sum
            to 1 (renormalised internally).
        n_axes: number of positive-eigenvalue axes to return.
        return_diagnostics: if True, also return a dict with
            n_positive_eigenvalues, variance_explained (fraction of the
            summed positive eigenvalues captured by the returned axes,
            per axis and cumulative).

    Returns:
        (n_cities, n_axes) array of PCoA scores, or (scores, diagnostics)
        if return_diagnostics. Axes beyond the number of positive
        eigenvalues are zero-filled.
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
        scores = np.zeros((n, n_axes))
    else:
        scores = eigvecs[:, :k] * np.sqrt(eigvals[:k])
        if k < n_axes:
            scores = np.hstack([scores, np.zeros((n, n_axes - k))])

    if not return_diagnostics:
        return scores

    pos_eigvals = eigvals[pos]
    total_pos = float(pos_eigvals.sum())
    per_axis = (eigvals[:k] / total_pos).tolist() if total_pos > 0 else []
    diagnostics = {
        "n_positive_eigenvalues": int(pos.sum()),
        "pcoa_variance_explained_per_axis": per_axis,
        "pcoa_variance_explained_cumulative": float(sum(per_axis)),
    }
    return scores, diagnostics


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
    return_rank: bool = False,
):
    """Partial R^2 for each block, holding all other blocks in the model.

    Args:
        Y: (n_cities, n_axes) response matrix.
        blocks: {block_name: (n_cities, n_cols) design matrix}, one entry
            per predictor block (e.g. climate, colonial, gdp, population).
        return_rank: if True, also return the rank of the full design
            matrix [intercept | all blocks] (via np.linalg.matrix_rank),
            for callers that want to flag rank-deficient fits (e.g. a
            bootstrap resample that drops a rare category entirely).

    Returns:
        {block_name: partial_r2, "r2_full": r2_full}, or
        ({...}, design_rank_full) if return_rank.
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

    if not return_rank:
        return result
    return result, int(np.linalg.matrix_rank(X_full))


def _cluster_bootstrap_ci(
    df: pd.DataFrame,
    Y: np.ndarray,
    climate_col: str,
    colonial_col: str,
    log_gdp_col: str,
    log_pop_col: str,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """Country-level cluster bootstrap 95% CI for each block's partial R^2.

    Conditions on the full-sample PCoA: `Y` (the PCoA scores) is fixed by
    the caller and only its ROWS are resampled here, along with the
    covariate rows in `df`. Each replicate re-derives the design blocks
    (dummy encoding can differ if a resample drops a category entirely)
    and refits full/reduced OLS on the resampled rows — no distance matrix
    or eigendecomposition is recomputed. This is O(n_boot * n * k^2)
    rather than O(n_boot * n^3), so it runs in seconds even for the
    largest bins (~2500 cities) instead of requiring a worker pool.

    Returns:
        (ci, failure_counts) — failure_counts breaks down how many of the
        n_boot replicates were dropped and why, so a low valid-replicate
        count is diagnosable rather than silently swallowed.
    """
    countries = df["country"].unique()
    n_index = np.arange(len(df))
    country_to_idx = {c: n_index[df["country"].values == c] for c in countries}

    boot_results: dict[str, list[float]] = {}
    failure_counts = {"degenerate_resample": 0, "fit_exception": 0}

    for _ in range(n_boot):
        sampled_countries = rng.choice(countries, size=len(countries), replace=True)
        idx = np.concatenate([country_to_idx[c] for c in sampled_countries])

        if len(set(sampled_countries.tolist())) < 2 or len(idx) < MIN_CITIES:
            failure_counts["degenerate_resample"] += 1
            continue

        boot_df = df.iloc[idx].reset_index(drop=True)
        Y_boot = Y[idx]
        try:
            blocks = build_design_blocks(
                boot_df, climate_col, colonial_col, log_gdp_col, log_pop_col
            )
            r2 = partial_r2_by_block(Y_boot, blocks)
        except Exception:
            failure_counts["fit_exception"] += 1
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
    return ci, failure_counts


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
        pcoa_variance_explained | design_rank_full |
        n_boot_requested | n_boot_valid |
        partial_r2_climate | partial_r2_colonial | partial_r2_gdp |
        partial_r2_population | r2_full
        (+ *_ci_lo / *_ci_hi columns if compute_ci).

    Raises:
        RuntimeError: if compute_ci and, for any bin, fewer than
            MIN_VALID_BOOT_FRAC of the requested bootstrap replicates
            produced a usable fit — a silently small effective bootstrap
            sample understates CI width without any visible signal, so
            this is treated as a hard failure rather than an NaN CI.
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
        sub = merged[merged["radial_bin"] == rbin].reset_index(drop=True)
        if len(sub) < MIN_CITIES:
            continue

        mat = sub[composition_cols].to_numpy(dtype=float)
        Y, pcoa_diag = pcoa_axes(mat, n_axes=N_PCA_AXES, return_diagnostics=True)
        blocks = build_design_blocks(sub, climate_col, colonial_col, log_gdp_col, log_pop_col)
        r2, design_rank = partial_r2_by_block(Y, blocks, return_rank=True)

        row = {
            "radial_bin": rbin,
            "n_cities": len(sub),
            "n_countries": sub["country"].nunique(),
            "n_pcoa_axes": pcoa_diag["n_positive_eigenvalues"] and min(N_PCA_AXES, pcoa_diag["n_positive_eigenvalues"]),
            "pcoa_variance_explained": round(pcoa_diag["pcoa_variance_explained_cumulative"], 4),
            "design_rank_full": design_rank,
            "n_boot_requested": n_boot if compute_ci else 0,
            "n_boot_valid": 0,
            "partial_r2_climate": r2["climate"],
            "partial_r2_colonial": r2["colonial"],
            "partial_r2_gdp": r2["gdp"],
            "partial_r2_population": r2["population"],
            "r2_full": r2["r2_full"],
        }

        if compute_ci:
            ci, failures = _cluster_bootstrap_ci(
                sub, Y, climate_col, colonial_col, log_gdp_col, log_pop_col,
                n_boot, rng,
            )
            n_valid = n_boot - sum(failures.values())
            row["n_boot_valid"] = n_valid
            valid_frac = n_valid / n_boot if n_boot > 0 else 0.0
            if valid_frac < MIN_VALID_BOOT_FRAC:
                raise RuntimeError(
                    f"regional_effect_by_bin: bin {rbin} — only {n_valid}/{n_boot} "
                    f"({valid_frac:.1%}) bootstrap replicates were valid "
                    f"(below the required {MIN_VALID_BOOT_FRAC:.0%}). "
                    f"Failure breakdown: {failures}."
                )
            for block in ["climate", "colonial", "gdp", "population", "r2_full"]:
                lo, hi = ci.get(block, (np.nan, np.nan))
                key = "r2_full" if block == "r2_full" else f"partial_r2_{block}"
                row[f"{key}_ci_lo"] = lo
                row[f"{key}_ci_hi"] = hi

        rows.append(row)
        log.info(
            f"  bin {rbin}: n={len(sub)}  n_boot_valid={row['n_boot_valid']}/{row['n_boot_requested']}  "
            f"pcoa_var_explained={row['pcoa_variance_explained']:.3f}  "
            f"R2_climate={r2['climate']:.3f}  R2_colonial={r2['colonial']:.3f}  "
            f"R2_gdp={r2['gdp']:.3f}  R2_pop={r2['population']:.3f}  "
            f"R2_full={r2['r2_full']:.3f}"
        )

    return pd.DataFrame(rows)
