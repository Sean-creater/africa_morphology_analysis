"""
09_robustness.py
================
Q4 — Robustness checks for the two main findings.

Finding 1 (spatial division of labour): R_H = β_H_mid / β_H_core ≈ 1.58;
    R_r = β_r_core / β_r_mid ≈ 1.74.
Finding 2 (core convergence): C_core / C_mid ≈ 2.08;
    C_core / C_outer ≈ 2.26.

Three checks
------------
Check A — Leave-one-country-out (LOCO)
  Run 47 iterations, each dropping one country from the estimation.
  For each iteration compute R_H, R_r (spatial division), and the
  C_core/C_outer ratio (core convergence) from the remaining 46 countries.
  Report the min/median/max of each statistic across the 47 iterations.
  Robustness criterion: R_H > 1 AND R_r > 1 AND C_core/C_outer > 1 in
  all (or overwhelmingly all) iterations.

Check B — Alternative radial coordinate (q = dist_m / R98)
  Re-run the main OLS models (β_H per bin, β_r per bin, β_JS per bin)
  using q = dist_m / R98 as the radial coordinate instead of
  u = dist_m / Pop^0.4534, keeping the same 10-bin equal-width scheme.
  Compute zone-mean betas and verify the Core/Mid/Outer ordering:
    H17: β_H_mid > β_H_core  AND  β_H_mid > β_H_outer
    r17: β_r_core > β_r_mid  AND  β_r_core > β_r_outer
    JS : C_core > C_mid  AND  C_core > C_outer (C = |β_JS|)
  Report zone-mean betas and ratio estimates R_H(q), R_r(q), C_ratio(q).

Check C — Alternative centre definition (population centroid)
  Replace dist_m_primary (distance from primary morphological cluster)
  with dist_m_pop (distance from population-weighted centroid).
  Compute u_pop = dist_m_pop / Pop^α (same α = 0.4534), re-bin into
  10 equal-width bins (same P95 cutoff strategy), re-run the per-bin
  OLS models and report R_H, R_r, C_core/C_outer.

Zone definitions (matching 08_statistical_models.py)
-----------------------------------------------------
  Core  : bins 0–1
  Middle: bins 2–5
  Outer : bins 6–9

Outputs (all to outputs/tables/robustness/)
-------------------------------------------
  robustness_checkA_loco.csv
      statistic | n_iter | min | median | max | all_gt_threshold
  robustness_checkB_qcoord.csv
      outcome | zone | beta_mean | R_or_C_ratio
      + binary flags: pattern_preserved (all 3 ordering relations hold)
  robustness_checkC_popcenter.csv
      statistic | estimate_main | estimate_pop | diff | pct_change
  robustness_summary.csv
      check | criterion | result | note
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
ROBUST_DIR = OUT_TABLES / "robustness"

CELLS_PQ   = OUT_TABLES / "radial_cells.parquet"
TIERS_PQ   = OUT_TABLES / "city_tiers.parquet"

N_TYPES        = 17
MIN_OBS_REG    = 10
MIN_CTRY_REG   = 4
MIN_EDGES_MAIN = 30
MIN_TYPES_EDGE = 2

ALPHA = 0.4534   # same population-scaling exponent as 08

# Zone definitions — must match 08_statistical_models.py
ZONE_CORE  = [0, 1]
ZONE_MID   = [2, 3, 4, 5]
ZONE_OUTER = [6, 7, 8, 9]


# ── shared helpers ────────────────────────────────────────────────────────

def _ols_beta(df_bin: pd.DataFrame) -> tuple[float, float, float]:
    """
    Within-country demeaned OLS: Y ~ log_pop, country-clustered SE.
    Returns (beta, se, p_value); all NaN if insufficient data.
    """
    if len(df_bin) < MIN_OBS_REG or df_bin["country"].nunique() < MIN_CTRY_REG:
        return np.nan, np.nan, np.nan
    df = df_bin.copy()
    df["Y_dm"]       = df["Y"]       - df.groupby("country")["Y"].transform("mean")
    df["log_pop_dm"] = df["log_pop"] - df.groupby("country")["log_pop"].transform("mean")
    X = df[["log_pop_dm"]].values
    y = df["Y_dm"].values
    try:
        res = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
        cov = res.get_robustcov_results(cov_type="cluster",
                                        groups=df["country"].values)
        beta = float(cov.params[1])
        se   = float(cov.bse[1])
        t    = beta / se if se > 0 else np.nan
        G    = df["country"].nunique()
        pval = float(2 * sp_stats.t.sf(abs(t), df=G - 1))
        return beta, se, pval
    except Exception:
        return np.nan, np.nan, np.nan


def _newman_assortativity(types_i, types_j, n_types=N_TYPES) -> float:
    e = np.zeros((n_types, n_types), float)
    for ti, tj in zip(types_i, types_j):
        ti -= 1; tj -= 1
        if 0 <= ti < n_types and 0 <= tj < n_types:
            e[ti, tj] += 1; e[tj, ti] += 1
    total = e.sum()
    if total == 0:
        return np.nan
    e /= total
    a = e.sum(axis=1)
    b = e.sum(axis=0)
    tr_e = float(np.trace(e))
    sab  = float(np.dot(a, b))
    d    = 1.0 - sab
    return np.nan if abs(d) < 1e-12 else float((tr_e - sab) / d)


def _build_edges(cx, cy, tol=1.55):
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([cx, cy]))
    dists, _ = tree.query(np.column_stack([cx, cy]), k=2)
    spacing  = float(np.median(dists[:, 1][dists[:, 1] > 0]))
    pairs    = tree.query_pairs(r=tol * spacing, output_type="ndarray")
    if len(pairs) == 0:
        return np.array([], int), np.array([], int)
    return pairs[:, 0], pairs[:, 1]


def _H17_rarefied(labels: np.ndarray, n: int = 20, reps: int = 50,
                  rng=None) -> float:
    if rng is None:
        rng = np.random.default_rng()
    if len(labels) < n:
        return np.nan
    h_vals = []
    for _ in range(reps):
        samp   = rng.choice(labels, size=n, replace=False)
        counts = np.bincount(samp, minlength=N_TYPES + 1)[1:N_TYPES + 1].astype(float)
        counts /= counts.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.nansum(counts * np.log(counts + 1e-15)) / np.log(N_TYPES)
        h_vals.append(float(h))
    return float(np.mean(h_vals))


def _zmean(beta_dict: dict, bins: list[int]) -> float:
    return float(np.nanmean([beta_dict.get(b, np.nan) for b in bins]))


def _fit_per_bin(city_bin_df: pd.DataFrame, n_bins: int,
                 outcome_col: str) -> dict[int, float]:
    """
    Fit OLS per radial_bin; return {bin: beta} dict.
    city_bin_df must have columns: city, country, log_pop, radial_bin, <outcome_col>.
    """
    city_bin_df = city_bin_df.rename(columns={outcome_col: "Y"})
    betas = {}
    for b in range(n_bins):
        sbin = city_bin_df[city_bin_df["radial_bin"] == b]
        beta, _, _ = _ols_beta(sbin)
        betas[b] = beta
    return betas


# ── Check A: Leave-one-country-out ────────────────────────────────────────

def checkA_loco(city_bin_H: pd.DataFrame, city_bin_r: pd.DataFrame,
                city_bin_JS: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """
    47 leave-one-country-out iterations.

    For each, compute:
      R_H   = beta_H_mid / beta_H_core       (spatial division, H17)
      R_r   = beta_r_core / beta_r_mid       (spatial division, r17)
      C_ratio = C_core / C_outer  where C = |beta_JS|  (core convergence)

    Reports min/median/max and whether all iterations satisfy threshold > 1.
    """
    log.info("Check A: Leave-one-country-out (%d iterations) ...",
             city_bin_H["country"].nunique())

    countries = sorted(city_bin_H["country"].unique())
    detail_rows = []

    for ctry in countries:
        dH = city_bin_H[city_bin_H["country"] != ctry]
        dr = city_bin_r[city_bin_r["country"] != ctry]
        dJ = city_bin_JS[city_bin_JS["country"] != ctry]

        bH = _fit_per_bin(dH.copy(), n_bins, "Y")
        br = _fit_per_bin(dr.copy(), n_bins, "Y")
        bJ = _fit_per_bin(dJ.copy(), n_bins, "Y")

        h_core  = _zmean(bH, ZONE_CORE)
        h_mid   = _zmean(bH, ZONE_MID)
        r_core  = _zmean(br, ZONE_CORE)
        r_mid   = _zmean(br, ZONE_MID)
        c_core  = abs(_zmean(bJ, ZONE_CORE))
        c_outer = abs(_zmean(bJ, ZONE_OUTER))

        R_H = h_mid  / h_core  if (not np.isnan(h_core)  and h_core  > 0) else np.nan
        R_r = r_core / r_mid   if (not np.isnan(r_mid)   and r_mid   > 0) else np.nan
        C_r = c_core / c_outer if (not np.isnan(c_outer) and c_outer > 0) else np.nan

        detail_rows.append({
            "country_omitted": ctry,
            "R_H": R_H, "R_r": R_r, "C_core_C_outer": C_r,
        })

    detail_df = pd.DataFrame(detail_rows)
    R_H_vals   = detail_df["R_H"].tolist()
    R_r_vals   = detail_df["R_r"].tolist()
    C_ratio_vals = detail_df["C_core_C_outer"].tolist()

    rows = []
    for stat, vals, threshold in [
        ("R_H",           R_H_vals,     1.0),
        ("R_r",           R_r_vals,     1.0),
        ("C_core/C_outer", C_ratio_vals, 1.0),
    ]:
        arr  = np.array([v for v in vals if not np.isnan(v)])
        rows.append({
            "statistic":       stat,
            "n_iter":          len(arr),
            "min":             float(np.min(arr))    if len(arr) > 0 else np.nan,
            "median":          float(np.median(arr)) if len(arr) > 0 else np.nan,
            "max":             float(np.max(arr))    if len(arr) > 0 else np.nan,
            "all_gt_threshold": bool(np.all(arr > threshold)),
            "pct_gt_threshold": float(100.0 * np.mean(arr > threshold))
                                if len(arr) > 0 else np.nan,
        })

    return pd.DataFrame(rows), detail_df


# ── Check B: q-coordinate (dist_m / R98) ──────────────────────────────────

def checkB_qcoord(cells: pd.DataFrame, tiers: pd.DataFrame,
                  grid: dict, n_bins: int) -> tuple[pd.DataFrame, dict]:
    """
    Re-run β_H, β_r, β_JS per bin under q = dist_m / R98 coordinate.
    Uses pre-computed radial_bin_q column and q_relative column in cells.

    Returns per-zone summary DataFrame and a dict with pattern-preservation flags.
    """
    log.info("Check B: q-coordinate robustness ...")

    pop_map = tiers.set_index(["city", "country"])["log_pop"].to_dict()
    rng     = np.random.default_rng(2025)

    # Filter to cells with valid q-bins
    cq = cells[cells["radial_bin_q"].notna()].copy()
    cq["radial_bin"] = cq["radial_bin_q"].astype(int)

    # ── H17 per city × bin ──
    div_records = []
    for (city, country, tier, rbin), grp in cq.groupby(
            ["city", "country", "tier", "radial_bin"]):
        labels = grp["label_k17"].values.astype(int)
        h_r    = _H17_rarefied(labels, n=20, reps=50, rng=rng)
        div_records.append({
            "city": city, "country": country,
            "radial_bin": int(rbin),
            "Y": h_r,
            "log_pop": pop_map.get((city, country), np.nan),
        })
    div_q = pd.DataFrame(div_records).dropna(subset=["Y", "log_pop"])

    # ── r17 per city × bin (q-bins, edge midpoint) ──
    q_edges_arr = np.array(grid["q_edges"])
    assort_records = []
    for (city, country, tier), city_df in cq.groupby(["city", "country", "tier"]):
        cx = city_df["cx"].values.astype(float)
        cy = city_df["cy"].values.astype(float)
        q  = city_df["q_relative"].values.astype(float)
        t  = city_df["label_k17"].values.astype(int)
        if len(cx) < 10:
            continue
        try:
            ei, ej = _build_edges(cx, cy)
        except Exception:
            continue
        if len(ei) == 0:
            continue
        q_mid   = (q[ei] + q[ej]) / 2.0
        within  = (q_mid >= q_edges_arr[0]) & (q_mid <= q_edges_arr[-1])
        ei_w = ei[within]; ej_w = ej[within]; q_mid_w = q_mid[within]
        edge_bin = np.clip(
            np.searchsorted(q_edges_arr[1:], q_mid_w, side="right"),
            0, n_bins - 1)
        log_pop = pop_map.get((city, country), np.nan)
        for rb in range(n_bins):
            mask = edge_bin == rb
            if mask.sum() == 0:
                continue
            ti = t[ei_w[mask]]; tj = t[ej_w[mask]]
            n_e = int(mask.sum()); n_ty = len(set(ti) | set(tj))
            r17 = _newman_assortativity(ti, tj) \
                  if n_e >= MIN_EDGES_MAIN and n_ty >= MIN_TYPES_EDGE else np.nan
            assort_records.append({
                "city": city, "country": country,
                "radial_bin": rb, "Y": r17, "log_pop": log_pop,
            })
    assort_q = pd.DataFrame(assort_records).dropna(subset=["Y", "log_pop"])

    # ── JS divergence per city × bin (q-bins) ──
    # Recompute from cells: type distribution per city×bin (q), JS to global centroid.

    # Compute city×bin (q) type distributions
    type_cols = [f"t{k}" for k in range(1, N_TYPES + 1)]
    js_records = []
    for (city, country, tier, rbin), grp in cq.groupby(
            ["city", "country", "tier", "radial_bin"]):
        labels = grp["label_k17"].values.astype(int)
        if len(labels) < 5:
            continue
        counts = np.bincount(labels, minlength=N_TYPES + 1)[1:N_TYPES + 1].astype(float)
        total  = counts.sum()
        if total == 0:
            continue
        row = {"city": city, "country": country, "radial_bin": int(rbin),
               "log_pop": pop_map.get((city, country), np.nan)}
        for k in range(N_TYPES):
            row[type_cols[k]] = counts[k] / total
        js_records.append(row)
    js_df = pd.DataFrame(js_records).dropna(subset=["log_pop"])

    # Global centroid per bin (q): average composition across all cities
    js_beta_q = {}
    for rb in range(n_bins):
        sbin = js_df[js_df["radial_bin"] == rb]
        if len(sbin) < MIN_OBS_REG:
            js_beta_q[rb] = np.nan
            continue
        # Global centroid = mean type distribution in this bin (equal city weight)
        centroid = sbin[type_cols].mean().values
        # JS divergence of each city from global centroid
        def js_div(p, q_c):
            p = p + 1e-12; p /= p.sum()
            q_c = q_c + 1e-12; q_c /= q_c.sum()
            m   = 0.5 * (p + q_c)
            with np.errstate(divide="ignore", invalid="ignore"):
                kl1 = np.nansum(p   * np.log(p   / m))
                kl2 = np.nansum(q_c * np.log(q_c / m))
            return 0.5 * (kl1 + kl2)

        js_vals = sbin[type_cols].apply(
            lambda row: js_div(row.values.copy(), centroid.copy()), axis=1)
        sbin2 = sbin.copy()
        sbin2["Y"] = js_vals.values
        beta, _, _ = _ols_beta(sbin2)
        js_beta_q[rb] = beta

    # ── Per-bin OLS for H17 and r17 ──
    h_betas = {}
    for rb in range(n_bins):
        sbin = div_q[div_q["radial_bin"] == rb]
        beta, _, _ = _ols_beta(sbin)
        h_betas[rb] = beta

    r_betas = {}
    for rb in range(n_bins):
        sbin = assort_q[assort_q["radial_bin"] == rb]
        beta, _, _ = _ols_beta(sbin)
        r_betas[rb] = beta

    # ── Zone-mean betas ──
    h_core  = _zmean(h_betas, ZONE_CORE)
    h_mid   = _zmean(h_betas, ZONE_MID)
    h_outer = _zmean(h_betas, ZONE_OUTER)
    r_core  = _zmean(r_betas, ZONE_CORE)
    r_mid   = _zmean(r_betas, ZONE_MID)
    r_outer = _zmean(r_betas, ZONE_OUTER)
    c_core  = abs(_zmean(js_beta_q, ZONE_CORE))
    c_mid   = abs(_zmean(js_beta_q, ZONE_MID))
    c_outer = abs(_zmean(js_beta_q, ZONE_OUTER))

    R_H_q  = h_mid  / h_core  if (not np.isnan(h_core)  and h_core  > 0) else np.nan
    R_r_q  = r_core / r_mid   if (not np.isnan(r_mid)   and r_mid   > 0) else np.nan
    C_rat_q = c_core / c_outer if (not np.isnan(c_outer) and c_outer > 0) else np.nan

    rows = [
        {"outcome": "H17", "zone": "core",  "beta_mean": h_core,  "ratio_name": "R_H",  "ratio": R_H_q},
        {"outcome": "H17", "zone": "mid",   "beta_mean": h_mid,   "ratio_name": "R_H",  "ratio": R_H_q},
        {"outcome": "H17", "zone": "outer", "beta_mean": h_outer, "ratio_name": "R_H",  "ratio": R_H_q},
        {"outcome": "r17", "zone": "core",  "beta_mean": r_core,  "ratio_name": "R_r",  "ratio": R_r_q},
        {"outcome": "r17", "zone": "mid",   "beta_mean": r_mid,   "ratio_name": "R_r",  "ratio": R_r_q},
        {"outcome": "r17", "zone": "outer", "beta_mean": r_outer, "ratio_name": "R_r",  "ratio": R_r_q},
        {"outcome": "JS",  "zone": "core",  "beta_mean": c_core,  "ratio_name": "C_core/C_outer", "ratio": C_rat_q},
        {"outcome": "JS",  "zone": "mid",   "beta_mean": c_mid,   "ratio_name": "C_core/C_outer", "ratio": C_rat_q},
        {"outcome": "JS",  "zone": "outer", "beta_mean": c_outer, "ratio_name": "C_core/C_outer", "ratio": C_rat_q},
    ]

    flags = {
        "H_mid_gt_core":  bool(h_mid  > h_core),
        "H_mid_gt_outer": bool(h_mid  > h_outer),
        "r_core_gt_mid":  bool(r_core > r_mid),
        "r_core_gt_outer":bool(r_core > r_outer),
        "C_core_gt_outer":bool(c_core > c_outer),
        "R_H_q": round(R_H_q,  4) if not np.isnan(R_H_q)  else np.nan,
        "R_r_q": round(R_r_q,  4) if not np.isnan(R_r_q)  else np.nan,
        "C_ratio_q": round(C_rat_q, 4) if not np.isnan(C_rat_q) else np.nan,
    }

    return pd.DataFrame(rows), flags


# ── Check C: Population-centroid distance ─────────────────────────────────

def checkC_popcenter(cells: pd.DataFrame, tiers: pd.DataFrame,
                     grid: dict, n_bins: int) -> pd.DataFrame:
    """
    Replace dist_m_primary with dist_m_pop (population-weighted centroid).
    Compute u_pop = dist_m_pop / pop^ALPHA, bin into n_bins equal-width bins
    using the same P95 scheme, re-run per-bin OLS for H17, r17, JS.
    Report R_H, R_r, C_core/C_outer vs. main estimates.
    """
    log.info("Check C: population centroid alternative ...")

    pop_map = tiers.set_index(["city", "country"])["log_pop"].to_dict()
    rng     = np.random.default_rng(2025)

    # Compute u_pop
    cells2 = cells[cells["dist_m_pop"].notna()].copy()
    cells2["pop_scaled"] = np.exp(
        cells2.apply(lambda r: pop_map.get((r["city"], r["country"]), np.nan), axis=1)
    )
    cells2["u_pop"] = cells2["dist_m_pop"] / (cells2["pop_scaled"] ** ALPHA)

    # Build bin edges from u_pop P95
    u_pop_vals = cells2["u_pop"].dropna()
    u_pop_p95  = float(np.percentile(u_pop_vals, 95))
    u_pop_edges = np.linspace(0.0, u_pop_p95, n_bins + 1)

    # Assign radial_bin_pop
    cells2 = cells2[(cells2["u_pop"] >= 0) & (cells2["u_pop"] <= u_pop_p95)].copy()
    cells2["radial_bin_pop"] = np.clip(
        np.searchsorted(u_pop_edges[1:], cells2["u_pop"].values, side="right"),
        0, n_bins - 1
    ).astype(int)

    # ── H17 per city × bin (pop-centroid) ──
    div_records = []
    for (city, country, tier, rbin), grp in cells2.groupby(
            ["city", "country", "tier", "radial_bin_pop"]):
        labels = grp["label_k17"].values.astype(int)
        h_r    = _H17_rarefied(labels, n=20, reps=50, rng=rng)
        div_records.append({
            "city": city, "country": country,
            "radial_bin": int(rbin),
            "Y": h_r,
            "log_pop": pop_map.get((city, country), np.nan),
        })
    div_pop = pd.DataFrame(div_records).dropna(subset=["Y", "log_pop"])

    # ── r17 per city × bin (pop-centroid edges) ──
    assort_records = []
    for (city, country, tier), city_df in cells2.groupby(["city", "country", "tier"]):
        cx = city_df["cx"].values.astype(float)
        cy = city_df["cy"].values.astype(float)
        up = city_df["u_pop"].values.astype(float)
        t  = city_df["label_k17"].values.astype(int)
        if len(cx) < 10:
            continue
        try:
            ei, ej = _build_edges(cx, cy)
        except Exception:
            continue
        if len(ei) == 0:
            continue
        up_mid  = (up[ei] + up[ej]) / 2.0
        within  = (up_mid >= u_pop_edges[0]) & (up_mid <= u_pop_edges[-1])
        ei_w = ei[within]; ej_w = ej[within]; up_mid_w = up_mid[within]
        edge_bin = np.clip(
            np.searchsorted(u_pop_edges[1:], up_mid_w, side="right"),
            0, n_bins - 1)
        log_pop = pop_map.get((city, country), np.nan)
        for rb in range(n_bins):
            mask = edge_bin == rb
            if mask.sum() == 0:
                continue
            ti = t[ei_w[mask]]; tj = t[ej_w[mask]]
            n_e = int(mask.sum()); n_ty = len(set(ti) | set(tj))
            r17 = _newman_assortativity(ti, tj) \
                  if n_e >= MIN_EDGES_MAIN and n_ty >= MIN_TYPES_EDGE else np.nan
            assort_records.append({
                "city": city, "country": country,
                "radial_bin": rb, "Y": r17, "log_pop": log_pop,
            })
    assort_pop = pd.DataFrame(assort_records).dropna(subset=["Y", "log_pop"])

    # ── JS divergence per city × bin (pop-centroid) ──
    type_cols = [f"t{k}" for k in range(1, N_TYPES + 1)]
    js_records = []
    for (city, country, tier, rbin), grp in cells2.groupby(
            ["city", "country", "tier", "radial_bin_pop"]):
        labels = grp["label_k17"].values.astype(int)
        if len(labels) < 5:
            continue
        counts = np.bincount(labels, minlength=N_TYPES + 1)[1:N_TYPES + 1].astype(float)
        total  = counts.sum()
        if total == 0:
            continue
        row = {"city": city, "country": country, "radial_bin": int(rbin),
               "log_pop": pop_map.get((city, country), np.nan)}
        for k in range(N_TYPES):
            row[type_cols[k]] = counts[k] / total
        js_records.append(row)
    js_df = pd.DataFrame(js_records).dropna(subset=["log_pop"])

    def js_div(p, q_c):
        p = p + 1e-12; p /= p.sum()
        q_c = q_c + 1e-12; q_c /= q_c.sum()
        m   = 0.5 * (p + q_c)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl1 = np.nansum(p   * np.log(p   / m))
            kl2 = np.nansum(q_c * np.log(q_c / m))
        return 0.5 * (kl1 + kl2)

    js_betas = {}
    for rb in range(n_bins):
        sbin = js_df[js_df["radial_bin"] == rb]
        if len(sbin) < MIN_OBS_REG:
            js_betas[rb] = np.nan
            continue
        centroid = sbin[type_cols].mean().values
        js_vals  = sbin[type_cols].apply(
            lambda row: js_div(row.values.copy(), centroid.copy()), axis=1)
        sbin2 = sbin.copy()
        sbin2["Y"] = js_vals.values
        beta, _, _ = _ols_beta(sbin2)
        js_betas[rb] = beta

    # ── Zone-mean betas ──
    h_betas = {}
    for rb in range(n_bins):
        sbin = div_pop[div_pop["radial_bin"] == rb]
        beta, _, _ = _ols_beta(sbin)
        h_betas[rb] = beta

    r_betas = {}
    for rb in range(n_bins):
        sbin = assort_pop[assort_pop["radial_bin"] == rb]
        beta, _, _ = _ols_beta(sbin)
        r_betas[rb] = beta

    h_core  = _zmean(h_betas, ZONE_CORE)
    h_mid   = _zmean(h_betas, ZONE_MID)
    r_core  = _zmean(r_betas, ZONE_CORE)
    r_mid   = _zmean(r_betas, ZONE_MID)
    c_core  = abs(_zmean(js_betas, ZONE_CORE))
    c_outer = abs(_zmean(js_betas, ZONE_OUTER))

    R_H_pop  = h_mid  / h_core  if (not np.isnan(h_core)  and h_core  > 0) else np.nan
    R_r_pop  = r_core / r_mid   if (not np.isnan(r_mid)   and r_mid   > 0) else np.nan
    C_rat_pop = c_core / c_outer if (not np.isnan(c_outer) and c_outer > 0) else np.nan

    # Main estimates from formal_models.csv
    main_h = pd.read_csv(OUT_TABLES / "formal_models.csv")
    main_h = main_h[main_h["outcome"] == "H17"].set_index("radial_bin")["beta"].to_dict()
    main_r = pd.read_csv(OUT_TABLES / "formal_models.csv")
    main_r = main_r[main_r["outcome"] == "r17"].set_index("radial_bin")["beta"].to_dict()
    main_JS = pd.read_csv(OUT_TABLES / "formal_models.csv")
    main_JS = main_JS[main_JS["outcome"] == "JS"].set_index("radial_bin")["beta"].to_dict()

    mH_core  = _zmean(main_h,  ZONE_CORE)
    mH_mid   = _zmean(main_h,  ZONE_MID)
    mr_core  = _zmean(main_r,  ZONE_CORE)
    mr_mid   = _zmean(main_r,  ZONE_MID)
    mC_core  = abs(_zmean(main_JS, ZONE_CORE))
    mC_outer = abs(_zmean(main_JS, ZONE_OUTER))

    R_H_main  = mH_mid  / mH_core  if (not np.isnan(mH_core)  and mH_core  > 0) else np.nan
    R_r_main  = mr_core / mr_mid   if (not np.isnan(mr_mid)   and mr_mid   > 0) else np.nan
    C_rat_main = mC_core / mC_outer if (not np.isnan(mC_outer) and mC_outer > 0) else np.nan

    rows = []
    for stat, est_main, est_pop in [
        ("R_H",           R_H_main,   R_H_pop),
        ("R_r",           R_r_main,   R_r_pop),
        ("C_core/C_outer", C_rat_main, C_rat_pop),
    ]:
        diff     = est_pop - est_main if not (np.isnan(est_main) or np.isnan(est_pop)) else np.nan
        pct_chg  = 100.0 * diff / est_main if (not np.isnan(diff) and est_main != 0) else np.nan
        rows.append({
            "statistic":     stat,
            "estimate_main": est_main,
            "estimate_pop":  est_pop,
            "diff":          diff,
            "pct_change":    pct_chg,
        })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    ROBUST_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading radial_cells.parquet ...")
    cells = pd.read_parquet(
        CELLS_PQ,
        columns=["city", "country", "label_k17", "tier",
                 "cx", "cy",
                 "dist_m_primary", "dist_m_pop",
                 "u_scaled", "q_relative",
                 "radial_bin", "radial_bin_q"])
    log.info("  %d cells.", len(cells))

    log.info("Loading city_tiers.parquet ...")
    tiers = pd.read_parquet(TIERS_PQ,
                            columns=["city", "country", "log_pop", "tier"])

    log.info("Loading bin_grid.json ...")
    with open(OUT_TABLES / "bin_grid.json") as f:
        grid = json.load(f)
    n_bins = int(grid["n_bins"])

    pop_map = tiers.set_index(["city", "country"])["log_pop"].to_dict()

    # ── Build city×bin dataframes from main (u-coordinate) ────────────────
    log.info("Building city×bin summaries (u-coordinate) ...")
    rng = np.random.default_rng(2025)
    cells_u = cells[cells["radial_bin"].notna()].copy()
    cells_u["radial_bin"] = cells_u["radial_bin"].astype(int)

    # H17 city×bin
    div_records = []
    for (city, country, tier, rbin), grp in cells_u.groupby(
            ["city", "country", "tier", "radial_bin"]):
        labels = grp["label_k17"].values.astype(int)
        h_r    = _H17_rarefied(labels, n=20, reps=50, rng=rng)
        div_records.append({
            "city": city, "country": country,
            "radial_bin": int(rbin),
            "Y": h_r,
            "log_pop": pop_map.get((city, country), np.nan),
        })
    city_bin_H = pd.DataFrame(div_records).dropna(subset=["Y", "log_pop"])

    # r17 city×bin (reuse from parquet, qc30 filter)
    assort_pq = OUT_TABLES / "city_bin_assortativity.parquet"
    if assort_pq.exists():
        cb_r = pd.read_parquet(assort_pq,
                               columns=["city", "country", "radial_bin", "r17", "qc30"])
        cb_r = cb_r[cb_r["qc30"] == 1].rename(columns={"r17": "Y"})
        cb_r = cb_r.merge(tiers[["city", "country", "log_pop"]],
                          on=["city", "country"], how="left")
        city_bin_r = cb_r.dropna(subset=["Y", "log_pop"])
    else:
        log.warning("city_bin_assortativity.parquet not found; r17 check A/C will be limited.")
        city_bin_r = pd.DataFrame(columns=["city","country","radial_bin","Y","log_pop"])

    # JS city×bin — load from precomputed city_bin_divergence.parquet
    div_js_pq = OUT_TABLES / "city_bin_divergence.parquet"
    if div_js_pq.exists():
        cb_js = pd.read_parquet(div_js_pq,
                                columns=["city", "country", "radial_bin", "JS_to_centroid"])
        cb_js = cb_js.rename(columns={"JS_to_centroid": "Y"})
        cb_js = cb_js.merge(tiers[["city", "country", "log_pop"]],
                            on=["city", "country"], how="left")
        city_bin_JS = cb_js.dropna(subset=["Y", "log_pop"])
    else:
        log.warning("city_bin_divergence.parquet not found; JS check A will be limited.")
        city_bin_JS = pd.DataFrame(columns=["city","country","radial_bin","Y","log_pop"])

    # ── Check A ───────────────────────────────────────────────────────────
    cA, cA_detail = checkA_loco(city_bin_H, city_bin_r, city_bin_JS, n_bins)
    cA.to_csv(ROBUST_DIR / "robustness_checkA_loco.csv",
              index=False, float_format="%.6f")
    cA_detail.to_csv(ROBUST_DIR / "robustness_checkA_loco_detail.csv",
                     index=False, float_format="%.6f")
    log.info("Saved: robustness_checkA_loco.csv + _detail.csv")

    # ── Check B ───────────────────────────────────────────────────────────
    cB, flagsB = checkB_qcoord(cells, tiers, grid, n_bins)
    cB.to_csv(ROBUST_DIR / "robustness_checkB_qcoord.csv",
              index=False, float_format="%.6f")
    log.info("Saved: robustness_checkB_qcoord.csv")

    # ── Check C ───────────────────────────────────────────────────────────
    cC = checkC_popcenter(cells, tiers, grid, n_bins)
    cC.to_csv(ROBUST_DIR / "robustness_checkC_popcenter.csv",
              index=False, float_format="%.6f")
    log.info("Saved: robustness_checkC_popcenter.csv")

    # ── Summary ───────────────────────────────────────────────────────────
    summary_rows = []

    # Check A summary
    for _, row in cA.iterrows():
        thr = "> 1"
        summary_rows.append({
            "check":    "A_loco",
            "criterion": f"{row['statistic']} {thr} in all {row['n_iter']} iterations",
            "result":   str(row["all_gt_threshold"]),
            "note":     f"min={row['min']:.3f}, med={row['median']:.3f}, "
                        f"max={row['max']:.3f}, pct_pass={row['pct_gt_threshold']:.1f}%",
        })

    # Check B summary
    for flag_name, flag_val in [
        ("H_mid_gt_core",  flagsB["H_mid_gt_core"]),
        ("H_mid_gt_outer", flagsB["H_mid_gt_outer"]),
        ("r_core_gt_mid",  flagsB["r_core_gt_mid"]),
        ("r_core_gt_outer",flagsB["r_core_gt_outer"]),
        ("C_core_gt_outer",flagsB["C_core_gt_outer"]),
    ]:
        summary_rows.append({
            "check":    "B_qcoord",
            "criterion": flag_name,
            "result":   str(flag_val),
            "note":     f"R_H(q)={flagsB['R_H_q']}, "
                        f"R_r(q)={flagsB['R_r_q']}, "
                        f"C_ratio(q)={flagsB['C_ratio_q']}",
        })

    # Check C summary
    for _, row in cC.iterrows():
        summary_rows.append({
            "check":    "C_popcenter",
            "criterion": f"{row['statistic']}: pop-centroid vs primary",
            "result":   f"{row['estimate_pop']:.4f}",
            "note":     f"main={row['estimate_main']:.4f}, "
                        f"pop={row['estimate_pop']:.4f}, "
                        f"pct_change={row['pct_change']:.1f}%",
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ROBUST_DIR / "robustness_summary.csv",
                   index=False, float_format="%.4f")
    log.info("Saved: robustness_summary.csv")

    # ── Print results ─────────────────────────────────────────────────────
    print("\n=== Check A: Leave-one-country-out ===")
    print(cA.to_string(index=False))

    print("\n=== Check B: q-coordinate zone contrasts ===")
    print(f"  R_H(q)     = {flagsB['R_H_q']} "
          f"({'H_mid>H_core: '+str(flagsB['H_mid_gt_core'])})")
    print(f"  R_r(q)     = {flagsB['R_r_q']} "
          f"({'r_core>r_mid: '+str(flagsB['r_core_gt_mid'])})")
    print(f"  C_ratio(q) = {flagsB['C_ratio_q']} "
          f"({'C_core>C_outer: '+str(flagsB['C_core_gt_outer'])})")
    print(f"  All 5 ordering flags: "
          f"{all(flagsB[k] for k in ['H_mid_gt_core','H_mid_gt_outer','r_core_gt_mid','r_core_gt_outer','C_core_gt_outer'])}")

    print("\n=== Check C: Population-centroid alternative ===")
    print(cC.to_string(index=False))

    log.info("Done.")


if __name__ == "__main__":
    main()
