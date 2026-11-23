"""
04_radial_assortativity.py
==========================
Q4 — Experiment C: 17-type spatial assortativity profile r_17(u).

For each city × radial_bin compute Newman categorical assortativity
using distance-based hex adjacency: cells are connected when their
centroid distance is within tol_factor x the estimated hex spacing (see
_build_edges_from_coords) — this approximates Queen contiguity but is a
different graph construction from core/04_radial_network_metrics.py's
polygon-based libpysal.Queen adjacency; the two modules' r17 values are
not computed on identical graphs.

Edge assignment: edge midpoint rule — an edge (i, j) is assigned to
the radial bin that contains  u_e = (u_i + u_j) / 2.

QC thresholds
-------------
  qc30  n_edges ≥ 30  AND  n_types ≥ 2   ← main analysis
  qc50  n_edges ≥ 50  AND  n_types ≥ 2   ← robustness (stricter)

Outputs
-------
  city_bin_assortativity.parquet
      city | country | tier | radial_bin | r17 | n_edges | n_types | qc30 | qc50

  tier_assortativity_profile.csv
      tier | radial_bin | n_cities | r17_mean | r17_se | r17_ci_lo | r17_ci_hi
      (qc30 cities only; country-cluster bootstrap CI)

  beta_r_profile.csv
      r17_i(u) ~ log(Pop_i)  per bin  →  β_r(u)
      radial_bin | n_cities | beta_r | beta_r_se | p_value
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

N_TYPES     = 17
MIN_EDGES_MAIN   = 30
MIN_EDGES_STRICT = 50
MIN_TYPES   = 2
N_BOOT      = 300
MIN_CITIES_REG = 10


# ── Newman assortativity ──────────────────────────────────────────────────

def _newman_assortativity(types_i: np.ndarray,
                          types_j: np.ndarray,
                          n_types: int = N_TYPES) -> float:
    """
    Newman 2003 categorical assortativity for one set of edges.
    r = (sum_k e_kk - sum_k a_k b_k) / (1 - sum_k a_k b_k)
    Returns NaN if denominator = 0.
    """
    m = len(types_i)
    if m == 0:
        return np.nan
    # mixing matrix e (n_types × n_types), symmetric, normalised
    e = np.zeros((n_types, n_types), dtype=float)
    for ti, tj in zip(types_i, types_j):
        ti -= 1; tj -= 1          # 0-indexed
        if 0 <= ti < n_types and 0 <= tj < n_types:
            e[ti, tj] += 1
            e[tj, ti] += 1
    total = e.sum()
    if total == 0:
        return np.nan
    e /= total
    a = e.sum(axis=1)   # row sums
    b = e.sum(axis=0)   # col sums
    tr_e = float(np.trace(e))
    sum_ab = float(np.dot(a, b))
    denom  = 1.0 - sum_ab
    if abs(denom) < 1e-12:
        return np.nan
    return float((tr_e - sum_ab) / denom)


# ── Distance-based hex adjacency (approximates Queen contiguity) ──────────

def _build_edges_from_coords(cx: np.ndarray, cy: np.ndarray,
                              tol_factor: float = 1.55) -> tuple:
    """
    Build adjacency edges for hex cells using 2D spatial proximity (NOT
    polygon-based libpysal.Queen — see module docstring). Cells are
    connected if their centroid distance <= tol_factor x hex_spacing.
    hex_spacing is estimated as the minimum non-zero inter-centroid distance.

    Returns (row_idx, col_idx) arrays of edge endpoints.
    """
    from scipy.spatial import cKDTree
    coords = np.column_stack([cx, cy])
    tree   = cKDTree(coords)

    # estimate hex spacing from nearest-neighbour distance
    dists, _ = tree.query(coords, k=2)
    nn_dist  = dists[:, 1]
    spacing  = float(np.median(nn_dist[nn_dist > 0]))
    radius   = tol_factor * spacing

    pairs = tree.query_pairs(r=radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return pairs[:, 0], pairs[:, 1]


def _cluster_boot_ci(values: np.ndarray, groups: np.ndarray,
                     B: int = N_BOOT, alpha: float = 0.05) -> tuple:
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
    cells = pd.read_parquet(CELLS_PQ,
                            columns=["city", "country", "label_k17",
                                     "cx", "cy", "u_scaled",
                                     "radial_bin", "tier"])
    cells = cells[cells["radial_bin"].notna()].copy()
    cells["radial_bin"] = cells["radial_bin"].astype(int)
    log.info(f"  {len(cells):,} cells.")

    log.info("Loading city_tiers.parquet ...")
    tiers = pd.read_parquet(TIERS_PQ, columns=["city", "country", "log_pop"])
    pop_map = tiers.set_index(["city", "country"])["log_pop"].to_dict()

    log.info("Loading bin_grid.json ...")
    with open(OUT_TABLES / "bin_grid.json") as f:
        grid = json.load(f)
    n_bins   = int(grid["n_bins"])
    u_edges  = np.array(grid["u_edges"])

    # ── per-city assortativity ────────────────────────────────────────────
    log.info("Computing assortativity per city × radial bin ...")
    records = []
    cities  = cells.groupby(["city", "country", "tier"])

    n_cities = cells.groupby(["city", "country"]).ngroups
    done = 0
    for (city, country, tier), city_df in cities:
        done += 1
        if done % 200 == 0:
            log.info(f"  {done} / {n_cities} cities ...")

        cx_arr = city_df["cx"].values.astype(float)
        cy_arr = city_df["cy"].values.astype(float)
        u_arr  = city_df["u_scaled"].values.astype(float)
        t_arr  = city_df["label_k17"].values.astype(int)
        rb_arr = city_df["radial_bin"].values.astype(int)

        if len(cx_arr) < 10:
            continue

        # build edges for entire city
        try:
            ei, ej = _build_edges_from_coords(cx_arr, cy_arr)
        except Exception:
            continue
        if len(ei) == 0:
            continue

        # edge midpoint in scaled distance
        u_mid = (u_arr[ei] + u_arr[ej]) / 2.0

        # assign each edge to a radial bin
        edge_bin = np.searchsorted(u_edges[1:], u_mid, side="right")
        edge_bin = np.clip(edge_bin, 0, n_bins - 1)

        # mask: only edges whose midpoint is within the common support
        within = (u_mid >= u_edges[0]) & (u_mid <= u_edges[-1])
        ei_w = ei[within]; ej_w = ej[within]; eb_w = edge_bin[within]

        log_pop = pop_map.get((city, country), np.nan)

        for rbin in range(n_bins):
            mask   = eb_w == rbin
            if mask.sum() == 0:
                continue
            ti = t_arr[ei_w[mask]]
            tj = t_arr[ej_w[mask]]
            n_edges = int(mask.sum())
            n_typ   = int(len(set(ti) | set(tj)))
            r17     = _newman_assortativity(ti, tj) \
                      if (n_edges >= MIN_EDGES_MAIN and n_typ >= MIN_TYPES) else np.nan
            records.append({
                "city":       city,
                "country":    country,
                "tier":       tier,
                "radial_bin": rbin,
                "r17":        r17,
                "n_edges":    n_edges,
                "n_types":    n_typ,
                "qc30":       int(n_edges >= 30 and n_typ >= MIN_TYPES),
                "qc50":       int(n_edges >= 50 and n_typ >= MIN_TYPES),
                "log_pop":    log_pop,
            })

    assort = pd.DataFrame(records)
    _path = OUT_TABLES / "city_bin_assortativity.parquet"
    assort.to_parquet(_path, index=False)
    _chk = pd.read_parquet(_path, columns=["city"])
    assert len(_chk) == len(assort), \
        f"Parquet read-back mismatch: wrote {len(assort)}, read {len(_chk)}"
    log.info(f"Saved + verified: city_bin_assortativity.parquet  ({len(assort):,} rows)")

    # ── tier × bin profiles ───────────────────────────────────────────────
    log.info("Computing tier × bin assortativity profiles ...")
    df_qc = assort[assort["qc30"] == 1].dropna(subset=["r17"])
    tier_rows = []
    for tier in ["T1", "T2", "T3", "T4"]:
        sub = df_qc[df_qc["tier"] == tier]
        for rbin in range(n_bins):
            sbin = sub[sub["radial_bin"] == rbin]
            if len(sbin) < 3:
                continue
            vals   = sbin["r17"].values
            groups = sbin["country"].values
            lo, hi = _cluster_boot_ci(vals, groups)
            tier_rows.append({
                "tier":       tier,
                "radial_bin": rbin,
                "n_cities":   len(sbin),
                "r17_mean":   float(np.nanmean(vals)),
                "r17_se":     float(np.nanstd(vals, ddof=1) / np.sqrt(len(vals))),
                "r17_ci_lo":  lo,
                "r17_ci_hi":  hi,
            })
    tier_df = pd.DataFrame(tier_rows)
    tier_df.to_csv(OUT_TABLES / "tier_assortativity_profile.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: tier_assortativity_profile.csv")

    # ── β_r(u): population effect per bin ────────────────────────────────
    log.info("Estimating β_r(u): OLS r17 ~ log(Pop) per bin ...")
    df_reg   = df_qc.dropna(subset=["log_pop", "r17"])
    beta_rows = []
    for rbin in range(n_bins):
        sbin = df_reg[df_reg["radial_bin"] == rbin]
        if len(sbin) < MIN_CITIES_REG:
            continue
        X = sm.add_constant(sbin["log_pop"].values)
        y = sbin["r17"].values
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
                "beta_r":     beta,
                "beta_r_se":  se,
                "p_value":    pval,
            })
        except Exception:
            continue

    beta_df = pd.DataFrame(beta_rows)
    beta_df.to_csv(OUT_TABLES / "beta_r_profile.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: beta_r_profile.csv")

    # ── summary ───────────────────────────────────────────────────────────
    print("\n=== r17 tier profiles (qc30 cities) ===")
    print(tier_df[["tier","radial_bin","n_cities","r17_mean"]].to_string(index=False))
    print("\n=== β_r(u) — population effect on r17 ===")
    if len(beta_df):
        print(beta_df.to_string(index=False))

    log.info("Done.")


if __name__ == "__main__":
    main()
