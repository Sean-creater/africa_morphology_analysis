"""
01_compute_city_metrics.py
==========================
Compute city-level morphological metrics for urban hierarchy analysis.

For each city (n≈6918) this script computes:
  - p1..p4   : share of grid cells belonging to each higher-order morphology group
  - H_star   : normalised Shannon entropy  H* = H / log(4),  H* ∈ [0,1]
               0 = single group dominates, 1 = all four groups equal
  - r        : Newman categorical spatial assortativity of the four groups
               computed from Queen adjacency built directly from SHP geometry
               (bypasses cached cell graphs, which have incomplete edge sets
               for ~57% of cities due to an upstream CRS issue)
               r > 0 → same-group cells cluster spatially (segregation)
               r ≈ 0 → random mixing
               r < 0 → cross-group cells cluster (interleaving)
  - H17_star : robustness — normalised Shannon entropy over 17 types (Supplement)
  - r17      : robustness — assortativity over 17 types

Covariates joined from existing tables:
  - community  (1–8, Leiden k=100; community 0 is a valid Leiden community)
  - log_pop    (natural log of city population, from regression analysis_input)
  - country
  - n_cells    (number of grid cells in city, QC)
  - n_edges    (total adjacency edges in city, QC)

Output
------
outputs/tables/city_metrics.parquet
    One row per city; all columns above plus city, country.

Usage
-----
    python 01_compute_city_metrics.py [--workers N]

    --workers   Number of parallel workers (default: 16)

Note: requires geopandas + libpysal (available in DL_env / Africa_env).
Run with:
    /home/sjs96_file_share/anaconda3/envs/DL_env/bin/python3 01_compute_city_metrics.py
"""

import argparse
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent
SHP_ROOT    = Path("/home/sjs96_file_share/Momepy_workflow/Metrix/"
                   "data_cleaning_20260224/SHP_cleaned")
LABELS_PATH = (ROOT.parent / "01_typology_construction" / "typology_construction"
               / "outputs" / "clusters" / "grid_type_labels.csv")
COMM_PATH   = (ROOT.parent / "02_regional_structure" /
               "community_detection" / "outputs" / "city_communities.parquet")
REG_PATH    = (ROOT.parent / "02_regional_structure" /
               "regression" / "outputs" / "tables" / "analysis_input.csv")
OUT_DIR     = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Higher-order group mapping  (label_k17 → group 1..4)
# ---------------------------------------------------------------------------
# Group 1 — Medium-density built-up fabric        : C2, C17, C11, C16, C9, C10
# Group 2 — High-density compact urban fabric     : C14, C15, C12, C13
# Group 3 — Sparse low-built vegetated settlement : C4, C5
# Group 4 — Low-density heterogeneous built-up    : C1, C6, C7, C3, C8
GROUP_MAP = {
    2: 1, 17: 1, 11: 1, 16: 1,  9: 1, 10: 1,
    14: 2, 15: 2, 12: 2, 13: 2,
    4: 3,  5: 3,
    1: 4,  6: 4,  7: 4,  3: 4,  8: 4,
}
N_GROUPS = 4
N_TYPES  = 17


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------
def _shannon(counts: np.ndarray, n_cat: int) -> float:
    """Normalised Shannon entropy H* = H / log(n_cat)."""
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts / total
    p = p[p > 0]
    h = -np.sum(p * np.log(p))
    return h / np.log(n_cat)


# ---------------------------------------------------------------------------
# Newman assortativity from edge mixing matrix
# ---------------------------------------------------------------------------
def _assortativity_from_edges(labels_a: np.ndarray, labels_b: np.ndarray,
                               n_cat: int) -> float:
    """
    Compute Newman categorical assortativity from arrays of edge endpoint labels.
    Works on undirected edges (each edge appears once).

    r = (Tr(e) - sum(a_i^2)) / (1 - sum(a_i^2))

    where e_ij = fraction of edges with endpoints in categories i, j
    and a_i = sum_j e_ij  (row marginal).
    """
    if len(labels_a) == 0:
        return np.nan
    # Build mixing matrix
    e = np.zeros((n_cat, n_cat), dtype=float)
    for a, b in zip(labels_a, labels_b):
        if 0 < a <= n_cat and 0 < b <= n_cat:
            e[a - 1, b - 1] += 1
            e[b - 1, a - 1] += 1   # undirected: count both directions
    total = e.sum()
    if total == 0:
        return np.nan
    e /= total
    trace = np.trace(e)
    a     = e.sum(axis=1)
    denom = 1 - np.sum(a ** 2)
    if abs(denom) < 1e-12:
        return np.nan
    return (trace - np.sum(a ** 2)) / denom


# ---------------------------------------------------------------------------
# Per-city computation
# ---------------------------------------------------------------------------
def _process_city(args):
    """
    Worker function.  Loads SHP, computes Queen adjacency, returns metrics dict.
    """
    country, city, city_labels_df = args

    # ---- Label arrays ----
    grid_ids  = city_labels_df["grid_id"].values
    type_arr  = city_labels_df["label_k17"].values
    group_arr = np.array([GROUP_MAP.get(int(t), 0) for t in type_arr])

    n_cells = len(grid_ids)
    if n_cells == 0:
        return None

    # ---- Group share & entropy ----
    group_counts = np.zeros(N_GROUPS + 1, dtype=int)
    for g in group_arr:
        if 1 <= g <= N_GROUPS:
            group_counts[g] += 1
    type_counts = np.zeros(N_TYPES + 1, dtype=int)
    for t in type_arr:
        if 1 <= t <= N_TYPES:
            type_counts[t] += 1

    n_valid = int(group_counts[1:].sum())
    if n_valid == 0:
        return None
    p = group_counts[1:] / n_valid

    h_star   = _shannon(group_counts[1:], N_GROUPS)
    h17_star = _shannon(type_counts[1:],  N_TYPES)

    # ---- Queen adjacency from SHP ----
    r = np.nan; r17 = np.nan; n_edges = 0
    shp_dir = SHP_ROOT / country / city
    shps    = list(shp_dir.glob("*.shp")) if shp_dir.exists() else []
    if shps:
        try:
            import geopandas as gpd
            from libpysal.weights import Queen

            gdf = gpd.read_file(shps[0])
            # Merge labels onto gdf
            lbl_df = pd.DataFrame({"GridID": grid_ids,
                                   "label_k17": type_arr,
                                   "group4": group_arr})
            gdf = gdf.merge(lbl_df, on="GridID", how="inner")
            if len(gdf) == 0:
                raise ValueError("Empty merge")

            # Reproject to Web Mercator (EPSG:3857) before Queen computation.
            # Some UTM-zoned SHPs produce zero edges with libpysal when the
            # CRS causes floating-point gaps at nominally shared hex boundaries.
            # EPSG:3857 resolves this consistently across all countries.
            if gdf.crs is not None and gdf.crs.to_epsg() != 3857:
                gdf = gdf.to_crs(epsg=3857)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w = Queen.from_dataframe(gdf, silence_warnings=True)

            # Collect edge endpoint labels
            gdf = gdf.reset_index(drop=True)
            g4_vals  = gdf["group4"].values
            t17_vals = gdf["label_k17"].values

            ea, eb     = [], []
            e17a, e17b = [], []
            for focal, neighbours in w.neighbors.items():
                ga = g4_vals[focal]
                ta = t17_vals[focal]
                for nb in neighbours:
                    if nb > focal:  # count each undirected edge once
                        gb = g4_vals[nb]
                        tb = t17_vals[nb]
                        ea.append(ga); eb.append(gb)
                        e17a.append(ta); e17b.append(tb)

            n_edges = len(ea)
            r   = _assortativity_from_edges(np.array(ea),  np.array(eb),  N_GROUPS)
            r17 = _assortativity_from_edges(np.array(e17a), np.array(e17b), N_TYPES)

        except Exception:
            pass   # leave r as nan

    return {
        "country":  country,
        "city":     city,
        "n_cells":  n_cells,
        "n_edges":  n_edges,
        "p1":       p[0],
        "p2":       p[1],
        "p3":       p[2],
        "p4":       p[3],
        "H_star":   h_star,
        "H17_star": h17_star,
        "r":        r,
        "r17":      r17,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_workers: int = 16) -> None:
    # --- Load grid labels ---
    logging.info("Loading grid labels ...")
    labels_df = pd.read_csv(LABELS_PATH, usecols=["grid_id", "city", "country", "label_k17"])

    city_groups = (
        labels_df.groupby(["country", "city"])
        .apply(lambda x: x[["grid_id", "label_k17"]], include_groups=False)
        .reset_index(level=[0, 1])
        .groupby(["country", "city"])
    )

    tasks = [
        (name[0], name[1], grp.reset_index(drop=True))
        for name, grp in city_groups
    ]
    logging.info("  %d cities to process.", len(tasks))

    # --- Parallel processing ---
    results = []
    logging.info("Computing metrics with %d workers ...", n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_process_city, t): t for t in tasks}
        done = 0
        for fut in as_completed(futs):
            done += 1
            if done % 500 == 0:
                logging.info("  %d / %d cities done ...", done, len(tasks))
            res = fut.result()
            if res is not None:
                results.append(res)

    metrics = pd.DataFrame(results)
    logging.info("Metrics computed for %d cities.", len(metrics))

    # --- Join community ---
    comm = pd.read_parquet(COMM_PATH)[["country", "city", "community"]]
    metrics = metrics.merge(comm, on=["country", "city"], how="left")

    # --- Join log_pop and covariates ---
    reg = pd.read_csv(REG_PATH, usecols=[
        "city", "country", "log_pop", "climate_zone", "colonial", "log_gdp"
    ])
    metrics = metrics.merge(reg, on=["country", "city"], how="left")

    # --- QC ---
    n_miss_comm = metrics["community"].isna().sum()
    n_miss_pop  = metrics["log_pop"].isna().sum()
    n_miss_r    = metrics["r"].isna().sum()
    logging.info("QC — missing community: %d, missing log_pop: %d, missing r: %d",
                 n_miss_comm, n_miss_pop, n_miss_r)
    logging.info("r range: %.3f – %.3f (median %.3f)",
                 metrics["r"].min(), metrics["r"].max(), metrics["r"].median())
    logging.info("H* range: %.3f – %.3f (median %.3f)",
                 metrics["H_star"].min(), metrics["H_star"].max(),
                 metrics["H_star"].median())

    # --- Save ---
    out = OUT_DIR / "city_metrics.parquet"
    metrics.to_parquet(out, index=False)
    logging.info("Saved: %s  (%d rows × %d cols)", out, len(metrics), len(metrics.columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute city-level morphological metrics for urban hierarchy analysis."
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Number of parallel worker processes (default: 16)."
    )
    args = parser.parse_args()
    main(n_workers=args.workers)
