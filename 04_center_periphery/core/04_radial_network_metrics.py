"""
04_radial_network_metrics.py
============================
Compute radial spatial assortativity for each city x ring using an
edge-midpoint assignment strategy, for THREE distance frameworks.

Frameworks
----------
rel  — relative within-city radius: d_edge* = mean(d_norm_i, d_norm_j)
       same ring5 thresholds [0,.2) .. [.8,1]  (robustness)
abs  — absolute distance: d_edge_km = mean(dist_km_i, dist_km_j)
       digitized into km_bins from allometric_params.json
scl  — population-scaled: d_edge_scl = mean(d_scaled_i, d_scaled_j)
       digitized into scl_bins from allometric_params.json

Method
------
For every hex cell pair that is Queen-adjacent in a city, assign the edge
to a radial band based on the mean distance of its two endpoints.

Advantage over induced subgraph: no edges are dropped; cross-ring pairs are
assigned to the intermediate zone, preserving spatial continuity information.

Metrics per city x ring_type x ring
-------------------------------------
n_edges_ring    — edges with midpoint in this ring
n_types_ring    — number of distinct label_k17 values on these edges' endpoints
r17             — Newman categorical assortativity (17 types) <- main
r4              — Newman categorical assortativity (4 groups) <- robustness

Quality flags
-------------
qc_assort_main     — n_edges_ring >= 30 AND n_types_ring >= 2
qc_assort_strict   — n_edges_ring >= 50 AND n_types_ring >= 2
qc_assort_balanced — qc_assort_main AND all 5 rings present (per ring_type)

Output
------
    outputs/tables/radial_network_metrics.parquet
    Columns: city, country, ring_type, ring, n_edges_ring, n_types_ring, r17, r4
             community, log_pop, n_cells
             qc_assort_main, qc_assort_strict, qc_assort_balanced

Usage
-----
    python 04_radial_network_metrics.py [--workers N]
"""

import argparse
import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT          = Path(__file__).resolve().parent
SHP_ROOT      = Path("/home/sjs96_file_share/Momepy_workflow/Metrix/"
                     "data_cleaning_20260224/SHP_cleaned")
LABELS_PATH   = (ROOT.parent.parent / "01_typology_construction" / "typology_construction"
                 / "outputs" / "clusters" / "grid_type_labels.csv")
CELL_POS_PATH = ROOT / "outputs" / "tables" / "cell_radial_position.parquet"
CITY_METRICS  = (ROOT.parent.parent / "03_urban_hierarchy" /
                 "outputs" / "tables" / "city_metrics.parquet")
ALLOMETRIC    = ROOT / "outputs" / "tables" / "allometric_params.json"
OUT_DIR       = ROOT / "outputs" / "tables"
OUT_PATH      = OUT_DIR / "radial_network_metrics.parquet"

N_TYPES  = 17
N_GROUPS = 4

GROUP_MAP = {
    2: 1, 17: 1, 11: 1, 16: 1,  9: 1, 10: 1,
    14: 2, 15: 2, 12: 2, 13: 2,
    4: 3,  5: 3,
    1: 4,  6: 4,  7: 4,  3: 4,  8: 4,
}

MIN_EDGES_MAIN   = 30
MIN_EDGES_STRICT = 50
MIN_TYPES        = 2

# Loaded in main(), passed to workers via global dict
_PARAMS = {}


# ---------------------------------------------------------------------------
# Newman assortativity (same as Part 3)
# ---------------------------------------------------------------------------
def _assortativity(labels_a: np.ndarray, labels_b: np.ndarray,
                   n_cat: int) -> float:
    if len(labels_a) == 0:
        return np.nan
    e = np.zeros((n_cat, n_cat), dtype=float)
    for a, b in zip(labels_a, labels_b):
        if 1 <= a <= n_cat and 1 <= b <= n_cat:
            e[a - 1, b - 1] += 1
            e[b - 1, a - 1] += 1
    total = e.sum()
    if total == 0:
        return np.nan
    e /= total
    tr  = np.trace(e)
    a   = e.sum(axis=1)
    den = 1 - np.sum(a ** 2)
    if abs(den) < 1e-12:
        return np.nan
    return float((tr - np.sum(a ** 2)) / den)


# ---------------------------------------------------------------------------
# Per-city worker
# ---------------------------------------------------------------------------
def _process_city(args):
    (country, city, city_labels_df, city_radial_df) = args

    shp_dir = SHP_ROOT / country / city
    if not shp_dir.exists():
        return None
    shps = list(shp_dir.glob("*.shp"))
    if not shps:
        return None

    try:
        from libpysal.weights import Queen

        # Load geometry + GridID
        gdf = gpd.read_file(shps[0], columns=["GridID", "geometry"])
        if gdf.empty:
            return None

        # Merge labels
        lbl_df = (city_labels_df[["grid_id", "label_k17"]]
                  .rename(columns={"grid_id": "GridID"}))
        gdf = gdf.merge(lbl_df, on="GridID", how="inner")
        if gdf.empty:
            return None

        # Merge radial position columns
        rad_df = city_radial_df[[
            "GridID",
            "dist_norm_primary",   # for rel framework
            "dist_km_primary",     # for abs framework
            "d_scaled",            # for scl framework (precomputed)
        ]].copy()
        gdf = gdf.merge(rad_df, on="GridID", how="inner")
        if gdf.empty:
            return None

        # group4
        gdf["group4"] = gdf["label_k17"].map(GROUP_MAP).fillna(0).astype(int)

        # Reproject to EPSG:3857 for stable Queen adjacency
        if gdf.crs is not None and gdf.crs.to_epsg() != 3857:
            gdf = gdf.to_crs(epsg=3857)

        gdf = gdf.reset_index(drop=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = Queen.from_dataframe(gdf, silence_warnings=True)

        t17_vals = gdf["label_k17"].values
        g4_vals  = gdf["group4"].values
        dn_vals  = gdf["dist_norm_primary"].values     # rel
        dk_vals  = gdf["dist_km_primary"].values       # abs
        ds_vals  = gdf["d_scaled"].values              # scl

        # Params for ring assignment
        km_edges  = _PARAMS["km_edges"]    # [1, 3, 5, 10]
        scl_edges = _PARAMS["scl_edges"]   # [q20, q40, q60, q80]
        rel_bins  = np.array([0.0, 0.2, 0.4, 0.6, 0.8])

        # ring_data[ring_type][ring] = {t17a, t17b, g4a, g4b}
        ring_data = {
            rt: {r: {"t17a": [], "t17b": [], "g4a": [], "g4b": []}
                 for r in range(1, 6)}
            for rt in ("rel", "abs", "scl")
        }

        for focal, neighbours in w.neighbors.items():
            t17_f = t17_vals[focal]
            g4_f  = g4_vals[focal]
            dn_f  = dn_vals[focal]
            dk_f  = dk_vals[focal]
            ds_f  = ds_vals[focal]

            for nb in neighbours:
                if nb <= focal:
                    continue
                # Midpoint values
                mid_rel = (dn_f + dn_vals[nb]) / 2.0
                mid_abs = (dk_f + dk_vals[nb]) / 2.0
                mid_scl = (ds_f + ds_vals[nb]) / 2.0

                # Ring assignment
                r_rel = int(np.clip(int(np.digitize(mid_rel, rel_bins)),  1, 5))
                r_abs = int(np.clip(int(np.digitize(mid_abs, km_edges)) + 1, 1, 5))
                r_scl = int(np.clip(int(np.digitize(mid_scl, scl_edges)) + 1, 1, 5))

                t17_nb = t17_vals[nb]
                g4_nb  = g4_vals[nb]

                for rt, ring in [("rel", r_rel), ("abs", r_abs), ("scl", r_scl)]:
                    rd = ring_data[rt][ring]
                    rd["t17a"].append(t17_f)
                    rd["t17b"].append(t17_nb)
                    rd["g4a"].append(g4_f)
                    rd["g4b"].append(g4_nb)

        records = []
        for rt in ("rel", "abs", "scl"):
            for ring in range(1, 6):
                rd   = ring_data[rt][ring]
                t17a = np.array(rd["t17a"], dtype=int)
                t17b = np.array(rd["t17b"], dtype=int)
                g4a  = np.array(rd["g4a"],  dtype=int)
                g4b  = np.array(rd["g4b"],  dtype=int)

                n_edges = len(t17a)
                n_types = len(set(t17a) | set(t17b))

                r17 = _assortativity(t17a, t17b, N_TYPES)
                r4  = _assortativity(g4a,  g4b,  N_GROUPS)

                records.append({
                    "city":         city,
                    "country":      country,
                    "ring_type":    rt,
                    "ring":         ring,
                    "n_edges_ring": n_edges,
                    "n_types_ring": n_types,
                    "r17":          r17,
                    "r4":           r4,
                })

        return pd.DataFrame(records)

    except Exception as exc:
        logging.warning("  %s / %s failed: %s", country, city, exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_workers: int = 14) -> None:
    global _PARAMS

    # Load allometric params
    logging.info("Loading allometric_params.json ...")
    with open(ALLOMETRIC) as f:
        params = json.load(f)
    km_bins  = [float(b) for b in params["km_bins"]]
    scl_bins = [float(b) for b in params["scl_bins"]]
    alpha    = float(params["alpha_R98"])
    km_bins[-1]  = np.inf
    scl_bins[-1] = np.inf
    _PARAMS["km_edges"]  = km_bins[1:-1]    # interior edges
    _PARAMS["scl_edges"] = scl_bins[1:-1]
    logging.info("  alpha=%.4f  km_edges=%s", alpha, _PARAMS["km_edges"])

    # Load cell radial positions
    logging.info("Loading cell radial positions (%s) ...", CELL_POS_PATH.name)
    pos = pd.read_parquet(CELL_POS_PATH,
                          columns=["GridID", "city", "country",
                                   "dist_norm_primary",
                                   "dist_km_primary",
                                   "dist_m_primary"])
    logging.info("  %d cells loaded.", len(pos))

    # Compute d_scaled = dist_m / pop^alpha per cell
    logging.info("Loading city metrics ...")
    city_meta = pd.read_parquet(CITY_METRICS)[
        ["city", "country", "community", "log_pop", "n_cells"]
    ]
    pos = pos.merge(city_meta[["city", "country", "log_pop"]],
                    on=["city", "country"], how="inner")
    pop_city = np.exp(pos["log_pop"].to_numpy(dtype="float64"))
    pos["d_scaled"] = (pos["dist_m_primary"].to_numpy(dtype="float64") /
                       np.power(pop_city, alpha)).astype("float32")
    pos = pos.drop(columns=["log_pop", "dist_m_primary"])

    # Load labels
    logging.info("Loading grid labels (%s) ...", LABELS_PATH.name)
    labels_df = pd.read_csv(LABELS_PATH, usecols=["grid_id", "city", "country", "label_k17"])

    # Build per-city groups
    city_label_groups  = labels_df.groupby(["country", "city"])
    city_radial_groups = pos.groupby(["country", "city"])

    label_keys  = set(city_label_groups.groups.keys())
    radial_keys = set(city_radial_groups.groups.keys())
    common_keys = label_keys & radial_keys
    logging.info("  %d cities with labels + radial data.", len(common_keys))

    tasks = []
    for (country, city) in sorted(common_keys):
        tasks.append((
            country, city,
            city_label_groups.get_group((country, city)).reset_index(drop=True),
            city_radial_groups.get_group((country, city)).reset_index(drop=True),
        ))

    logging.info("Computing radial assortativity with %d workers ...", n_workers)
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_process_city, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            if done % 500 == 0:
                logging.info("  %d / %d cities done ...", done, len(tasks))
            res = fut.result()
            if res is not None:
                results.append(res)

    logging.info("  %d / %d cities produced output.", len(results), len(tasks))
    if not results:
        logging.error("No results.")
        return

    out = pd.concat(results, ignore_index=True)
    logging.info("  Total city x ring_type x ring rows: %d", len(out))

    # Merge city covariates
    out = out.merge(city_meta, on=["city", "country"], how="left")

    # QC flags
    out["qc_assort_main"]   = ((out["n_edges_ring"] >= MIN_EDGES_MAIN) &
                                (out["n_types_ring"] >= MIN_TYPES))
    out["qc_assort_strict"] = ((out["n_edges_ring"] >= MIN_EDGES_STRICT) &
                                (out["n_types_ring"] >= MIN_TYPES))

    # qc_assort_balanced: all 5 rings pass qc_assort_main per city x ring_type
    bal_ok = (
        out[out["qc_assort_main"]]
        .groupby(["city", "country", "ring_type"])["ring"]
        .apply(lambda s: set(s) >= {1, 2, 3, 4, 5})
        .rename("qc_assort_balanced")
        .reset_index()
    )
    out = out.merge(bal_ok, on=["city", "country", "ring_type"], how="left")
    out["qc_assort_balanced"] = out["qc_assort_balanced"].fillna(False)

    out.to_parquet(OUT_PATH, index=False)
    logging.info("Saved: %s  (%d rows)", OUT_PATH, len(out))

    # Validation
    logging.info("Validation:")
    for rt in ["rel", "abs", "scl"]:
        sub_rt  = out[out["ring_type"] == rt]
        n_main  = sub_rt["qc_assort_main"].sum()
        n_bal   = sub_rt[sub_rt["qc_assort_balanced"]]["city"].nunique()
        bal_sub = sub_rt[sub_rt["qc_assort_balanced"]]
        if len(bal_sub) > 0:
            r17_by_ring = {
                r: float(bal_sub[bal_sub["ring"] == r]["r17"].mean())
                for r in range(1, 6)
            }
        else:
            r17_by_ring = {}
        logging.info("  [%s] qc_main=%d rows  balanced=%d cities  r17=%s",
                     rt, n_main, n_bal,
                     "  ".join(f"R{r}:{v:.4f}" for r, v in r17_by_ring.items()))

    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    main(n_workers=args.workers)
