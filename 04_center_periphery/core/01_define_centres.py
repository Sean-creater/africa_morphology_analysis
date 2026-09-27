"""
01_define_centres.py
====================
Define three city reference centres for each African city, using population
data (Pop2022) already aggregated to the 250-m hexagonal analysis grid.

Three centre definitions
------------------------
primary_pop   — MAIN
    Population-weighted centroid of the largest contiguous high-population
    cluster. Steps:
      1. One-hop neighbour smoothing of Pop2022 (prevents single-cell peaks
         from dominating).
      2. Candidate cells: top max(10%, 10 cells) by smoothed population.
      3. Contiguous clusters via Queen (8-neighbour) adjacency on grid index.
      4. Primary cluster = largest by total (raw) Pop2022.
      5. Centre = population-weighted centroid of primary cluster cells.

pop_centroid  — ROBUSTNESS A
    Whole-city population-weighted centroid of all hex cells.

geom_centroid — ROBUSTNESS B
    Population-unweighted (geometric) centroid of all hex cell centroids,
    equivalent to the polygon geometric centroid. Matches the previous
    analysis_input.csv latitude/longitude.

All spatial computations are performed in each city's local projected CRS
(from the SHP file), avoiding continental-scale projection distortions.
Outputs are stored as both local-CRS (x, y) and WGS84 (lon, lat) coordinates.

Polycentric indicators
----------------------
n_pop_clusters          — number of candidate high-population clusters
primary_cluster_pop     — total Pop2022 in primary cluster
primary_cluster_share   — primary cluster share of city total Pop2022
secondary_cluster_pop   — total Pop2022 in second-largest cluster (0 if none)
secondary_to_primary    — ratio secondary/primary (0 if none; proxy for polycentricity)

Compactness
-----------
compactness = 4π·A / P²  (Polsby–Popper, ∈ (0, 1]; 1 = perfect circle)
Computed from the union of all hex cell geometries. Used as a robustness
filter in downstream scripts (exclude bottom 10%).

Output
------
    outputs/tables/city_centres.parquet

Usage
-----
    python 01_define_centres.py [--workers N]

Notes
-----
- Pop2022 is estimated population count (area-weighted WorldPop 2022,
  aggregated to the 250-m hex grid; see the companion
  africa_hex_grid_workflow repository). It is the primary population
  variable used throughout the analysis (see
  02_regional_structure/regression/00_prepare_data.py).
- Cities with no SHP or total Pop2022 = 0 are flagged centre_qc = False and
  fall back to geom_centroid for all three centre fields.
- The city list is read directly from SHP_ROOT's directory structure
  (SHP_ROOT/<country>/<city>/), not from analysis_input.csv, since that
  file is itself produced downstream of this script's output
  (city_centres.parquet feeds population_worldpop_2022 into
  02_regional_structure/regression/00_prepare_data.py) — reading it here
  would make the two scripts depend on each other's output.
- Parallelised over cities using ProcessPoolExecutor.
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy import ndimage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT     = Path(__file__).resolve().parent
SHP_ROOT = Path("/home/sjs96_file_share/Momepy_workflow/Metrix/"
                "data_cleaning_20260224/SHP_cleaned")
OUT_DIR  = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum candidate cells for high-pop cluster definition
MIN_CANDIDATE_FRAC = 0.10   # top 10%
MIN_CANDIDATE_ABS  = 10     # but at least 10 cells


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _weighted_centroid(x, y, w):
    """Population-weighted centroid in projected coordinates."""
    w = np.asarray(w, dtype=float)
    total = w.sum()
    if total <= 0:
        return float(np.mean(x)), float(np.mean(y))
    return float((x * w).sum() / total), float((y * w).sum() / total)


def _project_to_wgs84(x_proj, y_proj, crs_from):
    """Convert a single (x, y) point from crs_from to WGS84 lon/lat."""
    t = Transformer.from_crs(crs_from, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x_proj, y_proj)
    return float(lon), float(lat)


def _queen_clusters(indices_2d, shape):
    """
    Label contiguous clusters using Queen (8-neighbour) connectivity on a
    sparse 2-D boolean array defined by (row, col) index pairs.

    Returns an array of cluster labels (1-based) aligned with indices_2d,
    and the total number of clusters.
    """
    mask = np.zeros(shape, dtype=bool)
    rows, cols = indices_2d[:, 0], indices_2d[:, 1]
    mask[rows, cols] = True
    labeled, n_clusters = ndimage.label(mask, structure=np.ones((3, 3)))
    labels = labeled[rows, cols]
    return labels, n_clusters


# ---------------------------------------------------------------------------
# Per-city worker
# ---------------------------------------------------------------------------
def _process_city(args):
    country, city = args

    shp_dir = SHP_ROOT / country / city
    if not shp_dir.exists():
        return None
    shps = list(shp_dir.glob("*.shp"))
    if not shps:
        return None

    try:
        gdf = gpd.read_file(shps[0], columns=["GridID", "center_x", "center_y",
                                                "Pop2022", "geometry"])
        if gdf.empty:
            return None

        crs = gdf.crs
        cx  = gdf["center_x"].to_numpy(dtype=float)   # local projected CRS
        cy  = gdf["center_y"].to_numpy(dtype=float)
        pop_raw = gdf["Pop2022"].to_numpy(dtype=float)
        valid_mask = np.isfinite(pop_raw) & (pop_raw > 0)
        pop = np.where(valid_mask, pop_raw, 0.0)

        n_cells                  = len(gdf)
        n_valid_population_cells = int(valid_mask.sum())
        population_coverage      = n_valid_population_cells / n_cells if n_cells > 0 else 0.0
        total_pop = pop.sum()

        # ------------------------------------------------------------------
        # ROBUSTNESS B: geometric centroid (unweighted mean of cell centres)
        # ------------------------------------------------------------------
        geom_x = float(np.mean(cx))
        geom_y = float(np.mean(cy))
        geom_lon, geom_lat = _project_to_wgs84(geom_x, geom_y, crs)

        # ------------------------------------------------------------------
        # ROBUSTNESS A: whole-city population-weighted centroid
        # ------------------------------------------------------------------
        if total_pop > 0:
            pop_cx, pop_cy = _weighted_centroid(cx, cy, pop)
        else:
            pop_cx, pop_cy = geom_x, geom_y
        pop_clon, pop_clat = _project_to_wgs84(pop_cx, pop_cy, crs)

        # ------------------------------------------------------------------
        # MAIN: primary population cluster centre
        # ------------------------------------------------------------------
        # Step 1 — one-hop neighbour smoothing
        # Build a GridID-based neighbour lookup from the SHP grid.
        # For speed, use a 2-D raster approach: bin cells into a regular grid,
        # apply a 3×3 mean filter, then look up smoothed values by cell index.

        # Estimate grid resolution from IQR of cell centre spacing
        cx_sorted = np.sort(np.unique(np.round(cx, -1)))
        if len(cx_sorted) > 1:
            diffs = np.diff(cx_sorted)
            res = float(np.median(diffs[diffs > 0]))
        else:
            res = 250.0  # fallback: 250 m hexagon grid

        # Map cell centres to 2-D raster indices
        x_min, y_min = cx.min(), cy.min()
        col_idx = np.round((cx - x_min) / res).astype(int)
        row_idx = np.round((cy - y_min) / res).astype(int)
        n_rows  = row_idx.max() + 1
        n_cols  = col_idx.max() + 1

        pop_grid = np.zeros((n_rows, n_cols), dtype=float)
        cnt_grid = np.zeros((n_rows, n_cols), dtype=int)
        np.add.at(pop_grid, (row_idx, col_idx), pop)
        np.add.at(cnt_grid, (row_idx, col_idx), 1)

        # Uniform 3×3 mean filter for smoothing (only over occupied cells)
        kernel     = np.ones((3, 3), dtype=float)
        pop_smooth = ndimage.convolve(pop_grid, kernel, mode="constant", cval=0)
        cnt_smooth = ndimage.convolve(cnt_grid.astype(float), kernel,
                                      mode="constant", cval=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            smooth_grid = np.where(cnt_smooth > 0,
                                   pop_smooth / cnt_smooth, 0.0)

        smooth_vals = smooth_grid[row_idx, col_idx]

        # Step 2 — candidate cells: top max(10%, 10) by smoothed population
        n_candidates = max(int(np.ceil(MIN_CANDIDATE_FRAC * n_cells)),
                           MIN_CANDIDATE_ABS)
        n_candidates = min(n_candidates, n_cells)
        threshold    = np.partition(smooth_vals, -n_candidates)[-n_candidates]
        candidate    = smooth_vals >= threshold

        cand_rows = row_idx[candidate]
        cand_cols = col_idx[candidate]
        cand_idx  = np.column_stack([cand_rows, cand_cols])

        # Step 3 — Queen-adjacency clusters among candidate cells
        cluster_labels, n_clusters = _queen_clusters(
            cand_idx, shape=(n_rows, n_cols))

        # Step 4 — primary cluster = highest total raw Pop2022
        cand_pop = pop[candidate]
        cluster_pops = {}
        for lbl, p in zip(cluster_labels, cand_pop):
            cluster_pops[lbl] = cluster_pops.get(lbl, 0.0) + p

        sorted_clusters = sorted(cluster_pops.items(),
                                 key=lambda kv: kv[1], reverse=True)
        primary_lbl         = sorted_clusters[0][0]
        primary_cluster_pop = sorted_clusters[0][1]
        secondary_cluster_pop = sorted_clusters[1][1] if len(sorted_clusters) > 1 else 0.0
        secondary_to_primary  = (secondary_cluster_pop / primary_cluster_pop
                                  if primary_cluster_pop > 0 else 0.0)

        # Step 5 — population-weighted centroid of primary cluster
        # Map cluster labels (len = n_candidates) back to all-cell index array
        full_labels = np.zeros(n_cells, dtype=int)
        full_labels[candidate] = cluster_labels
        in_primary = full_labels == primary_lbl

        prim_cx, prim_cy = _weighted_centroid(cx[in_primary], cy[in_primary],
                                              pop[in_primary])
        prim_lon, prim_lat = _project_to_wgs84(prim_cx, prim_cy, crs)

        # ------------------------------------------------------------------
        # Compactness: Polsby–Popper = 4π·A / P²
        # Using union of all hex polygons
        # ------------------------------------------------------------------
        try:
            union     = gdf.geometry.unary_union
            area_m2   = union.area
            perim_m   = union.length
            compactness = (4 * np.pi * area_m2 / (perim_m ** 2)
                           if perim_m > 0 else np.nan)
        except Exception:
            area_m2 = perim_m = compactness = np.nan

        # ------------------------------------------------------------------
        # centre_qc: True if population data is valid
        # ------------------------------------------------------------------
        centre_qc = (total_pop > 0) and np.isfinite(prim_lon)

        return {
            "city":    city,
            "country": country,
            "n_cells": n_cells,
            "total_pop2022": total_pop,
            "n_valid_population_cells": n_valid_population_cells,
            "population_coverage": population_coverage,

            # Primary population centre (MAIN)
            "primary_pop_x":   prim_cx,
            "primary_pop_y":   prim_cy,
            "primary_pop_lon": prim_lon,
            "primary_pop_lat": prim_lat,

            # Whole-city population-weighted centroid (ROBUSTNESS A)
            "pop_centroid_x":   pop_cx,
            "pop_centroid_y":   pop_cy,
            "pop_centroid_lon": pop_clon,
            "pop_centroid_lat": pop_clat,

            # Geometric centroid (ROBUSTNESS B)
            "geom_centroid_x":   geom_x,
            "geom_centroid_y":   geom_y,
            "geom_centroid_lon": geom_lon,
            "geom_centroid_lat": geom_lat,

            # Polycentric indicators
            "n_pop_clusters":        n_clusters,
            "primary_cluster_pop":   primary_cluster_pop,
            "primary_cluster_share": (primary_cluster_pop / total_pop
                                      if total_pop > 0 else np.nan),
            "secondary_cluster_pop": secondary_cluster_pop,
            "secondary_to_primary":  secondary_to_primary,

            # Shape
            "compactness": compactness,

            # QC
            "centre_qc": centre_qc,
            "crs_epsg":  crs.to_epsg() if crs else None,
        }

    except Exception as exc:
        logging.warning("  %s / %s failed: %s", country, city, exc)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_workers: int = 14) -> None:
    logging.info("Scanning city list from %s ...", SHP_ROOT)
    tasks = [
        (country_dir.name, city_dir.name)
        for country_dir in sorted(SHP_ROOT.iterdir())
        if country_dir.is_dir()
        for city_dir in sorted(country_dir.iterdir())
        if city_dir.is_dir()
    ]
    logging.info("  %d cities.", len(tasks))

    logging.info("Defining centres with %d workers ...", n_workers)
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
        logging.error("No results — check warnings above.")
        return

    out_df = pd.DataFrame(results)

    # --- Validation ---
    logging.info("Validation:")
    logging.info("  centre_qc=True:  %d cities", out_df["centre_qc"].sum())
    logging.info("  centre_qc=False: %d cities", (~out_df["centre_qc"]).sum())
    logging.info("  compactness: mean=%.3f  median=%.3f  min=%.4f  max=%.4f",
                 out_df["compactness"].mean(), out_df["compactness"].median(),
                 out_df["compactness"].min(), out_df["compactness"].max())
    logging.info("  secondary_to_primary: median=%.3f  >0.5: %d cities",
                 out_df["secondary_to_primary"].median(),
                 (out_df["secondary_to_primary"] > 0.5).sum())

    # Distance between primary_pop and geom_centroid (sanity check)
    dx = out_df["primary_pop_lon"] - out_df["geom_centroid_lon"]
    dy = out_df["primary_pop_lat"] - out_df["geom_centroid_lat"]
    dist_deg = np.sqrt(dx**2 + dy**2)
    logging.info("  primary_pop vs geom_centroid distance (degrees):")
    logging.info("    median=%.4f°  P90=%.4f°  P99=%.4f°",
                 dist_deg.median(), dist_deg.quantile(0.90),
                 dist_deg.quantile(0.99))

    out_path = OUT_DIR / "city_centres.parquet"
    out_df.to_parquet(out_path, index=False)
    logging.info("Saved: %s  (%d cities)", out_path, len(out_df))
    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    main(n_workers=args.workers)
