"""
03_ring_profiles.py
===================
Aggregate hex-cell radial data to city × ring level for THREE distance frameworks:

  abs  — absolute geographic distance (km bands from allometric_params.json)
  scl  — population-scaled distance  (d_scaled = dist_m / Pop^α̂, quintile bands)
  rel  — relative within-city radius (ring5_primary, 5 equal [0,1] bands) ← robustness

For each city × ring_type × ring computes:
  - 4-group composition shares  p1–p4
  - 17-type composition shares  p01–p17
  - Normalised Shannon diversity H4_star (over 4 groups)
  - Normalised Shannon diversity H17_star (over 17 types)
  - Rarefied H17* at n=20 cells (H17_rare) — corrects sample-size bias
  - Population share, land share, population-land ratio (RPC)

Centre version: primary_pop only (main analysis).
Pop/geom robustness available by re-running with dist_km_pop / ring5_pop if needed.

Quality flags (applied per ring_type independently)
----------------------------------------------------
qc_comp     — n_cells_ring >= 5
qc_div      — n_cells_ring >= 20  (required for H17_rare; otherwise nan)
qc_balanced — qc_comp AND all 5 rings present for this city x ring_type

Note: for abs framework, large ring numbers (4,5) are only reached by larger
cities (R98 > 5 km / >10 km). No balanced requirement enforced for abs.

Output
------
    outputs/tables/radial_profiles.parquet
    ~3 x 34,000 rows; key columns:
        city, country, ring_type, ring, n_cells_ring
        p1, p2, p3, p4
        p01 to p17
        H4_star, H17_star, H17_rare
        population_ring, population_city
        population_share, land_share, population_land_ratio
        log_pop, n_cells, community
        qc_comp, qc_div, qc_balanced

Usage
-----
    python 03_ring_profiles.py

Notes
-----
- City-equally-weighted means: compute per-city per-ring values, then average
  across cities -- NOT pooled-cell averages.
- Hex cells have uniform area; land_share = n_cells_ring / n_cells_city.
- Rarefaction: randomly subsample N_RARE=20 cells per ring 100 times,
  average H17* -- removes sample-size bias when comparing rings of different sizes.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "tables"

CELL_POS_PATH   = OUT_DIR / "cell_radial_position.parquet"
CITY_METRICS    = (ROOT.parent.parent / "03_urban_hierarchy" /
                   "outputs" / "tables" / "city_metrics.parquet")
ALLOMETRIC_PATH = OUT_DIR / "allometric_params.json"
OUT_PATH        = OUT_DIR / "radial_profiles.parquet"

N_TYPES  = 17
N_GROUPS = 4
N_RARE   = 20    # rarefaction subsample size
N_BOOT   = 100   # rarefaction replicates

MIN_CELLS_RING_COMP = 5
MIN_CELLS_RING_DIV  = 20


# ---------------------------------------------------------------------------
# Rarefied Shannon diversity
# ---------------------------------------------------------------------------
def _h17_rare(labels17_array: np.ndarray,
              n_rare: int = N_RARE,
              n_boot: int = N_BOOT,
              rng=None) -> float:
    """
    Rarefied normalised Shannon H17*.
    Returns nan if n_cells < n_rare.
    """
    n = len(labels17_array)
    if n < n_rare:
        return np.nan
    if rng is None:
        rng = np.random.default_rng()
    vals = []
    for _ in range(n_boot):
        samp   = rng.choice(labels17_array, size=n_rare, replace=False)
        counts = np.bincount(samp.astype(int), minlength=N_TYPES + 1)[1:N_TYPES + 1]
        total  = counts.sum()
        if total == 0:
            continue
        p   = counts / total
        pos = p[p > 0]
        vals.append(-np.sum(pos * np.log(pos)) / np.log(N_TYPES))
    return float(np.mean(vals)) if vals else np.nan


# ---------------------------------------------------------------------------
# Per-ring profile
# ---------------------------------------------------------------------------
def _ring_profile(sub: pd.DataFrame, city_pop_total: float,
                  n_cells_city: int, rng) -> dict:
    """
    Compute all metrics for one city x ring subset.
    sub: DataFrame with columns label_k17, group4, Pop2022.
    """
    n = len(sub)

    # 4-group shares
    vc4     = sub["group4"].value_counts()
    p4      = {f"p{g}": int(vc4.get(g, 0)) / n for g in [1, 2, 3, 4]}
    counts4 = np.array([int(vc4.get(g, 0)) for g in [1, 2, 3, 4]], dtype=float)
    pos4    = counts4[counts4 > 0] / n
    h4      = float(-np.sum(pos4 * np.log(pos4)) / np.log(N_GROUPS)) if len(pos4) > 1 else 0.0

    # 17-type shares
    vc17     = sub["label_k17"].value_counts()
    counts17 = np.array([int(vc17.get(t, 0)) for t in range(1, N_TYPES + 1)], dtype=float)
    total17  = counts17.sum()
    if total17 > 0:
        p17   = {f"p{t:02d}": float(counts17[t - 1] / total17) for t in range(1, N_TYPES + 1)}
        pos17 = counts17[counts17 > 0] / total17
        h17   = float(-np.sum(pos17 * np.log(pos17)) / np.log(N_TYPES)) if len(pos17) > 1 else 0.0
    else:
        p17 = {f"p{t:02d}": np.nan for t in range(1, N_TYPES + 1)}
        h17 = np.nan

    # Rarefied H17*
    labels17_arr = sub["label_k17"].to_numpy(dtype=int)
    h17_rare = _h17_rare(labels17_arr, rng=rng)

    # Population & land
    pop_ring  = float(sub["Pop2022"].sum())
    pop_share = pop_ring / city_pop_total if city_pop_total > 0 else np.nan
    land_share = float(n) / n_cells_city if n_cells_city > 0 else np.nan
    plr        = (pop_share / land_share
                  if (land_share and land_share > 0
                      and pop_share == pop_share)  # nan check
                  else np.nan)

    return {
        **p4,
        **p17,
        "H4_star":               h4,
        "H17_star":              h17,
        "H17_rare":              h17_rare,
        "n_cells_ring":          n,
        "population_ring":       pop_ring,
        "population_share":      pop_share,
        "land_share":            land_share,
        "population_land_ratio": plr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Load parameters
    # ------------------------------------------------------------------
    logging.info("Loading allometric_params.json ...")
    with open(ALLOMETRIC_PATH) as f:
        params = json.load(f)
    km_bins  = [float(b) for b in params["km_bins"]]
    scl_bins = [float(b) for b in params["scl_bins"]]
    alpha    = float(params["alpha_R98"])
    km_bins[-1]  = np.inf
    scl_bins[-1] = np.inf
    logging.info("  alpha = %.4f", alpha)
    logging.info("  km_bins  = %s", km_bins)
    logging.info("  scl_bins = %s", [f"{b:.4g}" for b in scl_bins])

    # Interior edges for np.digitize  (exclude 0 and inf)
    km_edges  = km_bins[1:-1]    # [1, 3, 5, 10]
    scl_edges = scl_bins[1:-1]   # [q20, q40, q60, q80]

    # ------------------------------------------------------------------
    # Load cell data
    # ------------------------------------------------------------------
    logging.info("Loading cell_radial_position.parquet ...")
    cell_pos = pd.read_parquet(
        CELL_POS_PATH,
        columns=[
            "city", "country",
            "label_k17", "group4", "Pop2022",
            "dist_km_primary",
            "dist_m_primary",
            "ring5_primary",
        ],
    )
    logging.info("  %d cells.", len(cell_pos))

    # ------------------------------------------------------------------
    # Load city metrics
    # ------------------------------------------------------------------
    logging.info("Loading city_metrics.parquet ...")
    city_metrics = pd.read_parquet(
        CITY_METRICS,
        columns=["city", "country", "community", "log_pop", "n_cells"],
    )
    logging.info("  %d cities.", len(city_metrics))

    cell_pos = cell_pos.merge(city_metrics, on=["city", "country"], how="inner")
    logging.info("  %d cells after merge.", len(cell_pos))

    # ------------------------------------------------------------------
    # Compute population-scaled distance
    # ------------------------------------------------------------------
    logging.info("Computing d_scaled = dist_m / pop^alpha ...")
    pop_city = np.exp(cell_pos["log_pop"].to_numpy(dtype="float64"))
    d_scaled = (cell_pos["dist_m_primary"].to_numpy(dtype="float64") /
                np.power(pop_city, alpha))
    cell_pos = cell_pos.copy()
    cell_pos["d_scaled"] = d_scaled.astype("float32")

    # ------------------------------------------------------------------
    # Assign rings for each framework
    # ------------------------------------------------------------------
    logging.info("Assigning abs / scl rings ...")
    cell_pos["abs_ring"] = (
        np.digitize(cell_pos["dist_km_primary"].to_numpy(dtype="float64"), km_edges) + 1
    ).clip(1, 5).astype("int8")

    cell_pos["scl_ring"] = (
        np.digitize(cell_pos["d_scaled"].to_numpy(dtype="float64"), scl_edges) + 1
    ).clip(1, 5).astype("int8")

    cell_pos["rel_ring"] = cell_pos["ring5_primary"].astype("int8")

    # ------------------------------------------------------------------
    # Aggregate to city x ring_type x ring
    # ------------------------------------------------------------------
    logging.info("Aggregating profiles ...")

    rng = np.random.default_rng(42)
    records = []

    city_groups  = cell_pos.groupby(["city", "country"])
    total_cities = city_groups.ngroups
    done = 0

    for (city, country), city_df in city_groups:
        done += 1
        if done % 1000 == 0:
            logging.info("  %d / %d cities done ...", done, total_cities)

        n_cells_city   = len(city_df)
        city_pop_total = float(city_df["Pop2022"].sum())
        first          = city_df.iloc[0]
        community      = first["community"]
        log_pop        = first["log_pop"]
        n_cells        = first["n_cells"]

        for ring_type, ring_col in [("abs", "abs_ring"),
                                    ("scl", "scl_ring"),
                                    ("rel", "rel_ring")]:
            ring_vals = city_df[ring_col].to_numpy(dtype=int)

            for ring in range(1, 6):
                mask = ring_vals == ring
                n    = int(mask.sum())
                if n == 0:
                    continue

                sub  = city_df[mask]
                prof = _ring_profile(sub, city_pop_total, n_cells_city, rng)

                records.append({
                    "city":              city,
                    "country":           country,
                    "ring_type":         ring_type,
                    "ring":              ring,
                    "community":         community,
                    "log_pop":           log_pop,
                    "n_cells":           n_cells,
                    "population_city":   city_pop_total,
                    **prof,
                    "qc_comp":           n >= MIN_CELLS_RING_COMP,
                    "qc_div":            n >= MIN_CELLS_RING_DIV,
                })

    logging.info("  %d records assembled.", len(records))
    out = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # qc_balanced: qc_comp AND all 5 rings present per city x ring_type
    # ------------------------------------------------------------------
    logging.info("Computing qc_balanced ...")
    bal_map = (
        out[out["qc_comp"]]
        .groupby(["city", "country", "ring_type"])["ring"]
        .nunique()
        .rename("n_rings_qc")
        .reset_index()
    )
    bal_map["qc_balanced"] = (bal_map["n_rings_qc"] == 5)
    out = out.merge(
        bal_map[["city", "country", "ring_type", "qc_balanced"]],
        on=["city", "country", "ring_type"],
        how="left",
    )
    out["qc_balanced"] = out["qc_balanced"].fillna(False)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    logging.info("Validation:")
    for rt in ["abs", "scl", "rel"]:
        sub_rt        = out[out["ring_type"] == rt]
        n_bal         = sub_rt[sub_rt["qc_balanced"]]["city"].nunique()
        n_div_bal     = sub_rt[sub_rt["qc_balanced"] & sub_rt["qc_div"]]["city"].nunique()
        h_mean        = sub_rt[sub_rt["qc_balanced"]]["H17_rare"].mean()
        logging.info(
            "  [%s] rows=%d  balanced=%d cities  div_balanced=%d cities  "
            "mean_H17_rare=%.4f",
            rt, len(sub_rt), n_bal, n_div_bal,
            h_mean if not np.isnan(h_mean) else -1.0,
        )

    # Abs framework: cities per ring (no balanced required)
    logging.info("  [abs] cities per ring (qc_comp):")
    abs_comp = out[(out["ring_type"] == "abs") & out["qc_comp"]]
    for r in range(1, 6):
        n_cities = abs_comp[abs_comp["ring"] == r]["city"].nunique()
        logging.info("    Ring %d: %d cities", r, n_cities)

    # H17_rare radial profile (rel, balanced)
    rel_bal = out[(out["ring_type"] == "rel") & out["qc_balanced"]]
    logging.info("  H17_rare by ring [rel, balanced, city-equally-weighted]:")
    for r in range(1, 6):
        cm = rel_bal[rel_bal["ring"] == r].groupby(["city","country"])["H17_rare"].mean()
        logging.info("    Ring %d: H17_rare=%.4f  (n=%d)", r,
                     float(cm.mean()) if len(cm) > 0 else np.nan, len(cm))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out.to_parquet(OUT_PATH, index=False)
    logging.info("Saved: %s  (%d rows, %d cols)", OUT_PATH, len(out), out.shape[1])
    logging.info("Done.")


if __name__ == "__main__":
    main()
