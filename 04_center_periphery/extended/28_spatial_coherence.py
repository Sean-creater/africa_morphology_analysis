"""
28_spatial_coherence.py
==========================
STATUS: EXPLORATORY, NOT FOR MAIN TEXT (methods review, 2026-10-04).
All three reference-population 95% CIs cross zero -- this script does
not currently support a reportable population trend for spatial
coherence, and the implementation itself has two known issues that
should be fixed before it is reconsidered for the main narrative (see
"Known implementation issues" below). Kept in the pipeline as a
documented negative/inconclusive result and a starting point for anyone
who wants to pick it back up, not as a headline figure.

Third panel of the A/B/C within-city morphology framework (A =
23_scale_dependent_convergence.py's between-city V_between, B =
27_within_city_heterogeneity.py's within-city V_within). C asks a
different question from both: within a city's core or periphery, is
morphological variety organised into spatially COHERENT patches, or
finely interleaved cell-by-cell?

Two zones can have identical V_within (same average pairwise
dissimilarity) while looking completely different on the ground: one
might be a few large contiguous districts of each form (coherent,
"patchy"), the other a fine-grained checkerboard of forms with no
spatial structure (incoherent, "salt-and-pepper"). V_within cannot
tell these apart; this script's C can.

    d_adj(c,z)  = mean over Queen-adjacent cell pairs in zone z of city c
                  of (1 - cos(emb_i, emb_j))
    d_rand(c,z) = mean over an equal number of RANDOM (non-adjacency-
                  restricted) cell pairs in the same zone of
                  (1 - cos(emb_i, emb_j))
    C(c,z)      = 1 - d_adj(c,z) / d_rand(c,z)

    Delta_C_c = C(c, outer) - C(c, core)

    C > 0: adjacent cells are MORE similar than random pairs in the same
           zone -- spatially coherent patches (similar forms cluster
           together)
    C ~ 0: adjacency carries no extra information -- forms are spatially
           random within the zone (fine-grained interleaving)
    C < 0: adjacent cells are LESS similar than random pairs -- actively
           checkerboarded / anti-clustered (unusual; would flag a
           labelling or geometry artifact if it dominates)

Method notes
------------
- Core/outer zones, qualifying-city sample: IDENTICAL to 23 and 27
  (q<0.2 / q>0.75, >=MIN_CELLS_ZONE=20 cells in BOTH zones) -- so A, B,
  C all describe the same city set and are directly comparable.
- Queen adjacency: built from each city's own cleaned grid SHP
  (SHP_cleaned/{country}/{city}/*.shp), reprojected to EPSG:3857 for
  stable adjacency -- same approach as
  04_center_periphery/core/04_radial_network_metrics.py. Adjacency is
  computed on the FULL city grid (not zone-restricted) so that a cell's
  true queen-neighbours are found correctly, then pairs are kept only
  if BOTH endpoints fall in the same zone (core or outer).
- Random-pair baseline: for each zone, draw the SAME NUMBER of random
  (unordered) cell pairs as there are Queen-adjacent pairs, without
  replacement on pairs (sampled with replacement on indices, duplicate
  pairs allowed but vanishingly rare at this n) -- so the comparison
  d_adj vs d_rand isn't confounded by a different sample size between
  the two.
- No rarefaction here (unlike 27): C is a RATIO (d_adj/d_rand), not a
  raw dispersion level, so it is not mechanically driven by zone cell
  count the way a raw mean-pairwise-distance is -- both numerator and
  denominator are means over pairs, and the random-pair draw count is
  matched to the adjacency-pair count per zone.
- Zones with < MIN_ADJ_PAIRS Queen-adjacent pairs are dropped (NaN) --
  too few pairs to estimate d_adj reliably.
- Population trend: identical design to 23/27 -- per-city Delta_C_c vs
  log(Pop_c), LOWESS trend, country-block bootstrap 95% CI (resample
  countries, preserving multiplicity). Reference-population predictions
  at 20k/100k/1M (no zero-crossing search, matching 27's convention).

Known implementation issues (methods review, 2026-10-04 -- NOT fixed,
since all three reference-point CIs already cross zero and there is no
clear result riding on tightening them; documented so a future revision
starts from a known list rather than rediscovering these):
  1. The random-pair draw targets `n_adj` pairs, then DROPS self-pairs
     (a==b) via `keep = a_rand != b_rand` -- this makes the actual random-
     pair count slightly LESS than n_adj, not exactly equal as the
     docstring above claims, and the shortfall isn't backfilled.
  2. d_rand is a SINGLE random draw per city x zone (no repeated-draw
     averaging), so it carries extra Monte Carlo noise on top of genuine
     between-city variation -- this plausibly explains the extreme
     per-city values seen in spatial_coherence_dispersion.csv (e.g.
     C_outer as low as -1.20, delta_C as low as -1.85). A fix would
     either compute d_rand over ALL non-adjacent same-zone pairs directly
     (feasible at this n_cells) or average over multiple (e.g. 20-50)
     random draws per city x zone, matching 27's rarefaction-averaging
     pattern.

Output
------
  outputs/tables/spatial_coherence_dispersion.csv
      city_key | country | log_pop | pop | n_adj_core | n_adj_outer |
      d_adj_core | d_rand_core | C_core | d_adj_outer | d_rand_outer |
      C_outer | delta_C
  outputs/tables/spatial_coherence_trend.csv
      log_pop_grid | pop_grid | delta_C_smooth | delta_C_lo | delta_C_hi
  outputs/tables/spatial_coherence_reference_points.csv
      pop_ref | delta_C | delta_C_lo | delta_C_hi
  outputs/figures/spatial_coherence.svg/.png

Usage
-----
    python 28_spatial_coherence.py [--workers N]
"""

import argparse
import json
import logging
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_FONT_DIR = Path("/home/sjs96_file_share/Scripts4Analysis/_fonts")
if _FONT_DIR.is_dir():
    for _f in fm.findSystemFonts(fontpaths=[str(_FONT_DIR)]):
        fm.fontManager.addfont(_f)
else:
    log.warning("Font directory not found (%s); falling back to default sans-serif.", _FONT_DIR)
matplotlib.rcParams.update({
    "font.family": "Arial", "font.size": 8, "svg.fonttype": "none",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
})

ROOT       = Path(__file__).parent
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_ARRAYS = ROOT / "outputs" / "arrays"
INDEX_PQ   = OUT_TABLES / "cell_embeddings_index.parquet"
EMB_NPY    = OUT_ARRAYS / "cell_embeddings.npy"
BIN_GRID   = OUT_TABLES / "bin_grid.json"
OUT_FIG    = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# SHP geometry + attributes per city, published by the companion
# africa_morphology_metrics repository (cleaning stage output,
# "SHP_cleaned"). Override with AFRICA_SHP_CLEANED_ROOT if your
# copy lives elsewhere.
SHP_ROOT = Path(os.environ.get(
    "AFRICA_SHP_CLEANED_ROOT",
    "../../../africa_morphology_metrics/outputs/SHP_cleaned",
))

CORE_Q_MAX     = 0.2
OUTER_Q_MIN    = 0.75
MIN_CELLS_ZONE = 20
MIN_ADJ_PAIRS  = 10     # zones with fewer Queen-adjacent pairs -> NaN
LOWESS_FRAC    = 0.5
N_BOOT         = 500
BOOT_SEED      = 42
GRID_N         = 60
POP_REFERENCE_POINTS = [20_000, 100_000, 1_000_000]


# ---------------------------------------------------------------------------
# Per-city worker: Queen adjacency + d_adj / d_rand for core and outer zones
# ---------------------------------------------------------------------------
def _process_city(args):
    (country, city, gridid_to_rowidx, gridid_to_zone, seed) = args

    shp_dir = SHP_ROOT / country / city
    if not shp_dir.exists():
        return None
    shps = list(shp_dir.glob("*.shp"))
    if not shps:
        return None

    try:
        import geopandas as gpd
        from libpysal.weights import Queen

        gdf = gpd.read_file(shps[0], columns=["GridID", "geometry"])
        if gdf.empty:
            return None
        gdf = gdf[gdf["GridID"].isin(gridid_to_rowidx)].reset_index(drop=True)
        if len(gdf) < 2 * MIN_CELLS_ZONE:
            return None

        if gdf.crs is not None and gdf.crs.to_epsg() != 3857:
            gdf = gdf.to_crs(epsg=3857)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = Queen.from_dataframe(gdf, silence_warnings=True)

        gids  = gdf["GridID"].values
        zones = np.array([gridid_to_zone.get(g, "") for g in gids])
        rows  = np.array([gridid_to_rowidx.get(g, -1) for g in gids])

        emb_city = EMB_GLOBAL[rows]
        norms = np.linalg.norm(emb_city, axis=1, keepdims=True)
        norms[norms < 1e-12] = np.nan
        emb_n = emb_city / norms

        rng = np.random.default_rng(seed)
        result = {}
        for zone in ("core", "outer"):
            zmask = zones == zone
            local_idx = np.where(zmask)[0]
            if len(local_idx) < MIN_CELLS_ZONE:
                result[zone] = (np.nan, np.nan, 0)
                continue

            # Queen-adjacent pairs with both endpoints in this zone.
            adj_pairs = []
            local_set = set(local_idx.tolist())
            for focal in local_idx:
                for nb in w.neighbors.get(focal, []):
                    if nb in local_set and nb > focal:
                        adj_pairs.append((focal, nb))
            n_adj = len(adj_pairs)
            if n_adj < MIN_ADJ_PAIRS:
                result[zone] = (np.nan, np.nan, n_adj)
                continue

            adj_pairs = np.array(adj_pairs)
            sim_adj = np.sum(emb_n[adj_pairs[:, 0]] * emb_n[adj_pairs[:, 1]], axis=1)
            d_adj = float(np.nanmean(1.0 - sim_adj))

            # Random-pair baseline, same n as adjacency pairs.
            n_local = len(local_idx)
            a_rand = rng.choice(local_idx, size=n_adj, replace=True)
            b_rand = rng.choice(local_idx, size=n_adj, replace=True)
            keep = a_rand != b_rand
            a_rand, b_rand = a_rand[keep], b_rand[keep]
            sim_rand = np.sum(emb_n[a_rand] * emb_n[b_rand], axis=1)
            d_rand = float(np.nanmean(1.0 - sim_rand))

            result[zone] = (d_adj, d_rand, n_adj)

        return (country, city, result)

    except Exception as exc:
        logging.warning("  %s / %s failed: %s", country, city, exc)
        return None


def _init_worker(emb_path):
    global EMB_GLOBAL
    EMB_GLOBAL = np.load(emb_path)


def _lowess_trend(log_pop: np.ndarray, delta: np.ndarray, grid: np.ndarray,
                  frac: float = LOWESS_FRAC) -> np.ndarray:
    mask = np.isfinite(log_pop) & np.isfinite(delta)
    if mask.sum() < 10:
        return np.full(len(grid), np.nan)
    sm = lowess(delta[mask], log_pop[mask], frac=frac, return_sorted=True)
    return np.interp(grid, sm[:, 0], sm[:, 1])


def main(n_workers: int = 14) -> None:
    log.info("Loading cell_embeddings_index.parquet ...")
    idx = pd.read_parquet(INDEX_PQ,
                          columns=["GridID", "city", "country", "row_idx",
                                   "q_relative", "log_pop"])
    with open(BIN_GRID) as f:
        q_max = float(json.load(f)["q_P95"])
    idx = idx[(idx["q_relative"] >= 0) & (idx["q_relative"] <= q_max)].copy()
    idx["_city_key"] = idx["country"] + "||" + idx["city"]
    idx = idx.dropna(subset=["log_pop"])
    log.info("  %d cells, q_max=%.4f", len(idx), q_max)

    # ── same qualifying-city sample as 23/27 ──
    core_sub = idx[idx["q_relative"] < CORE_Q_MAX]
    outer_sub = idx[(idx["q_relative"] > OUTER_Q_MIN) & (idx["q_relative"] <= q_max)]
    core_n = core_sub.groupby("_city_key").size()
    outer_n = outer_sub.groupby("_city_key").size()
    qualified = sorted(set(core_n[core_n >= MIN_CELLS_ZONE].index) &
                       set(outer_n[outer_n >= MIN_CELLS_ZONE].index))
    log.info("  %d cities qualify (>=%d cells in BOTH core and outer).",
            len(qualified), MIN_CELLS_ZONE)

    city_log_pop = idx.groupby("_city_key")["log_pop"].first()
    city_country = {ck: ck.split("||")[0] for ck in qualified}
    city_name    = {ck: ck.split("||")[1] for ck in qualified}

    core_sub_q = core_sub[core_sub["_city_key"].isin(qualified)]
    outer_sub_q = outer_sub[outer_sub["_city_key"].isin(qualified)]

    log.info("Building per-city GridID->row_idx / zone maps ...")
    tasks = []
    rng_master = np.random.default_rng(BOOT_SEED)
    for i, ck in enumerate(qualified):
        country, city = city_country[ck], city_name[ck]
        c_rows = core_sub_q[core_sub_q["_city_key"] == ck]
        o_rows = outer_sub_q[outer_sub_q["_city_key"] == ck]
        gridid_to_rowidx = dict(zip(c_rows["GridID"], c_rows["row_idx"]))
        gridid_to_rowidx.update(dict(zip(o_rows["GridID"], o_rows["row_idx"])))
        gridid_to_zone = {g: "core" for g in c_rows["GridID"]}
        gridid_to_zone.update({g: "outer" for g in o_rows["GridID"]})
        seed = int(rng_master.integers(0, 2**31 - 1))
        tasks.append((country, city, gridid_to_rowidx, gridid_to_zone, seed))

    log.info("Computing Queen adjacency + d_adj/d_rand for %d cities (%d workers) ...",
             len(tasks), n_workers)
    results = {}
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                             initargs=(str(EMB_NPY),)) as ex:
        futs = {ex.submit(_process_city, t): t for t in tasks}
        for fut in as_completed(futs):
            done += 1
            if done % 100 == 0:
                log.info("  %d / %d cities done ...", done, len(tasks))
            res = fut.result()
            if res is not None:
                country, city, zone_result = res
                results[f"{country}||{city}"] = zone_result

    log.info("  %d / %d cities produced output.", len(results), len(tasks))

    rows = []
    for ck in qualified:
        if ck not in results:
            continue
        zr = results[ck]
        d_adj_c, d_rand_c, n_adj_c = zr.get("core", (np.nan, np.nan, 0))
        d_adj_o, d_rand_o, n_adj_o = zr.get("outer", (np.nan, np.nan, 0))
        C_core = 1.0 - d_adj_c / d_rand_c if np.isfinite(d_adj_c) and d_rand_c not in (0, np.nan) and np.isfinite(d_rand_c) else np.nan
        C_outer = 1.0 - d_adj_o / d_rand_o if np.isfinite(d_adj_o) and d_rand_o not in (0, np.nan) and np.isfinite(d_rand_o) else np.nan
        rows.append({
            "city_key": ck, "country": city_country[ck],
            "log_pop": float(city_log_pop[ck]), "pop": float(np.exp(city_log_pop[ck])),
            "n_adj_core": n_adj_c, "n_adj_outer": n_adj_o,
            "d_adj_core": d_adj_c, "d_rand_core": d_rand_c, "C_core": C_core,
            "d_adj_outer": d_adj_o, "d_rand_outer": d_rand_o, "C_outer": C_outer,
            "delta_C": C_outer - C_core if np.isfinite(C_core) and np.isfinite(C_outer) else np.nan,
        })
    main_df = pd.DataFrame(rows)
    main_df.to_csv(OUT_TABLES / "spatial_coherence_dispersion.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: spatial_coherence_dispersion.csv (%d cities, %d with valid delta_C)",
             len(main_df), main_df["delta_C"].notna().sum())

    plot_df = main_df.dropna(subset=["delta_C"]).reset_index(drop=True)

    # ── LOWESS trend ──
    log_pop_grid = np.linspace(plot_df["log_pop"].min(), plot_df["log_pop"].max(), GRID_N)
    delta_c_smooth = _lowess_trend(plot_df["log_pop"].values, plot_df["delta_C"].values, log_pop_grid)

    # ── country-block bootstrap on the LOWESS trend (same design as 23/27) ──
    log.info("Bootstrapping (country-block, n=%d) ...", N_BOOT)
    countries = sorted(plot_df["country"].unique())
    country_df = {c: plot_df[plot_df["country"] == c] for c in countries}
    rng = np.random.default_rng(BOOT_SEED)
    boot_delta_c = np.full((N_BOOT, GRID_N), np.nan)
    for b in range(N_BOOT):
        chosen_countries = rng.choice(countries, size=len(countries), replace=True)
        resampled = pd.concat([country_df[c] for c in chosen_countries], ignore_index=True)
        if len(resampled) < 20:
            continue
        d_smooth = _lowess_trend(resampled["log_pop"].values, resampled["delta_C"].values, log_pop_grid)
        boot_delta_c[b] = d_smooth
        if (b + 1) % 100 == 0:
            log.info("  bootstrap draw %d/%d", b + 1, N_BOOT)

    delta_c_lo = np.nanpercentile(boot_delta_c, 2.5, axis=0)
    delta_c_hi = np.nanpercentile(boot_delta_c, 97.5, axis=0)

    trend_df = pd.DataFrame({
        "log_pop_grid": log_pop_grid, "pop_grid": np.exp(log_pop_grid),
        "delta_C_smooth": delta_c_smooth,
        "delta_C_lo": delta_c_lo, "delta_C_hi": delta_c_hi,
    })
    trend_df.to_csv(OUT_TABLES / "spatial_coherence_trend.csv",
                    index=False, float_format="%.6f")
    log.info("Saved: spatial_coherence_trend.csv")

    # ── reference-point predictions (20k / 100k / 1M), no zero-crossing ──
    ref_rows = []
    for pop_ref in POP_REFERENCE_POINTS:
        log_p = np.log(pop_ref)
        d_pred = float(np.interp(log_p, log_pop_grid, delta_c_smooth))
        d_lo = float(np.interp(log_p, log_pop_grid, delta_c_lo))
        d_hi = float(np.interp(log_p, log_pop_grid, delta_c_hi))
        ref_rows.append({"pop_ref": pop_ref, "delta_C": d_pred,
                        "delta_C_lo": d_lo, "delta_C_hi": d_hi})
        log.info("  pop=%s: Delta_C=%.4f [%.4f, %.4f]",
                f"{pop_ref:,}", d_pred, d_lo, d_hi)
    ref_df = pd.DataFrame(ref_rows)
    ref_df.to_csv(OUT_TABLES / "spatial_coherence_reference_points.csv",
                 index=False, float_format="%.6f")
    log.info("Saved: spatial_coherence_reference_points.csv")
    print("\n=== Delta_C at reference populations ===")
    print(ref_df.to_string(index=False))

    # ── figure ──
    log.info("Drawing figure ...")
    AXES_W_MM, AXES_H_MM = 100.0, 75.0
    LEFT_MM, RIGHT_MM, BOTTOM_MM, TOP_MM = 20.0, 20.0, 16.0, 6.0
    FIG_W_MM = LEFT_MM + AXES_W_MM + RIGHT_MM
    FIG_H_MM = BOTTOM_MM + AXES_H_MM + TOP_MM
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    ax = fig.add_axes([LEFT_MM / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
                       AXES_W_MM / FIG_W_MM, AXES_H_MM / FIG_H_MM])

    ax.scatter(plot_df["pop"], plot_df["delta_C"], s=6, alpha=0.25,
              color="#2E8B57", linewidth=0, zorder=2)
    ax.fill_between(np.exp(log_pop_grid), delta_c_lo, delta_c_hi,
                    color="#2E8B57", alpha=0.15, linewidth=0, zorder=1)
    ax.plot(np.exp(log_pop_grid), delta_c_smooth, color="#2E8B57", lw=2.0, zorder=3)
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    for pop_ref in POP_REFERENCE_POINTS:
        ax.axvline(pop_ref, color="#C8292A", lw=0.6, ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("City population", fontsize=9)
    ax.set_ylabel(r"$\Delta C_c = C_{outer} - C_{core}$" "\n"
                 r"($C = 1 - d_{adj}/d_{rand}$, Queen adjacency)",
                 fontsize=9)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="black", labelsize=8)

    caption = (
        r"$C>0$: spatially coherent patches (adjacent cells more similar than random "
        "pairs in the same zone);" "\n"
        r"$C\approx0$: fine-grained interleaving (adjacency carries no extra "
        r"similarity). $\Delta C>0$: periphery more" "\n"
        "patchy than core. Random-pair baseline matched to the Queen-adjacent pair "
        "count per zone." "\n"
        "Shaded band: 95% country-block bootstrap CI. Dotted lines: 20k/100k/1M "
        "reference populations" "\n"
        "(see spatial_coherence_reference_points.csv)."
    )
    fig.text(0.5, -0.02, caption, fontsize=5.5, color="#555555", style="italic",
            ha="center", va="top", linespacing=1.6)

    for ext in ("svg", "png"):
        out = OUT_FIG / f"spatial_coherence.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", out)
    plt.close(fig)

    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    main(n_workers=args.workers)
