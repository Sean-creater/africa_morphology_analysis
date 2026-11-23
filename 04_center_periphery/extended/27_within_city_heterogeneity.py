"""
27_within_city_heterogeneity.py
==================================
Complement to 23_scale_dependent_convergence.py's BETWEEN-city
differentiation (V_between(q,P): how dissimilar are same-sized cities'
cores/peripheries from EACH OTHER) with WITHIN-city heterogeneity:
how dissimilar are the cells INSIDE one city's own core or periphery
from each other?

    V_within(c, z) = mean_{i<j in zone z of city c} [1 - cos(emb_i, emb_j)]

    Delta_V_within_c = V_within(c, outer) - V_within(c, core)

    Delta_V_within_c > 0: periphery is internally MORE heterogeneous
                          than the core (more varied urban forms mixed
                          together at the periphery)
    Delta_V_within_c < 0: core is internally more heterogeneous
    Delta_V_within_c ~ 0: core and outer are similarly heterogeneous
                          internally

This is a DIFFERENT question from 23's V_between: V_between asks "do
same-sized cities look like each other at this radius"; V_within asks
"does a single city's own zone contain a wide mix of forms or a narrow
one". The two together let us distinguish, e.g., "large cities' cores
converge across cities (low V_between) AND each core is internally
uniform (low V_within)" from "... AND each core is internally mixed but
the MIX itself is similar across cities (low V_between, high V_within)".

Method notes
------------
- Core/outer zones: SAME definition as 23 (q<0.2 / q>0.75).
  SAMPLE COVERAGE (methods review, 2026-10-05, same fix as 23): every
  city with >=MIN_CELLS_ZONE=2 cells in BOTH zones qualifies (2 is V_
  within's own mathematical floor -- a pairwise-distance statistic needs
  at least 2 observations), not the previous >=20 cutoff, which excluded
  the same ~75% non-random subset of mostly-small cities documented in
  23's docstring. Each city gets a reliability weight w_c =
  sqrt(min(n_core, n_outer)) applied at aggregation (spline regression +
  LOWESS sensitivity), not an inclusion filter.
- Rarefaction DROPPED (methods review, 2026-10-05): the previous version
  fixed every city x zone to a SAME sample size (N_RAREFY=20 cells,
  matching the old MIN_CELLS_ZONE=20), which is incompatible with
  admitting cities with <20 cells in a zone. V_within and the Shannon-
  entropy verification metric are now computed directly on EVERY cell in
  each city's zone (no subsampling), with the zone's cell count instead
  feeding the reliability weight described above -- a city with more
  cells gets a more precise estimate AND more aggregation weight, rather
  than being truncated down to a fixed small sample. (This also resolves
  the earlier note that rarefaction didn't actually remove a "mechanical
  dependence" on cell count -- it wasn't needed for that reason, and
  dropping it removes a now-unnecessary complication entirely.)
- Verification metric: 17-type Shannon entropy (normalized by log(17)),
  computed on the SAME full cell set as V_within, for the SAME city x
  zone. If both move the same direction, the embedding-based finding
  isn't an artifact of the embedding space's particular geometry.
- Population trend: PRIMARY trend is a country-fixed-effects natural-
  spline model, delta_V_within ~ spline(log_pop, 4 df) + C(country), with
  predictions reported as the AVERAGE MARGINAL prediction across the
  observed country composition at each reference population (i.e. for a
  given log_pop, predict once per city's actual country, then average --
  not a prediction for one arbitrarily-chosen reference country). This
  was added after a methods review found the raw (unadjusted) LOWESS
  trend's apparent reversal to positive Delta_V_within near 1M
  population could be an artifact of which countries happen to have
  large cities in this sample, rather than a population effect per se.
  The raw LOWESS trend (no country adjustment) is retained as a
  documented SENSITIVITY curve, not the headline. 95% CI via
  country-block bootstrap (resample countries, not cities, matching 23's
  and this project's established clustered-uncertainty convention; each
  bootstrap draw refits the country-FE spline model, not just the
  trend). No zero-crossing is reported (per request -- the within-city
  question doesn't presuppose a reversal); instead, predicted
  Delta_V_within is reported at three reference populations (20k, 100k,
  1M).

Output
------
  outputs/tables/within_city_heterogeneity_dispersion.csv
      city_key | log_pop | pop | V_within_core | V_within_outer |
      H17_rare_core | H17_rare_outer | delta_V_within | delta_H17_rare
  outputs/tables/within_city_heterogeneity_trend.csv
      log_pop_grid | pop_grid |
      delta_V_within_smooth | delta_V_within_lo | delta_V_within_hi
          (PRIMARY -- country-FE spline, average marginal prediction)
      delta_V_within_raw_lowess (SENSITIVITY -- unadjusted LOWESS, no CI)
      delta_H17_rare_smooth
  outputs/tables/within_city_heterogeneity_reference_points.csv
      pop_ref | delta_V_within | delta_V_within_lo | delta_V_within_hi
  outputs/figures/within_city_heterogeneity.svg/.png

Usage
-----
    python 27_within_city_heterogeneity.py
"""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import SplineTransformer
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

N_TYPES        = 17
CORE_Q_MAX     = 0.2
OUTER_Q_MIN    = 0.75
MIN_CELLS_ZONE = 2      # V_within's own mathematical floor (needs >=2
                        # cells for a pairwise distance); reliability
                        # weight (sqrt(min(n_core,n_outer))), not a
                        # stricter cutoff, handles small-n cities
LOWESS_FRAC    = 0.5
N_BOOT         = 500
BOOT_SEED      = 42
GRID_N         = 60
N_SPLINE_KNOTS = 4     # SplineTransformer(n_knots=4, degree=3) on log_pop
                        # gives 5 non-constant basis columns (~4 df after
                        # the model's own intercept absorbs one) -- verified
                        # empirically on this script's actual log_pop range,
                        # same "check the real column count, don't assume
                        # the generic knot-count formula" rule as 26's q-spline.
POP_REFERENCE_POINTS = [20_000, 100_000, 1_000_000]


def _shannon_normalized(counts: np.ndarray, n_cat: int) -> float:
    total = counts.sum()
    if total == 0:
        return np.nan
    p = counts[counts > 0] / total
    h = -np.sum(p * np.log(p))
    return float(h / np.log(n_cat))


def _within_city_dispersion(emb_zone: np.ndarray, types_zone: np.ndarray) -> tuple:
    """
    Full-sample (no subsampling) mean pairwise cosine distance among the
    embedding rows in one city-zone (V_within), and normalized Shannon
    entropy of the 17-type labels. NaN if fewer than 2 cells (V_within's
    own mathematical floor). Replaces the former rarefied/fixed-n version
    (methods review, 2026-10-05): cell count now feeds the reliability
    weight used at aggregation instead of being truncated to a fixed
    sample size, so cities with <20 cells in a zone are no longer
    excluded outright.
    """
    n = len(emb_zone)
    if n < 2:
        return np.nan, np.nan
    norms = np.linalg.norm(emb_zone, axis=1, keepdims=True)
    norms[norms < 1e-12] = np.nan
    En = emb_zone / norms
    sim = En @ En.T
    iu = np.triu_indices(n, k=1)
    v_val = float(np.nanmean(1.0 - sim[iu]))

    counts = np.bincount(types_zone.astype(int), minlength=N_TYPES + 1)[1:N_TYPES + 1]
    h_val = _shannon_normalized(counts, N_TYPES)
    return v_val, h_val


def _lowess_trend(log_pop: np.ndarray, delta: np.ndarray, grid: np.ndarray,
                  frac: float = LOWESS_FRAC, weight: np.ndarray = None) -> np.ndarray:
    """Weighted via integer row-repetition, same convention as
    23_scale_dependent_convergence.py's _lowess_trend (statsmodels'
    lowess() has no native weight argument)."""
    mask = np.isfinite(log_pop) & np.isfinite(delta)
    if weight is not None:
        mask &= np.isfinite(weight) & (weight > 0)
    if mask.sum() < 10:
        return np.full(len(grid), np.nan)
    x, y = log_pop[mask], delta[mask]
    if weight is not None:
        reps = np.maximum(np.round(weight[mask]).astype(int), 1)
        x = np.repeat(x, reps)
        y = np.repeat(y, reps)
    sm = lowess(y, x, frac=frac, return_sorted=True)
    return np.interp(grid, sm[:, 0], sm[:, 1])


def _country_fe_spline_trend(df: pd.DataFrame, grid: np.ndarray,
                             spline: SplineTransformer) -> np.ndarray:
    """
    Fit delta_V_within ~ spline(log_pop) + C(country) on df, then predict
    at each grid point as the AVERAGE MARGINAL prediction across df's
    observed country composition (predict once per row's actual country
    at that log_pop, then average) -- not a prediction for one arbitrary
    reference country. This removes the confound where a raw log_pop
    LOWESS trend can be driven by which countries happen to have large
    (or small) cities in the sample, rather than by population itself
    (methods review finding: the raw trend's apparent reversal to
    positive Delta_V_within near 1M population was partly a country-
    composition artifact).

    Returns NaN at every grid point if df has <2 countries or too few
    rows for the model to fit (e.g. inside a bootstrap draw where a
    resampled country set collapses the effective sample).

    Weighted by df's reliability_weight column (sqrt(min(n_core,
    n_outer))) if present -- both the model fit (WLS, not OLS) and the
    average-marginal-prediction step use this weight, so a low-cell-count
    city influences the fit and the prediction average proportionally
    less, consistent with 23_scale_dependent_convergence.py's weighting
    convention.
    """
    d = df.dropna(subset=["log_pop", "delta_V_within", "country"]).copy()
    if d["country"].nunique() < 2 or len(d) < 30:
        return np.full(len(grid), np.nan)
    has_weight = "reliability_weight" in d.columns
    w = d["reliability_weight"].values if has_weight else np.ones(len(d))

    X = spline.transform(d[["log_pop"]].values)
    for i in range(X.shape[1]):
        d[f"_sp{i}"] = X[:, i]
    formula = ("delta_V_within ~ " +
              " + ".join(f"_sp{i}" for i in range(X.shape[1])) +
              " + C(country)")
    try:
        model = smf.wls(formula, data=d, weights=w).fit()
    except Exception:
        return np.full(len(grid), np.nan)

    preds = np.full(len(grid), np.nan)
    for gi, log_p in enumerate(grid):
        Xg = spline.transform(np.array([[log_p]]))
        pred_df = d.copy()
        for i in range(Xg.shape[1]):
            pred_df[f"_sp{i}"] = Xg[0, i]
        try:
            row_preds = model.predict(pred_df)
            preds[gi] = np.average(row_preds, weights=w)
        except Exception:
            preds[gi] = np.nan
    return preds


def main():
    log.info("Loading cell_embeddings_index.parquet + cell_embeddings.npy ...")
    idx = pd.read_parquet(INDEX_PQ,
                          columns=["city", "country", "row_idx", "q_relative",
                                   "log_pop", "label_k17"])
    emb = np.load(EMB_NPY)
    with open(BIN_GRID) as f:
        q_max = float(json.load(f)["q_P95"])
    idx = idx[(idx["q_relative"] >= 0) & (idx["q_relative"] <= q_max)].copy()
    idx["_city_key"] = idx["country"] + "||" + idx["city"]
    idx = idx.dropna(subset=["log_pop"])
    log.info("  %d cells, q_max=%.4f", len(idx), q_max)

    # ── same qualifying-city sample as 23_scale_dependent_convergence.py ──
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
    city_weight = {ck: float(np.sqrt(min(core_n[ck], outer_n[ck]))) for ck in qualified}
    log.info("  reliability weight: median=%.1f, min=%.1f, max=%.1f",
            np.median(list(city_weight.values())), min(city_weight.values()), max(city_weight.values()))

    rng = np.random.default_rng(BOOT_SEED)
    log.info("Computing within-city dispersion (full sample, no rarefaction) ...")
    rows = []
    for ck in qualified:
        c_rows = core_sub[core_sub["_city_key"] == ck]
        o_rows = outer_sub[outer_sub["_city_key"] == ck]
        v_core, h_core = _within_city_dispersion(
            emb[c_rows["row_idx"].values], c_rows["label_k17"].values)
        v_outer, h_outer = _within_city_dispersion(
            emb[o_rows["row_idx"].values], o_rows["label_k17"].values)
        rows.append({
            "city_key": ck, "country": city_country[ck],
            "log_pop": float(city_log_pop[ck]), "pop": float(np.exp(city_log_pop[ck])),
            "reliability_weight": city_weight[ck],
            "V_within_core": v_core, "V_within_outer": v_outer,
            "H17_rare_core": h_core, "H17_rare_outer": h_outer,
            "delta_V_within": v_outer - v_core, "delta_H17_rare": h_outer - h_core,
        })
    main_df = pd.DataFrame(rows)
    main_df.to_csv(OUT_TABLES / "within_city_heterogeneity_dispersion.csv",
                   index=False, float_format="%.6f")
    log.info("Saved: within_city_heterogeneity_dispersion.csv (%d cities)", len(main_df))

    # ── PRIMARY trend: country-FE spline, average marginal prediction ──
    log_pop_grid = np.linspace(main_df["log_pop"].min(), main_df["log_pop"].max(), GRID_N)
    spline = SplineTransformer(n_knots=N_SPLINE_KNOTS, degree=3,
                               extrapolation="linear", include_bias=False)
    spline.fit(main_df[["log_pop"]].values)
    n_spline_cols = spline.transform(main_df[["log_pop"]].values[:1]).shape[1]
    log.info("  country-FE spline: n_knots=%d -> %d basis columns",
            N_SPLINE_KNOTS, n_spline_cols)
    delta_v_fe = _country_fe_spline_trend(main_df, log_pop_grid, spline)

    # ── SENSITIVITY trend: raw (unadjusted) LOWESS, no country FE ──
    delta_v_lowess = _lowess_trend(main_df["log_pop"].values, main_df["delta_V_within"].values, log_pop_grid,
                                   weight=main_df["reliability_weight"].values)
    delta_h_smooth = _lowess_trend(main_df["log_pop"].values, main_df["delta_H17_rare"].values, log_pop_grid,
                                   weight=main_df["reliability_weight"].values)

    # ── country-block bootstrap on the PRIMARY (country-FE) trend ──
    # Each draw refits the country-FE spline model on the resampled rows
    # (not just a LOWESS trend) -- the model formula already includes
    # C(country), so a country drawn K times contributing K duplicate
    # city-rows is fine here (unlike 23's KNN statistic, nothing here is
    # RE-DERIVED from inter-row distances where duplicate rows could
    # become each other's spurious nearest neighbour; OLS with repeated
    # rows is a standard, valid way to implement resampling weights).
    log.info("Bootstrapping (country-block, n=%d, country-FE spline refit) ...", N_BOOT)
    countries = sorted(set(city_country.values()))
    country_df = {c: main_df[main_df["country"] == c] for c in countries}
    boot_delta_v = np.full((N_BOOT, GRID_N), np.nan)
    for b in range(N_BOOT):
        chosen_countries = rng.choice(countries, size=len(countries), replace=True)
        # True country-block bootstrap: a country drawn K times contributes
        # its cities K times (preserves resampling multiplicity -- see the
        # methods-review fix already applied in 23_scale_dependent_
        # convergence.py for why naive set-based resampling is wrong here).
        resampled = pd.concat([country_df[c] for c in chosen_countries], ignore_index=True)
        if len(resampled) < 20:
            continue
        boot_delta_v[b] = _country_fe_spline_trend(resampled, log_pop_grid, spline)
        if (b + 1) % 100 == 0:
            log.info("  bootstrap draw %d/%d", b + 1, N_BOOT)

    delta_v_lo = np.nanpercentile(boot_delta_v, 2.5, axis=0)
    delta_v_hi = np.nanpercentile(boot_delta_v, 97.5, axis=0)

    trend_df = pd.DataFrame({
        "log_pop_grid": log_pop_grid, "pop_grid": np.exp(log_pop_grid),
        "delta_V_within_smooth": delta_v_fe,
        "delta_V_within_lo": delta_v_lo, "delta_V_within_hi": delta_v_hi,
        "delta_V_within_raw_lowess": delta_v_lowess,
        "delta_H17_rare_smooth": delta_h_smooth,
    })
    trend_df.to_csv(OUT_TABLES / "within_city_heterogeneity_trend.csv",
                    index=False, float_format="%.6f")
    log.info("Saved: within_city_heterogeneity_trend.csv")

    # ── reference-point predictions (20k / 100k / 1M), no zero-crossing ──
    ref_rows = []
    for pop_ref in POP_REFERENCE_POINTS:
        log_p = np.log(pop_ref)
        d_pred = float(np.interp(log_p, log_pop_grid, delta_v_fe))
        d_lo = float(np.interp(log_p, log_pop_grid, delta_v_lo))
        d_hi = float(np.interp(log_p, log_pop_grid, delta_v_hi))
        d_raw = float(np.interp(log_p, log_pop_grid, delta_v_lowess))
        ref_rows.append({"pop_ref": pop_ref, "delta_V_within": d_pred,
                        "delta_V_within_lo": d_lo, "delta_V_within_hi": d_hi,
                        "delta_V_within_raw_lowess": d_raw})
        log.info("  pop=%s: Delta_V_within(country-FE)=%.4f [%.4f, %.4f]  raw LOWESS=%.4f",
                f"{pop_ref:,}", d_pred, d_lo, d_hi, d_raw)
    ref_df = pd.DataFrame(ref_rows)
    ref_df.to_csv(OUT_TABLES / "within_city_heterogeneity_reference_points.csv",
                 index=False, float_format="%.6f")
    log.info("Saved: within_city_heterogeneity_reference_points.csv")
    print("\n=== Delta_V_within at reference populations (country-FE primary) ===")
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

    # Point size scales with reliability_weight, same convention as
    # 23_scale_dependent_convergence.py's panel a.
    point_size = 3 + 10 * (main_df["reliability_weight"] / main_df["reliability_weight"].max())
    ax.scatter(main_df["pop"], main_df["delta_V_within"], s=point_size, alpha=0.18,
              color="#293890", linewidth=0, zorder=2)
    ax.fill_between(np.exp(log_pop_grid), delta_v_lo, delta_v_hi,
                    color="#293890", alpha=0.15, linewidth=0, zorder=1)
    ax.plot(np.exp(log_pop_grid), delta_v_fe, color="#293890", lw=2.0, zorder=3,
           label="Country-FE spline (primary)")
    ax.plot(np.exp(log_pop_grid), delta_v_lowess, color="#293890", lw=1.2,
           ls="--", alpha=0.6, zorder=3, label="Raw LOWESS (sensitivity, no country FE)")
    ax.axhline(0, color="#AAAAAA", lw=0.8, ls="--")
    for pop_ref in POP_REFERENCE_POINTS:
        ax.axvline(pop_ref, color="#C8292A", lw=0.6, ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("City population", fontsize=9)
    ax.set_ylabel(r"$\Delta V_{within,c} = V_{within,outer} - V_{within,core}$" "\n(full cell sample per zone)",
                 fontsize=9)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("black")
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="black", labelsize=8)

    caption = (
        r"$\Delta V_{within}>0$: periphery internally more heterogeneous than core "
        "(wider mix of forms);" "\n"
        r"$\Delta V_{within}<0$: core internally more heterogeneous. Primary trend: "
        "country-fixed-effects" "\n"
        "spline, weighted average marginal prediction (controls for country "
        "composition differing by city" "\n"
        "size). V_within uses every cell in each city's zone (no rarefaction); "
        "cities get a reliability weight" "\n"
        "sqrt(min(n_core, n_outer)) at aggregation instead of a hard cell-count "
        "cutoff, so all qualifying" "\n"
        "cities contribute (down-weighted, not excluded, if their zone has few "
        "cells). Shaded band: 95%" "\n"
        "country-block bootstrap CI on the country-FE trend. Dotted lines: "
        "20k/100k/1M reference" "\n"
        "populations (see within_city_heterogeneity_reference_points.csv)."
    )
    fig.text(0.5, -0.02, caption, fontsize=5.5, color="#555555", style="italic",
            ha="center", va="top", linespacing=1.6)

    for ext in ("svg", "png"):
        out = OUT_FIG / f"within_city_heterogeneity.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", out)
    plt.close(fig)

    log.info("Done.")


if __name__ == "__main__":
    main()
