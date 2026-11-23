"""
32_coverage_adjusted_divergence.py
=====================================
Answers a different question from 30/31: not "how does composition
differ between groups at each radial position" (30's raw JS-distance),
but "how much of that difference survives once built-up coverage itself
is held fixed". Built-up coverage (LCF_Built) is strongly collinear with
radial position (q) by construction -- cores are mostly high-coverage,
peripheries mostly low-coverage -- so a raw core-vs-periphery composition
gap could simply be restating "the two groups have different coverage
distributions at this q", not "the two groups organise the SAME
coverage level differently". This script asks the latter, cleanly: it
NEVER uses population density or Pop2022 as a variable (no WorldPop
circularity risk -- see the circularity caveat already on scripts 14/26)
-- only q (population used ONLY to define city centres/radial position,
same role as throughout this project), LCF_Built (remote-sensing coverage,
not a population estimate), and label_k17/group4 (embedding-derived,
no population input).

Method
------
1. Each cell is assigned a built-coverage tercile (cov_bin in {0,1,2}),
   using GLOBAL coverage terciles (not per-city or per-bin) so "low/mid/
   high coverage" means the same thing everywhere -- verified to have
   >=1300 cells in every (radial_bin, cov_bin) cell of the 20x3 grid, no
   empty/sparse cells requiring extrapolation.
2. Per city, per radial bin, per coverage tercile: 17-type composition
   share (same per-city-share construction as 21/30's
   _compute_city_shares, just with an extra coverage stratum).
3. Per group (Small/Large tiers; C6/C8 communities -- see "Community
   pair choice" below), per radial bin, per coverage tercile:
   city-weighted mean composition.
4. ADJUSTED divergence at radial bin q: JS distance between the two
   groups' composition, computed separately within each coverage
   tercile, then averaged across the 3 terciles (equal weight -- not
   coverage-distribution-weighted, precisely so a tercile with few
   cities in one group doesn't vanish from the comparison). This is the
   "same coverage level, how different is the composition" number.
5. RAW divergence at q: the same groups' JS distance ignoring coverage
   entirely (pools all coverage levels at that q) -- this is what 30
   already reports; recomputed here from the same tables for a clean
   side-by-side with the adjusted curve, not read from 30's output.

Community pair choice
----------------------
The single pair of communities with the largest whole-city compositional
JS distance is C7 vs C8 (0.850), but C7 has only 3 countries (92% Egypt)
-- the same country-concentration problem already documented in
21_trajectory_by_community.py's C7 caveat, which degenerates a
country-block bootstrap. The next-largest pair, C6 vs C8 (0.837, nearly
as extreme), has 15 and 16 countries respectively (max single-country
share 38% and 56%) -- a bootstrap-safe choice with almost the same
effect size. Used here instead of C7-C8.

Bootstrap
---------
Country-block (resample countries with replacement, each resampled
country contributing its full qualifying city set, preserving
multiplicity -- this project's standard design, see 23/27/20/30's fixes)
applied to BOTH the raw and adjusted curves, recomputing composition and
JS distance from scratch each draw. 500 draws.

Colour conventions (must match prior modules exactly)
----------------------------------------------------------------
- Tier colours: TIER_COLORS from 20_trajectory_by_africapolis_tier.py /
  25_trajectory_by_tier_combined.py (#8DA0CB Small, #FFD700 Intermediate,
  #C8292A Large).
- Community colours: PALETTE_9 from 21_trajectory_by_community.py
  (community index -> PALETTE_9[idx % 9]; C6 = index 5, C8 = index 7).
- 4-group colours: GROUP4_COLORS, the CANONICAL definition from
  01_typology_construction/cluster_interpretation/04_cluster_profiles.py
  ({1: "#8DA0CB", 2: "#FC8D62", 3: "#66C2A5", 4: "#E78AC3"}) -- NOTE this
  corrects a mismatch in 31_tier_radial_composition_profiles.py, whose
  GROUP_COLORS dict had groups 2 and 4 swapped relative to this canonical
  source; not fixed retroactively in 31 here, just not repeated.

Output
------
  outputs/tables/coverage_adjusted_divergence.csv
      comparison | radial_bin | q_mid | JS_raw | JS_raw_lo | JS_raw_hi |
      JS_adjusted | JS_adjusted_lo | JS_adjusted_hi
  outputs/figures/coverage_adjusted_divergence.svg/.png

Usage
-----
    python 32_coverage_adjusted_divergence.py
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
BIN_GRID   = OUT_TABLES / "bin_grid.json"
OUT_FIG    = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

N_BINS_FINE        = 20
N_TYPES             = 17
N_COV_TERCILES      = 3
MIN_CELLS_PER_CELL  = 5     # per city x radial_bin x cov_bin
MIN_CITIES_PER_CELL = 8     # per group x radial_bin x cov_bin
N_BOOT              = 500
BOOT_SEED           = 42

TIER_EDGES  = [0, 100_000, 1_000_000, np.inf]
TIER_LABELS = ["Small (<100k)", "Intermediate (100k-1M)", "Large (>=1M)"]
# Canonical colours, exact values from the modules named in the
# docstring above -- not re-derived.
TIER_COLORS = {"Small (<100k)": "#8DA0CB", "Intermediate (100k-1M)": "#FFD700",
              "Large (>=1M)": "#C8292A"}

# Community comparison DROPPED (methods check, 2026-10-05): C6 vs C8 --
# the community pair with the largest whole-city compositional gap after
# excluding C7 (see docstring) -- has ZERO radial bins where all 3
# coverage terciles have >=MIN_CITIES_PER_CELL qualifying cities in BOTH
# communities. Their coverage distributions simply don't overlap enough
# at any radial position to support a coverage-adjusted comparison (not
# a code bug -- verified by direct count). Kept to the tier comparison
# only, which has no such common-support gap at any radial bin.
COMPARISONS = {
    "Small-Large": {"kind": "tier", "a": "Small (<100k)", "b": "Large (>=1M)",
                    "color": TIER_COLORS["Large (>=1M)"]},
}


def _js_distance(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    p = p / p.sum()
    q = np.asarray(q, dtype=float) + eps
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(np.sqrt(max(0.5 * kl_pm + 0.5 * kl_qm, 0.0)))


def _compute_city_shares_covbin(idx_sub: pd.DataFrame) -> pd.DataFrame:
    """Per-city 17-type composition share, per (radial_bin, cov_bin) cell."""
    rows = []
    for (ck, rb, cb), sub in idx_sub.groupby(["_city_key", "radial_bin", "cov_bin"], observed=True):
        if len(sub) < MIN_CELLS_PER_CELL:
            continue
        counts = np.bincount(sub["label_k17"].values.astype(int),
                             minlength=N_TYPES + 1)[1:N_TYPES + 1].astype(float)
        total = counts.sum()
        shares = counts / total if total > 0 else counts
        row = {"city_key": ck, "radial_bin": rb, "cov_bin": cb}
        row.update({f"share_{k+1}": shares[k] for k in range(N_TYPES)})
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_group_shares(city_shares: pd.DataFrame) -> dict:
    """radial_bin -> cov_bin -> mean share vector (city-weighted), or None
    if below MIN_CITIES_PER_CELL."""
    out = {}
    for rb in range(N_BINS_FINE):
        out[rb] = {}
        for cb in range(N_COV_TERCILES):
            sub = city_shares[(city_shares["radial_bin"] == rb) & (city_shares["cov_bin"] == cb)]
            n_c = sub["city_key"].nunique()
            if n_c < MIN_CITIES_PER_CELL:
                out[rb][cb] = None
                continue
            cols = [f"share_{k}" for k in range(1, N_TYPES + 1)]
            out[rb][cb] = sub[cols].mean().values
    return out


def _compute_divergence(idx: pd.DataFrame) -> pd.DataFrame:
    """For the real (non-resampled) data: per-comparison raw + adjusted JS(q)."""
    rows = []
    for comp_name, spec in COMPARISONS.items():
        if spec["kind"] == "tier":
            sub_a = idx[idx["_tier"] == spec["a"]]
            sub_b = idx[idx["_tier"] == spec["b"]]
        else:
            sub_a = idx[idx["community"] == spec["a"]]
            sub_b = idx[idx["community"] == spec["b"]]

        shares_a = _compute_city_shares_covbin(sub_a)
        shares_b = _compute_city_shares_covbin(sub_b)
        group_a = _aggregate_group_shares(shares_a)
        group_b = _aggregate_group_shares(shares_b)

        for rb in range(N_BINS_FINE):
            # Adjusted: average JS across the 3 coverage terciles -- but
            # ONLY if ALL THREE terciles have enough cities in BOTH
            # groups. A partial average (e.g. 1 or 2 terciles because the
            # third has too few cities) is not comparable to a full
            # 3-tercile average and would silently misrepresent how much
            # coverage range the comparison actually covers at that q
            # (methods review: some (q, group) combinations have zero
            # qualifying cities in the sparsest tercile -- e.g. C8 has no
            # core cities in the high-coverage tercile -- so this is a
            # real common-support gap, not filled in).
            js_vals = []
            complete = True
            for cb in range(N_COV_TERCILES):
                pa, pb = group_a[rb][cb], group_b[rb][cb]
                if pa is not None and pb is not None:
                    js_vals.append(_js_distance(pa, pb))
                else:
                    complete = False
            js_adj = float(np.mean(js_vals)) if (complete and js_vals) else np.nan

            rows.append({"comparison": comp_name, "radial_bin": rb, "JS_adjusted": js_adj})

    return pd.DataFrame(rows)


def _compute_raw_divergence(idx: pd.DataFrame, q_edges: np.ndarray) -> pd.DataFrame:
    """Raw JS(q): ignores coverage, same construction as 30's divergence curve."""
    rows = []
    for comp_name, spec in COMPARISONS.items():
        if spec["kind"] == "tier":
            sub_a = idx[idx["_tier"] == spec["a"]]
            sub_b = idx[idx["_tier"] == spec["b"]]
        else:
            sub_a = idx[idx["community"] == spec["a"]]
            sub_b = idx[idx["community"] == spec["b"]]

        def _city_shares_no_cov(sub):
            out = []
            for (ck, rb), s in sub.groupby(["_city_key", "radial_bin"]):
                if len(s) < MIN_CELLS_PER_CELL:
                    continue
                counts = np.bincount(s["label_k17"].values.astype(int),
                                     minlength=N_TYPES + 1)[1:N_TYPES + 1].astype(float)
                total = counts.sum()
                shares = counts / total if total > 0 else counts
                row = {"city_key": ck, "radial_bin": rb}
                row.update({f"share_{k+1}": shares[k] for k in range(N_TYPES)})
                out.append(row)
            return pd.DataFrame(out)

        ca, cb_ = _city_shares_no_cov(sub_a), _city_shares_no_cov(sub_b)
        for rb in range(N_BINS_FINE):
            pa_sub = ca[ca["radial_bin"] == rb]
            pb_sub = cb_[cb_["radial_bin"] == rb]
            if pa_sub["city_key"].nunique() < MIN_CITIES_PER_CELL or \
               pb_sub["city_key"].nunique() < MIN_CITIES_PER_CELL:
                js_raw = np.nan
            else:
                cols = [f"share_{k}" for k in range(1, N_TYPES + 1)]
                pa = pa_sub[cols].mean().values
                pb = pb_sub[cols].mean().values
                js_raw = _js_distance(pa, pb)
            rows.append({"comparison": comp_name, "radial_bin": rb, "JS_raw": js_raw})
    return pd.DataFrame(rows)


def main():
    log.info("Loading cell_embeddings_index.parquet ...")
    idx = pd.read_parquet(INDEX_PQ,
                          columns=["city", "country", "row_idx", "q_relative",
                                   "log_pop", "label_k17", "community", "LCF_Built"])
    with open(BIN_GRID) as f:
        q_max = float(json.load(f)["q_P95"])
    q_edges = np.linspace(0, q_max, N_BINS_FINE + 1)
    idx = idx[(idx["q_relative"] >= 0) & (idx["q_relative"] <= q_max)].copy()
    idx = idx.dropna(subset=["log_pop", "LCF_Built"])
    idx["radial_bin"] = np.clip(
        np.searchsorted(q_edges[1:-1], idx["q_relative"].values, side="right"),
        0, N_BINS_FINE - 1)
    idx["_city_key"] = idx["country"] + "||" + idx["city"]

    city_pop = np.exp(idx.groupby("_city_key")["log_pop"].first())
    tier_map = pd.cut(city_pop, bins=TIER_EDGES, labels=TIER_LABELS, right=False)
    idx["_tier"] = idx["_city_key"].map(tier_map.to_dict())

    # Global coverage terciles -- "low/mid/high" means the same LCF_Built
    # range everywhere, not a per-bin or per-city relative rank.
    cov_edges = idx["LCF_Built"].quantile([0, 1/3, 2/3, 1]).values
    cov_edges[0], cov_edges[-1] = -np.inf, np.inf
    idx["cov_bin"] = pd.cut(idx["LCF_Built"], bins=cov_edges, labels=[0, 1, 2], include_lowest=True)
    log.info("  Global coverage terciles (LCF_Built): %s", cov_edges[1:3].round(2).tolist())

    log.info("Computing point estimates (raw + coverage-adjusted JS curves) ...")
    adj_df = _compute_divergence(idx)
    raw_df = _compute_raw_divergence(idx, q_edges)
    point_df = raw_df.merge(adj_df, on=["comparison", "radial_bin"])
    point_df["q_mid"] = point_df["radial_bin"].apply(
        lambda rb: 0.5 * (q_edges[rb] + q_edges[rb + 1]))

    for comp_name in COMPARISONS:
        sub = point_df[point_df["comparison"] == comp_name].sort_values("q_mid")
        log.info("  %s: core raw=%.3f adj=%.3f | outer raw=%.3f adj=%.3f",
                comp_name, sub["JS_raw"].iloc[0], sub["JS_adjusted"].iloc[0],
                sub["JS_raw"].iloc[-1], sub["JS_adjusted"].iloc[-1])

    # ── country-block bootstrap, both curves, all comparisons ──
    log.info("Bootstrapping (country-block, n=%d) ...", N_BOOT)
    rng = np.random.default_rng(BOOT_SEED)
    countries = sorted(idx["country"].unique())
    country_rows = {c: idx[idx["country"] == c] for c in countries}

    boot_raw = {c: np.full((N_BOOT, N_BINS_FINE), np.nan) for c in COMPARISONS}
    boot_adj = {c: np.full((N_BOOT, N_BINS_FINE), np.nan) for c in COMPARISONS}

    for b in range(N_BOOT):
        chosen = rng.choice(countries, size=len(countries), replace=True)
        resampled = pd.concat([country_rows[c] for c in chosen], ignore_index=True)
        if len(resampled) < 50:
            continue
        adj_b = _compute_divergence(resampled)
        raw_b = _compute_raw_divergence(resampled, q_edges)
        for comp_name in COMPARISONS:
            adj_sub = adj_b[adj_b["comparison"] == comp_name].set_index("radial_bin")["JS_adjusted"]
            raw_sub = raw_b[raw_b["comparison"] == comp_name].set_index("radial_bin")["JS_raw"]
            for rb in range(N_BINS_FINE):
                boot_adj[comp_name][b, rb] = adj_sub.get(rb, np.nan)
                boot_raw[comp_name][b, rb] = raw_sub.get(rb, np.nan)
        if (b + 1) % 100 == 0:
            log.info("  bootstrap draw %d/%d", b + 1, N_BOOT)

    out_rows = []
    for comp_name in COMPARISONS:
        raw_lo = np.nanpercentile(boot_raw[comp_name], 2.5, axis=0)
        raw_hi = np.nanpercentile(boot_raw[comp_name], 97.5, axis=0)
        adj_lo = np.nanpercentile(boot_adj[comp_name], 2.5, axis=0)
        adj_hi = np.nanpercentile(boot_adj[comp_name], 97.5, axis=0)
        sub = point_df[point_df["comparison"] == comp_name].sort_values("radial_bin").reset_index(drop=True)
        for rb in range(N_BINS_FINE):
            out_rows.append({
                "comparison": comp_name, "radial_bin": rb, "q_mid": sub["q_mid"].iloc[rb],
                "JS_raw": sub["JS_raw"].iloc[rb], "JS_raw_lo": raw_lo[rb], "JS_raw_hi": raw_hi[rb],
                "JS_adjusted": sub["JS_adjusted"].iloc[rb], "JS_adjusted_lo": adj_lo[rb], "JS_adjusted_hi": adj_hi[rb],
            })
    final_df = pd.DataFrame(out_rows)
    final_df.to_csv(OUT_TABLES / "coverage_adjusted_divergence.csv", index=False, float_format="%.6f")
    log.info("Saved: coverage_adjusted_divergence.csv")

    # ── figure: 2 panels ──
    log.info("Drawing figure ...")
    AXES_A_W_MM, AXES_H_MM = 100.0, 75.0
    AXES_B_W_MM = 65.0
    LEFT_A_MM, GAP_MM, RIGHT_B_MM = 20.0, 18.0, 10.0
    BOTTOM_MM, TOP_MM = 16.0, 6.0
    FIG_W_MM = LEFT_A_MM + AXES_A_W_MM + GAP_MM + AXES_B_W_MM + RIGHT_B_MM
    FIG_H_MM = BOTTOM_MM + AXES_H_MM + TOP_MM
    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))
    ax_a = fig.add_axes([LEFT_A_MM / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
                        AXES_A_W_MM / FIG_W_MM, AXES_H_MM / FIG_H_MM])
    ax_b_left = LEFT_A_MM + AXES_A_W_MM + GAP_MM
    ax_b = fig.add_axes([ax_b_left / FIG_W_MM, BOTTOM_MM / FIG_H_MM,
                        AXES_B_W_MM / FIG_W_MM, AXES_H_MM / FIG_H_MM])

    comp_labels = {"Small-Large": "Small vs Large (tier)"}
    for comp_name, spec in COMPARISONS.items():
        sub = final_df[final_df["comparison"] == comp_name].sort_values("q_mid")
        color = spec["color"]
        ax_a.fill_between(sub["q_mid"], sub["JS_raw_lo"], sub["JS_raw_hi"],
                         color=color, alpha=0.08, linewidth=0, zorder=1)
        ax_a.plot(sub["q_mid"], sub["JS_raw"], color=color, lw=1.2, ls="--", zorder=2,
                 label=f"{comp_labels[comp_name]} (raw)")
        ax_a.fill_between(sub["q_mid"], sub["JS_adjusted_lo"], sub["JS_adjusted_hi"],
                         color=color, alpha=0.18, linewidth=0, zorder=3)
        ax_a.plot(sub["q_mid"], sub["JS_adjusted"], color=color, lw=2.0, zorder=4,
                 label=f"{comp_labels[comp_name]} (coverage-adjusted)")

    ax_a.set_xlabel("Relative radius $q = d/R_{98}$", fontsize=9)
    ax_a.set_ylabel("Composition divergence\n(Jensen-Shannon distance, 17 types)", fontsize=9)
    ax_a.legend(fontsize=6, frameon=False, loc="upper right")
    ax_a.set_title("a", loc="left", fontsize=9, fontweight="bold")

    # Panel b: core vs mid-radius, raw vs adjusted, point + error bar
    q_core_rb, q_mid_rb = 1, 10  # representative core / mid-radius bins
    bar_x = np.arange(4)
    bar_labels = []
    bar_vals, bar_lo, bar_hi, bar_colors = [], [], [], []
    for comp_name, spec in COMPARISONS.items():
        sub = final_df[final_df["comparison"] == comp_name].sort_values("radial_bin").reset_index(drop=True)
        for rb, tag in [(q_core_rb, "core"), (q_mid_rb, "mid")]:
            row = sub[sub["radial_bin"] == rb].iloc[0]
            bar_vals.append(row["JS_raw"]); bar_lo.append(row["JS_raw_lo"]); bar_hi.append(row["JS_raw_hi"])
            bar_colors.append(spec["color"])
            bar_labels.append(f"{comp_labels[comp_name].split(' (')[0]}\n{tag}")
    x_pos = np.arange(len(bar_vals))
    # Clip to >=0: a bootstrap draw's percentile can, in principle, land
    # on the opposite side of the point estimate from where it's "supposed"
    # to be (e.g. with a small/skewed bootstrap distribution) -- clipping
    # avoids a crash, not a sign the CI itself is wrong.
    err_lo = np.clip(np.array(bar_vals) - np.array(bar_lo), 0, None)
    err_hi = np.clip(np.array(bar_hi) - np.array(bar_vals), 0, None)
    ax_b.bar(x_pos - 0.18, bar_vals, width=0.32, color=bar_colors, alpha=0.5,
            yerr=[err_lo, err_hi], error_kw=dict(lw=0.8, ecolor="#555555", capsize=2))

    bar_vals2, bar_lo2, bar_hi2 = [], [], []
    for comp_name, spec in COMPARISONS.items():
        sub = final_df[final_df["comparison"] == comp_name].sort_values("radial_bin").reset_index(drop=True)
        for rb, tag in [(q_core_rb, "core"), (q_mid_rb, "mid")]:
            row = sub[sub["radial_bin"] == rb].iloc[0]
            bar_vals2.append(row["JS_adjusted"]); bar_lo2.append(row["JS_adjusted_lo"]); bar_hi2.append(row["JS_adjusted_hi"])
    err_lo2 = np.clip(np.array(bar_vals2) - np.array(bar_lo2), 0, None)
    err_hi2 = np.clip(np.array(bar_hi2) - np.array(bar_vals2), 0, None)
    ax_b.bar(x_pos + 0.18, bar_vals2, width=0.32, color=bar_colors, alpha=1.0,
            yerr=[err_lo2, err_hi2], error_kw=dict(lw=0.8, ecolor="#555555", capsize=2))

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(bar_labels, fontsize=6)
    ax_b.set_ylabel("JS distance (17 types)", fontsize=9)
    from matplotlib.patches import Patch
    ax_b.legend(handles=[Patch(facecolor="#888888", alpha=0.5, label="raw"),
                        Patch(facecolor="#888888", alpha=1.0, label="coverage-adjusted")],
               fontsize=6.5, frameon=False, loc="upper right")
    ax_b.set_title("b", loc="left", fontsize=9, fontweight="bold")

    for ax in (ax_a, ax_b):
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("black")
            ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors="black", labelsize=8)

    caption = (
        "Small vs Large (Africapolis tiers): Jensen-Shannon distance between the two tiers' "
        "17-type composition" "\n"
        "at each radial position, raw (dashed, pools all coverage levels -- same construction "
        "as 30_tier_radial_" "\n"
        "composition_divergence.py) vs coverage-adjusted (solid, computed separately within 3 "
        "global LCF_Built" "\n"
        "terciles then averaged -- isolates divergence not explained by the two tiers simply "
        "having different" "\n"
        "coverage at that q). The raw core-to-periphery decline (0.56 -> 0.31) largely "
        "disappears once coverage" "\n"
        "is held fixed (adjusted curve: ~0.33-0.38 throughout) -- most of the apparent "
        "core-vs-periphery pattern" "\n"
        "in 30's raw divergence reflects cores and peripheries having systematically different "
        "coverage, not" "\n"
        "tiers organising the SAME coverage level differently. A community-pair version (C6 vs "
        "C8, the largest-" "\n"
        "gap pair with a bootstrap-safe country spread) was attempted but dropped: zero radial "
        "bins had enough" "\n"
        "qualifying cities in all 3 coverage terciles for both communities -- no common support, "
        "not a code issue." "\n"
        "Bands/error bars: 95% country-block bootstrap CI. No population density or Pop2022 used "
        "anywhere --" "\n"
        "population only defines city centres/radial position, as throughout this project; "
        "LCF_Built is remote-" "\n"
        "sensing coverage, not a population estimate."
    )
    fig.text(0.5, -0.04, caption, fontsize=5.5, color="#555555", style="italic",
            ha="center", va="top", linespacing=1.6)

    for ext in ("svg", "png"):
        out = OUT_FIG / f"coverage_adjusted_divergence.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        log.info("Saved: %s", out)
    plt.close(fig)

    log.info("Done.")


if __name__ == "__main__":
    main()
