"""
10_figure_main.py
=================
Main 3-panel figure for the morphological transition analysis.

Layout (three-column, single row):

  ┌──────────────────────────┬──────────────────┬──────────────┐
  │  A  Core restructuring   │  B  Spatial       │  C  Regional │
  │     & convergence        │     partitioning  │     context  │
  │  T1/T4 ribbons           │  β_H(u) top       │  R²_climate  │
  │  + S(u)                  │  β_r(u) bottom    │  R²_colonial │
  │  + C(u) twin-axis        │                   │              │
  └──────────────────────────┴──────────────────┴──────────────┘

Panels
------
A  Core restructuring and convergence:
   Top:    T1 and T4 stacked-area ribbons (mean type share, core→periphery).
   Middle: S(u) = JS[P_T4(u), P_T1(u)], compositional restructuring strength.
           S_core/S_outer ≈ 2.12 annotated.
   Bottom: C(u) = |β_JS(u)|, cross-city convergence intensity with city size.
           C_core/C_outer = 2.26 [bootstrap CI] annotated.

B  Spatial partitioning of size effects:
   B1: β_H(u) diversity scaling per bin, ±95% CI. R_H = 1.58 [1.28, 2.06].
   B2: β_r(u) assortativity scaling per bin, ±95% CI. R_r = 1.74 [1.53, 2.14].

C  Regional/environmental context:
   Partial R²_climate(u) and R²_colonial(u) on JS divergence per bin.
   Climate R² rises core→outer; colonial R² is highest at core.

Outputs
-------
  outputs/figures/<YYYY-MM-DD>/figure_main.pdf
  outputs/figures/<YYYY-MM-DD>/figure_main.png  (300 dpi)
"""

import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from pathlib import Path
from datetime import date

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _regional_effect_shared import regional_effect_by_bin

ROOT       = Path(__file__).parent
OUT_TABLES = ROOT / "outputs" / "tables"
ROBUST_DIR = OUT_TABLES / "robustness"
FIG_DATE   = date.today().strftime("%Y-%m-%d")
OUT_FIG    = ROOT / "outputs" / "figures" / FIG_DATE
OUT_FIG.mkdir(parents=True, exist_ok=True)

CM_PATH    = ROOT.parent.parent / "03_urban_hierarchy" / "outputs" / "tables" / "city_metrics.parquet"
AI_PATH    = ROOT.parent.parent / "02_regional_structure" / "regression" / "outputs" / "tables" / "analysis_input.csv"

# ── Constants ──────────────────────────────────────────────────────────────

N_TYPES    = 17
N_BINS     = 10
ZONE_CORE  = [0, 1]
ZONE_MID   = [2, 3, 4, 5]
ZONE_OUTER = [6, 7, 8, 9]

# 17-type palette — canonical PALETTE_17 from 0_embedding_visualization/01_tsne.py
TYPE_COLORS = {
    1:  "#DBDBDB",
    2:  "#FEEFCF",
    3:  "#FCBBC5",
    4:  "#F5E6C8",
    5:  "#E5152C",
    6:  "#FFFF4C",
    7:  "#5BA04A",
    8:  "#FBB732",
    9:  "#6F391F",
    10: "#B0E1E7",
    11: "#7B67EE",
    12: "#E89679",
    13: "#58C4EC",
    14: "#828F9B",
    15: "#1C1C5F",
    16: "#992FA3",
    17: "#C49A3C",
}

# Zone colours
COL_CORE  = "#2471A3"
COL_MID   = "#27AE60"
COL_OUTER = "#888888"
ZONE_ALPHA_CORE = 0.06
ZONE_ALPHA_MID  = 0.035

# ── Morphological family definitions (from 01_compute_city_metrics.py) ────
# Family 1 — Medium-density built-up fabric : types 2,9,10,11,16,17
# Family 2 — High-density compact urban     : types 12,13,14,15
# Family 3 — Sparse low-built vegetated     : types 4,5
# Family 4 — Low-density heterogeneous      : types 1,3,6,7,8
TYPE_FAMILY = {}
for t in [2, 9, 10, 11, 16, 17]: TYPE_FAMILY[t] = 1
for t in [12, 13, 14, 15]:       TYPE_FAMILY[t] = 2
for t in [4, 5]:                  TYPE_FAMILY[t] = 3
for t in [1, 3, 6, 7, 8]:        TYPE_FAMILY[t] = 4

FAMILY_LABELS = {
    1: "Medium-density\nbuilt-up",
    2: "High-density\ncompact",
    3: "Sparse\nvegetated",
    4: "Low-density\nheterogeneous",
}
FAMILY_COLORS = {1: "#4575B4", 2: "#D73027", 3: "#91BFDB", 4: "#FEE090"}

# Short type labels for heatmap y-axis
TYPE_SHORT = {
    1:  "T1  sparse-open",
    2:  "T2  med-density",
    3:  "T3  low-irreg",
    4:  "T4  sparse-veg",
    5:  "T5  bare-veg",
    6:  "T6  low-grid",
    7:  "T7  low-organic",
    8:  "T8  low-mixed",
    9:  "T9  med-organic",
    10: "T10 med-grid",
    11: "T11 med-compact",
    12: "T12 hi-compact",
    13: "T13 hi-dense",
    14: "T14 hi-formal",
    15: "T15 hi-intense",
    16: "T16 med-formal",
    17: "T17 med-mixed",
}


# ── JS helper ──────────────────────────────────────────────────────────────

def _js_divergence(p, q, eps=1e-9):
    """Symmetric JS divergence between two probability vectors."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _shannon(p, eps=1e-9):
    """Normalised Shannon entropy (base 17)."""
    p = np.asarray(p, dtype=float)
    p = p[p > eps]
    p = p / p.sum()
    return -np.sum(p * np.log(p)) / np.log(N_TYPES)


# ── Zone helpers ───────────────────────────────────────────────────────────

def _zone_shade(ax):
    ax.axvspan(-0.5, 1.5, alpha=ZONE_ALPHA_CORE, color=COL_CORE, zorder=0)
    ax.axvspan(1.5,  5.5, alpha=ZONE_ALPHA_MID,  color=COL_MID,  zorder=0)


def _zone_labels(ax, fontsize=6.5):
    for xpos, label, color in [
        (0.5, "Core",  COL_CORE),
        (3.5, "Mid",   COL_MID),
        (7.5, "Outer", COL_OUTER),
    ]:
        ax.text(xpos, 1.015, label, ha="center", va="bottom",
                fontsize=fontsize, color=color, clip_on=False,
                transform=ax.get_xaxis_transform())


# ── Data loading & pre-computation ─────────────────────────────────────────

def _load_all():
    fm    = pd.read_csv(OUT_TABLES / "formal_models.csv")
    sdt   = pd.read_csv(OUT_TABLES / "spatial_division_test.csv",
                        index_col="statistic")
    cd    = pd.read_csv(OUT_TABLES / "convergence_decay.csv")
    cbp   = pd.read_parquet(OUT_TABLES / "city_bin_profiles.parquet")
    cbdiv = pd.read_parquet(OUT_TABLES / "city_bin_diversity.parquet")
    cba   = pd.read_parquet(OUT_TABLES / "city_bin_assortativity.parquet")
    tdd   = pd.read_csv(OUT_TABLES / "tier_divergence_profile.csv")
    sep   = pd.read_csv(OUT_TABLES / "size_effect_profiles.csv")
    ai    = pd.read_csv(AI_PATH, usecols=["city", "country", "climate_zone", "colonial",
                                          "log_gdp", "log_pop_worldpop_2022"]
                        ).rename(columns={"log_pop_worldpop_2022": "log_pop"})
    return fm, sdt, cd, cbp, cbdiv, cba, tdd, sep, ai


def _panel_A_data(cbp, sep):
    """
    Prepare data for Panel A (four ribbons + S(u)).

    Returns:
      ribbons      : dict {"T1": df, "T2": df, "T3": df, "T4": df}
                     each df has index=radial_bin, cols=p_01..p_17
      heatmap_matrix : ndarray (n_types × n_bins), β_k(u)
      sig_matrix     : ndarray (n_types × n_bins), bool: q<0.05
      type_order     : list of type IDs in display order
      S_u            : array length n_bins, JS[P_T4, P_T1] per bin
    """
    p_cols = [f"p_{k:02d}" for k in range(1, N_TYPES + 1)]

    # Tier ribbon means (T1–T4)
    ribbons = {}
    for tier in ["T1", "T2", "T3", "T4"]:
        ribbons[tier] = (cbp[cbp["tier"] == tier]
                         .groupby("radial_bin")[p_cols].mean()
                         .reindex(range(N_BINS)))

    # S(u) = JS[P_T4, P_T1]
    S_u = np.array([
        _js_divergence(
            ribbons["T1"].iloc[b].values,
            ribbons["T4"].iloc[b].values
        ) for b in range(N_BINS)
    ])

    # Type ordering: family → mid-zone β desc
    mid_beta = (sep[sep["radial_bin"].isin(ZONE_MID)]
                .groupby("type17")["beta"].mean())
    type_order = []
    for fam in [1, 2, 3, 4]:
        types_in_fam = [t for t, f in TYPE_FAMILY.items() if f == fam]
        types_sorted = sorted(types_in_fam,
                              key=lambda t: mid_beta.get(t, 0),
                              reverse=True)
        type_order.extend(types_sorted)

    # Build heatmap matrix (n_types × n_bins)
    n_types = len(type_order)
    heatmap_matrix = np.full((n_types, N_BINS), np.nan)
    sig_matrix     = np.zeros((n_types, N_BINS), dtype=bool)
    for ri, t in enumerate(type_order):
        for b in range(N_BINS):
            row = sep[(sep["type17"] == t) & (sep["radial_bin"] == b)]
            if len(row) == 1:
                heatmap_matrix[ri, b] = row["beta"].values[0]
                sig_matrix[ri, b]     = row["q_value_bh"].values[0] < 0.05

    return ribbons, heatmap_matrix, sig_matrix, type_order, S_u


def _tier_metric_profiles(cbdiv, cba):
    """
    Returns per-bin mean H17 and r17 for T1 and T4.
    cbdiv: city_bin_diversity, cba: city_bin_assortativity
    """
    p_cols_h = [c for c in cbdiv.columns if c not in ["city","country","tier","radial_bin","log_pop"]]
    p_cols_r = [c for c in cba.columns if c not in ["city","country","tier","radial_bin","log_pop","qc30"]]

    result = {}
    for tier in ["T1", "T4"]:
        # H17
        sub_h = cbdiv[cbdiv["tier"] == tier]
        if "H17" in sub_h.columns:
            h_prof = sub_h.groupby("radial_bin")["H17"].mean()
        elif "rarefied_H17" in sub_h.columns:
            h_prof = sub_h.groupby("radial_bin")["rarefied_H17"].mean()
        else:
            # compute from p_01..p_17 if available
            h_prof = None

        # r17
        sub_r = cba[cba["tier"] == tier]
        r_col = "r17" if "r17" in sub_r.columns else None
        r_prof = sub_r.groupby("radial_bin")[r_col].mean() if r_col else None

        result[tier] = {"H": h_prof, "r": r_prof}

    return result


def _panel_D_data(cbp, ai):
    """
    Partial R^2 of climate and colonial legacy on the 17-type composition
    PCoA space, net of population, per radial bin. Delegates to the same
    `regional_effect_by_bin()` used by 06_regional_effect_by_distance.py,
    so this figure and that script's table report identical numbers.

    Returns (bins, r2_climate, r2_colonial, n_obs) — the same shape this
    function has always returned, for compatibility with `_panel_D` below.
    """
    result = regional_effect_by_bin(
        profiles=cbp,
        covariates=ai,
        composition_cols=[f"p_{k:02d}" for k in range(1, N_TYPES + 1)],
        compute_ci=False,  # figure only needs point estimates; the
                            # bootstrap CIs are reported in
                            # regional_effect_profile.csv (06_*.py)
    )
    bins = result["radial_bin"].to_numpy()
    r2_climate = result["partial_r2_climate"].to_numpy()
    r2_colonial = result["partial_r2_colonial"].to_numpy()
    n_obs = result["n_cities"].to_numpy()
    return bins, r2_climate, r2_colonial, n_obs


# ── Panel A: T1/T2/T3/T4 ribbons + S(u) ──────────────────────────────────

# Tier display metadata: (tier_key, short_label, tier_size_hint)
_TIER_META = [
    ("T1", "T1  (<20 k)",   "#F5F5F5"),  # lightest bg badge
    ("T2", "T2  (20–200 k)","#E8F4F8"),
    ("T3", "T3  (200 k–1 M)","#D6EAF8"),
    ("T4", "T4  (≥1 M)",    "#1A5276"),  # darkest — for white text contrast
]


def _draw_ribbon(ax, ribbon, label, badge_dark=False, show_yticks=True,
                 show_xticks=False, show_zone_labels=False):
    """Draw one stacked-area ribbon on ax."""
    x = np.arange(N_BINS)
    stack_order = list(range(N_TYPES, 0, -1))
    bottom = np.zeros(N_BINS)
    for k in stack_order:
        col = f"p_{k:02d}"
        vals = ribbon[col].values if col in ribbon.columns else np.zeros(N_BINS)
        ax.fill_between(x, bottom, bottom + vals,
                        color=TYPE_COLORS[k], linewidth=0)
        bottom += vals
    _zone_shade(ax)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(0, 1)
    if show_yticks:
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", ".5", "1"], fontsize=6.0)
    else:
        ax.set_yticks([])
    if show_xticks:
        ax.set_xticks(range(10))
        ax.set_xticklabels([str(b) for b in range(10)], fontsize=6.0)
    else:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    txt_color = "white" if badge_dark else "#111111"
    bg_color  = "#1A5276" if badge_dark else "#EEEEEE"
    ax.text(0.012, 0.93, label, transform=ax.transAxes,
            fontsize=7.5, fontweight="bold", va="top", ha="left",
            color=txt_color,
            bbox=dict(boxstyle="round,pad=0.16", fc=bg_color,
                      ec="none", alpha=0.82))
    if show_zone_labels:
        _zone_labels(ax, fontsize=7)


def _city_schematic(ax):
    """
    Draw a schematic concentric-ring city diagram showing Core / Mid / Outer zones.
    Placed as an inset in Panel A (top-right corner of T1 ribbon).
    """
    ax.set_aspect("equal")
    ax.axis("off")

    # Radii represent scaled distance bins: 0-1=Core, 2-5=Mid, 6-9=Outer
    # Draw filled annuli from outside in
    for r_outer, r_inner, color, alpha, lbl, lbl_r in [
        (1.00, 0.55, COL_OUTER, 0.18, "Outer\n(bins 6–9)", 0.78),
        (0.55, 0.28, COL_MID,   0.25, "Mid\n(bins 2–5)",  0.415),
        (0.28, 0.00, COL_CORE,  0.40, "Core\n(bins 0–1)", 0.14),
    ]:
        theta = np.linspace(0, 2 * np.pi, 200)
        # Outer ring
        xo = r_outer * np.cos(theta)
        yo = r_outer * np.sin(theta)
        xi = r_inner * np.cos(theta[::-1])
        yi = r_inner * np.sin(theta[::-1])
        ax.fill(
            np.concatenate([xo, xi]),
            np.concatenate([yo, yi]),
            color=color, alpha=alpha, linewidth=0
        )
        # Zone label — right side
        ax.text(lbl_r, 0, lbl, ha="center", va="center",
                fontsize=4.5, color=color, fontweight="bold",
                linespacing=1.2)

    # Radial tick marks at bin boundaries
    for r, lbl in [(0.28, ""), (0.55, "")]:
        theta_t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(r * np.cos(theta_t), r * np.sin(theta_t),
                color="white", lw=0.8, alpha=0.7)

    # Scaled-distance axis arrow
    ax.annotate("", xy=(0, -1.08), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.9))
    ax.text(0, -1.16, r"$u$ (scaled dist.)", ha="center", va="top",
            fontsize=4.2, color="#555555")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.25, 1.2)


def _panel_A(ribbon_axes, ax_su, ribbons, S_u):
    """
    ribbon_axes : list of 4 axes [ax_T1, ax_T2, ax_T3, ax_T4]
    ax_su       : S(u) curve axis (bottom)
    ribbons     : dict {"T1":df, "T2":df, "T3":df, "T4":df}
    S_u         : array length N_BINS
    """
    tiers_in_order = ["T1", "T2", "T3", "T4"]
    labels = ["T1  (<20 k)", "T2  (20–200 k)", "T3  (200 k–1 M)", "T4  (≥1 M)"]

    for i, (ax, tier, lbl) in enumerate(zip(ribbon_axes, tiers_in_order, labels)):
        is_first = (i == 0)
        _draw_ribbon(ax, ribbons[tier], lbl,
                     badge_dark=(tier == "T4"),
                     show_yticks=True,
                     show_xticks=False,
                     show_zone_labels=is_first)

    # City schematic inset — placed in S(u) subplot at far right
    # (drawn after ax_su is set up, so we defer to after S(u) section;
    #  store reference here and draw later via return or pass ax_su to panel)

    # Type legend — placed below T4 ribbon, above S(u)
    patches = [mpatches.Patch(color=TYPE_COLORS[k], label=str(k))
               for k in range(1, N_TYPES + 1)]
    ribbon_axes[-1].legend(
        handles=patches, ncol=17, fontsize=4.2,
        loc="upper left", bbox_to_anchor=(0.0, -0.14),
        title="Morphotype", title_fontsize=4.8,
        framealpha=0.0, handlelength=0.75, handletextpad=0.25,
        columnspacing=0.35, borderpad=0.1)

    # ── S(u) curve ───────────────────────────────────────────────────────
    x = np.arange(N_BINS)
    col_s = "#6C3483"
    ax_su.fill_between(x, 0, S_u, color=col_s, alpha=0.18)
    ax_su.plot(x, S_u, "o-", color=col_s, linewidth=2.0, markersize=5, zorder=3)

    S_core  = float(np.nanmean(S_u[np.array(ZONE_CORE)]))
    S_outer = float(np.nanmean(S_u[np.array(ZONE_OUTER)]))
    S_ratio = S_core / S_outer

    # Zone mean dashes (subtle)
    ax_su.plot([ZONE_CORE[0],  ZONE_CORE[-1]  + 1], [S_core,  S_core],
               "--", color=COL_CORE,  lw=1.0, alpha=0.6)
    ax_su.plot([ZONE_OUTER[0], ZONE_OUTER[-1] + 1], [S_outer, S_outer],
               "--", color=COL_OUTER, lw=1.0, alpha=0.6)

    # Prominent callout box — upper-right area
    ax_su.text(0.97, 0.93,
               f"Core / Outer\n"
               f"$\\times${S_ratio:.2f}",
               transform=ax_su.transAxes,
               fontsize=9, fontweight="bold", color=col_s,
               va="top", ha="right",
               bbox=dict(boxstyle="round,pad=0.35", fc="#F5EEF8",
                         ec=col_s, alpha=0.95, lw=1.2))

    _zone_shade(ax_su)
    ax_su.set_xlim(-0.5, 9.5)
    ax_su.set_ylim(0, max(S_u) * 1.60)
    ax_su.set_xticks(range(10))
    ax_su.set_xticklabels([str(b) for b in range(10)], fontsize=6.5)
    ax_su.set_xlabel("Radial bin  (core → periphery)", fontsize=8)
    ax_su.set_ylabel(r"$S(u)=JS[P_{T4},P_{T1}]$", fontsize=7.5, color=col_s)
    ax_su.tick_params(axis="y", labelsize=6.5, labelcolor=col_s)
    _zone_labels(ax_su, fontsize=6.5)

    # City schematic inset — lower-right of S(u) (outer zone, data is low there)
    ax_inset = ax_su.inset_axes([0.70, 0.18, 0.28, 0.72])
    _city_schematic(ax_inset)


# ── Supplement: 17×10 β heatmap (not called from main) ────────────────────

def _supplement_heatmap(ax_heat, heatmap_matrix, sig_matrix, type_order):
    """SI figure: type-specific size effects β_k(u) as 17×10 heatmap."""
    vmax = np.nanpercentile(np.abs(heatmap_matrix), 95)
    vmax = max(vmax, 0.01)

    im = ax_heat.imshow(heatmap_matrix,
                        aspect="auto", cmap=plt.cm.RdBu_r,
                        vmin=-vmax, vmax=vmax,
                        interpolation="nearest",
                        extent=[-0.5, N_BINS - 0.5,
                                len(type_order) - 0.5, -0.5])

    for ri in range(len(type_order)):
        for bi in range(N_BINS):
            if not sig_matrix[ri, bi]:
                ax_heat.add_patch(plt.Rectangle(
                    (bi - 0.5, ri - 0.5), 1, 1, fc="white", alpha=0.52, zorder=2))
                ax_heat.text(bi, ri, "×", ha="center", va="center",
                             fontsize=4, color="#777777", zorder=3)

    fam_boundaries = []
    prev_fam = TYPE_FAMILY[type_order[0]]
    for ri, t in enumerate(type_order):
        fam = TYPE_FAMILY[t]
        if fam != prev_fam:
            fam_boundaries.append(ri - 0.5)
            prev_fam = fam
    for yb in fam_boundaries:
        ax_heat.axhline(yb, color="white", lw=1.5, zorder=4)

    fam_start = 0
    prev_fam  = TYPE_FAMILY[type_order[0]]
    for ri, t in enumerate(type_order):
        fam = TYPE_FAMILY[t]
        if fam != prev_fam or ri == len(type_order) - 1:
            end = ri if fam != prev_fam else ri + 1
            mid = (fam_start + end - 1) / 2
            ax_heat.text(-1.4, mid, FAMILY_LABELS[prev_fam],
                         ha="right", va="center", fontsize=5,
                         color=FAMILY_COLORS[prev_fam], fontweight="bold",
                         transform=ax_heat.get_yaxis_transform(), clip_on=False)
            fam_start = ri
            prev_fam  = fam

    ax_heat.set_yticks(range(len(type_order)))
    ax_heat.set_yticklabels([TYPE_SHORT[t] for t in type_order], fontsize=5)
    ax_heat.tick_params(axis="y", length=0, pad=1)
    ax_heat.set_xticks(range(N_BINS))
    ax_heat.set_xticklabels([str(b) for b in range(N_BINS)], fontsize=6)
    ax_heat.set_xlabel("Radial bin  (core → periphery)", fontsize=8)
    _zone_shade(ax_heat)

    cbar = plt.colorbar(im, ax=ax_heat, orientation="vertical",
                        fraction=0.03, pad=0.01, shrink=0.85)
    cbar.set_label(r"$\beta_k(u)$ = $\partial p_k / \partial \log Pop$", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)


# ── Panel B: β + T1/T4 profiles ───────────────────────────────────────────

def _panel_B(ax_H, ax_r, fm, sdt, tier_profiles):
    """β_H(u) and β_r(u) scaling coefficients — clean, no ghost profiles."""
    bins = list(range(N_BINS))
    x    = np.array(bins)

    fH = fm[fm["outcome"] == "H17"].set_index("radial_bin")
    fr = fm[fm["outcome"] == "r17"].set_index("radial_bin")

    bH  = fH["beta"].reindex(bins).values
    seH = fH["beta_se"].reindex(bins).values
    br  = fr["beta"].reindex(bins).values
    ser = fr["beta_se"].reindex(bins).values

    # ── B1: β_H ──────────────────────────────────────────────────────────
    ax_H.fill_between(x, bH - 1.96*seH, bH + 1.96*seH,
                      color=COL_CORE, alpha=0.18)
    ax_H.plot(x, bH, "o-", color=COL_CORE, linewidth=2.0, markersize=4.5, zorder=3)
    _zone_shade(ax_H)
    ax_H.axhline(0, color="#AAAAAA", lw=0.7, ls=":")
    ax_H.set_xlim(-0.5, 9.5)
    ax_H.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_H.set_ylabel(r"$\beta_H(u)$", fontsize=9, color=COL_CORE)
    ax_H.tick_params(axis="y", labelcolor=COL_CORE, labelsize=7)
    _zone_labels(ax_H)

    R_H   = sdt.loc["R_H", "estimate"]
    RH_lo = sdt.loc["R_H", "ci_lo_95"]
    RH_hi = sdt.loc["R_H", "ci_hi_95"]
    # Callout box — upper right
    ax_H.text(0.97, 0.97,
              f"$R_H = {R_H:.2f}$\n[{RH_lo:.2f}, {RH_hi:.2f}]",
              transform=ax_H.transAxes, fontsize=7.5, color=COL_CORE,
              va="top", ha="right",
              bbox=dict(boxstyle="round,pad=0.30", fc="#EBF5FB",
                        ec=COL_CORE, alpha=0.95, lw=1.0))

    # ── B2: β_r ──────────────────────────────────────────────────────────
    col_r = "#C0392B"
    ax_r.fill_between(x, br - 1.96*ser, br + 1.96*ser, color=col_r, alpha=0.18)
    ax_r.plot(x, br, "s-", color=col_r, linewidth=2.0, markersize=4.5, zorder=3)
    _zone_shade(ax_r)
    ax_r.axhline(0, color="#AAAAAA", lw=0.7, ls=":")
    ax_r.set_xlim(-0.5, 9.5)
    ax_r.set_xticks(range(10))
    ax_r.set_xticklabels([str(b) for b in range(10)], fontsize=6.5)
    ax_r.set_xlabel("Radial bin  (core → periphery)", fontsize=8)
    ax_r.set_ylabel(r"$\beta_r(u)$", fontsize=9, color=col_r)
    ax_r.tick_params(axis="y", labelcolor=col_r, labelsize=7)

    R_r   = sdt.loc["R_r", "estimate"]
    Rr_lo = sdt.loc["R_r", "ci_lo_95"]
    Rr_hi = sdt.loc["R_r", "ci_hi_95"]
    ax_r.text(0.97, 0.97,
              f"$R_r = {R_r:.2f}$\n[{Rr_lo:.2f}, {Rr_hi:.2f}]",
              transform=ax_r.transAxes, fontsize=7.5, color=col_r,
              va="top", ha="right",
              bbox=dict(boxstyle="round,pad=0.30", fc="#FDEDEC",
                        ec=col_r, alpha=0.95, lw=1.0))


# ── Panel C: C(u) convergence — single axis, no overlays ─────────────────

def _panel_C(ax, fm, cd, tdd):
    """
    C(u) = |β_JS(u)| only.  Single y-axis, single colour, callout box.
    No inset, no dual axis, no T1/T4 JS overlay.
    """
    bins = list(range(N_BINS))
    x    = np.array(bins)

    fJ  = fm[fm["outcome"] == "JS"].set_index("radial_bin")
    bJ  = fJ["beta"].reindex(bins).values
    seJ = fJ["beta_se"].reindex(bins).values
    C   = np.abs(bJ)
    seC = seJ

    col_C = "#8E44AD"

    ax.fill_between(x, np.maximum(0, C - seC), C + seC,
                    color=col_C, alpha=0.20)          # ±1 SE band
    ax.plot(x, C, "D-", color=col_C, linewidth=2.2, markersize=5.5, zorder=3)

    _zone_shade(ax)
    _zone_labels(ax)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(b) for b in range(10)], fontsize=6)
    ax.set_xlabel("Radial bin  (core → periphery)", fontsize=7)
    ax.set_ylabel(r"$C(u) = |\beta_{JS}(u)|$", fontsize=7, color=col_C)
    ax.tick_params(axis="y", labelcolor=col_C, labelsize=6.5)

    row_mid   = cd[cd["zone_pair"].str.strip() == "C_core / C_mid"].iloc[0]
    row_outer = cd[cd["zone_pair"].str.strip() == "C_core / C_outer"].iloc[0]
    r_mid   = float(row_mid["ratio"])
    r_outer = float(row_outer["ratio"])
    ax.text(0.97, 0.97,
            f"Core / Mid   ×{r_mid:.2f}\n"
            f"Core / Outer ×{r_outer:.2f}",
            transform=ax.transAxes, fontsize=7.5, fontweight="bold",
            color=col_C, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.30", fc="#F5EEF8",
                      ec=col_C, alpha=0.95, lw=1.1))


# ── Panel D: R²_climate(u) — single axis, no overlays ────────────────────

def _panel_D(ax, bins, r2_climate, r2_colonial, n_obs):
    """
    Partial R²_climate(u) only.  Single y-axis, orange, callout box.
    Colonial mentioned in a plain-text footnote at bottom; no inset, no overlay.
    """
    x = np.array(bins)

    col_clim = "#E67E22"

    r2_climate  = np.clip(r2_climate,  0, None)
    r2_colonial = np.clip(r2_colonial, 0, None)

    ax.fill_between(x, 0, r2_climate, color=col_clim, alpha=0.15)
    ax.plot(x, r2_climate, "o-", color=col_clim, linewidth=2.2,
            markersize=5.5, zorder=3)

    _zone_shade(ax)
    _zone_labels(ax)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(b) for b in range(10)], fontsize=6)
    ax.set_xlabel("Radial bin  (core → periphery)", fontsize=7)
    ax.set_ylabel(r"Partial $R^2_{\rm climate}$ on JS divergence", fontsize=7,
                  color=col_clim)
    ax.tick_params(axis="y", labelsize=6, labelcolor=col_clim)

    c_core  = float(np.nanmean(r2_climate[[0, 1]]))
    c_outer = float(np.nanmean(r2_climate[[6, 7, 8, 9]]))
    ratio   = c_outer / c_core if c_core > 1e-6 else float("nan")
    ax.text(0.97, 0.97,
            f"Core  {c_core:.3f}\n"
            f"Outer {c_outer:.3f}   (×{ratio:.2f})",
            transform=ax.transAxes, fontsize=7.5, fontweight="bold",
            color=col_clim, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.30", fc="#FEF9EC",
                      ec=col_clim, alpha=0.95, lw=1.1))

    # One-line colonial footnote — bottom of axes
    c_core_col  = float(np.nanmean(r2_colonial[[0, 1]]))
    c_outer_col = float(np.nanmean(r2_colonial[[6, 7, 8, 9]]))
    ax.text(0.03, 0.03,
            f"Colonial $R^2$: {c_core_col:.3f} → {c_outer_col:.3f}  (see SI)",
            transform=ax.transAxes, fontsize=5.5, va="bottom", ha="left",
            color="#999999", style="italic")


# ── LOCO strip plot (SI) ───────────────────────────────────────────────────

def _panel_loco(ax, loco: pd.DataFrame, sdt: pd.DataFrame):
    """SI: Leave-one-country-out strip plot (47 iterations)."""
    stat_map = {
        "R_H":            ("R_H",           COL_CORE,  r"$R_H$"),
        "R_r":            ("R_r",           "#C0392B", r"$R_r$"),
        "C_core/C_outer": ("C_core_C_outer","#8E44AD", r"$C_{\rm core}/C_{\rm outer}$"),
    }
    main_vals = {
        "R_H":            sdt.loc["R_H", "estimate"],
        "R_r":            sdt.loc["R_r", "estimate"],
        "C_core/C_outer": 2.262858,
    }
    rng = np.random.default_rng(42)
    y_labels, y_pos, all_vals = [], [], []

    for row_idx, (stat_key, (col_key, color, label)) in enumerate(stat_map.items()):
        vals = loco[col_key].dropna().values
        y    = row_idx
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(vals, y + jitter, s=18, color=color, alpha=0.55,
                   linewidths=0, zorder=3)
        med = float(np.median(vals))
        ax.plot([med, med], [y - 0.28, y + 0.28],
                color=color, linewidth=2.5, solid_capstyle="round", zorder=4)
        main = main_vals[stat_key]
        ax.scatter([main], [y], s=60, marker="D", color=color,
                   edgecolors="white", linewidths=0.6, zorder=5)
        mn = float(vals.min())
        ax.text(mn - 0.01, y, f"{mn:.2f}",
                fontsize=6, color=color, va="center", ha="right", alpha=0.75)
        y_labels.append(label); y_pos.append(y)
        all_vals.extend(vals.tolist())

    ax.axvline(1.0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.text(0.99, 0.04, "47/47 iterations > 1",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            style="italic", color="#333333")
    x_lo = max(0.5, float(np.min(all_vals)) - 0.15)
    x_hi = float(np.max(all_vals)) + 0.2
    ax.set_xlim(x_lo, x_hi); ax.set_ylim(-0.6, 2.4)
    ax.set_yticks(y_pos); ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Ratio estimate (leave-one-country-out)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7); ax.tick_params(axis="y", length=0)
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
    ax.grid(axis="x", lw=0.5, alpha=0.35)


# ── Main assembly ──────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    fm, sdt, cd, cbp, cbdiv, cba, tdd, sep, ai = _load_all()

    print("Panel A data ...")
    ribbons, hm, sig, type_order, S_u = _panel_A_data(cbp, sep)

    print("T1/T4 metric profiles ...")
    tier_profiles = _tier_metric_profiles(cbdiv, cba)

    print("Panel C partial R² ...")
    d_bins, r2_clim, r2_col, n_obs = _panel_D_data(cbp, ai)

    # ─────────────────────────────────────────────────────────────────────
    # Figure geometry:
    #
    #  ┌─────────────────────────┬──────────────────────────┐  ← tall row
    #  │  A  T1–T4 ribbons+S(u)  │  B  β_H(u) / β_r(u)     │
    #  ├─────────────────────────┴──────────────────────────┤  ← thin row
    #  │  C1  C(u) conv.         │  C2  R²_climate/colonial │
    #  └─────────────────────────┴──────────────────────────┘
    #
    # height_ratios: top row ≈ 2.0, bottom (C) ≈ 0.75  → C is ~37% of total
    # A and B share equal width; C1 and C2 each take half of full width
    # ─────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("white")

    # Outer 2-row grid (rows: top=AB, bottom=C)
    outer = GridSpec(2, 1, figure=fig,
                     left=0.07, right=0.975,
                     bottom=0.07, top=0.95,
                     hspace=0.22,
                     height_ratios=[2.0, 0.75])

    # Top row: A and B side by side
    gs_top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                     wspace=0.26,
                                     width_ratios=[1, 1])

    # Bottom row: C1 and C2 side by side (very flat)
    gs_bot = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1],
                                     wspace=0.28)

    # ── Panel A: 4 ribbons (T1→T4) + S(u) ───────────────────────────────
    # S(u) gets 1.4× height of each ribbon row to be the visual focal point
    gs_A = GridSpecFromSubplotSpec(5, 1, subplot_spec=gs_top[0],
                                   hspace=0.06,
                                   height_ratios=[1, 1, 1, 1, 1.4])
    ax_A_T1 = fig.add_subplot(gs_A[0])
    ax_A_T2 = fig.add_subplot(gs_A[1])
    ax_A_T3 = fig.add_subplot(gs_A[2])
    ax_A_T4 = fig.add_subplot(gs_A[3])
    ax_A_su = fig.add_subplot(gs_A[4])

    # ── Panel B: β_H (top) + β_r (bottom) ───────────────────────────────
    gs_B = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_top[1],
                                   hspace=0.06, height_ratios=[1, 1])
    ax_B1 = fig.add_subplot(gs_B[0])
    ax_B2 = fig.add_subplot(gs_B[1])

    # ── Panel C: C(u) left | R² right  (flat bottom strip) ───────────────
    ax_C1 = fig.add_subplot(gs_bot[0])   # C(u) convergence
    ax_C2 = fig.add_subplot(gs_bot[1])   # Partial R²

    # ── Draw panels ────────────────────────────────────────────────────────
    print("Panel A ...")
    _panel_A(
        ribbon_axes=[ax_A_T1, ax_A_T2, ax_A_T3, ax_A_T4],
        ax_su=ax_A_su,
        ribbons=ribbons,
        S_u=S_u,
    )

    print("Panel B ...")
    _panel_B(ax_B1, ax_B2, fm, sdt, tier_profiles)

    print("Panel C1 (convergence) ...")
    _panel_C(ax_C1, fm, cd, tdd)

    print("Panel C2 (R²) ...")
    _panel_D(ax_C2, d_bins, r2_clim, r2_col, n_obs)

    # ── Panel labels ───────────────────────────────────────────────────────
    lkw = dict(fontsize=13, fontweight="bold", va="top")
    # Place labels just inside top-left corner of each panel's first axis
    ax_A_T1.text(0.01, 1.02, "A", transform=ax_A_T1.transAxes, **lkw)
    ax_B1.text(  0.01, 1.02, "B", transform=ax_B1.transAxes,   **lkw)
    ax_C1.text(  0.01, 1.08, "C", transform=ax_C1.transAxes,   **lkw)
    ax_C2.text(  0.01, 1.08, "D", transform=ax_C2.transAxes,   **lkw)

    # Shared y-label for ribbon column
    fig.text(0.010, 0.73, "Type share", va="center",
             rotation="vertical", fontsize=7.5, color="#444444")

    # ── Save combined figure ───────────────────────────────────────────────
    for fmt, dpi in [("pdf", None), ("png", 300)]:
        out_path = OUT_FIG / f"figure_main.{fmt}"
        kw = {"dpi": dpi} if dpi else {}
        fig.savefig(out_path, bbox_inches="tight", **kw)
        print(f"Saved: {out_path}")

    plt.close(fig)

    # ── Save individual PPT panels ─────────────────────────────────────────
    print("Saving individual panels ...")
    _save_individual_panels(
        fm=fm, sdt=sdt, cd=cd,
        ribbons=ribbons, S_u=S_u,
        tier_profiles=tier_profiles,
        d_bins=d_bins, r2_clim=r2_clim, r2_col=r2_col, n_obs=n_obs,
        tdd=tdd,
    )
    print("Done.")


# ── Individual PPT-sized panels ────────────────────────────────────────────

def _save_individual_panels(fm, sdt, cd, ribbons, S_u,
                             tier_profiles, d_bins, r2_clim, r2_col, n_obs, tdd):
    """
    Save panels A–D as separate PPT-friendly 16:9 landscape figures.

    All panels use figsize ≈ (13.33, 7.5) in (standard 16:9 at 96 dpi).

    A : Left = 4 stacked ribbons (T1–T4); Right = S(u) curve  [no schematic]
    B : Left = β_H(u);  Right = β_r(u)   (side by side)
    C : Single wide axes — T1/T4 JS left-axis + C(u) right-axis
    D : Single wide axes — partial R² curves
    """
    W, H = 13.33, 7.50   # 16:9 inches
    SAVE_FMTS = [("pdf", None), ("png", 300)]

    def _save(fig, stem):
        for fmt, dpi in SAVE_FMTS:
            p = OUT_FIG / f"{stem}.{fmt}"
            kw = {"dpi": dpi} if dpi else {}
            fig.savefig(p, bbox_inches="tight", **kw)
            print(f"  Saved: {p}")
        plt.close(fig)

    # ── Panel A: ribbons LEFT | S(u) RIGHT ───────────────────────────────
    fig_A = plt.figure(figsize=(W, H))
    fig_A.patch.set_facecolor("white")

    # Left block: 4 ribbon rows stacked
    gs_A = GridSpec(1, 2, figure=fig_A,
                    left=0.07, right=0.97,
                    bottom=0.07, top=0.95,
                    wspace=0.22,
                    width_ratios=[1.1, 0.9])

    # Left: 4 equal ribbon rows
    gs_ribbons = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_A[0],
                                         hspace=0.05)
    ax_r = [fig_A.add_subplot(gs_ribbons[i]) for i in range(4)]

    tiers_in_order = ["T1", "T2", "T3", "T4"]
    labels_A = ["T1  (<20 k)", "T2  (20–200 k)", "T3  (200 k–1 M)", "T4  (≥1 M)"]
    for i, (ax, tier, lbl) in enumerate(zip(ax_r, tiers_in_order, labels_A)):
        _draw_ribbon(ax, ribbons[tier], lbl,
                     badge_dark=(tier == "T4"),
                     show_yticks=True,
                     show_xticks=(i == 3),
                     show_zone_labels=(i == 0))
    # x-label on bottom ribbon
    ax_r[3].set_xlabel("Radial bin  (core → periphery)", fontsize=10)

    # Type legend below bottom ribbon
    patches = [mpatches.Patch(color=TYPE_COLORS[k], label=str(k))
               for k in range(1, N_TYPES + 1)]
    ax_r[3].legend(handles=patches, ncol=17, fontsize=5,
                   loc="upper left", bbox_to_anchor=(0.0, -0.22),
                   title="Morphotype", title_fontsize=5.5,
                   framealpha=0.0, handlelength=0.8, handletextpad=0.3,
                   columnspacing=0.4, borderpad=0.1)

    # Shared y-label
    fig_A.text(0.01, 0.55, "Type share", va="center",
               rotation="vertical", fontsize=10, color="#444444")

    # Right: S(u) — no inset schematic
    ax_su = fig_A.add_subplot(gs_A[1])
    x = np.arange(N_BINS)
    col_s = "#6C3483"
    ax_su.fill_between(x, 0, S_u, color=col_s, alpha=0.18)
    ax_su.plot(x, S_u, "o-", color=col_s, linewidth=2.2, markersize=6, zorder=3)

    S_core  = float(np.nanmean(S_u[np.array(ZONE_CORE)]))
    S_outer = float(np.nanmean(S_u[np.array(ZONE_OUTER)]))
    S_ratio = S_core / S_outer
    ax_su.plot([ZONE_CORE[0],  ZONE_CORE[-1]  + 1], [S_core,  S_core],
               "--", color=COL_CORE,  lw=1.2, alpha=0.7)
    ax_su.plot([ZONE_OUTER[0], ZONE_OUTER[-1] + 1], [S_outer, S_outer],
               "--", color=COL_OUTER, lw=1.2, alpha=0.7)
    ax_su.text(0.97, 0.95,
               f"Core / Outer\n$\\times${S_ratio:.2f}",
               transform=ax_su.transAxes,
               fontsize=13, fontweight="bold", color=col_s,
               va="top", ha="right",
               bbox=dict(boxstyle="round,pad=0.40", fc="#F5EEF8",
                         ec=col_s, alpha=0.95, lw=1.5))
    _zone_shade(ax_su)
    _zone_labels(ax_su, fontsize=9)
    ax_su.set_xlim(-0.5, 9.5)
    ax_su.set_ylim(0, max(S_u) * 1.65)
    ax_su.set_xticks(range(10))
    ax_su.set_xticklabels([str(b) for b in range(10)], fontsize=9)
    ax_su.set_xlabel("Radial bin  (core → periphery)", fontsize=10)
    ax_su.set_ylabel(r"$S(u) = JS[P_{T4},\,P_{T1}]$", fontsize=10, color=col_s)
    ax_su.tick_params(axis="y", labelsize=9, labelcolor=col_s)

    _save(fig_A, "panel_A")

    # ── Panel B: β_H LEFT | β_r RIGHT ────────────────────────────────────
    fig_B = plt.figure(figsize=(W, H))
    fig_B.patch.set_facecolor("white")
    gs_B = GridSpec(1, 2, figure=fig_B,
                    left=0.08, right=0.97,
                    bottom=0.10, top=0.94,
                    wspace=0.28)
    ax_B1 = fig_B.add_subplot(gs_B[0])
    ax_B2 = fig_B.add_subplot(gs_B[1])

    bins = list(range(N_BINS))
    x    = np.array(bins)
    fH   = fm[fm["outcome"] == "H17"].set_index("radial_bin")
    fr   = fm[fm["outcome"] == "r17"].set_index("radial_bin")
    bH   = fH["beta"].reindex(bins).values
    seH  = fH["beta_se"].reindex(bins).values
    br   = fr["beta"].reindex(bins).values
    ser  = fr["beta_se"].reindex(bins).values

    for ax, b, se, col, marker, ylabel, stat_key in [
        (ax_B1, bH, seH, COL_CORE, "o", r"$\beta_H(u)$", "R_H"),
        (ax_B2, br, ser, "#C0392B", "s", r"$\beta_r(u)$", "R_r"),
    ]:
        ax.fill_between(x, b - 1.96*se, b + 1.96*se, color=col, alpha=0.18)
        ax.plot(x, b, f"{marker}-", color=col, linewidth=2.2, markersize=6, zorder=3)
        _zone_shade(ax)
        ax.axhline(0, color="#AAAAAA", lw=0.8, ls=":")
        ax.set_xlim(-0.5, 9.5)
        ax.set_xticks(range(10))
        ax.set_xticklabels([str(i) for i in range(10)], fontsize=9)
        ax.set_xlabel("Radial bin  (core → periphery)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=12, color=col)
        ax.tick_params(axis="y", labelcolor=col, labelsize=9)
        _zone_labels(ax, fontsize=9)

        est   = sdt.loc[stat_key, "estimate"]
        ci_lo = sdt.loc[stat_key, "ci_lo_95"]
        ci_hi = sdt.loc[stat_key, "ci_hi_95"]
        fc    = "#EBF5FB" if col == COL_CORE else "#FDEDEC"
        ax.text(0.97, 0.97,
                f"$R = {est:.2f}$\n[{ci_lo:.2f}, {ci_hi:.2f}]",
                transform=ax.transAxes, fontsize=10, color=col,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.35", fc=fc,
                          ec=col, alpha=0.95, lw=1.2))

    _save(fig_B, "panel_B")

    # ── Panel C: T1/T4 JS + C(u) — single wide axes ──────────────────────
    fig_C = plt.figure(figsize=(W, H))
    fig_C.patch.set_facecolor("white")
    ax_C = fig_C.add_axes([0.08, 0.11, 0.84, 0.82])
    _panel_C(ax_C, fm, cd, tdd)
    # Upscale for standalone slide — single axis, no twin to handle
    ax_C.set_xlabel("Radial bin  (core → periphery)", fontsize=12)
    ax_C.set_ylabel(r"$C(u) = |\beta_{JS}(u)|$", fontsize=12, color="#8E44AD")
    ax_C.tick_params(axis="both", labelsize=10)
    for txt in ax_C.texts:
        txt.set_fontsize(max(txt.get_fontsize(), 10))
    _save(fig_C, "panel_C")

    # ── Panel D: R²_climate — single wide axes ────────────────────────────
    fig_D = plt.figure(figsize=(W, H))
    fig_D.patch.set_facecolor("white")
    ax_D = fig_D.add_axes([0.08, 0.11, 0.84, 0.82])
    _panel_D(ax_D, d_bins, r2_clim, r2_col, n_obs)
    ax_D.set_xlabel("Radial bin  (core → periphery)", fontsize=12)
    ax_D.set_ylabel(r"Partial $R^2_{\rm climate}$ on JS", fontsize=12,
                    color="#E67E22")
    ax_D.tick_params(axis="both", labelsize=10)
    for txt in ax_D.texts:
        txt.set_fontsize(max(txt.get_fontsize(), 10))
    _save(fig_D, "panel_D")


if __name__ == "__main__":
    main()
