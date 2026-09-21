"""
03_figures.py
=============
Three-panel publication figure for the regression analysis.

Panel A — Factor contributions (Q1)
    Horizontal bar chart showing the relative contribution of each factor
    group (climate, GDP+population, colonial legacy) measured by:
      • Δ McFadden pseudo-R²  (drop-one from M3)
      • Δ CV log-loss          (drop-one from M3, sign-flipped so larger = better)
    Error bar for CV log-loss omitted (single estimate); statistical
    significance from LRT shown as star annotation.

Panel B — Community-specific associations (Q2)
    Dual heatmap (community × predictor):
      Left  : AME of log_gdp_std and log_pop_std (+1 SD, probability scale)
      Right : AAP deviation from reference for climate-zone groups and
              colonial categories (average adjusted prediction minus the
              grand mean predicted probability for that community)
    Cells coloured by value; ±95% CI significance marks for AME.

Panel C — Country-level variance decomposition
    Line plot: ICC across B0→B1→B2→B3 for each community.
    Shows how much country-level heterogeneity remains after adding covariates.

Output
------
outputs/figures/regression_main.pdf
outputs/figures/regression_main.png

Usage
-----
    python 03_figures.py
"""

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent
TAB_DIR  = ROOT / "outputs" / "tables"
FIG_DIR  = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Community palette — same as community_detection scripts (0-based idx = comm-1)
PALETTE_9 = [
    "#f36424",  # comm 1 orange
    "#672e92",  # comm 2 purple
    "#8bcf02",  # comm 3 lime green
    "#139ed3",  # comm 4 sky blue
    "#ff0000",  # comm 5 red
    "#ffd700",  # comm 6 gold
    "#293890",  # comm 7 dark blue
    "#74201e",  # comm 8 dark red
    "#bebebe",  # comm 9 grey
]


def comm_color(c):
    """Color for community c (1-based)."""
    return PALETTE_9[(int(c) - 1) % len(PALETTE_9)]


# ---------------------------------------------------------------------------
# Load tables (with graceful fallback when a file is missing)
# ---------------------------------------------------------------------------
def load(fname):
    p = TAB_DIR / fname
    if not p.exists():
        log.warning("Missing: %s", p)
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Panel A: factor contributions — dot-and-line, publication style
# ---------------------------------------------------------------------------
def draw_panel_A(ax, factor_df):
    """
    Lollipop chart: one row per factor group, two metrics shown as
    separate coloured dots on a shared horizontal scale.
    Cleaner than grouped bars for 4 factors × 2 metrics.
    """
    if factor_df is None or factor_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_title("A", loc="left", fontsize=8, fontweight="bold")
        return

    labels = {"climate":  "Climate zone",
              "gdp":      "GDP per capita",
              "pop":      "City population",
              "colonial": "Colonial legacy"}

    df = factor_df.copy()
    df["label"] = df["factor_group"].map(labels).fillna(df["factor_group"])
    df = df.sort_values("delta_pseudo_r2", ascending=False)   # top-to-bottom

    y       = np.arange(len(df))
    GREY    = "#444444"
    C_R2    = "#1a6faf"   # steel blue  → pseudo-R²
    C_CV    = "#c0392b"   # deep red    → CV log-loss

    r2_vals = df["delta_pseudo_r2"].fillna(0).values
    cv_vals = df["delta_cv_logloss"].fillna(0).values

    # Normalise CV log-loss to same approximate scale as pseudo-R²
    # (both measure "how much worse without this factor")
    # pseudo-R²: 0–1 range; CV log-loss: ~0–0.35 range → scale to match
    scale = r2_vals.max() / cv_vals.max() if cv_vals.max() > 0 else 1.0
    cv_scaled = cv_vals * scale   # for display only; axis labelled separately

    offset = 0.18   # vertical jitter between two metrics per row

    for i, (r2, cv) in enumerate(zip(r2_vals, cv_scaled)):
        # stem lines — thicker
        ax.plot([0, r2], [y[i] + offset, y[i] + offset],
                color=C_R2, lw=1.8, alpha=0.75, solid_capstyle="round")
        ax.plot([0, cv], [y[i] - offset, y[i] - offset],
                color=C_CV, lw=1.8, alpha=0.75, solid_capstyle="round")
        # dots — no edge
        ax.scatter(r2, y[i] + offset, color=C_R2, s=72, zorder=4,
                   linewidths=0)
        ax.scatter(cv, y[i] - offset, color=C_CV, s=72, zorder=4,
                   linewidths=0)

    # significance annotation (all p < 0.001 in practice; show "***" once)
    for i, (_, row) in enumerate(df.iterrows()):
        pv = row.get("lrt_p", np.nan)
        star = "***" if (not pd.isna(pv) and pv < 0.001) else (
               "**"  if (not pd.isna(pv) and pv < 0.01)  else (
               "*"   if (not pd.isna(pv) and pv < 0.05)  else ""))
        if star:
            xmax = max(r2_vals[i], cv_scaled[i]) + 0.003
            ax.text(xmax, y[i], star, va="center", ha="left",
                    fontsize=6, color=GREY)

    # y-axis: factor labels
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=7, color=GREY)

    # dual x-axis: bottom = pseudo-R², top = CV log-loss (original scale)
    ax.set_xlabel("Δ McFadden pseudo-R²  (drop-one from M3)",
                  fontsize=6.5, color="#222222")
    ax.tick_params(axis="x", labelsize=6, colors="#222222")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xlim(left=0)

    ax2 = ax.twiny()
    ax2.set_xlim(0, ax.get_xlim()[1] / scale)
    ax2.set_xlabel("Δ CV log-loss  (drop-one from M3)",
                   fontsize=6.5, color="#222222")
    ax2.tick_params(axis="x", labelsize=6, colors="#222222")
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax2.spines[["bottom", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    ax.set_ylim(-0.6, len(df) - 0.4)

    # legend — placed outside plot area to avoid overlap with lollipops
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_R2,
               markersize=6, label="Δ pseudo-R²"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_CV,
               markersize=6, label="Δ CV log-loss"),
    ]
    ax.legend(handles=legend_els, fontsize=6, frameon=False,
              loc="upper right", bbox_to_anchor=(1.0, -0.18),
              ncol=2, handletextpad=0.3, columnspacing=1.0)
    ax.set_title("a", loc="left", fontsize=9, fontweight="bold", pad=4,
                 color="#222222")


# ---------------------------------------------------------------------------
# Panel B: community × predictor heatmap (AME + AAP)
# ---------------------------------------------------------------------------
def draw_panel_B(ax, ame_df, aap_clim_df, aap_col_df):
    """
    Heatmap: rows = communities, cols = predictors.
    Left block  : AME of log_gdp_std, log_pop_std (probability Δ per +1 SD)
    Middle block: AAP deviation for each climate zone present in the data
    Right block : AAP deviation for each colonial legacy category
    """
    if ame_df is None:
        ax.text(0.5, 0.5, "No AME data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_title("B  Community-specific associations", loc="left",
                     fontsize=8, fontweight="bold")
        return

    comms = sorted(ame_df["community"].unique())
    n_comms = len(comms)

    # ---- AME block ----
    ame_vars = ["log_gdp_std", "log_pop_std"]
    ame_labels = {"log_gdp_std": "GDP\n(+1 SD)", "log_pop_std": "Pop.\n(+1 SD)"}
    ame_mat = np.full((n_comms, len(ame_vars)), np.nan)
    ame_sig = np.zeros((n_comms, len(ame_vars)), dtype=bool)

    for i, comm in enumerate(comms):
        for j, var in enumerate(ame_vars):
            row = ame_df[(ame_df["community"] == comm) &
                         (ame_df["focal_var"] == var)]
            if not row.empty:
                r = row.iloc[0]
                ame_mat[i, j] = r["ame"]
                ame_sig[i, j] = (r["ci_lo_95"] * r["ci_hi_95"]) > 0

    # ---- Climate AAP block: all zones present in data ----
    # Köppen order: Tropical → Arid → Temperate (then reference Aw first so it shows)
    clim_order_pref = ["Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk",
                       "Cfa", "Cfb", "Csa", "Csb", "Cwa", "Cwb"]
    if aap_clim_df is not None and not aap_clim_df.empty:
        clim_present = [z for z in clim_order_pref
                        if z in aap_clim_df["group_value"].values]
    else:
        clim_present = []

    # Community baseline: n_cities-weighted mean AAP across all climate zones.
    # Equals the community's overall predicted prevalence in the sample.
    # ΔP = AAP(climate=k) − baseline  →  same unit/interpretation as AME.
    clim_baseline = {}   # comm → baseline probability
    if aap_clim_df is not None and not aap_clim_df.empty:
        for comm in comms:
            sub = aap_clim_df[aap_clim_df["community"] == comm]
            tot = sub["n_cities"].sum()
            clim_baseline[comm] = (
                (sub["aap"] * sub["n_cities"]).sum() / tot if tot > 0 else np.nan
            )

    clim_mat = np.full((n_comms, len(clim_present)), np.nan)
    if aap_clim_df is not None and not aap_clim_df.empty:
        for j, zone in enumerate(clim_present):
            sub = aap_clim_df[aap_clim_df["group_value"] == zone]
            for i, comm in enumerate(comms):
                cs = sub[sub["community"] == comm]
                if not cs.empty:
                    clim_mat[i, j] = cs.iloc[0]["aap"] - clim_baseline.get(comm, np.nan)

    # ---- Colonial AAP block ----
    col_order = ["British", "French", "Portuguese", "Belgian",
                 "Independent", "Other European"]
    if aap_col_df is not None and not aap_col_df.empty:
        col_present = [c for c in col_order
                       if c in aap_col_df["group_value"].values]
    else:
        col_present = []

    # Same baseline logic for colonial (use n_cities weights)
    col_baseline = {}
    if aap_col_df is not None and not aap_col_df.empty:
        for comm in comms:
            sub = aap_col_df[aap_col_df["community"] == comm]
            tot = sub["n_cities"].sum()
            col_baseline[comm] = (
                (sub["aap"] * sub["n_cities"]).sum() / tot if tot > 0 else np.nan
            )

    col_mat = np.full((n_comms, len(col_present)), np.nan)
    if aap_col_df is not None and not aap_col_df.empty:
        for j, cat in enumerate(col_present):
            sub = aap_col_df[aap_col_df["group_value"] == cat]
            for i, comm in enumerate(comms):
                cs = sub[sub["community"] == comm]
                if not cs.empty:
                    col_mat[i, j] = cs.iloc[0]["aap"] - col_baseline.get(comm, np.nan)

    # ---- Combine ----
    blocks = [ame_mat]
    if clim_mat.shape[1] > 0:
        blocks.append(clim_mat)
    if col_mat.shape[1] > 0:
        blocks.append(col_mat)
    full_mat   = np.hstack(blocks)
    col_labels = ([ame_labels.get(v, v) for v in ame_vars]
                  + clim_present + col_present)
    n_ame      = len(ame_vars)
    n_clim     = len(clim_present)
    n_total    = full_mat.shape[1]

    # Symmetric colour scale
    vmax = np.nanpercentile(np.abs(full_mat), 95)
    vmax = max(vmax, 0.01)

    im = ax.imshow(full_mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    # Significance marks: ● = 95% bootstrap CI excludes zero
    for i in range(n_comms):
        for j in range(n_ame):
            if ame_sig[i, j]:
                ax.text(j, i, "●", ha="center", va="center",
                        fontsize=4.5, color="white")

    # Separator lines between blocks
    for sep in [n_ame - 0.5, n_ame + n_clim - 0.5]:
        ax.axvline(sep, color="white", lw=1.5)
        ax.axvline(sep, color="#333333", lw=0.8, ls="--")

    ax.set_xticks(range(n_total))
    ax.set_xticklabels(col_labels, fontsize=6, rotation=40, ha="right")
    ax.set_yticks(range(n_comms))
    ax.set_yticklabels(
        [f"Comm {c}" for c in comms],
        fontsize=6.5, color="#222222"
    )

    # Group labels below x-tick labels (use axes-fraction y so they
    # sit beneath the rotated tick labels regardless of figure size)
    label_y = -0.38   # axes fraction — below tick labels
    ax.text((n_ame - 0.5) / n_total / 2 * n_total / n_total,
            label_y, "Socioeconomic",
            ha="center", va="top", fontsize=6, color="#555555",
            transform=ax.transAxes, style="italic")
    if n_clim > 0:
        cx = (n_ame + n_clim / 2 - 0.5) / n_total
        ax.text(cx, label_y, "Climate zone",
                ha="center", va="top", fontsize=6, color="#555555",
                transform=ax.transAxes, style="italic")
    if len(col_present) > 0:
        cx = (n_ame + n_clim + len(col_present) / 2 - 0.5) / n_total
        ax.text(cx, label_y, "Colonial legacy",
                ha="center", va="top", fontsize=6, color="#555555",
                transform=ax.transAxes, style="italic")

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
    cb.set_label("ΔP (probability vs community baseline)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.set_title("b", loc="left", fontsize=9, fontweight="bold", pad=14,
                 color="#222222")


# ---------------------------------------------------------------------------
# Panel C: ICC variance decomposition
# ---------------------------------------------------------------------------
def draw_panel_C(ax, vd_df):
    """Line chart: ICC across B0→B3 per community."""
    if vd_df is None or vd_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_title("C  Country-level variance decomposition",
                     loc="left", fontsize=8, fontweight="bold")
        return

    model_order = ["B0", "B1", "B2", "B2b", "B3"]
    model_labels = {
        "B0":  "Null\n(country RE)",
        "B1":  "+Climate",
        "B2":  "+GDP",
        "B2b": "+Population",
        "B3":  "+Colonial",
    }
    comms = sorted(vd_df["community"].unique())
    x = np.arange(len(model_order))

    for comm in comms:
        sub = vd_df[vd_df["community"] == comm].copy()
        sub = sub[sub["model"].isin(model_order)]
        sub["model_ord"] = sub["model"].map({m: i for i, m in enumerate(model_order)})
        sub = sub.sort_values("model_ord")
        icc_vals = sub["icc"].values
        ax.plot(range(len(sub)), icc_vals,
                color=comm_color(comm), lw=1.2, marker="o", markersize=3,
                label=f"Comm {comm}", alpha=0.85)

    ax.set_xticks(range(len(model_order)))
    ax.set_xticklabels([model_labels.get(m, m) for m in model_order],
                       fontsize=6.5)
    ax.set_ylabel("ICC (country-level variance share)", fontsize=7)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.tick_params(axis="both", labelsize=6.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=5.5, frameon=False, ncol=3,
              loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("C  Country-level residual heterogeneity (ICC)",
                 loc="left", fontsize=8, fontweight="bold", pad=4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":       7,
        "axes.linewidth":  0.5,
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
    })

    factor_df  = load("mnl_factor_contribution.csv")
    ame_df     = load("mnl_ame_M3.csv")
    aap_clim   = load("mnl_aap_climate_M3.csv")
    aap_col    = load("mnl_aap_colonial_M3.csv")

    # Figure: 180 mm wide × 150 mm tall (two-panel, Nature sub-journal)
    fig_w_mm, fig_h_mm = 180, 150
    fig, axes = plt.subplots(
        2, 1,
        figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4),
        gridspec_kw={"height_ratios": [1.1, 2.5]},
    )
    fig.subplots_adjust(left=0.14, right=0.92, top=0.96,
                        bottom=0.06, hspace=0.52)

    draw_panel_A(axes[0], factor_df)
    draw_panel_B(axes[1], ame_df, aap_clim, aap_col)

    for fmt in ("pdf", "png"):
        p = FIG_DIR / f"regression_main.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log.info("Saved → %s", p)
    plt.close(fig)
    log.info("Done.")


if __name__ == "__main__":
    main()
