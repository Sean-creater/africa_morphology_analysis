"""
03_figures.py
=============
Publication figure for the regression analysis: factor contributions (Q1).

Panel A — Factor contributions (Q1)
    Horizontal bar chart showing the relative contribution of each factor
    group (climate, GDP+population, colonial legacy) measured by:
      • Δ McFadden pseudo-R²  (drop-one from M3)
      • Δ CV log-loss          (drop-one from M3, sign-flipped so larger = better)
    Error bar for CV log-loss omitted (single estimate); statistical
    significance from LRT shown as star annotation.

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
def draw_panel_A(ax, factor_df, fit_df):
    """
    Grouped horizontal bar chart: for each factor group, two bars —
      - Unique (drop-one): pseudo-R² lost when that factor alone is
        removed from the full M3 model, all others held in. Does NOT sum
        to the total M3 R² when factor groups share explanatory power
        (e.g. colonial legacy correlates with climate/geography).
      - Sequential (incremental): pseudo-R² gained when that factor is
        added, in the fixed nested order climate -> GDP -> population ->
        colonial (M0->M1->M2->M2b->M3). Sums EXACTLY to the total M3 R²
        by construction (telescoping sum), but each individual increment
        is order-dependent (whichever factor enters first "claims" any
        variance shared with factors not yet in the model).
    Together the two bars show both kinds of explanatory power: what a
    factor uniquely needs (vs. what's redundant with the others) and what
    it contributes on a path that must add up to the whole.
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
    seq_order = ["climate", "gdp", "pop", "colonial"]
    seq_steps = {"climate":  ("M0", "M1"),
                "gdp":      ("M1", "M2"),
                "pop":      ("M2", "M2b"),
                "colonial": ("M2b", "M3")}

    df = factor_df.copy()
    df["label"] = df["factor_group"].map(labels).fillna(df["factor_group"])
    df = df.set_index("factor_group").loc[seq_order].reset_index()

    # Sequential (incremental) pseudo-R², computed from the nested fit
    # table so it is never hand-entered — always consistent with whatever
    # 01_mnl_main.py actually produced.
    r2_by_model = fit_df.set_index("model")["pseudo_r2"] if fit_df is not None else None
    seq_vals = []
    for fg in df["factor_group"]:
        lo, hi = seq_steps[fg]
        if r2_by_model is not None and lo in r2_by_model.index and hi in r2_by_model.index:
            seq_vals.append(r2_by_model[hi] - r2_by_model[lo])
        else:
            seq_vals.append(np.nan)
    df["seq_pseudo_r2"] = seq_vals

    # Sort rows by sequential contribution (the metric that sums to the
    # total) so the chart's top-to-bottom order reflects the model-
    # building sequence's own ranking.
    df = df.sort_values("seq_pseudo_r2", ascending=False)

    y        = np.arange(len(df))
    GREY     = "#555555"
    C_UNIQUE = "#BFBFBF"   # lighter grey — unique (drop-one)
    C_SEQ    = "#7A7A7A"   # darker grey  — sequential (incremental)

    uniq_vals = df["delta_pseudo_r2"].fillna(0).values
    seq_vals  = df["seq_pseudo_r2"].fillna(0).values
    total_r2  = (r2_by_model["M3"] if r2_by_model is not None and "M3" in r2_by_model.index
                else np.nansum(seq_vals))

    xmax = max(uniq_vals.max(), seq_vals.max(), total_r2) * 1.22
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    BAR_H = 0.32
    ax.barh(y + BAR_H / 2 + 0.02, seq_vals, height=BAR_H, color=C_SEQ, zorder=2,
           label="Sequential (sums to total R²)")
    ax.barh(y - BAR_H / 2 - 0.02, uniq_vals, height=BAR_H, color=C_UNIQUE, zorder=2,
           label="Unique (drop-one, overlap excluded)")

    for i in range(len(df)):
        pv = df.iloc[i].get("lrt_p", np.nan)
        star = "***" if (not pd.isna(pv) and pv < 0.001) else (
               "**"  if (not pd.isna(pv) and pv < 0.01)  else (
               "*"   if (not pd.isna(pv) and pv < 0.05)  else ""))
        ax.text(seq_vals[i] + xmax * 0.015, y[i] + BAR_H / 2 + 0.02,
                f"{seq_vals[i]:.2f}", va="center", ha="left",
                fontsize=6, color=GREY)
        ax.text(uniq_vals[i] + xmax * 0.015, y[i] - BAR_H / 2 - 0.02,
                f"{uniq_vals[i]:.2f}{star}", va="center", ha="left",
                fontsize=6, color=GREY)

    # Reference line at the full M3 model's total pseudo-R² — the
    # sequential bars' sum (not any single bar) lands exactly on this.
    # Label placed inside the axes, to the line's right, vertically
    # centred — avoids the title/legend areas above and below the axes.
    if not pd.isna(total_r2):
        ax.axvline(total_r2, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
        ax.text(total_r2 + xmax * 0.015, (len(df) - 1) / 2,
                f"total R²\n= {total_r2:.2f}",
                fontsize=6, color="#777777", va="center", ha="left")

    # y-axis: factor labels
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].tolist(), fontsize=7.5, color="#333333")

    ax.set_xlabel("Pseudo-R²", fontsize=7.5, color="#222222", labelpad=5)
    ax.tick_params(axis="x", labelsize=6, colors="#555555")

    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(left=False, bottom=False)
    ax.set_ylim(-0.6, len(df) - 0.1)
    ax.set_axisbelow(True)

    ax.legend(fontsize=6, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, -0.14), ncol=1, handlelength=1.2,
              handletextpad=0.5)

    ax.set_title("a", loc="left", fontsize=9, fontweight="bold", pad=4,
                 color="#222222")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mpl.rcParams.update({
        "font.family":     "Arial",
        "font.size":       8,
        "axes.linewidth":  0.5,
        "svg.fonttype": "none",
    })

    factor_df  = load("mnl_factor_contribution.csv")
    fit_df     = load("mnl_fit.csv")

    fig_w_mm, fig_h_mm = 115, 80
    fig, ax = plt.subplots(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4))
    fig.subplots_adjust(left=0.26, right=0.95, top=0.90, bottom=0.26)

    draw_panel_A(ax, factor_df, fit_df)

    for fmt in ("svg", "png"):
        p = FIG_DIR / f"regression_main.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        log.info("Saved → %s", p)
    plt.close(fig)
    log.info("Done.")


if __name__ == "__main__":
    main()
