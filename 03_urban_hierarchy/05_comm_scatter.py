"""
05_comm_scatter.py
==================
Six figures (p1–p4, H17_star, r17): grey scatter + per-community LOESS curves
+ 95% country-cluster bootstrap CI bands + two reference vlines (200k, 1M pop).

Communities are ranked by city count (largest → first) and labelled by
geographic name. Colours match the community_detection PALETTE_9 (by original
community ID 0–7).

Output
------
    outputs/figures/comm_scatter_p1.pdf/png  … comm_scatter_r17.pdf/png

Usage
-----
    python 05_comm_scatter.py [--boots N] [--workers N]
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

matplotlib.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

ROOT      = Path(__file__).resolve().parent
METRICS   = ROOT / "outputs" / "tables" / "city_metrics.parquet"
FIG_DIR   = ROOT / "outputs" / "figures"
CACHE_DIR = ROOT / "outputs" / "cache"
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4

# Community colour palette — indexed by original community ID 0–7
# Must match PALETTE_9 in community_detection scripts
PALETTE_9 = [
    "#f36424",   # 0  orange
    "#672e92",   # 1  purple
    "#8bcf02",   # 2  lime green
    "#139ed3",   # 3  sky blue
    "#ff0000",   # 4  red
    "#ffd700",   # 5  gold
    "#293890",   # 6  dark blue
    "#74201e",   # 7  dark red
]

# Geographic community names assigned in descending city-count order (C1=largest)
COMM_NAMES_RANKED = [
    "West African Inland",
    "Gulf of Guinea Coastal",
    "Southern–East African",
    "East–Central African",
    "Sahel–Central African Interior",
    "North African–Nile",
    "Nile Urban Corridor",
    "East African Highlands",
]

GROUP_NAMES = {
    "p1": "G1 — Medium-density built-up",
    "p2": "G2 — High-density compact",
    "p3": "G3 — Sparse vegetated",
    "p4": "G4 — Low-density heterogeneous",
}

SCATTER_SIZE   = 4
SCATTER_ALPHA  = 0.20
LOESS_FRAC     = 0.50   # wider bandwidth for per-community curves (smaller n)
N_BOOTS        = 500
N_BOOT_WORKERS = 14

# Reference vlines: log of 200 000 and 1 000 000
VLINE_X = [np.log(200_000), np.log(1_000_000)]


# ---------------------------------------------------------------------------
# LOESS + country-cluster bootstrap CI (cached)
# ---------------------------------------------------------------------------
def fit_loess(x, y, x_grid, frac=LOESS_FRAC):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.full(len(x_grid), np.nan)
    sm = lowess(y[mask], x[mask], frac=frac, return_sorted=True)
    return np.interp(x_grid, sm[:, 0], sm[:, 1])


def _boot_worker(args):
    seed, x_vals, y_vals, countries, x_grid, frac = args
    rng      = np.random.default_rng(seed)
    unique_c = np.unique(countries)
    chosen   = rng.choice(unique_c, size=len(unique_c), replace=True)
    idx      = np.concatenate([np.where(countries == c)[0] for c in chosen])
    xb, yb   = x_vals[idx], y_vals[idx]
    mask     = np.isfinite(xb) & np.isfinite(yb)
    if mask.sum() < 20:
        return None
    try:
        sm = lowess(yb[mask], xb[mask], frac=frac, return_sorted=True)
        return np.interp(x_grid, sm[:, 0], sm[:, 1])
    except Exception:
        return None


def loess_ci(df_sub, y_col, x_grid, n_boots, n_workers, frac, cache_key=None):
    """LOESS + country-cluster bootstrap CI. Results cached to disk.

    The cache is invalidated whenever METRICS' mtime changes, so a
    regenerated city_metrics.parquet always triggers a fresh bootstrap
    rather than silently reusing a CI computed from stale input data.
    """
    x   = df_sub["log_pop"].values
    y   = df_sub[y_col].values
    cty = df_sub["country"].values
    y_hat = fit_loess(x, y, x_grid, frac=frac)
    input_mtime = METRICS.stat().st_mtime

    if cache_key:
        cache_file = CACHE_DIR / f"{cache_key}.npz"
        if cache_file.exists():
            cached = np.load(cache_file)
            if float(cached.get("input_mtime", -1)) == input_mtime:
                return y_hat, cached["ci_lo"], cached["ci_hi"]
            logging.info("  Cache stale for %s (input changed), recomputing ...", cache_key)

    seeds = [int(sq.generate_state(1)[0])
             for sq in np.random.SeedSequence(42).spawn(n_boots)]
    args  = [(sd, x, y, cty, x_grid, frac) for sd in seeds]
    preds = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_boot_worker, args):
            if res is not None:
                preds.append(res)
    preds = np.array(preds)
    ci_lo = np.percentile(preds, 2.5,  axis=0)
    ci_hi = np.percentile(preds, 97.5, axis=0)

    if cache_key:
        np.savez(cache_file, ci_lo=ci_lo, ci_hi=ci_hi, input_mtime=input_mtime)
        logging.info("  Cached: %s", cache_key)

    return y_hat, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Axes style
# ---------------------------------------------------------------------------
def _style_ax(ax):
    ax.set_facecolor("white")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#AAAAAA")
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors="#555555", labelsize=7)
    ax.xaxis.label.set_color("#555555")
    ax.yaxis.label.set_color("#555555")


# ---------------------------------------------------------------------------
# One figure
# ---------------------------------------------------------------------------
def plot_one_group(df, y_col, title, n_boots, n_workers,
                   y_label="Group share", y_lim=None):
    # Rank communities by city count (descending); assign geographic names
    communities_ranked = df["community"].value_counts().index.tolist()
    rank_label = {orig: COMM_NAMES_RANKED[i]
                  for i, orig in enumerate(communities_ranked)}
    rank_color = {orig: PALETTE_9[orig] for orig in communities_ranked}

    x_lo   = np.nanpercentile(df["log_pop"], 1)
    x_hi   = np.nanpercentile(df["log_pop"], 99.5)
    x_grid = np.linspace(x_lo, x_hi, 400)

    fig, ax = plt.subplots(figsize=(150 * MM, 115 * MM))
    fig.patch.set_facecolor("white")

    # Grey scatter
    ax.scatter(df["log_pop"], df[y_col],
               c="#BBBBBB", s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
               edgecolors="none", linewidths=0, rasterized=True, zorder=2)

    # Per-community LOESS + CI
    for comm in communities_ranked:
        sub = df[df["community"] == comm]
        logging.info("  Comm %d → %s (n=%d) ...", comm, rank_label[comm], len(sub))
        cache_key = f"comm{comm}_{y_col}_b{n_boots}"
        y_hat, ci_lo, ci_hi = loess_ci(
            sub, y_col, x_grid, n_boots, n_workers,
            frac=LOESS_FRAC, cache_key=cache_key,
        )
        col = rank_color[comm]
        ax.fill_between(x_grid, ci_lo, ci_hi,
                        color=col, alpha=0.12, linewidth=0, zorder=3)
        ax.plot(x_grid, y_hat, color=col, lw=1.6, zorder=4,
                label=rank_label[comm])

    # Reference vlines
    for vx in VLINE_X:
        if x_lo <= vx <= x_hi:
            ax.axvline(vx, color="#AAAAAA", lw=0.8, ls="--", zorder=1, alpha=0.8)

    ax.set_xlim(x_lo - 0.1, x_hi + 0.1)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    else:
        ax.set_ylim(-0.02, min(df[y_col].quantile(0.995) + 0.05, 1.02))
    ax.set_title(title, fontsize=9, pad=6, fontweight="bold")
    ax.set_xlabel("log(population)", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.tick_params(which="major", labelsize=7, length=4)
    ax.tick_params(which="minor", length=2, labelbottom=False, labelleft=False)
    _style_ax(ax)

    comm_handles = [
        mlines.Line2D([], [], color=rank_color[c], lw=1.6, label=rank_label[c])
        for c in communities_ranked
    ]
    ax.legend(handles=comm_handles,
              fontsize=5.5, frameon=False, loc="upper right", ncol=1)

    fig.tight_layout(pad=1.2)
    safe = y_col.replace("*", "star")
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"comm_scatter_{safe}.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved: comm_scatter_%s.pdf/png", safe)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_boots=N_BOOTS, n_workers=N_BOOT_WORKERS):
    logging.info("Loading city metrics ...")
    df = pd.read_parquet(METRICS)
    df = df.dropna(subset=["log_pop", "community", "p1", "p2", "p3", "p4"]).copy()
    df["community"] = df["community"].astype(int)
    logging.info("  %d cities, %d communities.", len(df), df["community"].nunique())

    for y_col, title in GROUP_NAMES.items():
        logging.info("=== %s ===", y_col)
        plot_one_group(df, y_col, title, n_boots=n_boots, n_workers=n_workers)

    logging.info("=== H17_star ===")
    df_h = df.dropna(subset=["H17_star"]).copy()
    plot_one_group(df_h, "H17_star", "Morphological diversity (H*, 17 types)",
                   n_boots=n_boots, n_workers=n_workers,
                   y_label="Normalised Shannon diversity H* (k=17)",
                   y_lim=(-0.02, 1.02))

    logging.info("=== r17 ===")
    df_r = df.dropna(subset=["r17"]).copy()
    plot_one_group(df_r, "r17", "Spatial assortativity (r, 17 types)",
                   n_boots=n_boots, n_workers=n_workers,
                   y_label="Newman assortativity r (k=17)",
                   y_lim=(-0.25, 0.55))

    logging.info("Done. Figures in %s", FIG_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boots",   type=int, default=N_BOOTS)
    parser.add_argument("--workers", type=int, default=N_BOOT_WORKERS)
    args = parser.parse_args()
    main(n_boots=args.boots, n_workers=args.workers)
