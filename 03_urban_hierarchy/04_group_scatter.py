"""
04_group_scatter.py
===================
Four figures (one per morphological group p1–p4):
scatter coloured by community + per-community LOESS curves
+ continental LOESS reference (black dashed) + 95% bootstrap CI band.

Output
------
    outputs/figures/group_scatter_p1.pdf/png  … group_scatter_p4.pdf/png

Usage
-----
    python 04_group_scatter.py [--boots N] [--workers N]
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

# Community colours (original community ID 0–7)
COMM_COLORS = {
    0: "#4C72B0",
    1: "#DD8452",
    2: "#55A868",
    3: "#C44E52",
    4: "#9674B4",
    5: "#8C7156",
    6: "#D5739D",
    7: "#797979",
}

GROUP_NAMES = {
    "p1": "G1 — Medium-density built-up",
    "p2": "G2 — High-density compact",
    "p3": "G3 — Sparse vegetated",
    "p4": "G4 — Low-density heterogeneous",
}

SCATTER_SIZE   = 6
SCATTER_ALPHA  = 0.40
LOESS_FRAC     = 0.35
N_BOOTS        = 500
N_BOOT_WORKERS = 14


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


def continental_loess_ci(df, y_col, x_grid, n_boots, n_workers,
                         frac=LOESS_FRAC, cache_key=None):
    """Return (y_hat, ci_lo, ci_hi) from country-cluster bootstrap LOESS.

    The cache (keyed by cache_key) is invalidated whenever METRICS' mtime
    changes, so a regenerated city_metrics.parquet always triggers a fresh
    bootstrap rather than silently reusing a CI computed from stale input.
    """
    x   = df["log_pop"].values
    y   = df[y_col].values
    cty = df["country"].values
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
        ax.spines[sp].set_linewidth(0.8)
    ax.tick_params(colors="#555555", labelsize=7)
    ax.xaxis.label.set_color("#555555")
    ax.yaxis.label.set_color("#555555")
    ax.title.set_color("#222222")
    ax.grid(axis="y", color="#EEEEEE", lw=0.6, zorder=0)


# ---------------------------------------------------------------------------
# One figure
# ---------------------------------------------------------------------------
def plot_one_group(df, y_col, title, n_boots, n_workers):
    communities = sorted(df["community"].unique())
    x_lo   = np.nanpercentile(df["log_pop"], 1)
    x_hi   = np.nanpercentile(df["log_pop"], 99.5)
    x_grid = np.linspace(x_lo, x_hi, 400)

    fig, ax = plt.subplots(figsize=(130 * MM, 108 * MM))
    fig.patch.set_facecolor("white")

    # Scatter: coloured by community
    x_all = df["log_pop"].values
    y_all = df[y_col].values
    for comm in communities:
        mask = df["community"].values == comm
        ax.scatter(x_all[mask], y_all[mask],
                   c=COMM_COLORS[comm], s=SCATTER_SIZE, alpha=SCATTER_ALPHA,
                   edgecolors="none", linewidths=0, rasterized=True, zorder=2)

    # Per-community LOESS curves
    for comm in communities:
        mask = df["community"].values == comm
        yhat = fit_loess(x_all[mask], y_all[mask], x_grid)
        ax.plot(x_grid, yhat, color=COMM_COLORS[comm], lw=1.1, zorder=3)

    # Continental LOESS + CI
    logging.info("  Continental LOESS + CI for %s ...", y_col)
    cache_key = f"loess_ci_{y_col}_b{n_boots}"
    y_cont, ci_lo, ci_hi = continental_loess_ci(
        df, y_col, x_grid, n_boots, n_workers, cache_key=cache_key)
    ax.fill_between(x_grid, ci_lo, ci_hi,
                    color="#BBBBBB", alpha=0.55, linewidth=0, zorder=4)
    ax.plot(x_grid, y_cont, color="black", lw=2.0, linestyle="--", zorder=5)

    ax.set_xlim(x_lo - 0.1, x_hi + 0.1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=9, pad=6, fontweight="bold")
    ax.set_xlabel("log(population)", fontsize=8)
    ax.set_ylabel("Group share", fontsize=8)
    _style_ax(ax)

    comm_handles = [
        mlines.Line2D([], [], color=COMM_COLORS[c], lw=1.5,
                      label=f"Community {c}")
        for c in communities
    ]
    africa_handle = mlines.Line2D(
        [], [], color="black", lw=2.0, linestyle="--",
        label="Africa LOESS ± 95% CI"
    )
    ax.legend(handles=comm_handles + [africa_handle],
              fontsize=5.5, frameon=False, loc="upper right", ncol=1)

    fig.tight_layout(pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"group_scatter_{y_col}.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved: group_scatter_%s.pdf/png", y_col)


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

    logging.info("Done. Figures in %s", FIG_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boots",   type=int, default=N_BOOTS)
    parser.add_argument("--workers", type=int, default=N_BOOT_WORKERS)
    args = parser.parse_args()
    main(n_boots=args.boots, n_workers=args.workers)
