"""
02_plot_hierarchy.py
====================
Continental urban hierarchy figure for publication.

Output
------
    outputs/figures/urban_hierarchy.pdf/png

    Single axes: four morphological group shares (p1–p4) vs log(population).
    LOESS smooth + 95% country-cluster bootstrap CI per group.
    Curve crossings marked with light dashed vertical lines.

Usage
-----
    python 02_plot_hierarchy.py [--boots N] [--workers N]

    --boots     Bootstrap resamples (default: 500)
    --workers   Parallel workers (default: 14)
"""

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

GROUP_NAMES = {
    1: "Medium-density built-up",
    2: "High-density compact",
    3: "Sparse vegetated",
    4: "Low-density heterogeneous",
}
GROUP_COLORS = {
    1: "#714F91",   # purple
    2: "#E4391B",   # red
    3: "#F9992A",   # amber
    4: "#739CCD",   # sky blue
}

LOESS_FRAC        = 0.35
N_BOOTS_DEFAULT   = 500
N_WORKERS_DEFAULT = 14


# ---------------------------------------------------------------------------
# LOESS + country-cluster bootstrap CI
# ---------------------------------------------------------------------------
def _fit_loess(x, y, x_grid, frac=LOESS_FRAC):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.full(len(x_grid), np.nan)
    sm = lowess(y[mask], x[mask], frac=frac, return_sorted=True)
    return np.interp(x_grid, sm[:, 0], sm[:, 1])


def _loess_boot_worker(args):
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


def _loess_ci(df, y_col, x_grid, n_boots, n_workers, frac=LOESS_FRAC):
    """LOESS point estimate + 95% country-cluster bootstrap CI (cached)."""
    cache_file = CACHE_DIR / f"loess_ci_{y_col}_b{n_boots}.npz"
    x   = df["log_pop"].values
    y   = df[y_col].values
    cty = df["country"].values
    y_hat = _fit_loess(x, y, x_grid, frac=frac)

    if cache_file.exists():
        logging.info("  Loading cached CI for %s ...", y_col)
        cached = np.load(cache_file)
        return y_hat, cached["ci_lo"], cached["ci_hi"]

    seeds = [int(sq.generate_state(1)[0])
             for sq in np.random.SeedSequence(42).spawn(n_boots)]
    args  = [(sd, x, y, cty, x_grid, frac) for sd in seeds]
    preds = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_loess_boot_worker, args):
            if res is not None:
                preds.append(res)
    preds = np.array(preds)
    ci_lo = np.percentile(preds, 2.5,  axis=0)
    ci_hi = np.percentile(preds, 97.5, axis=0)
    np.savez(cache_file, ci_lo=ci_lo, ci_hi=ci_hi)
    logging.info("  CI cached: %s", cache_file.name)
    return y_hat, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Axes style
# ---------------------------------------------------------------------------
def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#AAAAAA")
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors="#555555", labelsize=7)
    ax.xaxis.label.set_color("#555555")
    ax.yaxis.label.set_color("#555555")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def plot_hierarchy(df, n_boots, n_workers):
    x_lo   = np.nanpercentile(df["log_pop"], 1)
    x_hi   = np.nanpercentile(df["log_pop"], 99.5)
    x_grid = np.linspace(x_lo, x_hi, 400)

    fig, ax = plt.subplots(figsize=(130 * MM, 100 * MM))
    fig.patch.set_facecolor("white")

    curves_hat = {}
    for g in range(1, 5):
        y_col = f"p{g}"
        col   = GROUP_COLORS[g]
        label = GROUP_NAMES[g]
        logging.info("  LOESS + CI: group %d ...", g)
        y_hat, lo, hi = _loess_ci(df, y_col, x_grid,
                                   n_boots=n_boots, n_workers=n_workers)
        curves_hat[g] = y_hat
        ax.fill_between(x_grid, lo, hi, color=col, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x_grid, y_hat, color=col, lw=1.4, label=label, zorder=3)

    # Curve crossings: light dashed vlines
    for ga, gb in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]:
        diff = curves_hat[ga] - curves_hat[gb]
        for idx in np.where(np.diff(np.sign(diff)))[0]:
            x0, x1 = x_grid[idx], x_grid[idx + 1]
            d0, d1 = diff[idx], diff[idx + 1]
            x_cross = x0 - d0 * (x1 - x0) / (d1 - d0)
            ax.axvline(x_cross, color="#CCCCCC", lw=0.6, ls="--", zorder=1, alpha=0.7)

    ax.set_xlim(x_lo - 0.1, x_hi + 0.1)
    ax.set_ylim(-0.02, 0.52)
    ax.set_xlabel("log(population)", fontsize=8)
    ax.set_ylabel("Group share", fontsize=8)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.05))
    ax.tick_params(which="major", labelsize=7, length=4)
    ax.tick_params(which="minor", length=2, labelbottom=False, labelleft=False)
    ax.legend(fontsize=6, frameon=False, loc="upper right",
              ncol=1, handlelength=1.5, handletextpad=0.5)
    _despine(ax)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"urban_hierarchy.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved: urban_hierarchy.pdf/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_boots=N_BOOTS_DEFAULT, n_workers=N_WORKERS_DEFAULT):
    logging.info("Loading city metrics ...")
    df = pd.read_parquet(METRICS)
    df = df.dropna(subset=["log_pop", "community", "p1", "p2", "p3", "p4"]).copy()
    df["community"] = df["community"].astype(int)
    logging.info("  %d cities loaded.", len(df))

    plot_hierarchy(df, n_boots=n_boots, n_workers=n_workers)
    logging.info("Done. Figure in %s", FIG_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boots",   type=int, default=N_BOOTS_DEFAULT)
    parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
    args = parser.parse_args()
    main(n_boots=args.boots, n_workers=args.workers)
