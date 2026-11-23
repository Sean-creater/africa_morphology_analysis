"""
06_radial_models.py
===================
Statistical models and figures for intra-urban morphological radial gradient
analysis, across three distance frameworks.

Analyses
--------
Model A — CPC scaling (city-level)
    CPC_z ~ log(Pop)_z   (country-clustered SE)
    Three distance frameworks: rel / abs / scl
    Three specs: M1 raw, M2 community FE, M3 country FE

Model B — Radial H17_rare gradient (city x ring level)
    H17_rare ~ C(ring, Treatment(1)) * X_z   (city-clustered SE)
    ring as categorical factor (Treatment coding, ring 1 = reference)
    Three frameworks separately; balanced sample required for scl/rel

Model C — Radial r17 assortativity gradient
    r17 ~ C(ring, Treatment(1)) * X_z   (city-clustered SE)
    Same structure as Model B

Three figures
-------------
Figure A  (radial_abs.pdf/png):   absolute-km distance profiles
Figure B  (radial_scl.pdf/png):   population-scaled distance profiles
Figure C  (radial_rel.pdf/png):   relative-radius profiles (robustness)

Each figure has 4 panels:
  Row 1: 4-group composition by size class (3 facets)
  Row 2: H17_rare by size class (3 facets)
  Row 3: r17 by size class (3 facets)
  Row 4: cumulative population vs land share (1 centred panel)

Also outputs
------------
    outputs/tables/sample_audit.csv     -- cities per ring x ring_type
    outputs/tables/radial_models.csv    -- model coefficients

Usage
-----
    python 06_radial_models.py [--boots N] [--workers N]

Language constraints (cross-sectional data):
    Use: "across the city-size hierarchy", "is associated with"
    Avoid: "as cities grow", "causal", "drives"
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

matplotlib.rcParams.update({
    "font.family":     "Arial",
    "font.size":       8,
    "svg.fonttype": "none",
})

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PROFILES_PATH  = OUT_DIR / "radial_profiles.parquet"
NETWORK_PATH   = OUT_DIR / "radial_network_metrics.parquet"
CITY_DIFF_PATH = OUT_DIR / "city_radial_metrics.csv"
ALLOMETRIC     = OUT_DIR / "allometric_params.json"

MM = 1 / 25.4

GROUP_COLORS = {1: "#714F91", 2: "#E4391B", 3: "#F9992A", 4: "#739CCD"}
GROUP_LABELS = {
    1: "Cropland-Fringe Low-Rise",
    2: "Heterogeneous Sparse Fabric",
    3: "Large-Volume Compact Fabric",
    4: "Dense Small-Building Fabric",
}
SIZE_CLASSES = {
    "Small\n(<200k)":    (0,               np.log(200_000)),
    "Medium\n(200k-1M)": (np.log(200_000), np.log(1_000_000)),
    "Large\n(>1M)":      (np.log(1_000_000), np.inf),
}

# x-axis labels per framework
RING_XLABELS = {
    "abs": ["0-1\nkm", "1-3\nkm", "3-5\nkm", "5-10\nkm", ">10\nkm"],
    "scl": ["Band 1\n(inner)", "Band 2", "Band 3", "Band 4", "Band 5\n(outer)"],
    "rel": ["Core\n(1)", "2", "3", "4", "Periph.\n(5)"],
}

N_BOOTS        = 500
N_BOOT_WORKERS = 14


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _std(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def _run_ols(df, formula, cluster_col):
    return smf.ols(formula, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df[cluster_col]})


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#AAAAAA")
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors="#555555", labelsize=6.5)
    ax.xaxis.label.set_color("#555555")
    ax.yaxis.label.set_color("#555555")


def _set_ring_xticks(ax, rt):
    labels = RING_XLABELS[rt]
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(labels, fontsize=6)


# ---------------------------------------------------------------------------
# Bootstrap: city-equally-weighted mean per ring, country-clustered
# ---------------------------------------------------------------------------
def _boot_worker(args):
    seed, df, y_col, rings = args
    rng = np.random.default_rng(seed)
    countries = df["country"].unique()
    chosen = rng.choice(countries, size=len(countries), replace=True)
    boot = pd.concat([df[df["country"] == c] for c in chosen],
                     ignore_index=True)
    means = []
    for r in rings:
        city_means = boot[boot["ring"] == r].groupby(
            ["city", "country"])[y_col].mean()
        means.append(float(city_means.mean()) if len(city_means) > 0 else np.nan)
    return means


def _ring_means_ci(df, y_col, rings, n_boots, n_workers):
    """City-equally-weighted mean + 95% CI per ring."""
    means = []
    for r in rings:
        cm = df[df["ring"] == r].groupby(["city", "country"])[y_col].mean()
        means.append(float(cm.mean()) if len(cm) > 0 else np.nan)
    args_list = [(s, df, y_col, rings) for s in range(n_boots)]
    boot_mat = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_boot_worker, args_list):
            if res is not None:
                boot_mat.append(res)
    boot_mat = np.array(boot_mat)
    ci_lo = np.nanpercentile(boot_mat, 2.5,  axis=0).tolist()
    ci_hi = np.nanpercentile(boot_mat, 97.5, axis=0).tolist()
    return means, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Bootstrap: cumulative pop vs land (Panel 4)
# ---------------------------------------------------------------------------
def _boot_cumulative_worker(args):
    seed, df, rings = args
    rng = np.random.default_rng(seed)
    countries = df["country"].unique()
    chosen = rng.choice(countries, size=len(countries), replace=True)
    boot = pd.concat([df[df["country"] == c] for c in chosen],
                     ignore_index=True)
    cum_pop, cum_land = [], []
    for r in rings:
        sub = boot[boot["ring"] == r]
        cm_pop  = sub.groupby(["city","country"])["population_share"].mean()
        cm_land = sub.groupby(["city","country"])["land_share"].mean()
        cum_pop.append(float(cm_pop.mean())  if len(cm_pop)  > 0 else np.nan)
        cum_land.append(float(cm_land.mean()) if len(cm_land) > 0 else np.nan)
    return np.cumsum(cum_pop).tolist(), np.cumsum(cum_land).tolist()


# ---------------------------------------------------------------------------
# Panel 1: 4-group composition by size class
# ---------------------------------------------------------------------------
def plot_panel_comp(prof_rt, ax_list, rt, n_boots, n_workers):
    rings = [1, 2, 3, 4, 5]
    bal   = prof_rt[prof_rt["qc_balanced"]].copy()
    for ax, (label, (lo, hi)) in zip(ax_list, SIZE_CLASSES.items()):
        sub = bal[bal["log_pop"].between(lo, hi, inclusive="left")]
        n_cities = sub["city"].nunique()
        for g in [1, 2, 3, 4]:
            col  = GROUP_COLORS[g]
            means, ci_lo, ci_hi = _ring_means_ci(
                sub, f"p{g}", rings, min(n_boots, 200), n_workers)
            ax.fill_between(rings, ci_lo, ci_hi, color=col, alpha=0.15, lw=0)
            ax.plot(rings, means, color=col, lw=1.6, marker="o", markersize=3,
                    label=GROUP_LABELS[g] if ax is ax_list[0] else None)
        _set_ring_xticks(ax, rt)
        ax.set_ylim(-0.02, 0.75)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        if ax is ax_list[0]:
            ax.set_ylabel("Group share", fontsize=7)
        ax.set_title(f"{label}\n(n={n_cities})", fontsize=7, pad=3)
        _despine(ax)
    # Legend in first panel
    handles = [mlines.Line2D([], [], color=GROUP_COLORS[g], lw=1.6,
                             label=GROUP_LABELS[g]) for g in [1, 2, 3, 4]]
    ax_list[0].legend(handles=handles, fontsize=5, frameon=False,
                      loc="upper right", ncol=1)


# ---------------------------------------------------------------------------
# Panel 2: H17_rare by size class
# ---------------------------------------------------------------------------
def plot_panel_diversity(prof_rt, ax_list, rt, n_boots, n_workers):
    rings = [1, 2, 3, 4, 5]
    # Use qc_balanced AND qc_div (n_cells_ring >= 20)
    bal   = prof_rt[prof_rt["qc_balanced"] & prof_rt["qc_div"]].copy()
    color = "#2166AC"
    for ax, (label, (lo, hi)) in zip(ax_list, SIZE_CLASSES.items()):
        sub = bal[bal["log_pop"].between(lo, hi, inclusive="left")]
        n_cities = sub["city"].nunique()
        means, ci_lo, ci_hi = _ring_means_ci(
            sub, "H17_rare", rings, min(n_boots, 200), n_workers)
        ax.fill_between(rings, ci_lo, ci_hi, color=color, alpha=0.20, lw=0)
        ax.plot(rings, means, color=color, lw=1.8, marker="o", markersize=4)
        _set_ring_xticks(ax, rt)
        # Auto y-range with some padding
        valid = [v for v in means if np.isfinite(v)]
        if valid:
            ylo = max(0.0, min(valid) - 0.05)
            yhi = min(1.0, max(valid) + 0.05)
            ax.set_ylim(ylo, yhi)
        if ax is ax_list[0]:
            ax.set_ylabel("H*\u2081\u2087 (rarefied, n=20)", fontsize=7)
        ax.set_title(f"{label}\n(n={n_cities})", fontsize=7, pad=3)
        _despine(ax)


# ---------------------------------------------------------------------------
# Panel 3: r17 by size class
# ---------------------------------------------------------------------------
def plot_panel_assort(net_rt, ax_list, rt, n_boots, n_workers):
    rings = [1, 2, 3, 4, 5]
    bal   = net_rt[net_rt["qc_assort_balanced"]].copy()
    color = "#D6604D"
    for ax, (label, (lo, hi)) in zip(ax_list, SIZE_CLASSES.items()):
        sub = bal[bal["log_pop"].between(lo, hi, inclusive="left")]
        n_cities = sub["city"].nunique()
        means, ci_lo, ci_hi = _ring_means_ci(
            sub, "r17", rings, min(n_boots, 200), n_workers)
        ax.fill_between(rings, ci_lo, ci_hi, color=color, alpha=0.20, lw=0)
        ax.plot(rings, means, color=color, lw=1.8, marker="o", markersize=4)
        ax.axhline(0, color="#CCCCCC", lw=0.6, ls=":", zorder=1)
        _set_ring_xticks(ax, rt)
        valid_lo = [v for v in ci_lo if np.isfinite(v)]
        valid_hi = [v for v in ci_hi if np.isfinite(v)]
        if valid_lo and valid_hi:
            ylo = min(0.0, min(valid_lo)) - 0.01
            yhi = max(valid_hi) + 0.02
            ax.set_ylim(ylo, yhi)
        if ax is ax_list[0]:
            ax.set_ylabel("r\u2081\u2087 (Newman assortativity)", fontsize=7)
        ax.set_title(f"{label}\n(n={n_cities})", fontsize=7, pad=3)
        _despine(ax)


# ---------------------------------------------------------------------------
# Panel 4: cumulative population vs land
# ---------------------------------------------------------------------------
def plot_panel_popland(prof_rt, ax, rt, n_boots, n_workers):
    rings = [1, 2, 3, 4, 5]
    bal   = prof_rt[
        prof_rt["qc_balanced"] &
        prof_rt["population_share"].notna() &
        prof_rt["land_share"].notna()
    ].copy()
    n_cities = bal["city"].nunique()
    logging.info("    Panel 4 [%s]: %d cities.", rt, n_cities)
    if n_cities == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center")
        return {}

    # Point estimates
    cum_pop_pt, cum_land_pt = [], []
    for r in rings:
        sub = bal[bal["ring"] == r]
        cum_pop_pt.append(sub.groupby(["city","country"])["population_share"].mean().mean())
        cum_land_pt.append(sub.groupby(["city","country"])["land_share"].mean().mean())
    cum_pop_pt  = np.cumsum(cum_pop_pt)
    cum_land_pt = np.cumsum(cum_land_pt)

    # Bootstrap
    args_list = [(s, bal, rings) for s in range(min(n_boots, 200))]
    boot_pop, boot_land = [], []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for res in ex.map(_boot_cumulative_worker, args_list):
            if res:
                boot_pop.append(res[0])
                boot_land.append(res[1])
    boot_pop  = np.array(boot_pop)
    boot_land = np.array(boot_land)
    pop_lo  = np.nanpercentile(boot_pop,  2.5,  axis=0)
    pop_hi  = np.nanpercentile(boot_pop,  97.5, axis=0)
    land_lo = np.nanpercentile(boot_land, 2.5,  axis=0)
    land_hi = np.nanpercentile(boot_land, 97.5, axis=0)

    c_pop  = "#2166AC"
    c_land = "#92C5DE"

    ax.fill_between(rings, pop_lo,  pop_hi,  color=c_pop,  alpha=0.20, lw=0)
    ax.fill_between(rings, land_lo, land_hi, color=c_land, alpha=0.25, lw=0)
    ax.plot(rings, cum_pop_pt,  color=c_pop,  lw=2.0, marker="o", ms=4,
            label="Population share")
    ax.plot(rings, cum_land_pt, color=c_land, lw=2.0, marker="s", ms=4,
            ls="--", label="Urban land share")

    core_pop  = float(cum_pop_pt[0])
    core_land = float(cum_land_pt[0])
    ax.annotate(
        f"Ring 1: {core_pop:.0%} pop, {core_land:.0%} land",
        xy=(1, core_pop), xytext=(1.4, core_pop + 0.04),
        fontsize=5.5, color=c_pop,
        arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.5),
    )

    _set_ring_xticks(ax, rt)
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Cumulative share (inner to outer)", fontsize=7)
    ax.set_title(f"Population vs urban land\n(n={n_cities})", fontsize=7, pad=3)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    _despine(ax)

    return dict(core_pop=core_pop, core_land=core_land, n_cities=n_cities)


# ---------------------------------------------------------------------------
# Build one figure (abs / scl / rel)
# ---------------------------------------------------------------------------
def build_figure(rt, prof, network, n_boots, n_workers):
    logging.info("=== Figure [%s] ===", rt)
    prof_rt = prof[prof["ring_type"] == rt].copy()
    net_rt  = network[network["ring_type"] == rt].copy()

    fig = plt.figure(figsize=(190 * MM, 210 * MM))
    fig.patch.set_facecolor("white")

    gs = fig.add_gridspec(
        4, 3,
        left=0.08, right=0.97,
        top=0.96, bottom=0.05,
        hspace=0.60, wspace=0.32,
    )
    ax_1 = [fig.add_subplot(gs[0, i]) for i in range(3)]   # composition
    ax_2 = [fig.add_subplot(gs[1, i]) for i in range(3)]   # diversity
    ax_3 = [fig.add_subplot(gs[2, i]) for i in range(3)]   # assortativity
    ax_4  = fig.add_subplot(gs[3, 1])                       # pop-land (centred)

    logging.info("  Panel 1: composition ...")
    plot_panel_comp(prof_rt, ax_1, rt, n_boots, n_workers)

    logging.info("  Panel 2: H17_rare ...")
    plot_panel_diversity(prof_rt, ax_2, rt, n_boots, n_workers)

    logging.info("  Panel 3: r17 assortativity ...")
    plot_panel_assort(net_rt, ax_3, rt, n_boots, n_workers)

    logging.info("  Panel 4: population vs land ...")
    panel4 = plot_panel_popland(prof_rt, ax_4, rt, n_boots, n_workers)

    # Panel labels
    for ax, lbl in [(ax_1[0], "A"), (ax_2[0], "B"),
                    (ax_3[0], "C"), (ax_4, "D")]:
        ax.text(-0.22, 1.12, lbl, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="top")

    # Figure suptitle
    rt_labels = {"abs": "Absolute distance", "scl": "Population-scaled distance",
                 "rel": "Relative radius (robustness)"}
    fig.suptitle(rt_labels.get(rt, rt), fontsize=8.5, y=0.98)

    stem = f"radial_{rt}"
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"{stem}.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("  Saved: %s.pdf/png", stem)
    return panel4


# ---------------------------------------------------------------------------
# Statistical models
# ---------------------------------------------------------------------------
def run_models(prof, network, city_diff, model_records):

    # === Model A: CPC scaling ================================================
    logging.info("=== Model A: CPC scaling ===")
    for rt in ["rel", "abs", "scl"]:
        cpc_col = f"CPC_{rt}"
        qc_col  = f"qc_cpc_main_{rt}"
        if cpc_col not in city_diff.columns or qc_col not in city_diff.columns:
            continue
        df_m = city_diff[city_diff[qc_col].fillna(False)].dropna(
            subset=["log_pop", cpc_col]).copy()
        if len(df_m) < 50:
            continue
        df_m["X_z"] = _std(df_m["log_pop"])
        df_m["Y_z"] = _std(df_m[cpc_col])
        for spec, formula in [
            ("M1_raw",         "Y_z ~ X_z"),
            ("M2_community_FE","Y_z ~ X_z + C(community)"),
            ("M3_country_FE",  "Y_z ~ X_z + C(country)"),
        ]:
            try:
                res = _run_ols(df_m, formula, "country")
                ci  = res.conf_int().loc["X_z"]
                logging.info("  [%s] %s: beta=%.4f [%.4f,%.4f] R2=%.3f n=%d",
                             rt, spec, res.params["X_z"],
                             ci[0], ci[1], res.rsquared, int(res.nobs))
                model_records.append(dict(
                    outcome=f"CPC_{rt}", spec=spec,
                    beta=res.params["X_z"], ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                    pval=res.pvalues["X_z"], r2=res.rsquared, n=int(res.nobs)))
            except Exception as e:
                logging.warning("    [%s] %s failed: %s", rt, spec, e)

    # === Model B: H17_rare ~ C(ring) * X_z ====================================
    logging.info("=== Model B: H17_rare ~ C(ring)*X_z (city-clustered SE) ===")
    for rt in ["abs", "scl", "rel"]:
        prof_rt = prof[(prof["ring_type"] == rt) & prof["qc_balanced"] &
                       prof["qc_div"]].copy()
        if len(prof_rt) < 200:
            logging.warning("  [%s] too few rows (%d), skip.", rt, len(prof_rt))
            continue
        prof_rt["X_z"] = _std(prof_rt["log_pop"])
        try:
            res_b = smf.ols(
                "H17_rare ~ C(ring, Treatment(1)) * X_z",
                data=prof_rt,
            ).fit(cov_type="cluster", cov_kwds={"groups": prof_rt["city"]})
            logging.info("  [%s] H17_rare model: R2=%.3f n=%d",
                         rt, res_b.rsquared, int(res_b.nobs))
            # Overall ring effect F-test (Wald)
            ring_params = [p for p in res_b.params.index
                           if "ring" in p and "X_z" not in p and p != "Intercept"]
            if ring_params:
                from statsmodels.stats.wald import wald_test
                try:
                    wt = res_b.wald_test("(" + ", ".join(
                        [f"{p} = 0" for p in ring_params]) + ")",
                        use_f=True)
                    logging.info("    Ring F-test: F=%.3f p=%.4f",
                                 float(wt.statistic), float(wt.pvalue))
                except Exception:
                    pass
            for p in res_b.params.index:
                ci = res_b.conf_int().loc[p]
                logging.info("    %-40s beta=%.5f [%.5f,%.5f] p=%.4f",
                             p, res_b.params[p], ci[0], ci[1], res_b.pvalues[p])
            model_records.append(dict(
                outcome=f"H17_rare_{rt}", spec="ring_cat_x_logpop",
                beta=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                pval=np.nan, r2=res_b.rsquared, n=int(res_b.nobs)))
        except Exception as e:
            logging.warning("  [%s] Model B failed: %s", rt, e)

    # === Model C: r17 ~ C(ring) * X_z =========================================
    logging.info("=== Model C: r17 ~ C(ring)*X_z (city-clustered SE) ===")
    for rt in ["abs", "scl", "rel"]:
        net_rt = network[(network["ring_type"] == rt) &
                         network["qc_assort_balanced"]].copy()
        if len(net_rt) < 200:
            logging.warning("  [%s] too few rows (%d), skip.", rt, len(net_rt))
            continue
        net_rt["X_z"] = _std(net_rt["log_pop"])
        try:
            res_c = smf.ols(
                "r17 ~ C(ring, Treatment(1)) * X_z",
                data=net_rt,
            ).fit(cov_type="cluster", cov_kwds={"groups": net_rt["city"]})
            logging.info("  [%s] r17 model: R2=%.3f n=%d",
                         rt, res_c.rsquared, int(res_c.nobs))
            for p in res_c.params.index:
                ci = res_c.conf_int().loc[p]
                logging.info("    %-40s beta=%.5f [%.5f,%.5f] p=%.4f",
                             p, res_c.params[p], ci[0], ci[1], res_c.pvalues[p])
            model_records.append(dict(
                outcome=f"r17_{rt}", spec="ring_cat_x_logpop",
                beta=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                pval=np.nan, r2=res_c.rsquared, n=int(res_c.nobs)))
        except Exception as e:
            logging.warning("  [%s] Model C failed: %s", rt, e)


# ---------------------------------------------------------------------------
# Sample audit
# ---------------------------------------------------------------------------
def make_sample_audit(prof, network):
    records = []
    for rt in ["abs", "scl", "rel"]:
        prof_rt = prof[prof["ring_type"] == rt]
        net_rt  = network[network["ring_type"] == rt]
        for ring in range(1, 6):
            p_sub = prof_rt[prof_rt["ring"] == ring]
            n_sub = net_rt[net_rt["ring"] == ring]
            records.append({
                "ring_type":       rt,
                "ring":            ring,
                "n_cities_comp":   p_sub[p_sub["qc_comp"]]["city"].nunique(),
                "n_cities_div":    p_sub[p_sub["qc_div"]]["city"].nunique(),
                "n_cities_assort": n_sub[n_sub["qc_assort_main"]]["city"].nunique(),
            })
    audit = pd.DataFrame(records)
    audit.to_csv(OUT_DIR / "sample_audit.csv", index=False)
    logging.info("Saved: sample_audit.csv")
    return audit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_boots=N_BOOTS, n_workers=N_BOOT_WORKERS):
    logging.info("Loading data ...")
    prof     = pd.read_parquet(PROFILES_PATH)
    network  = pd.read_parquet(NETWORK_PATH)
    city_diff = pd.read_csv(CITY_DIFF_PATH)
    logging.info("  profiles: %d rows  network: %d rows  city_diff: %d cities",
                 len(prof), len(network), len(city_diff))

    # Sample audit
    audit = make_sample_audit(prof, network)
    print("\n=== Sample audit ===")
    print(audit.to_string(index=False))

    # Statistical models
    model_records = []
    run_models(prof, network, city_diff, model_records)

    # Save model table
    pd.DataFrame(model_records).to_csv(
        OUT_DIR / "radial_models.csv", index=False, float_format="%.6f")
    logging.info("Saved: radial_models.csv")

    # Build figures
    panel4_results = {}
    for rt in ["abs", "scl", "rel"]:
        p4 = build_figure(rt, prof, network, n_boots, n_workers)
        panel4_results[rt] = p4

    # Summary printout
    print("\n=== Panel 4 (population vs land) results ===")
    for rt, p4 in panel4_results.items():
        if p4:
            print(f"  [{rt}] Ring 1: {p4['core_pop']:.1%} pop, "
                  f"{p4['core_land']:.1%} land  (n={p4['n_cities']})")

    logging.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boots",   type=int, default=N_BOOTS)
    parser.add_argument("--workers", type=int, default=N_BOOT_WORKERS)
    args = parser.parse_args()
    main(n_boots=args.boots, n_workers=args.workers)
