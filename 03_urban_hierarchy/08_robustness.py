"""
08_robustness.py
================
Robustness checks for the city-size scaling regressions (06_scaling_regression.py).

Five specifications per outcome, all standardised OLS with country-clustered SE:

  Main          : Y_z ~ X_z
  Community FE  : Y_z ~ X_z + C(community)
  Country FE    : Y_z ~ X_z + C(country)
  ≥50 cells     : Y_z ~ X_z  [cities with n_cells ≥ 50 only]
  4-group       : H_star / r  [4-group versions of the main outcomes]

Outcomes: H17_star and r17 (main); H_star and r (4-group robustness).

Output
------
    outputs/tables/robustness.csv
    outputs/figures/robustness_coef.pdf/png

Usage
-----
    python 08_robustness.py
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

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

ROOT    = Path(__file__).resolve().parent
METRICS = ROOT / "outputs" / "tables" / "city_metrics.parquet"
FIG_DIR = ROOT / "outputs" / "figures"
TAB_DIR = ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4

MIN_CELLS = 50   # threshold for sample-sensitivity check

# Specification display labels (ordered for plot)
SPEC_LABELS = [
    "Main",
    "+ Community FE",
    "+ Country FE",
    r"Cities $\geq$50 cells",
    "4-group metrics",
]

OUTCOME_COLORS = {
    "H17_star": "#2166AC",
    "r17":      "#D6604D",
}

OUTCOME_LABELS = {
    "H17_star": r"$H^*$ (diversity)",
    "r17":      r"$r$ (assortativity)",
}


# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------
def run_ols(df, formula, cluster_var="country"):
    return smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_var]},
    )


def extract_coef(result, param="X_z"):
    ci = result.conf_int().loc[param]
    return dict(
        beta   = result.params[param],
        ci_lo  = ci[0],
        ci_hi  = ci[1],
        pval   = result.pvalues[param],
        r2     = result.rsquared,
        n      = int(result.nobs),
    )


def standardise(df, x_col, y_col):
    """Add X_z and Y_z columns (in-place copy)."""
    d = df.copy()
    d["X_z"] = (d[x_col] - d[x_col].mean()) / d[x_col].std()
    d["Y_z"] = (d[y_col] - d[y_col].mean()) / d[y_col].std()
    return d


# ---------------------------------------------------------------------------
# Run all specifications
# ---------------------------------------------------------------------------
def run_robustness(df):
    records = []

    for outcome in ["H17_star", "r17"]:
        # Determine 4-group analogue
        alt_outcome = "H_star" if outcome == "H17_star" else "r"

        base_cols = [outcome, alt_outcome, "log_pop", "community", "country", "n_cells"]
        df_base = df.dropna(subset=[outcome, "log_pop", "community", "country"]).copy()

        logging.info("=== %s (n=%d) ===", outcome, len(df_base))

        # 1. Main
        d = standardise(df_base, "log_pop", outcome)
        c = extract_coef(run_ols(d, "Y_z ~ X_z"))
        c.update(outcome=outcome, spec="Main")
        records.append(c)
        logging.info("  Main:          β=%.4f [%.4f, %.4f]  R²=%.3f",
                     c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

        # 2. Community FE
        d = standardise(df_base, "log_pop", outcome)
        c = extract_coef(run_ols(d, "Y_z ~ X_z + C(community)"))
        c.update(outcome=outcome, spec="Community FE")
        records.append(c)
        logging.info("  + Community FE: β=%.4f [%.4f, %.4f]  R²=%.3f",
                     c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

        # 3. Country FE
        d = standardise(df_base, "log_pop", outcome)
        c = extract_coef(run_ols(d, "Y_z ~ X_z + C(country)"))
        c.update(outcome=outcome, spec="Country FE")
        records.append(c)
        logging.info("  + Country FE:   β=%.4f [%.4f, %.4f]  R²=%.3f",
                     c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

        # 4. Sample sensitivity: n_cells >= MIN_CELLS
        df_large = df_base[df_base["n_cells"] >= MIN_CELLS].copy()
        d = standardise(df_large, "log_pop", outcome)
        c = extract_coef(run_ols(d, "Y_z ~ X_z"))
        c.update(outcome=outcome, spec=f"Cities ≥{MIN_CELLS} cells")
        records.append(c)
        logging.info("  ≥%d cells (n=%d): β=%.4f [%.4f, %.4f]  R²=%.3f",
                     MIN_CELLS, len(df_large),
                     c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

        # 5. 4-group resolution
        df_alt = df.dropna(subset=[alt_outcome, "log_pop", "community", "country"]).copy()
        d = standardise(df_alt, "log_pop", alt_outcome)
        c = extract_coef(run_ols(d, "Y_z ~ X_z"))
        c.update(outcome=outcome, spec="4-group")
        records.append(c)
        logging.info("  4-group (%s, n=%d): β=%.4f [%.4f, %.4f]  R²=%.3f",
                     alt_outcome, len(df_alt),
                     c["beta"], c["ci_lo"], c["ci_hi"], c["r2"])

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Coefficient plot
# ---------------------------------------------------------------------------
def plot_robustness(rob_df):
    """
    Two-panel coefficient plot (H17_star | r17).
    Rows = 5 specifications; points = β; horizontal bars = 95% CI.
    """
    spec_order = ["Main", "Community FE", "Country FE",
                  f"Cities ≥{MIN_CELLS} cells", "4-group"]

    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 90 * MM), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, outcome in zip(axes, ["H17_star", "r17"]):
        col = OUTCOME_COLORS[outcome]
        sub = rob_df[rob_df["outcome"] == outcome].copy()
        sub["spec_order"] = sub["spec"].map({s: i for i, s in enumerate(spec_order)})
        sub = sub.sort_values("spec_order")

        y_pos = np.arange(len(sub))
        ax.errorbar(
            sub["beta"], y_pos,
            xerr=[sub["beta"] - sub["ci_lo"], sub["ci_hi"] - sub["beta"]],
            fmt="o", color=col, ecolor=col, elinewidth=1.2,
            markersize=5, capsize=3, zorder=3,
        )

        # Annotate β
        for _, row in sub.iterrows():
            i = int(row["spec_order"])
            ax.text(row["ci_hi"] + 0.005, i,
                    f"β={row['beta']:.3f}",
                    va="center", ha="left", fontsize=6, color="#444444")

        ax.axvline(0, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
        ax.set_yticks(y_pos)

        # Y-tick labels only on left panel
        if ax is axes[0]:
            labels = []
            for s in spec_order:
                if s == f"Cities ≥{MIN_CELLS} cells":
                    labels.append(f"Cities ≥{MIN_CELLS} cells")
                elif s == "4-group":
                    labels.append("4-group metrics")
                else:
                    labels.append(s)
            ax.set_yticklabels(labels, fontsize=7)
        else:
            ax.set_yticklabels([])

        ax.set_xlabel("Standardised β (95% CI)", fontsize=7)
        ax.set_title(OUTCOME_LABELS[outcome], fontsize=8, fontweight="bold", pad=5)
        ax.invert_yaxis()

        ax.set_facecolor("white")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("#AAAAAA")
            ax.spines[sp].set_linewidth(0.6)
        ax.tick_params(colors="#555555", labelsize=7)
        ax.xaxis.label.set_color("#555555")

    fig.tight_layout(pad=1.5, w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"robustness_coef.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved: robustness_coef.pdf/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.info("Loading city metrics ...")
    df = pd.read_parquet(METRICS)
    df["community"] = pd.to_numeric(df["community"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["community"]).copy()
    df["community"] = df["community"].astype(int)
    logging.info("  %d cities loaded.", len(df))

    logging.info("Running robustness checks ...")
    rob_df = run_robustness(df)

    out_csv = TAB_DIR / "robustness.csv"
    rob_df.to_csv(out_csv, index=False, float_format="%.6f")
    logging.info("Saved: %s", out_csv)

    print("\n=== Robustness results ===")
    print(rob_df[["outcome", "spec", "beta", "ci_lo", "ci_hi", "pval", "r2", "n"]]
          .to_string(index=False, float_format="{:.4f}".format))

    logging.info("Plotting robustness coefficient figure ...")
    plot_robustness(rob_df)

    logging.info("Done.")


if __name__ == "__main__":
    main()
