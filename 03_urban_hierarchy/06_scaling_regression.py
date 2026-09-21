"""
06_scaling_regression.py
========================
Continent-wide standardized OLS scaling regressions for H17* and r17.

Two models per outcome:
  M1 (raw)      : Y_z ~ X_z                          (OLS, country-clustered SE)
  M2 (adjusted) : Y_z ~ X_z + community_FE           (OLS, country-clustered SE)

Where:
  Y_z = (Y - mean(Y)) / sd(Y)     [H17_star or r17, standardized]
  X_z = (log_pop - mean) / sd     [standardized log population]

Output
------
  outputs/tables/scaling_regression.csv   — coefficient table
  outputs/figures/scaling_coef.pdf/png    — coefficient plot (main text figure)

Usage
-----
    python 06_scaling_regression.py
"""

import logging
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
    "font.family":     "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

ROOT      = Path(__file__).resolve().parent
METRICS   = ROOT / "outputs" / "tables" / "city_metrics.parquet"
FIG_DIR   = ROOT / "outputs" / "figures"
TAB_DIR   = ROOT / "outputs" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4

# Community geographic names (original ID 0–7 → name)
COMM_NAMES = {
    0: "West African Inland",
    1: "Gulf of Guinea Coastal",
    2: "Southern–East African",
    3: "East–Central African",
    4: "Sahel–Central African Interior",
    5: "North African–Nile",
    6: "Nile Urban Corridor",
    7: "East African Highlands",
}

OUTCOMES = {
    "H17_star": "Morphological diversity H* (k=17)",
    "r17":      "Spatial assortativity r (k=17)",
}

# Colours for the two outcomes in coefficient plot
OUTCOME_COLORS = {
    "H17_star": "#2166AC",  # blue
    "r17":      "#D6604D",  # red-orange
}


# ---------------------------------------------------------------------------
# OLS with country-clustered SE
# ---------------------------------------------------------------------------
def run_ols(df, formula, cluster_var="country"):
    """Fit OLS and return country-clustered SE results."""
    model = smf.ols(formula, data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": df[cluster_var]},
    )
    return model


def extract_coef(result, param="X_z"):
    """Extract β, 95% CI, R², p-value for a named parameter."""
    beta   = result.params[param]
    ci_lo  = result.conf_int().loc[param, 0]
    ci_hi  = result.conf_int().loc[param, 1]
    pval   = result.pvalues[param]
    r2     = result.rsquared
    r2_adj = result.rsquared_adj
    n      = int(result.nobs)
    return dict(beta=beta, ci_lo=ci_lo, ci_hi=ci_hi,
                pval=pval, r2=r2, r2_adj=r2_adj, n=n)


# ---------------------------------------------------------------------------
# Main regression loop
# ---------------------------------------------------------------------------
def run_regressions(df):
    records = []

    for outcome, outcome_label in OUTCOMES.items():
        df_sub = df.dropna(subset=[outcome, "log_pop", "community", "country"]).copy()

        # Standardize X and Y
        mu_x, sd_x = df_sub["log_pop"].mean(), df_sub["log_pop"].std()
        mu_y, sd_y = df_sub[outcome].mean(),   df_sub[outcome].std()
        df_sub["X_z"] = (df_sub["log_pop"] - mu_x) / sd_x
        df_sub["Y_z"] = (df_sub[outcome]   - mu_y) / sd_y

        n_countries = df_sub["country"].nunique()
        logging.info("  %s: n=%d cities, %d countries, SD(Y)=%.4f",
                     outcome, len(df_sub), n_countries, sd_y)

        # M1: raw (no community FE)
        m1 = run_ols(df_sub, "Y_z ~ X_z")
        c1 = extract_coef(m1, "X_z")
        c1.update(outcome=outcome, model="M1_raw",
                  sd_x=sd_x, sd_y=sd_y, n_countries=n_countries)
        records.append(c1)
        logging.info("    M1 raw:      β=%.4f [%.4f, %.4f]  R²=%.3f  p=%.2e",
                     c1["beta"], c1["ci_lo"], c1["ci_hi"], c1["r2"], c1["pval"])

        # M2: community fixed effects
        m2 = run_ols(df_sub, "Y_z ~ X_z + C(community)")
        c2 = extract_coef(m2, "X_z")
        c2.update(outcome=outcome, model="M2_community_FE",
                  sd_x=sd_x, sd_y=sd_y, n_countries=n_countries)
        records.append(c2)
        logging.info("    M2 comm FE:  β=%.4f [%.4f, %.4f]  R²=%.3f  p=%.2e",
                     c2["beta"], c2["ci_lo"], c2["ci_hi"], c2["r2"], c2["pval"])

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Coefficient plot
# ---------------------------------------------------------------------------
def plot_coef(coef_df):
    """
    Two-panel coefficient plot:
      Left panel:  H17_star  M1 and M2
      Right panel: r17       M1 and M2
    Points = β, horizontal bars = 95% CI.
    """
    fig, axes = plt.subplots(1, 2, figsize=(140 * MM, 70 * MM), sharey=False)
    fig.patch.set_facecolor("white")

    model_labels = {"M1_raw": "Raw OLS", "M2_community_FE": "Community FE"}
    model_markers = {"M1_raw": "o", "M2_community_FE": "s"}
    model_offsets = {"M1_raw": 0.12, "M2_community_FE": -0.12}  # y jitter

    for ax, (outcome, outcome_label) in zip(axes, OUTCOMES.items()):
        col = OUTCOME_COLORS[outcome]
        sub = coef_df[coef_df["outcome"] == outcome].copy()

        for i, row in enumerate(sub.itertuples()):
            yo = model_offsets[row.model]
            ax.errorbar(
                row.beta, i + yo,
                xerr=[[row.beta - row.ci_lo], [row.ci_hi - row.beta]],
                fmt=model_markers[row.model],
                color=col, ecolor=col, elinewidth=1.2,
                markersize=5, capsize=3, zorder=4,
                label=model_labels[row.model] if i == 0 else "",
            )
            # Annotate β and R²
            ax.text(row.ci_hi + 0.005, i + yo,
                    f"β={row.beta:.3f}, R²={row.r2:.3f}",
                    va="center", ha="left", fontsize=5.5, color="#444444")

        ax.axvline(0, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(
            [model_labels[m] for m in sub["model"]], fontsize=6.5)
        ax.set_xlabel("Standardised β (95% CI)", fontsize=7)
        ax.set_title(outcome_label, fontsize=7.5, fontweight="bold", pad=5)

        # Style
        ax.set_facecolor("white")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("#AAAAAA")
            ax.spines[sp].set_linewidth(0.6)
        ax.tick_params(colors="#555555", labelsize=6.5)
        ax.xaxis.label.set_color("#555555")

    fig.tight_layout(pad=1.5, w_pad=3.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"scaling_coef.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logging.info("Saved: scaling_coef.pdf/png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logging.info("Loading city metrics ...")
    df = pd.read_parquet(METRICS)
    df = df.dropna(subset=["log_pop", "community"]).copy()
    df["community"] = df["community"].astype(int)
    logging.info("  %d cities loaded.", len(df))

    logging.info("Running continent-wide regressions ...")
    coef_df = run_regressions(df)

    # Save table
    out_csv = TAB_DIR / "scaling_regression.csv"
    coef_df.to_csv(out_csv, index=False, float_format="%.6f")
    logging.info("Saved: %s", out_csv)

    # Pretty print
    print("\n=== Scaling regression results ===")
    display_cols = ["outcome", "model", "beta", "ci_lo", "ci_hi",
                    "pval", "r2", "r2_adj", "n", "n_countries"]
    print(coef_df[display_cols].to_string(index=False, float_format="{:.4f}".format))

    logging.info("Plotting coefficient figure ...")
    plot_coef(coef_df)

    logging.info("Done.")


if __name__ == "__main__":
    main()
