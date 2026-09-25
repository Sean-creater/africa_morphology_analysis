"""
02_multilevel_sensitivity.py
============================
Sensitivity analysis: per-community binary mixed logit with country random
intercepts, answering the question "how much of community membership is
explained by national membership alone, beyond individual-city covariates?"

For each community k (1–9) we estimate four binary mixed logit models:
    B0k  null (intercept + country RE only)
    B1k  + climate zone
    B2k  + climate zone + GDP + population
    B3k  + climate zone + GDP + population + colonial legacy

From each model we extract:
    ICC   = σ²_u / (σ²_u + π²/3)       (country-level variance proportion)
    PCV   = 1 – σ²_u(model) / σ²_u(B0) (proportional change in variance,
                                          i.e. how much country RE is explained)

The variance decomposition table shows how ICC falls as covariates are added,
and what residual country-level heterogeneity remains after accounting for
all measured factors.

ICC is estimated via the observed-residual-variance approach: fit a
standard binary logit (no random effects), then compute the country-level
variance of the log-odds residuals as σ²_u. See the manuscript's Methods
and Supplementary Information for the full derivation.

Outputs
-------
outputs/tables/multilevel_variance_decomp.csv
    community × model × {loglik, n_params, sigma2_u, icc, pcv}

Usage
-----
    python 02_multilevel_sensitivity.py
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT    = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_CLIMATE  = "Aw"
REF_COLONIAL = "British"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_data():
    """Population is WorldPop 2022 (log_pop_worldpop_2022) — see
    00_prepare_data.py's docstring for the population-definition policy."""
    path = OUT_DIR / "analysis_input.csv"
    df = pd.read_csv(path).dropna(
        subset=["climate_zone", "colonial", "log_gdp", "log_pop_worldpop_2022"]
    ).reset_index(drop=True)
    df["log_gdp_std"] = (df["log_gdp"] - df["log_gdp"].mean()) / df["log_gdp"].std()
    df["log_pop_std"] = (
        (df["log_pop_worldpop_2022"] - df["log_pop_worldpop_2022"].mean())
        / df["log_pop_worldpop_2022"].std()
    )
    return df


def make_dummies(series, prefix, ref):
    d = pd.get_dummies(series, prefix=prefix).astype(int)
    ref_col = f"{prefix}_{ref}"
    return d.drop(columns=[ref_col], errors="ignore")


# ---------------------------------------------------------------------------
# Fit binary logit and compute residual ICC
# ---------------------------------------------------------------------------
def fit_binary_logit(y_bin, X_with_const):
    """Fit binary logit; return result or None."""
    for method in ("bfgs", "lbfgs", "newton"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = Logit(y_bin, X_with_const).fit(
                    method=method, maxiter=2000, disp=False, gtol=1e-5)
            if not np.isnan(res.llf):
                return res
        except Exception:
            continue
    return None


def residual_icc(res, df_sub):
    """
    Compute residual ICC from binary logit model via country-level residuals.
    ICC = σ²_u / (σ²_u + π²/3).
    σ²_u = variance of country-mean Pearson residuals on the log-odds scale.
    """
    mu  = res.predict()
    # Pearson residuals in logistic model
    resid = (df_sub["_y"].values - mu) / np.sqrt(mu * (1 - mu) + 1e-10)
    # Country-mean residuals
    df_r  = pd.DataFrame({"country": df_sub["country"].values, "resid": resid})
    c_var = df_r.groupby("country")["resid"].mean().var()
    icc   = c_var / (c_var + (np.pi ** 2 / 3))
    return float(c_var), float(icc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df   = load_data()
    comms = sorted(df["community"].unique())
    log.info("Communities: %s", comms)

    clim_dum = make_dummies(df["climate_zone"], "clim", REF_CLIMATE)
    col_dum  = make_dummies(df["colonial"],     "col",  REF_COLONIAL)

    rows = []

    for comm in comms:
        log.info("Community %d ...", comm)
        df_sub = df.copy()
        df_sub["_y"] = (df_sub["community"] == comm).astype(int)

        # Build 5 design matrices (GDP and population entered separately)
        const = pd.DataFrame({"const": np.ones(len(df))})
        X_dict = {
            "B0": const.copy(),
            "B1": pd.concat([const, clim_dum], axis=1),
            "B2": pd.concat([const, clim_dum,
                             df[["log_gdp_std"]]], axis=1),
            "B2b": pd.concat([const, clim_dum,
                              df[["log_gdp_std", "log_pop_std"]]], axis=1),
            "B3": pd.concat([const, clim_dum,
                             df[["log_gdp_std", "log_pop_std"]],
                             col_dum], axis=1),
        }

        sigma2_null = None

        for mname, X in X_dict.items():
            res = fit_binary_logit(df_sub["_y"].values, X.values.astype(float))
            if res is None:
                log.warning("Community %d model %s failed.", comm, mname)
                rows.append({
                    "community": comm, "model": mname,
                    "n_obs": len(df_sub), "n_positive": int(df_sub["_y"].sum()),
                    "loglik": np.nan, "n_params": X.shape[1],
                    "sigma2_u": np.nan, "icc": np.nan, "pcv": np.nan,
                    "pseudo_r2": np.nan,
                })
                continue

            sigma2_u, icc = residual_icc(res, df_sub)
            if mname == "B0":
                sigma2_null = sigma2_u

            pcv = (1 - sigma2_u / sigma2_null) if (
                sigma2_null and sigma2_null > 0) else np.nan
            r2  = 1 - res.llf / res.llnull

            rows.append({
                "community":    comm,
                "model":        mname,
                "n_obs":        len(df_sub),
                "n_positive":   int(df_sub["_y"].sum()),
                "loglik":       round(float(res.llf), 3),
                "n_params":     X.shape[1],
                "pseudo_r2":    round(r2, 4),
                "sigma2_u":     round(sigma2_u, 6),
                "icc":          round(icc, 4),
                "pcv":          round(pcv, 4) if not np.isnan(pcv) else np.nan,
            })
            log.info("  %s  llf=%.1f  ICC=%.3f  PCV=%.3f",
                     mname, res.llf, icc, pcv if not np.isnan(pcv) else 0)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "multilevel_variance_decomp.csv", index=False)
    log.info("Saved multilevel_variance_decomp.csv (%d rows).", len(out))

    # Console pivot
    pivot = out.pivot_table(
        index="community", columns="model",
        values=["icc", "pcv", "pseudo_r2"],
        aggfunc="first"
    )
    print("\n=== Residual ICC by community and model ===")
    print(pivot["icc"].to_string())
    print("\n=== PCV (proportion of country variance explained) ===")
    print(pivot["pcv"].to_string())


if __name__ == "__main__":
    main()
