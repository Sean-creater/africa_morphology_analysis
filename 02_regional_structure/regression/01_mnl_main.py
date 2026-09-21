"""
01_mnl_main.py
==============
Main multinomial logit analysis answering Q1 and Q2.

Q1 — What structures the continental morphology regions?
    Nested model sequence M0–M4 with drop-one factor comparisons.
    Contribution of each factor group measured by:
      (a) drop-one LRT (Δχ²) against full M3
      (b) partial pseudo-R²  (Δ McFadden R² from null)
      (c) 5-fold cross-validated log-loss drop

Q2 — Do associations differ across communities?
    Average Adjusted Predictions (AAP) for climate zone and colonial legacy.
    Average Marginal Effects (AME, +1 SD) for GDP and population.
    Both computed at M3; bootstrap 95% CI clustered by country.

Model sequence
--------------
M0  intercept only
M1  + climate zone          (12 dummies, ref Aw)
M2  + log_gdp               (standardised)
M2b + log_gdp + log_pop     (standardised)
M3  + colonial legacy       (5 dummies, ref British)   ← primary model
M4  + spatial polynomial    (lat_c, lon_c, lat², lon², lat×lon)

Drop-one from M3
----------------
M3_nc   M3 without climate
M3_ng   M3 without GDP
M3_np   M3 without population
M3_nl   M3 without colonial legacy

Outputs
-------
outputs/tables/mnl_fit.csv
outputs/tables/mnl_lrt.csv
outputs/tables/mnl_factor_contribution.csv   ← Q1 summary
outputs/tables/mnl_coefs_M3.csv
outputs/tables/mnl_aap_climate_M3.csv        ← Q2
outputs/tables/mnl_aap_colonial_M3.csv       ← Q2
outputs/tables/mnl_ame_M3.csv                ← Q2

Usage
-----
    python 01_mnl_main.py
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold

import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT      = Path(__file__).resolve().parent
OUT_DIR   = ROOT / "outputs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REF_CLIMATE  = "Aw"
REF_COLONIAL = "British"
REF_COMM     = 1          # largest community → reference
N_BOOT       = 500
CV_FOLDS     = 5
RNG_SEED     = 42


# ---------------------------------------------------------------------------
# Load & build design matrices
# ---------------------------------------------------------------------------
def load_data():
    path = OUT_DIR / "analysis_input.csv"
    df = pd.read_csv(path).dropna(
        subset=["climate_zone", "colonial", "log_gdp", "log_pop"]
    ).reset_index(drop=True)
    df["log_gdp_std"] = (df["log_gdp"] - df["log_gdp"].mean()) / df["log_gdp"].std()
    df["log_pop_std"] = (df["log_pop"] - df["log_pop"].mean()) / df["log_pop"].std()
    log.info("Loaded %d cities, %d countries, %d communities.",
             len(df), df["country"].nunique(), df["community"].nunique())
    return df


def make_dummies(series, prefix, ref):
    d = pd.get_dummies(series, prefix=prefix).astype(int)
    ref_col = f"{prefix}_{ref}"
    return d.drop(columns=[ref_col], errors="ignore")


def build_X(df, climate=False, gdp=False, pop=False, colonial=False,
            spatial=False):
    parts = []
    if climate:
        parts.append(make_dummies(df["climate_zone"], "clim", REF_CLIMATE))
    if gdp:
        parts.append(df["log_gdp_std"])
    if pop:
        parts.append(df["log_pop_std"])
    if colonial:
        parts.append(make_dummies(df["colonial"], "col", REF_COLONIAL))
    if spatial:
        parts.append(df[["lat_c", "lon_c", "lat2", "lon2", "lat_lon"]])
    if not parts:
        return pd.DataFrame({"_placeholder": np.ones(len(df))}, index=df.index)
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Clustered SE
# ---------------------------------------------------------------------------
def clustered_se(result, clusters):
    unique_g = np.unique(clusters)
    G, n     = len(unique_g), len(clusters)
    params   = np.asarray(result.params)
    scores   = result.model.score_obs(params)
    try:
        bread = np.linalg.inv(-result.model.hessian(params))
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(-result.model.hessian(params))
    meat = np.zeros((scores.shape[1], scores.shape[1]))
    for g in unique_g:
        sg = scores[clusters == g].sum(axis=0)
        meat += np.outer(sg, sg)
    corr = (G / (G - 1)) * ((n - 1) / (n - scores.shape[1]))
    return np.sqrt(np.diag(corr * bread @ meat @ bread))


# ---------------------------------------------------------------------------
# Fit MNLogit
# ---------------------------------------------------------------------------
def fit_mnl(X, y, df, name):
    communities = [REF_COMM] + [c for c in sorted(y.unique()) if c != REF_COMM]
    y_cat = pd.Categorical(y, categories=communities)
    X_c   = sm.add_constant(X, has_constant="add")

    result = None
    for method in ("bfgs", "lbfgs"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = MNLogit(y_cat.codes, X_c).fit(
                    method=method, maxiter=3000, disp=False, gtol=1e-5)
            if not np.isnan(res.llf):
                result = res
                log.info("[%s] converged (%s).", name, method)
                break
        except Exception as e:
            log.warning("[%s] %s failed: %s", name, method, e)

    if result is None:
        log.error("[%s] all optimisers failed.", name)
        return None

    try:
        se_flat = clustered_se(result, df["country"].values)
    except Exception as e:
        log.warning("[%s] clustered SE failed (%s); using default SE.", name, e)
        se_flat = np.asarray(result.bse)

    n_feat   = X_c.shape[1]
    non_ref  = [c for c in communities if c != REF_COMM]
    coef_rows = []
    for j, comm in enumerate(non_ref):
        for k, fname in enumerate(X_c.columns):
            idx  = j * n_feat + k
            coef = float(result.params.iloc[k, j])
            se   = float(se_flat[idx]) if idx < len(se_flat) else np.nan
            z    = coef / se if se and se > 0 else np.nan
            pval = 2 * stats.norm.sf(abs(z)) if not np.isnan(z) else np.nan
            coef_rows.append({
                "model": name, "community": comm, "term": fname,
                "coef": round(coef, 6), "se": round(se, 6),
                "z": round(z, 4) if not np.isnan(z) else np.nan,
                "p_value": round(pval, 6) if not np.isnan(pval) else np.nan,
                "ci_lower": round(coef - 1.96 * se, 6) if not np.isnan(se) else np.nan,
                "ci_upper": round(coef + 1.96 * se, 6) if not np.isnan(se) else np.nan,
            })

    k_params = int(result.params.size)
    return {
        "result":       result,
        "coef_table":   pd.DataFrame(coef_rows),
        "loglik":       float(result.llf),
        "llnull":       float(result.llnull),
        "k_params":     k_params,
        "aic":          -2 * result.llf + 2 * k_params,
        "bic":          -2 * result.llf + k_params * np.log(len(df)),
        "pseudo_r2":    1 - result.llf / result.llnull,
        "communities":  communities,
        "feature_names": list(X_c.columns),
        "X_const":      X_c,
    }


# ---------------------------------------------------------------------------
# Cross-validated log-loss
# ---------------------------------------------------------------------------
def cv_logloss(X, y, df, n_splits=CV_FOLDS, seed=RNG_SEED):
    """Mean per-observation log-loss over stratified k-fold CV."""
    communities = sorted(y.unique())
    ref_comm    = REF_COMM
    comm_order  = [ref_comm] + [c for c in communities if c != ref_comm]
    y_cat       = pd.Categorical(y, categories=comm_order)
    y_codes     = np.array(y_cat.codes)

    X_c  = sm.add_constant(X, has_constant="add")
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    losses = []

    for fold, (tr, te) in enumerate(skf.split(X_c, y_codes)):
        X_tr, y_tr = X_c.iloc[tr], y_codes[tr]
        X_te, y_te = X_c.iloc[te], y_codes[te]
        res = None
        for method in ("bfgs", "lbfgs"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = MNLogit(y_tr, X_tr).fit(
                        method=method, maxiter=2000, disp=False, gtol=1e-4)
                if not np.isnan(r.llf):
                    res = r
                    break
            except Exception:
                continue
        if res is None:
            log.warning("CV fold %d failed — skipping.", fold)
            continue

        params = np.asarray(res.params)      # (n_feat, J-1)
        logits = np.hstack([np.zeros((len(X_te), 1)),
                            X_te.values @ params])
        logits -= logits.max(axis=1, keepdims=True)
        probs   = np.exp(logits)
        probs  /= probs.sum(axis=1, keepdims=True)
        probs   = np.clip(probs, 1e-10, 1.0)
        ll_fold = -np.mean(np.log(probs[np.arange(len(y_te)), y_te]))
        losses.append(ll_fold)

    return float(np.mean(losses)) if losses else np.nan


# ---------------------------------------------------------------------------
# LRT
# ---------------------------------------------------------------------------
def lrt(r_null, r_alt, label):
    if r_null is None or r_alt is None:
        return {"comparison": label, "lrt_stat": np.nan, "df": np.nan,
                "p_value": np.nan}
    stat    = 2 * (r_alt["loglik"] - r_null["loglik"])
    df_diff = max(r_alt["k_params"] - r_null["k_params"], 1)
    pval    = stats.chi2.sf(stat, df=df_diff)
    log.info("LRT %s: χ²=%.1f df=%d p=%.3g", label, stat, df_diff, pval)
    return {"comparison": label, "lrt_stat": round(stat, 2),
            "df": df_diff, "p_value": round(pval, 6)}


# ---------------------------------------------------------------------------
# Q2 helpers: predicted probabilities
# ---------------------------------------------------------------------------
def pred_probs(X_arr, params):
    """(N, J) predicted probability matrix."""
    logits = np.hstack([np.zeros((len(X_arr), 1)), X_arr @ params])
    logits -= logits.max(axis=1, keepdims=True)
    exp_l   = np.exp(logits)
    return exp_l / exp_l.sum(axis=1, keepdims=True)


def compute_ame(res, df, focal_vars, n_boot=N_BOOT, seed=RNG_SEED):
    """AME for continuous predictors (+1 SD), bootstrap CI clustered by country."""
    if res is None:
        return pd.DataFrame()
    params     = res["result"].params.values
    X_arr      = res["X_const"].values.astype(float)
    col_map    = {n: i for i, n in enumerate(res["feature_names"])}
    communities= res["communities"]
    rng        = np.random.default_rng(seed)
    countries  = df["country"].values
    unique_c   = np.unique(countries)

    rows = []
    for var in focal_vars:
        if var not in col_map:
            continue
        ci        = col_map[var]
        X_hi      = X_arr.copy(); X_hi[:, ci] += 1.0
        dP        = pred_probs(X_hi, params) - pred_probs(X_arr, params)
        ame_pt    = dP.mean(axis=0)

        boot = np.zeros((n_boot, len(communities)))
        for b in range(n_boot):
            samp = rng.choice(unique_c, size=len(unique_c), replace=True)
            idx  = np.concatenate([np.where(countries == c)[0] for c in samp])
            boot[b] = (pred_probs(X_hi[idx], params) -
                       pred_probs(X_arr[idx], params)).mean(axis=0)
        ci_lo = np.percentile(boot, 2.5,  axis=0)
        ci_hi = np.percentile(boot, 97.5, axis=0)

        for j, comm in enumerate(communities):
            rows.append({"focal_var": var, "community": comm,
                         "ame": round(float(ame_pt[j]), 6),
                         "ci_lo_95": round(float(ci_lo[j]), 6),
                         "ci_hi_95": round(float(ci_hi[j]), 6)})
    return pd.DataFrame(rows)


def compute_aap(res, df, group_var, dummy_prefix, ref_val, all_vals):
    """Average Adjusted Predictions via marginal standardisation."""
    if res is None:
        return pd.DataFrame()
    params     = res["result"].params.values
    X_const    = res["X_const"].copy()
    col_names  = res["feature_names"]
    communities= res["communities"]
    dummy_cols = [c for c in col_names if c.startswith(dummy_prefix + "_")]

    rows = []
    target_vals = [ref_val] + [v for v in all_vals if v != ref_val]

    for val in target_vals:
        X_mod = X_const.values.astype(float).copy()
        for c in dummy_cols:
            X_mod[:, col_names.index(c)] = 0.0
        target_col = f"{dummy_prefix}_{val}"
        if target_col in col_names:
            X_mod[:, col_names.index(target_col)] = 1.0
        probs = pred_probs(X_mod, params).mean(axis=0)
        n_val = (df[group_var] == val).sum()
        for j, comm in enumerate(communities):
            rows.append({"group_var": group_var, "group_value": val,
                         "community": comm,
                         "aap": round(float(probs[j]), 6),
                         "n_cities": int(n_val)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    y  = df["community"]

    # ---- Design matrices ----
    Xs = {
        "M0":    build_X(df),
        "M1":    build_X(df, climate=True),
        "M2":    build_X(df, climate=True, gdp=True),
        "M2b":   build_X(df, climate=True, gdp=True, pop=True),
        "M3":    build_X(df, climate=True, gdp=True, pop=True, colonial=True),
        "M4":    build_X(df, climate=True, gdp=True, pop=True, colonial=True,
                         spatial=True),
        "M3_nc": build_X(df, gdp=True, pop=True, colonial=True),
        "M3_ng": build_X(df, climate=True, pop=True, colonial=True),
        "M3_np": build_X(df, climate=True, gdp=True, colonial=True),
        "M3_nl": build_X(df, climate=True, gdp=True, pop=True),
    }

    # ---- Fit ----
    results = {}
    coef_tables = []
    fit_rows    = []

    for name in ["M0", "M1", "M2", "M2b", "M3", "M4",
                 "M3_nc", "M3_ng", "M3_np", "M3_nl"]:
        log.info("Fitting %s ...", name)
        res = fit_mnl(Xs[name], y, df, name)
        results[name] = res
        if res:
            coef_tables.append(res["coef_table"])
            fit_rows.append({
                "model":    name,
                "n_cities": len(df),
                "n_params": res["k_params"],
                "log_lik":  round(res["loglik"], 3),
                "aic":      round(res["aic"], 3),
                "bic":      round(res["bic"], 3),
                "pseudo_r2": round(res["pseudo_r2"], 4),
            })

    # ---- CV log-loss for main models ----
    log.info("Computing CV log-loss (5-fold) ...")
    cv_models = ["M0", "M1", "M2", "M2b", "M3", "M4"]
    cv_loss   = {}
    for name in cv_models:
        log.info("  CV %s ...", name)
        cv_loss[name] = cv_logloss(Xs[name], y, df)
        log.info("  %s CV log-loss = %.4f", name, cv_loss[name])

    for row in fit_rows:
        row["cv_logloss"] = cv_loss.get(row["model"], np.nan)

    # ---- Sequential LRT ----
    lrt_rows = []
    for null, alt in [("M0","M1"),("M1","M2"),("M2","M2b"),
                      ("M2b","M3"),("M3","M4")]:
        lrt_rows.append(lrt(results[null], results[alt], f"{null}→{alt}"))

    # ---- Drop-one LRT (vs M3) ----
    drop_map = {
        "M3_nc": "climate",
        "M3_ng": "gdp",
        "M3_np": "pop",
        "M3_nl": "colonial",
    }
    for key, label in drop_map.items():
        r = lrt(results[key], results["M3"], f"M3_drop({label})")
        lrt_rows.append(r)

    # ---- Q1: factor contribution summary ----
    m3_r2   = results["M3"]["pseudo_r2"]  if results["M3"] else np.nan
    m0_r2   = results["M0"]["pseudo_r2"]  if results["M0"] else np.nan  # = 0
    m3_cv   = cv_loss.get("M3", np.nan)
    m0_cv   = cv_loss.get("M0", np.nan)

    factor_rows = []
    for key, label in drop_map.items():
        res_drop = results[key]
        if res_drop and results["M3"]:
            dr2  = results["M3"]["pseudo_r2"] - res_drop["pseudo_r2"]
            dcv  = cv_logloss(Xs[key], y, df) - m3_cv   # positive = M3 better
            chi2 = 2 * (results["M3"]["loglik"] - res_drop["loglik"])
            df_  = results["M3"]["k_params"] - res_drop["k_params"]
            pv   = stats.chi2.sf(chi2, df_)
        else:
            dr2 = dcv = chi2 = df_ = pv = np.nan
        factor_rows.append({
            "factor_group":    label,
            "delta_pseudo_r2": round(dr2, 4) if not np.isnan(dr2) else np.nan,
            "delta_cv_logloss":round(dcv, 4) if not np.isnan(dcv) else np.nan,
            "lrt_chi2":        round(chi2, 2) if not np.isnan(chi2) else np.nan,
            "lrt_df":          int(df_) if not np.isnan(df_) else np.nan,
            "lrt_p":           round(pv, 6) if not np.isnan(pv) else np.nan,
        })

    # ---- Q2: AAP and AME at M3 ----
    clim_vals   = sorted(df["climate_zone"].unique())
    col_vals    = sorted(df["colonial"].unique())

    aap_clim = compute_aap(results["M3"], df,
                           "climate_zone", "clim", REF_CLIMATE, clim_vals)
    aap_col  = compute_aap(results["M3"], df,
                           "colonial",     "col",  REF_COLONIAL, col_vals)
    ame      = compute_ame(results["M3"], df,
                           ["log_gdp_std", "log_pop_std"])

    # ---- Save ----
    pd.DataFrame(fit_rows).to_csv(OUT_DIR / "mnl_fit.csv", index=False)
    pd.DataFrame(lrt_rows).to_csv(OUT_DIR / "mnl_lrt.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(OUT_DIR / "mnl_factor_contribution.csv",
                                     index=False)
    if coef_tables:
        pd.concat(coef_tables).to_csv(OUT_DIR / "mnl_coefs_M3.csv",
                                       index=False)
    if not aap_clim.empty:
        aap_clim.to_csv(OUT_DIR / "mnl_aap_climate_M3.csv", index=False)
    if not aap_col.empty:
        aap_col.to_csv(OUT_DIR / "mnl_aap_colonial_M3.csv", index=False)
    if not ame.empty:
        ame.to_csv(OUT_DIR / "mnl_ame_M3.csv", index=False)

    # ---- Console summary ----
    print("\n=== Model fit ===")
    print(pd.DataFrame(fit_rows).to_string(index=False))
    print("\n=== Factor contributions (Q1) ===")
    print(pd.DataFrame(factor_rows).to_string(index=False))


if __name__ == "__main__":
    main()
