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

Population definition
----------------------
run_analysis(pop_col, output_suffix) runs this entire analysis for one
population column. main() calls it twice: once for the primary analysis
(WorldPop 2022, output_suffix="") and once as a sensitivity check
(Africapolis 2020, output_suffix="_sensitivity_africapolis2020"), so the
M3 population coefficient, AME, and drop-one contribution can be compared
directly across population definitions.

Outputs
-------
outputs/tables/mnl_fit{suffix}.csv
outputs/tables/mnl_lrt{suffix}.csv
outputs/tables/mnl_factor_contribution{suffix}.csv   ← Q1 summary
outputs/tables/mnl_coefs_M3{suffix}.csv
outputs/tables/mnl_aap_climate_M3{suffix}.csv        ← Q2
outputs/tables/mnl_aap_colonial_M3{suffix}.csv       ← Q2
outputs/tables/mnl_ame_M3{suffix}.csv                ← Q2
  where {suffix} is "" for the primary (WorldPop 2022) run and
  "_sensitivity_africapolis2020" for the sensitivity run.

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
from sklearn.model_selection import StratifiedGroupKFold

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
def load_data(pop_col="log_pop_worldpop_2022"):
    """Population is WorldPop 2022 by default (log_pop_worldpop_2022,
    aggregated by 00_prepare_data.py from the area-weighted WorldPop 2022
    grid within each Africapolis urban agglomeration) — see that script's
    docstring for the population-definition policy.

    Args:
        pop_col: which column in analysis_input.csv to use as the model's
            population covariate. "log_pop_worldpop_2022" (default,
            primary analysis) or "log_pop_africapolis_2020" (sensitivity
            analysis — see run_analysis / main). Internally always
            re-exposed to the rest of this module as the generic "log_pop"
            / "log_pop_std" columns, so build_X / cv_logloss / compute_ame
            don't need to know which definition is in use.
    """
    path = OUT_DIR / "analysis_input.csv"
    df = pd.read_csv(path).dropna(
        subset=["climate_zone", "colonial", "log_gdp", pop_col]
    ).reset_index(drop=True)
    df["log_gdp_std"] = (df["log_gdp"] - df["log_gdp"].mean()) / df["log_gdp"].std()
    # "log_pop" is the generic name build_X / cv_logloss standardise from —
    # it holds whichever population definition pop_col names, so the rest
    # of this module never needs to know if that's WorldPop2022 or
    # Africapolis2020.
    df["log_pop"] = df[pop_col]
    df["log_pop_std"] = (df["log_pop"] - df["log_pop"].mean()) / df["log_pop"].std()
    log.info("Loaded %d cities, %d countries, %d communities (population column: %s).",
             len(df), df["country"].nunique(), df["community"].nunique(), pop_col)
    return df


def make_dummies(series, prefix, ref):
    d = pd.get_dummies(series, prefix=prefix).astype(int)
    ref_col = f"{prefix}_{ref}"
    return d.drop(columns=[ref_col], errors="ignore")


def build_X(df, climate=False, gdp=False, pop=False, colonial=False,
            spatial=False, std_stats=None):
    """Build the design matrix for one model spec.

    Args:
        std_stats: optional {"log_gdp": (mean, std), "log_pop":
            (mean, std)} to standardise gdp/pop with externally-supplied
            statistics rather than this df's own mean/std — used by
            cv_logloss so each CV fold standardises using only its
            training-fold statistics, not statistics that have already
            seen the held-out fold.
    """
    std_stats = std_stats or {}
    parts = []
    if climate:
        parts.append(make_dummies(df["climate_zone"], "clim", REF_CLIMATE))
    if gdp:
        if "log_gdp" in std_stats:
            m, s = std_stats["log_gdp"]
            parts.append(((df["log_gdp"] - m) / s).rename("log_gdp_std"))
        else:
            parts.append(df["log_gdp_std"])
    if pop:
        if "log_pop" in std_stats:
            m, s = std_stats["log_pop"]
            parts.append(((df["log_pop"] - m) / s).rename("log_pop_std"))
        else:
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
def cv_logloss(df, y, X_spec, n_splits=CV_FOLDS, seed=RNG_SEED):
    """Mean per-observation log-loss over country-grouped stratified k-fold CV.

    Uses StratifiedGroupKFold with groups=df["country"] rather than plain
    StratifiedKFold: GDP and colonial legacy are country-level variables,
    so a random per-city split puts cities from the same country — sharing
    the same GDP/colonial values — in both the training and test fold,
    which overstates how well the model generalises to unseen countries.

    Standardisation of log_gdp / log_pop is refit on each
    training fold and applied to that fold's test set (via build_X's
    std_stats), rather than using statistics computed on the full sample
    before splitting, which would leak test-fold information into the
    standardisation the model is fit on.
    """
    communities = sorted(y.unique())
    ref_comm    = REF_COMM
    comm_order  = [ref_comm] + [c for c in communities if c != ref_comm]
    y_cat       = pd.Categorical(y, categories=comm_order)
    y_codes     = np.array(y_cat.codes)
    groups      = df["country"].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    losses = []

    for fold, (tr, te) in enumerate(sgkf.split(df, y_codes, groups=groups)):
        df_tr, y_tr = df.iloc[tr].reset_index(drop=True), y_codes[tr]
        df_te       = df.iloc[te].reset_index(drop=True)
        y_te        = y_codes[te]

        n_countries_tr = df_tr["country"].nunique()
        n_countries_te = df_te["country"].nunique()
        te_communities = set(np.unique(y_te))
        tr_communities = set(np.unique(y_tr))
        if not te_communities <= tr_communities:
            log.warning(
                "CV fold %d: test fold contains community codes absent "
                "from the training fold (%s) — skipping.",
                fold, te_communities - tr_communities,
            )
            continue
        log.info(
            "  CV fold %d: %d train cities / %d countries, "
            "%d test cities / %d countries, %d/%d outcome communities in test.",
            fold, len(df_tr), n_countries_tr, len(df_te), n_countries_te,
            len(te_communities), len(tr_communities),
        )

        std_stats = {}
        if X_spec.get("gdp"):
            std_stats["log_gdp"] = (df_tr["log_gdp"].mean(), df_tr["log_gdp"].std())
        if X_spec.get("pop"):
            std_stats["log_pop"] = (
                df_tr["log_pop"].mean(),
                df_tr["log_pop"].std(),
            )

        X_tr = sm.add_constant(build_X(df_tr, std_stats=std_stats, **X_spec), has_constant="add")
        X_te = sm.add_constant(build_X(df_te, std_stats=std_stats, **X_spec), has_constant="add")
        X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)

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
            log.warning("CV fold %d failed to fit — skipping.", fold)
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


def compute_ame(res, df, focal_vars, X_spec, n_boot=N_BOOT, seed=RNG_SEED):
    """AME for continuous predictors (+1 SD), bootstrap CI clustered by
    country.

    Each bootstrap replicate resamples countries with replacement and
    REFITS MNLogit on that resampled design (not just the point-estimate
    coefficients from the full-sample fit), so the resulting CI reflects
    both which cities/countries were resampled and the coefficient
    estimation uncertainty that resampling induces. A CI built by holding
    params fixed and only resampling the predicted-probability inputs (as
    an earlier version of this function did) only reflects the first of
    those two sources.

    Args:
        X_spec: build_X(df, ...) kwargs used for the original fit, so each
            replicate builds its design matrix the same way.
    """
    if res is None:
        return pd.DataFrame()
    params     = res["result"].params.values
    X_arr      = res["X_const"].values.astype(float)
    col_map    = {n: i for i, n in enumerate(res["feature_names"])}
    communities= res["communities"]
    rng        = np.random.default_rng(seed)
    countries  = df["country"].values
    unique_c   = np.unique(countries)

    ame_pt = {}
    for var in focal_vars:
        if var not in col_map:
            continue
        ci     = col_map[var]
        X_hi   = X_arr.copy(); X_hi[:, ci] += 1.0
        dP     = pred_probs(X_hi, params) - pred_probs(X_arr, params)
        ame_pt[var] = dP.mean(axis=0)

    boot = {var: [] for var in ame_pt}
    n_failed = 0
    for b in range(n_boot):
        samp = rng.choice(unique_c, size=len(unique_c), replace=True)
        idx  = np.concatenate([np.where(countries == c)[0] for c in samp])
        df_b = df.iloc[idx].reset_index(drop=True)
        X_b  = build_X(df_b, **X_spec)
        y_b  = df_b["community"]

        res_b = fit_mnl(X_b, y_b, df_b, name=f"AME_boot_{b}")
        if res_b is None or res_b["communities"] != communities:
            n_failed += 1
            continue
        params_b  = res_b["result"].params.values
        X_b_arr   = res_b["X_const"].values.astype(float)
        col_map_b = {n: i for i, n in enumerate(res_b["feature_names"])}

        for var in ame_pt:
            if var not in col_map_b:
                continue
            ci_b   = col_map_b[var]
            X_b_hi = X_b_arr.copy(); X_b_hi[:, ci_b] += 1.0
            dP_b   = pred_probs(X_b_hi, params_b) - pred_probs(X_b_arr, params_b)
            boot[var].append(dP_b.mean(axis=0))

    if n_failed:
        log.warning("compute_ame: %d/%d bootstrap replicates failed to fit "
                    "or produced a different community set and were dropped.",
                    n_failed, n_boot)

    rows = []
    for var, pt in ame_pt.items():
        boot_arr = np.array(boot[var]) if boot[var] else np.full((1, len(communities)), np.nan)
        ci_lo = np.percentile(boot_arr, 2.5,  axis=0)
        ci_hi = np.percentile(boot_arr, 97.5, axis=0)
        for j, comm in enumerate(communities):
            rows.append({"focal_var": var, "community": comm,
                         "ame": round(float(pt[j]), 6),
                         "ci_lo_95": round(float(ci_lo[j]), 6),
                         "ci_hi_95": round(float(ci_hi[j]), 6),
                         "n_boot_valid": len(boot[var])})
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
# Main analysis (parameterised over the population definition, so the
# Africapolis-2020 sensitivity run below is the SAME analysis re-run on a
# different population column, not a separate comparison of raw population
# values only)
# ---------------------------------------------------------------------------
def run_analysis(pop_col="log_pop_worldpop_2022", output_suffix=""):
    """Run the full M0-M4 / drop-one / AAP / AME analysis using pop_col as
    the population covariate, writing outputs/tables/mnl_*{output_suffix}.csv.

    Returns:
        {"fit": fit_rows, "factor_contribution": factor_rows} — the two
        tables most useful for comparing population definitions without
        re-reading the CSVs.
    """
    df = load_data(pop_col=pop_col)
    y  = df["community"]

    # ---- Design matrices ----
    X_specs = {
        "M0":    dict(),
        "M1":    dict(climate=True),
        "M2":    dict(climate=True, gdp=True),
        "M2b":   dict(climate=True, gdp=True, pop=True),
        "M3":    dict(climate=True, gdp=True, pop=True, colonial=True),
        "M4":    dict(climate=True, gdp=True, pop=True, colonial=True, spatial=True),
        "M3_nc": dict(gdp=True, pop=True, colonial=True),
        "M3_ng": dict(climate=True, pop=True, colonial=True),
        "M3_np": dict(climate=True, gdp=True, colonial=True),
        "M3_nl": dict(climate=True, gdp=True, pop=True),
    }
    Xs = {name: build_X(df, **spec) for name, spec in X_specs.items()}

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
        cv_loss[name] = cv_logloss(df, y, X_specs[name])
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
            dcv  = cv_logloss(df, y, X_specs[key]) - m3_cv   # positive = M3 better
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
                           ["log_gdp_std", "log_pop_std"],
                           X_spec=dict(climate=True, gdp=True, pop=True, colonial=True))

    # ---- Save ----
    suffix = output_suffix
    pd.DataFrame(fit_rows).to_csv(OUT_DIR / f"mnl_fit{suffix}.csv", index=False)
    pd.DataFrame(lrt_rows).to_csv(OUT_DIR / f"mnl_lrt{suffix}.csv", index=False)
    pd.DataFrame(factor_rows).to_csv(OUT_DIR / f"mnl_factor_contribution{suffix}.csv",
                                     index=False)
    if coef_tables:
        pd.concat(coef_tables).to_csv(OUT_DIR / f"mnl_coefs_M3{suffix}.csv",
                                       index=False)
    if not aap_clim.empty:
        aap_clim.to_csv(OUT_DIR / f"mnl_aap_climate_M3{suffix}.csv", index=False)
    if not aap_col.empty:
        aap_col.to_csv(OUT_DIR / f"mnl_aap_colonial_M3{suffix}.csv", index=False)
    if not ame.empty:
        ame.to_csv(OUT_DIR / f"mnl_ame_M3{suffix}.csv", index=False)

    # ---- Console summary ----
    print(f"\n=== Model fit (population={pop_col}) ===")
    print(pd.DataFrame(fit_rows).to_string(index=False))
    print(f"\n=== Factor contributions (Q1, population={pop_col}) ===")
    print(pd.DataFrame(factor_rows).to_string(index=False))

    return {"fit": pd.DataFrame(fit_rows), "factor_contribution": pd.DataFrame(factor_rows)}


def main():
    """Primary analysis: WorldPop 2022. Then re-runs the identical analysis
    with Africapolis 2020 as a sensitivity check (output_suffix
    "_sensitivity_africapolis2020"), so the M3 population coefficient,
    AME, drop-one LRT and delta-pseudo-R^2 can be compared directly across
    population definitions rather than only comparing the two raw
    population values (population_definition_comparison.csv, produced by
    00_prepare_data.py, does the latter)."""
    log.info("=== Primary analysis: population = WorldPop 2022 ===")
    primary = run_analysis(pop_col="log_pop_worldpop_2022", output_suffix="")

    log.info("=== Sensitivity analysis: population = Africapolis 2020 ===")
    sensitivity = run_analysis(pop_col="log_pop_africapolis_2020",
                               output_suffix="_sensitivity_africapolis2020")

    pop_row_primary     = primary["factor_contribution"].query("factor_group == 'pop'")
    pop_row_sensitivity = sensitivity["factor_contribution"].query("factor_group == 'pop'")
    log.info(
        "\n=== Population coefficient sensitivity: WorldPop2022 vs Africapolis2020 ===\n"
        "  WorldPop2022   delta_pseudo_r2=%.4f  lrt_p=%.3g\n"
        "  Africapolis2020 delta_pseudo_r2=%.4f  lrt_p=%.3g",
        pop_row_primary["delta_pseudo_r2"].iloc[0], pop_row_primary["lrt_p"].iloc[0],
        pop_row_sensitivity["delta_pseudo_r2"].iloc[0], pop_row_sensitivity["lrt_p"].iloc[0],
    )


if __name__ == "__main__":
    main()
