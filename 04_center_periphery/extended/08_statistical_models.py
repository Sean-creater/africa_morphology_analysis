"""
08_statistical_models.py
========================
Q4 — Formal statistical inference.

Continuous log-population models, one per outcome per radial bin:

  Y_i(u) = α(u) + β(u) log(Pop_i) + CountryFE_i + ε_i

where Y ∈ {H17_rarefied, r17, JS_to_centroid, rpc}.

These models are the primary inference vehicle — the Tier comparison is
for visualisation and interpretation only.

Estimation
----------
  OLS with country fixed effects (explicit country dummies, within-
  country variation in log(Pop) identifies β).
  Standard errors: country-clustered (G-1 degrees of freedom, where G
  is the number of countries).
  Reported: β(u), 95% CI, p-value, partial R² of log(Pop) after
  partialling out country FE.

  partial R² = 1 - SS_res(full) / SS_res(country-only reduced model)
  This is the incremental R² of log(Pop) after accounting for all
  between-country variation.

  IMPORTANT: Scripts 02–05 use pooled OLS (no FE) for descriptive
  within-bin summaries.  Script 08 is the confirmatory model with
  country FE and must NOT be mixed with descriptive β estimates.

Additional model
----------------
  Full spatial profile model (GAM-style interaction via bin dummies):
  Y_i × bin_b ~ log(Pop_i) + bin_b + log(Pop_i) × bin_b + CountryFE

Main findings structure (three headlines)
-----------------------------------------
  Finding 1 — Spatial division of labour (Analysis 1):
      Diversity scaling (β_H) peaks in mid-zone; assortativity scaling
      (β_r) peaks in core. Both contrasts are significant by bootstrap.
      Ratios R_H = β_H_mid/β_H_core and R_r = β_r_core/β_r_mid quantify
      the fold-difference for abstract-ready wording.

  Finding 2 — Core convergence (Analysis 2):
      Cross-city compositional convergence C(u) = |β_JS(u)| is ~2×
      stronger in the core than in mid or outer zones.

Outputs
-------
  formal_models.csv
      outcome | radial_bin | n_obs | n_countries | beta | beta_se
              | ci_lo | ci_hi | p_value | partial_r2

  profile_interaction_model.csv
      outcome | radial_bin | interaction_coef | se | p_value
      (from bin × log(Pop) interaction in pooled model with country FE)

  spatial_division_test.csv  [Finding 1]
      Analysis 1 — Formal test of spatial division of labour (H17 vs r17):
        Delta_H = beta_H(mid) - beta_H(core)   [peak diversity zone > core]
        Delta_r = beta_r(core) - beta_r(mid)   [core assortativity > mid]
        R_H     = beta_H_mid / beta_H_core      [fold ratio, diversity]
        R_r     = beta_r_core / beta_r_mid      [fold ratio, assortativity]
      Bootstrap CI (B=2000, city-level resampling within country blocks).
      Columns: statistic | estimate | ci_lo_95 | ci_hi_95 | p_one_sided

  convergence_decay.csv  [Finding 2]
      Analysis 2 — Radial decay of core convergence C(u) = |beta_D(u)|:
        ratio_mid_core, ratio_outer_core with bootstrap CI.
      Columns: zone_pair | ratio | ci_lo | ci_hi

  morphology_allocation_coupling.csv  [SUPPLEMENTARY ONLY]
      Analysis 3 — Coupling of beta_H(u) and RPC(u):
        Spearman/Pearson corr across bins; mid-zone RPC vs core RPC contrast.
        NOTE: mechanically confounded — RPC and beta_H both derive from
        population-defined spatial units; not used in main findings.
      Columns: metric | value | p_value | note
"""

import logging
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sp_stats
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).parent
OUT_TABLES = ROOT / "outputs" / "tables"

MIN_OBS   = 15   # min cities per bin for per-bin model
MIN_CTRY  = 5    # min countries for clustered SE to be reliable


def _load_outcome(name: str) -> pd.DataFrame | None:
    """Load and standardise an outcome file."""
    paths = {
        "H17":  OUT_TABLES / "city_bin_diversity.parquet",
        "r17":  OUT_TABLES / "city_bin_assortativity.parquet",
        "JS":   OUT_TABLES / "city_bin_divergence.parquet",
        "rpc":  OUT_TABLES / "city_bin_rpc.parquet",
    }
    col_map = {
        "H17": "H17_rarefied",
        "r17": "r17",
        "JS":  "JS_to_centroid",
        "rpc": "rpc",
    }
    qc_map = {
        "r17": "qc30",
    }
    pq = paths.get(name)
    if pq is None or not pq.exists():
        log.warning(f"  {name}: file not found ({pq})")
        return None
    df = pd.read_parquet(pq)
    ycol = col_map[name]
    if ycol not in df.columns:
        log.warning(f"  {name}: column {ycol} not found")
        return None
    df = df.rename(columns={ycol: "Y"})
    # apply QC filter if applicable
    qc = qc_map.get(name)
    if qc and qc in df.columns:
        df = df[df[qc] == 1]
    return df[["city", "country", "tier", "radial_bin", "Y"]].dropna(subset=["Y"])


def _load_log_pop() -> dict:
    tiers = pd.read_parquet(OUT_TABLES / "city_tiers.parquet",
                            columns=["city", "country", "log_pop"])
    return tiers.set_index(["city", "country"])["log_pop"].to_dict()


def _fit_per_bin(df: pd.DataFrame, pop_map: dict,
                 outcome_name: str, n_bins: int) -> pd.DataFrame:
    """
    Per-bin OLS: Y ~ log(Pop) + C(country), country-clustered SE.

    Country fixed effects absorb all between-country heterogeneity so
    that β is identified from within-country variation in log(Pop).

    partial R² = incremental R² of log(Pop) over the country-only model:
        1 - SSE(full) / SSE(reduced)   where reduced = Y ~ C(country) only.
    Equivalently: the fraction of within-country residual variance explained
    by log(Pop) after partialling out country FE.

    Returns DataFrame with one row per radial_bin.
    """
    df = df.copy()
    df["log_pop"] = df.apply(
        lambda r: pop_map.get((r["city"], r["country"]), np.nan), axis=1)
    df = df.dropna(subset=["log_pop"])

    rows = []
    for rbin in range(n_bins):
        sbin = df[df["radial_bin"] == rbin].copy()
        n_ctry = sbin["country"].nunique()
        if len(sbin) < MIN_OBS or n_ctry < MIN_CTRY:
            continue
        # Need at least 2 cities per country on average to have within-country
        # variation; skip bins where FE saturate the model
        if n_ctry >= len(sbin):
            continue

        try:
            # Full model: Y ~ log_pop + C(country)
            formula_full = "Y ~ log_pop + C(country)"
            mod_full = smf.ols(formula_full, data=sbin).fit(
                cov_type="cluster", cov_kwds={"groups": sbin["country"]})

            beta  = float(mod_full.params["log_pop"])
            se    = float(mod_full.bse["log_pop"])
            t     = beta / se if se > 0 else np.nan
            df_t  = n_ctry - 1
            pval  = float(2 * sp_stats.t.sf(abs(t), df=df_t))
            t_crit = float(sp_stats.t.ppf(0.975, df=df_t))
            ci_lo = beta - t_crit * se
            ci_hi = beta + t_crit * se

            # Reduced model: Y ~ C(country) only (for partial R²)
            formula_red = "Y ~ C(country)"
            mod_red = smf.ols(formula_red, data=sbin).fit()
            ss_res_full = mod_full.ssr
            ss_res_red  = mod_red.ssr
            pr2 = 1 - ss_res_full / ss_res_red if ss_res_red > 0 else np.nan

            rows.append({
                "outcome":     outcome_name,
                "radial_bin":  rbin,
                "n_obs":       len(sbin),
                "n_countries": n_ctry,
                "beta":        beta,
                "beta_se":     se,
                "ci_lo":       ci_lo,
                "ci_hi":       ci_hi,
                "p_value":     pval,
                "partial_r2":  float(pr2),
            })
        except Exception as e:
            log.debug(f"  bin {rbin} ({outcome_name}): {e}")
            continue
    return pd.DataFrame(rows)


def _fit_interaction_model(df: pd.DataFrame, pop_map: dict,
                           outcome_name: str) -> pd.DataFrame:
    """
    Pooled model Y ~ log(Pop) * bin + C(country) with country-clustered SE.
    The bin × log(Pop) interaction coefficients show how β(u) varies across
    radial positions in a single unified model. Reference bin is bin 0.
    Returns interaction coefficients per bin.
    """
    df = df.copy()
    df["log_pop"] = df.apply(
        lambda r: pop_map.get((r["city"], r["country"]), np.nan), axis=1)
    df = df.dropna(subset=["log_pop"])
    df["radial_bin"] = df["radial_bin"].astype("category")

    try:
        # Country FE via explicit dummies; clustered SE handles inference
        formula = "Y ~ log_pop * C(radial_bin) + C(country)"
        mod = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["country"]})
        rows = []
        for coef_name, coef_val in mod.params.items():
            if "log_pop:C(radial_bin)" in coef_name:
                rbin = int(coef_name.split("[T.")[-1].rstrip("]"))
                se   = float(mod.bse[coef_name])
                t    = coef_val / se if se > 0 else np.nan
                pval = float(2 * sp_stats.t.sf(abs(t), df=df["country"].nunique()-1))
                rows.append({
                    "outcome":          outcome_name,
                    "radial_bin":       rbin,
                    "interaction_coef": float(coef_val),
                    "se":               se,
                    "p_value":          pval,
                })
        return pd.DataFrame(rows)
    except Exception as e:
        log.warning(f"  Interaction model failed for {outcome_name}: {e}")
        return pd.DataFrame()


# ── Zone definitions (bins 0–9) ───────────────────────────────────────────
# Core  : bins 0–1   (innermost two bins, u < 2×7.29 = 14.58)
# Middle: bins 2–5   (mid-city zone where H17 beta peaks)
# Outer : bins 6–9   (distal periphery)
ZONE_CORE   = [0, 1]
ZONE_MID    = [2, 3, 4, 5]
ZONE_OUTER  = [6, 7, 8, 9]


def _zone_mean_beta(fm: pd.DataFrame, outcome: str, bins: list) -> float:
    """Mean β across specified bins for a given outcome."""
    sub = fm[(fm["outcome"] == outcome) & (fm["radial_bin"].isin(bins))]
    return float(sub["beta"].mean()) if len(sub) > 0 else np.nan


def _bootstrap_zone_delta(df_H: pd.DataFrame, df_r: pd.DataFrame,
                           pop_map: dict, n_bins: int,
                           B: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Analysis 1: Bootstrap test of spatial division of labour.

    For each bootstrap replicate:
      - Resample cities within country blocks (cluster bootstrap)
      - Re-fit per-bin OLS with country FE for H17 and r17
      - Compute zone-mean betas and deltas

    Delta_H = mean(beta_H, mid-bins) - mean(beta_H, core-bins)
    Delta_r = mean(beta_r, core-bins) - mean(beta_r, mid-bins)

    Null hypothesis: Delta <= 0 (no spatial division)
    p-value: fraction of bootstrap replicates where Delta <= 0.
    """
    import statsmodels.formula.api as _smf

    rng = np.random.default_rng(seed)
    countries = df_H["country"].unique()

    def _bin_beta(df_sub, outcome_label):
        """Fit per-bin OLS Y ~ log_pop + C(country), return zone means."""
        betas = {}
        for rbin in range(n_bins):
            sbin = df_sub[df_sub["radial_bin"] == rbin].copy()
            n_c  = sbin["country"].nunique()
            if len(sbin) < MIN_OBS or n_c < MIN_CTRY or n_c >= len(sbin):
                betas[rbin] = np.nan
                continue
            try:
                mod = _smf.ols("Y ~ log_pop + C(country)", data=sbin).fit(
                    cov_type="cluster", cov_kwds={"groups": sbin["country"]})
                betas[rbin] = float(mod.params["log_pop"])
            except Exception:
                betas[rbin] = np.nan
        core_m = np.nanmean([betas.get(b, np.nan) for b in ZONE_CORE])
        mid_m  = np.nanmean([betas.get(b, np.nan) for b in ZONE_MID])
        return core_m, mid_m

    # Observed values
    def _prep(df, pop_map):
        df = df.copy()
        df["log_pop"] = df.apply(
            lambda r: pop_map.get((r["city"], r["country"]), np.nan), axis=1)
        return df.dropna(subset=["log_pop"])

    df_H = _prep(df_H, pop_map)
    df_r = _prep(df_r, pop_map)

    core_H_obs, mid_H_obs = _bin_beta(df_H, "H17")
    core_r_obs, mid_r_obs = _bin_beta(df_r, "r17")
    delta_H_obs = mid_H_obs  - core_H_obs   # should be > 0
    delta_r_obs = core_r_obs - mid_r_obs    # should be > 0
    R_H_obs = mid_H_obs / core_H_obs if core_H_obs > 0 else np.nan
    R_r_obs = core_r_obs / mid_r_obs if mid_r_obs  > 0 else np.nan

    # Single bootstrap loop — collect all four statistics simultaneously
    delta_H_boot = []
    delta_r_boot = []
    R_H_boot     = []
    R_r_boot     = []
    for _ in range(B):
        c_boot = rng.choice(countries, size=len(countries), replace=True)
        dH_b = pd.concat([df_H[df_H["country"] == c] for c in c_boot],
                         ignore_index=True)
        dr_b = pd.concat([df_r[df_r["country"] == c] for c in c_boot],
                         ignore_index=True)
        c_H, m_H = _bin_beta(dH_b, "H17")
        c_r, m_r = _bin_beta(dr_b, "r17")
        delta_H_boot.append(m_H - c_H)
        delta_r_boot.append(c_r - m_r)
        R_H_boot.append(m_H / c_H if (not np.isnan(c_H) and c_H > 0) else np.nan)
        R_r_boot.append(c_r / m_r if (not np.isnan(m_r) and m_r > 0) else np.nan)

    dH_arr  = np.array(delta_H_boot)
    dr_arr  = np.array(delta_r_boot)
    R_H_arr = np.array(R_H_boot, dtype=float)
    R_r_arr = np.array(R_r_boot, dtype=float)

    rows = []
    for stat, obs, arr, is_ratio in [
            ("Delta_H", delta_H_obs, dH_arr,  False),
            ("Delta_r", delta_r_obs, dr_arr,  False),
            ("R_H",     R_H_obs,     R_H_arr, True),
            ("R_r",     R_r_obs,     R_r_arr, True),
    ]:
        ci_lo = float(np.nanpercentile(arr, 2.5))
        ci_hi = float(np.nanpercentile(arr, 97.5))
        # one-sided p: for deltas, fraction <= 0; for ratios, fraction <= 1
        threshold = 1.0 if is_ratio else 0.0
        p_perm = float(np.nanmean(arr <= threshold))
        rows.append({
            "statistic":       stat,
            "estimate":        float(obs) if not np.isnan(obs) else np.nan,
            "ci_lo_95":        ci_lo,
            "ci_hi_95":        ci_hi,
            "p_one_sided":     p_perm,
            "core_bins":       str(ZONE_CORE),
            "mid_bins":        str(ZONE_MID),
            "n_boot":          B,
        })
    return pd.DataFrame(rows)


def _convergence_decay(fm: pd.DataFrame, B: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Analysis 2: Radial decay of core convergence C(u) = |beta_D(u)|.

    Uses the per-bin JS beta estimates from formal_models.csv.
    Computes zone-mean |beta_D| and ratios:
      ratio_core_mid   = C_core / C_mid
      ratio_core_outer = C_core / C_outer
    Bootstrap CI via percentile method on the zone-mean estimates,
    propagating uncertainty from per-bin beta SEs via parametric sampling.
    """
    js = fm[fm["outcome"] == "JS"][["radial_bin", "beta", "beta_se"]].copy()
    js["C"] = js["beta"].abs()

    rng = np.random.default_rng(seed)

    def _zone_C(js_sample, bins):
        return float(js_sample[js_sample["radial_bin"].isin(bins)]["C"].mean())

    # Observed
    C_core  = _zone_C(js, ZONE_CORE)
    C_mid   = _zone_C(js, ZONE_MID)
    C_outer = _zone_C(js, ZONE_OUTER)

    # Parametric bootstrap: sample beta ~ N(beta, se^2) per bin
    ratios_cm = []
    ratios_co = []
    for _ in range(B):
        js_s = js.copy()
        js_s["C"] = np.abs(
            rng.normal(js_s["beta"].values, js_s["beta_se"].values))
        rc  = _zone_C(js_s, ZONE_CORE)
        rm  = _zone_C(js_s, ZONE_MID)
        ro  = _zone_C(js_s, ZONE_OUTER)
        ratios_cm.append(rc / rm  if rm  > 0 else np.nan)
        ratios_co.append(rc / ro  if ro  > 0 else np.nan)

    rows = []
    for label, obs_ratio, boot_arr in [
            ("C_core / C_mid",   C_core / C_mid   if C_mid   > 0 else np.nan, ratios_cm),
            ("C_core / C_outer", C_core / C_outer if C_outer > 0 else np.nan, ratios_co),
    ]:
        arr = np.array(boot_arr)
        rows.append({
            "zone_pair":  label,
            "C_core":     round(C_core, 6),
            "C_mid":      round(C_mid, 6),
            "C_outer":    round(C_outer, 6),
            "ratio":      float(obs_ratio) if not np.isnan(obs_ratio) else np.nan,
            "ci_lo_95":   float(np.nanpercentile(arr, 2.5)),
            "ci_hi_95":   float(np.nanpercentile(arr, 97.5)),
            "n_boot":     B,
        })
    return pd.DataFrame(rows)


def _morphology_allocation_coupling(fm: pd.DataFrame,
                                    rpc_csv: Path) -> pd.DataFrame:
    """
    Analysis 3: Spatial coupling of beta_H(u) and RPC(u).

    Uses tier-averaged RPC (T2–T4 mean, the comparable subset) per bin
    and formal_models beta_H per bin.

    Reports:
      - Spearman rho between beta_H(u) and mean_RPC(u) across bins
      - Mean RPC in mid-zone vs core (RPC_mid / RPC_core)
      - Mean RPC in outer vs core (RPC_outer / RPC_core)
    """
    from scipy.stats import spearmanr, pearsonr as _pearsonr

    bH = fm[fm["outcome"] == "H17"][["radial_bin", "beta"]].set_index("radial_bin")

    rpc_tier = pd.read_csv(rpc_csv)
    # Use T2+T3+T4 (exclude T1 — tiny settlements with few bins)
    rpc_main = rpc_tier[rpc_tier["tier"].isin(["T2", "T3", "T4"])].groupby(
        "radial_bin")["rpc_mean"].mean()

    common = bH.index.intersection(rpc_main.index)
    bH_v   = bH.loc[common, "beta"].values
    rpc_v  = rpc_main.loc[common].values

    rho, p_spear = spearmanr(bH_v, rpc_v)
    r,   p_pear  = _pearsonr(bH_v,  rpc_v)

    # Zone RPC contrasts
    def _zrpc(bins):
        return float(rpc_main[rpc_main.index.isin(bins)].mean())

    rpc_core  = _zrpc(ZONE_CORE)
    rpc_mid   = _zrpc(ZONE_MID)
    rpc_outer = _zrpc(ZONE_OUTER)

    rows = [
        {"metric": "Spearman_rho(beta_H, RPC)",
         "value": round(float(rho), 4), "p_value": round(float(p_spear), 4),
         "note": f"n_bins={len(common)}"},
        {"metric": "Pearson_r(beta_H, RPC)",
         "value": round(float(r), 4), "p_value": round(float(p_pear), 4),
         "note": f"n_bins={len(common)}"},
        {"metric": "RPC_core_mean",
         "value": round(rpc_core, 4), "p_value": np.nan, "note": f"bins={ZONE_CORE}"},
        {"metric": "RPC_mid_mean",
         "value": round(rpc_mid, 4), "p_value": np.nan, "note": f"bins={ZONE_MID}"},
        {"metric": "RPC_outer_mean",
         "value": round(rpc_outer, 4), "p_value": np.nan, "note": f"bins={ZONE_OUTER}"},
        {"metric": "RPC_mid / RPC_core",
         "value": round(rpc_mid / rpc_core, 4) if rpc_core > 0 else np.nan,
         "p_value": np.nan, "note": "zone ratio"},
        {"metric": "RPC_outer / RPC_core",
         "value": round(rpc_outer / rpc_core, 4) if rpc_core > 0 else np.nan,
         "p_value": np.nan, "note": "zone ratio"},
        {"metric": "beta_H_mid_mean",
         "value": round(_zone_mean_beta(fm, "H17", ZONE_MID), 6),
         "p_value": np.nan, "note": f"bins={ZONE_MID}"},
        {"metric": "beta_H_core_mean",
         "value": round(_zone_mean_beta(fm, "H17", ZONE_CORE), 6),
         "p_value": np.nan, "note": f"bins={ZONE_CORE}"},
    ]
    return pd.DataFrame(rows)


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    log.info("Loading bin_grid.json ...")
    with open(OUT_TABLES / "bin_grid.json") as f:
        grid = json.load(f)
    n_bins = int(grid["n_bins"])

    log.info("Loading log_pop map ...")
    pop_map = _load_log_pop()

    outcomes = ["H17", "r17", "JS", "rpc"]
    all_per_bin   = []
    all_interact  = []

    for name in outcomes:
        log.info(f"Fitting models for {name} ...")
        df = _load_outcome(name)
        if df is None:
            continue
        # per-bin models
        pbin = _fit_per_bin(df, pop_map, name, n_bins)
        all_per_bin.append(pbin)
        log.info(f"  {name}: {len(pbin)} bins with estimates.")
        # interaction model
        inter = _fit_interaction_model(df, pop_map, name)
        all_interact.append(inter)

    # ── save ──────────────────────────────────────────────────────────────
    if all_per_bin:
        result_pb = pd.concat(all_per_bin, ignore_index=True)
        result_pb.to_csv(OUT_TABLES / "formal_models.csv",
                         index=False, float_format="%.6f")
        log.info("Saved: formal_models.csv")
    else:
        log.warning("No per-bin model results to save.")

    if all_interact:
        result_int = pd.concat(all_interact, ignore_index=True)
        result_int.to_csv(OUT_TABLES / "profile_interaction_model.csv",
                          index=False, float_format="%.6f")
        log.info("Saved: profile_interaction_model.csv")

    # ── summary print ─────────────────────────────────────────────────────
    if all_per_bin and len(result_pb) > 0:
        print("\n=== Formal model β(u) estimates by outcome ===")
        for name in outcomes:
            sub = result_pb[result_pb["outcome"] == name]
            if len(sub) == 0:
                continue
            print(f"\n  {name}:")
            print(sub[["radial_bin","n_obs","beta","beta_se",
                        "p_value","partial_r2"]].to_string(index=False))

    # ── Analysis 1: Spatial division of labour — bootstrap test ──────────
    log.info("Analysis 1: Bootstrap test of spatial division (H17 vs r17) ...")
    df_H_full = _load_outcome("H17")
    df_r_full = _load_outcome("r17")
    if df_H_full is not None and df_r_full is not None:
        sdiv = _bootstrap_zone_delta(df_H_full, df_r_full, pop_map, n_bins,
                                     B=2000, seed=42)
        sdiv.to_csv(OUT_TABLES / "spatial_division_test.csv",
                    index=False, float_format="%.6f")
        log.info("Saved: spatial_division_test.csv")
        print("\n=== Analysis 1: Spatial division of labour (H17 vs r17) ===")
        print(sdiv[["statistic","estimate","ci_lo_95","ci_hi_95",
                     "p_one_sided"]].to_string(index=False))
    else:
        log.warning("Skipping Analysis 1: H17 or r17 data not available.")

    # ── Analysis 2: Convergence decay C(u) = |beta_D(u)| ─────────────────
    log.info("Analysis 2: Convergence decay ratios ...")
    if all_per_bin and len(result_pb) > 0 and "JS" in result_pb["outcome"].values:
        cdec = _convergence_decay(result_pb, B=2000, seed=42)
        cdec.to_csv(OUT_TABLES / "convergence_decay.csv",
                    index=False, float_format="%.6f")
        log.info("Saved: convergence_decay.csv")
        print("\n=== Analysis 2: Convergence decay (C_core / C_mid / C_outer) ===")
        print(cdec[["zone_pair","C_core","C_mid","C_outer",
                     "ratio","ci_lo_95","ci_hi_95"]].to_string(index=False))
    else:
        log.warning("Skipping Analysis 2: JS estimates not available.")

    # ── Analysis 3: Morphology–allocation coupling (SUPPLEMENTARY) ──────────
    # NOTE: mechanically confounded — both beta_H and RPC derive from
    # population-defined spatial units (centre defined by population).
    # Saved for supplement only; NOT used in main findings.
    log.info("Analysis 3 (supplementary): beta_H(u) ~ RPC(u) coupling ...")
    rpc_tier_csv = OUT_TABLES / "tier_rpc_profile.csv"
    if all_per_bin and len(result_pb) > 0 and rpc_tier_csv.exists():
        coup = _morphology_allocation_coupling(result_pb, rpc_tier_csv)
        coup.to_csv(OUT_TABLES / "morphology_allocation_coupling.csv",
                    index=False, float_format="%.6f")
        log.info("Saved: morphology_allocation_coupling.csv (supplementary)")
        print("\n=== Analysis 3 (supplementary): Morphology–allocation coupling ===")
        print("    [Not in main findings — see docstring for confounding note]")
        print(coup[["metric", "value", "p_value", "note"]].to_string(index=False))
    else:
        log.warning("Skipping Analysis 3: tier_rpc_profile.csv not available.")

    log.info("Done.")


if __name__ == "__main__":
    main()
