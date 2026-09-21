"""
00_prepare_data.py
------------------
Merge city-level and country-level covariates with Leiden community labels
from the community detection pipeline.

No category merging or recoding is performed beyond what is documented
below.  All original Köppen climate classes and colonial legacy categories
are retained as-is.  Dummy encoding is deferred to the modelling scripts
so that reference-category choices are explicit and auditable there.

Inputs
------
input/city.xlsx
    City-level: country, city, latitude, longitude, grid_cell_count,
    climate_zone(dominant), population_2020
input/Country.xlsx
    Country-level: Country, Geographic_Region,
    GDP_per_Capita(2020 dollar, source: World Bank),
    Colonial History (source: Our World in Data Colonizers Dataset)
../community_detection/outputs/city_communities.parquet
    City-level Leiden community labels (1–10).

Outputs
-------
outputs/tables/analysis_input.csv
    One row per city.  Contains:
      city, country, community (1–10),
      latitude, longitude, n_cells,
      climate_zone  (original Köppen code, 13 classes),
      colonial      (original 6-class encoding; see encode_colonial()),
      log_gdp       (log GDP per capita, 2020 USD, World Bank),
      log_pop       (log city population 2020),
      lat_c, lon_c  (mean-centred, SD-scaled latitude/longitude),
      lat2, lon2, lat_lon  (second-order spatial polynomial terms).
outputs/tables/covariate_summary.csv
    Counts and country coverage for climate_zone and colonial,
    and community × covariate cross-tabulations.  Used to assess
    separation risk before modelling.

Source data notes
-----------------
Country.xlsx uses the same country-name spellings as city.xlsx and the
companion morphology-metrics repository's country folder structure.
Population is still missing for a small number of cities (see log
warnings).

Colonial history encoding
-------------------------
Source: Our World in Data Colonizers Dataset (as recorded in Country.xlsx).
Multi-coloniser entries are semicolon-separated; the LAST entry is used as
the primary administrative legacy (reflects the final colonial period before
independence).  Countries with no formal colonisation are coded 'Independent'.
Unmatched entries are coded 'Other European' (Italy, Germany, Spain, etc.).

  British        – United Kingdom administration
  French         – France administration
  Belgian        – Belgium administration (DRC, Rwanda, Burundi)
  Portuguese     – Portugal administration
  Independent    – No formal colonisation recorded
                   NOTE: in this dataset this category consists entirely of
                   Ethiopian cities; coefficients reflect Ethiopian urban
                   characteristics, not a generalised "never-colonised" effect.
  Other European – Italy, Germany, Spain, and other minor colonial powers

The Independent category is retained without merging.  Its interpretive
limits are documented in covariate_summary.csv (country coverage) and
must be acknowledged in any publication.

Usage
-----
    python 00_prepare_data.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent
INPUT_DIR  = ROOT / "input"
OUT_TABLES = ROOT / "outputs" / "tables"
CD_OUT     = ROOT.parent / "community_detection" / "outputs"

COLONIAL_MAP = {
    "United Kingdom": "British",
    "France":         "French",
    "Portugal":       "Portuguese",
    "Belgium":        "Belgian",
}

# Country name harmonisation: city.xlsx spelling → Country.xlsx spelling.
# Populated only if the two source files' country-name spellings diverge;
# no recoding of country identity is implied.
COUNTRY_NAME_MAP: dict[str, str] = {}


# ---------------------------------------------------------------------------
def encode_colonial(raw: str) -> str:
    """
    Assign a primary colonial administrative legacy category.

    Rule: for semicolon-separated multi-coloniser strings, use the LAST
    entry (rightmost), which typically reflects the final administrative
    period before independence.  Countries with no formal colonisation are
    coded 'Independent'.  All remaining cases (Italy, Germany, Spain, etc.)
    are coded 'Other European'.
    """
    if pd.isna(raw):
        return "Unknown"
    raw = str(raw).strip()
    if "independent" in raw.lower() and "american" not in raw.lower():
        return "Independent"
    parts = [p.strip() for p in raw.split(";")]
    last  = parts[-1]
    for keyword, label in COLONIAL_MAP.items():
        if keyword in last:
            return label
    return "Other European"


def standardise_spatial(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add centred/scaled spatial terms (computed on the analysis sample):
      lat_c, lon_c  : mean-centred, unit-SD scaled
      lat2, lon2    : squared terms
      lat_lon       : interaction
    Centering and scaling parameters are logged for reproducibility.
    """
    df = df.copy()
    lat_mean, lat_sd = df["latitude"].mean(),  df["latitude"].std()
    lon_mean, lon_sd = df["longitude"].mean(), df["longitude"].std()
    logging.info("Spatial standardisation: lat mean=%.4f sd=%.4f; "
                 "lon mean=%.4f sd=%.4f", lat_mean, lat_sd, lon_mean, lon_sd)
    df["lat_c"]   = (df["latitude"]  - lat_mean) / lat_sd
    df["lon_c"]   = (df["longitude"] - lon_mean) / lon_sd
    df["lat2"]    = df["lat_c"] ** 2
    df["lon2"]    = df["lon_c"] ** 2
    df["lat_lon"] = df["lat_c"] * df["lon_c"]
    return df


def write_covariate_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Write covariate_summary.csv: for each categorical covariate, report
    N cities, N countries, and country coverage.
    Used to document interpretive limits (e.g. Independent=Ethiopia,
    Belgian≈DRC) and to assess separation risk before modelling.
    """
    rows = []

    for var in ["climate_zone", "colonial"]:
        for cat in sorted(df[var].dropna().unique()):
            sub = df[df[var] == cat]
            rows.append({
                "variable":   var,
                "category":   cat,
                "n_cities":   len(sub),
                "n_countries": sub["country"].nunique(),
                "pct_cities": round(len(sub) / len(df) * 100, 1),
                "top_countries": "; ".join(
                    sub["country"].value_counts().head(3).index.tolist()
                ),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "covariate_summary.csv", index=False)
    logging.info("Covariate summary saved.")


# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    # --- Load raw inputs ---
    city_xlsx    = INPUT_DIR / "city.xlsx"
    country_xlsx = INPUT_DIR / "Country.xlsx"
    city_df    = pd.read_excel(city_xlsx)
    country_df = pd.read_excel(country_xlsx)
    logging.info("Loaded city.xlsx (%d rows) and Country.xlsx (%d rows).",
                 len(city_df), len(country_df))

    # --- City table ---
    city_df = city_df.rename(columns={"climate_zone(dominant)": "climate_zone"})
    city_df["log_pop"] = np.log(city_df["population_2020"].clip(lower=1))
    city_df = city_df.drop(columns=["population_2020"])
    # Harmonise country names to match Country.xlsx spelling
    n_remapped = city_df["country"].isin(COUNTRY_NAME_MAP).sum()
    city_df["country"] = city_df["country"].replace(COUNTRY_NAME_MAP)
    logging.info("Country name harmonisation: %d cities remapped (%s).",
                 n_remapped,
                 ", ".join(f"{k}→{v}" for k, v in COUNTRY_NAME_MAP.items()))

    # --- Country table ---
    country_df = country_df.rename(columns={
        "Country": "country",
        " Geographic_Region": "geographic_region",
        "GDP_per_Capita(2020 dollar, source: World Bank)": "gdp_pc_2020",
        "Colonial History (source: Our World in Data Colonizers Dataset)":
            "colonial_raw",
    })
    country_df = country_df[
        ["country", "geographic_region", "gdp_pc_2020", "colonial_raw"]
    ].copy()
    country_df["log_gdp"] = np.log(country_df["gdp_pc_2020"].clip(lower=1))
    country_df["colonial"] = country_df["colonial_raw"].apply(encode_colonial)
    country_df = country_df.drop(columns=["colonial_raw", "gdp_pc_2020"])

    logging.info("Colonial legacy distribution (country level):\n%s",
                 country_df["colonial"].value_counts().to_string())

    unknown = (country_df["colonial"] == "Unknown").sum()
    if unknown > 0:
        logging.warning("%d countries have unknown colonial history.", unknown)

    # --- Community labels ---
    # Source: city_communities.parquet from 02_leiden_communities.py
    # community column is 0-based (0–8); convert to 1-based for analysis.
    part_path = CD_OUT / "city_communities.parquet"
    if not part_path.exists():
        raise FileNotFoundError(
            f"Community partition not found: {part_path}\n"
            "Run community_detection/02_leiden_communities.py first."
        )
    part_df = pd.read_parquet(part_path)[["city", "country", "community"]]
    part_df["community"] = part_df["community"] + 1   # 0-based → 1-based
    part_df["country"] = part_df["country"].replace(COUNTRY_NAME_MAP)
    logging.info("Loaded community partition: %d cities, %d communities.",
                 len(part_df), part_df["community"].nunique())

    # --- Merge ---
    df = (
        part_df
        .merge(city_df,    on=["city", "country"], how="inner")
        .merge(country_df, on="country",           how="left")
    )
    logging.info("After merge: %d cities, %d countries.",
                 len(df), df["country"].nunique())

    # --- Climate zone completeness check ---
    if df["climate_zone"].isna().sum() > 0:
        logging.warning("Unexpected missing climate_zone values: %d cities",
                        df["climate_zone"].isna().sum())
        logging.warning("%s", df.loc[df["climate_zone"].isna(),
                                     ["city", "country"]].to_string())

    # --- Spatial terms ---
    df = standardise_spatial(df)

    # --- Missing value report ---
    key_cols = ["community", "climate_zone", "colonial", "log_gdp",
                "log_pop", "lat_c", "lon_c"]
    missing = df[key_cols].isna().sum()
    logging.info("Missing values:\n%s", missing.to_string())
    if missing.any():
        logging.warning("Rows with any missing key covariate: %d",
                        df[key_cols].isna().any(axis=1).sum())

    # --- Output columns (no dummy encoding here) ---
    out_cols = [
        "city", "country", "community",
        "latitude", "longitude",
        "climate_zone", "colonial",
        "log_gdp", "log_pop",
        "lat_c", "lon_c", "lat2", "lon2", "lat_lon",
    ]
    out = df[out_cols].copy()
    out.to_csv(OUT_TABLES / "analysis_input.csv", index=False)
    logging.info("Analysis input saved: %d rows × %d cols.",
                 len(out), out.shape[1])

    # --- Covariate summary and cross-tabulations ---
    write_covariate_summary(out, OUT_TABLES)

    # --- Console summary ---
    print(f"\n  Cities            : {len(out)}")
    print(f"  Countries         : {out['country'].nunique()}")
    print(f"  Communities       : {out['community'].nunique()}  "
          f"(sizes: {out['community'].value_counts().sort_index().to_dict()})")
    print(f"\n  Climate zones     : {sorted(out['climate_zone'].dropna().unique())}")
    print(f"  Colonial categories: {sorted(out['colonial'].dropna().unique())}")
    print(f"\n  Missing values (key columns):\n"
          + "\n".join(f"    {c}: {missing[c]}" for c in key_cols))


if __name__ == "__main__":
    main()
