# Regional-Structure Regression

Relates city community membership (from `../community_detection/`) to
climate, socioeconomic, and colonial-history covariates, via multinomial
logit models.

## Method

1. `00_prepare_data.py` — merges city- and country-level covariates
   (`input/city.xlsx`, `input/Country.xlsx`) with the Leiden community
   labels, into one row per city. No category merging or recoding beyond
   what each input column already represents; dummy encoding is left to
   the modelling scripts so reference-category choices stay explicit.
2. `01_mnl_main.py` — main multinomial logit analysis, in two parts:
   - **Q1** — what structures the continental partition into communities:
     a nested model sequence (M0–M4) with drop-one factor-group
     comparisons (likelihood-ratio test and partial pseudo-R²).
   - **Q2** — community-specific associations: average marginal effects
     and average adjusted predictions per community.
3. `02_multilevel_sensitivity.py` — a per-community binary mixed-logit
   sensitivity check with country random intercepts, asking how much of
   community membership is explained by national membership alone
   (variance-partition coefficient), beyond the individually measured
   covariates. See the script's docstring for the implementation note on
   binary mixed-effects estimation.
4. `03_figures.py` — the combined three-panel figure: factor
   contributions (Q1), community-specific associations (Q2), and the
   multilevel sensitivity result.

## Data requirements

- `input/city.xlsx` / `input/Country.xlsx` — city- and country-level
  covariates (coordinates, population, GDP, climate zone, colonial
  history), bundled in this repository.
- `../community_detection/outputs/city_communities.parquet` — community
  labels, produced by `../community_detection/02_leiden_communities.py`.

## Usage

```bash
python 00_prepare_data.py
python 01_mnl_main.py
python 02_multilevel_sensitivity.py
python 03_figures.py
```

## Outputs

- `outputs/tables/analysis_input.csv` — merged city-level table (one row
  per city), the shared input to steps 1–3.
- `outputs/tables/covariate_summary.csv` — coverage and cross-tabulation
  checks used to assess separation risk before modelling.
- `outputs/figures/` — the combined three-panel figure and supporting
  tables from steps 1–3.
