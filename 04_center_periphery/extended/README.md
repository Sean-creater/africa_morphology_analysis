# Center-Periphery (Extended): Morphological Transition

Extends the center-periphery radial framework in `../core/` with city-size
stratification, asking how the radial transition in urban form differs
across small, medium, large, and metropolitan cities, and how much of the
cross-city variation it captures is attributable to national/regional
context versus population-driven spatial organisation alone.

## Canonical pipeline (current, 2026-10-04)

This directory accumulated three generations of analysis as the project's
method evolved (discrete T1-T4 tiers -> continuous embedding framework ->
Africapolis tiers + community stratification + the A/B/C within-city
framework). Running an old script can silently overwrite a shared output
table that a newer script also reads, so **run only the canonical chain
below** unless you specifically need an older script's own output for
comparison.

```
core/01_define_centres.py -> core/02_cell_radial_position.py -> core/00_allometric_scaling.py
extended/13_embedding_radial_join.py
extended/20_trajectory_by_africapolis_tier.py -> extended/25_trajectory_by_tier_combined.py
extended/30_tier_radial_composition_divergence.py (composition explanation for the tier
                                              trajectory, parallel to community's composition_by_
                                              community.svg -- JS-distance curves + country-
                                              block bootstrap CI, 17-type and 4-group
                                              resolutions; reuses 20's composition tables
                                              directly, no new metric. Imports
                                              20_trajectory_by_africapolis_tier.py by path for
                                              its aggregation functions. A "3 stacked-area
                                              panels" alternative presentation of the same
                                              result adds no information beyond what 30
                                              already shows and is not included here.)
extended/32_coverage_adjusted_divergence.py (methods follow-up to 30: isolates how much of the
                                              Small-vs-Large core-periphery composition
                                              divergence survives once built-up coverage
                                              (LCF_Built) is held fixed within 3 global
                                              terciles, vs the raw (coverage-pooled) curve --
                                              answers "do differently-sized cities organise the
                                              SAME coverage level differently, or does the raw
                                              core-vs-periphery gap just reflect cores/
                                              peripheries having different coverage". Finding:
                                              the raw core-to-periphery decline (0.56->0.31)
                                              largely disappears once coverage is held fixed
                                              (~0.33-0.38 throughout) -- most of 30's raw
                                              divergence pattern reflects coverage, not
                                              tier-specific organisation of the same coverage.
                                              No population density/Pop2022 used (population
                                              only defines radial position, as throughout this
                                              project) -- LCF_Built is remote-sensing coverage,
                                              not a population estimate, so this carries none
                                              of the WorldPop-circularity risk flagged on
                                              scripts 14/26. A community-pair version (C6 vs C8)
                                              was attempted and dropped: zero radial bins had
                                              common support across all 3 coverage terciles for
                                              both communities.)
extended/33_tier_coverage_divergence.py (within-tier, not tier-vs-tier: for each Africapolis
                                              size tier SEPARATELY, full pairwise cross-city
                                              embedding divergence (V_between) at each city's
                                              own built-up coverage level -- "at X% coverage, how
                                              unpredictable are same-tier cities from each
                                              other", not "how different do the tiers look from
                                              each other" (that's 32's question). Two figures,
                                              core and periphery, reusing 23's qualifying
                                              636-city sample and core/outer embeddings. Finding:
                                              at the CORE, Intermediate and Large tiers converge
                                              with rising coverage (less cross-city divergence at
                                              high coverage) while Small tier DIVERGES (more
                                              divergence at high coverage) -- opposite direction,
                                              not just a different slope. At the PERIPHERY, all
                                              three tiers diverge with rising coverage (same
                                              direction), so "higher coverage -> more predictable
                                              form" only holds for mid/large cities' cores, not
                                              small cities' cores and not any tier's periphery.
                                              No population density/Pop2022 used.)
extended/21_trajectory_by_community.py (single-panel combined figure, self-contained)
extended/23_scale_dependent_convergence.py  (A: between-city, reversal P*~126k)
extended/29_core_outer_divergence_mechanism.py (mechanism decomposition of A --
                                              core vs periphery levels separately;
                                              reuses 23's dispersion table, no new metric)
extended/27_within_city_heterogeneity.py    (B: within-city)
extended/28_spatial_coherence.py            (C: spatial coherence -- EXPLORATORY, see its docstring)
extended/14_coverage_vs_morphology.py       (built-up coverage vs. morphology: out-of-sample R^2 and
                                              density-ratio result, see caveat below)
extended/17_figure_main_embedding.py        (main 4-panel figure; panel b sources from 14 above)
```

Scripts `02`-`09`/`10`/`12` (the original 17-type radial-profile /
divergence / T1-T4-tier robustness scripts, see "Method" below) are a
separate, still-valid analysis at the type-composition level, not
superseded by the embedding framework.

**Caveats:**
- `14_coverage_vs_morphology.py` predicts WorldPop-constrained population
  (or log-population) from built coverage + embedding. WorldPop's own
  constrained-disaggregation methodology uses building footprints / GHSL
  built-settlement masks / roads as covariates, so this risks re-discovering
  WorldPop's modelling inputs rather than finding independent morphology
  evidence; its R²/density-ratio numbers (feeding `17_figure_main_embedding.py`
  panel b) should be read as descriptive, not as a causal morphology effect.

**Exploratory / deprecated -- do not treat as the main-text source:**
- `19_trajectory_curve_fit_stratified.py` -- T1-T4-tier curve fits,
  superseded by `20_trajectory_by_africapolis_tier.py`'s Africapolis tiers.
- `22_community_vs_population_confound.py` -- a confound check for
  `21_trajectory_by_community.py`; informative but not a main-text figure.
- `24_continuous_q_dispersion_heatmap.py` -- a continuous-q visualisation
  of the SAME pattern `23_scale_dependent_convergence.py` reports formally;
  23 remains the formal result with the country-block bootstrap CI.
- `26_radial_coverage_vs_morphology.py` -- methods follow-up to `14`,
  stopped mid-run with no saved output.
- `00_define_size_tiers.py` through `12_tier_rings_comparison.py`'s T1-T4
  fixed-breakpoint framing specifically (as opposed to the still-valid
  type-composition analyses built on top of it, see "Method" above).

## Method

The T1–T4 fixed population-size tiers computed in `00_define_size_tiers.py`
are a legacy framing, superseded by the continuous-population embedding
framework (`13`–`33` below) as the primary basis for population-driven
claims in the manuscript. The type-composition-level analyses built on
those same four tiers (`02`–`09`/`10`/`12`) remain independently valid and
reported: read their results as tier-conditional descriptive profiles,
not as evidence for the T1–T4 breakpoints' validity as a population
segmentation.

1. `00_define_size_tiers.py` — four fixed population-size tiers (T1–T4,
   inherited from `../../03_urban_hierarchy/`'s scaling breakpoints), plus
   a supplementary two-breakpoint segmented regression checking that
   `H17_star`/`r17` also show non-linear change near those boundaries.
2. `01_radial_coordinates.py` — 10 equal-width radial bins (core = bin 0,
   periphery = bin 9) over the common-support range of the population-
   scaled distance from `../core/`, applied consistently across all
   distance coordinates.
3. `02_radial_type_profiles.py` — mean k = 17 type-share profile per tier
   × radial bin.
4. `03_radial_diversity.py` — rarefied 17-type Shannon diversity profile
   per tier × radial bin.
5. `04_radial_assortativity.py` — 17-type spatial assortativity profile
   (Newman categorical, Queen adjacency, edge-midpoint bin assignment) per
   tier × radial bin.
6. `05_between_city_divergence.py` — each city's morphological divergence
   (Jensen–Shannon, against the global per-bin composition) as a function
   of radial position.
7. `06_regional_effect_by_distance.py` — at each radial bin, how much of
   the cross-city variation in composition is explained by climate zone,
   colonial legacy, and GDP per capita, connecting the regional context
   from `../../02_regional_structure/regression/` to intra-urban position.
   Shares its partial-R² model-fitting helpers with `_regional_effect_shared.py`.
8. `07_population_land_profiles.py` — population-to-land allocation ratio
   (RPC) per tier × radial bin × type, relative to SDG 11.3 land
   consumption efficiency.
9. `08_statistical_models.py` — the formal per-bin regression models
   underlying the diversity and assortativity profiles above.
10. `09_robustness.py` — three robustness checks against the two main
    findings (spatial division of labour between core and mid-ring; core
    convergence across cities): leave-one-country-out, an alternative
    radial coordinate (distance / R98 instead of population-scaled
    distance), and an alternative centre definition (population centroid
    instead of the primary high-population cluster).
11. `10_figure_main.py` — the main three-panel figure combining core
    restructuring, spatial partitioning, and regional context.
12. `11_accra_rings_viz.py` / `12_tier_rings_comparison.py` — illustrative
    single-city and cross-tier ring visualisations.

## Data requirements

- `city_centres.parquet`, `cell_radial_position.parquet`, `city_R98.parquet`,
  `allometric_params.json`, produced by `../core/`.
- `city_metrics.parquet`, produced by `../../03_urban_hierarchy/`.
- `analysis_input.csv`, produced by `../../02_regional_structure/regression/`.

## Python environment

Run everything inside the repository's pinned conda environment, not a
bare `python3` / `python` off the shell's default `$PATH` -- a system or
other default Python build (e.g. 3.12/3.13) can fail to read this
project's parquet files (`city_centres.parquet` and others raise a
"Repetition level histogram size mismatch" pyarrow error under an
unpinned interpreter):

```bash
conda env create -f ../../environment.yml
conda activate africa_morphology_analysis
python <script>.py
```

Either env reads the project's parquet files correctly; the split above is
only about which env has `geopandas`/`libpysal` installed.

## Usage

```bash
python 00_define_size_tiers.py
python 01_radial_coordinates.py
python 02_radial_type_profiles.py
python 03_radial_diversity.py
python 04_radial_assortativity.py
python 05_between_city_divergence.py
python 06_regional_effect_by_distance.py
python 07_population_land_profiles.py
python 08_statistical_models.py
python 09_robustness.py
python 10_figure_main.py
python 11_accra_rings_viz.py
python 12_tier_rings_comparison.py
```

## Outputs

- `outputs/tables/city_tiers.parquet` / `radial_cells.parquet` — the
  shared tier and radial-bin assignments used throughout.
- `outputs/tables/city_bin_profiles.parquet` / `city_bin_diversity.parquet`
  / `city_bin_assortativity.parquet` / `city_bin_divergence.parquet` /
  `city_bin_rpc.parquet` — the per-tier-and-bin profiles from steps 2–7.
- `outputs/tables/formal_models.csv` — the regression models underlying
  the diversity and assortativity profiles.
- `outputs/tables/robustness/robustness_check{A,B,C}_*.csv` — the three
  robustness checks from `09_robustness.py` (leave-one-country-out,
  alternative radial coordinate, alternative centre definition).
- `outputs/figures/` — the main three-panel figure and the illustrative
  ring visualisations.
