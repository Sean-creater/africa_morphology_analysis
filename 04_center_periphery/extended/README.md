# Center-Periphery (Extended): Morphological Transition

Extends the center-periphery radial framework in `../core/` with city-size
stratification, asking how the radial transition in urban form differs
across small, medium, large, and metropolitan cities, and how much of the
cross-city variation it captures is attributable to national/regional
context versus population-driven spatial organisation alone.

## Method

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
  robustness checks with generating code (leave-one-country-out,
  alternative radial coordinate, alternative centre definition).
  `robustness_check{1,2,3}_*.csv` (alternative distance metric,
  rarefaction, and a stricter 50% quality-control threshold) are
  additional robustness results without a corresponding script in this
  repository; the coefficient table format matches `formal_models.csv`.
- `outputs/figures/` — the main three-panel figure and the illustrative
  ring visualisations.
