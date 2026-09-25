# Center-Periphery (Core)

Computes each city's population-driven centre and the radial position of
every grid cell relative to it, in three parallel distance frameworks, and
tests whether morphological composition and network structure differ
systematically between centre and periphery.

## Method

Run order follows the dependency chain, not the file-number prefix —
`00_allometric_scaling.py` consumes `02_cell_radial_position.py`'s output
(the R98 distances), so it runs third. See Usage below.

1. `01_define_centres.py` — three city centre definitions: the
   population-weighted centroid of the dominant high-population cluster
   (main), the whole-city population-weighted centroid, and the
   unweighted geometric centroid (the latter two as robustness
   alternatives).
2. `02_cell_radial_position.py` — for every grid cell, absolute radial
   distance (km) and relative radius (fraction of R98) from each centre
   definition.
3. `00_allometric_scaling.py` — R98 (98th-percentile radial distance from
   each city's population centre) vs. population, giving the scaling
   exponents used to build the population-scaled distance framework
   (distance / population^α) that `03_ring_profiles.py` onward also use.
4. `03_ring_profiles.py` — aggregates cell-level radial data to city ×
   ring level, across all three distance frameworks.
5. `04_radial_network_metrics.py` — radial spatial assortativity per city
   × ring, using an edge-midpoint assignment strategy.
6. `05_city_differentiation.py` — city-level centre-periphery contrast
   metrics (CPC) from the ring profiles.
7. `06_radial_models.py` — the statistical models and figures: CPC scaling
   against population, and the radial gradient of morphological diversity
   and assortativity across rings, each across all three distance
   frameworks.

## Data requirements

- Per-city cleaned SHP grid geometries (`SHP_ROOT`), produced by the
  companion
  [`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics)
  repository.
- `grid_type_labels.csv`, produced by
  `../../01_typology_construction/typology_construction/02_assign_labels.py`.
- `analysis_input.csv` (city latitude/longitude), produced by
  `../../02_regional_structure/regression/00_prepare_data.py`. City
  population itself comes from each SHP's `Pop2022` field (area-weighted
  WorldPop 2022, aggregated within each Africapolis urban agglomeration;
  Ethiopia uses WorldPop 2020, the closest year with coverage) — this is
  the primary population variable used throughout the analysis, matching
  `../../02_regional_structure/regression/00_prepare_data.py`'s
  `population_worldpop_2022`.

## Usage

```bash
python 01_define_centres.py
python 02_cell_radial_position.py
python 00_allometric_scaling.py
python 03_ring_profiles.py
python 04_radial_network_metrics.py
python 05_city_differentiation.py
python 06_radial_models.py
```

## Outputs

- `outputs/tables/city_centres.parquet` — the three centre definitions per
  city, consumed by `../extended/`.
- `outputs/tables/cell_radial_position.parquet` / `city_R98.parquet` /
  `allometric_params.json` — the shared radial-coordinate inputs to
  `../extended/`.
- `outputs/tables/city_radial_metrics.csv` — city-level CPC metrics.
- `outputs/figures/` — the CPC scaling and radial gradient figures.
