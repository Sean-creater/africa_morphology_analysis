# Regional Structure

Builds a continent-level picture of how the k = 17 morphological types
combine at the city scale, groups cities into communities of similar
composition and spatial configuration, and relates that grouping to
climate, socioeconomic, and colonial-history covariates.

## Contents

- [`subgraph_construction/`](subgraph_construction/) — per-city spatial
  graphs and compact type-interaction summaries.
- [`community_detection/`](community_detection/) — Leiden communities of
  cities, from a similarity graph over the per-city summaries.
- [`regression/`](regression/) — multinomial logit models relating
  community membership to climate, GDP, population, and colonial history.

## Pipeline order

```
subgraph_construction/  ──type graphs──►  community_detection/  ──community labels──►  regression/
```

## Data requirements

Per-city cleaned SHP grid geometries and `grid_type_labels.csv`, produced
by the companion
[`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics)
and `../01_typology_construction/` respectively. See each submodule's
README for the specific paths to set.
