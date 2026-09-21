# Cluster Interpretation

Characterises the k = 17 morphological types in terms of the underlying
morphometric indicators: which features distinguish the clusters, and how
each cluster deviates from the continent-wide average on those features.

## Method

1. `01_merge_data.py` — merges the 168 per-cell morphometric indicators
   with the k = 17 type labels from `../typology_construction/`.
2. `02_deduplicate_features.py` — removes redundant indicators via
   Spearman-correlation hierarchical clustering (complete linkage,
   |ρ| ≥ 0.90), retaining one representative per collinear group, ranked
   by η².
3. `03_feature_importance.py` — ranks the deduplicated features by global
   discriminative power (η², after rank-based Inverse Normal Transform)
   across the 17 clusters.
4. `04_cluster_profiles.py` — for each cluster and feature, the signed
   standardised mean deviation from the continent-wide average
   (D_cj), giving each cluster's morphological profile relative to the
   full dataset.

η² is reported as a descriptive variance-decomposition ratio; no p-values
are computed, consistent with the descriptive framework used throughout
this analysis at continental (~1M-cell) scale. See the manuscript's
Methods and Supplementary Information for the full statistical treatment.

## Data requirements

- Per-city cleaned morphology CSVs (`CLEANED_DIR` in `01_merge_data.py`),
  produced by the companion
  [`africa_morphology_metrics`](https://github.com/Sean-creater/africa_morphology_metrics)
  repository.
- `grid_type_labels.csv`, produced by
  `../typology_construction/02_assign_labels.py`.

## Usage

```bash
python 01_merge_data.py
python 02_deduplicate_features.py
python 03_feature_importance.py
python 04_cluster_profiles.py
```

## Outputs

- `outputs/tables/model_input_k17.csv` — merged indicator + label table,
  the shared input to steps 2–4 and to `../embedding_visualization/`.
- `outputs/tables/feature_groups.csv` / `selected_features.csv` — the
  redundancy-grouped and deduplicated feature lists.
- `outputs/tables/feature_importance.csv` — deduplicated features ranked
  by η².
- `outputs/tables/cluster_profiles.csv` / `cluster_sizes.csv` — per-cluster
  morphological profile (D_cj matrix) and grid-cell counts.
