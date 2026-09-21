# Embedding Visualisation

2D projections of the continent-scale student embeddings, for visual
inspection of cluster separation and how individual morphometric features
vary across the embedding space.

## Method

1. `01_tsne.py` — full-dataset t-SNE (openTSNE, Barnes-Hut, cosine metric)
   on all ~1M embeddings, coloured by k = 17 cluster label.
2. `02_feature_overlay.py` — the same t-SNE coordinates, coloured by the
   top-12 morphometric features (ranked by η² in
   `../cluster_interpretation/`) instead of cluster label.
3. `03_umap.py` — the same visualisation using UMAP instead of t-SNE.

Embeddings are L2-normalised before projection (cosine metric).
Degenerate embeddings (L2 norm outside the IQR-based range) and projection
outliers (>5σ on either axis) are dropped before plotting; see each
script's docstring for the exact thresholds.

## Data requirements

- Continent-scale student embeddings (`EMBEDDINGS_DIR` in `config.py`),
  produced by the companion
  [`africa_morphology_embedding`](https://github.com/Sean-creater/africa_morphology_embedding)
  repository.
- `model_input_k17.csv` (`MODEL_INPUT_CSV` in `config.py`), produced by
  `../cluster_interpretation/01_merge_data.py`.

## Usage

```bash
python 01_tsne.py
python 02_feature_overlay.py   # requires 01_tsne.py to have run first
python 03_umap.py
```

## Outputs

- `outputs/tsne_coords.parquet` / `umap_coords.parquet` — cached 2D
  coordinates (re-running a script skips recomputation if its cache
  exists).
- `outputs/figures/tsne_k17.*` / `umap_k17.*` — cluster-coloured scatter
  plots.
- `outputs/figures/tsne_feature_overlay_1.*` / `_2.*` — feature-coloured
  scatter plots, top-12 features by η².
