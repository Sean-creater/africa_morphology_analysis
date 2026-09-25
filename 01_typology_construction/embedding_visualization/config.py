"""
config.py
---------
All input/output paths for the embedding visualisation pipeline.

Edit the paths in this file to match your local data layout before running
any of the analysis scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (directory containing this file)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# INPUT PATHS  — edit these to match your data location
# ---------------------------------------------------------------------------

# Directory containing per-country sub-folders of per-city .npz embedding files.
# Each .npz must contain:
#   embeddings : float32 array of shape (N, 256)
#   grid_ids   : str array of shape (N,)
EMBEDDINGS_DIR = Path(
    "/home/sjs96_file_share/Momepy_workflow/Metrix"
    "/data_cleaning_20260224/data_embeddings_20260513"
)

# Parquet table with cluster labels and morphometric features, produced by
# ../cluster_interpretation/01_merge_data.py.
# Required columns: GridID (str), label_k17 (int), plus all feature columns
# listed in FEATURES (see 02_feature_overlay.py).
MODEL_INPUT = (
    ROOT.parent / "cluster_interpretation" / "outputs" / "tables" / "merged_k17.parquet"
)

# ---------------------------------------------------------------------------
# OUTPUT PATHS  — created automatically if they do not exist
# ---------------------------------------------------------------------------

# Intermediate: full-dataset t-SNE coordinates (GridID, tsne1, tsne2, label_k17, …)
TSNE_PARQUET = ROOT / "outputs" / "tsne_coords.parquet"

# Intermediate: full-dataset UMAP coordinates (GridID, umap1, umap2, label_k17, …)
UMAP_PARQUET = ROOT / "outputs" / "umap_coords.parquet"

# Figures
OUT_FIG_DIR = ROOT / "outputs" / "figures"
