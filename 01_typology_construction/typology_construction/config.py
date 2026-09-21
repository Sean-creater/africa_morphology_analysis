"""
0_typology_construction — configuration
========================================
Hierarchical typology of urban morphology across African cities.

Pipeline overview
-----------------
1. L2-normalised 256-dim student embeddings (one per hexagonal grid cell)
2. Mini-batch K-Means compression to N_CENTROIDS representative centroids
3. Ward hierarchical agglomerative clustering (Euclidean distance) on centroids
4. Cut dendrogram at k=2–30; expert-guided selection of optimal k
"""
from pathlib import Path

# ── Input: student embeddings and K-Means centroids ───────────────────────────
# Per-city embedding files produced by the student distillation model.
# Each file contains L2-normalised 256-dim vectors (norm ≡ 1.0).
EMBEDDINGS_ROOT = Path(
    '/home/sjs96_file_share/Momepy_workflow/Metrix'
    '/data_cleaning_20260224/data_embeddings_20260513'
)

# 2,000 Mini-batch K-Means centroids derived from the full embedding set.
# Used to make Ward clustering tractable at continental scale (~1 M cells).
# Produced by ../k_selection/sampling/build_dendrogram_and_sample.py.
CENTROIDS_NPY = Path(__file__).parent.parent / 'k_selection' / 'outputs' / 'dendrogram' / 'centroids.npy'

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR     = Path(__file__).parent / 'outputs'
CLUSTER_DIR = OUT_DIR / 'clusters'
LOG_DIR     = OUT_DIR / 'logs'
FIG_DIR     = OUT_DIR / 'figures'

# ── Clustering parameters ─────────────────────────────────────────────────────
N_CENTROIDS = 2_000
K_LIST      = list(range(2, 31))   # candidate k values evaluated
K_FOCAL     = 17                   # k selected via expert pairwise evaluation, cross-checked
                                    # for stability across neighbouring k (see ../k_selection/)

# ── Linkage specification ─────────────────────────────────────────────────────
# Ward agglomerative clustering minimises the increase in within-cluster
# variance at each merge step. Requires Euclidean distance; on L2-normalised
# vectors, Euclidean and cosine rankings are monotonically equivalent
# (d_E² = 2 · d_cos), preserving the intended angular similarity structure.
LINKAGE_METHOD = 'ward'
LINKAGE_METRIC = 'euclidean'

# ── Misc ──────────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
SIL_SAMPLE     = 20_000   # subsample size for silhouette computation
