"""
02_assign_labels.py
===================
Assign morphological type labels (k=2–30) to all grid cells.

Each grid cell is mapped to its nearest centroid (Euclidean distance on
L2-normalised vectors), then the Ward dendrogram is cut at each candidate k
to produce the final cell-level type assignment.

Outputs
-------
outputs/clusters/grid_type_labels.csv
    Columns: grid_id, city, country, label_k02 … label_k30
outputs/logs/02_assign_labels.log
"""

import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import pairwise_distances_argmin

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import EMBEDDINGS_ROOT, CLUSTER_DIR, LOG_DIR, K_LIST

# Reference file that carries the canonical city/country name strings.
# These are used by all downstream scripts (subgraph_construction, etc.)
# and must be full names (e.g. "Ghana", not "GHA").
_REF_LABELS_CSV = Path(
    '/home/sjs96_file_share/DL_workflow/K_Selection/'
    'outputs/grid_cluster_labels_k02_k30.csv'
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
CLUSTER_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_DIR / '02_assign_labels.log', mode='w'),
        logging.StreamHandler(),
    ],
)
log = logging.info

log('=' * 60)
log('02_ASSIGN_LABELS')
log('=' * 60)

# ── Load dendrogram and centroids ─────────────────────────────────────────────
linkage_path  = CLUSTER_DIR / 'linkage_matrix.npy'
centroids_path = CLUSTER_DIR / 'centroids.npy'
assert linkage_path.exists(),   f'Run 01_build_linkage.py first: {linkage_path}'
assert centroids_path.exists(), f'Run 01_build_linkage.py first: {centroids_path}'

Z         = np.load(linkage_path)
centroids = np.load(centroids_path).astype(np.float32)
log(f'\nLinkage matrix: {Z.shape}')
log(f'Centroids:      {centroids.shape}')

# ── Cut dendrogram at k=2–30 ──────────────────────────────────────────────────
log(f'\nCutting dendrogram at k={K_LIST[0]}–{K_LIST[-1]} …')
cent_labels = {k: fcluster(Z, k, criterion='maxclust') - 1 for k in K_LIST}

# ── Load per-city embeddings ───────────────────────────────────────────────────
log(f'\nLoading embeddings from:\n  {EMBEDDINGS_ROOT}')
t0 = time.time()
npz_paths = sorted(EMBEDDINGS_ROOT.rglob('*.npz'))
log(f'  Found {len(npz_paths):,} city files')

all_embs, all_gids = [], []
for path in npz_paths:
    d    = np.load(path, allow_pickle=True)
    all_embs.append(d['embeddings'].astype(np.float32))
    all_gids.extend(d['grid_ids'].astype(str).tolist())

embeddings = np.vstack(all_embs)
grid_ids   = np.array(all_gids)
N = len(grid_ids)
log(f'  Loaded {N:,} cells in {time.time() - t0:.1f}s')

# ── Load canonical city/country names from reference file ─────────────────────
log(f'\nLoading city/country names from reference: {_REF_LABELS_CSV}')
ref = pd.read_csv(_REF_LABELS_CSV, usecols=['grid_id', 'city', 'country'])
ref = ref.set_index('grid_id')

norms = np.linalg.norm(embeddings, axis=1)
log(f'  Embedding norm — mean: {norms.mean():.4f}  std: {norms.std():.4f}')

# ── Assign each cell to its nearest centroid ───────────────────────────────────
log('\nAssigning cells to nearest centroid …')
t1 = time.time()
CHUNK = 50_000
pre_labels = np.empty(N, dtype=np.int32)
for start in range(0, N, CHUNK):
    end = min(start + CHUNK, N)
    pre_labels[start:end] = pairwise_distances_argmin(
        embeddings[start:end], centroids, metric='euclidean'
    )
    if (start // CHUNK) % 20 == 0:
        log(f'  … {end:,} / {N:,}')

log(f'  Completed in {time.time() - t1:.1f}s  '
    f'({len(np.unique(pre_labels))} / {len(centroids)} centroids populated)')

# ── Build output table ─────────────────────────────────────────────────────────
log('\nBuilding output table …')
df = pd.DataFrame({'grid_id': grid_ids})
df = df.join(ref[['city', 'country']], on='grid_id', how='left')
for k in K_LIST:
    df[f'label_k{k:02d}'] = (cent_labels[k][pre_labels] + 1).astype(np.int16)

log(f'  Shape: {df.shape}')
log(f'  Type distribution at k=17:\n'
    f'{df["label_k17"].value_counts().sort_index().to_string()}')

out_csv = CLUSTER_DIR / 'grid_type_labels.csv'
df.to_csv(out_csv, index=False)
log(f'\nSaved: {out_csv}  ({out_csv.stat().st_size / 1e6:.1f} MB)  {N:,} rows')
log('\nNext: python3 03_kselection_diagnostics.py')
