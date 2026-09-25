"""
verify_knn_graph_row_order_invariance.py
=========================================
One-off robustness check for the symmetric k-NN graph construction in
02_leiden_communities.py: confirms that the resulting community partition
does not depend on the row order of the input city feature table.

Method
------
1. Load the cached city distance matrices (D_p, D_a) and build the main
   k-NN graph + Leiden partition, as in 02_leiden_communities.py's main run.
2. Independently, apply a random permutation to the city order, rebuild
   the distance matrices, k-NN graph, and Leiden partition from scratch.
3. Compare the two partitions via the adjusted Rand index (label identity
   is arbitrary under a permutation, so this is the correct comparison,
   not a direct label match).

An ARI close to 1.0 confirms the graph — and therefore the resulting
communities — depends only on inter-city distances, not on city order.
This is not part of the regular pipeline; run it once after any change to
build_knn_graph to confirm the fix holds.

Usage
-----
    python verify_knn_graph_row_order_invariance.py
"""
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

lc = import_module("02_leiden_communities")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RANDOM_SEED = 42


def main():
    feat_path = lc.OUT_DIR / "city_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"{feat_path} not found — run 01_build_city_features.py first.")

    import pandas as pd
    df = pd.read_csv(feat_path) if feat_path.suffix == ".csv" else pd.read_parquet(feat_path)
    P = df[lc.P_COLS].values.astype(np.float32)
    A = df[lc.A_COLS].values.astype(np.float32)

    log.info("Building original-order partition ...")
    D_p, D_a = lc.load_or_compute_distances(feat_path, P, A, n_workers=8)
    D_main = lc.combined_distances(D_p, D_a, alpha=0.5)
    G_main = lc.build_knn_graph(D_main, k=100)
    mem_main = lc.run_leiden(G_main, n_seeds=5)

    log.info("Building permuted-order partition ...")
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(df))
    inv_perm = np.argsort(perm)

    D_main_perm = D_main[np.ix_(perm, perm)]
    G_perm = lc.build_knn_graph(D_main_perm, k=100)
    mem_perm = lc.run_leiden(G_perm, n_seeds=5)

    # Map permuted-order labels back to original order for comparison.
    mem_perm_original_order = np.asarray(mem_perm)[inv_perm]

    ari = adjusted_rand_score(mem_main, mem_perm_original_order)
    log.info(f"\nAdjusted Rand Index (original vs. row-permuted): {ari:.4f}")
    if ari > 0.99:
        log.info("PASS: k-NN graph construction is row-order invariant.")
    else:
        log.warning("FAIL: partition differs materially under row permutation "
                     "(ARI below 0.99) — investigate build_knn_graph.")


if __name__ == "__main__":
    main()
