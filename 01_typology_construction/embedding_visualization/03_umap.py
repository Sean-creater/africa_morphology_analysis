"""
03_umap.py
----------
Full-dataset UMAP on all ~1 M M6 StudentMLP embeddings, followed by a
publication-quality cluster scatter plot.

Pipeline
--------
1. Load all per-city .npz embedding files (embeddings, grid_ids).
2. Pre-clean: remove cells whose L2 norm falls outside
   [Q1 − 5·IQR, Q3 + 5·IQR] to filter degenerate embeddings.
3. L2-normalise (cosine-equivalent, consistent with triplet training).
4. Full UMAP: n_neighbors=30, min_dist=0.1, cosine metric, random_state=42.
   Runtime: ~30–60 min on a 32-core CPU.
5. Join k=17 cluster labels from merged_k17.parquet (column GridID).
6. Post-clean: drop cells whose UMAP coordinate deviates > 5σ on
   either axis.
7. Save full coordinates to outputs/umap_coords.parquet (cached —
   re-running the script skips steps 1–7 if the file exists).
8. Plot all retained cells with stratified sampling; annotate each cluster
   with its index.

Output
------
outputs/umap_coords.parquet
outputs/figures/umap_k17.pdf
outputs/figures/umap_k17.png

Usage
-----
    python 03_umap.py

Paths
-----
All input/output paths are defined in config.py.
"""

import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import umap
from sklearn.preprocessing import normalize

from config import EMBEDDINGS_PATH, MODEL_INPUT, OUT_FIG_DIR

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT            = Path(__file__).resolve().parent
UMAP_PARQUET    = ROOT / "outputs" / "umap_coords.parquet"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PRE_CLEAN_IQR_FACTOR = 5.0
POST_CLEAN_SIGMA     = 5.0

UMAP_PARAMS = dict(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
    low_memory=True,
    verbose=True,
    n_jobs=-1,
)

# ---------------------------------------------------------------------------
# Colour palette — 17 qualitative colours (matching 01_tsne.py)
# ---------------------------------------------------------------------------
PALETTE_17 = [
    "#DBDBDB", "#FEEFCF", "#FCBBC5", "#40B273", "#E5152C",
    "#FFFF4C", "#5BA04A", "#FBB732", "#6F391F", "#B0E1E7",
    "#7B67EE", "#E89679", "#58C4EC", "#828F9B", "#1C1C5F",
    "#992FA3", "#C49A3C",
]

FIG_W_MM, FIG_H_MM = 150, 150
MM_TO_IN = 1 / 25.4

# Stratified sampling: plot up to this many points total
PLOT_N_MAX     = 100_000
SMALL_CLUSTER  = 5_000   # clusters below this size are kept in full


# ---------------------------------------------------------------------------
# Step 1 — Load embeddings
# ---------------------------------------------------------------------------
def load_all_embeddings():
    t0 = time.time()
    log.info("Loading embeddings from %s ...", EMBEDDINGS_PATH)
    f = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    embeddings = f["embeddings"].astype(np.float32)
    grid_ids   = f["grid_ids"].astype(str)
    cities     = f["city"].astype(str)
    countries  = f["country"].astype(str)
    n_cities = len({(c, ci) for c, ci in zip(countries, cities)})
    log.info("Loaded %s cells from %d cities  [%.1fs]",
             f"{len(embeddings):,}", n_cities, time.time() - t0)
    return embeddings, grid_ids, cities, countries


# ---------------------------------------------------------------------------
# Step 2–3 — Pre-clean + L2 normalise
# ---------------------------------------------------------------------------
def pre_clean(embeddings, grid_ids, cities, countries):
    norms = np.linalg.norm(embeddings, axis=1)
    q1, q3 = np.percentile(norms, [25, 75])
    iqr = q3 - q1
    lo  = q1 - PRE_CLEAN_IQR_FACTOR * iqr
    hi  = q3 + PRE_CLEAN_IQR_FACTOR * iqr
    mask = (norms >= lo) & (norms <= hi) & np.isfinite(norms)
    log.info("Pre-clean: removed %d cells (L2 norm outside [%.3f, %.3f])",
             (~mask).sum(), lo, hi)
    return embeddings[mask], grid_ids[mask], cities[mask], countries[mask]


# ---------------------------------------------------------------------------
# Step 4 — Full UMAP
# ---------------------------------------------------------------------------
def run_umap(embeddings):
    log.info("Running UMAP on %s cells (n_neighbors=%d, min_dist=%.2f, metric=%s) ...",
             f"{len(embeddings):,}",
             UMAP_PARAMS["n_neighbors"],
             UMAP_PARAMS["min_dist"],
             UMAP_PARAMS["metric"])
    t0      = time.time()
    reducer = umap.UMAP(**UMAP_PARAMS)
    coords  = reducer.fit_transform(embeddings)
    log.info("UMAP done: shape %s  [%.1fs]", coords.shape, time.time() - t0)
    return coords.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 5–6 — Join labels + post-clean
# ---------------------------------------------------------------------------
def join_and_clean(grid_ids, cities, countries, coords):
    labels_df = pd.read_parquet(MODEL_INPUT, columns=["GridID", "label_k17"])
    labels_df["GridID"] = labels_df["GridID"].astype(str)
    df = pd.DataFrame({
        "GridID":  grid_ids,
        "city":    cities,
        "country": countries,
        "umap1":   coords[:, 0],
        "umap2":   coords[:, 1],
    })
    df = df.merge(labels_df, on="GridID", how="left")
    missing = df["label_k17"].isna().sum()
    if missing:
        log.warning("%d cells missing label — dropping", missing)
        df = df.dropna(subset=["label_k17"])
    df["label_k17"] = df["label_k17"].astype(int)

    for axis in ("umap1", "umap2"):
        mu, s = df[axis].mean(), df[axis].std()
        mask  = (df[axis] - mu).abs() <= POST_CLEAN_SIGMA * s
        removed = (~mask).sum()
        if removed:
            log.info("Post-clean %s: removed %d cells beyond %.1fσ",
                     axis, removed, POST_CLEAN_SIGMA)
        df = df[mask]

    log.info("After join + post-clean: %s cells", f"{len(df):,}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 7 — Save parquet
# ---------------------------------------------------------------------------
def save_parquet(df):
    UMAP_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    log.info("Saving %s ...", UMAP_PARQUET)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False),
        UMAP_PARQUET, compression="snappy",
    )
    log.info("Saved %s rows → %s", f"{len(df):,}", UMAP_PARQUET)


# ---------------------------------------------------------------------------
# Step 8 — Stratified sample + plot
# ---------------------------------------------------------------------------
def stratified_sample(df, n_max=PLOT_N_MAX, small_thresh=SMALL_CLUSTER,
                      seed=0):
    """
    Per-cluster stratified sample: keep all cells in clusters < small_thresh;
    proportionally downsample the rest to hit ~n_max total.
    """
    rng       = np.random.default_rng(seed)
    counts    = df["label_k17"].value_counts()
    small     = counts[counts < small_thresh].index
    large     = counts[counts >= small_thresh].index

    keep_idx  = df[df["label_k17"].isin(small)].index.tolist()
    budget    = n_max - len(keep_idx)

    if budget > 0 and len(large):
        large_df  = df[df["label_k17"].isin(large)]
        total_lg  = len(large_df)
        frac      = min(1.0, budget / total_lg)
        sampled   = large_df.groupby("label_k17", group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=int(rng.integers(1e6)))
        )
        keep_idx  = keep_idx + sampled.index.tolist()

    sub = df.loc[keep_idx].reset_index(drop=True)
    log.info("Stratified sample: %s → %s cells", f"{len(df):,}", f"{len(sub):,}")
    return sub


def plot(df):
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family":   "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":     7,
        "pdf.fonttype":  42,
        "ps.fonttype":   42,
    })

    sub = stratified_sample(df)

    clusters  = sorted(df["label_k17"].unique())
    color_map = {c: PALETTE_17[(c - 1) % len(PALETTE_17)] for c in clusters}

    fig, ax = plt.subplots(
        figsize=(FIG_W_MM * MM_TO_IN, FIG_H_MM * MM_TO_IN), dpi=300)
    fig.patch.set_facecolor("white")

    # Draw largest clusters first so smaller ones appear on top
    draw_order = sub["label_k17"].value_counts().sort_values(
        ascending=False).index.tolist()
    for c in draw_order:
        chunk = sub[sub["label_k17"] == c]
        ax.scatter(chunk["umap1"], chunk["umap2"],
                   c=color_map[c], s=1.5, alpha=0.5,
                   linewidths=0, rasterized=True, zorder=2)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Cluster index labels: hollow circle + number at cluster median
    for c in clusters:
        chunk = df[df["label_k17"] == c]
        cx, cy = chunk["umap1"].median(), chunk["umap2"].median()
        ax.scatter(cx, cy, s=80, color="none", zorder=5,
                   linewidths=0.5, edgecolors="black")
        ax.text(cx, cy, str(c), fontsize=5, fontweight="bold",
                color="black", ha="center", va="center", zorder=6,
                fontfamily="Arial")

    plt.tight_layout()

    for fmt in ("pdf", "png"):
        path = OUT_FIG_DIR / f"umap_k17.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight", format=fmt)
        log.info("Saved: %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log.info("=== 03_umap.py ===")

    if UMAP_PARQUET.exists():
        log.info("Cache found — loading %s", UMAP_PARQUET)
        df = pq.read_table(UMAP_PARQUET).to_pandas()
        log.info("Loaded %s cells from cache.", f"{len(df):,}")
    else:
        embeddings, grid_ids, cities, countries = load_all_embeddings()
        embeddings, grid_ids, cities, countries = pre_clean(
            embeddings, grid_ids, cities, countries)
        embeddings = normalize(embeddings, norm="l2").astype(np.float32)
        coords = run_umap(embeddings)
        df     = join_and_clean(grid_ids, cities, countries, coords)
        save_parquet(df)

    log.info("Plotting (stratified sample of %s pts) ...", f"{PLOT_N_MAX:,}")
    plot(df)
    log.info("=== Done ===")


if __name__ == "__main__":
    main()
