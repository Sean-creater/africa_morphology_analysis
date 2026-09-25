"""
04_cluster_profiles.py
----------------------
Characterise each of the 17 urban-form clusters by the signed standardised
mean deviation (D_cj) of morphometric indicators across the features
identified in steps 02–03.

Method
------
For each cluster c and feature j, after rank-based Inverse Normal Transform:

    D_cj = (x̄_cj − x̄_j) / s_j

Because rank-INT maps every feature to mean ≈ 0, s ≈ 1 globally,
D_cj ≈ x̄_cj (the cluster mean on the transformed scale).  Positive values
indicate that the cluster lies above the global average for that feature;
negative values indicate below.  The magnitude reflects how many standard
deviations the cluster deviates from the full-Africa mean.

Features are ordered by their global η² rank (from step 03) within each
dimension group.  Both cluster rows and feature columns are reordered by
Ward hierarchical clustering of the D matrix to reveal block structure.

Outputs
-------
outputs/tables/cluster_profiles.csv
    (17 clusters × N features) D_cj matrix; index = cluster label.
outputs/tables/cluster_sizes.csv
    Grid-cell count and percentage share per cluster.
outputs/figures/cluster_profiles_heatmap.pdf
    Publication-ready clustered heatmap with Ward dendrograms on both
    axes, dimension-group colour bands above feature labels, and cluster
    sizes on the y-axis.

Usage
-----
    python 04_cluster_profiles.py [--top N]

    --top   Maximum number of features to include (default: all selected,
            capped at 40 for readability).
"""

import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.special import ndtri

N_JOBS = 16

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT               = Path(__file__).resolve().parent
MODEL_INPUT        = ROOT / "outputs" / "tables" / "merged_k17.parquet"
IMPORTANCE_CSV     = ROOT / "outputs" / "tables" / "feature_importance.csv"
OUT_TABLES         = ROOT / "outputs" / "tables"
OUT_FIGURES        = ROOT / "outputs" / "figures"
CACHE_DIR          = ROOT / "outputs" / "cache"
INT_CACHE          = CACHE_DIR / "rank_int_top.parquet"  # rank-INT data + label_k17


# ---------------------------------------------------------------------------
# Rank-based Inverse Normal Transform — vectorised
# ---------------------------------------------------------------------------
def rank_int(series: pd.Series) -> np.ndarray:
    """Blom-offset rank-INT; vectorised, no row-wise Python loop."""
    arr  = pd.to_numeric(series, errors="coerce").values.astype(float)
    mask = np.isfinite(arr)
    n    = mask.sum()
    out  = np.full(len(arr), np.nan)
    if n == 0:
        return out
    sub   = arr[mask]
    order = sub.argsort()
    ranked = np.empty(len(sub))
    ranked[order] = np.arange(1, len(sub) + 1, dtype=float)
    vals_sorted = sub[order]
    i = 0
    while i < len(vals_sorted):
        j = i + 1
        while j < len(vals_sorted) and vals_sorted[j] == vals_sorted[i]:
            j += 1
        if j > i + 1:
            ranked[order[i:j]] = ranked[order[i:j]].mean()
        i = j
    out[mask] = ndtri((ranked - 0.375) / (n + 0.25))
    return out


# ---------------------------------------------------------------------------
# D_cj computation
# ---------------------------------------------------------------------------
def compute_profiles(
    df: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Compute signed standardised mean deviation D_cj for all clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Rank-INT transformed data; columns include `features`.
    features : list[str]
        Ordered feature names.
    labels : np.ndarray
        Cluster label per grid cell (integer 1–17).

    Returns
    -------
    pd.DataFrame
        Shape (17, len(features)); index = cluster label (sorted).
    """
    global_mean = df[features].mean()
    global_std  = df[features].std().replace(0, np.nan)

    profiles = {}
    for k, grp in df.groupby(labels):
        profiles[k] = (grp[features].mean() - global_mean) / global_std

    return pd.DataFrame(profiles).T.sort_index()


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
def plot_heatmap(
    profiles: pd.DataFrame,
    importance: pd.DataFrame,
    sizes: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    Clustered heatmap of D_cj with variable-width columns (proportional to η²),
    coloured y-axis cluster dots, and x-axis labels for the top-20 features only.

    Parameters
    ----------
    profiles : pd.DataFrame
        (17 × N) D_cj matrix; index = cluster label; columns = col_labels.
    importance : pd.DataFrame
        Feature metadata; must contain 'feature' and 'eta2' columns.
    sizes : pd.DataFrame
        Cluster sizes; columns label_k17, n_cells, share_pct.
    out_path : Path
        Output PDF path.
    """
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable

    # Cluster colour palette (cluster 1 → index 0, …, cluster 17 → index 16)
    PALETTE_17 = [
        "#DBDBDB", "#FEEFCF", "#FCBBC5", "#F5E6C8", "#E5152C",
        "#FFFF4C", "#5BA04A", "#FBB732", "#6F391F", "#B0E1E7",
        "#7B67EE", "#E89679", "#58C4EC", "#828F9B", "#1C1C5F",
        "#992FA3", "#C49A3C",
    ]
    cluster_color = {i + 1: PALETTE_17[i] for i in range(17)}

    col_labels   = list(profiles.columns)
    clust_labels = list(profiles.index)   # integer cluster IDs, sorted 1–17
    Z_mat        = profiles.values        # (17 × N)
    n_feat       = len(col_labels)

    # Shorten column labels: "Group N (feat)" → "feat", singletons unchanged
    def _short(lbl: str) -> str:
        if lbl.startswith("Group ") and "(" in lbl:
            return lbl.split("(", 1)[1].rstrip(")")
        return lbl
    col_labels_short = [_short(l) for l in col_labels]

    # --- Ordering ---
    # Rows: canonical Ward-derived order (fixed across runs)
    # Columns: top-40 by η², displayed left→right most→least important
    CANONICAL_ROW_ORDER = [2, 17, 11, 16, 9, 10, 14, 15, 12, 13, 4, 5, 1, 6, 7, 3, 8]
    clust_id_to_pos = {int(cid): i for i, cid in enumerate(clust_labels)}
    row_order = [clust_id_to_pos[c] for c in CANONICAL_ROW_ORDER
                 if c in clust_id_to_pos]

    # Build η² array; select top-40 columns, then Ward-cluster those 40
    imp_lookup    = importance.set_index("feature")["eta2"].to_dict()
    eta2_arr      = np.array(
        [imp_lookup.get(col_labels_short[i], 0.0) for i in range(n_feat)],
        dtype=float,
    )
    TOP_COLS      = 40
    top40_idx     = np.argsort(-eta2_arr)[:TOP_COLS]   # indices into original columns

    # Reorder rows first, then slice top-40 columns
    Z_rows        = Z_mat[row_order, :]                 # (17 × n_feat)
    Z_top40       = Z_rows[:, top40_idx]                # (17 × 40)

    # Ward clustering on the 40-column submatrix to create visual blocks
    Z_finite_top  = np.nan_to_num(Z_top40, nan=0.0)
    col_link      = linkage(Z_finite_top.T, method="ward", metric="euclidean")
    col_sub_order = dendrogram(col_link, no_plot=True)["leaves"]  # permutation of 0..39

    Z_ord      = Z_top40[:, col_sub_order]
    labels_ord = [col_labels_short[top40_idx[i]] for i in col_sub_order]
    clust_ord  = [clust_labels[i] for i in row_order]
    n_clust    = len(clust_ord)
    n_show     = TOP_COLS

    # --- Figure layout ---
    fig_w  = max(14, n_show * 0.50)
    fig_h  = max(5.5, n_clust * 0.55)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[fig_w - 1.5, 0.22],
        wspace=0.03,
        left=0.06, right=0.97, top=0.93, bottom=0.30,
    )
    ax_heat = fig.add_subplot(gs[0])
    ax_cb   = fig.add_subplot(gs[1])

    # --- Heatmap: equal-width columns via imshow ---
    vmax = np.percentile(np.abs(Z_ord[np.isfinite(Z_ord)]), 98)
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    ax_heat.imshow(
        Z_ord, aspect="auto", cmap=cmap, norm=norm,
        interpolation="nearest",
    )

    ax_heat.tick_params(axis="x", length=0)
    ax_heat.tick_params(axis="y", length=0, left=False, labelleft=False)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    # x-axis: all shown column labels, rotated 45°
    ax_heat.set_xticks(range(n_show))
    ax_heat.set_xticklabels(
        labels_ord, rotation=45, ha="right", fontsize=8,
        rotation_mode="anchor",
    )

    # imshow centres row i at y=i; invert so row 0 is at top
    ax_heat.set_ylim(n_clust - 0.5, -0.5)
    ax_heat.set_xlim(-0.5, n_show - 0.5)

    # --- y-axis: coloured dot + cluster number ---
    # Strategy: place a narrow auxiliary axes immediately left of ax_heat,
    # sharing the same y data range. Dots and labels drawn inside it are
    # always within the figure bounds and are never clipped by bbox_inches.
    heat_pos = ax_heat.get_position()   # in figure fraction [l, b, w, h]
    DOT_AX_W = 0.035                    # figure-fraction width of dot strip
    ax_dot = fig.add_axes([
        heat_pos.x0 - DOT_AX_W - 0.005,  # left
        heat_pos.y0,                       # bottom
        DOT_AX_W,                          # width
        heat_pos.height,                   # height
    ])
    ax_dot.set_xlim(0, 1)
    ax_dot.set_ylim(n_clust - 0.5, -0.5)   # same inversion as heatmap
    ax_dot.axis("off")

    for row_i, cid in enumerate(clust_ord):
        col = cluster_color.get(int(cid), "#888888")
        # Number label
        ax_dot.text(0.38, row_i, str(int(cid)),
                    ha="right", va="center", fontsize=11, clip_on=False)
        # Colour rectangle spanning the full row height (y: row_i±0.5)
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (0.52, row_i - 0.5),   # (x_left, y_bottom) in data coords
            0.23, 1.0,              # width, height
            facecolor=col, linewidth=0,
            clip_on=False, zorder=5,
        )
        ax_dot.add_patch(rect)

    # --- Colourbar ---
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        cax=ax_cb, orientation="vertical",
    )
    cbar.set_label(
        r"Standardised deviation ($D_{cj}$)",
        fontsize=9, labelpad=6,
    )
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_visible(False)

    # --- Title ---
    fig.text(
        0.5, 0.965,
        "Urban-form cluster profiles",
        ha="center", va="top", fontsize=13, fontweight="bold",
    )

    # --- Save ---
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logging.info("Heatmap saved: %s (+ .png)", out_path)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _cache_key(features: list[str]) -> str:
    mtime = MODEL_INPUT.stat().st_mtime
    payload = json.dumps({"mtime": mtime, "features": sorted(features)}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_int_cache(features: list[str]) -> pd.DataFrame | None:
    """Return cached rank-INT DataFrame (features + label_k17) if valid, else None."""
    meta = CACHE_DIR / "rank_int_top.meta.json"
    if not INT_CACHE.exists() or not meta.exists():
        return None
    if json.loads(meta.read_text())["key"] != _cache_key(features):
        return None
    return pd.read_parquet(INT_CACHE)


def save_int_cache(features: list[str], df_int: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df_int.to_parquet(INT_CACHE, index=False)
    meta = CACHE_DIR / "rank_int_top.meta.json"
    meta.write_text(json.dumps({"key": _cache_key(features)}))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(top_n: int | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Load feature importance ranking ---
    # Reduce to one representative per collinear group (intragroup_rank == 1),
    # then take the top_n groups by η² (None = all).  Singletons have
    # intragroup_rank == 1 by definition, so they are included automatically.
    importance_all = pd.read_csv(IMPORTANCE_CSV)
    importance_sorted = (
        importance_all[importance_all["intragroup_rank"] == 1]
        .sort_values("eta2", ascending=False)
        .reset_index(drop=True)
    )
    if top_n is not None:
        importance_sorted = importance_sorted.head(top_n)
    importance = importance_sorted
    features = importance["feature"].tolist()
    logging.info(
        "Using top %d representative features (one per group/singleton) for cluster profiles.",
        len(features),
    )

    # Build display labels: "Group N (feat)" for multi-feature groups, feature name for singletons.
    # dimension_label column already encodes "Group N (feat1, feat2, feat3)"; extract "Group N (top1)".
    def _col_label(row: pd.Series) -> str:
        lbl = row["dimension_label"]
        if lbl == "Singleton":
            return row["feature"]
        # lbl is "Group N (feat1, feat2[, feat3])" — rewrite to "Group N (top1_feat)"
        group_part = lbl.split("(")[0].strip()   # "Group N"
        return f"{group_part} ({row['feature']})"

    importance["col_label"] = importance.apply(_col_label, axis=1)

    # --- Load and transform data (with cache) ---
    df_int = load_int_cache(features)
    if df_int is not None:
        logging.info("Rank-INT data loaded from cache (%s).", INT_CACHE)
        labels = df_int["label_k17"].values
    else:
        logging.info("Cache miss — loading model input ...")
        df_raw = pd.read_parquet(MODEL_INPUT, columns=features + ["label_k17"])
        labels = df_raw["label_k17"].values

        logging.info("Applying rank-INT (parallel, %d workers) ...", N_JOBS)
        def _rint(f):
            return f, rank_int(df_raw[f])
        with ThreadPoolExecutor(max_workers=N_JOBS) as ex:
            rint_results = dict(ex.map(_rint, features))
        df_int = pd.DataFrame(rint_results)
        df_int["label_k17"] = labels
        save_int_cache(features, df_int)
        logging.info("Rank-INT data cached to %s.", INT_CACHE)
        df_raw = None  # free memory

    # --- Cluster sizes ---
    sizes = (
        pd.DataFrame({"label_k17": labels})
        .groupby("label_k17").size()
        .reset_index(name="n_cells")
    )
    sizes["share_pct"] = (sizes["n_cells"] / sizes["n_cells"].sum() * 100).round(2)
    sizes.to_csv(OUT_TABLES / "cluster_sizes.csv", index=False)
    logging.info("Cluster sizes saved.")

    # --- D_cj profiles ---
    logging.info("Computing D_cj profiles ...")
    profiles = compute_profiles(df_int, features, labels)
    # Rename columns to display labels before saving
    profiles.columns = importance.set_index("feature").loc[features, "col_label"].tolist()
    profiles.to_csv(OUT_TABLES / "cluster_profiles.csv")
    logging.info("Profiles saved: %s", OUT_TABLES / "cluster_profiles.csv")

    # --- Heatmap ---
    plot_heatmap(
        profiles,
        importance[importance["feature"].isin(features)],
        sizes,
        out_path=OUT_FIGURES / "cluster_profiles_heatmap.pdf",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and visualise k=17 cluster profiles (D_cj)."
    )
    parser.add_argument(
        "--top", type=int, default=None,
        help="Maximum number of features (by η²) to include (default: all).",
    )
    args = parser.parse_args()
    main(top_n=args.top)
