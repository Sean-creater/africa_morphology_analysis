"""
03_city_showcase.py
===================
Publication figure: one representative city per Leiden community (0–8).
For each city, two panels side-by-side:
  Left  — hex-grid spatial layout (cells coloured by k=17 morphological type)
  Right — 17-node type interaction graph

Layout: 3 rows × 3 columns, each cell = [spatial map | subgraph]
Footer: shared 17-type colour legend.

Usage
-----
    python 03_city_showcase.py
    # (requires Africa_env: geopandas, networkx, matplotlib)
"""

import logging
import os
import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent
CACHE_DIR   = ROOT / "outputs" / "cache"
OUT_DIR     = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SHP geometry + attributes per city, published by the companion
# africa_morphology_metrics repository (cleaning stage output,
# "SHP_cleaned"). Override with AFRICA_SHP_CLEANED_ROOT if your
# copy lives elsewhere.
SHP_ROOT    = Path(os.environ.get(
    "AFRICA_SHP_CLEANED_ROOT",
    "../../../africa_morphology_metrics/outputs/SHP_cleaned",
))
LABELS_PATH = (ROOT.parent.parent / "01_typology_construction" / "typology_construction"
               / "outputs" / "clusters" / "grid_type_labels.csv")

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
PALETTE_9 = [
    "#f36424",   # 0  orange
    "#672e92",   # 1  purple
    "#8bcf02",   # 2  lime green
    "#139ed3",   # 3  sky blue
    "#ff0000",   # 4  red
    "#ffd700",   # 5  gold
    "#293890",   # 6  dark blue
    "#74201e",   # 7  dark red
    "#bebebe",   # 8  grey
]

PALETTE_17 = [
    "#DBDBDB", "#FEEFCF", "#FCBBC5", "#40B273", "#E5152C",
    "#FFFF4C", "#5BA04A", "#FBB732", "#6F391F", "#B0E1E7",
    "#7B67EE", "#E89679", "#58C4EC", "#828F9B", "#1C1C5F",
    "#992FA3", "#C49A3C",
]
TYPE_COLORS = {i + 1: PALETTE_17[i] for i in range(17)}

# ---------------------------------------------------------------------------
# Representative cities — one per community (0-indexed)
# ---------------------------------------------------------------------------
CITY_REPS = [
    # (community, country, city, display_label)
    (0, "Ghana",        "Accra",         "Accra, Ghana"),
    (1, "Somalia",      "Mogadishu",     "Mogadishu, Somalia"),
    (2, "Kenya",        "Kisumu",        "Kisumu, Kenya"),
    (3, "Democratic Republic of the Congo", "Kananga", "Kananga, DR Congo"),
    (4, "Tunisia",      "Tunis",         "Tunis, Tunisia"),
    (5, "South Africa", "Johannesburg",  "Johannesburg, S. Africa"),
    (6, "Burkina Faso", "Ouagadougou",   "Ouagadougou, Burkina Faso"),
    (7, "Uganda",       "Soroti",        "Soroti, Uganda"),
    (8, "Ghana",        "Bolgatanga",    "Bolgatanga, Ghana"),
]

N_SEG = 60


# ---------------------------------------------------------------------------
# Helpers — subgraph drawing
# ---------------------------------------------------------------------------
def hex_to_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i+2], 16) / 255 for i in (0, 2, 4)])


def _bezier_gradient(ax, p0, p1, c0, c1, width, alpha, linestyle):
    mid = (p0 + p1) / 2
    toward = -mid
    norm = np.linalg.norm(toward)
    if norm > 1e-9:
        toward = toward / norm * 0.25
    ctrl = mid + toward
    t = np.linspace(0, 1, N_SEG + 1)
    pts = (np.outer((1-t)**2, p0)
           + np.outer(2*(1-t)*t, ctrl)
           + np.outer(t**2, p1))
    segs  = np.stack([pts[:-1], pts[1:]], axis=1)
    t_mid = (t[:-1] + t[1:]) / 2
    colors = c0*(1-t_mid[:,None]) + c1*t_mid[:,None]
    rgba   = np.hstack([colors, np.full((len(colors), 1), alpha)])
    lc = LineCollection(segs, colors=rgba, linewidth=width,
                        capstyle="round", linestyle=linestyle)
    ax.add_collection(lc)


def draw_subgraph(ax, G):
    p_attr = nx.get_node_attributes(G, "p")
    w_attr = nx.get_edge_attributes(G, "weight")
    nodes  = sorted(G.nodes())
    pos    = nx.circular_layout(G, scale=1.0)

    regular = [(u, v) for u, v in G.edges() if u != v]
    all_w   = [abs(w_attr.get((u, v), w_attr.get((v, u), 0.0)))
               for u, v in regular]
    max_w   = max(all_w) if all_w else 1.0

    for u, v in regular:
        w  = w_attr.get((u, v), w_attr.get((v, u), 0.0))
        aw = abs(w)
        wid = 0.15 + (aw / max_w) * 1.8
        alp = 0.18 + (aw / max_w) * 0.60
        ls  = "solid" if w >= 0 else "dashed"
        _bezier_gradient(ax,
                         np.array(pos[u]), np.array(pos[v]),
                         hex_to_rgb(TYPE_COLORS.get(u, "#888888")),
                         hex_to_rgb(TYPE_COLORS.get(v, "#888888")),
                         wid, alp, ls)

    max_p = max(p_attr.values()) if p_attr else 1.0
    for n in nodes:
        cx, cy = pos[n]
        r  = 0.045 + (p_attr.get(n, 0) / max_p) * 0.085
        circle = mpatches.Circle((cx, cy), radius=r,
                                 color=TYPE_COLORS.get(n, "#888888"),
                                 zorder=5, linewidth=0.5, edgecolor="k")
        ax.add_patch(circle)
        ax.text(cx, cy, str(n), ha="center", va="center",
                fontsize=5.5, fontweight="bold", color="k", zorder=6)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Helpers — spatial map drawing
# ---------------------------------------------------------------------------
def load_gdf(country, city, labels):
    shp_path = SHP_ROOT / country / city
    shps = list(shp_path.glob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"No SHP in {shp_path}")
    gdf = gpd.read_file(shps[0])
    city_labels = labels[labels["city"] == city][["grid_id", "label_k17"]]
    gdf = gdf.merge(city_labels, left_on="GridID", right_on="grid_id", how="inner")
    return gdf.reset_index(drop=True)


def draw_spatial(ax, gdf):
    colors = gdf["label_k17"].map(TYPE_COLORS).tolist()
    gdf.plot(ax=ax, color=colors, linewidth=0.05, edgecolor="white")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":       7,
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
    })

    log.info("Loading labels …")
    labels = pd.read_csv(LABELS_PATH, usecols=["grid_id", "city", "country", "label_k17"])

    # Figure: 3×3 grid, no footer legend, tight margins
    fig = plt.figure(figsize=(180/25.4, 175/25.4))

    gs_cities = fig.add_gridspec(
        3, 3,
        hspace=0.10, wspace=0.05,
        left=0.01, right=0.99,
        top=0.99, bottom=0.01,
    )

    for idx, (comm, country, city, label) in enumerate(CITY_REPS):
        row_i, col_i = divmod(idx, 3)

        # Each city cell: plain text label (top) + [spatial | subgraph]
        inner = gs_cities[row_i, col_i].subgridspec(
            2, 1, height_ratios=[0.06, 1.0], hspace=0.02,
        )

        # --- Plain text label (no colour bar) ---
        ax_title = fig.add_subplot(inner[0])
        ax_title.set_facecolor("white")
        ax_title.text(0.5, 0.5,
                      f"Comm {comm+1}  ·  {label}",
                      ha="center", va="center",
                      fontsize=5.5, color="#333333",
                      transform=ax_title.transAxes)
        ax_title.set_xticks([]); ax_title.set_yticks([])
        for sp in ax_title.spines.values():
            sp.set_visible(False)

        # --- Content: spatial map (left) + subgraph (right) ---
        content = inner[1].subgridspec(1, 2, wspace=0.03, width_ratios=[1, 1])
        ax_spatial = fig.add_subplot(content[0])
        ax_graph   = fig.add_subplot(content[1])

        # Spatial map
        log.info("Loading SHP for %s …", city)
        try:
            gdf = load_gdf(country, city, labels)
            draw_spatial(ax_spatial, gdf)
        except Exception as e:
            log.error("SHP failed for %s: %s", city, e)
            ax_spatial.axis("off")

        # Subgraph
        safe = f"{country}_{city}".replace(" ", "_")
        pkl  = CACHE_DIR / f"{safe}_type_graph.pkl"
        if not pkl.exists():
            log.error("Missing graph: %s", pkl)
            ax_graph.axis("off")
        else:
            with open(pkl, "rb") as f:
                G = pickle.load(f)
            n_reg = len([(u,v) for u,v in G.edges() if u!=v])
            log.info("Loaded %s (%d nodes, %d edges)", city,
                     G.number_of_nodes(), n_reg)
            draw_subgraph(ax_graph, G)

    # --- Save ---
    pdf_path = OUT_DIR / "city_showcase.pdf"
    png_path = OUT_DIR / "city_showcase.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    log.info("Saved → %s", pdf_path)
    log.info("Saved → %s", png_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
