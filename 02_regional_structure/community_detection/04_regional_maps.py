"""
04_regional_maps.py
===================
Four publication-quality regional close-up maps, each highlighting a
geographic narrative of community structure:

  West Africa  — coastal–inland divergence
  North Africa — strong isolation / independence
  East Africa  — high mixing with residual regional patterning
  South Africa — clear community compartmentalisation

Each panel uses the same visual language as 03_visualize_communities.py
(palette, point sizing) so the two figures are directly comparable.
(Edges between cities were dropped — their interpretation was not
meaningful for the analysis.)

Inputs
------
outputs/plot_data.parquet       merged city + community + coordinate table
/…/africa_boundaries.shp        country polygons

Output
------
outputs/figures/regional_west.pdf  / .png
outputs/figures/regional_north.pdf / .png
outputs/figures/regional_east.pdf  / .png
outputs/figures/regional_south.pdf / .png

Usage
-----
    python 04_regional_maps.py
"""

import logging
import os
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd


def _polygon_to_mpl_path(geom) -> MplPath:
    """
    Build a matplotlib Path with an explicit MOVETO at the start of every
    ring (exterior + holes, and every part of a MultiPolygon).

    geopandas' batched GeoDataFrame.plot() draws the whole boundary layer
    as one PathCollection; when serialised to SVG, the break between two
    unrelated rings (e.g. mainland Africa's coastline and Madagascar's, or
    two different countries) can be silently dropped, leaving a LINETO
    where a MOVETO belongs. matplotlib/most browsers still render this
    correctly by luck, but Adobe Illustrator's SVG importer does not: it
    draws a stray straight line connecting the end of one ring to the
    start of the next, visible as long diagonal/horizontal artifacts
    crossing empty ocean. Building the Path explicitly, ring by ring,
    avoids relying on that shared code path entirely.
    """
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    vertices, codes = [], []
    for poly in polys:
        for ring in [poly.exterior, *poly.interiors]:
            coords = list(ring.coords)
            vertices.extend(coords)
            codes.append(MplPath.MOVETO)
            codes.extend([MplPath.LINETO] * (len(coords) - 2))
            codes.append(MplPath.CLOSEPOLY)
    return MplPath(vertices, codes)


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")

# ---------------------------------------------------------------------------
# Paths (mirror 03_visualize_communities.py)
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent
OUT_FIG   = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

PLOT_DATA  = ROOT / "outputs" / "plot_data.parquet"
# African country boundary polygons (e.g. derived from Africapolis/GADM
# administrative boundaries). Override with AFRICA_BOUNDARY_SHP to point
# at your own copy.
BOUNDARY   = Path(os.environ.get(
    "AFRICA_BOUNDARY_SHP",
    "/path/to/africa_boundaries.shp",
))

# ---------------------------------------------------------------------------
# Visual parameters — kept identical to 03_visualize_communities.py
# ---------------------------------------------------------------------------
LAND_COLOR    = "white"
BORDER_COLOR  = "#aaaaaa"
BORDER_LW     = 0.4

POINT_SCALE  = 0.14   # slightly smaller for regional close-ups
POINT_ALPHA  = 0.80
POINT_MIN_S  = 4.0
POINT_MAX_S  = 150.0

LABEL_FONTSIZE = 5.0

# Each panel: 90 mm × variable height (single-column Nature sub-journal)
FIG_W_MM = 120
DPI       = 300

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

# ---------------------------------------------------------------------------
# Region definitions  (lon_min, lon_max, lat_min, lat_max, title, subtitle)
# ---------------------------------------------------------------------------
REGIONS = {
    "west": (
        -18, 15, 3, 18,
        "West Africa",
        "Coastal–inland morphological divergence",
    ),
    "north": (
        -6, 42, 18, 38,
        "North Africa",
        "Strong regional independence",
    ),
    "east": (
        28, 52, -5, 18,
        "East Africa",
        "High community mixing with residual regional patterning",
    ),
    "south": (
        10, 40, -35, -10,
        "Southern Africa",
        "Clear community compartmentalisation",
    ),
}

# Margin around each bounding box (degrees)
MARGIN = 1.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def mm_to_in(mm): return mm / 25.4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mpl.rcParams.update({
        "font.family":     "Arial",
        "font.size":       8,
        "axes.linewidth":  0.5,
        "svg.fonttype": "none",
    })

    df     = pd.read_parquet(PLOT_DATA)
    africa = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")
    # Simplify: the source boundary layer has ~2.3M vertices total, far more
    # detail than visible even at these zoomed-in regional scales, and the
    # main reason the SVG was too heavy for Illustrator to open. A finer
    # tolerance than the continental map (these views are more zoomed in).
    africa["geometry"] = africa.geometry.simplify(0.005, preserve_topology=True)

    # Global arrays
    lon_all  = df["longitude"].values
    lat_all  = df["latitude"].values
    comm_all = df["community"].values.astype(int)
    sz_all   = np.clip(POINT_SCALE * np.sqrt(df["n_cells"].values),
                       POINT_MIN_S, POINT_MAX_S)

    # Skip labels for tiny/remote territories
    _skip_labels = {
        "French Southern Territories", "Mayotte", "Reunion",
        "Seychelles", "Comoros", "Cape Verde", "Sao Tome and Principe",
        "Mauritius", "Western Sahara",
    }

    for tag, (lon0, lon1, lat0, lat1, title, subtitle) in REGIONS.items():
        logging.info("Drawing region: %s", tag)

        # --- Subset cities inside bounding box ---
        mask = (
            (lon_all >= lon0) & (lon_all <= lon1) &
            (lat_all >= lat0) & (lat_all <= lat1)
        )
        if mask.sum() == 0:
            logging.warning("No cities in %s — skipping.", tag)
            continue

        lon  = lon_all[mask]
        lat  = lat_all[mask]
        comm = comm_all[mask]
        sz   = sz_all[mask]

        colours = np.array(PALETTE_9)[comm]

        # --- Figure geometry: width fixed, height from aspect ratio ---
        x_span = (lon1 - lon0) + 2 * MARGIN
        y_span = (lat1 - lat0) + 2 * MARGIN
        aspect = y_span / x_span
        fig_h_mm = FIG_W_MM * aspect
        fig, ax = plt.subplots(
            figsize=(mm_to_in(FIG_W_MM), mm_to_in(fig_h_mm)),
            constrained_layout=False)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.06)

        # Region label only (no subtitle)
        fig.text(0.5, 0.97, title,
                 ha="center", va="top",
                 fontsize=8, fontweight="bold",
                 fontfamily="Arial")

        # Backdrop — one PathPatch per country, each built ring-by-ring with
        # an explicit MOVETO (see _polygon_to_mpl_path), instead of a single
        # batched africa.plot() call whose merged PathCollection can lose
        # ring breaks when serialised to SVG (stray straight lines
        # connecting unrelated rings).
        #
        # Fill and stroke in TWO separate passes (all fills, then all
        # borders on top): neighbouring countries share near-identical
        # boundary coordinates, so with a single facecolor+edgecolor patch
        # per country, a later country's opaque fill can sit exactly on
        # top of an earlier country's border stroke along their shared
        # edge. Browsers' anti-aliasing often leaves a faint seam there by
        # chance, but Illustrator's precise vector rendering does not — the
        # shared border vanishes completely. Drawing every stroke after
        # all fills guarantees no fill can ever cover a border.
        geoms = list(africa.geometry)
        for geom in geoms:
            ax.add_patch(PathPatch(
                _polygon_to_mpl_path(geom),
                facecolor=LAND_COLOR, edgecolor="none", linewidth=0, zorder=1,
            ))
        for geom in geoms:
            ax.add_patch(PathPatch(
                _polygon_to_mpl_path(geom),
                facecolor="none", edgecolor=BORDER_COLOR,
                linewidth=BORDER_LW, zorder=2,
            ))

        # Points
        order = np.argsort(sz)
        ax.scatter(lon[order], lat[order],
                   s=sz[order], c=colours[order],
                   alpha=POINT_ALPHA, linewidths=0,
                   zorder=3, rasterized=True)

        # Extent
        ax.set_xlim(lon0 - MARGIN, lon1 + MARGIN)
        ax.set_ylim(lat0 - MARGIN, lat1 + MARGIN)
        ax.set_aspect("equal")
        ax.axis("off")

        # Country labels
        for _, row in africa.iterrows():
            name = row["NAME_0"]
            if name in _skip_labels:
                continue
            pt = row.geometry.representative_point()
            x, y = pt.x, pt.y
            if not (lon0 - MARGIN <= x <= lon1 + MARGIN and
                    lat0 - MARGIN <= y <= lat1 + MARGIN):
                continue
            ax.text(x, y, name,
                    fontsize=LABEL_FONTSIZE, fontfamily="Arial",
                    fontweight="bold", color="#333333",
                    ha="center", va="center", zorder=10, clip_on=True,
                    path_effects=[
                        mpl.patheffects.withStroke(
                            linewidth=1.5, foreground="white")])

        # Save
        for fmt in ("svg", "png"):
            p = OUT_FIG / f"regional_{tag}.{fmt}"
            fig.savefig(p, dpi=DPI, bbox_inches="tight")
            logging.info("Saved → %s", p)
        plt.close(fig)

    logging.info("All regional maps done.")


if __name__ == "__main__":
    main()
