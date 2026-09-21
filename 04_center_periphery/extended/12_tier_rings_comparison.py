"""
12_tier_rings_comparison.py
===========================
Publication-quality 1×4 comparison of population-scaled radial binning
across four city-size tiers (T1–T4), illustrating how the same relative
binning scheme maps onto cities of very different physical scales.

Cities selected (one per tier, all 10 bins present, uniform bin distribution):
  T1 — Ophondweni,       South Africa  (~17 k pop,   379 cells,  12×23 km)
  T2 — Jerba Houmet Souk, Tunisia      (~82 k pop, 1 270 cells,  28×17 km)
  T3 — Pietermaritzburg, South Africa  (~932 k pop, 1 465 cells, 35×21 km)
  T4 — Bafoussam,        Cameroon      (~1.8 M pop, 9 313 cells, 72×73 km)

Layout: single figure, four equal sub-panels side by side.
Each panel shows:
  • 250 m grid cells coloured by radial bin (0–9)
  • Clipped bin-boundary arcs (city footprint ∩ circle)
  • City footprint boundary (black outline)
  • Population-weighted primary centre (red star)
  • Scale bar (physical km — to make size difference explicit)
  • Tier label + city name

Shared colour strip at right edge of figure.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
import matplotlib.colors as mcolors
from shapely.geometry import Point as ShpPoint, MultiPoint, box
from shapely.ops import unary_union
from pathlib import Path
from datetime import date

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.linewidth":   0.6,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
OUT_TABLES = ROOT / "outputs" / "tables"
FIG_DATE   = date.today().strftime("%Y-%m-%d")
OUT_FIG    = ROOT / "outputs" / "figures" / FIG_DATE
OUT_FIG.mkdir(parents=True, exist_ok=True)

# ── Colour scheme (shared with 11_accra_rings_viz.py) ─────────────────────
BIN_COLORS = {
    0: "#C39BD3", 1: "#7D3C98",   # Core:  light purple → deep purple
    2: "#82E0AA", 3: "#27AE60", 4: "#1E8449", 5: "#145A32",  # Mid: light→dark green
    6: "#FAD7A0", 7: "#F39C12", 8: "#D35400", 9: "#784212",  # Outer: light→dark amber
}
ZONE_COL  = {"Core": "#7D3C98", "Mid": "#1E8449", "Outer": "#D35400"}
ZONE_BINS = {"Core": [0, 1], "Mid": [2, 3, 4, 5], "Outer": [6, 7, 8, 9]}
NA_COLOR  = "#E8E4DC"

# ── City catalogue ─────────────────────────────────────────────────────────
CITIES = [
    dict(tier="T1", city="Ophondweni",       country="South Africa", crs_epsg=32736),
    dict(tier="T2", city="Jerba Houmet Souk", country="Tunisia",      crs_epsg=32632),
    dict(tier="T3", city="Pietermaritzburg", country="South Africa", crs_epsg=32736),
    dict(tier="T4", city="Bafoussam",         country="Cameroon",    crs_epsg=32632),
]

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading radial_cells …")
rc_all = pd.read_parquet(OUT_TABLES / "radial_cells.parquet")
cc_all = pd.read_parquet(
    Path(__file__).parent.parent
    / "core/outputs/tables/city_centres.parquet"
)


def _build_city_data(cfg):
    """Return all spatial data needed to draw one city panel."""
    city, country = cfg["city"], cfg["country"]
    crs_epsg = cfg["crs_epsg"]
    crs = f"EPSG:{crs_epsg}"

    # --- radial cells for this city ---
    rc = rc_all[(rc_all["city"] == city) & (rc_all["country"] == country)].copy()
    if rc.empty:
        raise ValueError(f"No cells found for {city}, {country}")

    # --- primary centre ---
    row_cc = cc_all[(cc_all["city"] == city) & (cc_all["country"] == country)]
    if len(row_cc):
        cx_c = float(row_cc["primary_pop_x"].values[0])
        cy_c = float(row_cc["primary_pop_y"].values[0])
    else:
        # fallback: cell closest to dist_m_primary == 0
        cx_c = float(rc.loc[rc["dist_m_primary"].idxmin(), "cx"])
        cy_c = float(rc.loc[rc["dist_m_primary"].idxmin(), "cy"])

    # --- build city footprint from buffered centroids ---
    # Use buffer = hex row pitch / 2 ≈ 217 m so adjacent cells merge into
    # a single polygon.  Build on ALL cells (inc. NaN-bin) for arc clipping,
    # and on binned-only cells for the visible boundary outline.
    HEX_PITCH = 250 * 3**0.5        # ≈ 433 m — actual centre-to-centre spacing
    BUFFER    = HEX_PITCH / 2 + 5   # ≈ 222 m — just enough to bridge the gap

    def _buffered_union(xy_arr):
        pts_l = [ShpPoint(x, y) for x, y in xy_arr]
        if len(pts_l) <= 2000:
            return unary_union([p.buffer(BUFFER) for p in pts_l])
        # tile for speed on large cities
        xs_t = xy_arr[:, 0]; ys_t = xy_arr[:, 1]
        tile_step = 3000
        x0t, y0t = xs_t.min(), ys_t.min()
        idx_df = pd.DataFrame({
            "i": np.floor((xs_t - x0t) / tile_step).astype(int),
            "j": np.floor((ys_t - y0t) / tile_step).astype(int),
            "k": np.arange(len(pts_l))})
        tiles = []
        for _, grp in idx_df.groupby(["i", "j"]):
            sub = [pts_l[k] for k in grp["k"]]
            tiles.append(unary_union([p.buffer(BUFFER) for p in sub]))
        return unary_union(tiles)

    bins_present = sorted(int(b) for b in rc["radial_bin"].dropna().unique())

    # bounds from binned cells only (excludes NaN-bin outliers)
    binned = rc[rc["radial_bin"].notna()]
    bounds = (binned["cx"].min(), binned["cy"].min(),
              binned["cx"].max(), binned["cy"].max())

    return dict(
        rc=rc,
        cx_c=cx_c, cy_c=cy_c,
        bins_present=bins_present,
        bounds=bounds,
    )


# ── Pre-compute all four cities ────────────────────────────────────────────
city_data = []
for cfg in CITIES:
    print(f"  Building {cfg['city']} ({cfg['tier']}) …")
    city_data.append(_build_city_data(cfg))


def _draw_panel(ax, cfg, data):
    """Draw one city panel onto `ax`."""
    rc           = data["rc"]
    cx_c, cy_c   = data["cx_c"], data["cy_c"]
    bins_present = data["bins_present"]
    bounds       = data["bounds"]   # (minx, miny, maxx, maxy) of binned cells

    # --- axes limits: square, centred on city, with small margin ---
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    side   = max(w, h) * 1.10   # add 5 % each side
    cx_mid = (bounds[0] + bounds[2]) / 2
    cy_mid = (bounds[1] + bounds[3]) / 2
    xmin = cx_mid - side / 2
    xmax = cx_mid + side / 2
    ymin = cy_mid - side / 2
    ymax = cy_mid + side / 2
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # --- cells rendered via RGBA imshow grid ---
    # The project uses a 250 m hex grid whose cell centres are spaced
    # 250 × √3 ≈ 433 m apart (hex row pitch).  Using 250 m as the grid
    # step leaves ~67 % of slots empty → ragged "polka-dot" appearance.
    # Fix: snap each cell centre to the nearest 433 m slot so the grid
    # is fully packed, then display each slot as a solid coloured square.
    xs_c     = rc["cx"].values
    ys_c     = rc["cy"].values
    bins_arr = rc["radial_bin"].values

    # Measure the actual median nearest-neighbour spacing as the step size
    from scipy.spatial import cKDTree as _KDTree
    _tree  = _KDTree(np.column_stack([xs_c, ys_c]))
    _d, _  = _tree.query(np.column_stack([xs_c, ys_c]), k=2)
    STEP   = float(np.median(_d[:, 1]))          # ≈ 433 m for all cities
    HALF   = STEP / 2

    xi = np.round((xs_c - xs_c.min()) / STEP).astype(int)
    yi = np.round((ys_c - ys_c.min()) / STEP).astype(int)
    x0g = float(xs_c.min()) - HALF
    y0g = float(ys_c.min()) - HALF
    nx  = int(xi.max()) + 2
    ny  = int(yi.max()) + 2

    # pre-convert colours
    bin_rgb = {b: np.array(mcolors.to_rgb(BIN_COLORS[b]), dtype=np.float32)
               for b in range(10)}
    na_rgb  = np.array(mcolors.to_rgb(NA_COLOR), dtype=np.float32)

    rgba_grid          = np.ones((ny, nx, 4), dtype=np.float32)
    rgba_grid[:, :, 3] = 0.0   # transparent background

    na_mask = np.isnan(bins_arr.astype(float))
    for b in range(10):
        m = ~na_mask & (bins_arr.astype(float) == b)
        if not m.any(): continue
        rgba_grid[yi[m], xi[m], :3] = bin_rgb[b]
        rgba_grid[yi[m], xi[m],  3] = 1.0
    # NaN-bin cells (u > 95th-pct common support) — leave transparent.
    # They are part of the urban footprint but outside the binning range;
    # rendering them grey inside the boundary would confuse readers.

    ax.imshow(rgba_grid, origin="lower",
              extent=[x0g, x0g + nx * STEP, y0g, y0g + ny * STEP],
              interpolation="nearest", zorder=2, rasterized=True)

    # 3 — primary centre
    ax.plot(cx_c, cy_c, "*", color="#E74C3C", markersize=10, zorder=9,
            markeredgecolor="white", markeredgewidth=0.8)

    # 5 — scale bar below panel (axes-fraction coords, centred)
    _draw_scalebar(ax, xmin, xmax, cfg["tier"])

    # 6 — tier label top-left inside panel (only tier number, small footprint)
    tier  = cfg["tier"]
    city  = cfg["city"]
    country = cfg["country"]
    pop   = rc["Pop2022"].sum()
    pop_s = f"{pop/1e6:.2f} M" if pop >= 1e6 else f"{pop/1e3:.0f} k"
    tier_full = tier.replace("T", "Tier ")
    # Tier label stays inside panel top-left — short text, minimal overlap risk
    ax.text(0.03, 0.97, tier_full, transform=ax.transAxes,
            fontsize=11, fontweight="bold", color="#222222",
            va="top", ha="left", zorder=11,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.75))
    # City / country / pop below the panel, left-aligned, outside map area
    ax.text(0.02, -0.08, f"{city}, {country}  ·  {pop_s}",
            transform=ax.transAxes, fontsize=9, color="#444444",
            va="top", ha="left", clip_on=False)



def _plot_geom(ax, geom, **kw):
    """Plot a shapely geometry (LineString / MultiLineString) onto ax."""
    from shapely.geometry import MultiLineString, GeometryCollection, LineString
    if geom is None or geom.is_empty:
        return
    if geom.geom_type in ("LineString", "LinearRing"):
        xs, ys = geom.xy
        ax.plot(xs, ys, **kw)
    elif geom.geom_type in ("MultiLineString", "GeometryCollection"):
        for g in geom.geoms:
            _plot_geom(ax, g, **kw)
    elif geom.geom_type == "Polygon":
        xs, ys = geom.exterior.xy
        ax.plot(xs, ys, **kw)
    elif geom.geom_type == "MultiPolygon":
        for g in geom.geoms:
            xs, ys = g.exterior.xy
            ax.plot(xs, ys, **kw)


def _draw_scalebar(ax, xmin, xmax, tier=""):
    """Draw a scale bar below the panel (outside map area) in axes-fraction coords.

    Bar is capped at MAX_FRAC of panel width so physical length stays consistent
    across panels of very different city sizes.
    """
    span_km = (xmax - xmin) / 1000
    MAX_FRAC = 0.20   # bar never exceeds 20 % of panel width on screen

    # Pick the largest round number whose bar fits within MAX_FRAC
    sb_km = None
    for candidate in [100, 50, 20, 10, 5, 2, 1, 0.5]:
        frac = (candidate * 1000) / (xmax - xmin)
        if frac <= MAX_FRAC:
            sb_km = candidate
            break
    if sb_km is None:
        sb_km = 0.5   # fallback

    # bar width as fraction of axes width; right-aligned at 98 %
    bar_frac = (sb_km * 1000) / (xmax - xmin)
    x1 = 0.98
    x0 = x1 - bar_frac
    y   = -0.05   # below panel in axes coords
    cap = 0.012   # cap height in axes coords

    tr = ax.transAxes
    for cx in [x0, x1]:
        ax.plot([cx, cx], [y - cap, y + cap], transform=tr,
                color="#333333", linewidth=0.9, zorder=10, clip_on=False)
    ax.plot([x0, x1], [y, y], transform=tr,
            color="#333333", linewidth=1.4, solid_capstyle="butt",
            zorder=10, clip_on=False)

    label = f"{int(sb_km)} km" if sb_km >= 1 else f"{int(sb_km*1000)} m"
    ax.text((x0 + x1) / 2, y + cap * 1.8, label, transform=tr,
            ha="center", va="bottom", fontsize=10, color="#333333",
            zorder=10, clip_on=False)


# ── Figure layout — 2×2 grid ───────────────────────────────────────────────
# T1 T2   (top row)
# T3 T4   (bottom row)
# Colour strip + zone labels on the far right.
print("Drawing figure …")

STRIP_W  = 0.025    # colour strip width (normalised)
LABELS_W = 0.09     # zone label panel width
HGAP     = 0.03     # horizontal gap between panels / legend
VGAP     = 0.10     # vertical gap between rows (extra space for scale bar below panels)
TITLE_H  = 0.04     # space reserved for title above panels

# Two columns of map panels
PANEL_W = (1.0 - STRIP_W - LABELS_W - 3 * HGAP) / 2
PANEL_H = (1.0 - TITLE_H - 3 * VGAP) / 2   # two rows + top/bottom margin

# Row bottoms (y): row 0 = top, row 1 = bottom
row_y = [VGAP * 2 + PANEL_H, VGAP]          # bottom-left corner of each row
col_x = [HGAP, HGAP * 2 + PANEL_W]          # left edges of col 0 and col 1

fig = plt.figure(figsize=(12, 10))
fig.patch.set_facecolor("white")

# panel positions: [(col, row), ...] matching CITIES order T1,T2,T3,T4
panel_positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
axes = []
for (col, row) in panel_positions:
    ax = fig.add_axes([col_x[col], row_y[row], PANEL_W, PANEL_H])
    axes.append(ax)

# draw panels
for ax, cfg, data in zip(axes, CITIES, city_data):
    _draw_panel(ax, cfg, data)

# ── Shared colour strip (right side, vertically centred) ──────────────────
N_SLOTS   = 10
strip_left = 1.0 - LABELS_W - STRIP_W - HGAP
strip_bot  = VGAP
strip_h    = 1.0 - TITLE_H - 2 * VGAP

ax_strip = fig.add_axes([strip_left, strip_bot, STRIP_W, strip_h])
ax_strip.set_xlim(0, 1); ax_strip.set_ylim(0, N_SLOTS)
ax_strip.set_xticks([]); ax_strip.set_yticks([])
for sp in ax_strip.spines.values(): sp.set_visible(False)

all_bins_present = set()
for d in city_data:
    all_bins_present.update(d["bins_present"])

for b in range(N_SLOTS):
    fc = BIN_COLORS[b] if b in all_bins_present else "#DDDDDD"
    ax_strip.add_patch(mpatches.Rectangle(
        (0, b), 1, 1, facecolor=fc, edgecolor="white", linewidth=0.4))
    tc = "white" if b in all_bins_present else "#AAAAAA"
    ax_strip.text(0.5, b + 0.5, str(b), fontsize=9, ha="center",
                  va="center", color=tc, fontweight="bold")

ax_strip.set_title("Bin", fontsize=10, pad=4, color="#444444")

# zone brackets
ax_labels = fig.add_axes([strip_left + STRIP_W, strip_bot, LABELS_W, strip_h])
ax_labels.set_xlim(0, 1); ax_labels.set_ylim(0, N_SLOTS)
ax_labels.set_xticks([]); ax_labels.set_yticks([])
for sp in ax_labels.spines.values(): sp.set_visible(False)

for zone, bins in ZONE_BINS.items():
    y_lo  = min(bins)
    y_hi  = max(bins) + 1
    y_mid = (y_lo + y_hi) / 2
    ax_labels.plot([0.08, 0.08], [y_lo + 0.1, y_hi - 0.1],
                   color=ZONE_COL[zone], lw=2.0)
    ax_labels.text(0.22, y_mid, zone, fontsize=11, fontweight="bold",
                   color=ZONE_COL[zone], va="center", ha="left")

# ── Shared title — flush against top of panels ─────────────────────────────
title_y = strip_bot + strip_h + 0.01
fig.text(0.02, title_y,
         "Population-scaled radial bins across city-size tiers",
         fontsize=12, fontweight="bold", va="bottom", ha="left")

# ── Save ───────────────────────────────────────────────────────────────────
for fmt, dpi in [("pdf", None), ("png", 300)]:
    p = OUT_FIG / f"tier_rings_comparison.{fmt}"
    kw = {"dpi": dpi} if dpi else {}
    fig.savefig(p, bbox_inches="tight", **kw)
    print(f"Saved: {p}")
plt.close(fig)
print("Done.")
