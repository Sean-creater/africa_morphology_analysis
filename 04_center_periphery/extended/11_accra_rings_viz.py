"""
11_accra_rings_viz.py
=====================
Publication-quality illustration of population-scaled radial bins — Accra.

Figure caption (to be used outside the figure):
    Illustration of population-scaled radial binning for Accra, Ghana (T4,
    Pop ≈ 4.7 M).  Each 250 m grid cell is assigned a bin (0 = core,
    7 = periphery for Accra) based on its scaled distance
    u = d / Pop^{0.4534} from the population-weighted primary centre.
    Ten equal-width bins are defined on the pooled u distribution across
    all 561 cities; bins 0–1 form the Core, 2–5 the Mid zone, and 6–9
    the Outer zone.  Dashed arcs show the bin-boundary circles clipped to
    the city footprint.  Inset: population density (Pop2022 per cell)
    within the core zone, illustrating how the primary centre is defined
    as the population-weighted centroid of the largest contiguous
    high-density cluster.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.patheffects as mpe
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
from shapely.geometry import Point as ShpPoint
from scipy.spatial import cKDTree
from pathlib import Path
from datetime import date

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size":        9,
    "axes.linewidth":   0.6,
    "pdf.fonttype":     42,   # embeds fonts in PDF
    "ps.fonttype":      42,
})

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
OUT_TABLES = ROOT / "outputs" / "tables"
FIG_DATE   = date.today().strftime("%Y-%m-%d")
OUT_FIG    = ROOT / "outputs" / "figures" / FIG_DATE
OUT_FIG.mkdir(parents=True, exist_ok=True)

ACCRA_SHP  = Path("/home/sjs96_file_share/Momepy_workflow/Metrix/"
                  "data_cleaning_20260224/Crowley-poster/Accra.shp")
CITY_BOUND = Path("/home/sjs96_file_share/Momepy_workflow/Metrix/"
                  "data_cleaning_20260224/Crowley-poster/city_boundaries.shp")

CRS = "EPSG:32630"   # UTM zone 30N — Ghana

# ── Colour scheme ──────────────────────────────────────────────────────────
# Core (0–1): purple  |  Mid (2–5): mint→teal  |  Outer (6–9): pale→amber
# Within each zone: lighter for lower bins, darker for higher bins
# Palette shared with 12_tier_rings_comparison.py
BIN_COLORS = {
    0: "#C39BD3",
    1: "#7D3C98",
    2: "#82E0AA",
    3: "#27AE60",
    4: "#1E8449",
    5: "#145A32",
    6: "#FAD7A0",
    7: "#F39C12",
    8: "#D35400",
    9: "#784212",
}
ZONE_COL = {"Core": "#7D3C98", "Mid": "#1E8449", "Outer": "#D35400"}
ZONE_BINS = {"Core": [0,1], "Mid": [2,3,4,5], "Outer": [6,7,8,9]}

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data …")
rc       = pd.read_parquet(OUT_TABLES / "radial_cells.parquet")
acc      = rc[(rc["city"] == "Accra") & (rc["country"] == "Ghana")].copy()
acc_grid = gpd.read_file(ACCRA_SHP).to_crs(CRS)
boundary = gpd.read_file(CITY_BOUND)
boundary = boundary[boundary["city"] == "Accra"].to_crs(CRS)

# ── Nearest-neighbour merge ────────────────────────────────────────────────
print("Merging bins …")
acc_grid["cx_m"] = acc_grid.geometry.centroid.x
acc_grid["cy_m"] = acc_grid.geometry.centroid.y

tree = cKDTree(acc[["cx", "cy"]].values)
dists, idxs = tree.query(acc_grid[["cx_m", "cy_m"]].values, k=1)
matched = dists < 200

acc_grid["radial_bin"]     = np.where(matched, acc.iloc[idxs]["radial_bin"].values,     np.nan)
acc_grid["dist_m_primary"] = np.where(matched, acc.iloc[idxs]["dist_m_primary"].values, np.nan)
acc_grid["Pop2022"]        = np.where(matched, acc.iloc[idxs]["Pop2022"].values,         np.nan)
acc_grid["radial_bin"]     = pd.to_numeric(acc_grid["radial_bin"], errors="coerce")
print(f"  {matched.sum()}/{len(acc_grid)} cells matched")

# ── Primary centre ─────────────────────────────────────────────────────────
cx_c = float(acc.loc[acc["dist_m_primary"].idxmin(), "cx"])
cy_c = float(acc.loc[acc["dist_m_primary"].idxmin(), "cy"])

# ── Bin edge distances ─────────────────────────────────────────────────────
bin_max_dist = {}   # metres
for b in range(10):
    sub = acc[acc["radial_bin"] == b]
    if len(sub):
        bin_max_dist[b] = float(sub["dist_m_primary"].max())

# ── Clipped radial arcs ────────────────────────────────────────────────────
print("Clipping arcs …")
city_union = boundary.geometry.union_all()
clipped_arcs = {}
for b, r in bin_max_dist.items():
    arc = ShpPoint(cx_c, cy_c).buffer(r, resolution=512).exterior
    clp = arc.intersection(city_union)
    if not clp.is_empty:
        clipped_arcs[b] = clp

# ── Figure layout ──────────────────────────────────────────────────────────
print("Drawing …")
fig = plt.figure(figsize=(9.5, 8.0))
ax = fig.add_axes([0.01, 0.01, 0.84, 0.95])
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# 1 — cells coloured by bin
bins_present = sorted(int(b) for b in acc_grid["radial_bin"].dropna().unique())
for b in bins_present:
    acc_grid[acc_grid["radial_bin"] == b].plot(
        ax=ax, color=BIN_COLORS[b], linewidth=0, alpha=0.88, zorder=2)

# cells outside common support — same hue as outer zone, very light
acc_grid[acc_grid["radial_bin"].isna()].plot(
    ax=ax, color="#E8E4DC", linewidth=0, alpha=0.80, zorder=2)

# 2 — clipped arcs (very thin, grey, dashed)
for b, arc in clipped_arcs.items():
    gpd.GeoDataFrame(geometry=[arc], crs=CRS).plot(
        ax=ax, color="#444444", linewidth=0.55,
        linestyle=(0, (5, 4)), alpha=0.50, zorder=4)

# 3 — city boundary
boundary.plot(ax=ax, facecolor="none", edgecolor="#111111",
              linewidth=1.4, zorder=5)

# 4 — primary centre
ax.plot(cx_c, cy_c, "*", color="white", markersize=11, zorder=9,
        markeredgecolor="#222222", markeredgewidth=0.8)

# ── Axes limits tight to footprint ────────────────────────────────────────
bounds = boundary.total_bounds
margin = 2500
ax.set_xlim(bounds[0] - margin, bounds[2] + margin)
ax.set_ylim(bounds[1] - margin, bounds[3] + margin)
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)


# ── Zone labels — placed at mass centroid of each zone's cells ────────────
for zone, bins in ZONE_BINS.items():
    present = [b for b in bins if b in bins_present]
    if not present:
        continue
    sub = acc[acc["radial_bin"].isin(present)]
    lx = float(sub["cx"].mean())
    ly = float(sub["cy"].mean())
    ax.text(lx, ly, zone, fontsize=12, fontweight="bold",
            color="white", ha="center", va="center", zorder=10,
            path_effects=[mpe.withStroke(linewidth=3.5, foreground=ZONE_COL[zone])])

# ── Centre: bright marker + clear annotation ──────────────────────────────
ax.plot(cx_c, cy_c, "*", color="#E74C3C", markersize=14, zorder=9,
        markeredgecolor="white", markeredgewidth=1.0)
ax.annotate("Population-weighted\nprimary centre",
            xy=(cx_c, cy_c),
            xytext=(xmin + (xmax - xmin) * 0.78, ymin + (ymax - ymin) * 0.10),
            fontsize=8.5, color="#111111", va="center", ha="center",
            fontweight="bold", linespacing=1.5, zorder=11,
            arrowprops=dict(arrowstyle="-|>", color="#444444",
                            lw=0.9, mutation_scale=8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#CCCCCC", alpha=0.95, lw=0.7))

# ── Scale bar ─────────────────────────────────────────────────────────────
sb_len = 10000
sb_x   = xmin + (xmax - xmin) * 0.04
sb_y   = ymin + (ymax - ymin) * 0.04
for cap_x in [sb_x, sb_x + sb_len]:
    ax.plot([cap_x, cap_x], [sb_y - 400, sb_y + 400],
            color="#333333", linewidth=0.9, zorder=10)
ax.plot([sb_x, sb_x + sb_len], [sb_y, sb_y],
        color="#333333", linewidth=1.5, solid_capstyle="butt", zorder=10)
ax.text(sb_x + sb_len / 2, sb_y + 1000, "10 km",
        ha="center", va="bottom", fontsize=8, color="#333333", zorder=10)

# ── Colour strip — separate axes on the right ─────────────────────────────
# Colour strip: always show all 10 slots (0–9), grey for absent bins
N_SLOTS = 10
ax_strip = fig.add_axes([0.860, 0.15, 0.024, 0.72])
ax_strip.set_xlim(0, 1)
ax_strip.set_ylim(0, N_SLOTS)
ax_strip.set_xticks([]); ax_strip.set_yticks([])
for sp in ax_strip.spines.values():
    sp.set_visible(False)

for b in range(N_SLOTS):
    fc = BIN_COLORS[b] if b in bins_present else "#DDDDDD"
    ax_strip.add_patch(mpatches.Rectangle(
        (0, b), 1, 1, facecolor=fc, edgecolor="white", linewidth=0.4))
    tc = "white" if b in bins_present else "#AAAAAA"
    ax_strip.text(0.5, b + 0.5, str(b),
                  fontsize=7, ha="center", va="center",
                  color=tc, fontweight="bold")

# zone bracket + label
ax_labels = fig.add_axes([0.887, 0.15, 0.11, 0.72])
ax_labels.set_xlim(0, 1); ax_labels.set_ylim(0, N_SLOTS)
ax_labels.set_xticks([]); ax_labels.set_yticks([])
for sp in ax_labels.spines.values(): sp.set_visible(False)

for zone, bins in ZONE_BINS.items():
    y_lo = min(bins)
    y_hi = max(bins) + 1
    y_mid = (y_lo + y_hi) / 2
    ax_labels.plot([0.08, 0.08], [y_lo + 0.08, y_hi - 0.08],
                   color=ZONE_COL[zone], lw=1.8)
    ax_labels.text(0.22, y_mid, zone, fontsize=8.5, fontweight="bold",
                   color=ZONE_COL[zone], va="center", ha="left")

ax_strip.set_title("Bin", fontsize=7.5, pad=4, color="#444444")


# ── Title ─────────────────────────────────────────────────────────────────
ax.set_title("Accra — population-scaled radial bins",
             fontsize=11, fontweight="bold", pad=8, loc="left")

# ── Save ───────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.2)
for fmt, dpi in [("pdf", None), ("png", 300)]:
    p = OUT_FIG / f"accra_rings.{fmt}"
    kw = {"dpi": dpi} if dpi else {}
    fig.savefig(p, bbox_inches="tight", **kw)
    print(f"Saved: {p}")
plt.close(fig)
print("Done.")
