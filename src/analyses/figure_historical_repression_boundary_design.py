#!/usr/bin/env python3
"""Historical Repression Boundary Design.

Plan: Map treatment assignment, signed distance, segments, village locations, and frozen support.
Framework: AnaSOP Sections 5.3-5.4, 6.8-6.9, and the boundary-reproduction workflow in Section 7.
No NPP or other outcome column is read.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = (
    ROOT / "data/processed/historical_boundary_annual_spatial_climate_preprocessed.parquet"
)
ZONE_ARCHIVE = (
    ROOT
    / "data/exp/data-preprocessing/historical-boundary-source/"
    "Democratic_Kampuchea_Zones.zip"
)
COMMUNES = ROOT / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
SEGMENTS = (
    ROOT
    / "data/exp/data-preprocessing/historical-boundary-source/boundary_segment_points.csv"
)
OUTPUT = ROOT / "data/results/figures/Figure_historical_repression_boundary_design.png"

DESIGN_COLUMNS = [
    "Village Code",
    "Year",
    "Longitude",
    "Latitude",
    "Historical Repression Side",
    "Higher-Repression Southwest Zone",
    "Signed Distance to Historical Repression Boundary km",
    "Absolute Distance to Historical Repression Boundary km",
    "Historical Boundary Segment",
    "Historical-Boundary Common Support 5 km",
]


def load_geography() -> tuple[gpd.GeoSeries, gpd.GeoSeries, gpd.GeoSeries, gpd.GeoSeries]:
    with tempfile.TemporaryDirectory(prefix="mj02-boundary-figure-") as temp_dir:
        with zipfile.ZipFile(ZONE_ARCHIVE) as archive:
            archive.extractall(temp_dir)
        shapefiles = list(Path(temp_dir).rglob("*.shp"))
        if len(shapefiles) != 1:
            raise RuntimeError(f"Expected one zone shapefile, found {len(shapefiles)}")
        zones = gpd.read_file(shapefiles[0]).to_crs(32648)
    southwest = zones.loc[zones["ZONE_NAME"].eq("Southwest"), "geometry"].union_all()
    west = zones.loc[zones["ZONE_NAME"].eq("West"), "geometry"].union_all()
    communes = gpd.read_file(COMMUNES).to_crs(32648)
    province = communes.loc[communes["pro_code"].astype(str).eq("5"), "geometry"].union_all()
    boundary = southwest.boundary.intersection(west.boundary).intersection(province)
    southwest_clip = southwest.intersection(province)
    west_clip = west.intersection(province)
    support = boundary.buffer(5000).intersection(province)
    return tuple(
        gpd.GeoSeries([geometry], crs=32648).to_crs(4326)
        for geometry in (southwest_clip, west_clip, boundary, support)
    )


def setup_axis(ax: plt.Axes, extent: tuple[float, float, float, float]) -> None:
    min_x, max_x, min_y, max_y = extent
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="#D9D9D9", linewidth=0.55, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")


def main() -> None:
    design = pd.read_parquet(PANEL_PATH, columns=DESIGN_COLUMNS)
    design = design.loc[design["Year"].eq(design["Year"].min())].copy()
    villages = gpd.GeoDataFrame(
        design,
        geometry=gpd.points_from_xy(design["Longitude"], design["Latitude"], crs=4326),
        crs=4326,
    )
    southwest, west, boundary, support = load_geography()
    province = southwest.union(west).boundary
    support_villages = villages.loc[
        villages["Historical-Boundary Common Support 5 km"].eq(1)
    ].copy()
    segment_table = pd.read_csv(SEGMENTS)
    segment_points = gpd.GeoDataFrame(
        segment_table,
        geometry=gpd.points_from_xy(
            segment_table["Easting EPSG 32648"],
            segment_table["Northing EPSG 32648"],
            crs=32648,
        ),
        crs=32648,
    ).to_crs(4326)

    bounds = southwest.union(west).total_bounds
    x_pad = (bounds[2] - bounds[0]) * 0.035
    y_pad = (bounds[3] - bounds[1]) * 0.035
    overview_extent = (
        bounds[0] - x_pad,
        bounds[2] + x_pad,
        bounds[1] - y_pad,
        bounds[3] + y_pad,
    )
    local_bounds = support.total_bounds
    local_x_pad = (local_bounds[2] - local_bounds[0]) * 0.05
    local_y_pad = (local_bounds[3] - local_bounds[1]) * 0.15
    local_extent = (
        local_bounds[0] - local_x_pad,
        local_bounds[2] + local_x_pad,
        local_bounds[1] - local_y_pad,
        local_bounds[3] + local_y_pad,
    )
    colors = {"Southwest": "#C94C35", "West": "#3C78B5"}

    fig = plt.figure(figsize=(13.4, 7.2))
    grid = GridSpec(
        2,
        4,
        figure=fig,
        width_ratios=(1, 1, 1, 1),
        height_ratios=(1.25, 0.75),
        wspace=0.34,
        hspace=0.34,
    )
    ax_a = fig.add_subplot(grid[:, :2])
    ax_b = fig.add_subplot(grid[0, 2:])
    ax_c = fig.add_subplot(grid[1, 2])
    ax_d = fig.add_subplot(grid[1, 3])

    setup_axis(ax_a, overview_extent)
    setup_axis(ax_b, local_extent)
    ax_b.set_aspect("equal", adjustable="datalim")
    province.plot(ax=ax_a, color="#252525", linewidth=0.9, zorder=4)
    province.plot(ax=ax_b, color="#252525", linewidth=0.9, zorder=4)

    southwest.plot(ax=ax_a, color="#F2B8AA", edgecolor="none", alpha=0.90)
    west.plot(ax=ax_a, color="#AFCBE5", edgecolor="none", alpha=0.90)
    support.plot(ax=ax_a, color="#F7E6A6", edgecolor="none", alpha=0.52, zorder=2)
    villages.plot(ax=ax_a, color="#3A3A3A", markersize=3.2, alpha=0.32, zorder=3)
    boundary.plot(ax=ax_a, color="#111111", linewidth=2.0, zorder=5)
    ax_a.legend(
        handles=[
            Patch(facecolor="#F2B8AA", label="Southwest (higher repression)"),
            Patch(facecolor="#AFCBE5", label="West (comparison)"),
            Patch(facecolor="#F7E6A6", label="5 km boundary corridor"),
            Line2D([0], [0], color="#111111", linewidth=2, label="Historical boundary"),
        ],
        loc="lower left",
        frameon=True,
        fontsize=7.2,
    )

    support.plot(ax=ax_b, color="#F0F0F0", edgecolor="#777777", linewidth=0.7)
    boundary.plot(ax=ax_b, color="#111111", linewidth=2.1, zorder=4)
    for side, group in support_villages.groupby("Historical Repression Side", observed=True):
        group.plot(
            ax=ax_b,
            color=colors[side],
            markersize=20,
            alpha=0.80,
            label=f"{side}: {len(group)} villages",
            zorder=5,
        )
    segment_points.plot(
        ax=ax_b,
        marker="X",
        color="#111111",
        edgecolor="white",
        linewidth=0.5,
        markersize=48,
        zorder=6,
        label="Boundary segment anchors",
    )
    ax_b.legend(loc="upper left", frameon=True, fontsize=7.4, ncol=3)

    within_30 = villages.loc[
        villages["Absolute Distance to Historical Repression Boundary km"].le(30)
    ]
    distance_bins = np.arange(-30, 32, 2)
    ax_c.axvspan(-5, 5, color="#F7E6A6", alpha=0.55, zorder=0)
    for side in ("West", "Southwest"):
        values = within_30.loc[
            within_30["Historical Repression Side"].eq(side),
            "Signed Distance to Historical Repression Boundary km",
        ]
        ax_c.hist(
            values,
            bins=distance_bins,
            color=colors[side],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
            label=side,
        )
    ax_c.axvline(0, color="#111111", linewidth=1.4)
    ax_c.axvline(-5, color="#777777", linewidth=0.8, linestyle="--")
    ax_c.axvline(5, color="#777777", linewidth=0.8, linestyle="--")
    ax_c.set_xlabel("Signed distance (km)")
    ax_c.set_ylabel("Villages per 2 km bin")
    ax_c.set_xlim(-30, 30)
    ax_c.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax_c.legend(
        handles=[
            Patch(facecolor=colors["West"], label="West"),
            Patch(facecolor=colors["Southwest"], label="Southwest"),
            Patch(facecolor="#F7E6A6", label="Primary window"),
        ],
        loc="upper left",
        frameon=True,
        fontsize=7.0,
    )

    counts = (
        support_villages.groupby(
            ["Historical Boundary Segment", "Historical Repression Side"], observed=True
        )
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(1, 6), columns=["West", "Southwest"], fill_value=0)
    )
    positions = np.arange(1, 6)
    width = 0.36
    for offset, side in ((-width / 2, "West"), (width / 2, "Southwest")):
        bars = ax_d.bar(
            positions + offset,
            counts[side],
            width=width,
            color=colors[side],
            alpha=0.86,
            label=side,
        )
        ax_d.bar_label(bars, padding=2, fontsize=7)
    ax_d.set_xlabel("Boundary segment")
    ax_d.set_ylabel("Villages within 5 km")
    ax_d.set_xticks(positions)
    ax_d.set_ylim(0, counts.to_numpy().max() * 1.18)
    ax_d.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax_d.legend(loc="upper left", frameon=True, fontsize=7.2)

    for ax in (ax_c, ax_d):
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color("#333333")

    for label, ax in zip("abcd", (ax_a, ax_b, ax_c, ax_d), strict=True):
        ax.text(
            -0.11,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(
        f"Outcome columns read: none; 5 km villages={len(support_villages)}; "
        f"Southwest={int(support_villages['Higher-Repression Southwest Zone'].sum())}; "
        f"West={int((1 - support_villages['Higher-Repression Southwest Zone']).sum())}"
    )


if __name__ == "__main__":
    main()
