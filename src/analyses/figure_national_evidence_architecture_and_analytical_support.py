#!/usr/bin/env python3
"""National Evidence Architecture and Analytical Support.

Plan: Establish the nationwide climate-village frame, the nested linked CSES
sample, survey-wave support, and the two parallel evidence paths.
Framework: AnaSOP Sections 5-7 linkage audit and the approved non-mediation
interpretation.  The ecological and household paths are shown separately.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLIMATE = ROOT / "data/processed/cambodia_national_005deg_climate_cells_preprocessed.parquet"
VILLAGES = ROOT / "data/processed/cambodia_public_village_points_preprocessed.parquet"
CSES = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
BOUNDARY = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/figures/Figure_national_evidence_architecture_and_analytical_support.png"

NAVY = "#264653"
BLUE = "#2a6f97"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
GOLD = "#e9c46a"
LIGHT = "#e8ecef"
GRAY = "#a7adb2"


def add_panel_header(
    ax: plt.Axes,
    label: str,
    label_x: float = -0.10,
) -> None:
    ax.text(label_x, 1.055, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")


def style_map(
    ax: plt.Axes,
    boundary: gpd.GeoDataFrame,
    provinces: gpd.GeoDataFrame,
    neighbors: gpd.GeoDataFrame,
) -> None:
    neighbors.plot(
        ax=ax, facecolor="#f1f2f1", edgecolor="#a6adb2",
        linewidth=0.55, zorder=-2,
    )
    provinces.boundary.plot(ax=ax, color="#9ba3a8", linewidth=0.32, zorder=3)
    boundary.boundary.plot(ax=ax, color="#4f5960", linewidth=0.78, zorder=4)
    country_labels = {
        "Thailand": (102.10, 13.15),
        "Laos": (105.70, 14.72),
        "Vietnam": (107.72, 12.55),
    }
    for name, (longitude, latitude) in country_labels.items():
        ax.text(
            longitude, latitude, name, fontsize=8, color="#737b80",
            fontstyle="italic", ha="center", va="center", zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.8},
        )
    ax.set_xlim(101.75, 108.15)
    ax.set_ylim(10.05, 14.95)
    ax.set_xticks(np.arange(102, 109, 1))
    ax.set_yticks(np.arange(11, 15, 1))
    ax.grid(color="#d9dde0", linewidth=0.55, linestyle="--", zorder=0)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#59636b")
        spine.set_linewidth(0.75)


def main() -> None:
    climate = pd.read_parquet(CLIMATE)
    villages = gpd.read_parquet(VILLAGES).to_crs(4326)
    communes = gpd.read_file(BOUNDARY).to_crs(4326)
    boundary = communes.dissolve()
    provinces = communes.dissolve(by="ADM1_PCODE").reset_index()
    countries = gpd.read_file(COUNTRIES).to_crs(4326)
    neighbors = countries.loc[countries["ADMIN"].isin(["Thailand", "Laos", "Vietnam"])].copy()
    cses = pd.read_parquet(
        CSES,
        columns=[
            "Survey Year", "Household ID", "Village Code", "Climate Ecology Link Available",
            "National Village Point ID", "Point Longitude", "Point Latitude",
        ],
    )
    linked = cses.loc[cses["Climate Ecology Link Available"].eq(1)].copy()
    linked_points = linked.drop_duplicates("National Village Point ID")

    coverage = (
        cses.groupby("Survey Year", observed=True)
        .agg(
            Total_households=("Household ID", "size"),
            Linked_households=("Climate Ecology Link Available", "sum"),
            Total_villages=("Village Code", "nunique"),
        )
        .reset_index()
    )
    coverage["Unlinked households"] = coverage["Total_households"] - coverage["Linked_households"]
    coverage["Linked share"] = coverage["Linked_households"] / coverage["Total_households"]

    fig = plt.figure(figsize=(13.6, 9.6), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [["a", "b"], ["c", "c"]],
        height_ratios=[1.08, 0.92],
    )

    # a: nationwide measurement frame
    ax = axes["a"]
    ax.scatter(
        climate["Climate Cell Longitude"], climate["Climate Cell Latitude"],
        s=1.5, marker="s", color="#c9d6df", alpha=0.55, linewidths=0, zorder=1,
    )
    ax.scatter(
        villages["Point Longitude"], villages["Point Latitude"],
        s=2.0, color=TEAL, alpha=0.60, linewidths=0, zorder=2,
    )
    style_map(ax, boundary, provinces, neighbors)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#c9d6df",
                   markeredgecolor="none", markersize=6, label=f"Climate cells ({len(climate):,})"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=TEAL,
                   markeredgecolor="none", markersize=5, label=f"National villages ({len(villages):,})"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8.5,
    )
    add_panel_header(ax, "a")

    # b: linked survey support nested in the national frame
    ax = axes["b"]
    ax.scatter(
        villages["Point Longitude"], villages["Point Latitude"],
        s=2.0, color="#d9dde0", alpha=0.60, linewidths=0, zorder=1,
    )
    ax.scatter(
        linked_points["Point Longitude"], linked_points["Point Latitude"],
        s=5.0, color=ORANGE, alpha=0.80, linewidths=0, zorder=2,
    )
    style_map(ax, boundary, provinces, neighbors)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#d9dde0",
                   markeredgecolor="none", markersize=5, label="Other national villages"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE,
                   markeredgecolor="none", markersize=5,
                   label=f"CSES-linked villages ({linked_points['National Village Point ID'].nunique():,})"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8.5,
    )
    add_panel_header(ax, "b")

    # c: wave-level household linkage
    ax = axes["c"]
    x = np.arange(len(coverage))
    ax.bar(x, coverage["Linked_households"], color=BLUE, width=0.72, label="Linked")
    ax.bar(
        x, coverage["Unlinked households"], bottom=coverage["Linked_households"],
        color=LIGHT, edgecolor="#c8cdd1", linewidth=0.5, width=0.72, label="Unlinked",
    )
    maximum = float(coverage["Total_households"].max())
    for xpos, total, share in zip(x, coverage["Total_households"], coverage["Linked share"]):
        ax.text(xpos, total + maximum * 0.022, f"{share:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, coverage["Survey Year"].astype(str), rotation=40, ha="right")
    ax.set_ylabel("Households")
    ax.set_ylim(0, maximum * 1.14)
    ax.grid(axis="y", color="#d9dde0", linewidth=0.6, linestyle="--")
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    add_panel_header(
        ax,
        "c",
        label_x=0.0,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(
        f"Climate cells={len(climate):,}; national villages={len(villages):,}; "
        f"linked households={len(linked):,}; linked villages="
        f"{linked_points['National Village Point ID'].nunique():,}"
    )


if __name__ == "__main__":
    main()
