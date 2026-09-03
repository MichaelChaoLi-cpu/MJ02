#!/usr/bin/env python3
"""Spatial NPP-welfare clustering and province coefficient map.

Plan: Improve the exploratory spatial diagnostic without changing any estimates.
Framework: Display the continuous local bivariate Moran statistic alongside the
province-interacted NPP-food-welfare coefficients, with multiplicity-adjusted
significance shown explicitly.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = (
    ROOT
    / "data/exp/analysis/climate-welfare/spatial-cropland-npp-household-welfare"
)
BOUNDARIES = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
COUNTRIES = (
    ROOT
    / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
)
LOCAL = ANALYSIS / "local_bivariate_moran_clusters.csv"
PROVINCES = ANALYSIS / "province_npp_food_coefficients.csv"
GLOBAL_TESTS = ANALYSIS / "global_moran_tests.csv"
OUTPUT = ANALYSIS / "spatial_clustering_and_province_coefficients_revised.png"

MAP_EXTENT = (101.45, 108.45, 9.75, 15.35)
CONTEXT_COUNTRIES = ["Cambodia", "Thailand", "Laos", "Vietnam"]


def add_context(
    ax: plt.Axes,
    countries: gpd.GeoDataFrame,
    provinces: gpd.GeoDataFrame,
    communes: gpd.GeoDataFrame | None = None,
) -> None:
    """Draw consistent geographic context and coordinate guides."""
    countries.plot(
        ax=ax,
        color="#f1eee6",
        edgecolor="#a7a39a",
        linewidth=0.55,
        zorder=0,
    )
    cambodia = countries.loc[countries["ADMIN"].eq("Cambodia")]
    cambodia.plot(
        ax=ax,
        color="#fbfaf6",
        edgecolor="#575757",
        linewidth=0.9,
        zorder=1,
    )
    if communes is not None:
        communes.boundary.plot(
            ax=ax,
            color="#c9c6be",
            linewidth=0.12,
            alpha=0.55,
            zorder=2,
        )
    provinces.boundary.plot(
        ax=ax,
        color="#676767",
        linewidth=0.45,
        alpha=0.9,
        zorder=3,
    )
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_xticks(np.arange(102, 109, 1))
    ax.set_yticks(np.arange(10, 16, 1))
    ax.grid(color="#c8c8c8", linestyle=(0, (2, 3)), linewidth=0.45, alpha=0.65)
    ax.tick_params(axis="both", labelsize=7, length=2.5, color="#666666")
    ax.set_xlabel("Longitude (°E)", fontsize=8, labelpad=5)
    ax.set_ylabel("Latitude (°N)", fontsize=8, labelpad=5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#4c4c4c")
        spine.set_linewidth(0.7)


def add_country_labels(ax: plt.Axes) -> None:
    labels = {
        "THAILAND": (101.72, 13.55),
        "LAO PDR": (107.18, 15.12),
        "VIETNAM": (108.10, 12.65),
    }
    for label, (longitude, latitude) in labels.items():
        ax.text(
            longitude,
            latitude,
            label,
            color="#77736a",
            fontsize=7,
            fontstyle="italic",
            ha="center",
            va="center",
            zorder=8,
        )


def add_panel_heading(ax: plt.Axes, label: str, heading: str) -> None:
    ax.text(
        0.018,
        0.982,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.2},
    )
    ax.text(
        0.072,
        0.982,
        heading,
        transform=ax.transAxes,
        fontsize=9.2,
        fontweight="semibold",
        ha="left",
        va="top",
        zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.2},
    )


def add_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    scalar: plt.cm.ScalarMappable,
    label: str,
) -> None:
    cax = inset_axes(
        ax,
        width="78%",
        height="3.2%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.155, 1.0, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    colorbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.outline.set_linewidth(0.55)
    colorbar.set_label(label, fontsize=8, labelpad=4)


def main() -> None:
    local = pd.read_csv(LOCAL, dtype={"Province Code": str})
    coefficients = pd.read_csv(PROVINCES, dtype={"Province Code": str})
    tests = pd.read_csv(GLOBAL_TESTS)

    communes = gpd.read_file(BOUNDARIES).to_crs("EPSG:4326")
    province_map = communes.dissolve(by="ADM1_PCODE", as_index=False)
    province_map["Province Code"] = province_map["ADM1_PCODE"].str[-2:]
    province_map = province_map.merge(coefficients, on="Province Code", how="left")
    countries = gpd.read_file(COUNTRIES).to_crs("EPSG:4326")
    countries = countries.loc[countries["ADMIN"].isin(CONTEXT_COUNTRIES)].copy()
    points = gpd.GeoDataFrame(
        local,
        geometry=gpd.points_from_xy(local["Point Longitude"], local["Point Latitude"]),
        crs="EPSG:4326",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.45))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.975, bottom=0.19, wspace=0.075)

    # Panel a: show the complete local diagnostic, not only the zero FDR discoveries.
    add_context(axes[0], countries, province_map, communes=communes)
    local_stat = points["Local bivariate Moran statistic"].astype(float)
    local_bound = float(np.quantile(np.abs(local_stat), 0.98))
    local_norm = TwoSlopeNorm(vmin=-local_bound, vcenter=0, vmax=local_bound)
    points.plot(
        ax=axes[0],
        column="Local bivariate Moran statistic",
        cmap="PuOr_r",
        norm=local_norm,
        markersize=9,
        alpha=0.80,
        edgecolor="none",
        zorder=5,
        rasterized=True,
    )
    significant_points = points.loc[
        points["Local BH-adjusted probability value"].astype(float).lt(0.05)
    ]
    if len(significant_points):
        significant_points.plot(
            ax=axes[0],
            facecolor="none",
            edgecolor="black",
            linewidth=0.65,
            markersize=23,
            zorder=7,
        )
    add_country_labels(axes[0])
    add_panel_heading(axes[0], "a", "Local NPP–welfare spatial concordance")
    global_row = tests.loc[tests["Test"].eq("Bivariate NPP-food welfare")].iloc[0]
    axes[0].text(
        0.025,
        0.035,
        (
            f"Global bivariate Moran’s I = {global_row['Moran Statistic']:.3f}  "
            f"(permutation p = {global_row['Permutation Probability Value']:.3f})\n"
            f"Local FDR q < 0.05: {len(significant_points):,}/{len(points):,} villages"
        ),
        transform=axes[0].transAxes,
        fontsize=7.6,
        ha="left",
        va="bottom",
        linespacing=1.35,
        zorder=20,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#a8a8a8",
            "linewidth": 0.55,
            "alpha": 0.92,
        },
    )
    local_scalar = plt.cm.ScalarMappable(norm=local_norm, cmap="PuOr_r")
    add_colorbar(
        fig,
        axes[0],
        local_scalar,
        "Local bivariate Moran statistic (clipped at 98th percentile)",
    )

    # Panel b: coefficients are centered at zero; black borders encode FDR significance.
    add_context(axes[1], countries, province_map)
    values = province_map["Coefficient"].dropna().astype(float)
    coefficient_bound = float(max(abs(values.min()), abs(values.max())))
    coefficient_norm = TwoSlopeNorm(
        vmin=-coefficient_bound,
        vcenter=0,
        vmax=coefficient_bound,
    )
    province_map.plot(
        ax=axes[1],
        column="Coefficient",
        cmap="RdBu_r",
        norm=coefficient_norm,
        edgecolor="#5f5f5f",
        linewidth=0.45,
        missing_kwds={"color": "#dedbd3", "edgecolor": "#77736a"},
        zorder=4,
    )
    significant_provinces = province_map.loc[
        province_map["FDR Significant"].fillna(False).astype(bool)
    ].copy()
    if len(significant_provinces):
        significant_provinces.boundary.plot(
            ax=axes[1], color="#171717", linewidth=1.75, zorder=7
        )
        label_points = significant_provinces.to_crs("EPSG:32648").representative_point()
        label_points = gpd.GeoSeries(label_points, crs="EPSG:32648").to_crs("EPSG:4326")
        for (_, province), point in zip(
            significant_provinces.iterrows(), label_points, strict=True
        ):
            label = province["ADM1_EN"].replace(" ", "\n")
            text = axes[1].text(
                point.x,
                point.y,
                label,
                fontsize=6.5,
                fontweight="semibold",
                color="#1c1c1c",
                ha="center",
                va="center",
                linespacing=0.92,
                zorder=9,
            )
            text.set_path_effects(
                [path_effects.withStroke(linewidth=2.2, foreground="white", alpha=0.90)]
            )
    add_country_labels(axes[1])
    add_panel_heading(axes[1], "b", "Province-specific NPP–food-welfare coefficients")
    legend_items = [
        Line2D(
            [0],
            [0],
            color="#171717",
            linewidth=1.75,
            label="FDR-adjusted q < 0.05",
        ),
        Patch(
            facecolor="#dedbd3",
            edgecolor="#77736a",
            label="Insufficient survey support",
        ),
    ]
    axes[1].legend(
        handles=legend_items,
        loc="lower left",
        bbox_to_anchor=(0.018, 0.018),
        fontsize=7.3,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="#a8a8a8",
        borderpad=0.5,
        handlelength=2.2,
    )
    coefficient_scalar = plt.cm.ScalarMappable(norm=coefficient_norm, cmap="RdBu_r")
    add_colorbar(
        fig,
        axes[1],
        coefficient_scalar,
        "Coefficient per 0.1 kg C m⁻² cropland NPP",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
