#!/usr/bin/env python3
"""National Geography of Climate Exposure and Cropland NPP.

Plan: Show the 2001-2021 national village geography of absolute heat,
extreme rainfall, consecutive dry days, and strict-cropland NPP.
Framework: AnaSOP Section 7 descriptive spatial audit preceding the Stage-1
fixed-effect model. Values are village-level temporal means in natural units;
the maps contain no regression coefficients or causal effects.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/cses_public_village_cropland_npp_absolute_climate_panel_preprocessed.parquet"
COMMUNES = ROOT / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
OUTPUT = ROOT / "data/results/figures/Figure_national_geography_of_climate_exposure_and_cropland_npp.png"

LONGITUDE = "Point Longitude"
LATITUDE = "Point Latitude"
HEAT = "Village Buffer Mean Annual Heat Days at or Above 35 C"
RX5DAY = "Village Buffer Mean Annual Maximum Consecutive Five-Day Precipitation Rx5day mm"
DRY_DAYS = "Village Buffer Mean Annual Maximum Consecutive Dry Days Below 1 mm"
NPP = "Annual Strict-Cropland Mean NPP kg C per m2"

DARK_GRAY = "#4D5960"
MID_GRAY = "#A8B0B4"
LIGHT_LAND = "#F1F3F2"
CAMBODIA_FILL = "#FFFDF7"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=30,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.4},
    )


def degree_e(value: float, _: int) -> str:
    return f"{value:.0f}°E"


def degree_n(value: float, _: int) -> str:
    return f"{value:.0f}°N"


def style_map(
    ax: plt.Axes,
    region: gpd.GeoDataFrame,
    cambodia: gpd.GeoDataFrame,
    provinces: gpd.GeoDataFrame,
) -> None:
    region.plot(ax=ax, facecolor=LIGHT_LAND, edgecolor="#AAB2B6", linewidth=0.50, zorder=0)
    cambodia.plot(ax=ax, facecolor=CAMBODIA_FILL, edgecolor=DARK_GRAY, linewidth=0.85, zorder=1)
    provinces.boundary.plot(ax=ax, color="#A0A9AE", linewidth=0.34, zorder=3)

    country_labels = {
        "Thailand": (102.05, 13.15),
        "Laos": (105.70, 14.86),
        "Vietnam": (107.85, 12.55),
        "Cambodia": (104.85, 12.62),
    }
    for name, (longitude, latitude) in country_labels.items():
        ax.text(
            longitude,
            latitude,
            name,
            ha="center",
            va="center",
            fontsize=7.4 if name != "Cambodia" else 7.8,
            color="#7A8388" if name != "Cambodia" else "#314D60",
            fontstyle="italic" if name != "Cambodia" else "normal",
            zorder=5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.58, "pad": 0.6},
        )

    ax.set_xlim(101.75, 108.15)
    ax.set_ylim(9.95, 15.25)
    ax.set_xticks(np.arange(102, 109, 1))
    ax.set_yticks(np.arange(10, 16, 1))
    ax.xaxis.set_major_formatter(FuncFormatter(degree_e))
    ax.yaxis.set_major_formatter(FuncFormatter(degree_n))
    ax.grid(color="#D9DEE1", linewidth=0.52, linestyle="--", zorder=-1)
    ax.tick_params(labelsize=7.8, colors=DARK_GRAY, length=3)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#667279")
        spine.set_linewidth(0.72)


def map_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    region: gpd.GeoDataFrame,
    cambodia: gpd.GeoDataFrame,
    provinces: gpd.GeoDataFrame,
    variable: str,
    cmap: str,
    colorbar_label: str,
    label: str,
) -> None:
    observed = frame.loc[frame[variable].notna()].copy()
    missing = frame.loc[frame[variable].isna()].copy()
    lower, upper = observed[variable].quantile([0.02, 0.98])
    norm = Normalize(vmin=float(lower), vmax=float(upper), clip=True)

    if not missing.empty:
        ax.scatter(
            missing[LONGITUDE],
            missing[LATITUDE],
            s=7.0,
            color=MID_GRAY,
            alpha=0.78,
            linewidths=0,
            zorder=2,
        )
    points = ax.scatter(
        observed[LONGITUDE],
        observed[LATITUDE],
        c=observed[variable],
        cmap=cmap,
        norm=norm,
        s=7.5,
        alpha=0.94,
        linewidths=0,
        zorder=2,
        rasterized=True,
    )
    style_map(ax, region, cambodia, provinces)
    colorbar = ax.figure.colorbar(
        points,
        ax=ax,
        orientation="vertical",
        pad=0.014,
        shrink=0.82,
        aspect=24,
        extend="both",
    )
    colorbar.set_label(colorbar_label, fontsize=8.1, color=DARK_GRAY, labelpad=5)
    colorbar.locator = MaxNLocator(nbins=5)
    colorbar.update_ticks()
    colorbar.ax.tick_params(labelsize=7.5, colors=DARK_GRAY, length=2.5)
    colorbar.outline.set_linewidth(0.45)
    panel_label(ax, label)


def main() -> None:
    columns = ["National Village Point ID", LONGITUDE, LATITUDE, "Year", HEAT, RX5DAY, DRY_DAYS, NPP]
    panel = pd.read_parquet(PANEL, columns=columns)
    panel = panel.loc[panel["Year"].between(2001, 2021)].copy()
    frame = (
        panel.groupby(["National Village Point ID", LONGITUDE, LATITUDE], as_index=False)[
            [HEAT, RX5DAY, DRY_DAYS, NPP]
        ]
        .mean()
        .sort_values("National Village Point ID")
    )

    communes = gpd.read_file(COMMUNES).to_crs(4326)
    provinces = communes.dissolve(by="pro_code").reset_index()
    cambodia = communes.dissolve()
    countries = gpd.read_file(COUNTRIES).to_crs(4326)
    region = countries.cx[101.4:108.7, 9.6:15.6].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 10.6), facecolor="white")
    fig.subplots_adjust(left=0.045, right=0.965, top=0.985, bottom=0.055, wspace=0.16, hspace=0.14)

    specifications = [
        (axes[0, 0], HEAT, "YlOrRd", "Mean annual days at or above 35°C", "a"),
        (axes[0, 1], RX5DAY, "Blues", "Mean annual Rx5day (mm)", "b"),
        (axes[1, 0], DRY_DAYS, "YlOrBr", "Mean annual maximum dry spell (days)", "c"),
        (axes[1, 1], NPP, "YlGn", "Mean annual cropland NPP (kg C m⁻² yr⁻¹)", "d"),
    ]
    for ax, variable, cmap, colorbar_label, label in specifications:
        map_panel(
            ax,
            frame,
            region,
            cambodia,
            provinces,
            variable,
            cmap,
            colorbar_label,
            label,
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"period=2001-2021; villages={len(frame):,}; NPP_missing={frame[NPP].isna().sum():,}")
    for variable in [HEAT, RX5DAY, DRY_DAYS, NPP]:
        values = frame[variable].dropna()
        print(
            f"{variable}: mean={values.mean():.3f}; "
            f"p02={values.quantile(0.02):.3f}; p98={values.quantile(0.98):.3f}"
        )


if __name__ == "__main__":
    main()
