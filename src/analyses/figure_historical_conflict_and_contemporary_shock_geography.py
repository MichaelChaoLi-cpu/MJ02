#!/usr/bin/env python3
"""Historical Conflict and Contemporary Shock Geography.

Plan: Show where historical conflict overlaps with drought, local food-price
pressure, and satellite-observed inundation across Cambodia.
Framework: AnaSOP Sections 5.1 and 5.5, Section 6.2, and the first workflow
step in Section 7. This is a spatial-support diagnostic, not a causal result.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
BOUNDARY_PATH = ROOT / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
HOUSEHOLD_PATH = (
    ROOT / "data/processed/direction3_household_conflict_shock_preprocessed.parquet"
)
CONFLICT_PATH = ROOT / "data/processed/historical_conflict_commune_preprocessed.parquet"
OUTPUT_PATH = (
    ROOT
    / "data/results/figures/Figure_historical_conflict_and_contemporary_shock_geography.png"
)

YEAR = "Survey Year"
PSU = "PSU"
RESOLUTION = "Climate Geography Resolution"
GEOGRAPHY = "Climate Geography Code"
PROVINCE = "Province Code Component"
PRICE_DATE = "Price Exposure Date"
CONFLICT = "Log Bombing Unique Locations per 100 km2"
SPI12 = "Interview Month SPI 12 Month"
PRICE = "Local Relative Log Wholesale Rice Price"
FLOOD = "Survey Year Maximum Flooded Geography Share"


def read_map_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    communes = gpd.read_file(BOUNDARY_PATH).to_crs(4326)
    communes["Geography Code"] = (
        communes["com_code"].astype(int).astype(str).str.zfill(6)
    )
    communes["Province Code"] = (
        communes["pro_code"].astype(int).astype(str).str.zfill(2)
    )
    assert communes["Geography Code"].is_unique
    assert len(communes) == 1_633

    conflict = pd.read_parquet(
        CONFLICT_PATH, columns=["Geography Code", CONFLICT]
    )
    assert conflict["Geography Code"].is_unique
    conflict_map = communes.merge(
        conflict, on="Geography Code", how="left", validate="one_to_one"
    )
    assert conflict_map[CONFLICT].notna().all()

    household_columns = [
        YEAR,
        PSU,
        RESOLUTION,
        GEOGRAPHY,
        PROVINCE,
        PRICE_DATE,
        SPI12,
        PRICE,
        FLOOD,
    ]
    households = pd.read_parquet(HOUSEHOLD_PATH, columns=household_columns)
    psu_wave = households.drop_duplicates([YEAR, PSU]).copy()
    assert psu_wave[PSU].nunique() == 1_220
    assert len(psu_wave) == 5_616

    commune_psu_wave = psu_wave.loc[psu_wave[RESOLUTION].eq("commune")].copy()
    drought = (
        commune_psu_wave.groupby(GEOGRAPHY, as_index=False)[SPI12]
        .min()
        .rename(columns={GEOGRAPHY: "Geography Code", SPI12: "Lowest linked SPI-12"})
    )
    drought["Drought intensity"] = (-drought["Lowest linked SPI-12"]).clip(lower=0)
    drought_map = communes.merge(
        drought[["Geography Code", "Drought intensity"]],
        on="Geography Code",
        how="left",
        validate="one_to_one",
    )

    flood = (
        commune_psu_wave.groupby(GEOGRAPHY, as_index=False)[FLOOD]
        .max()
        .rename(columns={GEOGRAPHY: "Geography Code", FLOOD: "Maximum inundated share"})
    )
    flood_map = communes.merge(
        flood, on="Geography Code", how="left", validate="one_to_one"
    )

    price = (
        psu_wave.dropna(subset=[PRICE])
        .drop_duplicates([PROVINCE, PRICE_DATE])
        .groupby(PROVINCE, as_index=False)[PRICE]
        .max()
        .rename(columns={PROVINCE: "Province Code", PRICE: "Maximum local price pressure"})
    )
    provinces = communes.dissolve(by="Province Code", as_index=False)
    price_map = provinces.merge(
        price, on="Province Code", how="left", validate="one_to_one"
    )

    assert drought_map["Drought intensity"].notna().sum() == 1_402
    assert flood_map["Maximum inundated share"].notna().sum() == 1_235
    assert price_map["Maximum local price pressure"].notna().sum() == 22
    return conflict_map, drought_map, price_map, flood_map


def robust_limit(values: pd.Series, quantile: float = 0.98) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    assert not finite.empty
    return float(finite.quantile(quantile))


def plot_panel(
    ax: plt.Axes,
    data: gpd.GeoDataFrame,
    column: str,
    cmap: str,
    panel_label: str,
    annotation: str,
    colorbar_label: str,
    vmax: float,
    outline: gpd.GeoDataFrame,
) -> None:
    vmin = 0.0
    data.plot(
        column=column,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidth=0.12,
        edgecolor="#FFFFFF",
        missing_kwds={"color": "#E5E7EB", "edgecolor": "#FFFFFF"},
    )
    outline.boundary.plot(ax=ax, color="#4B5563", linewidth=0.45)
    min_x, min_y, max_x, max_y = outline.total_bounds
    longitude_ticks = np.arange(np.ceil(min_x), np.floor(max_x) + 0.01, 1.0)
    latitude_ticks = np.arange(np.ceil(min_y), np.floor(max_y) + 0.01, 1.0)
    ax.set_xlim(min_x - 0.10, max_x + 0.10)
    ax.set_ylim(min_y - 0.10, max_y + 0.10)
    ax.set_xticks(longitude_ticks)
    ax.set_yticks(latitude_ticks)
    ax.set_xticklabels([f"{value:.0f}°E" for value in longitude_ticks])
    ax.set_yticklabels([f"{value:.0f}°N" for value in latitude_ticks])
    ax.tick_params(
        axis="both",
        colors="#4B5563",
        labelsize=7.2,
        length=2.5,
        width=0.55,
        pad=2.0,
    )
    ax.set_axisbelow(False)
    ax.grid(
        color="#6B7280",
        linewidth=0.45,
        linestyle=(0, (2, 3)),
        alpha=0.45,
        zorder=4,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#6B7280")
        spine.set_linewidth(0.65)
    ax.text(
        -0.04,
        1.03,
        panel_label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.01,
        0.99,
        annotation,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="semibold",
        ha="left",
        va="top",
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.5},
    )
    scalar = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap
    )
    scalar.set_array([])
    colorbar = ax.figure.colorbar(
        scalar, ax=ax, orientation="horizontal", fraction=0.038, pad=0.018
    )
    colorbar.set_label(colorbar_label, fontsize=8.5, color="#374151")
    colorbar.ax.tick_params(labelsize=8, colors="#374151", length=2)
    colorbar.outline.set_linewidth(0.4)


def main() -> None:
    conflict, drought, price, flood = read_map_data()
    country_outline = conflict.dissolve()

    sns.set_theme(style="white", context="paper")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 11.0))
    plot_panel(
        axes[0, 0],
        conflict,
        CONFLICT,
        "magma_r",
        "a",
        "Historical conflict · commune",
        "Log bombing-location density per 100 km²",
        robust_limit(conflict[CONFLICT]),
        country_outline,
    )
    plot_panel(
        axes[0, 1],
        drought,
        "Drought intensity",
        "YlOrBr",
        "b",
        "Most severe linked drought · commune",
        "max(0, −minimum interview-month SPI-12)",
        robust_limit(drought["Drought intensity"]),
        country_outline,
    )
    plot_panel(
        axes[1, 0],
        price,
        "Maximum local price pressure",
        "Reds",
        "c",
        "Maximum linked wholesale rice-price pressure · province",
        "Local relative log wholesale rice price",
        float(price["Maximum local price pressure"].max()),
        country_outline,
    )
    plot_panel(
        axes[1, 1],
        flood,
        "Maximum inundated share",
        "Blues",
        "d",
        "Maximum linked satellite inundation · commune",
        "Survey-year maximum flooded geography share",
        robust_limit(flood["Maximum inundated share"]),
        country_outline,
    )

    figure.subplots_adjust(
        left=0.035,
        right=0.985,
        top=0.985,
        bottom=0.025,
        wspace=0.08,
        hspace=0.12,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print("Panels: 4 (a-d), no figure title")
    print(f"Conflict support: {conflict[CONFLICT].notna().sum():,} communes")
    print(f"Drought support: {drought['Drought intensity'].notna().sum():,} communes")
    print(f"Price support: {price['Maximum local price pressure'].notna().sum():,} provinces")
    print(f"Flood support: {flood['Maximum inundated share'].notna().sum():,} communes")


if __name__ == "__main__":
    main()
