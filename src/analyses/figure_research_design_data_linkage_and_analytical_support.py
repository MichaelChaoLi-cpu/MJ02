#!/usr/bin/env python3
"""Research Design Data Linkage and Analytical Support.

Plan: Document national village support, temporal coverage, NPP linkage across
buffer scales and land-cover definitions, and household sample retention before
coefficient interpretation.
Framework: AnaSOP Section 5 associational two-stage design and Section 7 Steps
1, 2, and 8. This figure contains no estimated climate or welfare effect.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HOUSEHOLDS = ROOT / "data/processed/cses_household_cropland_npp_analysis_preprocessed.parquet"
NPP = ROOT / "data/processed/cses_public_village_pixel_cropland_npp_annual_preprocessed.parquet"
COMMUNES = ROOT / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
OUTPUT = ROOT / "data/results/figures/Figure_research_design_data_linkage_and_analytical_support.png"

NAVY = "#183B56"
BLUE = "#287C8E"
TEAL = "#3A9D8F"
ORANGE = "#D97757"
GOLD = "#D6A84B"
LIGHT_BLUE = "#DDECEF"
LIGHT_ORANGE = "#F3E1D9"
LIGHT_GRAY = "#ECEFF1"
MID_GRAY = "#9AA3A8"
DARK_GRAY = "#48545B"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=30,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="both", color="#D8DEE2", linewidth=0.55, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#707A80")
    ax.tick_params(labelsize=8.5, colors=DARK_GRAY)


def main() -> None:
    household_columns = [
        "Survey Wave",
        "Interview Calendar Year",
        "National Village Point ID",
        "Point Longitude",
        "Point Latitude",
        "Total Consumption Main Outcome Eligible",
        "Prior-Year Strict-Cropland NPP at 2 km",
        "Prior-Year Inclusive-Agriculture NPP at 2 km",
        "Prior-Year Strict-Cropland NPP",
        "Prior-Year Inclusive-Agriculture NPP",
        "Prior-Year Strict-Cropland NPP at 10 km",
        "Prior-Year Inclusive-Agriculture NPP at 10 km",
        "Stage 2 Total Consumption Complete Case",
        "Stage 2 Food Consumption Complete Case",
    ]
    households = pd.read_parquet(HOUSEHOLDS, columns=household_columns)
    villages = pd.read_parquet(
        NPP,
        columns=[
            "National Village Point ID",
            "Point Longitude",
            "Point Latitude",
        ],
    ).drop_duplicates("National Village Point ID")
    total_points = households.loc[
        households["Stage 2 Total Consumption Complete Case"],
        ["National Village Point ID", "Point Longitude", "Point Latitude"],
    ].drop_duplicates("National Village Point ID")

    communes = gpd.read_file(COMMUNES).to_crs(4326)
    provinces = communes.dissolve(by="pro_code").reset_index()
    cambodia = communes.dissolve()
    countries = gpd.read_file(COUNTRIES).to_crs(4326)
    region = countries.cx[101.4:108.7, 9.6:15.6].copy()

    fig = plt.figure(figsize=(19.1, 8.5), facecolor="white")
    axes = fig.subplot_mosaic(
        [["a", "a", "b", "b"], ["a", "a", "c", "d"]],
        gridspec_kw={"width_ratios": [1.32, 1.32, 1.0, 1.0], "height_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.975, bottom=0.105, wspace=0.34, hspace=0.34)

    # a: national village and stage-2 analytical support
    ax = axes["a"]
    region.plot(ax=ax, facecolor="#F2F3F1", edgecolor="#A5ADB1", linewidth=0.55, zorder=0)
    cambodia.plot(ax=ax, facecolor="#FFFDF7", edgecolor=DARK_GRAY, linewidth=0.9, zorder=1)
    provinces.boundary.plot(ax=ax, color="#9CA5AA", linewidth=0.36, zorder=2)
    ax.scatter(
        villages["Point Longitude"],
        villages["Point Latitude"],
        s=3.0,
        color="#B8C1C5",
        alpha=0.62,
        linewidths=0,
        zorder=3,
    )
    ax.scatter(
        total_points["Point Longitude"],
        total_points["Point Latitude"],
        s=7.0,
        color=ORANGE,
        alpha=0.84,
        linewidths=0,
        zorder=4,
    )
    country_labels = {
        "Thailand": (101.95, 13.15),
        "Laos": (105.65, 15.0),
        "Vietnam": (108.05, 12.65),
        "Cambodia": (104.85, 12.65),
    }
    for name, (longitude, latitude) in country_labels.items():
        ax.text(
            longitude,
            latitude,
            name,
            ha="center",
            va="center",
            fontsize=8.2 if name != "Cambodia" else 8.8,
            color="#747D82" if name != "Cambodia" else NAVY,
            fontstyle="italic" if name != "Cambodia" else "normal",
            zorder=5,
        )
    ax.set_xlim(101.6, 108.35)
    ax.set_ylim(9.9, 15.35)
    ax.set_xticks(np.arange(102, 109, 1))
    ax.set_yticks(np.arange(10, 16, 1))
    ax.grid(color="#D9DEE1", linewidth=0.55, linestyle="--", zorder=-1)
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#657178")
        spine.set_linewidth(0.75)
    ax.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="#B8C1C5",
                markeredgecolor="none", markersize=5.2,
                label=f"NPP-supported public villages ({len(villages):,})",
            ),
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=ORANGE,
                markeredgecolor="none", markersize=5.2,
                label=f"Total-consumption complete villages ({len(total_points):,})",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.095),
        ncol=2,
        frameon=False,
        fontsize=8.2,
        handletextpad=0.45,
        columnspacing=1.1,
    )
    panel_label(ax, "a")

    # b: household NPP linkage across buffer and land-cover definitions
    ax = axes["b"]
    eligible_mask = households["Total Consumption Main Outcome Eligible"]
    eligible_households = households.loc[eligible_mask]
    strict_columns = [
        "Prior-Year Strict-Cropland NPP at 2 km",
        "Prior-Year Strict-Cropland NPP",
        "Prior-Year Strict-Cropland NPP at 10 km",
    ]
    inclusive_columns = [
        "Prior-Year Inclusive-Agriculture NPP at 2 km",
        "Prior-Year Inclusive-Agriculture NPP",
        "Prior-Year Inclusive-Agriculture NPP at 10 km",
    ]
    strict_rates = [eligible_households[column].notna().mean() for column in strict_columns]
    inclusive_rates = [eligible_households[column].notna().mean() for column in inclusive_columns]
    x = np.arange(3)
    width = 0.34
    strict_bars = ax.bar(
        x - width / 2,
        strict_rates,
        width,
        color=TEAL,
        label="Strict cropland",
        zorder=2,
    )
    inclusive_bars = ax.bar(
        x + width / 2,
        inclusive_rates,
        width,
        color=GOLD,
        label="Inclusive agriculture",
        zorder=2,
    )
    for bars in (strict_bars, inclusive_bars):
        for bar in bars:
            rate = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 0.015,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color=DARK_GRAY,
            )
    ax.set_xticks(x, ["2 km", "5 km\n(primary)", "10 km"])
    ax.set_ylim(0, 0.90)
    ax.set_ylabel("Eligible households with prior-year NPP", fontsize=8.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.58, 0.99), frameon=False, fontsize=8.1, ncol=2)
    style_axes(ax)
    panel_label(ax, "b")

    # c: temporal support
    ax = axes["c"]
    rows = [
        ("Daily climate", 1991, 2024, BLUE),
        ("Annual NPP", 2001, 2021, TEAL),
    ]
    for y, (label, start, end, color) in enumerate(rows[::-1]):
        ax.barh(y, end - start + 1, left=start, height=0.34, color=color, alpha=0.90, zorder=2)
        ax.text(start + 0.35, y, f"{start}–{end}", va="center", ha="left", fontsize=7.7, color="white", fontweight="bold")
    wave_years = np.array([2007, 2009, 2011, 2013, 2014, 2016, 2017, 2019, 2020, 2021])
    ax.hlines(2, wave_years.min(), wave_years.max(), color=ORANGE, linewidth=1.4, zorder=1)
    ax.scatter(wave_years, np.full(len(wave_years), 2), s=23, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_yticks([0, 1, 2], ["NPP", "Climate", "CSES"])
    ax.set_xlim(1990, 2025)
    ax.set_xlabel("Calendar year", fontsize=8.8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    style_axes(ax)
    panel_label(ax, "c")

    # d: household analytical support
    ax = axes["d"]
    eligible = int(households["Total Consumption Main Outcome Eligible"].sum())
    linked = int(
        (
            households["Total Consumption Main Outcome Eligible"]
            & households["Prior-Year Strict-Cropland NPP"].notna()
        ).sum()
    )
    total_complete = int(households["Stage 2 Total Consumption Complete Case"].sum())
    food_complete = int(households["Stage 2 Food Consumption Complete Case"].sum())
    labels = ["Outcome eligible", "5 km NPP linked", "Total complete", "Food complete"]
    values = [eligible, linked, total_complete, food_complete]
    colors = [MID_GRAY, TEAL, NAVY, ORANGE]
    y = np.arange(len(values))[::-1]
    ax.barh(y, values, height=0.56, color=colors, zorder=2)
    for ypos, value in zip(y, values):
        ax.text(value + eligible * 0.018, ypos, f"{value:,}", va="center", fontsize=8.0, color=DARK_GRAY)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, eligible * 1.16)
    ax.set_xlabel("Households", fontsize=8.8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value / 1000)}k"))
    style_axes(ax)
    panel_label(ax, "d")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(
        f"public_villages={len(villages):,}; total_complete_villages={len(total_points):,}; "
        f"eligible_households={eligible:,}; npp_linked={linked:,}; "
        f"total_complete={total_complete:,}; food_complete={food_complete:,}"
    )


if __name__ == "__main__":
    main()
