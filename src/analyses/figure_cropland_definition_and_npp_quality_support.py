#!/usr/bin/env python3
"""Cropland-definition and NPP-quality support figure.

Plan: show where the inclusive agricultural definition adds pixels, document
the valid-pixel share under both definitions, and assess whether annual NPP is
sensitive to the land-cover definition.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/cses_public_village_cropland_npp_absolute_climate_panel_preprocessed.parquet"
COMMUNES = ROOT / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
OUTPUT = ROOT / "data/results/figures/Figure_cropland_definition_and_npp_quality_support.png"
EVIDENCE = ROOT / "data/exp/analysis/climate-npp/cropland-definition-and-npp-quality-support"

ID = "National Village Point ID"
LONGITUDE = "Point Longitude"
LATITUDE = "Point Latitude"
STRICT_CANDIDATE = "Strict-Cropland Candidate 500m Pixel Count"
INCLUSIVE_CANDIDATE = "Inclusive-Agriculture Candidate 500m Pixel Count"
STRICT_VALID = "Strict-Cropland Valid NPP 500m Pixel Count"
INCLUSIVE_VALID = "Inclusive-Agriculture Valid NPP 500m Pixel Count"
STRICT_SHARE = "Strict-Cropland Valid NPP Pixel Share"
INCLUSIVE_SHARE = "Inclusive-Agriculture Valid NPP Pixel Share"
STRICT_NPP = "Annual Strict-Cropland Mean NPP kg C per m2"
INCLUSIVE_NPP = "Annual Inclusive-Agriculture Mean NPP kg C per m2"

MAP_EXTENT = (101.9, 108.0, 10.0, 15.05)
DARK = "#40505A"
GRID = "#DDE3E5"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.013,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111111",
        zorder=10,
    )


def title_inside(ax: plt.Axes, title: str) -> None:
    ax.text(
        0.985,
        0.97,
        title,
        transform=ax.transAxes,
        fontsize=10.0,
        fontweight="bold",
        ha="right",
        va="top",
        color="#173C5C",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.6},
        zorder=10,
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=0.55, linestyle="--", alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.2, colors=DARK, length=2.5)
    for spine in ax.spines.values():
        spine.set_color("#89969C")
        spine.set_linewidth(0.65)


def map_context() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    communes = gpd.read_file(COMMUNES).to_crs(4326)
    province_field = next(
        column for column in ["pro_code", "PRO_CODE", "province", "pro_name_e"] if column in communes.columns
    )
    provinces = communes.dissolve(by=province_field).reset_index()
    cambodia = communes.dissolve()
    countries = gpd.read_file(COUNTRIES).to_crs(4326)
    surrounding = countries.cx[101.5:108.5, 9.7:15.5].copy()
    return surrounding, cambodia, provinces


def main() -> None:
    columns = [
        ID,
        LONGITUDE,
        LATITUDE,
        STRICT_CANDIDATE,
        INCLUSIVE_CANDIDATE,
        STRICT_VALID,
        INCLUSIVE_VALID,
        STRICT_SHARE,
        INCLUSIVE_SHARE,
        STRICT_NPP,
        INCLUSIVE_NPP,
    ]
    frame = pd.read_parquet(INPUT, columns=columns).copy()
    frame["Inclusive Minus Strict Candidate Pixels"] = frame[INCLUSIVE_CANDIDATE] - frame[STRICT_CANDIDATE]

    villages = (
        frame.groupby(ID, as_index=False)
        .agg(
            **{
                LONGITUDE: (LONGITUDE, "first"),
                LATITUDE: (LATITUDE, "first"),
                "Mean Added Candidate Pixels": ("Inclusive Minus Strict Candidate Pixels", "mean"),
                "Mean Strict Candidate Pixels": (STRICT_CANDIDATE, "mean"),
                "Mean Inclusive Candidate Pixels": (INCLUSIVE_CANDIDATE, "mean"),
                "Mean Strict Valid Pixels": (STRICT_VALID, "mean"),
                "Mean Inclusive Valid Pixels": (INCLUSIVE_VALID, "mean"),
            }
        )
    )

    paired = frame.dropna(subset=[STRICT_NPP, INCLUSIVE_NPP]).copy()
    npp_correlation = float(paired[[STRICT_NPP, INCLUSIVE_NPP]].corr().iloc[0, 1])
    mean_npp_difference = float((paired[INCLUSIVE_NPP] - paired[STRICT_NPP]).mean())
    strict_high_support = float(frame[STRICT_SHARE].dropna().ge(0.8).mean())
    inclusive_high_support = float(frame[INCLUSIVE_SHARE].dropna().ge(0.8).mean())

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    villages.to_parquet(EVIDENCE / "village_cropland_definition_pixel_support.parquet", index=False)
    frame[[STRICT_SHARE, INCLUSIVE_SHARE, STRICT_NPP, INCLUSIVE_NPP]].describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).T.to_csv(EVIDENCE / "cropland_definition_npp_quality_summary.csv")
    summary = {
        "village_year_rows": int(len(frame)),
        "villages": int(frame[ID].nunique()),
        "paired_npp_rows": int(len(paired)),
        "strict_valid_pixel_share_at_least_0_8": strict_high_support,
        "inclusive_valid_pixel_share_at_least_0_8": inclusive_high_support,
        "strict_inclusive_npp_pearson_correlation": npp_correlation,
        "mean_inclusive_minus_strict_npp_kg_c_per_m2": mean_npp_difference,
    }
    (EVIDENCE / "cropland_definition_and_npp_quality_support_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplot_mosaic(
        [["a", "a", "b"], ["a", "a", "c"]],
        figsize=(15.2, 8.7),
        facecolor="white",
        gridspec_kw={
            "width_ratios": [1.2, 1.2, 1.1],
            "left": 0.055,
            "right": 0.97,
            "top": 0.97,
            "bottom": 0.08,
            "wspace": 0.22,
            "hspace": 0.25,
        },
    )
    ax_a, ax_b, ax_c = axes["a"], axes["b"], axes["c"]

    surrounding, cambodia, provinces = map_context()
    surrounding.plot(ax=ax_a, facecolor="#F0F2F1", edgecolor="#AAB2B6", linewidth=0.42, zorder=0)
    cambodia.plot(ax=ax_a, facecolor="#FFFEFA", edgecolor="#4D5960", linewidth=0.78, zorder=1)
    provinces.boundary.plot(ax=ax_a, color="#A7B0B4", linewidth=0.32, zorder=2)
    upper_added = max(1.0, float(villages["Mean Added Candidate Pixels"].quantile(0.98)))
    points = ax_a.scatter(
        villages[LONGITUDE],
        villages[LATITUDE],
        c=villages["Mean Added Candidate Pixels"],
        cmap="YlGnBu",
        vmin=0,
        vmax=upper_added,
        s=9,
        alpha=0.94,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    ax_a.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax_a.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax_a.set_xticks(np.arange(102, 109, 1))
    ax_a.set_yticks(np.arange(10, 16, 1))
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.grid(color=GRID, linestyle="--", linewidth=0.48, alpha=0.82)
    ax_a.tick_params(labelsize=8.2, colors=DARK, length=2.5)
    for spine in ax_a.spines.values():
        spine.set_color("#89969C")
        spine.set_linewidth(0.65)
    panel_label(ax_a, "a")
    title_inside(ax_a, "Pixels added by inclusive agriculture definition")
    map_bar = fig.colorbar(points, ax=ax_a, orientation="vertical", pad=0.012, shrink=0.74, aspect=25)
    map_bar.set_label("Mean additional candidate 500 m pixels", fontsize=8.0, color=DARK, labelpad=5)
    map_bar.ax.tick_params(labelsize=7.4, colors=DARK, length=2)
    map_bar.outline.set_linewidth(0.45)

    bins = np.linspace(0, 1, 41)
    ax_b.hist(
        frame[STRICT_SHARE].dropna(),
        bins=bins,
        histtype="step",
        linewidth=1.8,
        color="#2166AC",
        label=f"Strict cropland (≥80%: {strict_high_support:.1%})",
    )
    ax_b.hist(
        frame[INCLUSIVE_SHARE].dropna(),
        bins=bins,
        histtype="step",
        linewidth=1.8,
        color="#D73027",
        label=f"Inclusive agriculture (≥80%: {inclusive_high_support:.1%})",
    )
    ax_b.axvline(0.8, color="#5A646A", linewidth=1.0, linestyle="--")
    ax_b.set_yscale("log")
    ax_b.set_xlim(0, 1.0)
    ax_b.set_xlabel("Valid annual NPP pixel share", fontsize=8.7, color=DARK)
    ax_b.set_ylabel("Village-year observations (log)", fontsize=8.7, color=DARK)
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.02, 0.88), frameon=False, fontsize=7.8)
    panel_label(ax_b, "b")
    title_inside(ax_b, "Annual NPP pixel support")
    style_axis(ax_b)

    x = paired[STRICT_NPP].to_numpy(float)
    y = paired[INCLUSIVE_NPP].to_numpy(float)
    lower = float(min(np.quantile(x, 0.002), np.quantile(y, 0.002)))
    upper = float(max(np.quantile(x, 0.998), np.quantile(y, 0.998)))
    density = ax_c.hexbin(x, y, gridsize=48, mincnt=1, bins="log", cmap="YlGnBu", linewidths=0)
    ax_c.plot([lower, upper], [lower, upper], color="#39464C", linewidth=1.1, linestyle="--")
    ax_c.set_xlim(lower, upper)
    ax_c.set_ylim(lower, upper)
    ax_c.set_xlabel("Strict-cropland annual NPP (kg C m⁻²)", fontsize=8.7, color=DARK)
    ax_c.set_ylabel("Inclusive-agriculture annual NPP (kg C m⁻²)", fontsize=8.7, color=DARK)
    panel_label(ax_c, "c")
    title_inside(ax_c, f"Definition agreement (r = {npp_correlation:.3f})")
    style_axis(ax_c)
    density_bar = fig.colorbar(density, ax=ax_c, orientation="vertical", pad=0.012, shrink=0.76, aspect=24)
    density_bar.set_label("Village-year density (log)", fontsize=7.8, color=DARK, labelpad=4)
    density_bar.ax.tick_params(labelsize=7.2, colors=DARK, length=2)
    density_bar.outline.set_linewidth(0.45)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved evidence: {EVIDENCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
