#!/usr/bin/env python3
"""GWR bandwidth and effective-sample diagnostics.

Plan: audit AICc-selected adaptive bandwidths, leave-one-village-out prediction,
effective local sample distributions, and the geography of the weakest stage-2
local information without selecting bandwidths by coefficient significance.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/exp/analysis/climate-npp/continuous-spatial-heterogeneity"
BANDWIDTH_INPUT = SOURCE / "adaptive_bandwidth_aicc_and_prediction_diagnostics.csv"
SURFACE_INPUT = SOURCE / "continuous_local_slope_surfaces.parquet"
COMMUNES = ROOT / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
OUTPUT = ROOT / "data/results/figures/Figure_gwr_bandwidth_and_effective_sample_diagnostics.png"
EVIDENCE = ROOT / "data/exp/analysis/climate-npp/gwr-bandwidth-and-effective-sample-diagnostics"

ID = "National Village Point ID"
LONGITUDE = "Point Longitude"
LATITUDE = "Point Latitude"
MAP_EXTENT = (101.9, 108.0, 10.0, 15.05)

MODEL_ORDER = [
    "Heat to cropland NPP",
    "Dry spell to cropland NPP",
    "Cropland NPP to total consumption",
    "Cropland NPP to food consumption",
]
MODEL_LABELS = {
    "Heat to cropland NPP": "Heat → NPP",
    "Dry spell to cropland NPP": "Dry spell → NPP",
    "Cropland NPP to total consumption": "NPP → total consumption",
    "Cropland NPP to food consumption": "NPP → food consumption",
}
MODEL_COLORS = {
    "Heat to cropland NPP": "#D97757",
    "Dry spell to cropland NPP": "#D4A72C",
    "Cropland NPP to total consumption": "#173F5F",
    "Cropland NPP to food consumption": "#3A9D8F",
}

NAVY = "#173F5F"
DARK = "#4D5960"
GRID = "#D9DEE1"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.2},
        zorder=20,
    )


def title_inside(ax: plt.Axes, title: str) -> None:
    ax.text(
        0.985,
        0.965,
        title,
        transform=ax.transAxes,
        fontsize=10.1,
        fontweight="bold",
        ha="right",
        va="top",
        color=NAVY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.4},
        zorder=20,
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", color=GRID, linewidth=0.55, linestyle="--", zorder=0)
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
    bandwidth = pd.read_csv(BANDWIDTH_INPUT)
    surfaces = pd.read_parquet(SURFACE_INPUT)
    bandwidth["AICc Difference"] = bandwidth["AICc"] - bandwidth.groupby("Model")["AICc"].transform("min")
    bandwidth["AICc Difference Plus One"] = bandwidth["AICc Difference"].clip(lower=0.0) + 1.0
    bandwidth["RMSE Percent Above Model Minimum"] = 100.0 * (
        bandwidth["Leave-One-Village-Out RMSE"]
        / bandwidth.groupby("Model")["Leave-One-Village-Out RMSE"].transform("min")
        - 1.0
    )
    selected_index = bandwidth.groupby("Model")["AICc"].idxmin()
    selected = bandwidth.loc[selected_index].copy().sort_values("Model")

    total = surfaces.loc[
        surfaces["Model"].eq("Cropland NPP to total consumption"),
        [ID, LONGITUDE, LATITUDE, "Effective Local Sample"],
    ].rename(columns={"Effective Local Sample": "Total-Consumption Effective Local Sample"})
    food = surfaces.loc[
        surfaces["Model"].eq("Cropland NPP to food consumption"),
        [ID, "Effective Local Sample"],
    ].rename(columns={"Effective Local Sample": "Food-Consumption Effective Local Sample"})
    household_support = total.merge(food, on=ID, how="inner", validate="one_to_one")
    household_support["Minimum Household Effective Local Sample"] = household_support[
        ["Total-Consumption Effective Local Sample", "Food-Consumption Effective Local Sample"]
    ].min(axis=1)

    effective_summary = (
        surfaces.groupby("Model")["Effective Local Sample"]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        .reset_index()
    )
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    selected.to_csv(EVIDENCE / "selected_adaptive_bandwidth_diagnostics.csv", index=False)
    effective_summary.to_csv(EVIDENCE / "effective_local_sample_distribution_summary.csv", index=False)
    household_support.to_parquet(EVIDENCE / "household_effective_local_sample_map.parquet", index=False)
    summary = {
        row["Model"]: {
            "selected_adaptive_neighbor_bandwidth": int(row["Adaptive Neighbor Bandwidth"]),
            "aicc": float(row["AICc"]),
            "leave_one_village_out_rmse": float(row["Leave-One-Village-Out RMSE"]),
            "median_effective_local_sample": float(row["Median Effective Local Sample"]),
        }
        for _, row in selected.iterrows()
    }
    (EVIDENCE / "gwr_bandwidth_and_effective_sample_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplot_mosaic(
        [["a", "a", "a"], ["b", "d", "d"], ["c", "d", "d"]],
        figsize=(15.2, 10.0),
        facecolor="white",
        gridspec_kw={
            "height_ratios": [0.78, 1.0, 1.0],
            "left": 0.09,
            "right": 0.975,
            "top": 0.97,
            "bottom": 0.075,
            "wspace": 0.27,
            "hspace": 0.31,
        },
    )

    ax = axes["a"]
    for model in MODEL_ORDER:
        model_data = bandwidth.loc[bandwidth["Model"].eq(model)].sort_values("Adaptive Neighbor Bandwidth")
        chosen = selected.loc[selected["Model"].eq(model)].iloc[0]
        ax.plot(
            model_data["Adaptive Neighbor Bandwidth"],
            model_data["AICc Difference Plus One"],
            color=MODEL_COLORS[model],
            linewidth=1.45,
            marker="o",
            markersize=2.7,
            label=MODEL_LABELS[model],
            zorder=2,
        )
        ax.scatter(
            chosen["Adaptive Neighbor Bandwidth"],
            max(float(chosen["AICc Difference"]), 0.0) + 1.0,
            s=48,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Adaptive neighbour bandwidth", fontsize=8.8, color=DARK)
    ax.set_ylabel("AICc difference + 1 (log scale)", fontsize=8.8, color=DARK)
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.05), frameon=False, fontsize=7.5, ncol=2)
    panel_label(ax, "a")
    title_inside(ax, "AICc bandwidth selection")
    style_axis(ax)

    ax = axes["b"]
    for model in MODEL_ORDER:
        model_data = bandwidth.loc[bandwidth["Model"].eq(model)].sort_values("Adaptive Neighbor Bandwidth")
        chosen = selected.loc[selected["Model"].eq(model)].iloc[0]
        ax.plot(
            model_data["Adaptive Neighbor Bandwidth"],
            model_data["RMSE Percent Above Model Minimum"],
            color=MODEL_COLORS[model],
            linewidth=1.45,
            marker="o",
            markersize=2.7,
            zorder=2,
        )
        ax.scatter(
            chosen["Adaptive Neighbor Bandwidth"],
            chosen["RMSE Percent Above Model Minimum"],
            s=48,
            color=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    ax.set_xscale("log")
    ax.set_ylim(-0.35, 14.5)
    ax.set_xlabel("Adaptive neighbour bandwidth", fontsize=8.8, color=DARK)
    ax.set_ylabel("LOVO RMSE above model-specific minimum (%)", fontsize=8.8, color=DARK)
    panel_label(ax, "b")
    title_inside(ax, "LOVO prediction")
    style_axis(ax)

    ax = axes["c"]
    box_values = [
        surfaces.loc[surfaces["Model"].eq(model), "Effective Local Sample"].dropna().to_numpy(float)
        for model in MODEL_ORDER
    ]
    box = ax.boxplot(
        box_values,
        orientation="horizontal",
        patch_artist=True,
        tick_labels=[MODEL_LABELS[model] for model in MODEL_ORDER],
        showfliers=False,
        widths=0.56,
        medianprops={"color": "#26343B", "linewidth": 1.2},
        whiskerprops={"color": "#68747A", "linewidth": 1.0},
        capprops={"color": "#68747A", "linewidth": 1.0},
    )
    for patch, model in zip(box["boxes"], MODEL_ORDER, strict=True):
        patch.set_facecolor(MODEL_COLORS[model])
        patch.set_alpha(0.78)
        patch.set_edgecolor("white")
    ax.set_xscale("log")
    ax.set_xlabel("Effective local sample size (log scale)", fontsize=8.8, color=DARK)
    panel_label(ax, "c")
    title_inside(ax, "Effective local sample size")
    style_axis(ax)

    ax = axes["d"]
    surrounding, cambodia, provinces = map_context()
    surrounding.plot(ax=ax, facecolor="#F0F2F1", edgecolor="#AAB2B6", linewidth=0.42, zorder=0)
    cambodia.plot(ax=ax, facecolor="#FFFEFA", edgecolor="#4D5960", linewidth=0.76, zorder=1)
    provinces.boundary.plot(ax=ax, color="#A7B0B4", linewidth=0.31, zorder=2)
    map_points = ax.scatter(
        household_support[LONGITUDE],
        household_support[LATITUDE],
        c=household_support["Minimum Household Effective Local Sample"],
        cmap="YlGnBu",
        s=9,
        alpha=0.94,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_xticks(np.arange(102, 109, 1))
    ax.set_yticks(np.arange(10, 16, 1))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=GRID, linewidth=0.48, linestyle="--", alpha=0.82)
    ax.tick_params(labelsize=8.1, colors=DARK, length=2.5)
    for spine in ax.spines.values():
        spine.set_color("#89969C")
        spine.set_linewidth(0.65)
    map_bar = fig.colorbar(map_points, ax=ax, orientation="vertical", pad=0.012, shrink=0.75, aspect=24)
    map_bar.set_label("Minimum effective household sample", fontsize=7.9, color=DARK, labelpad=5)
    map_bar.ax.tick_params(labelsize=7.3, colors=DARK, length=2)
    map_bar.outline.set_linewidth(0.45)
    panel_label(ax, "d")
    title_inside(ax, "Weakest stage-2 local support")

    # Align the right edge of the full-width AICc panel with the map frame,
    # leaving the map colourbar in its own visual column.
    panel_a_position = axes["a"].get_position()
    map_position = axes["d"].get_position()
    axes["a"].set_position(
        [
            panel_a_position.x0,
            panel_a_position.y0,
            map_position.x1 - panel_a_position.x0,
            panel_a_position.height,
        ]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(selected[["Model", "Adaptive Neighbor Bandwidth", "AICc", "Leave-One-Village-Out RMSE", "Median Effective Local Sample"]].to_string(index=False))
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved evidence: {EVIDENCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
