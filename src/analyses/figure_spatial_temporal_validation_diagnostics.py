#!/usr/bin/env python3
"""Spatial-Temporal Validation Diagnostics.

Plan: Show the spatial blocks, temporal folds, held-out prediction errors, and
coefficient stability underlying the post-monsoon vegetation validation gate.
Framework: AnaSOP strict 5-by-5 spatial-temporal cross-fitting and leave-one-
temporal-block-out validation for the November-February EVI response.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/cambodia_public_village_monsoon_ecology_panel_preprocessed.parquet"
CSES = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
COEFFICIENTS = ROOT / "data/exp/analysis/climate-welfare/postmonsoon-vegetation-validation/coefficients.csv"
FOLD_METRICS = ROOT / "data/exp/analysis/climate-welfare/postmonsoon-vegetation-validation/crossfit_fold_metrics.csv"
MODEL_COMPARISON = ROOT / "data/exp/analysis/climate-welfare/postmonsoon-vegetation-validation/crossfit_model_comparison.csv"
BOUNDARY = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
ANALYSIS_DIR = ROOT / "data/exp/analysis/climate-welfare/spatial-temporal-validation-diagnostics"
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/figures/Figure_spatial_temporal_validation_diagnostics.png"

ID = "National Village Point ID"
YEAR = "Year"
RADIUS = "Buffer Radius km"
LON = "Point Longitude"
LAT = "Point Latitude"
EVI = "Village Buffer Mean November-February Mean EVI Anomaly Z"
RAIN = "Village Buffer Mean May October Precipitation Total mm Anomaly Z"
ONSET = "Village Buffer Mean Wet-Season Onset DOY Candidate B Anomaly Z"
DRY = "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z"
HEAT = "Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate B"

FOLD_COLORS = ["#2a6f97", "#2a9d8f", "#e9c46a", "#e76f51", "#8a5fbf"]
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
BLUE = "#2a6f97"


def make_spatial_block(frame: pd.DataFrame) -> pd.Series:
    return (
        np.floor((frame[LON] - 102.0) / 0.75).astype("Int64").astype(str)
        + "_"
        + np.floor((frame[LAT] - 10.0) / 0.75).astype("Int64").astype(str)
    )


def assign_spatial_folds(frame: pd.DataFrame) -> pd.Series:
    counts = frame.groupby("Spatial Block", observed=True).size().sort_values(ascending=False)
    totals = [0] * 5
    mapping: dict[str, int] = {}
    for block, count in counts.items():
        target = int(np.argmin(totals))
        mapping[str(block)] = target
        totals[target] += int(count)
    return frame["Spatial Block"].map(mapping).astype(int)


def build_fold_map() -> pd.DataFrame:
    columns = [RADIUS, ID, YEAR, LON, LAT, EVI, RAIN, ONSET, DRY, HEAT]
    frame = pd.read_parquet(PANEL, columns=columns)
    frame = frame.loc[
        frame[RADIUS].eq(5) & frame[YEAR].between(2001, 2023)
    ].dropna(subset=[EVI, RAIN, ONSET, DRY, HEAT]).copy()
    frame["Spatial Block"] = make_spatial_block(frame)
    frame["Spatial Fold"] = assign_spatial_folds(frame)
    villages = frame[
        [ID, LON, LAT, "Spatial Block", "Spatial Fold"]
    ].drop_duplicates(ID).sort_values(ID)
    linked = pd.read_parquet(
        CSES, columns=[ID, "Climate Ecology Link Available"]
    )
    linked_ids = set(
        linked.loc[linked["Climate Ecology Link Available"].eq(1), ID].dropna().astype(str)
    )
    villages["CSES Linked Village"] = villages[ID].astype(str).isin(linked_ids).astype(int)
    return villages


def style_map(
    ax: plt.Axes,
    boundary: gpd.GeoDataFrame,
    provinces: gpd.GeoDataFrame,
    neighbors: gpd.GeoDataFrame,
) -> None:
    neighbors.plot(
        ax=ax, facecolor="#f1f2f1", edgecolor="#a6adb2", linewidth=0.55, zorder=-2
    )
    provinces.boundary.plot(ax=ax, color="#9ba3a8", linewidth=0.32, zorder=3)
    boundary.boundary.plot(ax=ax, color="#4f5960", linewidth=0.78, zorder=4)
    for name, (longitude, latitude) in {
        "Thailand": (102.10, 13.15),
        "Laos": (105.70, 14.72),
        "Vietnam": (107.72, 12.55),
    }.items():
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


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.07, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")


def prediction_grid(metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = metrics.pivot(
        index=["Spatial Fold", "Temporal Fold"], columns="Model", values="RMSE SD"
    ).reset_index()
    pivot["RMSE Improvement Percent"] = 100 * (
        pivot["Rainfall totals only"] - pivot["Joint rainfall and monsoon structure"]
    ) / pivot["Rainfall totals only"]
    return pivot


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    labels = [
        ("Primary: approved, 5 km", "Candidate B, 35 C, 5 km"),
        ("Approved, 2 km", "Candidate B, 35 C, 2 km"),
        ("Approved, 10 km", "Candidate B, 35 C, 10 km"),
        ("Exclude 2001–2005", "Candidate B, 35 C, 5 km, excluding years 2001-2005"),
        ("Exclude 2006–2010", "Candidate B, 35 C, 5 km, excluding years 2006-2010"),
        ("Exclude 2011–2015", "Candidate B, 35 C, 5 km, excluding years 2011-2015"),
        ("Exclude 2016–2019", "Candidate B, 35 C, 5 km, excluding years 2016-2019"),
        ("Exclude 2020–2023", "Candidate B, 35 C, 5 km, excluding years 2020-2023"),
    ]
    rows = []
    for label, specification in labels:
        row = coefficients.loc[
            coefficients["Specification"].eq(specification)
            & coefficients["Outcome"].eq(EVI)
            & coefficients["Exposure"].str.contains("Longest Intraseasonal Dry Spell")
        ]
        if len(row) != 1:
            raise ValueError(f"Expected one EVI dry-spell row for {specification!r}; found {len(row)}")
        selected = row.iloc[0].copy()
        selected["Display Label"] = label
        rows.append(selected)
    return pd.DataFrame(rows)


def main() -> None:
    villages = build_fold_map()
    metrics = pd.read_csv(FOLD_METRICS)
    model_comparison = pd.read_csv(MODEL_COMPARISON)
    coefficients = pd.read_csv(COEFFICIENTS)
    grid = prediction_grid(metrics)
    stability = coefficient_stability(coefficients)

    temporal_ranges = {
        0: "2001–2005", 1: "2006–2010", 2: "2011–2015",
        3: "2016–2019", 4: "2020–2023",
    }
    temporal = (
        grid.groupby("Temporal Fold", observed=True)["RMSE Improvement Percent"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    temporal["Years"] = temporal["Temporal Fold"].map(temporal_ranges)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    villages.to_parquet(ANALYSIS_DIR / "spatial_fold_villages.parquet", index=False)
    grid.to_csv(ANALYSIS_DIR / "spatial_temporal_fold_prediction_improvement.csv", index=False)
    temporal.to_csv(ANALYSIS_DIR / "temporal_fold_prediction_summary.csv", index=False)
    stability.to_csv(ANALYSIS_DIR / "dry_spell_coefficient_stability.csv", index=False)

    communes = gpd.read_file(BOUNDARY).to_crs(4326)
    boundary = communes.dissolve()
    provinces = communes.dissolve(by="ADM1_PCODE").reset_index()
    countries = gpd.read_file(COUNTRIES).to_crs(4326)
    neighbors = countries.loc[countries["ADMIN"].isin(["Thailand", "Laos", "Vietnam"])].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.7), constrained_layout=True)

    # a: spatial fold geography and linked-survey support
    ax = axes[0, 0]
    for fold, color in enumerate(FOLD_COLORS):
        subset = villages.loc[villages["Spatial Fold"].eq(fold)]
        ax.scatter(
            subset[LON], subset[LAT], s=3.2, color=color, alpha=0.70,
            linewidths=0, zorder=2,
        )
    linked = villages.loc[villages["CSES Linked Village"].eq(1)]
    ax.scatter(
        linked[LON], linked[LAT], s=11, facecolors="none", edgecolors="#20272b",
        linewidths=0.38, alpha=0.55, zorder=3,
    )
    style_map(ax, boundary, provinces, neighbors)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor="none", markersize=5, label=f"Spatial fold {fold + 1}")
        for fold, color in enumerate(FOLD_COLORS)
    ]
    handles.append(
        Line2D([0], [0], marker="o", color="#20272b", markerfacecolor="none",
               markersize=5, linewidth=0, label="CSES-linked village")
    )
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=3, frameon=False, fontsize=7.8, columnspacing=0.9, handletextpad=0.3,
    )
    panel_label(ax, "a")

    # b: average held-out improvement by temporal fold
    ax = axes[0, 1]
    work = temporal.sort_values("Temporal Fold", ascending=False)
    y = np.arange(len(work))
    values = work["mean"].to_numpy(float)
    colors = [TEAL if value > 0 else ORANGE for value in values]
    ax.axvline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    ax.barh(y, values, color=colors, height=0.55)
    for yi, value in zip(y, values):
        if value >= 0:
            ax.text(
                value + 0.16, yi, f"{value:+.2f}%",
                va="center", ha="left", fontsize=8.8,
            )
        else:
            ax.text(
                value / 2, yi, f"{value:+.2f}%",
                va="center", ha="center", fontsize=8.5,
                color="white", fontweight="bold",
            )
    ax.set_yticks(y, work["Years"])
    ax.set_xlabel("Mean held-out RMSE improvement versus rainfall only")
    ax.set_ylabel("Held-out temporal fold")
    ax.grid(axis="x", color="#d9dde0", linewidth=0.65, linestyle="--")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.12)
    r2_by_model = model_comparison.set_index("Model")[
        "R2 versus Zero-Anomaly Benchmark"
    ]
    absolute_boundary = (
        "Absolute held-out $R^2_0$ vs zero anomaly\n"
        f"Rainfall only  {r2_by_model['Rainfall totals only']:+.3f}\n"
        f"Monsoon only  {r2_by_model['Monsoon structure only']:+.3f}\n"
        f"Joint model   {r2_by_model['Joint rainfall and monsoon structure']:+.3f}\n"
        "All < 0: no absolute predictive skill"
    )
    ax.text(
        0.98, 0.04, absolute_boundary, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.2, linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white",
              "edgecolor": "#9aa1a6", "linewidth": 0.7, "alpha": 0.94},
    )
    panel_label(ax, "b")

    # c: exact 5-by-5 diagnostic grid
    ax = axes[1, 0]
    matrix = grid.pivot(
        index="Spatial Fold", columns="Temporal Fold", values="RMSE Improvement Percent"
    ).sort_index().sort_index(axis=1)
    limit = float(np.ceil(np.abs(matrix.to_numpy()).max()))
    image = ax.imshow(
        matrix.to_numpy(), cmap="RdYlGn", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        aspect="auto",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            ax.text(column, row, f"{value:+.1f}%", ha="center", va="center", fontsize=8.3)
    ax.set_xticks(np.arange(5), [temporal_ranges[i] for i in range(5)], rotation=30, ha="right")
    ax.set_yticks(np.arange(5), [f"Fold {i + 1}" for i in range(5)])
    ax.set_xlabel("Held-out temporal fold")
    ax.set_ylabel("Held-out spatial fold")
    colorbar = fig.colorbar(image, ax=ax, orientation="vertical", pad=0.02, shrink=0.82)
    colorbar.set_label("RMSE improvement (%)", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)
    improved = int((grid["RMSE Improvement Percent"] > 0).sum())
    ax.text(
        0.02, 0.98, f"Joint RMSE lower in {improved}/25 held-out cells",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
              "edgecolor": "#9aa1a6", "linewidth": 0.6, "alpha": 0.90},
    )
    panel_label(ax, "c")

    # d: coefficient stability across scale, definition, and temporal exclusions
    ax = axes[1, 1]
    work = stability.iloc[::-1].reset_index(drop=True)
    estimate = work["Coefficient Outcome SD per Exposure Unit"].to_numpy(float)
    lower = work["95 Percent CI Lower"].to_numpy(float)
    upper = work["95 Percent CI Upper"].to_numpy(float)
    y = np.arange(len(work))
    colors = [BLUE if label.startswith("Primary") else TEAL for label in work["Display Label"]]
    ax.axvline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    for index, color in enumerate(colors):
        ax.errorbar(
            estimate[index], y[index],
            xerr=[[estimate[index] - lower[index]], [upper[index] - estimate[index]]],
            fmt="o", color=color, ecolor=color, markersize=4.8,
            elinewidth=1.3, capsize=3,
        )
    ax.set_yticks(y, work["Display Label"])
    ax.set_xlabel("Post-monsoon EVI response per 1 SD longer dry spell")
    ax.grid(axis="x", color="#d9dde0", linewidth=0.65, linestyle="--")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_label(ax, "d")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Held-out cells improved: {improved}/25")
    print(temporal.to_string(index=False))
    print(stability[["Display Label", "Coefficient Outcome SD per Exposure Unit", "95 Percent CI Lower", "95 Percent CI Upper", "Probability Value"]].to_string(index=False))


if __name__ == "__main__":
    main()
