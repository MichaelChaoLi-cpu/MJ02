#!/usr/bin/env python3
"""National Geography of Dry-Spell and Absolute-Heat Risk.

Plan: Map natural-unit historical dry-spell and absolute-heat exposure around
all national villages, show their continuous overlap, and compare the national
exposure range with linked CSES villages.
Framework: AnaSOP national risk geography.  The maps do not multiply exposure
by regression coefficients and do not represent causal welfare-loss scores.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from table_survey_linkage_and_transport_support import add_support_score, public_context


ROOT = Path(__file__).resolve().parents[2]
MONSOON = ROOT / "data/processed/cambodia_national_monsoon_timing_preprocessed.parquet"
MEMBERSHIP = ROOT / "data/processed/cambodia_public_village_buffer_grid_crosswalk/radius_5_km.parquet"
GRID_CLIMATE = ROOT / "data/processed/cambodia_national_1km_to_climate_cell_preprocessed.parquet"
VILLAGES = ROOT / "data/processed/cambodia_public_village_points_preprocessed.parquet"
CSES = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
BOUNDARY = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
ANALYSIS_DIR = ROOT / "data/exp/analysis/climate-welfare/national-dryspell-absolute-heat-risk"
RISK_DATA = ANALYSIS_DIR / "village_risk_metrics_5km.parquet"
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/figures/Figure_national_geography_of_dry_spell_and_absolute_heat_risk.png"

BLUE_CMAP = "YlGnBu"
HEAT_CMAP = "YlOrRd"
OVERLAP_CMAP = "magma_r"
ORANGE = "#e76f51"
NAVY = "#264653"


def q(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def build_risk_data() -> pd.DataFrame:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    query = f"""
      COPY (
        WITH climate_weights AS (
          SELECT m."National Village Point ID", g."Climate Cell ID",
            count(*)::DOUBLE AS grid_weight
          FROM read_parquet('{q(MEMBERSHIP)}') m
          INNER JOIN read_parquet('{q(GRID_CLIMATE)}') g USING ("National Grid Cell ID")
          GROUP BY 1, 2
        ), climate_history AS (
          SELECT "Climate Cell ID",
            avg("Longest Intraseasonal Dry Spell Days Candidate B")
              AS mean_dry_spell_days,
            avg("Post-Onset Absolute Heat Day Count 35 C Candidate B")
              AS mean_heat_days_35,
            count("Longest Intraseasonal Dry Spell Days Candidate B")
              AS dry_valid_years,
            count("Post-Onset Absolute Heat Day Count 35 C Candidate B")
              AS heat_valid_years
          FROM read_parquet('{q(MONSOON)}')
          WHERE "Year" BETWEEN 1991 AND 2024
          GROUP BY 1
        ), village_history AS (
          SELECT w."National Village Point ID",
            sum(h.mean_dry_spell_days * w.grid_weight) FILTER (WHERE h.dry_valid_years >= 30)
              / nullif(sum(w.grid_weight) FILTER (WHERE h.dry_valid_years >= 30), 0)
              AS "Historical Mean Longest Dry Spell Days",
            sum(h.mean_heat_days_35 * w.grid_weight) FILTER (WHERE h.heat_valid_years >= 30)
              / nullif(sum(w.grid_weight) FILTER (WHERE h.heat_valid_years >= 30), 0)
              AS "Historical Mean Post-Onset Heat Days 35 C",
            min(h.dry_valid_years) AS "Minimum Valid Dry-Spell Years",
            min(h.heat_valid_years) AS "Minimum Valid Heat Years"
          FROM climate_weights w
          INNER JOIN climate_history h USING ("Climate Cell ID")
          GROUP BY 1
        ), linked AS (
          SELECT DISTINCT "National Village Point ID", 1 AS "CSES Linked Village"
          FROM read_parquet('{q(CSES)}')
          WHERE "Climate Ecology Link Available" = 1
        )
        SELECT v."National Village Point ID", v."Point Longitude", v."Point Latitude",
          v."Province Name", v."District Name",
          h."Historical Mean Longest Dry Spell Days",
          h."Historical Mean Post-Onset Heat Days 35 C",
          h."Minimum Valid Dry-Spell Years", h."Minimum Valid Heat Years",
          coalesce(l."CSES Linked Village", 0) AS "CSES Linked Village"
        FROM read_parquet('{q(VILLAGES)}') v
        INNER JOIN village_history h USING ("National Village Point ID")
        LEFT JOIN linked l USING ("National Village Point ID")
        ORDER BY v."National Village Point ID"
      ) TO '{q(RISK_DATA)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.close()
    frame = pd.read_parquet(RISK_DATA)
    frame["Dry-Spell National Percentile"] = frame[
        "Historical Mean Longest Dry Spell Days"
    ].rank(method="average", pct=True)
    frame["Heat National Percentile"] = frame[
        "Historical Mean Post-Onset Heat Days 35 C"
    ].rank(method="average", pct=True)
    frame["Continuous Joint Exposure Rank"] = np.sqrt(
        frame["Dry-Spell National Percentile"] * frame["Heat National Percentile"]
    )
    frame.to_parquet(RISK_DATA, index=False, compression="zstd")
    frame[
        [
            "Historical Mean Longest Dry Spell Days",
            "Historical Mean Post-Onset Heat Days 35 C",
            "Continuous Joint Exposure Rank",
        ]
    ].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).T.to_csv(
        ANALYSIS_DIR / "risk_metric_summary.csv"
    )
    return frame


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=13,
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


def map_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    boundary: gpd.GeoDataFrame,
    provinces: gpd.GeoDataFrame,
    neighbors: gpd.GeoDataFrame,
    variable: str,
    cmap: str,
    colorbar_label: str,
) -> None:
    low, high = frame[variable].quantile([0.01, 0.99])
    points = ax.scatter(
        frame["Point Longitude"], frame["Point Latitude"],
        c=frame[variable], cmap=cmap, vmin=low, vmax=high,
        s=5.0, alpha=0.88, linewidths=0, zorder=2,
    )
    style_map(ax, boundary, provinces, neighbors)
    colorbar = ax.figure.colorbar(points, ax=ax, orientation="vertical", pad=0.018, shrink=0.80)
    colorbar.set_label(colorbar_label, fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.outline.set_linewidth(0.5)


def main() -> None:
    frame = build_risk_data()
    support, threshold95, _ = add_support_score(public_context())
    frame = frame.merge(
        support[
            [
                "National Village Point ID",
                "Primary 95 Percent Support",
                "Survey Support Score",
            ]
        ],
        on="National Village Point ID",
        how="left",
        validate="one_to_one",
    )
    communes = gpd.read_file(BOUNDARY).to_crs(4326)
    boundary = communes.dissolve()
    provinces = communes.dissolve(by="ADM1_PCODE").reset_index()
    countries = gpd.read_file(COUNTRIES).to_crs(4326)
    neighbors = countries.loc[countries["ADMIN"].isin(["Thailand", "Laos", "Vietnam"])].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.8), constrained_layout=True)

    map_panel(
        axes[0, 0], frame, boundary, provinces, neighbors,
        "Historical Mean Longest Dry Spell Days", BLUE_CMAP,
        "Mean annual longest dry spell (days)",
    )
    add_panel_label(axes[0, 0], "a")

    map_panel(
        axes[0, 1], frame, boundary, provinces, neighbors,
        "Historical Mean Post-Onset Heat Days 35 C", HEAT_CMAP,
        "Mean annual post-onset days ≥35°C",
    )
    add_panel_label(axes[0, 1], "b")

    map_panel(
        axes[1, 0], frame, boundary, provinces, neighbors,
        "Continuous Joint Exposure Rank", OVERLAP_CMAP,
        "Joint exposure rank",
    )
    add_panel_label(axes[1, 0], "c")

    ax = axes[1, 1]
    supported = frame.loc[frame["Primary 95 Percent Support"]]
    unsupported = frame.loc[~frame["Primary 95 Percent Support"]]
    linked = frame.loc[frame["CSES Linked Village"].eq(1)]
    ax.scatter(
        supported["Point Longitude"], supported["Point Latitude"],
        s=4.0, color="#86b6c9", alpha=0.62, linewidths=0, zorder=1,
    )
    ax.scatter(
        unsupported["Point Longitude"], unsupported["Point Latitude"],
        s=8.0, color=ORANGE, alpha=0.90, linewidths=0, zorder=2,
    )
    ax.scatter(
        linked["Point Longitude"], linked["Point Latitude"],
        s=10, facecolors="none", edgecolors="#20272b", linewidths=0.35,
        alpha=0.45, zorder=3,
    )
    style_map(ax, boundary, provinces, neighbors)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#86b6c9",
                   markeredgecolor="none", markersize=6,
                   label=f"Within primary support ({len(supported):,})"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE,
                   markeredgecolor="none", markersize=6,
                   label=f"Outside primary support ({len(unsupported):,})"),
            Line2D([0], [0], marker="o", color="#20272b", markerfacecolor="none",
                   linewidth=0, markersize=5, label=f"CSES-linked ({len(linked):,})"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1,
        frameon=False, fontsize=8.0,
    )
    add_panel_label(ax, "d")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(
        frame[
            [
                "Historical Mean Longest Dry Spell Days",
                "Historical Mean Post-Onset Heat Days 35 C",
                "Continuous Joint Exposure Rank",
            ]
        ].describe().to_string()
    )
    print(
        f"Primary support threshold={threshold95:.3f}; "
        f"supported={len(supported):,}; unsupported={len(unsupported):,}"
    )


if __name__ == "__main__":
    main()
