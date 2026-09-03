#!/usr/bin/env python3
"""Explore why high drought-placebo coefficients cluster spatially.

This is an exploratory data briefing, not a confirmatory result. It maps the
124 random two-sector assignments whose drought coefficient is at least the
actual-conflict estimate and compares their predetermined geography, baseline
land system, settlement, rainfall, drought, and pre-conflict NPP features with
the remaining assignments.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/exp/.matplotlib"))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(ROOT / "src/analyses"))
from table_predetermined_balance_and_common_support import build_sample  # noqa: E402


PLACEBO_DIR = (
    ROOT
    / "data/exp/experiments/cambodia-thailand-village-area-itt-climate"
    / "drought-random-pair-placebo"
)
ASSIGNMENTS = PLACEBO_DIR / "drought_random_pair_village_assignments.csv"
ESTIMATES = PLACEBO_DIR / "drought_random_pair_estimates.csv"
INFERENCE = PLACEBO_DIR / "drought_random_pair_inference.csv"
PANEL = ROOT / "data/processed/cambodia_public_village_npp_conflict_panel_candidate_preprocessed.parquet"
COMMUNES = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"

OUT = ROOT / "data/exp/data-briefing/drought-placebo-geography"
ASSIGNMENT_FEATURES_OUT = OUT / "assignment_level_diagnostic_features.csv"
COMPARISON_OUT = OUT / "high_coefficient_placebo_feature_comparison.csv"
HIGH_ASSIGNMENTS_OUT = OUT / "high_coefficient_placebo_assignments.csv"
FIGURE_OUT = OUT / "high_coefficient_placebo_geography.png"
METADATA_OUT = OUT / "diagnostic_metadata.json"

ID = "National Village Point ID"
YEAR = "Year"
SECTOR = "Conflict Sector"
DRY = "Village Buffer Mean May October Dry Rainfall Intensity"
RAIN = "Village Buffer Mean May October Precipitation Total mm"
FOREST = "Village Buffer Mean Baseline Forest Share"
POP_DENSITY = "Village Buffer Mean Baseline Population Density per km2"

SECTOR_YEARS = {"Preah Vihear": 2008, "Ta Moan-Ta Krabey": 2011}

BASELINE_RENAME = {
    "Point Longitude": "Mean village longitude",
    "Point Latitude": "Mean village latitude",
    "Border distance": "Mean border distance km",
    "Elevation": "Mean elevation m",
    "Slope": "Mean slope degrees",
    "Historical-road distance": "Mean historical-road distance km",
    "Baseline cropland share": "Mean baseline cropland share",
    "Log baseline population": "Mean log baseline population",
    "Pre-conflict NPP mean": "Mean pre-conflict NPP",
    "Pre-conflict NPP trend": "Mean pre-conflict NPP trend",
    "Pre-conflict NPP SD": "Mean pre-conflict NPP SD",
    "Pre-conflict rainfall mean": "Mean common-period pre-conflict rainfall mm",
    "Pre-conflict rainfall SD": "Mean common-period pre-conflict rainfall SD",
    "Pre-conflict dry intensity": "Mean common-period pre-conflict dry intensity",
    "Pre-conflict heat intensity": "Mean common-period pre-conflict heat intensity",
    FOREST: "Mean baseline forest share",
    POP_DENSITY: "Mean baseline population density per km2",
}

COMPARISON_FEATURES = [
    "Mean village latitude",
    "Mean village longitude",
    "Mean border distance km",
    "Mean elevation m",
    "Mean slope degrees",
    "Mean historical-road distance km",
    "Mean baseline cropland share",
    "Mean baseline forest share",
    "Mean log baseline population",
    "Mean baseline population density per km2",
    "Mean common-period pre-conflict rainfall mm",
    "Mean common-period pre-conflict rainfall SD",
    "Mean common-period pre-conflict dry intensity",
    "Mean common-period pre-conflict heat intensity",
    "Mean assigned-period pre drought intensity",
    "Mean assigned-period post drought intensity",
    "Post-minus-pre drought intensity",
    "Mean assigned-period pre rainfall mm",
    "Mean assigned-period post rainfall mm",
    "Post-minus-pre rainfall mm",
    "Mean pre-conflict NPP",
    "Mean pre-conflict NPP trend",
    "Mean pre-conflict NPP SD",
    "Anchor separation km",
    "Mean zone radius km",
    "Maximum zone radius km",
]


def load_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    treated, controls = build_sample()
    baseline = pd.concat([treated, controls], ignore_index=True).copy()
    baseline[ID] = baseline[ID].astype(str)
    static = pd.read_parquet(
        PANEL,
        columns=[ID, YEAR, FOREST, POP_DENSITY],
        filters=[("Buffer Radius km", "=", 5), (YEAR, "=", 2001)],
    ).drop_duplicates(ID)
    static[ID] = static[ID].astype(str)
    baseline = baseline.merge(
        static[[ID, FOREST, POP_DENSITY]],
        on=ID,
        how="left",
        validate="one_to_one",
    )
    return baseline, treated


def build_sector_climate() -> pd.DataFrame:
    panel = pd.read_parquet(
        PANEL,
        columns=[ID, YEAR, DRY, RAIN],
        filters=[("Buffer Radius km", "=", 5)],
    )
    panel[ID] = panel[ID].astype(str)
    rows: list[pd.DataFrame] = []
    for sector, onset in SECTOR_YEARS.items():
        sample = panel.assign(period=np.where(panel[YEAR].lt(onset), "pre", "post"))
        summary = (
            sample.groupby([ID, "period"])[[DRY, RAIN]]
            .mean()
            .unstack("period")
        )
        summary.columns = [f"{variable}__{period}" for variable, period in summary.columns]
        summary = summary.reset_index()
        summary[SECTOR] = sector
        summary["Assigned-period pre drought intensity"] = summary[f"{DRY}__pre"]
        summary["Assigned-period post drought intensity"] = summary[f"{DRY}__post"]
        summary["Assigned-period pre rainfall mm"] = summary[f"{RAIN}__pre"]
        summary["Assigned-period post rainfall mm"] = summary[f"{RAIN}__post"]
        rows.append(
            summary[
                [
                    ID,
                    SECTOR,
                    "Assigned-period pre drought intensity",
                    "Assigned-period post drought intensity",
                    "Assigned-period pre rainfall mm",
                    "Assigned-period post rainfall mm",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def aggregate_assignment_features(
    assignments: pd.DataFrame,
    baseline: pd.DataFrame,
    climate: pd.DataFrame,
) -> pd.DataFrame:
    assignments = assignments.copy()
    assignments[ID] = assignments[ID].astype(str)
    baseline_columns = [ID, *BASELINE_RENAME]
    enriched = assignments.merge(
        baseline[baseline_columns],
        on=ID,
        how="left",
        validate="many_to_one",
    ).merge(
        climate,
        on=[ID, SECTOR],
        how="left",
        validate="many_to_one",
    )
    enriched = enriched.rename(columns=BASELINE_RENAME)
    mean_columns = [
        *BASELINE_RENAME.values(),
        "Assigned-period pre drought intensity",
        "Assigned-period post drought intensity",
        "Assigned-period pre rainfall mm",
        "Assigned-period post rainfall mm",
    ]
    grouped = (
        enriched.groupby("Simulation ID", as_index=False)[mean_columns]
        .mean()
    )
    anchor_summary = (
        enriched.groupby("Simulation ID", as_index=False)
        .agg(
            **{
                "Anchor separation km": ("Anchor Separation km", "first"),
                "Mean zone radius km": ("Anchor-to-village Distance km", "mean"),
                "Maximum zone radius km": ("Anchor-to-village Distance km", "max"),
            }
        )
    )
    grouped = grouped.merge(
        anchor_summary,
        on="Simulation ID",
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "Assigned-period pre drought intensity": "Mean assigned-period pre drought intensity",
            "Assigned-period post drought intensity": "Mean assigned-period post drought intensity",
            "Assigned-period pre rainfall mm": "Mean assigned-period pre rainfall mm",
            "Assigned-period post rainfall mm": "Mean assigned-period post rainfall mm",
        }
    )
    grouped["Post-minus-pre drought intensity"] = (
        grouped["Mean assigned-period post drought intensity"]
        - grouped["Mean assigned-period pre drought intensity"]
    )
    grouped["Post-minus-pre rainfall mm"] = (
        grouped["Mean assigned-period post rainfall mm"]
        - grouped["Mean assigned-period pre rainfall mm"]
    )
    return grouped


def actual_features(
    treated: pd.DataFrame,
    baseline: pd.DataFrame,
    climate: pd.DataFrame,
) -> pd.Series:
    actual = treated[[ID, "Candidate Conflict Sector"]].copy()
    actual[ID] = actual[ID].astype(str)
    actual = actual.rename(columns={"Candidate Conflict Sector": SECTOR}).merge(
        baseline[[ID, *BASELINE_RENAME]],
        on=ID,
        how="left",
        validate="one_to_one",
    ).merge(
        climate,
        on=[ID, SECTOR],
        how="left",
        validate="one_to_one",
    )
    actual = actual.rename(columns=BASELINE_RENAME)
    values: dict[str, float] = {}
    for feature in BASELINE_RENAME.values():
        values[feature] = float(actual[feature].mean())
    values["Mean assigned-period pre drought intensity"] = float(
        actual["Assigned-period pre drought intensity"].mean()
    )
    values["Mean assigned-period post drought intensity"] = float(
        actual["Assigned-period post drought intensity"].mean()
    )
    values["Post-minus-pre drought intensity"] = (
        values["Mean assigned-period post drought intensity"]
        - values["Mean assigned-period pre drought intensity"]
    )
    values["Mean assigned-period pre rainfall mm"] = float(
        actual["Assigned-period pre rainfall mm"].mean()
    )
    values["Mean assigned-period post rainfall mm"] = float(
        actual["Assigned-period post rainfall mm"].mean()
    )
    values["Post-minus-pre rainfall mm"] = (
        values["Mean assigned-period post rainfall mm"]
        - values["Mean assigned-period pre rainfall mm"]
    )
    values["Anchor separation km"] = np.nan
    values["Mean zone radius km"] = np.nan
    values["Maximum zone radius km"] = np.nan
    return pd.Series(values)


def comparison_table(
    diagnostics: pd.DataFrame,
    actual: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    high = diagnostics.loc[diagnostics["High coefficient assignment"]]
    other = diagnostics.loc[~diagnostics["High coefficient assignment"]]
    for feature in COMPARISON_FEATURES:
        high_values = high[feature].astype(float)
        other_values = other[feature].astype(float)
        pooled_sd = float(
            np.sqrt((high_values.var(ddof=1) + other_values.var(ddof=1)) / 2)
        )
        difference = float(high_values.mean() - other_values.mean())
        correlation = float(
            diagnostics[[feature, "Drought Triple-Interaction Estimate"]]
            .corr()
            .iloc[0, 1]
        )
        rows.append(
            {
                "Feature": feature,
                "Actual conflict geography": float(actual.get(feature, np.nan)),
                "High-coefficient assignments mean": float(high_values.mean()),
                "Other assignments mean": float(other_values.mean()),
                "High minus other": difference,
                "Pooled assignment SD": pooled_sd,
                "Standardized high-minus-other difference": difference / pooled_sd
                if pooled_sd > 0
                else np.nan,
                "Correlation with placebo coefficient": correlation,
                "High assignments with nonmissing feature": int(high_values.notna().sum()),
                "Other assignments with nonmissing feature": int(other_values.notna().sum()),
            }
        )
    output = pd.DataFrame(rows)
    output["Absolute standardized difference"] = output[
        "Standardized high-minus-other difference"
    ].abs()
    return output.sort_values(
        ["Absolute standardized difference", "Feature"],
        ascending=[False, True],
    ).reset_index(drop=True)


def draw_map(
    assignments: pd.DataFrame,
    diagnostics: pd.DataFrame,
    treated: pd.DataFrame,
    actual_estimate: float,
) -> None:
    map_frame = assignments.merge(
        diagnostics[
            [
                "Simulation ID",
                "Drought Triple-Interaction Estimate",
                "High coefficient assignment",
            ]
        ],
        on="Simulation ID",
        how="left",
        validate="many_to_one",
    )
    centroids = (
        map_frame.groupby(["Simulation ID", SECTOR], as_index=False)
        .agg(
            **{
                "Sector centroid longitude": ("Village Longitude", "mean"),
                "Sector centroid latitude": ("Village Latitude", "mean"),
                "Drought Triple-Interaction Estimate": (
                    "Drought Triple-Interaction Estimate",
                    "first",
                ),
                "High coefficient assignment": ("High coefficient assignment", "first"),
            }
        )
    )
    actual_centroids = (
        treated.groupby("Candidate Conflict Sector", as_index=False)
        .agg(
            **{
                "Sector centroid longitude": ("Point Longitude", "mean"),
                "Sector centroid latitude": ("Point Latitude", "mean"),
            }
        )
    )
    communes = gpd.read_file(COMMUNES).to_crs(4326)
    provinces = communes.dissolve(by="ADM1_PCODE")
    country = communes.dissolve()

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 6.0),
        gridspec_kw={"width_ratios": [1.25, 1]},
        constrained_layout=True,
    )
    country.plot(ax=axes[0], color="#F8F8F5", edgecolor="#3F3F3F", linewidth=0.9)
    provinces.boundary.plot(ax=axes[0], color="#C5C7C9", linewidth=0.45)
    ordinary = centroids.loc[~centroids["High coefficient assignment"]]
    high = centroids.loc[centroids["High coefficient assignment"]]
    axes[0].scatter(
        ordinary["Sector centroid longitude"],
        ordinary["Sector centroid latitude"],
        s=7,
        color="#8D979C",
        alpha=0.16,
        linewidth=0,
        label="Other random-sector centres",
        zorder=3,
    )
    axes[0].scatter(
        high["Sector centroid longitude"],
        high["Sector centroid latitude"],
        s=18,
        color="#D88727",
        alpha=0.65,
        linewidth=0,
        label="Centres from 124 high assignments",
        zorder=4,
    )
    sector_colors = {"Preah Vihear": "#B2182B", "Ta Moan-Ta Krabey": "#6A3D9A"}
    for _, row in actual_centroids.iterrows():
        sector = row["Candidate Conflict Sector"]
        axes[0].scatter(
            row["Sector centroid longitude"],
            row["Sector centroid latitude"],
            marker="*",
            s=145,
            color=sector_colors[sector],
            edgecolor="white",
            linewidth=0.8,
            label=f"Actual {sector}",
            zorder=6,
        )
    axes[0].set_xlim(102.2, 107.8)
    axes[0].set_ylim(9.7, 14.8)
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    axes[0].set_xticks(np.arange(103, 108, 1))
    axes[0].set_yticks(np.arange(10, 15, 1))
    axes[0].grid(True, linestyle="--", color="#D5D8DA", linewidth=0.5)
    axes[0].legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=7,
    )

    plot_data = diagnostics.copy()
    plot_data["Latitude quartile"] = pd.qcut(
        plot_data["Mean village latitude"],
        4,
        labels=["Q1 south", "Q2", "Q3", "Q4 north"],
    )
    sns.boxplot(
        data=plot_data,
        x="Latitude quartile",
        y="Drought Triple-Interaction Estimate",
        color="#A9B4B9",
        width=0.62,
        fliersize=2,
        linewidth=0.8,
        ax=axes[1],
    )
    axes[1].axhline(
        actual_estimate,
        color="#C4493D",
        linewidth=1.8,
        linestyle="--",
        label=f"Actual estimate: {actual_estimate:.3f}",
    )
    axes[1].axhline(0, color="#5A5A5A", linewidth=0.8)
    axes[1].set_xlabel("Random assignment latitude quartile")
    axes[1].set_ylabel("Drought slope-change coefficient")
    axes[1].grid(True, axis="y", color="#E3E7EA", linewidth=0.6)
    axes[1].grid(False, axis="x")
    axes[1].legend(frameon=False, loc="upper left", fontsize=8)

    for label, axis in zip("ab", axes):
        axis.text(
            -0.10,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )
    FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(ASSIGNMENTS, dtype={ID: str})
    estimates = pd.read_csv(ESTIMATES)
    inference = pd.read_csv(INFERENCE).iloc[0]
    actual_estimate = float(inference["Actual Common-Specification Drought Estimate"])
    baseline, treated = load_baseline()
    climate = build_sector_climate()
    features = aggregate_assignment_features(assignments, baseline, climate).merge(
        estimates,
        on="Simulation ID",
        how="left",
        validate="one_to_one",
    )
    features["High coefficient assignment"] = features[
        "Drought Triple-Interaction Estimate"
    ].ge(actual_estimate)
    if int(features["High coefficient assignment"].sum()) != 124:
        raise RuntimeError("High-coefficient assignment count no longer matches frozen inference")
    actual = actual_features(treated, baseline, climate)
    comparison = comparison_table(features, actual)

    features.to_csv(ASSIGNMENT_FEATURES_OUT, index=False)
    comparison.to_csv(COMPARISON_OUT, index=False)
    features.loc[features["High coefficient assignment"]].to_csv(
        HIGH_ASSIGNMENTS_OUT,
        index=False,
    )
    draw_map(assignments, features, treated, actual_estimate)

    top_features = comparison.head(8)[
        [
            "Feature",
            "Standardized high-minus-other difference",
            "Correlation with placebo coefficient",
        ]
    ].to_dict("records")
    metadata = {
        "status": "exploratory diagnostic; not confirmatory evidence",
        "human_decision_record": "MILI-D-20260823-014",
        "actual_estimate_threshold": actual_estimate,
        "high_assignments": int(features["High coefficient assignment"].sum()),
        "other_assignments": int((~features["High coefficient assignment"]).sum()),
        "features_compared": COMPARISON_FEATURES,
        "land_cover_limit": "baseline cropland and forest shares only; no annual land-cover transition is tested",
        "population_limit": "baseline population only; no migration mechanism is tested",
        "comparison_rule": "high assignments have placebo drought coefficient greater than or equal to actual common-specification coefficient",
        "top_features_by_absolute_standardized_difference": top_features,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    METADATA_OUT.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {FIGURE_OUT.relative_to(ROOT)}")
    print(f"Saved: {COMPARISON_OUT.relative_to(ROOT)}")
    print(f"Saved: {ASSIGNMENT_FEATURES_OUT.relative_to(ROOT)}")
    print("\nLargest high-versus-other standardized differences")
    print(
        comparison.head(12)[
            [
                "Feature",
                "Actual conflict geography",
                "High-coefficient assignments mean",
                "Other assignments mean",
                "Standardized high-minus-other difference",
                "Correlation with placebo coefficient",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
