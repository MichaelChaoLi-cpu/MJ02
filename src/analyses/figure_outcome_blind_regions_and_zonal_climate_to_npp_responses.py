#!/usr/bin/env python3
"""Outcome-Blind Regions and Zonal Climate-to-NPP Responses.

Plan: Map the one frozen six-region SKATER partition and compare its
region-specific heat, Rx5day, and consecutive-dry-day slopes.
Framework: AnaSOP Sections 5-7 outcome-blind regionalisation and common-sample
regional slope-interaction models with village and year fixed effects,
village-clustered inference, and joint slope-equality tests.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from linearmodels.iv import AbsorbingLS
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/cses_public_village_cropland_npp_absolute_climate_panel_preprocessed.parquet"
REGIONS = ROOT / "data/processed/outcome_blind_spatial_regions_preprocessed.parquet"
COMMUNES = ROOT / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
OUTPUT = ROOT / "data/results/figures/Figure_outcome_blind_regions_and_zonal_climate_to_npp_responses.png"
EVIDENCE = ROOT / "data/exp/analysis/climate-npp/outcome-blind-regions-and-zonal-climate-to-npp-responses"

ID = "National Village Point ID"
LONGITUDE = "Point Longitude"
LATITUDE = "Point Latitude"
YEAR = "Year"
OUTCOME = "Annual Strict-Cropland Mean NPP kg C per m2"
HEAT = "Village Buffer Mean Annual Heat Days at or Above 35 C"
RX5DAY = "Village Buffer Mean Annual Maximum Consecutive Five-Day Precipitation Rx5day mm"
DRY_DAYS = "Village Buffer Mean Annual Maximum Consecutive Dry Days Below 1 mm"
RAIN_TOTAL = "Village Buffer Mean Annual Precipitation Total mm"

HEAT_LABEL = "Heat days >=35 C per 10 days"
RX5DAY_LABEL = "Rx5day per 10 mm"
DRY_LABEL = "Maximum dry spell per 10 days"
RAIN_LABEL = "Annual precipitation per 100 mm"

EXPOSURES = [
    (HEAT_LABEL, "Heat days ≥35°C", "o"),
    (RX5DAY_LABEL, "Rx5day", "s"),
    (DRY_LABEL, "Maximum dry spell", "D"),
]
REGION_COLORS = ["#173F5F", "#2F80A2", "#3A9D8F", "#D4A72C", "#D97757", "#9B4A63"]
REGION_SHORT_NAMES = {
    1: "North / northwest",
    2: "Western Tonle Sap",
    3: "East / northeast Mekong",
    4: "Lower Mekong / southeast",
    5: "South-central interior",
    6: "Southern coast",
}
DARK_GRAY = "#4D5960"
GRID_GRAY = "#D9DEE1"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.012,
        0.988,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=30,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.3},
    )


def load_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_columns = [ID, YEAR, OUTCOME, HEAT, RX5DAY, DRY_DAYS, RAIN_TOTAL]
    sample = pd.read_parquet(PANEL, columns=panel_columns)
    region_columns = [ID, LONGITUDE, LATITUDE, "SKATER Region ID"]
    regions = pd.read_parquet(REGIONS, columns=region_columns)
    sample = sample.merge(regions[[ID, "SKATER Region ID"]], on=ID, validate="many_to_one")
    for column in [OUTCOME, HEAT, RX5DAY, DRY_DAYS, RAIN_TOTAL]:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    sample = sample.loc[sample[YEAR].between(2001, 2021)].dropna().reset_index(drop=True)
    sample[HEAT_LABEL] = sample[HEAT] / 10.0
    sample[RX5DAY_LABEL] = sample[RX5DAY] / 10.0
    sample[DRY_LABEL] = sample[DRY_DAYS] / 10.0
    sample[RAIN_LABEL] = sample[RAIN_TOTAL] / 100.0
    return sample, regions


def fit_regional_model(sample: pd.DataFrame, algorithm: str) -> tuple[object, list[str], dict[str, list[str]]]:
    region_column = f"{algorithm} Region ID"
    regressors: dict[str, np.ndarray] = {}
    terms_by_exposure: dict[str, list[str]] = {}
    for exposure, _, _ in EXPOSURES:
        terms: list[str] = []
        for region_id in range(1, 7):
            term = f"{exposure} - Region {region_id}"
            regressors[term] = sample[exposure].to_numpy(dtype=float) * sample[region_column].eq(region_id)
            terms.append(term)
        terms_by_exposure[exposure] = terms
    regressors[RAIN_LABEL] = sample[RAIN_LABEL].to_numpy(dtype=float)
    exog = pd.DataFrame(regressors, index=sample.index)
    absorb = pd.DataFrame(
        {
            "Village fixed effect": sample[ID].astype("category"),
            "Calendar-year fixed effect": sample[YEAR].astype("category"),
        },
        index=sample.index,
    )
    clusters = pd.DataFrame(
        {"Village cluster": pd.Categorical(sample[ID]).codes},
        index=sample.index,
    )
    result = AbsorbingLS(
        sample[OUTCOME].astype(float),
        exog,
        absorb=absorb,
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)
    return result, list(exog.columns), terms_by_exposure


def coefficient_evidence(
    sample: pd.DataFrame,
    algorithm: str,
    result: object,
    regressors: list[str],
    terms_by_exposure: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    intervals = result.conf_int(level=0.95)
    coefficient_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    exposure_lookup = {key: label for key, label, _ in EXPOSURES}
    for exposure, terms in terms_by_exposure.items():
        for region_id, term in enumerate(terms, start=1):
            coefficient_rows.append(
                {
                    "Algorithm": algorithm,
                    "Exposure": exposure_lookup[exposure],
                    "Exposure Term": exposure,
                    "Region ID": region_id,
                    "Estimate": float(result.params[term]),
                    "Clustered Standard Error": float(result.std_errors[term]),
                    "95 Percent CI Lower": float(intervals.loc[term, "lower"]),
                    "95 Percent CI Upper": float(intervals.loc[term, "upper"]),
                    "Probability Value": float(result.pvalues[term]),
                    "Observations": len(sample),
                    "Villages": sample[ID].nunique(),
                    "Village Fixed Effects": True,
                    "Calendar-Year Fixed Effects": True,
                    "Village-Clustered Inference": True,
                }
            )
        restriction = np.zeros((len(terms) - 1, len(regressors)))
        for row, comparison_term in enumerate(terms[1:]):
            restriction[row, regressors.index(comparison_term)] = 1.0
            restriction[row, regressors.index(terms[0])] = -1.0
        test = result.wald_test(restriction=restriction, value=np.zeros(len(terms) - 1))
        test_rows.append(
            {
                "Algorithm": algorithm,
                "Exposure": exposure_lookup[exposure],
                "Null Hypothesis": "All six regional slopes are equal",
                "Wald Statistic": float(test.stat),
                "Degrees of Freedom": int(test.df),
                "Probability Value": float(test.pval),
            }
        )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(test_rows)


def map_context() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    communes = gpd.read_file(COMMUNES).to_crs(4326)
    provinces = communes.dissolve(by="pro_code").reset_index()
    cambodia = communes.dissolve()
    countries = gpd.read_file(COUNTRIES).to_crs(4326)
    region = countries.cx[101.5:108.5, 9.7:15.5].copy()
    return region, cambodia, provinces


def draw_region_map(
    ax: plt.Axes,
    regions: pd.DataFrame,
    algorithm: str,
    context: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame],
    label: str,
) -> None:
    surrounding, cambodia, provinces = context
    surrounding.plot(ax=ax, facecolor="#F0F2F1", edgecolor="#AAB2B6", linewidth=0.42, zorder=0)
    cambodia.plot(ax=ax, facecolor="#FFFEFA", edgecolor="#4D5960", linewidth=0.75, zorder=1)
    provinces.boundary.plot(ax=ax, color="#A7B0B4", linewidth=0.31, zorder=2)
    region_column = f"{algorithm} Region ID"
    for region_id in range(1, 7):
        selected = regions[region_column].eq(region_id)
        ax.scatter(
            regions.loc[selected, LONGITUDE],
            regions.loc[selected, LATITUDE],
            s=8.0,
            color=REGION_COLORS[region_id - 1],
            linewidths=0,
            alpha=0.94,
            rasterized=True,
            zorder=3,
        )
    ax.text(
        0.985,
        0.965,
        f"{algorithm} regions",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#173F5F",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
        zorder=10,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5.2,
            color=color,
            label=f"R{index + 1}  {REGION_SHORT_NAMES[index + 1]}",
        )
        for index, color in enumerate(REGION_COLORS)
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.82,
        fontsize=7.1,
        handletextpad=0.35,
        columnspacing=1.2,
    )
    ax.set_xlim(101.9, 108.0)
    ax.set_ylim(10.0, 15.05)
    ax.set_xticks(np.arange(102, 109, 1))
    ax.set_yticks(np.arange(10, 16, 1))
    ax.tick_params(labelsize=7.7, colors=DARK_GRAY, length=2.5)
    ax.grid(color=GRID_GRAY, linewidth=0.5, linestyle="--", zorder=-1)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color("#69767C")
        spine.set_linewidth(0.65)
    panel_label(ax, label)


def forest_positions() -> tuple[np.ndarray, list[str], dict[str, float]]:
    positions: list[float] = []
    labels: list[str] = []
    group_centres: dict[str, float] = {}
    cursor = 0.0
    for exposure, display, _ in EXPOSURES:
        group_positions: list[float] = []
        for region_id in range(1, 7):
            positions.append(cursor)
            labels.append(f"{display} · R{region_id}")
            group_positions.append(cursor)
            cursor += 1.0
        group_centres[display] = float(np.mean(group_positions))
        cursor += 0.9
    return np.asarray(positions), labels, group_centres


def draw_forest(
    ax: plt.Axes,
    coefficients: pd.DataFrame,
    tests: pd.DataFrame,
    algorithm: str,
    label: str,
    x_limits: tuple[float, float],
) -> None:
    positions, labels, group_centres = forest_positions()
    display = coefficients.loc[coefficients["Algorithm"].eq(algorithm)].copy()
    display["Sort"] = display["Exposure"].map({item[1]: index for index, item in enumerate(EXPOSURES)})
    display = display.sort_values(["Sort", "Region ID"]).reset_index(drop=True)
    if len(display) != len(positions):
        raise ValueError(f"Expected 18 regional coefficients for {algorithm}.")
    for y, (_, row) in zip(positions, display.iterrows(), strict=True):
        region_id = int(row["Region ID"])
        ax.hlines(
            y,
            row["95 Percent CI Lower"],
            row["95 Percent CI Upper"],
            color=REGION_COLORS[region_id - 1],
            linewidth=1.8,
            zorder=2,
        )
        marker = next(item[2] for item in EXPOSURES if item[1] == row["Exposure"])
        ax.scatter(
            row["Estimate"],
            y,
            s=34,
            marker=marker,
            color=REGION_COLORS[region_id - 1],
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    ax.axvline(0, color="#59656B", linewidth=0.9, zorder=1)
    for boundary in [5.45, 12.35]:
        ax.axhline(boundary, color="#C7CED1", linewidth=0.65, linestyle="--", zorder=0)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Change in annual cropland NPP (kg C m⁻² yr⁻¹)", fontsize=9)
    ax.set_xlim(*x_limits)
    ax.grid(axis="x", color=GRID_GRAY, linewidth=0.55, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#707A80")
    ax.tick_params(labelsize=7.6, colors=DARK_GRAY)
    ax.text(
        0.985,
        0.985,
        f"{algorithm}: regional slopes",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color="#173F5F",
        fontweight="bold",
    )
    for exposure, centre in group_centres.items():
        p_value = float(
            tests.loc[(tests["Algorithm"].eq(algorithm)) & (tests["Exposure"].eq(exposure)), "Probability Value"].iloc[0]
        )
        p_text = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
        ax.text(
            0.985,
            centre,
            f"joint {p_text}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.5,
            color="#5D686E",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
        )
    panel_label(ax, label)


def main() -> None:
    sample, regions = load_sample()
    coefficient_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    for algorithm in ["SKATER"]:
        result, regressors, terms = fit_regional_model(sample, algorithm)
        coefficients, tests = coefficient_evidence(sample, algorithm, result, regressors, terms)
        coefficient_frames.append(coefficients)
        test_frames.append(tests)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    tests = pd.concat(test_frames, ignore_index=True)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(EVIDENCE / "regional_climate_to_npp_coefficients.csv", index=False)
    tests.to_csv(EVIDENCE / "regional_slope_equality_tests.csv", index=False)
    region_support = []
    for algorithm in ["SKATER"]:
        region_column = f"{algorithm} Region ID"
        support = (
            sample.groupby(region_column, as_index=False)
            .agg(Observations=(ID, "size"), Villages=(ID, "nunique"))
            .rename(columns={region_column: "Region ID"})
        )
        support.insert(0, "Algorithm", algorithm)
        region_support.append(support)
    pd.concat(region_support, ignore_index=True).to_csv(EVIDENCE / "regional_stage1_support.csv", index=False)

    fig = plt.figure(figsize=(19.0, 9.3), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.24)
    map_axis = fig.add_subplot(grid[0, 0])
    forest_axis = fig.add_subplot(grid[0, 1])
    fig.subplots_adjust(left=0.045, right=0.985, top=0.965, bottom=0.075)
    context = map_context()
    lower = float(coefficients["95 Percent CI Lower"].min())
    upper = float(coefficients["95 Percent CI Upper"].max())
    padding = 0.04 * (upper - lower)
    x_limits = (lower - padding, upper + padding)
    draw_region_map(map_axis, regions, "SKATER", context, "a")
    draw_forest(forest_axis, coefficients, tests, "SKATER", "b", x_limits)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved evidence: {EVIDENCE.relative_to(ROOT)}")
    print(f"observations={len(sample):,}; villages={sample[ID].nunique():,}")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
