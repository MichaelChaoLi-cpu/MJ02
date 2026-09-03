#!/usr/bin/env python3
"""Continuous Spatial Heterogeneity.

Plan: Map continuously varying local slopes for heat and dry-spell exposure
against cropland NPP and for prior-year cropland NPP against total and food
consumption.
Framework: AnaSOP Sections 5-7 residualise outcomes and focal regressors using
the corresponding national controls and fixed effects, then apply an adaptive
bi-square geographically weighted local regression. Bandwidth is selected by
AICc without reference to local coefficient signs or significance; leave-one-
village-out prediction and effective-sample diagnostics are retained.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
from linearmodels.iv import AbsorbingLS
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
STAGE1_PANEL = ROOT / "data/processed/cses_public_village_cropland_npp_absolute_climate_panel_preprocessed.parquet"
STAGE2_HOUSEHOLDS = ROOT / "data/processed/cses_household_cropland_npp_analysis_preprocessed.parquet"
VILLAGES = ROOT / "data/processed/outcome_blind_spatial_regions_preprocessed.parquet"
COMMUNES = ROOT / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
OUTPUT = ROOT / "data/results/figures/Figure_continuous_spatial_heterogeneity.png"
APPENDIX_OUTPUT = (
    ROOT
    / "data/results/figures/Figure_continuous_npp_to_consumption_spatial_diagnostics.png"
)
EVIDENCE = ROOT / "data/exp/analysis/climate-npp/continuous-spatial-heterogeneity"

ID = "National Village Point ID"
LONGITUDE = "Point Longitude"
LATITUDE = "Point Latitude"
YEAR = "Year"

STAGE1_OUTCOME = "Annual Strict-Cropland Mean NPP kg C per m2"
HEAT = "Village Buffer Mean Annual Heat Days at or Above 35 C"
RX5DAY = "Village Buffer Mean Annual Maximum Consecutive Five-Day Precipitation Rx5day mm"
DRY_DAYS = "Village Buffer Mean Annual Maximum Consecutive Dry Days Below 1 mm"
RAIN_TOTAL = "Village Buffer Mean Annual Precipitation Total mm"

WEIGHT = "Household Survey Weight"
SURVEY_YEAR = "Interview Calendar Year"
SURVEY_MONTH = "Interview Month"
NPP = "Prior-Year Strict-Cropland NPP"
TOTAL_OUTCOME = "Log Real 2021 Annual Total Consumption per Capita"
FOOD_OUTCOME = "Log Real 2021 Annual Food Consumption per Capita"
TOTAL_FLAG = "Stage 2 Total Consumption Complete Case"
FOOD_FLAG = "Stage 2 Food Consumption Complete Case"
COMPOSITION_CONTROLS = [
    "Household Size",
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
]

BANDWIDTH_CANDIDATES = [15, 20, 30, 40, 50, 60, 90, 130, 180, 250, 350, 500, 700, 950, 1250, 1650, 2100, 2600]
MAP_EXTENT = (101.9, 108.0, 10.0, 15.05)
DARK_GRAY = "#4D5960"
GRID_GRAY = "#D9DEE1"
SPATIAL_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "blue_green_white_yellow_red",
    ["#2166AC", "#1A9850", "#FFFFFF", "#FEE08B", "#D73027"],
    N=256,
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.014,
        0.986,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=30,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.3},
    )


def project_coordinates(frame: pd.DataFrame) -> np.ndarray:
    points = gpd.GeoDataFrame(
        frame[[LONGITUDE, LATITUDE]].copy(),
        geometry=gpd.points_from_xy(frame[LONGITUDE], frame[LATITUDE]),
        crs=4326,
    ).to_crs(32648)
    return np.column_stack([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])


def fit_residuals(
    sample: pd.DataFrame,
    dependent: str,
    controls: list[str],
    absorb_columns: list[str],
    weights: str | None = None,
) -> pd.Series:
    exog = sample[controls].astype(float)
    absorb = pd.DataFrame(index=sample.index)
    for column in absorb_columns:
        absorb[column] = sample[column].astype("category")
    model_weights = None if weights is None else sample[weights].astype(float)
    result = AbsorbingLS(
        sample[dependent].astype(float),
        exog,
        absorb=absorb,
        weights=model_weights,
        drop_absorbed=True,
    ).fit(cov_type="robust", debiased=True)
    return pd.Series(np.asarray(result.resids), index=sample.index)


def sufficient_statistics(
    frame: pd.DataFrame,
    x: str,
    y: str,
    observation_weight: str | None,
) -> pd.DataFrame:
    work = frame[[ID, LONGITUDE, LATITUDE, x, y] + ([] if observation_weight is None else [observation_weight])].copy()
    work["Local Weight"] = 1.0 if observation_weight is None else work[observation_weight].astype(float)
    work["sw"] = work["Local Weight"]
    work["sx"] = work["Local Weight"] * work[x]
    work["sy"] = work["Local Weight"] * work[y]
    work["sxx"] = work["Local Weight"] * work[x] ** 2
    work["sxy"] = work["Local Weight"] * work[x] * work[y]
    work["syy"] = work["Local Weight"] * work[y] ** 2
    work["sw2"] = work["Local Weight"] ** 2
    stats = (
        work.groupby([ID, LONGITUDE, LATITUDE], as_index=False)[["sw", "sx", "sy", "sxx", "sxy", "syy", "sw2"]]
        .sum()
        .sort_values(ID)
        .reset_index(drop=True)
    )
    return stats


def kernel_weights(distances: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    selected = distances[:, :k]
    bandwidth = np.maximum(selected[:, [-1]] * 1.000001, 1e-8)
    scaled = selected / bandwidth
    weights = np.square(np.clip(1.0 - np.square(scaled), 0.0, None))
    if exclude_self:
        weights[:, 0] = 0.0
    return weights


def local_fit(
    stats: pd.DataFrame,
    indexes: np.ndarray,
    spatial_weights: np.ndarray,
    calculate_diagnostics: bool,
    chunk_size: int = 192,
) -> dict[str, np.ndarray | float]:
    values = stats[["sw", "sx", "sy", "sxx", "sxy", "syy", "sw2"]].to_numpy(dtype=float)
    n_targets = len(stats)
    intercept = np.full(n_targets, np.nan)
    slope = np.full(n_targets, np.nan)
    standard_error = np.full(n_targets, np.nan)
    effective_n = np.full(n_targets, np.nan)
    own_sse = np.full(n_targets, np.nan)
    trace_contribution = np.full(n_targets, np.nan)

    for start in range(0, n_targets, chunk_size):
        stop = min(start + chunk_size, n_targets)
        neighbor = indexes[start:stop]
        kernel = spatial_weights[start:stop]
        gathered = values[neighbor]
        sw = (kernel * gathered[:, :, 0]).sum(axis=1)
        sx = (kernel * gathered[:, :, 1]).sum(axis=1)
        sy = (kernel * gathered[:, :, 2]).sum(axis=1)
        sxx = (kernel * gathered[:, :, 3]).sum(axis=1)
        sxy = (kernel * gathered[:, :, 4]).sum(axis=1)
        syy = (kernel * gathered[:, :, 5]).sum(axis=1)
        sw2 = (kernel**2 * gathered[:, :, 6]).sum(axis=1)
        determinant = sw * sxx - sx**2
        valid = (sw > 0) & (determinant > 1e-12)
        local_intercept = np.full(stop - start, np.nan)
        local_slope = np.full(stop - start, np.nan)
        local_intercept[valid] = (sxx[valid] * sy[valid] - sx[valid] * sxy[valid]) / determinant[valid]
        local_slope[valid] = (sw[valid] * sxy[valid] - sx[valid] * sy[valid]) / determinant[valid]
        local_sse = (
            syy
            - 2 * local_intercept * sy
            - 2 * local_slope * sxy
            + local_intercept**2 * sw
            + 2 * local_intercept * local_slope * sx
            + local_slope**2 * sxx
        )
        local_effective = sw**2 / np.maximum(sw2, 1e-12)
        sigma2 = np.maximum(local_sse, 0) / np.maximum(local_effective - 2.0, 1.0)
        local_se = np.sqrt(np.maximum(sigma2 * sw / np.maximum(determinant, 1e-12), 0))

        intercept[start:stop] = local_intercept
        slope[start:stop] = local_slope
        standard_error[start:stop] = local_se
        effective_n[start:stop] = local_effective

        if calculate_diagnostics:
            own = values[start:stop]
            own_sse[start:stop] = (
                own[:, 5]
                - 2 * local_intercept * own[:, 2]
                - 2 * local_slope * own[:, 4]
                + local_intercept**2 * own[:, 0]
                + 2 * local_intercept * local_slope * own[:, 1]
                + local_slope**2 * own[:, 3]
            )
            trace_contribution[start:stop] = (
                sxx * own[:, 0] - 2 * sx * own[:, 1] + sw * own[:, 3]
            ) / np.maximum(determinant, 1e-12)

    output: dict[str, np.ndarray | float] = {
        "intercept": intercept,
        "slope": slope,
        "standard_error": standard_error,
        "effective_n": effective_n,
    }
    if calculate_diagnostics:
        output["rss"] = float(np.nansum(np.maximum(own_sse, 0)))
        output["trace_s"] = float(np.nansum(np.maximum(trace_contribution, 0)))
    return output


def aicc(rss: float, n_observations: int, trace_s: float) -> float:
    denominator = n_observations - 2.0 - trace_s
    if rss <= 0 or denominator <= 0:
        return float("inf")
    return float(
        n_observations * np.log(rss / n_observations)
        + n_observations * np.log(2 * np.pi)
        + n_observations * (n_observations + trace_s) / denominator
    )


def select_and_fit_surface(
    stats: pd.DataFrame,
    model_name: str,
    outcome_unit: str,
    raw_observations: int,
    slope_transform: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    coordinates = project_coordinates(stats)
    tree = cKDTree(coordinates)
    candidates = [value for value in BANDWIDTH_CANDIDATES if value <= len(stats)]
    maximum = max(candidates)
    distances, indexes = tree.query(coordinates, k=maximum)
    n_observations = raw_observations
    bandwidth_rows: list[dict[str, object]] = []

    for k in candidates:
        included_kernel = kernel_weights(distances, k, exclude_self=False)
        included = local_fit(stats, indexes[:, :k], included_kernel, calculate_diagnostics=True)
        excluded_kernel = kernel_weights(distances, k, exclude_self=True)
        excluded = local_fit(stats, indexes[:, :k], excluded_kernel, calculate_diagnostics=True)
        cv_rss = float(excluded["rss"])
        bandwidth_rows.append(
            {
                "Model": model_name,
                "Adaptive Neighbor Bandwidth": k,
                "AICc": aicc(float(included["rss"]), n_observations, float(included["trace_s"])),
                "Trace of Smoother": float(included["trace_s"]),
                "Leave-One-Village-Out RMSE": float(np.sqrt(cv_rss / max(n_observations, 1))),
                "Median Effective Local Sample": float(np.nanmedian(included["effective_n"])),
                "Valid Local Slopes": int(np.isfinite(included["slope"]).sum()),
            }
        )
    bandwidth = pd.DataFrame(bandwidth_rows)
    selected_k = int(bandwidth.loc[bandwidth["AICc"].idxmin(), "Adaptive Neighbor Bandwidth"])
    selected_kernel = kernel_weights(distances, selected_k, exclude_self=False)
    selected = local_fit(stats, indexes[:, :selected_k], selected_kernel, calculate_diagnostics=True)

    selected_position = candidates.index(selected_k)
    adjacent = sorted(
        set(
            [
                candidates[max(0, selected_position - 1)],
                selected_k,
                candidates[min(len(candidates) - 1, selected_position + 1)],
            ]
        )
    )
    sensitivity: dict[int, np.ndarray] = {}
    for k in adjacent:
        fitted = local_fit(
            stats,
            indexes[:, :k],
            kernel_weights(distances, k, exclude_self=False),
            calculate_diagnostics=False,
        )
        sensitivity[k] = np.asarray(fitted["slope"])
    sign_matrix = np.column_stack([np.sign(sensitivity[k]) for k in adjacent])
    stable_sign = np.all(sign_matrix == sign_matrix[:, [0]], axis=1)

    surface = stats[[ID, LONGITUDE, LATITUDE]].copy()
    surface["Model"] = model_name
    raw_slope = np.asarray(selected["slope"])
    raw_standard_error = np.asarray(selected["standard_error"])
    if slope_transform == "identity":
        surface["Local Slope"] = raw_slope
        surface["Local Standard Error"] = raw_standard_error
        surface["Local 95 Percent CI Lower"] = raw_slope - 1.96 * raw_standard_error
        surface["Local 95 Percent CI Upper"] = raw_slope + 1.96 * raw_standard_error
    elif slope_transform == "percent_per_0.1":
        transform = lambda value: 100.0 * (np.exp(0.1 * value) - 1.0)
        surface["Local Slope"] = transform(raw_slope)
        surface["Local Standard Error"] = 10.0 * np.exp(0.1 * raw_slope) * raw_standard_error
        surface["Local 95 Percent CI Lower"] = transform(raw_slope - 1.96 * raw_standard_error)
        surface["Local 95 Percent CI Upper"] = transform(raw_slope + 1.96 * raw_standard_error)
    else:
        raise ValueError(f"Unknown slope transformation: {slope_transform}")
    surface["Effective Local Sample"] = np.asarray(selected["effective_n"])
    surface["Selected Adaptive Neighbor Bandwidth"] = selected_k
    surface["Slope Sign Stable Across Adjacent Bandwidths"] = stable_sign
    for k, slopes in sensitivity.items():
        if slope_transform == "identity":
            surface[f"Local Slope at Bandwidth {k}"] = slopes
        else:
            surface[f"Local Slope at Bandwidth {k}"] = 100.0 * (np.exp(0.1 * slopes) - 1.0)

    selected_row = bandwidth.loc[bandwidth["Adaptive Neighbor Bandwidth"].eq(selected_k)].iloc[0]
    summary: dict[str, object] = {
        "model": model_name,
        "outcome_unit": outcome_unit,
        "locations": int(len(surface)),
        "underlying_weighted_observations": float(stats["sw"].sum()),
        "selected_adaptive_neighbor_bandwidth": selected_k,
        "selected_aicc": float(selected_row["AICc"]),
        "leave_one_village_out_rmse": float(selected_row["Leave-One-Village-Out RMSE"]),
        "median_effective_local_sample": float(surface["Effective Local Sample"].median()),
        "local_slope_quantiles": {
            str(probability): float(surface["Local Slope"].quantile(probability))
            for probability in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        },
        "positive_local_slope_share": float(surface["Local Slope"].gt(0).mean()),
        "negative_local_slope_share": float(surface["Local Slope"].lt(0).mean()),
        "sign_stable_across_adjacent_bandwidth_share": float(stable_sign.mean()),
        "adjacent_bandwidths": adjacent,
    }
    return surface, bandwidth, summary


def prepare_stage1(focal: str, model_name: str) -> tuple[pd.DataFrame, int]:
    columns = [ID, YEAR, STAGE1_OUTCOME, HEAT, RX5DAY, DRY_DAYS, RAIN_TOTAL]
    sample = pd.read_parquet(STAGE1_PANEL, columns=columns)
    for column in [STAGE1_OUTCOME, HEAT, RX5DAY, DRY_DAYS, RAIN_TOTAL]:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    sample = sample.loc[sample[YEAR].between(2001, 2021)].dropna().reset_index(drop=True)
    sample["Heat per 10 days"] = sample[HEAT] / 10.0
    sample["Rx5day per 10 mm"] = sample[RX5DAY] / 10.0
    sample["Dry spell per 10 days"] = sample[DRY_DAYS] / 10.0
    sample["Rain per 100 mm"] = sample[RAIN_TOTAL] / 100.0
    focal_name = "Heat per 10 days" if focal == HEAT else "Dry spell per 10 days"
    controls = ["Heat per 10 days", "Rx5day per 10 mm", "Dry spell per 10 days", "Rain per 100 mm"]
    controls.remove(focal_name)
    sample["Residual Outcome"] = fit_residuals(
        sample, STAGE1_OUTCOME, controls, [ID, YEAR]
    )
    sample["Residual Exposure"] = fit_residuals(
        sample, focal_name, controls, [ID, YEAR]
    )
    villages = pd.read_parquet(VILLAGES, columns=[ID, LONGITUDE, LATITUDE])
    sample = sample.merge(villages, on=ID, how="left", validate="many_to_one")
    stats = sufficient_statistics(sample, "Residual Exposure", "Residual Outcome", None)
    return stats, len(sample)


def prepare_stage2(outcome: str, flag: str) -> tuple[pd.DataFrame, int]:
    columns = [
        ID,
        WEIGHT,
        SURVEY_YEAR,
        SURVEY_MONTH,
        NPP,
        outcome,
        flag,
        *COMPOSITION_CONTROLS,
    ]
    sample = pd.read_parquet(STAGE2_HOUSEHOLDS, columns=columns)
    numeric = [WEIGHT, SURVEY_YEAR, SURVEY_MONTH, NPP, outcome, *COMPOSITION_CONTROLS]
    for column in numeric:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    sample["Exact Survey Time"] = (
        sample[SURVEY_YEAR].astype("Int64").astype("string")
        + "-"
        + sample[SURVEY_MONTH].astype("Int64").astype("string").str.zfill(2)
    )
    required = [ID, WEIGHT, "Exact Survey Time", NPP, outcome, *COMPOSITION_CONTROLS]
    sample = sample.loc[sample[flag]].dropna(subset=required).copy()
    keys = [ID, "Exact Survey Time"]
    for column in [NPP, outcome, *COMPOSITION_CONTROLS]:
        sample[f"Weighted {column}"] = sample[WEIGHT] * sample[column]
    cells = (
        sample.groupby(keys, as_index=False)
        .agg(
            **{
                "Survey Weight Sum": (WEIGHT, "sum"),
                "Households": (ID, "size"),
                **{
                    f"Weighted Sum {column}": (f"Weighted {column}", "sum")
                    for column in [NPP, outcome, *COMPOSITION_CONTROLS]
                },
            }
        )
    )
    for column in [NPP, outcome, *COMPOSITION_CONTROLS]:
        cells[column] = cells[f"Weighted Sum {column}"] / cells["Survey Weight Sum"]
    cells["Residual Outcome"] = fit_residuals(
        cells,
        outcome,
        COMPOSITION_CONTROLS,
        ["Exact Survey Time"],
        weights="Survey Weight Sum",
    )
    cells["Residual Exposure"] = fit_residuals(
        cells,
        NPP,
        COMPOSITION_CONTROLS,
        ["Exact Survey Time"],
        weights="Survey Weight Sum",
    )
    cells["Weighted Residual Outcome"] = cells["Survey Weight Sum"] * cells["Residual Outcome"]
    cells["Weighted Residual Exposure"] = cells["Survey Weight Sum"] * cells["Residual Exposure"]
    village_frame = (
        cells.groupby(ID, as_index=False)
        .agg(
            **{
                "Survey Weight Sum": ("Survey Weight Sum", "sum"),
                "Households": ("Households", "sum"),
                "Weighted Residual Outcome": ("Weighted Residual Outcome", "sum"),
                "Weighted Residual Exposure": ("Weighted Residual Exposure", "sum"),
                "Survey Cells": ("Exact Survey Time", "size"),
            }
        )
    )
    village_frame["Residual Outcome"] = (
        village_frame["Weighted Residual Outcome"] / village_frame["Survey Weight Sum"]
    )
    village_frame["Residual Exposure"] = (
        village_frame["Weighted Residual Exposure"] / village_frame["Survey Weight Sum"]
    )
    village_frame["Household Precision Weight"] = village_frame["Households"].astype(float)
    villages = pd.read_parquet(VILLAGES, columns=[ID, LONGITUDE, LATITUDE])
    village_frame = village_frame.merge(villages, on=ID, how="left", validate="one_to_one")
    stats = sufficient_statistics(
        village_frame,
        "Residual Exposure",
        "Residual Outcome",
        "Household Precision Weight",
    )
    return stats, len(stats)


def map_context() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    communes = gpd.read_file(COMMUNES).to_crs(4326)
    provinces = communes.dissolve(by="pro_code").reset_index()
    cambodia = communes.dissolve()
    countries = gpd.read_file(COUNTRIES).to_crs(4326)
    region = countries.cx[101.5:108.5, 9.7:15.5].copy()
    return region, cambodia, provinces


def draw_surface(
    ax: plt.Axes,
    surface: pd.DataFrame,
    surrounding: gpd.GeoDataFrame,
    cambodia: gpd.GeoDataFrame,
    provinces: gpd.GeoDataFrame,
    display_label: str,
    colorbar_label: str,
    panel: str,
    bound: float,
) -> None:
    surrounding.plot(ax=ax, facecolor="#F0F2F1", edgecolor="#AAB2B6", linewidth=0.42, zorder=0)
    cambodia.plot(ax=ax, facecolor="#FFFEFA", edgecolor="#4D5960", linewidth=0.75, zorder=1)
    provinces.boundary.plot(ax=ax, color="#A7B0B4", linewidth=0.31, zorder=2)
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    scatter = ax.scatter(
        surface[LONGITUDE],
        surface[LATITUDE],
        c=surface["Local Slope"],
        cmap=SPATIAL_DIVERGING_CMAP,
        norm=norm,
        s=8.5,
        alpha=0.94,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_xticks(np.arange(102, 109, 1))
    ax.set_yticks(np.arange(10, 16, 1))
    ax.grid(color=GRID_GRAY, linewidth=0.5, linestyle="--", zorder=-1)
    ax.tick_params(labelsize=7.7, colors=DARK_GRAY, length=2.5)
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color("#69767C")
        spine.set_linewidth(0.65)
    ax.text(
        0.985,
        0.965,
        display_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color="#173F5F",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.4},
        zorder=10,
    )
    colorbar = ax.figure.colorbar(scatter, ax=ax, orientation="vertical", pad=0.012, shrink=0.76, aspect=24)
    colorbar.set_label(colorbar_label, fontsize=7.9, color=DARK_GRAY, labelpad=5)
    colorbar.ax.tick_params(labelsize=7.2, colors=DARK_GRAY, length=2.2)
    colorbar.outline.set_linewidth(0.45)
    panel_label(ax, panel)


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    model_inputs = [
        (
            "Heat to cropland NPP",
            *prepare_stage1(HEAT, "Heat to cropland NPP"),
            "NPP change per 10 heat days",
            "identity",
        ),
        (
            "Dry spell to cropland NPP",
            *prepare_stage1(DRY_DAYS, "Dry spell to cropland NPP"),
            "NPP change per 10 dry days",
            "identity",
        ),
        (
            "Cropland NPP to total consumption",
            *prepare_stage2(TOTAL_OUTCOME, TOTAL_FLAG),
            "Percent difference per 0.1 NPP",
            "percent_per_0.1",
        ),
        (
            "Cropland NPP to food consumption",
            *prepare_stage2(FOOD_OUTCOME, FOOD_FLAG),
            "Percent difference per 0.1 NPP",
            "percent_per_0.1",
        ),
    ]

    surfaces: list[pd.DataFrame] = []
    bandwidths: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for model_name, stats, raw_observations, outcome_unit, transform in model_inputs:
        surface, bandwidth, summary = select_and_fit_surface(
            stats,
            model_name,
            outcome_unit,
            raw_observations=raw_observations,
            slope_transform=transform,
        )
        summary["raw_observations"] = raw_observations
        surfaces.append(surface)
        bandwidths.append(bandwidth)
        summaries.append(summary)
        print(
            f"{model_name}: locations={len(surface):,}; "
            f"bandwidth={summary['selected_adaptive_neighbor_bandwidth']}; "
            f"median_effective_n={summary['median_effective_local_sample']:.1f}"
        )

    surface_frame = pd.concat(surfaces, ignore_index=True)
    bandwidth_frame = pd.concat(bandwidths, ignore_index=True)
    surface_frame.to_parquet(EVIDENCE / "continuous_local_slope_surfaces.parquet", index=False)
    bandwidth_frame.to_csv(EVIDENCE / "adaptive_bandwidth_aicc_and_prediction_diagnostics.csv", index=False)
    (EVIDENCE / "continuous_spatial_heterogeneity_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    top_bound = float(
        np.quantile(
            np.abs(
                surface_frame.loc[
                    surface_frame["Model"].isin(["Heat to cropland NPP", "Dry spell to cropland NPP"]),
                    "Local Slope",
                ]
            ),
            0.98,
        )
    )
    bottom_bound = float(
        np.quantile(
            np.abs(
                surface_frame.loc[
                    surface_frame["Model"].isin(
                        ["Cropland NPP to total consumption", "Cropland NPP to food consumption"]
                    ),
                    "Local Slope",
                ]
            ),
            0.98,
        )
    )
    context = map_context()
    main_specifications = [
        ("Heat to cropland NPP", "Heat days ≥35°C → cropland NPP", "Local NPP slope per 10 heat days", "a", top_bound),
        ("Dry spell to cropland NPP", "Maximum dry spell → cropland NPP", "Local NPP slope per 10 dry days", "b", top_bound),
    ]
    appendix_specifications = [
        (
            "Cropland NPP to total consumption",
            "Prior-year cropland NPP → total consumption",
            "Local percent difference per 0.1 NPP",
            "a",
            bottom_bound,
        ),
        (
            "Cropland NPP to food consumption",
            "Prior-year cropland NPP → food consumption",
            "Local percent difference per 0.1 NPP",
            "b",
            bottom_bound,
        ),
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    for output_path, specifications in [
        (OUTPUT, main_specifications),
        (APPENDIX_OUTPUT, appendix_specifications),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.35), facecolor="white")
        fig.subplots_adjust(left=0.045, right=0.965, top=0.97, bottom=0.08, wspace=0.17)
        for ax, (model_name, display_label, colorbar_label, panel, bound) in zip(
            axes.flat, specifications, strict=True
        ):
            surface = surface_frame.loc[surface_frame["Model"].eq(model_name)]
            draw_surface(ax, surface, *context, display_label, colorbar_label, panel, bound)
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {output_path.relative_to(ROOT)}")
    print(f"Saved evidence: {EVIDENCE.relative_to(ROOT)}")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
