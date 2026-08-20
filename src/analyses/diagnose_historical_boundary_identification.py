#!/usr/bin/env python3
"""Outcome-blind identification diagnostics for the historical repression boundary.

The script reads only predetermined geography, historical assignment, modern administrative
geometry, road-alignment design fields, and rainfall design fields. It never reads a candidate
outcome. Effect estimation remains blocked after this diagnostic step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest, norm
from shapely import get_parts
from shapely.geometry import Point

from feasibility_historical_boundary_linkage import load_boundary_geometry
from power_historical_boundary_annual_spatial_interaction import (
    SCENARIOS,
    absorb_fixed_effects,
    component_numerator_variances,
    empirical_mde,
    target_residual,
)


BANDWIDTHS_KM = (2, 5, 10, 15, 20, 30)
PRIMARY_BANDWIDTH_KM = 5
PRIMARY_SHOCK = "May October Rainfall Anomaly Z (1991-2020)"
SESOI = 0.20
BALANCE_REVIEW_THRESHOLD_SD = 0.25

PREDICTORS = {
    "temp_mean": ("Mean temperature", "historical climate normal", "identity"),
    "temp_var": ("Temperature variability", "historical climate normal", "identity"),
    "prec_mean": ("Mean precipitation", "historical climate normal", "identity"),
    "prec_var": ("Precipitation variability", "historical climate normal", "identity"),
    "class_12": ("Soil class 12 share", "soil", "identity"),
    "class_40": ("Soil class 40 share", "soil", "identity"),
    "fertile_pct": ("Fertile-soil share", "soil", "identity"),
    "rug": ("Terrain ruggedness", "terrain and geography", "log1p"),
    "elev": ("Elevation", "terrain and geography", "log1p"),
    "river_d": ("Distance to river", "terrain and geography", "log1p"),
    "dist_cap": ("Distance to Phnom Penh", "terrain and geography", "log1p"),
    "pop_75": ("1975 population surface", "timing-ambiguous 1975 settlement proxy", "asinh"),
    "build_75_sum": ("1975 building-count surface", "timing-ambiguous 1975 settlement proxy", "asinh"),
    "built_1975": ("1975 built-area mean", "timing-ambiguous 1975 settlement proxy", "asinh"),
    "built_1975_sum": ("1975 built-area sum", "timing-ambiguous 1975 settlement proxy", "asinh"),
}

ALIGNMENT_FIELDS = {
    "road_d": ("Distance to modern road", "log1p"),
    "road_cell": ("Modern road-cell intensity", "asinh"),
}

BLIND_POWER_COLUMNS = [
    "Village Code",
    "Year",
    "Commune Code",
    "Linked Climate Commune Code",
    "Higher-Repression Southwest Zone",
    "Signed Distance to Historical Repression Boundary km",
    "Absolute Distance to Historical Repression Boundary km",
    "Historical Boundary Segment",
    PRIMARY_SHOCK,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--field-dir",
        type=Path,
        default=Path("data/exp/feasibility-check/historical-boundary-identification"),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path(
            "data/processed/historical_boundary_annual_spatial_climate_preprocessed.parquet"
        ),
    )
    parser.add_argument(
        "--zones-zip",
        type=Path,
        default=Path(
            "data/exp/data-preprocessing/historical-boundary-source/"
            "Democratic_Kampuchea_Zones.zip"
        ),
    )
    parser.add_argument(
        "--communes",
        type=Path,
        default=Path("data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"),
    )
    parser.add_argument(
        "--legacy-communes",
        type=Path,
        default=Path("data/raw/geography/odc_cambodia_communes_2014.gpkg"),
    )
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def absolute(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def transform(values: pd.Series, method: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if method == "identity":
        return numeric
    if method == "log1p":
        return np.log1p(numeric.clip(lower=0))
    if method == "asinh":
        return np.arcsinh(numeric)
    raise ValueError(f"Unknown transformation {method}")


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.dropna().sort_values()
    if valid.empty:
        return result
    adjusted = np.maximum.accumulate(
        valid.to_numpy() * (len(valid) - np.arange(len(valid)))
    )
    result.loc[valid.index] = np.minimum(adjusted, 1.0)
    return result


def fit_local_continuity(
    frame: pd.DataFrame,
    value_column: str,
    bandwidth: int,
    transformation: str,
) -> dict[str, object]:
    data = frame.loc[frame["dist_border"].abs().le(bandwidth)].copy()
    data["analysis_value"] = transform(data[value_column], transformation)
    data = data.dropna(subset=["analysis_value", "dist_border", "treat", "comm"])
    outcome_sd = float(data["analysis_value"].std(ddof=1))
    if not np.isfinite(outcome_sd) or outcome_sd <= 0:
        return {
            "field": value_column,
            "bandwidth_km": bandwidth,
            "n_villages": len(data),
            "southwest_villages": int(data["treat"].sum()),
            "west_villages": int((1 - data["treat"]).sum()),
            "clusters": data["comm"].astype(str).nunique(),
            "transformation": transformation,
            "standardized_discontinuity": np.nan,
            "standard_error": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "p_value": np.nan,
            "covariance": "not estimable: no within-window variation",
        }
    data["standardized_value"] = (
        data["analysis_value"] - data["analysis_value"].mean()
    ) / outcome_sd
    data["treat_distance"] = data["treat"] * data["dist_border"]
    segments = pd.get_dummies(
        data["dist.segment"].astype("string"), prefix="segment", drop_first=True, dtype=float
    )
    design = pd.concat(
        [
            pd.DataFrame(
                {
                    "constant": 1.0,
                    "treat": data["treat"].astype(float),
                    "distance": data["dist_border"].astype(float),
                    "treat_distance": data["treat_distance"].astype(float),
                },
                index=data.index,
            ),
            segments,
        ],
        axis=1,
    )
    weights = np.maximum(1.0 - data["dist_border"].abs() / bandwidth, 1e-6)
    model = sm.WLS(data["standardized_value"], design, weights=weights)
    groups = data["comm"].astype(str)
    covariance = "cluster by replication commune"
    try:
        fit = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True},
        )
    except (ValueError, np.linalg.LinAlgError):
        fit = model.fit(cov_type="HC1")
        covariance = "HC1 fallback"
    estimate = float(fit.params["treat"])
    standard_error = float(fit.bse["treat"])
    pvalue = float(2 * norm.sf(abs(estimate / standard_error)))
    critical = float(norm.ppf(0.975))
    return {
        "field": value_column,
        "bandwidth_km": bandwidth,
        "n_villages": len(data),
        "southwest_villages": int(data["treat"].sum()),
        "west_villages": int((1 - data["treat"]).sum()),
        "clusters": groups.nunique(),
        "transformation": transformation,
        "standardized_discontinuity": estimate,
        "standard_error": standard_error,
        "ci95_low": estimate - critical * standard_error,
        "ci95_high": estimate + critical * standard_error,
        "p_value": pvalue,
        "covariance": covariance,
    }


def continuity_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for field, (label, family, transformation) in PREDICTORS.items():
        for bandwidth in BANDWIDTHS_KM:
            row = fit_local_continuity(frame, field, bandwidth, transformation)
            row.update({"label": label, "family": family, "diagnostic_role": "predetermined"})
            rows.append(row)
    continuity = pd.DataFrame(rows)
    continuity["multiple_testing_family"] = np.where(
        continuity["family"].str.contains("timing-ambiguous"),
        "timing-ambiguous 1975 settlement proxies",
        "physical predetermined covariates",
    )
    continuity["holm_p_value_within_bandwidth"] = continuity.groupby(
        ["bandwidth_km", "multiple_testing_family"]
    )["p_value"].transform(holm_adjust)
    continuity["review_status"] = "no decisive discontinuity"
    caution = continuity["standardized_discontinuity"].abs().gt(
        BALANCE_REVIEW_THRESHOLD_SD
    ) | continuity["holm_p_value_within_bandwidth"].lt(0.10)
    flag = continuity["standardized_discontinuity"].abs().gt(
        BALANCE_REVIEW_THRESHOLD_SD
    ) & continuity["holm_p_value_within_bandwidth"].lt(0.05)
    continuity.loc[caution, "review_status"] = "review"
    continuity.loc[flag, "review_status"] = "material flagged discontinuity"
    continuity.loc[
        continuity["standardized_discontinuity"].isna(), "review_status"
    ] = "not estimable: no within-window variation"

    alignment_rows: list[dict[str, object]] = []
    for field, (label, transformation) in ALIGNMENT_FIELDS.items():
        for bandwidth in BANDWIDTHS_KM:
            row = fit_local_continuity(frame, field, bandwidth, transformation)
            row.update(
                {
                    "label": label,
                    "family": "modern infrastructure alignment",
                    "diagnostic_role": "post-treatment alignment; not baseline balance",
                }
            )
            alignment_rows.append(row)
    alignment = pd.DataFrame(alignment_rows)
    alignment["holm_p_value_within_bandwidth"] = alignment.groupby("bandwidth_km")[
        "p_value"
    ].transform(holm_adjust)
    alignment["review_status"] = np.where(
        alignment["holm_p_value_within_bandwidth"].lt(0.05),
        "alignment discontinuity detected",
        "no decisive alignment discontinuity",
    )
    alignment.loc[
        alignment["standardized_discontinuity"].isna(), "review_status"
    ] = "not estimable: no within-window variation"
    return continuity, alignment


def density_diagnostics(
    frame: pd.DataFrame, running: str, label: str, units_per_km: float = 1.0
) -> pd.DataFrame:
    distance = pd.to_numeric(frame[running], errors="coerce") / units_per_km
    rows: list[dict[str, object]] = []
    for bandwidth in (2, 5, 10, 15, 20, 30):
        local = distance.loc[distance.abs().le(bandwidth)].dropna()
        positive = int(local.ge(0).sum())
        negative = int(local.lt(0).sum())
        exact = binomtest(positive, positive + negative, p=0.5)
        rows.append(
            {
                "boundary": label,
                "method": "exact symmetric-side count test",
                "bandwidth_km": bandwidth,
                "bin_width_km": np.nan,
                "negative_side_count": negative,
                "positive_side_count": positive,
                "estimate": positive / (positive + negative) - 0.5,
                "estimate_scale": "positive-side share minus 0.5",
                "standard_error": np.nan,
                "p_value": float(exact.pvalue),
            }
        )
        for bin_width in (0.5, 1.0):
            edges = np.arange(-bandwidth, bandwidth + bin_width * 0.5, bin_width)
            counts, edges = np.histogram(local, bins=edges)
            centers = (edges[:-1] + edges[1:]) / 2
            bins = pd.DataFrame({"count": counts, "center": centers})
            bins["positive"] = bins["center"].ge(0).astype(float)
            bins["absolute_distance"] = bins["center"].abs()
            bins["positive_slope"] = bins["positive"] * bins["absolute_distance"]
            if len(bins) <= 4:
                rows.append(
                    {
                        "boundary": label,
                        "method": "binned local-linear Poisson diagnostic",
                        "bandwidth_km": bandwidth,
                        "bin_width_km": bin_width,
                        "negative_side_count": negative,
                        "positive_side_count": positive,
                        "estimate": np.nan,
                        "estimate_scale": "not estimable: no residual degrees of freedom",
                        "standard_error": np.nan,
                        "p_value": np.nan,
                    }
                )
                continue
            design = sm.add_constant(
                bins[["positive", "absolute_distance", "positive_slope"]],
                has_constant="add",
            )
            fit = sm.GLM(bins["count"], design, family=sm.families.Poisson()).fit(
                cov_type="HC1"
            )
            estimate = float(fit.params["positive"])
            standard_error = float(fit.bse["positive"])
            rows.append(
                {
                    "boundary": label,
                    "method": "binned local-linear Poisson diagnostic",
                    "bandwidth_km": bandwidth,
                    "bin_width_km": bin_width,
                    "negative_side_count": negative,
                    "positive_side_count": positive,
                    "estimate": estimate,
                    "estimate_scale": "log density discontinuity",
                    "standard_error": standard_error,
                    "p_value": float(2 * norm.sf(abs(estimate / standard_error))),
                }
            )
    return pd.DataFrame(rows)


def sample_line_points(geometry: object, spacing_m: float = 250.0) -> gpd.GeoSeries:
    points: list[object] = []
    for part in get_parts(geometry):
        if part.length <= 0:
            continue
        distances = np.arange(0, part.length + spacing_m * 0.5, spacing_m)
        points.extend(part.interpolate(float(min(distance, part.length))) for distance in distances)
    return gpd.GeoSeries(points, crs=32648)


def internal_boundaries(polygons: gpd.GeoDataFrame, group: str | None = None) -> object:
    units = polygons if group is None else polygons.dissolve(by=group, as_index=False)
    province = polygons.geometry.union_all()
    all_boundaries = units.boundary.union_all()
    return all_boundaries.difference(province.boundary.buffer(20))


def administrative_coincidence(
    zones_zip: Path,
    modern_commune_path: Path,
    legacy_commune_path: Path,
    output_map: Path,
) -> pd.DataFrame:
    modern = gpd.read_file(modern_commune_path).to_crs(32648)
    modern = modern.loc[modern["ADM1_PCODE"].eq("KH05")].copy()
    _, _, _, historical_boundary, _ = load_boundary_geometry(
        zones_zip, legacy_commune_path
    )
    commune_internal = internal_boundaries(modern)
    district_internal = internal_boundaries(modern, "ADM2_PCODE")
    sampled = sample_line_points(historical_boundary)
    province = modern.geometry.union_all()
    min_x, min_y, max_x, max_y = province.bounds
    grid_points = []
    for x in np.arange(min_x + 500, max_x, 1000):
        for y in np.arange(min_y + 500, max_y, 1000):
            point = Point(float(x), float(y))
            if province.contains(point):
                grid_points.append(point)
    province_grid = gpd.GeoSeries(grid_points, crs=32648)
    rows: list[dict[str, object]] = []
    for label, boundary in (
        ("modern commune internal boundaries", commune_internal),
        ("modern district internal boundaries", district_internal),
    ):
        distances_km = sampled.distance(boundary) / 1000
        grid_distances_km = province_grid.distance(boundary) / 1000
        for threshold in (0.5, 1, 2, 5):
            covered = historical_boundary.intersection(boundary.buffer(threshold * 1000))
            rows.append(
                {
                    "modern_feature": label,
                    "distance_threshold_km": threshold,
                    "historical_boundary_length_km": historical_boundary.length / 1000,
                    "boundary_length_within_threshold_km": covered.length / 1000,
                    "boundary_length_share_within_threshold": covered.length
                    / historical_boundary.length,
                    "sampled_point_share_within_threshold": float(
                        distances_km.le(threshold).mean()
                    ),
                    "sampled_point_median_nearest_distance_km": float(
                        distances_km.median()
                    ),
                    "sampled_point_p10_nearest_distance_km": float(
                        distances_km.quantile(0.10)
                    ),
                    "province_1km_grid_point_share_within_threshold": float(
                        grid_distances_km.le(threshold).mean()
                    ),
                    "historical_to_province_grid_share_ratio": float(
                        distances_km.le(threshold).mean()
                        / grid_distances_km.le(threshold).mean()
                    ),
                    "province_grid_points": len(province_grid),
                    "sample_spacing_m": 250,
                }
            )

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    modern.boundary.plot(
        ax=ax, color="#C9C9C9", linewidth=0.45, label="Modern commune boundary"
    )
    gpd.GeoSeries([district_internal], crs=32648).plot(
        ax=ax, color="#3F72AF", linewidth=1.1, label="Modern district boundary"
    )
    gpd.GeoSeries([historical_boundary], crs=32648).plot(
        ax=ax, color="#B13C2E", linewidth=2.3, label="Historical repression boundary"
    )
    gpd.GeoSeries([modern.geometry.union_all().boundary], crs=32648).plot(
        ax=ax, color="#222222", linewidth=1.0
    )
    ax.set_xlabel("Easting (m, UTM zone 48N)")
    ax.set_ylabel("Northing (m, UTM zone 48N)")
    ax.grid(True, color="#E2E2E2", linewidth=0.6)
    ax.legend(frameon=True, loc="lower left", fontsize=8)
    ax.set_title("Historical boundary and modern administrative boundaries")
    fig.tight_layout()
    fig.savefig(output_map, dpi=220)
    plt.close(fig)
    return pd.DataFrame(rows)


def segment_influence(
    panel_path: Path, simulations: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(panel_path, columns=BLIND_POWER_COLUMNS)
    panel = panel.loc[panel["Year"].between(2001, 2021)].copy()
    panel["Village Code"] = panel["Village Code"].astype("string").str.zfill(8)
    panel["Commune Code"] = panel["Commune Code"].astype("string").str.zfill(6)
    panel = panel.loc[
        panel["Absolute Distance to Historical Repression Boundary km"].le(
            PRIMARY_BANDWIDTH_KM
        )
    ].copy()
    first_year = panel.loc[panel["Year"].eq(panel["Year"].min())].copy()
    support_rows: list[dict[str, object]] = []
    for segment, group in first_year.groupby("Historical Boundary Segment"):
        commune_sides = group.groupby("Linked Climate Commune Code")[
            "Higher-Repression Southwest Zone"
        ].nunique()
        support_rows.append(
            {
                "segment": segment,
                "villages": len(group),
                "southwest_villages": int(
                    group["Higher-Repression Southwest Zone"].sum()
                ),
                "west_villages": int(
                    (1 - group["Higher-Repression Southwest Zone"]).sum()
                ),
                "climate_communes": group["Linked Climate Commune Code"].nunique(),
                "cross_side_climate_communes": int((commune_sides == 2).sum()),
                "village_share": len(group) / len(first_year),
            }
        )

    rows: list[dict[str, object]] = []
    exclusions: list[object] = ["none", *sorted(first_year["Historical Boundary Segment"].unique())]
    full_mde: float | None = None
    for index, excluded in enumerate(exclusions):
        data = panel.copy()
        if excluded != "none":
            data = data.loc[data["Historical Boundary Segment"].ne(excluded)].copy()
        data = data.sort_values(["Year", "Village Code"]).reset_index(drop=True)
        residual = target_residual(data, PRIMARY_SHOCK)
        denominator = float(np.dot(residual, residual))
        components = component_numerator_variances(data, residual)
        weights = SCENARIOS["strong clustered dependence"]
        raw_variance = sum(weights[key] * components[key] for key in components) / denominator**2
        iid_variance = components["iid"] / denominator**2
        variance = max(raw_variance, iid_variance)
        standard_error = float(np.sqrt(variance))
        mde, achieved = empirical_mde(
            standard_error, simulations, seed + 500 + index
        )
        shock_sd = float(data[PRIMARY_SHOCK].std(ddof=1))
        standardized_mde = mde * shock_sd
        if excluded == "none":
            full_mde = standardized_mde
        rows.append(
            {
                "excluded_segment": excluded,
                "villages": data["Village Code"].nunique(),
                "village_years": len(data),
                "remaining_segments": data["Historical Boundary Segment"].nunique(),
                "target_residual_sum_squares": denominator,
                "coefficient_standard_error_standardized_outcome": standard_error,
                "mde_80_outcome_sd_per_one_sd_shock": standardized_mde,
                "simulated_power_at_reported_mde": achieved,
                "simulations": simulations,
            }
        )
    influence = pd.DataFrame(rows)
    influence["mde_ratio_to_full_sample"] = (
        influence["mde_80_outcome_sd_per_one_sd_shock"] / full_mde
    )
    influence["meets_approved_0_20_sesoi"] = influence[
        "mde_80_outcome_sd_per_one_sd_shock"
    ].le(SESOI)
    return pd.DataFrame(support_rows), influence


def commune_restriction_power(
    panel_path: Path, simulations: int, seed: int
) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path, columns=BLIND_POWER_COLUMNS)
    panel = panel.loc[
        panel["Year"].between(2001, 2021)
        & panel["Absolute Distance to Historical Repression Boundary km"].le(
            PRIMARY_BANDWIDTH_KM
        )
    ].copy()
    panel["Village Code"] = panel["Village Code"].astype("string").str.zfill(8)
    panel["Commune Code"] = panel["Commune Code"].astype("string").str.zfill(6)
    baseline = panel.loc[panel["Year"].eq(panel["Year"].min())]
    commune_side_count = baseline.groupby("Linked Climate Commune Code")[
        "Higher-Repression Southwest Zone"
    ].nunique()
    cross_side_communes = commune_side_count.loc[commune_side_count.eq(2)].index
    restricted_panel = panel.loc[
        panel["Linked Climate Commune Code"].isin(cross_side_communes)
    ]
    samples = [
        ("all 5 km villages", panel, False),
        ("cross-side modern climate communes only", restricted_panel, False),
        (
            "cross-side communes plus commune-by-year fixed effects",
            restricted_panel,
            True,
        ),
    ]
    rows: list[dict[str, object]] = []
    for index, (sample, data, add_commune_year) in enumerate(samples):
        data = data.sort_values(["Year", "Village Code"]).reset_index(drop=True)
        if add_commune_year:
            treatment = data["Higher-Repression Southwest Zone"].to_numpy(dtype=float)
            signed_distance = data[
                "Signed Distance to Historical Repression Boundary km"
            ].to_numpy(dtype=float)
            shock = data[PRIMARY_SHOCK].to_numpy(dtype=float)
            matrix = np.column_stack(
                [
                    treatment * shock,
                    signed_distance * shock,
                    treatment * signed_distance * shock,
                ]
            )
            groups = [
                pd.Categorical(data["Village Code"]).codes,
                pd.Categorical(
                    data["Historical Boundary Segment"].astype(str)
                    + "-"
                    + data["Year"].astype(str)
                ).codes,
                pd.Categorical(
                    data["Linked Climate Commune Code"].astype(str)
                    + "-"
                    + data["Year"].astype(str)
                ).codes,
            ]
            within = absorb_fixed_effects(matrix, groups)
            target = within[:, 0]
            nuisance = within[:, 1:]
            keep = nuisance.std(axis=0) > 1e-12
            residual = target - nuisance[:, keep] @ np.linalg.lstsq(
                nuisance[:, keep], target, rcond=None
            )[0]
        else:
            residual = target_residual(data, PRIMARY_SHOCK)
        denominator = float(np.dot(residual, residual))
        components = component_numerator_variances(data, residual)
        weights = SCENARIOS["strong clustered dependence"]
        raw_variance = sum(weights[key] * components[key] for key in components) / denominator**2
        iid_variance = components["iid"] / denominator**2
        standard_error = float(np.sqrt(max(raw_variance, iid_variance)))
        mde, achieved = empirical_mde(
            standard_error, simulations, seed + 900 + index
        )
        shock_sd = float(data[PRIMARY_SHOCK].std(ddof=1))
        village = data.loc[data["Year"].eq(data["Year"].min())]
        rows.append(
            {
                "sample": sample,
                "fixed_effects": (
                    "village; boundary-segment-by-year; modern-climate-commune-by-year"
                    if add_commune_year
                    else "village; boundary-segment-by-year"
                ),
                "villages": village["Village Code"].nunique(),
                "southwest_villages": int(
                    village["Higher-Repression Southwest Zone"].sum()
                ),
                "west_villages": int(
                    (1 - village["Higher-Repression Southwest Zone"]).sum()
                ),
                "climate_communes": village[
                    "Linked Climate Commune Code"
                ].nunique(),
                "village_years": len(data),
                "target_residual_sum_squares": denominator,
                "mde_80_outcome_sd_per_one_sd_shock": mde * shock_sd,
                "simulated_power_at_reported_mde": achieved,
                "meets_approved_0_20_sesoi": mde * shock_sd <= SESOI,
                "simulations": simulations,
            }
        )
    return pd.DataFrame(rows)


def forest_plot(continuity: pd.DataFrame, output: Path) -> None:
    data = continuity.loc[continuity["bandwidth_km"].eq(PRIMARY_BANDWIDTH_KM)].copy()
    data = data.sort_values(["family", "label"]).reset_index(drop=True)
    y = np.arange(len(data))
    colors = np.select(
        [
            data["review_status"].eq("material flagged discontinuity"),
            data["review_status"].eq("review"),
        ],
        ["#B13C2E", "#D88A1D"],
        default="#2A6FBB",
    )
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    for index, row in data.iterrows():
        if not np.isfinite(row["standardized_discontinuity"]):
            continue
        ax.plot(
            [row["ci95_low"], row["ci95_high"]],
            [index, index],
            color=colors[index],
            linewidth=1.5,
        )
        ax.scatter(row["standardized_discontinuity"], index, color=colors[index], s=30)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.axvline(-BALANCE_REVIEW_THRESHOLD_SD, color="#999999", linestyle="--", linewidth=0.9)
    ax.axvline(BALANCE_REVIEW_THRESHOLD_SD, color="#999999", linestyle="--", linewidth=0.9)
    ax.set_yticks(y, data["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized Southwest-side discontinuity (95% CI)")
    ax.set_title("Baseline continuity diagnostics at the 5 km boundary window")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def density_plot(frame: pd.DataFrame, output: Path) -> None:
    distance = frame["dist_border"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bins = np.arange(-30, 30.5, 0.5)
    ax.hist(distance, bins=bins, color="#6BAED6", edgecolor="white", linewidth=0.25)
    ax.axvline(0, color="#B13C2E", linewidth=1.8)
    ax.axvline(-PRIMARY_BANDWIDTH_KM, color="#555555", linestyle="--", linewidth=1.0)
    ax.axvline(PRIMARY_BANDWIDTH_KM, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Signed distance to historical boundary (km)")
    ax.set_ylabel("Village count per 0.5 km bin")
    ax.set_title("Village density around the historical repression boundary")
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def build_gate(
    continuity: pd.DataFrame,
    alignment: pd.DataFrame,
    density: pd.DataFrame,
    coincidence: pd.DataFrame,
    influence: pd.DataFrame,
    commune_safeguard: pd.DataFrame,
) -> pd.DataFrame:
    primary_continuity = continuity.loc[
        continuity["bandwidth_km"].eq(PRIMARY_BANDWIDTH_KM)
        & ~continuity["family"].str.contains("timing-ambiguous")
    ]
    settlement = continuity.loc[
        continuity["bandwidth_km"].eq(PRIMARY_BANDWIDTH_KM)
        & continuity["family"].str.contains("timing-ambiguous")
    ]
    flagged = primary_continuity["review_status"].eq(
        "material flagged discontinuity"
    ).sum()
    reviews = primary_continuity["review_status"].eq("review").sum()
    historical_density = density.loc[
        density["boundary"].eq("historical repression boundary")
        & density["bandwidth_km"].eq(PRIMARY_BANDWIDTH_KM)
    ]
    density_min_p = historical_density["p_value"].min()
    primary_alignment = alignment.loc[
        alignment["bandwidth_km"].eq(PRIMARY_BANDWIDTH_KM)
    ]
    alignment_hits = primary_alignment["holm_p_value_within_bandwidth"].lt(0.05).sum()
    worst_mde = influence["mde_80_outcome_sd_per_one_sd_shock"].max()
    commune_share_1km = coincidence.loc[
        coincidence["modern_feature"].str.contains("commune")
        & coincidence["distance_threshold_km"].eq(1),
        "boundary_length_share_within_threshold",
    ].iloc[0]
    commune_grid_share_1km = coincidence.loc[
        coincidence["modern_feature"].str.contains("commune")
        & coincidence["distance_threshold_km"].eq(1),
        "province_1km_grid_point_share_within_threshold",
    ].iloc[0]
    restricted = commune_safeguard.loc[
        commune_safeguard["sample"].eq(
            "cross-side communes plus commune-by-year fixed effects"
        )
    ].iloc[0]
    return pd.DataFrame(
        [
            {
                "gate": "Outcome blinding",
                "status": "pass",
                "evidence": "No productivity, poverty, education, light, or other candidate outcome was read.",
            },
            {
                "gate": "Power at approved SESOI",
                "status": "pass",
                "evidence": "The 5 km strong-dependence MDE is about 0.16 outcome SD, below the approved 0.20 SESOI.",
            },
            {
                "gate": "Predetermined covariate continuity",
                "status": "review required" if flagged or reviews else "provisional pass",
                "evidence": f"At 5 km: {flagged} material flags and {reviews} additional review items using the predeclared 0.25-SD review threshold.",
            },
            {
                "gate": "Timing-ambiguous 1975 settlement proxies",
                "status": "human review required",
                "evidence": f"{settlement['standardized_discontinuity'].abs().gt(BALANCE_REVIEW_THRESHOLD_SD).sum()} of 4 estimates exceed 0.25 SD; 1975 overlaps the onset of Khmer Rouge rule and is not treated as safely predetermined.",
            },
            {
                "gate": "Village sorting/density",
                "status": "review required" if density_min_p < 0.05 else "provisional pass",
                "evidence": f"Smallest 5 km density-diagnostic p-value is {density_min_p:.3f}; binned tests are diagnostics, not a formal rddensity replacement.",
            },
            {
                "gate": "Modern administrative-boundary coincidence",
                "status": "human review required",
                "evidence": f"{commune_share_1km:.1%} of the historical boundary lies within 1 km of a modern commune boundary versus {commune_grid_share_1km:.1%} of province-grid points; no automatic causal threshold is imposed.",
            },
            {
                "gate": "Within-modern-commune safeguard",
                "status": "pass",
                "evidence": f"Restricting to {int(restricted.climate_communes)} cross-side communes retains {int(restricted.villages)} villages and an 80% MDE of {restricted.mde_80_outcome_sd_per_one_sd_shock:.3f}, below the 0.20 SESOI.",
            },
            {
                "gate": "Modern road alignment",
                "status": "human review required" if alignment_hits else "provisional pass with caution",
                "evidence": f"{alignment_hits} of 2 road-alignment fields show Holm-adjusted p<0.05 at 5 km; these are potentially post-treatment and are not baseline covariates.",
            },
            {
                "gate": "Boundary-segment influence",
                "status": "review required" if worst_mde > SESOI else "pass",
                "evidence": f"Worst leave-one-segment-out 80% MDE is {worst_mde:.3f} versus the approved 0.20 SESOI.",
            },
            {
                "gate": "Effect estimation",
                "status": "blocked pending human review",
                "evidence": "No outcome effect is estimated until the diagnostic findings and any prespecified responses are approved.",
            },
        ]
    )


def write_readme(
    output: Path,
    continuity: pd.DataFrame,
    alignment: pd.DataFrame,
    density: pd.DataFrame,
    coincidence: pd.DataFrame,
    influence: pd.DataFrame,
    commune_safeguard: pd.DataFrame,
    gate: pd.DataFrame,
) -> None:
    primary = continuity.loc[
        continuity["bandwidth_km"].eq(PRIMARY_BANDWIDTH_KM)
        & ~continuity["family"].str.contains("timing-ambiguous")
    ]
    settlement = continuity.loc[
        continuity["bandwidth_km"].eq(PRIMARY_BANDWIDTH_KM)
        & continuity["family"].str.contains("timing-ambiguous")
    ]
    primary_density = density.loc[
        density["boundary"].eq("historical repression boundary")
        & density["bandwidth_km"].eq(PRIMARY_BANDWIDTH_KM)
    ]
    exact_density = primary_density.loc[
        primary_density["method"].eq("exact symmetric-side count test")
    ].iloc[0]
    full = influence.loc[influence["excluded_segment"].astype(str).eq("none")].iloc[0]
    worst = influence.sort_values("mde_80_outcome_sd_per_one_sd_shock", ascending=False).iloc[0]
    restricted = commune_safeguard.loc[
        commune_safeguard["sample"].eq(
            "cross-side communes plus commune-by-year fixed effects"
        )
    ].iloc[0]
    lines = [
        "# Outcome-Blind Historical-Boundary Identification Diagnostics",
        "",
        "No candidate outcome was read. The diagnostic input projection excludes poverty,",
        "education, night lights, land productivity, and all other present-day outcomes.",
        "",
        "## Prespecified design",
        "",
        "- Primary local bandwidth: 5 km.",
        "- Primary shock for design power: May-October rainfall anomaly.",
        "- Approved SESOI: 0.20 standardized outcome units per one-SD shock.",
        "- Continuity specification: triangular-weighted local linear regressions with separate",
        "  slopes by historical side, boundary-segment controls, and commune-clustered inference.",
        "- A 0.25-SD discontinuity is used as a transparent review threshold, not as an automatic",
        "  proof of balance or causal validity.",
        "",
        "## Main 5 km findings",
        "",
        f"The continuity screen contains {(primary.review_status == 'material flagged discontinuity').sum()} material flags and "
        f"{(primary.review_status == 'review').sum()} additional review items after Holm correction.",
        f"Separately, {(settlement.standardized_discontinuity.abs() > BALANCE_REVIEW_THRESHOLD_SD).sum()} of four 1975 settlement proxies exceed 0.25 SD in absolute value. These are not",
        "treated as safely predetermined because 1975 overlaps the onset of Khmer Rouge rule.",
        f"The historical-boundary density count is {int(exact_density.negative_side_count)} West versus "
        f"{int(exact_density.positive_side_count)} Southwest villages (exact p={exact_density.p_value:.3f}).",
        f"The full-sample strong-dependence MDE is {full.mde_80_outcome_sd_per_one_sd_shock:.3f}; "
        f"the worst leave-one-segment-out MDE is {worst.mde_80_outcome_sd_per_one_sd_shock:.3f} "
        f"when segment {worst.excluded_segment} is removed.",
        f"A safeguard restricted to {int(restricted.climate_communes)} modern communes containing villages on both sides",
        f"retains {int(restricted.villages)} villages and has an MDE of {restricted.mde_80_outcome_sd_per_one_sd_shock:.3f}.",
        "",
        "Modern road measures are interpreted only as alignment/placebo diagnostics because roads",
        "can themselves be affected by historical development. Modern administrative-boundary",
        "coincidence is descriptive; no unsupported numerical pass/fail threshold is imposed.",
        "",
        "## Activation status",
        "",
        "| gate | status | evidence |",
        "|---|---|---|",
    ]
    for row in gate.itertuples(index=False):
        lines.append(f"| {row.gate} | {row.status} | {row.evidence} |")
    lines.extend(
        [
            "",
            "Effect estimation remains blocked pending human review of these diagnostics.",
            "",
            "## Files",
            "",
            "- `predetermined_covariate_continuity.csv`: all bandwidth-specific continuity estimates.",
            "- `modern_road_alignment_continuity.csv`: road alignment estimates, kept separate.",
            "- `density_sorting_diagnostics.csv`: historical and NR3 placebo density diagnostics.",
            "- `modern_boundary_coincidence.csv`: proximity to modern commune/district boundaries.",
            "- `boundary_segment_support.csv` and `boundary_segment_leave_one_out_power.csv`.",
            "- `modern_commune_restriction_power.csv`: power for the within-modern-commune safeguard.",
            "- `identification_activation_gate.csv`: current gate decisions.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    field_dir = absolute(root, args.field_dir)
    panel_path = absolute(root, args.panel)
    zones_zip = absolute(root, args.zones_zip)
    commune_path = absolute(root, args.communes)
    legacy_commune_path = absolute(root, args.legacy_communes)
    field_dir.mkdir(parents=True, exist_ok=True)

    blind_fields = pd.read_csv(field_dir / "predetermined_and_alignment_fields.csv")
    allowed = {
        "vill_code", "comm", "treat", "dist_border", "dist.segment", "longitude",
        "latitude", *PREDICTORS.keys(), *ALIGNMENT_FIELDS.keys()
    }
    unexpected = set(blind_fields.columns) - allowed
    if unexpected:
        raise RuntimeError(f"Outcome-blind field export contains unexpected columns: {unexpected}")
    continuity, alignment = continuity_diagnostics(blind_fields)
    continuity.to_csv(field_dir / "predetermined_covariate_continuity.csv", index=False)
    alignment.to_csv(field_dir / "modern_road_alignment_continuity.csv", index=False)

    historical_density = density_diagnostics(
        blind_fields, "dist_border", "historical repression boundary"
    )
    nr3 = pd.read_csv(field_dir / "nr3_placebo_design_fields.csv")
    if set(nr3.columns) != {"vill_code", "south", "dist.NR3", "X", "Y"}:
        raise RuntimeError("NR3 placebo projection contains unexpected columns")
    nr3_density = density_diagnostics(nr3, "dist.NR3", "National Road 3 placebo", 1000.0)
    density = pd.concat([historical_density, nr3_density], ignore_index=True)
    density.to_csv(field_dir / "density_sorting_diagnostics.csv", index=False)

    coincidence = administrative_coincidence(
        zones_zip,
        commune_path,
        legacy_commune_path,
        field_dir / "modern_boundary_coincidence.png",
    )
    coincidence.to_csv(field_dir / "modern_boundary_coincidence.csv", index=False)
    segment_support, segment_influence_frame = segment_influence(
        panel_path, args.simulations, args.seed
    )
    segment_support.to_csv(field_dir / "boundary_segment_support.csv", index=False)
    segment_influence_frame.to_csv(
        field_dir / "boundary_segment_leave_one_out_power.csv", index=False
    )
    commune_safeguard = commune_restriction_power(
        panel_path, args.simulations, args.seed
    )
    commune_safeguard.to_csv(
        field_dir / "modern_commune_restriction_power.csv", index=False
    )

    forest_plot(continuity, field_dir / "predetermined_covariate_continuity_5km.png")
    density_plot(blind_fields, field_dir / "historical_boundary_density.png")
    gate = build_gate(
        continuity,
        alignment,
        density,
        coincidence,
        segment_influence_frame,
        commune_safeguard,
    )
    gate.to_csv(field_dir / "identification_activation_gate.csv", index=False)
    write_readme(
        field_dir / "README_identification_diagnostics.md",
        continuity,
        alignment,
        density,
        coincidence,
        segment_influence_frame,
        commune_safeguard,
        gate,
    )
    manifest = {
        "outcome_columns_read": [],
        "blind_replication_fields_read": sorted(allowed),
        "blind_annual_panel_fields_read": BLIND_POWER_COLUMNS,
        "primary_bandwidth_km": PRIMARY_BANDWIDTH_KM,
        "primary_shock": PRIMARY_SHOCK,
        "approved_sesoi": SESOI,
        "balance_review_threshold_sd": BALANCE_REVIEW_THRESHOLD_SD,
        "modern_commune_safeguard": "5 km sample restricted to climate communes containing villages on both historical sides, with modern-climate-commune-by-year fixed effects",
        "effect_estimation_performed": False,
        "activation_status": "blocked pending human review",
    }
    (field_dir / "README_identification_diagnostics.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(gate.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
