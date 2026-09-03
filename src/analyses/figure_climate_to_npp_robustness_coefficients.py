#!/usr/bin/env python3
"""Climate-to-NPP robustness coefficient figure.

Plan: compare the natural-unit stage-1 coefficients across buffer scale,
cropland definition, pixel-support gates, rainfall conditioning, MODIS NPP
quality support, functional form, and alternative covariance choices.
Leave-one-year coefficient envelopes are shown behind the primary estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

from linearmodels.iv import AbsorbingLS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[2]
CROSSWALK = ROOT / "data/processed/cses_to_cambodia_public_village_point_crosswalk_preprocessed.parquet"
CLIMATE = ROOT / "data/processed/cses_village_buffer_annual_absolute_climate_shocks_preprocessed.parquet"
NPP_FILES = {
    2: ROOT / "data/processed/cses_public_village_pixel_cropland_npp_annual_2km_preprocessed.parquet",
    5: ROOT / "data/processed/cses_public_village_pixel_cropland_npp_annual_preprocessed.parquet",
    10: ROOT / "data/processed/cses_public_village_pixel_cropland_npp_annual_10km_preprocessed.parquet",
}
OUTPUT = ROOT / "data/results/figures/Figure_climate_to_npp_robustness_coefficients.png"
EVIDENCE = ROOT / "data/exp/analysis/climate-npp/climate-to-npp-robustness-coefficients"

ID = "National Village Point ID"
YEAR = "Year"
LONGITUDE = "Point Longitude"
LATITUDE = "Point Latitude"
STRICT_OUTCOME = "Annual Strict-Cropland Mean NPP kg C per m2"
INCLUSIVE_OUTCOME = "Annual Inclusive-Agriculture Mean NPP kg C per m2"
STRICT_SHARE = "Strict-Cropland Valid NPP Pixel Share"
NPP_QC = "Mean Strict-Cropland Recoded NPP QC Filled Days Percent"

HEAT = "Village Buffer Mean Annual Heat Days at or Above 35 C"
HDD = "Village Buffer Mean Annual Heat Degree-Days Above 35 C"
RX5DAY = "Village Buffer Mean Annual Maximum Consecutive Five-Day Precipitation Rx5day mm"
DRY_DAYS = "Village Buffer Mean Annual Maximum Consecutive Dry Days Below 1 mm"
RAIN_TOTAL = "Village Buffer Mean Annual Precipitation Total mm"

SCALED = {
    HEAT: "Heat days per 10 days",
    HDD: "Heat degree-days per 10 degree-days",
    RX5DAY: "Rx5day per 10 mm",
    DRY_DAYS: "Dry spell per 10 days",
    RAIN_TOTAL: "Annual rainfall per 100 mm",
}
QC_10 = "NPP QC filled-days percentage / 10"
RESIDUAL_RAIN_100 = "Residual annual rainfall / 100 mm"
SCALE_DIVISOR = {HEAT: 10.0, HDD: 10.0, RX5DAY: 10.0, DRY_DAYS: 10.0, RAIN_TOTAL: 100.0}
DISPLAY = {
    HEAT: ("Heat exposure ≥35°C", "NPP change per 10 heat units"),
    HDD: ("Heat degree-days >35°C", "NPP change per 10 degree-days"),
    RX5DAY: ("Extreme five-day rainfall", "NPP change per 10 mm Rx5day"),
    DRY_DAYS: ("Maximum dry spell", "NPP change per 10 dry days"),
    RAIN_TOTAL: ("Annual rainfall", "NPP change per 100 mm rainfall"),
}
EXPOSURES = [HEAT, RX5DAY, DRY_DAYS, RAIN_TOTAL]
ALTERNATIVE_EXPOSURES = [HDD, RX5DAY, DRY_DAYS, RAIN_TOTAL]
ALL_EXPOSURES = [HEAT, HDD, RX5DAY, DRY_DAYS, RAIN_TOTAL]

SPECIFICATIONS = [
    ("Primary: 5 km strict cropland", "#173F5F"),
    ("Annual rainfall omitted", "#4B6F88"),
    ("Residual annual rainfall", "#557A95"),
    ("Continuous NPP QC adjustment", "#3A7D77"),
    ("NPP QC lower 75%", "#5A9F8C"),
    ("NPP QC lower 50%", "#74B49B"),
    ("2 km strict cropland", "#2F80A2"),
    ("10 km strict cropland", "#2F80A2"),
    ("5 km inclusive agriculture", "#3A9D8F"),
    ("Valid-pixel share ≥80%", "#D4A72C"),
    ("Valid-pixel share ≥95%", "#D4A72C"),
    ("Quadratic slope at median", "#8A62A6"),
    ("Alternative heat: degree-days", "#6F5AA8"),
    ("Village + year clustered", "#D97757"),
    ("0.5° spatial-block clustered", "#D97757"),
    ("Conley spatial HAC: 50 km", "#9B4A63"),
]

CONLEY_CUTOFFS_KM = [25.0, 50.0, 100.0]

DARK = "#4D5960"
GRID = "#D9DEE1"


def build_panel(radius: int, climate: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    radius_climate = climate.loc[climate["Buffer Radius km"].eq(radius)].merge(
        crosswalk[["Village Code", ID]],
        on="Village Code",
        how="inner",
        validate="many_to_one",
    )
    point_climate = (
        radius_climate.groupby([ID, YEAR], as_index=False)
        .agg(**{column: (column, "mean") for column in ALL_EXPOSURES})
    )
    npp = pd.read_parquet(
        NPP_FILES[radius],
        columns=[
            ID,
            YEAR,
            LONGITUDE,
            LATITUDE,
            STRICT_OUTCOME,
            INCLUSIVE_OUTCOME,
            STRICT_SHARE,
            NPP_QC,
        ],
    )
    panel = npp.merge(point_climate, on=[ID, YEAR], how="left", validate="one_to_one")
    panel["Buffer Radius km"] = radius
    return panel


def prepare(panel: pd.DataFrame, outcome: str, support_gate: float | None = None) -> pd.DataFrame:
    required = [ID, YEAR, LONGITUDE, LATITUDE, outcome, *ALL_EXPOSURES]
    selected = [*required, NPP_QC]
    if support_gate is not None:
        required.append(STRICT_SHARE)
        selected.append(STRICT_SHARE)
    sample = panel.loc[panel[YEAR].between(2001, 2021), list(dict.fromkeys(selected))].copy()
    for column in [outcome, NPP_QC, *ALL_EXPOSURES]:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    sample = sample.dropna(subset=required)
    if support_gate is not None:
        sample = sample.loc[sample[STRICT_SHARE].ge(support_gate)].copy()
    for exposure in ALL_EXPOSURES:
        sample[SCALED[exposure]] = sample[exposure] / SCALE_DIVISOR[exposure]
    sample[QC_10] = pd.to_numeric(sample[NPP_QC], errors="coerce") / 10.0
    return sample.reset_index(drop=True)


def clusters_for(sample: pd.DataFrame, covariance: str) -> pd.DataFrame:
    if covariance == "village":
        return pd.DataFrame({"Village": pd.Categorical(sample[ID]).codes}, index=sample.index)
    if covariance == "village_year":
        return pd.DataFrame(
            {
                "Village": pd.Categorical(sample[ID]).codes,
                "Year": pd.Categorical(sample[YEAR]).codes,
            },
            index=sample.index,
        )
    if covariance == "spatial_block":
        lon_block = np.floor((sample[LONGITUDE].to_numpy(float) - 101.5) / 0.5).astype(int)
        lat_block = np.floor((sample[LATITUDE].to_numpy(float) - 9.5) / 0.5).astype(int)
        blocks = pd.Series(lon_block.astype(str) + "_" + lat_block.astype(str), index=sample.index)
        return pd.DataFrame({"Spatial block": pd.Categorical(blocks).codes}, index=sample.index)
    raise ValueError(f"Unknown covariance: {covariance}")


def fit_linear(
    sample: pd.DataFrame,
    outcome: str,
    covariance: str = "village",
    exposures: list[str] | None = None,
) -> object:
    exposures = EXPOSURES if exposures is None else exposures
    absorb = pd.DataFrame(
        {
            "Village fixed effect": sample[ID].astype("category"),
            "Year fixed effect": sample[YEAR].astype("category"),
        },
        index=sample.index,
    )
    return AbsorbingLS(
        sample[outcome].astype(float),
        sample[[SCALED[exposure] for exposure in exposures]].astype(float),
        absorb=absorb,
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=clusters_for(sample, covariance),
        debiased=True,
    )


def fit_custom(
    sample: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    covariance: str = "village",
) -> object:
    absorb = pd.DataFrame(
        {
            "Village fixed effect": sample[ID].astype("category"),
            "Year fixed effect": sample[YEAR].astype("category"),
        },
        index=sample.index,
    )
    return AbsorbingLS(
        sample[outcome].astype(float),
        sample[regressors].astype(float),
        absorb=absorb,
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=clusters_for(sample, covariance),
        debiased=True,
    )


def rows_from_term_map(
    result: object,
    sample: pd.DataFrame,
    specification: str,
    covariance: str,
    term_map: dict[str, str],
) -> list[dict[str, object]]:
    intervals = result.conf_int(level=0.95)
    rows: list[dict[str, object]] = []
    for exposure, term in term_map.items():
        rows.append(
            {
                "Specification": specification,
                "Exposure": exposure,
                "Estimate": float(result.params[term]),
                "Standard Error": float(result.std_errors[term]),
                "95 Percent CI Lower": float(intervals.loc[term, "lower"]),
                "95 Percent CI Upper": float(intervals.loc[term, "upper"]),
                "Probability Value": float(result.pvalues[term]),
                "Observations": int(len(sample)),
                "Villages": int(sample[ID].nunique()),
                "Years": int(sample[YEAR].nunique()),
                "Buffer Radius km": 5,
                "Outcome": STRICT_OUTCOME,
                "Covariance": covariance,
            }
        )
    return rows


def residualise_annual_rainfall(sample: pd.DataFrame) -> pd.DataFrame:
    work = sample.copy()
    absorb = pd.DataFrame(
        {
            "Village fixed effect": work[ID].astype("category"),
            "Year fixed effect": work[YEAR].astype("category"),
        },
        index=work.index,
    )
    auxiliary = AbsorbingLS(
        work[RAIN_TOTAL].astype(float),
        work[[RX5DAY]].astype(float),
        absorb=absorb,
        drop_absorbed=True,
    ).fit(cov_type="unadjusted")
    work[RESIDUAL_RAIN_100] = auxiliary.resids.to_numpy(float) / 100.0
    return work


def fit_quadratic(sample: pd.DataFrame, outcome: str) -> tuple[object, dict[str, str]]:
    square_names: dict[str, str] = {}
    regressors: list[str] = []
    work = sample.copy()
    for exposure in EXPOSURES:
        linear = SCALED[exposure]
        square = f"Squared {linear}"
        work[square] = work[linear] ** 2
        square_names[exposure] = square
        regressors.extend([linear, square])
    absorb = pd.DataFrame(
        {
            "Village fixed effect": work[ID].astype("category"),
            "Year fixed effect": work[YEAR].astype("category"),
        },
        index=work.index,
    )
    result = AbsorbingLS(
        work[outcome].astype(float),
        work[regressors].astype(float),
        absorb=absorb,
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=clusters_for(work, "village"),
        debiased=True,
    )
    return result, square_names


def rows_from_result(
    result: object,
    sample: pd.DataFrame,
    specification: str,
    covariance: str,
    radius: int,
    outcome: str,
    exposures: list[str] | None = None,
) -> list[dict[str, object]]:
    exposures = EXPOSURES if exposures is None else exposures
    intervals = result.conf_int(level=0.95)
    return [
        {
            "Specification": specification,
            "Exposure": exposure,
            "Estimate": float(result.params[SCALED[exposure]]),
            "Standard Error": float(result.std_errors[SCALED[exposure]]),
            "95 Percent CI Lower": float(intervals.loc[SCALED[exposure], "lower"]),
            "95 Percent CI Upper": float(intervals.loc[SCALED[exposure], "upper"]),
            "Probability Value": float(result.pvalues[SCALED[exposure]]),
            "Observations": int(len(sample)),
            "Villages": int(sample[ID].nunique()),
            "Years": int(sample[YEAR].nunique()),
            "Buffer Radius km": radius,
            "Outcome": outcome,
            "Covariance": covariance,
        }
        for exposure in exposures
    ]


def haversine_distance_km(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:
    lon1_r = np.radians(lon1)
    lat1_r = np.radians(lat1)
    lon2_r = np.radians(lon2)
    lat2_r = np.radians(lat2)
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def conley_covariances(
    result: object,
    sample: pd.DataFrame,
    cutoffs_km: list[float],
) -> dict[float, np.ndarray]:
    """Same-year Conley spatial HAC covariance with a Bartlett distance kernel."""
    absorbed_x = result.model.absorbed_exog.to_numpy(dtype=float)
    residuals = result.resids.to_numpy(dtype=float)
    scores = absorbed_x * residuals[:, None]
    bread = np.linalg.inv(absorbed_x.T @ absorbed_x)
    meats = {cutoff: np.zeros((absorbed_x.shape[1], absorbed_x.shape[1])) for cutoff in cutoffs_km}
    maximum_cutoff = max(cutoffs_km)
    mean_latitude = float(sample[LATITUDE].mean())
    longitude_scale = 111.320 * np.cos(np.radians(mean_latitude))
    latitude_scale = 110.574

    for _, positions in sample.groupby(YEAR, sort=True).indices.items():
        positions = np.asarray(positions, dtype=int)
        year_scores = scores[positions]
        for cutoff in cutoffs_km:
            meats[cutoff] += year_scores.T @ year_scores

        lon = sample.iloc[positions][LONGITUDE].to_numpy(dtype=float)
        lat = sample.iloc[positions][LATITUDE].to_numpy(dtype=float)
        projected = np.column_stack([lon * longitude_scale, lat * latitude_scale])
        pairs = cKDTree(projected).query_pairs(maximum_cutoff * 1.02, output_type="ndarray")
        if len(pairs) == 0:
            continue
        distances = haversine_distance_km(
            lon[pairs[:, 0]],
            lat[pairs[:, 0]],
            lon[pairs[:, 1]],
            lat[pairs[:, 1]],
        )
        left_scores = year_scores[pairs[:, 0]]
        right_scores = year_scores[pairs[:, 1]]
        for cutoff in cutoffs_km:
            keep = distances < cutoff
            if not keep.any():
                continue
            weights = 1.0 - distances[keep] / cutoff
            left = left_scores[keep] * weights[:, None]
            right = right_scores[keep]
            cross = left.T @ right
            meats[cutoff] += cross + cross.T

    correction = len(sample) / (len(sample) - absorbed_x.shape[1])
    return {
        cutoff: correction * (bread @ meat @ bread)
        for cutoff, meat in meats.items()
    }


def conley_rows(
    result: object,
    sample: pd.DataFrame,
    cutoffs_km: list[float],
) -> list[dict[str, object]]:
    covariances = conley_covariances(result, sample, cutoffs_km)
    rows: list[dict[str, object]] = []
    for cutoff in cutoffs_km:
        covariance = covariances[cutoff]
        for position, exposure in enumerate(EXPOSURES):
            term = SCALED[exposure]
            estimate = float(result.params[term])
            standard_error = float(np.sqrt(max(covariance[position, position], 0.0)))
            rows.append(
                {
                    "Specification": f"Conley spatial HAC: {cutoff:g} km",
                    "Exposure": exposure,
                    "Estimate": estimate,
                    "Standard Error": standard_error,
                    "95 Percent CI Lower": estimate - 1.96 * standard_error,
                    "95 Percent CI Upper": estimate + 1.96 * standard_error,
                    "Probability Value": float(2.0 * norm.sf(abs(estimate / standard_error))),
                    "Observations": int(len(sample)),
                    "Villages": int(sample[ID].nunique()),
                    "Years": int(sample[YEAR].nunique()),
                    "Buffer Radius km": 5,
                    "Outcome": STRICT_OUTCOME,
                    "Covariance": f"conley_{cutoff:g}km",
                }
            )
    return rows


def quadratic_rows(
    result: object,
    square_names: dict[str, str],
    sample: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    covariance = result.cov
    for exposure in EXPOSURES:
        linear = SCALED[exposure]
        square = square_names[exposure]
        median = float(sample[linear].median())
        estimate = float(result.params[linear] + 2.0 * median * result.params[square])
        gradient = np.array([1.0, 2.0 * median])
        local_cov = covariance.loc[[linear, square], [linear, square]].to_numpy(float)
        standard_error = float(np.sqrt(gradient @ local_cov @ gradient))
        rows.append(
            {
                "Specification": "Quadratic slope at median",
                "Exposure": exposure,
                "Estimate": estimate,
                "Standard Error": standard_error,
                "95 Percent CI Lower": estimate - 1.96 * standard_error,
                "95 Percent CI Upper": estimate + 1.96 * standard_error,
                "Probability Value": np.nan,
                "Observations": int(len(sample)),
                "Villages": int(sample[ID].nunique()),
                "Years": int(sample[YEAR].nunique()),
                "Buffer Radius km": 5,
                "Outcome": STRICT_OUTCOME,
                "Covariance": "village clustered; delta method",
                "Evaluation Point": median,
            }
        )
    return rows


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
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
        zorder=20,
    )


def draw_panel(
    ax: plt.Axes,
    coefficients: pd.DataFrame,
    leave_one_year: pd.DataFrame,
    exposure: str,
    panel: str,
    show_y_labels: bool,
) -> None:
    labels = [label for label, _ in SPECIFICATIONS]
    colors = dict(SPECIFICATIONS)
    y_positions = {label: len(labels) - 1 - index for index, label in enumerate(labels)}
    loo = leave_one_year.loc[leave_one_year["Exposure"].eq(exposure)]
    primary_y = y_positions["Primary: 5 km strict cropland"]
    ax.hlines(
        primary_y,
        loo["Estimate"].min(),
        loo["Estimate"].max(),
        color="#C8CDD0",
        linewidth=7.0,
        zorder=1,
        label="Leave-one-year coefficient range",
    )
    for specification, color in SPECIFICATIONS:
        displayed_exposure = HDD if specification == "Alternative heat: degree-days" and exposure == HEAT else exposure
        subset = coefficients.loc[
            coefficients["Specification"].eq(specification)
            & coefficients["Exposure"].eq(displayed_exposure)
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        y = y_positions[specification]
        ax.hlines(
            y,
            row["95 Percent CI Lower"],
            row["95 Percent CI Upper"],
            color=color,
            linewidth=1.8,
            zorder=2,
        )
        ax.scatter(
            row["Estimate"],
            y,
            s=37,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
    ax.axvline(0, color="#667076", linewidth=0.9, zorder=0)
    ticks = [y_positions[label] for label in labels]
    if show_y_labels:
        ax.set_yticks(ticks, labels, fontsize=7.9)
    else:
        ax.set_yticks(ticks)
        ax.tick_params(axis="y", labelleft=False)
    ax.set_ylim(-0.8, len(labels) - 0.2)
    ax.set_xlabel(DISPLAY[exposure][1] + " (kg C m⁻²)", fontsize=8.5, color=DARK)
    ax.text(
        0.985,
        0.965,
        DISPLAY[exposure][0],
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color="#173F5F",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.4},
        zorder=10,
    )
    ax.grid(axis="both", color=GRID, linewidth=0.55, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=8.0, colors=DARK)
    for spine in ax.spines.values():
        spine.set_color("#89969C")
        spine.set_linewidth(0.65)
    panel_label(ax, panel)


def main() -> None:
    crosswalk = pd.read_parquet(CROSSWALK)
    crosswalk = crosswalk.loc[crosswalk["National Public Village Point Matched"].eq(1)].copy()
    climate = pd.read_parquet(CLIMATE)
    panels = {radius: build_panel(radius, climate, crosswalk) for radius in [2, 5, 10]}

    samples = {
        "Primary: 5 km strict cropland": prepare(panels[5], STRICT_OUTCOME),
        "2 km strict cropland": prepare(panels[2], STRICT_OUTCOME),
        "10 km strict cropland": prepare(panels[10], STRICT_OUTCOME),
        "5 km inclusive agriculture": prepare(panels[5], INCLUSIVE_OUTCOME),
        "Valid-pixel share ≥80%": prepare(panels[5], STRICT_OUTCOME, 0.80),
        "Valid-pixel share ≥95%": prepare(panels[5], STRICT_OUTCOME, 0.95),
    }

    rows: list[dict[str, object]] = []
    for specification, sample in samples.items():
        outcome = INCLUSIVE_OUTCOME if specification == "5 km inclusive agriculture" else STRICT_OUTCOME
        radius = 2 if specification.startswith("2 km") else 10 if specification.startswith("10 km") else 5
        result = fit_linear(sample, outcome, "village")
        rows.extend(rows_from_result(result, sample, specification, "village clustered", radius, outcome))

    primary_sample = samples["Primary: 5 km strict cropland"]
    primary_term_map = {exposure: SCALED[exposure] for exposure in EXPOSURES}

    no_rain_exposures = [HEAT, RX5DAY, DRY_DAYS]
    no_rain = fit_linear(
        primary_sample,
        STRICT_OUTCOME,
        "village",
        exposures=no_rain_exposures,
    )
    rows.extend(
        rows_from_result(
            no_rain,
            primary_sample,
            "Annual rainfall omitted",
            "village clustered; annual rainfall omitted",
            5,
            STRICT_OUTCOME,
            exposures=no_rain_exposures,
        )
    )

    residual_rain_sample = residualise_annual_rainfall(primary_sample)
    residual_terms = [
        SCALED[HEAT],
        SCALED[RX5DAY],
        SCALED[DRY_DAYS],
        RESIDUAL_RAIN_100,
    ]
    residual_rain = fit_custom(residual_rain_sample, STRICT_OUTCOME, residual_terms)
    residual_map = {
        HEAT: SCALED[HEAT],
        RX5DAY: SCALED[RX5DAY],
        DRY_DAYS: SCALED[DRY_DAYS],
        RAIN_TOTAL: RESIDUAL_RAIN_100,
    }
    rows.extend(
        rows_from_term_map(
            residual_rain,
            residual_rain_sample,
            "Residual annual rainfall",
            "village clustered; residual rainfall",
            residual_map,
        )
    )

    qc_adjusted = fit_custom(
        primary_sample,
        STRICT_OUTCOME,
        [*[SCALED[exposure] for exposure in EXPOSURES], QC_10],
    )
    rows.extend(
        rows_from_term_map(
            qc_adjusted,
            primary_sample,
            "Continuous NPP QC adjustment",
            "village clustered; NPP QC covariate",
            primary_term_map,
        )
    )

    qc_thresholds = {
        "NPP QC lower 75%": float(primary_sample[NPP_QC].quantile(0.75)),
        "NPP QC lower 50%": float(primary_sample[NPP_QC].quantile(0.50)),
    }
    qc_samples: dict[str, pd.DataFrame] = {}
    for specification, threshold in qc_thresholds.items():
        qc_sample = primary_sample.loc[primary_sample[NPP_QC].le(threshold)].reset_index(drop=True)
        qc_samples[specification] = qc_sample
        result = fit_linear(qc_sample, STRICT_OUTCOME, "village")
        rows.extend(
            rows_from_result(
                result,
                qc_sample,
                specification,
                "village clustered; NPP QC restriction",
                5,
                STRICT_OUTCOME,
            )
        )

    quadratic, square_names = fit_quadratic(primary_sample, STRICT_OUTCOME)
    rows.extend(quadratic_rows(quadratic, square_names, primary_sample))
    for specification, covariance in [
        ("Village + year clustered", "village_year"),
        ("0.5° spatial-block clustered", "spatial_block"),
    ]:
        result = fit_linear(primary_sample, STRICT_OUTCOME, covariance)
        rows.extend(
            rows_from_result(
                result,
                primary_sample,
                specification,
                covariance,
                5,
                STRICT_OUTCOME,
            )
        )

    alternative = fit_linear(
        primary_sample,
        STRICT_OUTCOME,
        "village",
        exposures=ALTERNATIVE_EXPOSURES,
    )
    rows.extend(
        rows_from_result(
            alternative,
            primary_sample,
            "Alternative heat: degree-days",
            "village clustered",
            5,
            STRICT_OUTCOME,
            exposures=ALTERNATIVE_EXPOSURES,
        )
    )
    primary_result = fit_linear(primary_sample, STRICT_OUTCOME, "village")
    rows.extend(conley_rows(primary_result, primary_sample, CONLEY_CUTOFFS_KM))

    coefficients = pd.DataFrame(rows)
    loo_rows: list[dict[str, object]] = []
    for omitted_year in sorted(primary_sample[YEAR].unique()):
        loo_sample = primary_sample.loc[primary_sample[YEAR].ne(omitted_year)].reset_index(drop=True)
        result = fit_linear(loo_sample, STRICT_OUTCOME, "village")
        for exposure in EXPOSURES:
            loo_rows.append(
                {
                    "Omitted Year": int(omitted_year),
                    "Exposure": exposure,
                    "Estimate": float(result.params[SCALED[exposure]]),
                    "Standard Error": float(result.std_errors[SCALED[exposure]]),
                    "Observations": int(len(loo_sample)),
                    "Villages": int(loo_sample[ID].nunique()),
                }
            )
    leave_one_year = pd.DataFrame(loo_rows)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(EVIDENCE / "climate_to_npp_robustness_coefficients.csv", index=False)
    leave_one_year.to_csv(EVIDENCE / "leave_one_year_coefficient_diagnostics.csv", index=False)
    sample_support = pd.DataFrame(
        [
            {
                "Specification": name,
                "Observations": len(sample),
                "Villages": sample[ID].nunique(),
                "Years": sample[YEAR].nunique(),
            }
            for name, sample in samples.items()
        ]
        + [
            {
                "Specification": "Annual rainfall omitted",
                "Observations": len(primary_sample),
                "Villages": primary_sample[ID].nunique(),
                "Years": primary_sample[YEAR].nunique(),
            },
            {
                "Specification": "Residual annual rainfall",
                "Observations": len(residual_rain_sample),
                "Villages": residual_rain_sample[ID].nunique(),
                "Years": residual_rain_sample[YEAR].nunique(),
            },
            {
                "Specification": "Continuous NPP QC adjustment",
                "Observations": len(primary_sample),
                "Villages": primary_sample[ID].nunique(),
                "Years": primary_sample[YEAR].nunique(),
            },
        ]
        + [
            {
                "Specification": name,
                "Observations": len(sample),
                "Villages": sample[ID].nunique(),
                "Years": sample[YEAR].nunique(),
            }
            for name, sample in qc_samples.items()
        ]
    )
    sample_support.to_csv(EVIDENCE / "robustness_sample_support.csv", index=False)
    summary = {
        "specifications": coefficients["Specification"].drop_duplicates().tolist(),
        "leave_one_year_omissions": int(leave_one_year["Omitted Year"].nunique()),
        "spatial_block_definition": "0.5 degree longitude-latitude cells",
        "conley_definition": "same-year Bartlett spatial HAC with 25, 50, and 100 km cutoffs; 50 km focal",
        "npp_qc_definition": "percentage of growing-season days using gap-filled FPAR/LAI inputs; lower is better",
        "npp_qc_thresholds": qc_thresholds,
        "rainfall_sensitivities": [
            "annual precipitation omitted",
            "annual precipitation residualised against Rx5day with village and year effects",
        ],
        "coefficient_units": {exposure: DISPLAY[exposure][1] for exposure in EXPOSURES},
    }
    (EVIDENCE / "climate_to_npp_robustness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 13.2), facecolor="white")
    fig.subplots_adjust(left=0.18, right=0.975, top=0.975, bottom=0.07, wspace=0.26, hspace=0.24)
    for ax, exposure, panel, show_y_labels in zip(
        axes.flat,
        EXPOSURES,
        ["a", "b", "c", "d"],
        [True, False, True, False],
        strict=True,
    ):
        draw_panel(ax, coefficients, leave_one_year, exposure, panel, show_y_labels)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(sample_support.to_string(index=False))
    for exposure in EXPOSURES:
        primary = coefficients.loc[
            coefficients["Exposure"].eq(exposure)
            & coefficients["Specification"].eq("Primary: 5 km strict cropland")
        ].iloc[0]
        loo = leave_one_year.loc[leave_one_year["Exposure"].eq(exposure), "Estimate"]
        print(
            f"{exposure}: primary={primary['Estimate']:.6f}; "
            f"CI=[{primary['95 Percent CI Lower']:.6f}, {primary['95 Percent CI Upper']:.6f}]; "
            f"LOO=[{loo.min():.6f}, {loo.max():.6f}]"
        )
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved evidence: {EVIDENCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
