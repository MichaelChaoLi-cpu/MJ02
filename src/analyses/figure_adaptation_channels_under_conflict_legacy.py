#!/usr/bin/env python3
"""Adaptation Channels Under Conflict Legacy.

Plan: Evaluate whether irrigation capacity, crop diversification, and
agricultural investment respond differently to drought across historical
conflict exposure.
Framework: AnaSOP Sections 5.1-5.2, 6.2-6.5, 6.11, and the adaptive-capacity
step in Section 7. The three household interactions belong to the expanded
six-test Holm family. Results are supporting mechanism evidence, not causal
mediation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels.iv import AbsorbingLS

from table_mechanism_families_and_multiplicity_checks import (
    MARKET as VILLAGE_MARKET,
    PSU,
    VILLAGE_IRRIGATION,
    VILLAGE_PATH,
    build_table as build_expanded_mechanism_table,
)


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    ROOT / "data/processed/direction3_household_conflict_shock_preprocessed.parquet"
)
OUTPUT_PATH = (
    ROOT / "data/exp/internal_output_archive/figures/Figure_adaptation_channels_under_conflict_legacy.png"
)

YEAR = "Survey Year"
RESOLUTION = "Climate Geography Resolution"
GEOGRAPHY = "Climate Geography Code"
PROVINCE = "Province Code Component"
WEIGHT = "Household Survey Weight"
HOUSEHOLD_SIZE = "Household Size"
AGRICULTURAL_HOUSEHOLD = "Agricultural Household"
CONFLICT = "Log Bombing Unique Locations per 100 km2"
SPI12 = "Interview Month SPI 12 Month"
IRRIGATION = "Irrigable Parcel Share"
DIVERSITY = "Crop Diversity Count"
INPUT_COST = "Real 2021 Agricultural Input Cost Riels"

CONFLICT_LEVELS = [
    ("Low conflict (−1 SD)", -1.0, "#2B8CBE"),
    ("Mean conflict", 0.0, "#7BCCC4"),
    ("High conflict (+1 SD)", 1.0, "#D95F0E"),
]


@dataclass(frozen=True)
class PanelSpec:
    letter: str
    outcome: str
    descriptor: str
    x_label: str
    transform: str


PANEL_SPECS = [
    PanelSpec(
        "a",
        IRRIGATION,
        "Irrigable parcel share",
        "Response to 1 SD greater drought severity (percentage points)",
        "percentage_points",
    ),
    PanelSpec(
        "b",
        DIVERSITY,
        "Crop diversity",
        "Response to 1 SD greater drought severity (crop count)",
        "level",
    ),
    PanelSpec(
        "c",
        INPUT_COST,
        "Agricultural input cost",
        "Response to 1 SD greater drought severity (asinh units)",
        "asinh",
    ),
]


@dataclass
class ModelResult:
    spec: PanelSpec
    sample_size: int
    cluster_count: int
    drought_coefficient: float
    interaction_coefficient: float
    drought_variance: float
    interaction_variance: float
    covariance: float


def weighted_standardize(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    standard_deviation = float(np.sqrt(variance))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Cannot standardize a variable with zero or invalid weighted variance")
    return (values - mean) / standard_deviation


def transform_outcome(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "percentage_points":
        if np.nanmin(values) < 0 or np.nanmax(values) > 1:
            raise ValueError("Share outcome falls outside [0, 1]")
        return 100.0 * values
    if transform == "level":
        return values
    if transform == "asinh":
        if np.nanmin(values) < 0:
            raise ValueError("Input-cost outcome unexpectedly contains negative values")
        return np.arcsinh(values)
    raise ValueError(f"Unknown transform: {transform}")


def fit_model(data: pd.DataFrame, spec: PanelSpec) -> ModelResult:
    required = [
        spec.outcome,
        SPI12,
        CONFLICT,
        WEIGHT,
        HOUSEHOLD_SIZE,
        GEOGRAPHY,
        PROVINCE,
        YEAR,
        AGRICULTURAL_HOUSEHOLD,
    ]
    sample = data.loc[data[RESOLUTION].eq("commune"), required].dropna().copy()
    sample = sample.loc[
        sample[WEIGHT].gt(0) & sample[AGRICULTURAL_HOUSEHOLD].astype(bool)
    ].copy()
    if sample[GEOGRAPHY].nunique() < 50:
        raise ValueError(f"Too few geography clusters for {spec.descriptor}")

    weights = sample[WEIGHT].to_numpy(dtype=float)
    conflict_z = weighted_standardize(sample[CONFLICT].to_numpy(dtype=float), weights)
    spi_z = weighted_standardize(sample[SPI12].to_numpy(dtype=float), weights)
    drought_severity = -spi_z
    outcome = transform_outcome(sample[spec.outcome].to_numpy(dtype=float), spec.transform)

    exogenous = pd.DataFrame(
        {
            "drought_severity": drought_severity,
            "conflict_x_drought": conflict_z * drought_severity,
            "household_size": sample[HOUSEHOLD_SIZE].to_numpy(dtype=float),
        },
        index=sample.index,
    )
    province_wave = (
        sample[PROVINCE].astype(str) + "_" + sample[YEAR].astype(int).astype(str)
    )
    absorbed = pd.DataFrame(
        {
            "geography": sample[GEOGRAPHY].astype("category"),
            "province_wave": province_wave.astype("category"),
        },
        index=sample.index,
    )
    fitted = AbsorbingLS(
        dependent=pd.Series(outcome, index=sample.index, name="outcome"),
        exog=exogenous,
        absorb=absorbed,
        weights=pd.Series(weights, index=sample.index),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=sample[[GEOGRAPHY]],
        debiased=True,
    )
    required_parameters = {"drought_severity", "conflict_x_drought"}
    if not required_parameters.issubset(fitted.params.index):
        raise ValueError(
            f"Required coefficients were absorbed for {spec.descriptor}: {fitted.params.index.tolist()}"
        )
    covariance = fitted.cov
    return ModelResult(
        spec=spec,
        sample_size=len(sample),
        cluster_count=int(sample[GEOGRAPHY].nunique()),
        drought_coefficient=float(fitted.params["drought_severity"]),
        interaction_coefficient=float(fitted.params["conflict_x_drought"]),
        drought_variance=float(covariance.loc["drought_severity", "drought_severity"]),
        interaction_variance=float(
            covariance.loc["conflict_x_drought", "conflict_x_drought"]
        ),
        covariance=float(covariance.loc["drought_severity", "conflict_x_drought"]),
    )


def marginal_response(result: ModelResult, conflict_level: float) -> tuple[float, float]:
    estimate = result.drought_coefficient + conflict_level * result.interaction_coefficient
    variance = (
        result.drought_variance
        + conflict_level**2 * result.interaction_variance
        + 2.0 * conflict_level * result.covariance
    )
    return estimate, float(np.sqrt(max(variance, 0.0)))


def plot_panel(ax: plt.Axes, result: ModelResult, holm_p: float) -> None:
    positions = np.arange(len(CONFLICT_LEVELS))[::-1]
    for position, (label, conflict_level, color) in zip(
        positions, CONFLICT_LEVELS, strict=True
    ):
        estimate, standard_error = marginal_response(result, conflict_level)
        ax.errorbar(
            estimate,
            position,
            xerr=1.96 * standard_error,
            fmt="o",
            markersize=6,
            color=color,
            ecolor=color,
            elinewidth=1.35,
            capsize=2.8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
        )
    ax.axvline(0, color="#6B7280", linewidth=0.9, linestyle="--")
    ax.set_yticks(positions, [item[0] for item in CONFLICT_LEVELS])
    ax.set_ylim(-0.7, len(CONFLICT_LEVELS) - 0.2)
    ax.set_xlabel(result.spec.x_label, fontsize=8.5)
    ax.text(
        -0.10,
        1.06,
        result.spec.letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.0,
        1.05,
        result.spec.descriptor,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.98,
        0.97,
        (
            f"N = {result.sample_size:,} · clusters = {result.cluster_count:,}\n"
            f"Interaction: six-test Holm p = {holm_p:.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color="#4B5563",
        linespacing=1.35,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.8},
        zorder=5,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(axis="x", colors="#374151", labelsize=8)
    ax.tick_params(axis="y", colors="#374151", labelsize=8.2, length=0)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)


def main() -> None:
    columns = [
        YEAR,
        RESOLUTION,
        GEOGRAPHY,
        PROVINCE,
        WEIGHT,
        HOUSEHOLD_SIZE,
        AGRICULTURAL_HOUSEHOLD,
        CONFLICT,
        SPI12,
        IRRIGATION,
        DIVERSITY,
        INPUT_COST,
        PSU,
    ]
    data = pd.read_parquet(INPUT_PATH, columns=columns)
    assert len(data) == 62_920
    results = [fit_model(data, spec) for spec in PANEL_SPECS]

    villages = pd.read_parquet(
        VILLAGE_PATH,
        columns=[YEAR, PSU, VILLAGE_IRRIGATION, VILLAGE_MARKET],
    )
    expanded_frame, _ = build_expanded_mechanism_table(villages, data)
    expanded_household = expanded_frame.loc[
        expanded_frame["Variable"].isin([IRRIGATION, DIVERSITY, INPUT_COST])
    ]
    expanded_holm = expanded_household.set_index("Variable")["Holm p-value"].to_dict()
    if set(expanded_holm) != {IRRIGATION, DIVERSITY, INPUT_COST}:
        raise ValueError("Expanded six-test mechanism family is incomplete")
    assert round(float(expanded_holm[IRRIGATION]), 3) == 0.106

    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.7))
    for ax, result in zip(axes, results, strict=True):
        plot_panel(ax, result, float(expanded_holm[result.spec.outcome]))
    figure.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.17, wspace=0.48)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print("Panels: 3 (a-c), no figure title")
    for result in results:
        interaction_se = np.sqrt(result.interaction_variance)
        print(
            f"{result.spec.letter} {result.spec.descriptor}: "
            f"N={result.sample_size:,}, clusters={result.cluster_count:,}, "
            f"conflict×drought={result.interaction_coefficient:.6f}, "
            f"SE={interaction_se:.6f}, "
            f"six-test Holm p={expanded_holm[result.spec.outcome]:.6f}"
        )
        for label, conflict_level, _ in CONFLICT_LEVELS:
            estimate, standard_error = marginal_response(result, conflict_level)
            print(f"  {label}: response={estimate:.6f}, SE={standard_error:.6f}")


if __name__ == "__main__":
    main()
