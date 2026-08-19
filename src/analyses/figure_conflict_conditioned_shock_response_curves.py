#!/usr/bin/env python3
"""Conflict-Conditioned Shock Response Curves.

Plan: Show how agricultural and welfare outcomes respond to drought,
extreme-wet rainfall, and 12-month local food-price shocks at low, mean, and
high historical conflict exposure.
Framework: AnaSOP Sections 5.1-5.2, 6.2-6.6, and the drought, wet-shock, and
food-price workflow steps in Section 7. Curves show differential sensitivity,
not a causal effect of historical conflict.
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


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    ROOT / "data/processed/direction3_household_conflict_shock_preprocessed.parquet"
)
OUTPUT_PATH = (
    ROOT / "data/results/figures/Figure_conflict_conditioned_shock_response_curves.png"
)

YEAR = "Survey Year"
RESOLUTION = "Climate Geography Resolution"
GEOGRAPHY = "Climate Geography Code"
PROVINCE = "Province Code Component"
WEIGHT = "Household Survey Weight"
HOUSEHOLD_SIZE = "Household Size"
CONFLICT = "Log Bombing Unique Locations per 100 km2"
SPI12 = "Interview Month SPI 12 Month"
EXTREME_WET = "Annual Rainfall Extreme Wet Shock"
PRICE_12M = "12 Month Change in Local Relative Log Wholesale Rice Price"
CROP_YIELD = "Crop Yield kg per ha"
LOSS_SHARE = "Post Harvest Loss Share"
FOOD_CONSUMPTION = "Real 2021 Food Consumption Value per Household Member Riels"
SEVERE_FOOD_INSECURITY = "Any Severe Food Insecurity Experience"

CONFLICT_LEVELS = {
    "Low conflict (−1 SD)": -1.0,
    "Mean conflict": 0.0,
    "High conflict (+1 SD)": 1.0,
}
COLORS = {
    "Low conflict (−1 SD)": "#2B8CBE",
    "Mean conflict": "#7BCCC4",
    "High conflict (+1 SD)": "#D95F0E",
}


@dataclass(frozen=True)
class PanelSpec:
    letter: str
    outcome: str
    shock: str
    descriptor: str
    x_label: str
    y_label: str
    outcome_transform: str
    binary_shock: bool = False


PANEL_SPECS = [
    PanelSpec(
        "a",
        CROP_YIELD,
        SPI12,
        "Crop yield · drought",
        "Interview-month SPI-12 (SD)",
        "Change in asinh crop yield",
        "asinh",
    ),
    PanelSpec(
        "b",
        FOOD_CONSUMPTION,
        SPI12,
        "Food consumption · drought",
        "Interview-month SPI-12 (SD)",
        "Change in asinh food consumption",
        "asinh",
    ),
    PanelSpec(
        "c",
        CROP_YIELD,
        EXTREME_WET,
        "Crop yield · extreme wet",
        "Annual extreme-wet rainfall",
        "Change in asinh crop yield",
        "asinh",
        True,
    ),
    PanelSpec(
        "d",
        LOSS_SHARE,
        EXTREME_WET,
        "Post-harvest loss · extreme wet",
        "Annual extreme-wet rainfall",
        "Change in loss share (percentage points)",
        "percentage_points",
        True,
    ),
    PanelSpec(
        "e",
        FOOD_CONSUMPTION,
        PRICE_12M,
        "Food consumption · rice-price shock",
        "12-month local relative rice-price shock (SD)",
        "Change in asinh food consumption",
        "asinh",
    ),
    PanelSpec(
        "f",
        SEVERE_FOOD_INSECURITY,
        PRICE_12M,
        "Severe food insecurity · rice-price shock",
        "12-month local relative rice-price shock (SD)",
        "Change in probability (percentage points)",
        "percentage_points",
    ),
]


@dataclass
class ModelResult:
    spec: PanelSpec
    sample_size: int
    cluster_count: int
    shock_coefficient: float
    interaction_coefficient: float
    shock_variance: float
    interaction_variance: float
    covariance: float
    x_values: np.ndarray
    conflict_mean: float
    conflict_sd: float
    shock_mean: float
    shock_sd: float


def weighted_standardize(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    standard_deviation = float(np.sqrt(variance))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Cannot standardize a variable with zero or invalid weighted variance")
    return (values - mean) / standard_deviation, mean, standard_deviation


def transform_outcome(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "asinh":
        if np.nanmin(values) < 0:
            raise ValueError("asinh outcome unexpectedly contains negative values")
        return np.arcsinh(values)
    if transform == "percentage_points":
        return 100.0 * values
    raise ValueError(f"Unknown outcome transform: {transform}")


def prepare_model_sample(data: pd.DataFrame, spec: PanelSpec) -> pd.DataFrame:
    required = [
        spec.outcome,
        spec.shock,
        CONFLICT,
        WEIGHT,
        HOUSEHOLD_SIZE,
        GEOGRAPHY,
        PROVINCE,
        YEAR,
    ]
    sample = data.loc[data[RESOLUTION].eq("commune"), required].dropna().copy()
    sample = sample.loc[sample[WEIGHT].gt(0)].copy()
    if spec.outcome in {CROP_YIELD, LOSS_SHARE}:
        sample = sample.loc[sample[spec.outcome].notna()].copy()
    if spec.outcome == LOSS_SHARE:
        assert sample[spec.outcome].between(0, 1).all()
    if spec.outcome == SEVERE_FOOD_INSECURITY:
        assert sample[spec.outcome].isin([0, 1, False, True]).all()
    if spec.binary_shock:
        assert sample[spec.shock].isin([0, 1, False, True]).all()
    if sample[GEOGRAPHY].nunique() < 50:
        raise ValueError(f"Too few geography clusters for {spec.descriptor}")
    return sample


def fit_model(data: pd.DataFrame, spec: PanelSpec) -> ModelResult:
    sample = prepare_model_sample(data, spec)
    weights = sample[WEIGHT].to_numpy(dtype=float)
    conflict_z, conflict_mean, conflict_sd = weighted_standardize(
        sample[CONFLICT].to_numpy(dtype=float), weights
    )

    raw_shock = sample[spec.shock].to_numpy(dtype=float)
    if spec.binary_shock:
        shock_model = raw_shock.copy()
        shock_mean = 0.0
        shock_sd = 1.0
        x_values = np.array([0.0, 1.0])
    else:
        shock_model, shock_mean, shock_sd = weighted_standardize(raw_shock, weights)
        lower, upper = np.quantile(shock_model, [0.05, 0.95])
        x_values = np.linspace(float(lower), float(upper), 80)

    outcome = transform_outcome(
        sample[spec.outcome].to_numpy(dtype=float), spec.outcome_transform
    )
    exogenous = pd.DataFrame(
        {
            "shock": shock_model,
            "conflict_x_shock": conflict_z * shock_model,
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
    clusters = sample[[GEOGRAPHY]].copy()

    model = AbsorbingLS(
        dependent=pd.Series(outcome, index=sample.index, name="outcome"),
        exog=exogenous,
        absorb=absorbed,
        weights=pd.Series(weights, index=sample.index),
        drop_absorbed=True,
    )
    fitted = model.fit(
        cov_type="clustered",
        clusters=clusters,
        debiased=True,
    )
    required_parameters = {"shock", "conflict_x_shock"}
    if not required_parameters.issubset(fitted.params.index):
        raise ValueError(
            f"Required coefficients were absorbed for {spec.descriptor}: {fitted.params.index.tolist()}"
        )
    covariance = fitted.cov
    return ModelResult(
        spec=spec,
        sample_size=len(sample),
        cluster_count=int(sample[GEOGRAPHY].nunique()),
        shock_coefficient=float(fitted.params["shock"]),
        interaction_coefficient=float(fitted.params["conflict_x_shock"]),
        shock_variance=float(covariance.loc["shock", "shock"]),
        interaction_variance=float(
            covariance.loc["conflict_x_shock", "conflict_x_shock"]
        ),
        covariance=float(covariance.loc["shock", "conflict_x_shock"]),
        x_values=x_values,
        conflict_mean=conflict_mean,
        conflict_sd=conflict_sd,
        shock_mean=shock_mean,
        shock_sd=shock_sd,
    )


def curve_and_interval(
    result: ModelResult, conflict_level: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope = result.shock_coefficient + conflict_level * result.interaction_coefficient
    slope_variance = (
        result.shock_variance
        + conflict_level**2 * result.interaction_variance
        + 2.0 * conflict_level * result.covariance
    )
    slope_standard_error = np.sqrt(max(slope_variance, 0.0))
    estimate = result.x_values * slope
    margin = 1.96 * np.abs(result.x_values) * slope_standard_error
    return estimate, estimate - margin, estimate + margin


def plot_result(ax: plt.Axes, result: ModelResult) -> None:
    for label, conflict_level in CONFLICT_LEVELS.items():
        estimate, lower, upper = curve_and_interval(result, conflict_level)
        color = COLORS[label]
        ax.plot(result.x_values, estimate, color=color, linewidth=1.8, label=label)
        ax.fill_between(result.x_values, lower, upper, color=color, alpha=0.10, linewidth=0)

    ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--")
    if result.spec.binary_shock:
        ax.set_xticks([0, 1], ["No", "Yes"])
        ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel(result.spec.x_label, fontsize=8.7)
    ax.set_ylabel(result.spec.y_label, fontsize=8.7)
    ax.text(
        -0.10,
        1.05,
        result.spec.letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.0,
        1.04,
        result.spec.descriptor,
        transform=ax.transAxes,
        fontsize=9.8,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.98,
        0.97,
        f"N = {result.sample_size:,} · clusters = {result.cluster_count:,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#4B5563",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#9CA3AF")
    ax.tick_params(colors="#374151", labelsize=8)
    ax.grid(color="#E5E7EB", linewidth=0.65)


def main() -> None:
    columns = sorted(
        {
            YEAR,
            RESOLUTION,
            GEOGRAPHY,
            PROVINCE,
            WEIGHT,
            HOUSEHOLD_SIZE,
            CONFLICT,
            SPI12,
            EXTREME_WET,
            PRICE_12M,
            CROP_YIELD,
            LOSS_SHARE,
            FOOD_CONSUMPTION,
            SEVERE_FOOD_INSECURITY,
        }
    )
    data = pd.read_parquet(INPUT_PATH, columns=columns)
    assert len(data) == 62_920

    results = [fit_model(data, spec) for spec in PANEL_SPECS]
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    figure, axes = plt.subplots(3, 2, figsize=(12.6, 12.2))
    for ax, result in zip(axes.flat, results, strict=True):
        plot_result(ax, result)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        top=0.975,
        bottom=0.07,
        wspace=0.23,
        hspace=0.30,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print("Panels: 6 (a-f), no figure title")
    for result in results:
        interaction_se = np.sqrt(result.interaction_variance)
        print(
            f"{result.spec.letter} {result.spec.descriptor}: "
            f"N={result.sample_size:,}, clusters={result.cluster_count:,}, "
            f"interaction={result.interaction_coefficient:.6f}, "
            f"SE={interaction_se:.6f}"
        )


if __name__ == "__main__":
    main()
