#!/usr/bin/env python3
"""Cross-Domain Amplification Effects.

Plan: Summarize standardized historical-conflict-by-shock estimates across
agriculture, consumption, food security, and education.
Framework: AnaSOP Sections 5.1-5.2, 6.2-6.6, and the synthesis step in
Section 7. Estimates measure differential sensitivity, not a causal effect of
historical conflict.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
HOUSEHOLD_PATH = (
    ROOT / "data/processed/direction3_household_conflict_shock_preprocessed.parquet"
)
EDUCATION_PATH = (
    ROOT / "data/processed/direction3_education_conflict_shock_preprocessed.parquet"
)
OUTPUT_PATH = ROOT / "data/results/figures/Figure_cross_domain_amplification_effects.png"

YEAR = "Survey Year"
RESOLUTION = "Climate Geography Resolution"
GEOGRAPHY = "Climate Geography Code"
HOUSEHOLD_PROVINCE = "Province Code Component"
EDUCATION_PROVINCE = "Province Code"
HOUSEHOLD_WEIGHT = "Household Survey Weight"
PERSON_WEIGHT = "Person Survey Weight"
HOUSEHOLD_SIZE = "Household Size"
AGE = "Age Years"
FEMALE = "Female"
AGRICULTURAL_HOUSEHOLD = "Agricultural Household"
CONFLICT = "Log Bombing Unique Locations per 100 km2"
SPI12 = "Interview Month SPI 12 Month"
EXTREME_WET = "Annual Rainfall Extreme Wet Shock"
PRICE_12M = "12 Month Change in Local Relative Log Wholesale Rice Price"
CROP_YIELD = "Crop Yield kg per ha"
CROP_VALUE = "Real 2021 Crop Production Value Riels"
FOOD_CONSUMPTION = "Real 2021 Food Consumption Value per Household Member Riels"
SEVERE_FOOD_INSECURITY = "Any Severe Food Insecurity Experience"
ATTENDANCE = "Currently Attending School"
EDUCATION_EXPENDITURE = "Real 2021 Education Expenditure Riels"

SHOCK_LABELS = {
    SPI12: "SPI-12",
    EXTREME_WET: "Extreme wet",
    PRICE_12M: "Rice-price change",
}
SHOCK_LEGEND_LABELS = {
    SPI12: "SPI-12 (1 SD)",
    EXTREME_WET: "Extreme wet (0→1)",
    PRICE_12M: "Rice-price change (1 SD)",
}
SHOCK_STYLES = {
    SPI12: {"color": "#2B8CBE", "marker": "o"},
    EXTREME_WET: {"color": "#66C2A4", "marker": "s"},
    PRICE_12M: {"color": "#D95F0E", "marker": "D"},
}
OUTCOME_ROW_LABELS = {
    CROP_YIELD: "Yield",
    CROP_VALUE: "Production value",
    FOOD_CONSUMPTION: "Food consumption",
    SEVERE_FOOD_INSECURITY: "Severe food insecurity",
    ATTENDANCE: "Attendance",
    EDUCATION_EXPENDITURE: "Expenditure",
}


@dataclass(frozen=True)
class EstimateSpec:
    domain: str
    outcome: str
    outcome_label: str
    shock: str
    source: str
    transform: str
    school_age_only: bool = False
    agriculture_only: bool = False


@dataclass
class Estimate:
    spec: EstimateSpec
    coefficient: float
    standard_error: float
    sample_size: int
    cluster_count: int

    @property
    def lower(self) -> float:
        return self.coefficient - 1.96 * self.standard_error

    @property
    def upper(self) -> float:
        return self.coefficient + 1.96 * self.standard_error


OUTCOMES = [
    ("Agriculture", CROP_YIELD, "Crop yield", "household", "asinh", False, True),
    ("Agriculture", CROP_VALUE, "Crop production value", "household", "asinh", False, True),
    ("Consumption", FOOD_CONSUMPTION, "Food consumption", "household", "asinh", False, False),
    ("Food security", SEVERE_FOOD_INSECURITY, "Severe food insecurity", "household", "level", False, False),
    ("Education", ATTENDANCE, "School attendance", "education", "level", True, False),
    ("Education", EDUCATION_EXPENDITURE, "Education expenditure", "education", "asinh", False, False),
]
SHOCKS = [SPI12, EXTREME_WET, PRICE_12M]
ESTIMATE_SPECS = [
    EstimateSpec(domain, outcome, label, shock, source, transform, school_age, agriculture)
    for domain, outcome, label, source, transform, school_age, agriculture in OUTCOMES
    for shock in SHOCKS
]


def weighted_standardize(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    standard_deviation = float(np.sqrt(variance))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Cannot standardize a variable with zero or invalid weighted variance")
    return (values - mean) / standard_deviation


def transform_outcome(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "asinh":
        if np.nanmin(values) < 0:
            raise ValueError("asinh outcome unexpectedly contains negative values")
        return np.arcsinh(values)
    if transform == "level":
        return values
    raise ValueError(f"Unknown outcome transform: {transform}")


def prepare_sample(
    household: pd.DataFrame, education: pd.DataFrame, spec: EstimateSpec
) -> tuple[pd.DataFrame, str, str, list[str]]:
    if spec.source == "household":
        source = household
        weight = HOUSEHOLD_WEIGHT
        province = HOUSEHOLD_PROVINCE
        controls = [HOUSEHOLD_SIZE]
    else:
        source = education
        weight = PERSON_WEIGHT
        province = EDUCATION_PROVINCE
        controls = [HOUSEHOLD_SIZE, AGE, FEMALE]

    required = [
        spec.outcome,
        spec.shock,
        CONFLICT,
        weight,
        GEOGRAPHY,
        province,
        YEAR,
        *controls,
    ]
    if spec.agriculture_only:
        required.append(AGRICULTURAL_HOUSEHOLD)
    sample = source.loc[source[RESOLUTION].eq("commune"), required].dropna().copy()
    sample = sample.loc[sample[weight].gt(0)].copy()
    if spec.school_age_only:
        sample = sample.loc[sample[AGE].between(6, 17)].copy()
    if spec.agriculture_only:
        sample = sample.loc[sample[AGRICULTURAL_HOUSEHOLD].astype(bool)].copy()
    if spec.outcome in {SEVERE_FOOD_INSECURITY, ATTENDANCE}:
        assert sample[spec.outcome].isin([0, 1, False, True]).all()
    if spec.shock == EXTREME_WET:
        assert sample[spec.shock].isin([0, 1, False, True]).all()
    if sample[GEOGRAPHY].nunique() < 50:
        raise ValueError(f"Too few clusters for {spec.outcome_label} × {SHOCK_LABELS[spec.shock]}")
    return sample, weight, province, controls


def fit_estimate(
    household: pd.DataFrame, education: pd.DataFrame, spec: EstimateSpec
) -> Estimate:
    sample, weight_column, province_column, controls = prepare_sample(
        household, education, spec
    )
    weights = sample[weight_column].to_numpy(dtype=float)
    conflict_z = weighted_standardize(sample[CONFLICT].to_numpy(dtype=float), weights)
    raw_shock = sample[spec.shock].to_numpy(dtype=float)
    shock_model = (
        raw_shock
        if spec.shock == EXTREME_WET
        else weighted_standardize(raw_shock, weights)
    )
    transformed_outcome = transform_outcome(
        sample[spec.outcome].to_numpy(dtype=float), spec.transform
    )
    outcome_z = weighted_standardize(transformed_outcome, weights)

    exogenous = pd.DataFrame(
        {
            "shock": shock_model,
            "conflict_x_shock": conflict_z * shock_model,
        },
        index=sample.index,
    )
    for control in controls:
        exogenous[control] = sample[control].to_numpy(dtype=float)
    province_wave = (
        sample[province_column].astype(str)
        + "_"
        + sample[YEAR].astype(int).astype(str)
    )
    absorbed = pd.DataFrame(
        {
            "geography": sample[GEOGRAPHY].astype("category"),
            "province_wave": province_wave.astype("category"),
        },
        index=sample.index,
    )
    fitted = AbsorbingLS(
        dependent=pd.Series(outcome_z, index=sample.index, name="outcome_z"),
        exog=exogenous,
        absorb=absorbed,
        weights=pd.Series(weights, index=sample.index),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=sample[[GEOGRAPHY]],
        debiased=True,
    )
    if "conflict_x_shock" not in fitted.params.index:
        raise ValueError(
            f"Interaction absorbed for {spec.outcome_label} × {SHOCK_LABELS[spec.shock]}"
        )
    return Estimate(
        spec=spec,
        coefficient=float(fitted.params["conflict_x_shock"]),
        standard_error=float(fitted.std_errors["conflict_x_shock"]),
        sample_size=len(sample),
        cluster_count=int(sample[GEOGRAPHY].nunique()),
    )


def plot_domain(ax: plt.Axes, estimates: list[Estimate], letter: str, domain: str) -> None:
    outcome_count = len({estimate.spec.outcome for estimate in estimates})
    labels = []
    for estimate in estimates:
        shock_label = SHOCK_LABELS[estimate.spec.shock]
        if outcome_count == 1:
            labels.append(shock_label)
        else:
            labels.append(
                f"{OUTCOME_ROW_LABELS[estimate.spec.outcome]} · {shock_label}"
            )
    positions = np.arange(len(estimates))[::-1]
    for position, estimate in zip(positions, estimates, strict=True):
        style = SHOCK_STYLES[estimate.spec.shock]
        ax.errorbar(
            estimate.coefficient,
            position,
            xerr=1.96 * estimate.standard_error,
            fmt=style["marker"],
            markersize=5.3,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.25,
            capsize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
    ax.axvline(0, color="#6B7280", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_yticks(positions, labels)
    ax.set_ylim(-0.8, len(estimates) - 0.2)
    ax.text(
        -0.10,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.0,
        1.05,
        domain,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(axis="x", colors="#374151", labelsize=8)
    ax.tick_params(axis="y", colors="#374151", labelsize=8, length=0)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)


def main() -> None:
    household_columns = sorted(
        {
            YEAR,
            RESOLUTION,
            GEOGRAPHY,
            HOUSEHOLD_PROVINCE,
            HOUSEHOLD_WEIGHT,
            HOUSEHOLD_SIZE,
            AGRICULTURAL_HOUSEHOLD,
            CONFLICT,
            *SHOCKS,
            CROP_YIELD,
            CROP_VALUE,
            FOOD_CONSUMPTION,
            SEVERE_FOOD_INSECURITY,
        }
    )
    education_columns = sorted(
        {
            YEAR,
            RESOLUTION,
            GEOGRAPHY,
            EDUCATION_PROVINCE,
            PERSON_WEIGHT,
            HOUSEHOLD_SIZE,
            AGE,
            FEMALE,
            CONFLICT,
            *SHOCKS,
            ATTENDANCE,
            EDUCATION_EXPENDITURE,
        }
    )
    household = pd.read_parquet(HOUSEHOLD_PATH, columns=household_columns)
    education = pd.read_parquet(EDUCATION_PATH, columns=education_columns)
    assert len(household) == 62_920
    assert len(education) == 268_485

    estimates = [fit_estimate(household, education, spec) for spec in ESTIMATE_SPECS]
    domains = ["Agriculture", "Consumption", "Food security", "Education"]
    grouped = {
        domain: [estimate for estimate in estimates if estimate.spec.domain == domain]
        for domain in domains
    }

    lower = min(estimate.lower for estimate in estimates)
    upper = max(estimate.upper for estimate in estimates)
    bound = 1.08 * max(abs(lower), abs(upper))
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("Invalid global forest-plot bounds")

    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 9.2), sharex=True)
    for ax, letter, domain in zip(axes.flat, "abcd", domains, strict=True):
        plot_domain(ax, grouped[domain], letter, domain)
        ax.set_xlim(-bound, bound)
    for ax in axes[1, :]:
        ax.set_xlabel("Conflict × shock estimate (outcome standard deviations)", fontsize=8.8)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=SHOCK_STYLES[shock]["marker"],
            color=SHOCK_STYLES[shock]["color"],
            linestyle="none",
            markersize=6,
            label=SHOCK_LEGEND_LABELS[shock],
        )
        for shock in SHOCKS
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.16,
        right=0.985,
        top=0.965,
        bottom=0.09,
        wspace=0.46,
        hspace=0.27,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print("Panels: 4 (a-d), no figure title")
    print(f"Estimates: {len(estimates)}")
    for estimate in estimates:
        z_value = estimate.coefficient / estimate.standard_error
        print(
            f"{estimate.spec.domain} | {estimate.spec.outcome_label} | "
            f"{SHOCK_LABELS[estimate.spec.shock]}: "
            f"beta={estimate.coefficient:.6f}, SE={estimate.standard_error:.6f}, "
            f"z={z_value:.3f}, N={estimate.sample_size:,}, "
            f"clusters={estimate.cluster_count:,}"
        )


if __name__ == "__main__":
    main()
