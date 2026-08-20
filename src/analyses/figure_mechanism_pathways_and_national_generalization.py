#!/usr/bin/env python3
"""Mechanism Pathways and National Generalization.

Plan: Contrast standardized village-infrastructure gradients, drought-sensitive
agricultural-capacity estimates, representative national outcome responses, and
the activated local boundary results without pooling distinct estimands.
Framework: AnaSOP Sections 5.1-5.6, 6.2, 6.9-6.11, and the mechanism and
integration steps in Section 7. Mechanism estimates are channel-consistent
associations; causal interpretation is restricted to the qualified local design.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
VILLAGE_PATH = ROOT / "data/processed/direction3_village_mechanisms_preprocessed.parquet"
HOUSEHOLD_PATH = ROOT / "data/processed/direction3_household_conflict_shock_preprocessed.parquet"
EDUCATION_PATH = ROOT / "data/processed/direction3_education_conflict_shock_preprocessed.parquet"
NPP_TABLE = ROOT / "data/results/tables/Table_historical_boundary_shock_response_estimates.xlsx"
VIIRS_TABLE = ROOT / "data/results/tables/Table_nighttime_activity_independent_validation_estimates.xlsx"
OUTPUT_PATH = ROOT / "data/results/figures/Figure_mechanism_pathways_and_national_generalization.png"
ESTIMATE_PATH = ROOT / "data/exp/mechanism-pathways-and-national-generalization/figure_estimates.csv"

YEAR = "Survey Year"
PSU = "PSU"
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
PRICE = "12 Month Change in Local Relative Log Wholesale Rice Price"

VILLAGE_IRRIGATION = "Village Irrigated Agricultural Land Share"
MARKET = "Permanent Market Access"
PARCEL_IRRIGATION = "Irrigable Parcel Share"
DIVERSITY = "Crop Diversity Count"
INPUT_COST = "Real 2021 Agricultural Input Cost Riels"
CROP_YIELD = "Crop Yield kg per ha"
FOOD_CONSUMPTION = "Real 2021 Food Consumption Value per Household Member Riels"
SEVERE_FOOD_INSECURITY = "Any Severe Food Insecurity Experience"
ATTENDANCE = "Currently Attending School"


@dataclass(frozen=True)
class PlotEstimate:
    panel: str
    label: str
    estimate: float
    lower: float
    upper: float
    sample_size: int | None
    source: str


def standardize(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        mean = float(np.mean(values))
        variance = float(np.mean((values - mean) ** 2))
    else:
        mean = float(np.average(values, weights=weights))
        variance = float(np.average((values - mean) ** 2, weights=weights))
    standard_deviation = float(np.sqrt(variance))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Cannot standardize a variable with zero or invalid variance")
    return (values - mean) / standard_deviation


def fit_model(
    sample: pd.DataFrame,
    outcome: np.ndarray,
    exogenous: pd.DataFrame,
    absorbed: pd.DataFrame,
    target: str,
    weights: np.ndarray | None,
    panel: str,
    label: str,
    source: str,
) -> PlotEstimate:
    fitted = AbsorbingLS(
        dependent=pd.Series(outcome, index=sample.index, name="outcome"),
        exog=exogenous,
        absorb=absorbed,
        weights=(
            pd.Series(weights, index=sample.index, name="weight")
            if weights is not None
            else None
        ),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=pd.DataFrame({"geography": sample[GEOGRAPHY]}, index=sample.index),
        debiased=True,
    )
    if target not in fitted.params.index:
        raise ValueError(f"Target parameter was absorbed: {label}")
    estimate = float(fitted.params[target])
    standard_error = float(fitted.std_errors[target])
    return PlotEstimate(
        panel=panel,
        label=label,
        estimate=estimate,
        lower=estimate - 1.96 * standard_error,
        upper=estimate + 1.96 * standard_error,
        sample_size=len(sample),
        source=source,
    )


def collapse_household_spine(households: pd.DataFrame) -> pd.DataFrame:
    keys = [YEAR, PSU]
    invariant = [RESOLUTION, GEOGRAPHY, HOUSEHOLD_PROVINCE, CONFLICT, SPI12]
    for column in invariant:
        if households.groupby(keys, observed=True)[column].nunique(dropna=False).max() > 1:
            raise ValueError(f"Exposure field varies within PSU-year: {column}")
    return households.groupby(keys, as_index=False, observed=True).agg(
        **{column: (column, "first") for column in invariant}
    )


def village_panel(villages: pd.DataFrame, households: pd.DataFrame) -> pd.DataFrame:
    panel = villages.merge(
        collapse_household_spine(households),
        on=[YEAR, PSU],
        how="left",
        validate="one_to_one",
    )
    panel[GEOGRAPHY] = panel[GEOGRAPHY].astype("string").str.zfill(6)
    return panel


def village_persistence_estimate(
    panel: pd.DataFrame, outcome: str, label: str
) -> PlotEstimate:
    required = [outcome, CONFLICT, GEOGRAPHY, HOUSEHOLD_PROVINCE, YEAR]
    sample = panel.loc[panel[RESOLUTION].eq("commune"), required].dropna().copy()
    outcome_z = standardize(sample[outcome].to_numpy(dtype=float))
    conflict_z = standardize(sample[CONFLICT].to_numpy(dtype=float))
    province_wave = (
        sample[HOUSEHOLD_PROVINCE].astype(str)
        + "_"
        + sample[YEAR].astype(int).astype(str)
    )
    return fit_model(
        sample,
        outcome_z,
        pd.DataFrame({"conflict": conflict_z}, index=sample.index),
        pd.DataFrame(
            {"province_wave": province_wave.astype("category")}, index=sample.index
        ),
        "conflict",
        None,
        "a",
        label,
        "Village infrastructure",
    )


def village_drought_estimate(panel: pd.DataFrame) -> PlotEstimate:
    required = [
        VILLAGE_IRRIGATION,
        CONFLICT,
        SPI12,
        GEOGRAPHY,
        HOUSEHOLD_PROVINCE,
        YEAR,
    ]
    sample = panel.loc[panel[RESOLUTION].eq("commune"), required].dropna().copy()
    outcome_z = standardize(sample[VILLAGE_IRRIGATION].to_numpy(dtype=float))
    conflict_z = standardize(sample[CONFLICT].to_numpy(dtype=float))
    drought = -standardize(sample[SPI12].to_numpy(dtype=float))
    province_wave = (
        sample[HOUSEHOLD_PROVINCE].astype(str)
        + "_"
        + sample[YEAR].astype(int).astype(str)
    )
    return fit_model(
        sample,
        outcome_z,
        pd.DataFrame(
            {
                "drought": drought,
                "conflict_x_drought": conflict_z * drought,
            },
            index=sample.index,
        ),
        pd.DataFrame(
            {
                "geography": sample[GEOGRAPHY].astype("category"),
                "province_wave": province_wave.astype("category"),
            },
            index=sample.index,
        ),
        "conflict_x_drought",
        None,
        "b",
        "Village irrigated land",
        "Village infrastructure",
    )


def transformed_outcome(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "asinh":
        if np.nanmin(values) < 0:
            raise ValueError("Asinh outcome contains negative values")
        return np.arcsinh(values)
    if transform in {"level", "share"}:
        return values
    raise ValueError(f"Unknown transform: {transform}")


def household_interaction_estimate(
    data: pd.DataFrame,
    *,
    panel: str,
    label: str,
    outcome: str,
    transform: str,
    shock: str,
    source: str,
    weight_column: str,
    province_column: str,
    controls: list[str],
    agriculture_only: bool = False,
    school_age_only: bool = False,
    resilience_flip: bool = False,
) -> PlotEstimate:
    required = [
        outcome,
        shock,
        CONFLICT,
        weight_column,
        GEOGRAPHY,
        province_column,
        YEAR,
        *controls,
    ]
    if agriculture_only:
        required.append(AGRICULTURAL_HOUSEHOLD)
    sample = data.loc[data[RESOLUTION].eq("commune"), required].dropna().copy()
    sample = sample.loc[sample[weight_column].gt(0)].copy()
    if agriculture_only:
        sample = sample.loc[sample[AGRICULTURAL_HOUSEHOLD].eq(1)].copy()
    if school_age_only:
        sample = sample.loc[sample[AGE].between(6, 17)].copy()
    sample[GEOGRAPHY] = sample[GEOGRAPHY].astype(str).str.zfill(6)
    weights = sample[weight_column].to_numpy(dtype=float)
    outcome_model = transformed_outcome(
        sample[outcome].to_numpy(dtype=float), transform
    )
    outcome_z = standardize(outcome_model, weights)
    if resilience_flip:
        outcome_z = -outcome_z
    conflict_z = standardize(sample[CONFLICT].to_numpy(dtype=float), weights)
    shock_z = standardize(sample[shock].to_numpy(dtype=float), weights)
    if shock == SPI12:
        shock_z = -shock_z
    exogenous = pd.DataFrame(
        {
            "shock": shock_z,
            "conflict_x_shock": conflict_z * shock_z,
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
    return fit_model(
        sample,
        outcome_z,
        exogenous,
        pd.DataFrame(
            {
                "geography": sample[GEOGRAPHY].astype("category"),
                "province_wave": province_wave.astype("category"),
            },
            index=sample.index,
        ),
        "conflict_x_shock",
        weights,
        panel,
        label,
        source,
    )


def parse_ci(value: object) -> tuple[float, float]:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value))
    if len(numbers) != 2:
        raise ValueError(f"Cannot parse confidence interval: {value}")
    return float(numbers[0]), float(numbers[1])


def local_boundary_estimates() -> list[PlotEstimate]:
    npp = pd.read_excel(NPP_TABLE, sheet_name="Shock Response")
    viirs = pd.read_excel(VIIRS_TABLE, sheet_name="NTL Validation")

    npp_rows = npp.loc[
        npp["Outcome"].eq("Annual land NPP anomaly (standardized)")
        & npp["Bandwidth (km)"].eq(5)
    ].iloc[:2]
    viirs_rows = viirs.iloc[:2]
    labels = [
        "Land NPP · primary",
        "Land NPP · within commune",
        "Nighttime activity · primary",
        "Nighttime activity · within commune",
    ]
    rows = [*npp_rows.to_dict("records"), *viirs_rows.to_dict("records")]
    output: list[PlotEstimate] = []
    for label, row in zip(labels, rows, strict=True):
        estimate_key = (
            "Interaction estimate" if "Interaction estimate" in row else "Interaction estimate"
        )
        lower, upper = parse_ci(row["95% CI"])
        output.append(
            PlotEstimate(
                panel="d",
                label=label,
                estimate=float(row[estimate_key]),
                lower=lower,
                upper=upper,
                sample_size=None,
                source="Historical-boundary validation",
            )
        )
    return output


def build_estimates(
    villages: pd.DataFrame,
    households: pd.DataFrame,
    education: pd.DataFrame,
) -> list[PlotEstimate]:
    panel = village_panel(villages, households)
    panel_a = [
        village_persistence_estimate(panel, VILLAGE_IRRIGATION, "Village irrigated land"),
        village_persistence_estimate(panel, MARKET, "Permanent market access"),
    ]
    panel_b = [
        village_drought_estimate(panel),
        household_interaction_estimate(
            households,
            panel="b",
            label="Irrigable household parcels",
            outcome=PARCEL_IRRIGATION,
            transform="share",
            shock=SPI12,
            source="Household agricultural capacity",
            weight_column=HOUSEHOLD_WEIGHT,
            province_column=HOUSEHOLD_PROVINCE,
            controls=[HOUSEHOLD_SIZE],
            agriculture_only=True,
        ),
        household_interaction_estimate(
            households,
            panel="b",
            label="Crop diversity",
            outcome=DIVERSITY,
            transform="level",
            shock=SPI12,
            source="Household agricultural capacity",
            weight_column=HOUSEHOLD_WEIGHT,
            province_column=HOUSEHOLD_PROVINCE,
            controls=[HOUSEHOLD_SIZE],
            agriculture_only=True,
        ),
        household_interaction_estimate(
            households,
            panel="b",
            label="Agricultural input cost",
            outcome=INPUT_COST,
            transform="asinh",
            shock=SPI12,
            source="Household agricultural capacity",
            weight_column=HOUSEHOLD_WEIGHT,
            province_column=HOUSEHOLD_PROVINCE,
            controls=[HOUSEHOLD_SIZE],
            agriculture_only=True,
        ),
    ]
    panel_c = [
        household_interaction_estimate(
            households,
            panel="c",
            label="Crop yield · drought",
            outcome=CROP_YIELD,
            transform="asinh",
            shock=SPI12,
            source="National breadth",
            weight_column=HOUSEHOLD_WEIGHT,
            province_column=HOUSEHOLD_PROVINCE,
            controls=[HOUSEHOLD_SIZE],
            agriculture_only=True,
        ),
        household_interaction_estimate(
            households,
            panel="c",
            label="Food consumption · rice price",
            outcome=FOOD_CONSUMPTION,
            transform="asinh",
            shock=PRICE,
            source="National breadth",
            weight_column=HOUSEHOLD_WEIGHT,
            province_column=HOUSEHOLD_PROVINCE,
            controls=[HOUSEHOLD_SIZE],
        ),
        household_interaction_estimate(
            households,
            panel="c",
            label="Food security · rice price",
            outcome=SEVERE_FOOD_INSECURITY,
            transform="level",
            shock=PRICE,
            source="National breadth",
            weight_column=HOUSEHOLD_WEIGHT,
            province_column=HOUSEHOLD_PROVINCE,
            controls=[HOUSEHOLD_SIZE],
            resilience_flip=True,
        ),
        household_interaction_estimate(
            education,
            panel="c",
            label="School attendance · rice price",
            outcome=ATTENDANCE,
            transform="level",
            shock=PRICE,
            source="National breadth",
            weight_column=PERSON_WEIGHT,
            province_column=EDUCATION_PROVINCE,
            controls=[HOUSEHOLD_SIZE, AGE, FEMALE],
            school_age_only=True,
        ),
    ]
    return [*panel_a, *panel_b, *panel_c, *local_boundary_estimates()]


def symmetric_limit(estimates: list[PlotEstimate], minimum: float = 0.12) -> float:
    maximum = max(max(abs(item.lower), abs(item.upper)) for item in estimates)
    return max(minimum, 1.16 * maximum)


def plot_panel(
    ax: plt.Axes,
    estimates: list[PlotEstimate],
    *,
    letter: str,
    heading: str,
    x_label: str,
    color: str,
    equivalence_band: tuple[float, float] | None = None,
    hollow_markers: bool = False,
    annotation: str | None = None,
) -> None:
    positions = np.arange(len(estimates))[::-1]
    if equivalence_band is not None:
        ax.axvspan(
            equivalence_band[0],
            equivalence_band[1],
            color="#E8EEF5",
            alpha=0.95,
            zorder=0,
        )
    for index, (position, estimate) in enumerate(zip(positions, estimates, strict=True)):
        if index % 2 == 0:
            ax.axhspan(position - 0.42, position + 0.42, color="#F7F9FB", zorder=0)
        ax.errorbar(
            estimate.estimate,
            position,
            xerr=np.array(
                [
                    [estimate.estimate - estimate.lower],
                    [estimate.upper - estimate.estimate],
                ]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            markersize=6.3,
            markerfacecolor="white" if hollow_markers else color,
            markeredgecolor=color if hollow_markers else "white",
            markeredgewidth=1.4 if hollow_markers else 0.6,
            elinewidth=1.45,
            capsize=3.0,
            zorder=3,
        )
    ax.axvline(0, color="#5F6B76", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_yticks(positions, [item.label for item in estimates])
    ax.set_ylim(-0.7, len(estimates) - 0.25)
    limit = 0.22 if equivalence_band is not None else symmetric_limit(estimates)
    ax.set_xlim(-limit, limit)
    ax.set_xlabel(x_label, fontsize=8.7, labelpad=8)
    ax.text(
        -0.08,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.0,
        1.08,
        heading,
        transform=ax.transAxes,
        fontsize=10.7,
        fontweight="bold",
        va="top",
        color="#18324A",
    )
    if annotation:
        ax.text(
            0.98,
            0.97,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.8,
            color="#64748B",
        )
    ax.grid(axis="x", color="#DCE3EA", linewidth=0.75)
    ax.tick_params(axis="x", labelsize=8.2, colors="#334155")
    ax.tick_params(axis="y", labelsize=8.5, colors="#334155", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9AA7B4")


def validate(estimates: list[PlotEstimate]) -> None:
    assert len(estimates) == 14
    assert pd.Series([item.panel for item in estimates]).value_counts().to_dict() == {
        "b": 4,
        "c": 4,
        "d": 4,
        "a": 2,
    }
    for item in estimates:
        assert np.isfinite([item.estimate, item.lower, item.upper]).all()
        assert item.lower <= item.estimate <= item.upper
    local = [item for item in estimates if item.panel == "d"]
    assert all(item.lower > -0.20 and item.upper < 0.20 for item in local)


def main() -> None:
    village_columns = [YEAR, PSU, VILLAGE_IRRIGATION, MARKET]
    household_columns = [
        YEAR,
        PSU,
        RESOLUTION,
        GEOGRAPHY,
        HOUSEHOLD_PROVINCE,
        HOUSEHOLD_WEIGHT,
        HOUSEHOLD_SIZE,
        AGRICULTURAL_HOUSEHOLD,
        CONFLICT,
        SPI12,
        PRICE,
        PARCEL_IRRIGATION,
        DIVERSITY,
        INPUT_COST,
        CROP_YIELD,
        FOOD_CONSUMPTION,
        SEVERE_FOOD_INSECURITY,
    ]
    education_columns = [
        YEAR,
        RESOLUTION,
        GEOGRAPHY,
        EDUCATION_PROVINCE,
        PERSON_WEIGHT,
        HOUSEHOLD_SIZE,
        AGE,
        FEMALE,
        CONFLICT,
        PRICE,
        ATTENDANCE,
    ]
    villages = pd.read_parquet(VILLAGE_PATH, columns=village_columns)
    households = pd.read_parquet(HOUSEHOLD_PATH, columns=household_columns)
    education = pd.read_parquet(EDUCATION_PATH, columns=education_columns)
    estimates = build_estimates(villages, households, education)
    validate(estimates)

    estimate_frame = pd.DataFrame(
        [
            {
                "Panel": item.panel,
                "Label": item.label,
                "Estimate": item.estimate,
                "Lower 95% CI": item.lower,
                "Upper 95% CI": item.upper,
                "N": item.sample_size,
                "Source family": item.source,
            }
            for item in estimates
        ]
    )
    ESTIMATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    estimate_frame.to_csv(ESTIMATE_PATH, index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.unicode_minus": True,
        }
    )
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(15.2, 10.2))
    plot_panel(
        axes[0, 0],
        [item for item in estimates if item.panel == "a"],
        letter="a",
        heading="Persistent village infrastructure",
        x_label="Conflict gradient (outcome SD per conflict SD)",
        color="#276FBF",
        hollow_markers=True,
        annotation="No Holm-adjusted p < 0.05",
    )
    plot_panel(
        axes[0, 1],
        [item for item in estimates if item.panel == "b"],
        letter="b",
        heading="Agricultural capacity under drought",
        x_label="Conflict × drought effect (outcome SD)",
        color="#2A9D8F",
        hollow_markers=True,
        annotation="No Holm-adjusted p < 0.05",
    )
    plot_panel(
        axes[1, 0],
        [item for item in estimates if item.panel == "c"],
        letter="c",
        heading="Representative national breadth",
        x_label="Conflict × shock effect (resilience-oriented outcome SD)",
        color="#D97706",
        annotation="Attendance result is definition-sensitive",
    )
    plot_panel(
        axes[1, 1],
        [item for item in estimates if item.panel == "d"],
        letter="d",
        heading="Local historical-boundary validation",
        x_label="Southwest − West rainfall response (within-unit outcome SD)",
        color="#7C3AED",
        equivalence_band=(-0.20, 0.20),
        annotation="All 95% CIs inside ±0.20 SD",
    )
    figure.subplots_adjust(left=0.205, right=0.975, top=0.93, bottom=0.09, wspace=0.56, hspace=0.48)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Saved: {ESTIMATE_PATH.relative_to(ROOT)}")
    print(estimate_frame.to_string(index=False))


if __name__ == "__main__":
    main()
