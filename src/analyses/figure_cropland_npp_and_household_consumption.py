#!/usr/bin/env python3
"""Cropland NPP and Household Consumption.

Plan: Compare the fixed national regression ladder for total and food
consumption and the corresponding slopes across the one frozen six-region
SKATER partition.
Framework: AnaSOP Sections 5-7 Stage-2 survey-weighted repeated-cross-section
model, exact survey-time controls, village-clustered inference, and expanded-
control common-sample regional interactions.
"""

from __future__ import annotations

from pathlib import Path

from linearmodels.iv import AbsorbingLS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HOUSEHOLDS = ROOT / "data/processed/cses_household_cropland_npp_analysis_preprocessed.parquet"
REGIONS = ROOT / "data/processed/outcome_blind_spatial_regions_preprocessed.parquet"
OUTPUT = ROOT / "data/results/figures/Figure_cropland_npp_and_household_consumption.png"
EVIDENCE = ROOT / "data/exp/analysis/climate-npp/cropland-npp-and-household-consumption"

ID = "National Village Point ID"
WEIGHT = "Household Survey Weight"
SURVEY_YEAR = "Interview Calendar Year"
SURVEY_MONTH = "Interview Month"
NPP = "Prior-Year Strict-Cropland NPP"
TOTAL_OUTCOME = "Log Real 2021 Annual Total Consumption per Capita"
FOOD_OUTCOME = "Log Real 2021 Annual Food Consumption per Capita"
TOTAL_FLAG = "Stage 2 Total Consumption Complete Case"
FOOD_FLAG = "Stage 2 Food Consumption Complete Case"
REGION = "SKATER Region ID"

COMPOSITION_CONTROLS = [
    "Household Size",
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
]
SOCIOECONOMIC_SOURCE = [
    "Urban Rural",
    "Agricultural Participation",
    "Household Head Ever Attended School",
]
URBAN = "Urban location"
AGRICULTURE = "Agricultural participation"
HEAD_SCHOOL = "Household head ever attended school"
SOCIOECONOMIC_CONTROLS = [URBAN, AGRICULTURE, HEAD_SCHOOL]

NPP_INCREMENT = 0.1
NAVY = "#173F5F"
BLUE = "#2F80A2"
TEAL = "#3A9D8F"
GOLD = "#D4A72C"
ORANGE = "#D97757"
PLUM = "#9B4A63"
REGION_COLORS = [NAVY, BLUE, TEAL, GOLD, ORANGE, PLUM]
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
        zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.3},
    )


def load_data() -> pd.DataFrame:
    columns = [
        ID,
        WEIGHT,
        SURVEY_YEAR,
        SURVEY_MONTH,
        NPP,
        TOTAL_OUTCOME,
        FOOD_OUTCOME,
        TOTAL_FLAG,
        FOOD_FLAG,
        *COMPOSITION_CONTROLS,
        *SOCIOECONOMIC_SOURCE,
    ]
    frame = pd.read_parquet(HOUSEHOLDS, columns=columns)
    regions = pd.read_parquet(REGIONS, columns=[ID, REGION])
    frame = frame.merge(regions, on=ID, how="left", validate="many_to_one")
    numeric = [
        WEIGHT,
        SURVEY_YEAR,
        SURVEY_MONTH,
        NPP,
        TOTAL_OUTCOME,
        FOOD_OUTCOME,
        *COMPOSITION_CONTROLS,
        "Household Head Ever Attended School",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[URBAN] = frame["Urban Rural"].astype("string").eq("1.0").astype(float)
    frame[AGRICULTURE] = frame["Agricultural Participation"].astype("boolean").astype(float)
    frame[HEAD_SCHOOL] = pd.to_numeric(frame["Household Head Ever Attended School"], errors="coerce")
    frame["Exact Survey Time"] = (
        frame[SURVEY_YEAR].astype("Int64").astype("string")
        + "-"
        + frame[SURVEY_MONTH].astype("Int64").astype("string").str.zfill(2)
    )
    return frame


def fit_weighted_absorbed(
    sample: pd.DataFrame,
    outcome: str,
    exog: pd.DataFrame,
    absorb_columns: list[str],
) -> object:
    absorb = pd.DataFrame(index=sample.index)
    for column in absorb_columns:
        absorb[column] = sample[column].astype("category")
    clusters = pd.DataFrame(
        {"Village cluster": pd.Categorical(sample[ID]).codes},
        index=sample.index,
    )
    return AbsorbingLS(
        sample[outcome].astype(float),
        exog.astype(float),
        absorb=absorb,
        weights=sample[WEIGHT].astype(float),
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)


def transformed_effect(beta: float) -> float:
    return 100.0 * (np.exp(NPP_INCREMENT * beta) - 1.0)


def coefficient_row(
    result: object,
    term: str,
    outcome_label: str,
    specification: str,
    sample: pd.DataFrame,
    region_id: int | None = None,
) -> dict[str, object]:
    interval = result.conf_int(level=0.95).loc[term]
    return {
        "Outcome": outcome_label,
        "Specification": specification,
        "Region ID": region_id,
        "NPP Coefficient": float(result.params[term]),
        "Clustered Standard Error": float(result.std_errors[term]),
        "95 Percent CI Lower": float(interval["lower"]),
        "95 Percent CI Upper": float(interval["upper"]),
        "Probability Value": float(result.pvalues[term]),
        "Percent Difference per 0.1 NPP": transformed_effect(float(result.params[term])),
        "Percent Difference 95 Percent CI Lower": transformed_effect(float(interval["lower"])),
        "Percent Difference 95 Percent CI Upper": transformed_effect(float(interval["upper"])),
        "Observations": len(sample),
        "Villages": sample[ID].nunique(),
        "Survey Weighted": True,
        "Village-Clustered Inference": True,
    }


def estimate_national_ladder(
    frame: pd.DataFrame,
    outcome: str,
    flag: str,
    outcome_label: str,
) -> pd.DataFrame:
    base_required = [ID, WEIGHT, "Exact Survey Time", NPP, outcome]
    specifications = [
        ("Survey-time controls", [], ["Exact Survey Time"]),
        ("+ Household composition (primary)", COMPOSITION_CONTROLS, ["Exact Survey Time"]),
        (
            "+ Socioeconomic controls",
            [*COMPOSITION_CONTROLS, *SOCIOECONOMIC_CONTROLS],
            ["Exact Survey Time"],
        ),
        ("+ Village fixed effects", COMPOSITION_CONTROLS, ["Exact Survey Time", ID]),
    ]
    rows: list[dict[str, object]] = []
    eligible = frame.loc[frame[flag]].copy()
    for specification, controls, absorb_columns in specifications:
        required = [*base_required, *controls, *absorb_columns]
        sample = eligible.dropna(subset=list(dict.fromkeys(required))).copy()
        exog = pd.DataFrame({NPP: sample[NPP]}, index=sample.index)
        for control in controls:
            exog[control] = sample[control]
        result = fit_weighted_absorbed(sample, outcome, exog, absorb_columns)
        row = coefficient_row(result, NPP, outcome_label, specification, sample)
        row["Household Composition Controls"] = bool(COMPOSITION_CONTROLS[0] in controls)
        row["Socioeconomic Controls"] = bool(SOCIOECONOMIC_CONTROLS[0] in controls)
        row["Village Fixed Effects"] = ID in absorb_columns
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_regional_slopes(
    frame: pd.DataFrame,
    outcome: str,
    flag: str,
    outcome_label: str,
    controls: list[str] | None = None,
    specification: str = "Frozen SKATER regional slope",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = COMPOSITION_CONTROLS if controls is None else controls
    required = [
        ID,
        REGION,
        WEIGHT,
        "Exact Survey Time",
        NPP,
        outcome,
        *controls,
    ]
    sample = frame.loc[frame[flag]].dropna(subset=required).copy()
    exog = pd.DataFrame(index=sample.index)
    terms: list[str] = []
    for region_id in range(1, 7):
        term = f"Prior-Year NPP - Region {region_id}"
        exog[term] = sample[NPP] * sample[REGION].eq(region_id)
        terms.append(term)
    for control in controls:
        exog[control] = sample[control]
    result = fit_weighted_absorbed(sample, outcome, exog, ["Exact Survey Time", REGION])
    rows = [
        coefficient_row(
            result,
            term,
            outcome_label,
            specification,
            sample,
            region_id,
        )
        for region_id, term in enumerate(terms, start=1)
    ]
    regressors = list(exog.columns)
    restriction = np.zeros((5, len(regressors)))
    for row, term in enumerate(terms[1:]):
        restriction[row, regressors.index(term)] = 1.0
        restriction[row, regressors.index(terms[0])] = -1.0
    test = result.wald_test(restriction=restriction, value=np.zeros(5))
    test_frame = pd.DataFrame(
        [
            {
                "Outcome": outcome_label,
                "Null Hypothesis": "All six regional NPP slopes are equal",
                "Wald Statistic": float(test.stat),
                "Degrees of Freedom": int(test.df),
                "Probability Value": float(test.pval),
                "Observations": len(sample),
                "Villages": sample[ID].nunique(),
            }
        ]
    )
    return pd.DataFrame(rows), test_frame


def style_forest(ax: plt.Axes) -> None:
    ax.axvline(0, color="#59656B", linewidth=0.9, zorder=1)
    ax.grid(axis="x", color=GRID_GRAY, linewidth=0.55, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#707A80")
    ax.tick_params(labelsize=8.1, colors=DARK_GRAY)
    ax.set_xlabel("Percent difference in consumption per 0.1 kg C m⁻² NPP", fontsize=8.8)


def draw_national_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    outcome_label: str,
    label: str,
    x_limits: tuple[float, float],
) -> None:
    display = frame.loc[frame["Outcome"].eq(outcome_label)].copy().iloc[::-1].reset_index(drop=True)
    y = np.arange(len(display))
    colors = ["#9AA5AA", "#7F8D94", NAVY, "#536A76"]
    for position, (_, row), color in zip(y, display.iterrows(), colors, strict=True):
        ax.hlines(
            position,
            row["Percent Difference 95 Percent CI Lower"],
            row["Percent Difference 95 Percent CI Upper"],
            color=color,
            linewidth=2.0,
            zorder=2,
        )
        ax.scatter(
            row["Percent Difference per 0.1 NPP"],
            position,
            s=42,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.set_yticks(y, display["Specification"])
    ax.set_ylim(-0.35, len(display) - 0.20)
    ax.set_xlim(*x_limits)
    ax.text(
        0.985,
        0.965,
        f"{outcome_label}: national ladder",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.3,
        color=NAVY,
        fontweight="bold",
    )
    style_forest(ax)
    panel_label(ax, label)


def draw_regional_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    tests: pd.DataFrame,
    outcome_label: str,
    label: str,
    x_limits: tuple[float, float],
) -> None:
    display = frame.loc[frame["Outcome"].eq(outcome_label)].sort_values("Region ID", ascending=False)
    y = np.arange(len(display))
    for position, (_, row) in zip(y, display.iterrows(), strict=True):
        region_id = int(row["Region ID"])
        ax.hlines(
            position,
            row["Percent Difference 95 Percent CI Lower"],
            row["Percent Difference 95 Percent CI Upper"],
            color=REGION_COLORS[region_id - 1],
            linewidth=2.0,
            zorder=2,
        )
        ax.scatter(
            row["Percent Difference per 0.1 NPP"],
            position,
            s=42,
            color=REGION_COLORS[region_id - 1],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.set_yticks(y, [f"Region {int(value)}" for value in display["Region ID"]])
    ax.set_ylim(-0.35, len(display) - 0.20)
    ax.set_xlim(*x_limits)
    p_value = float(tests.loc[tests["Outcome"].eq(outcome_label), "Probability Value"].iloc[0])
    p_text = "p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
    ax.text(
        0.985,
        0.965,
        f"{outcome_label}: regional slopes\njoint {p_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.3,
        color=NAVY,
        fontweight="bold",
    )
    style_forest(ax)
    panel_label(ax, label)


def main() -> None:
    frame = load_data()
    national = pd.concat(
        [
            estimate_national_ladder(frame, TOTAL_OUTCOME, TOTAL_FLAG, "Total consumption"),
            estimate_national_ladder(frame, FOOD_OUTCOME, FOOD_FLAG, "Food consumption"),
        ],
        ignore_index=True,
    )
    total_regional, total_test = estimate_regional_slopes(
        frame,
        TOTAL_OUTCOME,
        TOTAL_FLAG,
        "Total consumption",
        controls=[*COMPOSITION_CONTROLS, *SOCIOECONOMIC_CONTROLS],
        specification="Frozen SKATER regional slope + socioeconomic controls",
    )
    food_regional, food_test = estimate_regional_slopes(
        frame,
        FOOD_OUTCOME,
        FOOD_FLAG,
        "Food consumption",
        controls=[*COMPOSITION_CONTROLS, *SOCIOECONOMIC_CONTROLS],
        specification="Frozen SKATER regional slope + socioeconomic controls",
    )
    regional = pd.concat([total_regional, food_regional], ignore_index=True)
    tests = pd.concat([total_test, food_test], ignore_index=True)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    national.to_csv(EVIDENCE / "national_npp_to_consumption_regression_ladder.csv", index=False)
    regional.to_csv(EVIDENCE / "skater_regional_npp_to_consumption_coefficients.csv", index=False)
    tests.to_csv(EVIDENCE / "skater_regional_npp_slope_equality_tests.csv", index=False)

    all_limits = pd.concat(
        [
            national[["Percent Difference 95 Percent CI Lower", "Percent Difference 95 Percent CI Upper"]],
            regional[["Percent Difference 95 Percent CI Lower", "Percent Difference 95 Percent CI Upper"]],
        ]
    )
    lower = float(all_limits["Percent Difference 95 Percent CI Lower"].min())
    upper = float(all_limits["Percent Difference 95 Percent CI Upper"].max())
    padding = max(0.4, 0.06 * (upper - lower))
    x_limits = (lower - padding, upper + padding)

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.6), facecolor="white")
    fig.subplots_adjust(left=0.19, right=0.975, top=0.97, bottom=0.075, wspace=0.30, hspace=0.28)
    draw_national_panel(axes[0, 0], national, "Total consumption", "a", x_limits)
    draw_national_panel(axes[0, 1], national, "Food consumption", "b", x_limits)
    draw_regional_panel(axes[1, 0], regional, tests, "Total consumption", "c", x_limits)
    draw_regional_panel(axes[1, 1], regional, tests, "Food consumption", "d", x_limits)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved evidence: {EVIDENCE.relative_to(ROOT)}")
    print("\nNational ladder")
    print(
        national[
            [
                "Outcome",
                "Specification",
                "Percent Difference per 0.1 NPP",
                "Percent Difference 95 Percent CI Lower",
                "Percent Difference 95 Percent CI Upper",
                "Probability Value",
                "Observations",
                "Villages",
            ]
        ].to_string(index=False)
    )
    print("\nRegional equality tests")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
