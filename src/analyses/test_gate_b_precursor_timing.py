#!/usr/bin/env python3
"""Test whether the Preah Vihear light signal appears in precursor clash years."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mj02-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from diagnose_cambodia_thailand_common_support import PANEL
from estimate_cambodia_thailand_gate_b_longntl import OUTCOME, WEATHER_CONTROLS
from freeze_cambodia_thailand_gate_b_protocol import OUT_DIR
from validate_gate_b_with_ccnl_dmsp import scenario_samples


YEARS = tuple(range(2000, 2015))
REFERENCE_YEARS = (2005, 2006, 2007)
EVENT_DEATHS = {2008: 2, 2009: 2, 2011: 11}
COEFFICIENTS = OUT_DIR / "gate_b_precursor_timing_coefficients.csv"
TESTS = OUT_DIR / "gate_b_precursor_timing_tests.csv"
FIGURE = OUT_DIR / "gate_b_precursor_timing.png"
METADATA = OUT_DIR / "gate_b_precursor_timing_metadata.json"
SUMMARIES = OUT_DIR / "gate_b_precursor_timing_model_summaries.txt"


def fit_model(frame: pd.DataFrame):
    model_frame = frame.set_index(["National Grid Cell ID", "Year"]).sort_index()
    years = model_frame.index.get_level_values("Year").to_numpy()
    parameters: dict[int, str] = {}
    for year in YEARS:
        if year in REFERENCE_YEARS:
            continue
        column = f"Dose x {year}"
        model_frame[column] = model_frame["Continuous Conflict Dose"] * (years == year)
        parameters[year] = column
    exog = model_frame[[*parameters.values(), *WEATHER_CONTROLS]].copy()
    exog.insert(0, "Constant", 1.0)
    other_effects = pd.DataFrame(
        {"Sector Year": pd.Categorical(model_frame["Sector Year"]).codes},
        index=model_frame.index,
    )
    clusters = pd.DataFrame(
        {"Spatial Block": pd.Categorical(model_frame["Spatial Block ID"]).codes},
        index=model_frame.index,
    )
    result = PanelOLS(
        model_frame[OUTCOME],
        exog,
        weights=model_frame["Gate B Binary Support Overlap Weight"],
        entity_effects=True,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=False,
    ).fit(cov_type="clustered", clusters=clusters)
    return result, parameters


def linear_test(result, weights: dict[str, float], label: str) -> dict[str, float | str]:
    restriction = np.zeros((1, len(result.params)))
    for parameter, weight in weights.items():
        restriction[0, result.params.index.get_loc(parameter)] = weight
    test = result.wald_test(restriction)
    estimate = float(sum(result.params[p] * w for p, w in weights.items()))
    covariance = result.cov.loc[list(weights), list(weights)].to_numpy()
    vector = np.array([weights[p] for p in weights])
    standard_error = float(np.sqrt(vector @ covariance @ vector))
    return {
        "Test": label,
        "Contrast Estimate": estimate,
        "Contrast Standard Error": standard_error,
        "Wald Statistic": float(test.stat),
        "p-value": float(test.pval),
    }


def plot_results(coefficients: pd.DataFrame) -> None:
    scenarios = ["All 2011 event sectors", "Preah Vihear only"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    for axis, scenario in zip(axes, scenarios, strict=True):
        part = coefficients.loc[coefficients["Scenario"].eq(scenario)].set_index("Year")
        plotted_years = [year for year in YEARS if year not in REFERENCE_YEARS]
        shown = part.loc[plotted_years]
        axis.errorbar(
            shown.index,
            shown["Standardized Estimate"],
            yerr=1.96 * shown["Standardized Standard Error"],
            fmt="o-",
            color="#2C6E9B",
            capsize=2.5,
            linewidth=1.2,
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axvspan(2004.5, 2007.5, color="#D9D9D9", alpha=0.55)
        for year in EVENT_DEATHS:
            axis.axvline(year, color="#B54A3A", alpha=0.45, linestyle="--", linewidth=1)
        axis.set_title(scenario)
        axis.set_xticks(range(2000, 2015, 2))
        axis.grid(axis="y", alpha=0.2)
        axis.set_xlabel("Year (red dashed lines mark verified clash years)")
    axes[0].set_ylabel("Dose effect relative to 2005–2007 (pre-period SD)")
    fig.suptitle("Did the nighttime-light decline already appear during the 2008–2009 precursor clashes?")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    panel = pd.read_parquet(
        PANEL,
        columns=["National Grid Cell ID", "Year", OUTCOME, *WEATHER_CONTROLS],
    )
    panel = panel.loc[panel["Year"].isin(YEARS)].copy()
    coefficient_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    summaries: list[str] = []

    for scenario, (sample, support) in scenario_samples().items():
        frame = panel.merge(
            sample[
                [
                    "National Grid Cell ID",
                    "Border Analysis Sector",
                    "Spatial Block ID",
                    "Gate B Binary Support Overlap Weight",
                    "Continuous Conflict Dose",
                ]
            ],
            on="National Grid Cell ID",
            validate="many_to_one",
        )
        frame["Sector Year"] = frame["Border Analysis Sector"] + "__" + frame["Year"].astype(str)
        frame = frame.dropna(subset=[OUTCOME, *WEATHER_CONTROLS]).copy()
        pre_sd = float(frame.loc[frame["Year"].between(2000, 2007), OUTCOME].std(ddof=1))
        result, parameters = fit_model(frame)
        summaries.append(f"\n{'=' * 88}\n{scenario}\n{'=' * 88}\n{result.summary}\n")
        for year, parameter in parameters.items():
            estimate = float(result.params[parameter])
            standard_error = float(result.std_errors[parameter])
            coefficient_rows.append(
                {
                    "Scenario": scenario,
                    "Year": year,
                    "Verified Cambodia-Thailand Clash Year": year in EVENT_DEATHS,
                    "UCDP Best Deaths in Preah Vihear Sector": EVENT_DEATHS.get(year, 0),
                    "Reference Period": "2005-2007",
                    "Estimate": estimate,
                    "Standard Error": standard_error,
                    "p-value": float(result.pvalues[parameter]),
                    "Pre-2008 Outcome SD": pre_sd,
                    "Standardized Estimate": estimate / pre_sd,
                    "Standardized Standard Error": standard_error / pre_sd,
                    "Standardized Lower 95 CI": (estimate - 1.96 * standard_error) / pre_sd,
                    "Standardized Upper 95 CI": (estimate + 1.96 * standard_error) / pre_sd,
                    "Grid Cells": frame["National Grid Cell ID"].nunique(),
                    "Treated 10 km Block ESS": support["treated_10km_block_ess"],
                    "Maximum Absolute Weighted SMD": support["maximum_absolute_weighted_smd"],
                }
            )
        tests = [
            linear_test(
                result,
                {parameters[2008]: 1.0, parameters[2009]: 1.0},
                "2008 and 2009 precursor coefficients jointly equal zero",
            ),
            linear_test(result, {parameters[2010]: 1.0}, "2010 no-clash coefficient equals zero"),
            linear_test(result, {parameters[2011]: 1.0}, "2011 conflict coefficient equals zero"),
            linear_test(
                result,
                {parameters[2011]: 1.0, parameters[2008]: -0.5, parameters[2009]: -0.5},
                "2011 coefficient equals mean 2008-2009 precursor coefficient",
            ),
        ]
        test_rows.extend({"Scenario": scenario, **row} for row in tests)

    coefficients = pd.DataFrame(coefficient_rows)
    tests = pd.DataFrame(test_rows)
    coefficients.to_csv(COEFFICIENTS, index=False)
    tests.to_csv(TESTS, index=False)
    SUMMARIES.write_text("".join(summaries), encoding="utf-8")
    plot_results(coefficients)
    METADATA.write_text(
        json.dumps(
            {
                "status": "candidate Gate B precursor-timing test",
                "years": list(YEARS),
                "reference_period": list(REFERENCE_YEARS),
                "verified_clash_years_and_ucdp_best_deaths_in_preah_vihear_sector": EVENT_DEATHS,
                "outcome": OUTCOME,
                "weights": "scenario-specific outcome-independent Gate B overlap weights",
                "fixed_effects": "national 1 km grid cell and one-degree border-sector by year",
                "weather_controls": list(WEATHER_CONTROLS),
                "inference": "10 km spatial-block clustered covariance",
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(coefficients.loc[coefficients["Year"].between(2008, 2013)].to_string(index=False))
    print("\nTiming tests")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
