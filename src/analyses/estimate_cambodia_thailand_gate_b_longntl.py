#!/usr/bin/env python3
"""Estimate the frozen Gate B LongNTL distance-ring and continuous-dose models."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mj02-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

from diagnose_cambodia_thailand_common_support import PANEL
from freeze_cambodia_thailand_gate_b_protocol import OUT_DIR, RING_LABELS


WEIGHTS = OUT_DIR / "gate_b_binary_support_weights.parquet"
OUTCOME = "Asinh Annual NPP-VIIRS-like Radiance"
REFERENCE_PERIOD = "Late pre-period (2005-2007)"
PERIOD_ORDER = (
    "Early pre-period (2000-2004)",
    REFERENCE_PERIOD,
    "Precursor escalation (2008-2010)",
    "Conflict year (2011)",
    "Early recovery (2012-2014)",
    "Medium recovery (2015-2019)",
    "Long recovery (2020-2024)",
)
WEATHER_CONTROLS = (
    "Annual Precipitation Total mm Anomaly Z",
    "May October Maximum Five-Day Precipitation mm Anomaly Z",
    "May October Maximum Consecutive Dry Days Anomaly Z",
    "May October Maximum Daily Temperature C Anomaly Z",
)


def period_from_year(year: pd.Series) -> pd.Categorical:
    conditions = [
        year.between(2000, 2004),
        year.between(2005, 2007),
        year.between(2008, 2010),
        year.eq(2011),
        year.between(2012, 2014),
        year.between(2015, 2019),
        year.between(2020, 2024),
    ]
    values = np.select(conditions, PERIOD_ORDER, default="Outside scope")
    return pd.Categorical(values, categories=PERIOD_ORDER, ordered=True)


def load_panel() -> tuple[pd.DataFrame, float]:
    weights = pd.read_parquet(WEIGHTS)
    panel = pd.read_parquet(
        PANEL,
        columns=["National Grid Cell ID", "Year", OUTCOME, *WEATHER_CONTROLS],
    ).loc[lambda frame: frame["Year"].between(2000, 2024)]
    frame = panel.merge(weights, on="National Grid Cell ID", validate="many_to_one")
    frame = frame.dropna(subset=[OUTCOME, *WEATHER_CONTROLS]).copy()
    frame["Analysis Period"] = period_from_year(frame["Year"])
    frame["Sector Year"] = frame["Border Analysis Sector"] + "__" + frame["Year"].astype(str)
    frame["Continuous Conflict Dose"] = np.where(
        frame["Conflict Distance Ring"].eq("Over 60 km"),
        0.0,
        np.maximum(
            0.0,
            1 - frame["Candidate Conflict Year 2011 Nearest Event Distance km"] / 60,
        ),
    )
    pre_sd = float(frame.loc[frame["Year"].between(2000, 2007), OUTCOME].std(ddof=1))
    return frame, pre_sd


def fit_model(frame: pd.DataFrame, exog_columns: list[str]):
    panel_frame = frame.set_index(["National Grid Cell ID", "Year"]).sort_index()
    exog = panel_frame[[*exog_columns, *WEATHER_CONTROLS]].copy()
    exog.insert(0, "Constant", 1.0)
    other_effects = pd.DataFrame(
        {"Sector Year": pd.Categorical(panel_frame["Sector Year"]).codes},
        index=panel_frame.index,
    )
    clusters = pd.DataFrame(
        {"Spatial Block": pd.Categorical(panel_frame["Spatial Block ID"]).codes},
        index=panel_frame.index,
    )
    model = PanelOLS(
        panel_frame[OUTCOME],
        exog,
        weights=panel_frame["Gate B Binary Support Overlap Weight"],
        entity_effects=True,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=False,
    )
    return model.fit(cov_type="clustered", clusters=clusters)


def restriction_test(result, parameters: list[str]) -> tuple[float, float]:
    available = [parameter for parameter in parameters if parameter in result.params.index]
    restriction = np.zeros((len(available), len(result.params)))
    for row, parameter in enumerate(available):
        restriction[row, result.params.index.get_loc(parameter)] = 1
    test = result.wald_test(restriction)
    return float(test.stat), float(test.pval)


def estimate_ring_model(frame: pd.DataFrame, pre_sd: float):
    exog_columns = []
    mapping = []
    for ring in RING_LABELS[:-1]:
        ring_stem = ring.replace(" ", "").replace("-", "to")
        for period in PERIOD_ORDER:
            if period == REFERENCE_PERIOD:
                continue
            period_stem = PERIOD_ORDER.index(period)
            column = f"Ring{ring_stem}Period{period_stem}"
            frame[column] = (
                frame["Conflict Distance Ring"].eq(ring).astype(float)
                * frame["Analysis Period"].astype(str).eq(period).astype(float)
            )
            exog_columns.append(column)
            mapping.append((column, ring, period))
    result = fit_model(frame, exog_columns)
    rows = []
    for parameter, ring, period in mapping:
        estimate = float(result.params[parameter])
        standard_error = float(result.std_errors[parameter])
        rows.append(
            {
                "Distance Ring": ring,
                "Period": period,
                "Reference Period": REFERENCE_PERIOD,
                "Estimate": estimate,
                "Standard Error": standard_error,
                "Lower 95 CI": estimate - 1.96 * standard_error,
                "Upper 95 CI": estimate + 1.96 * standard_error,
                "Standardized Estimate": estimate / pre_sd,
                "Standardized Standard Error": standard_error / pre_sd,
                "Standardized Lower 95 CI": (estimate - 1.96 * standard_error) / pre_sd,
                "Standardized Upper 95 CI": (estimate + 1.96 * standard_error) / pre_sd,
            }
        )
    tests = []
    for period in PERIOD_ORDER:
        if period == REFERENCE_PERIOD:
            continue
        parameters = [parameter for parameter, _, candidate_period in mapping if candidate_period == period]
        statistic, pvalue = restriction_test(result, parameters)
        tests.append(
            {
                "Test": "All distance-ring coefficients equal zero",
                "Period": period,
                "Wald Statistic": statistic,
                "p-value": pvalue,
                "Restrictions": len(parameters),
            }
        )
    post_parameters = [
        parameter
        for parameter, _, period in mapping
        if period in PERIOD_ORDER[3:]
    ]
    statistic, pvalue = restriction_test(result, post_parameters)
    tests.append(
        {
            "Test": "All conflict and recovery ring coefficients equal zero",
            "Period": "2011-2024",
            "Wald Statistic": statistic,
            "p-value": pvalue,
            "Restrictions": len(post_parameters),
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(tests), result


def estimate_continuous_model(frame: pd.DataFrame, pre_sd: float):
    exog_columns = []
    mapping = []
    for period in PERIOD_ORDER:
        if period == REFERENCE_PERIOD:
            continue
        column = f"ContinuousDosePeriod{PERIOD_ORDER.index(period)}"
        frame[column] = (
            frame["Continuous Conflict Dose"]
            * frame["Analysis Period"].astype(str).eq(period).astype(float)
        )
        exog_columns.append(column)
        mapping.append((column, period))
    result = fit_model(frame, exog_columns)
    rows = []
    for parameter, period in mapping:
        estimate = float(result.params[parameter])
        standard_error = float(result.std_errors[parameter])
        rows.append(
            {
                "Period": period,
                "Reference Period": REFERENCE_PERIOD,
                "Dose Contrast": "event location versus 60 km or farther",
                "Estimate": estimate,
                "Standard Error": standard_error,
                "Lower 95 CI": estimate - 1.96 * standard_error,
                "Upper 95 CI": estimate + 1.96 * standard_error,
                "Standardized Estimate": estimate / pre_sd,
                "Standardized Standard Error": standard_error / pre_sd,
                "Standardized Lower 95 CI": (estimate - 1.96 * standard_error) / pre_sd,
                "Standardized Upper 95 CI": (estimate + 1.96 * standard_error) / pre_sd,
            }
        )
    return pd.DataFrame(rows), result


def plot_ring_results(coefficients: pd.DataFrame) -> None:
    periods = [period for period in PERIOD_ORDER if period != REFERENCE_PERIOD]
    x = np.arange(len(periods))
    colors = ["#8E3B46", "#D17C42", "#2C6E9B", "#5B8C5A"]
    fig, axis = plt.subplots(figsize=(12.5, 6.8))
    offsets = np.linspace(-0.27, 0.27, 4)
    for offset, color, ring in zip(offsets, colors, RING_LABELS[:-1], strict=True):
        part = coefficients.loc[coefficients["Distance Ring"].eq(ring)].set_index("Period").loc[periods]
        axis.errorbar(
            x + offset,
            part["Standardized Estimate"],
            yerr=1.96 * part["Standardized Standard Error"],
            fmt="o-",
            color=color,
            capsize=2,
            linewidth=1.3,
            label=ring,
        )
    axis.axhline(0, color="black", linewidth=0.9)
    axis.axvline(1.5, color="#777777", linestyle="--", linewidth=0.9)
    axis.set_xticks(x, [period.replace(" ", "\n", 1) for period in periods])
    axis.set_ylabel("LongNTL effect relative to 2005-2007 (pre-period SD)")
    axis.set_title("Cambodia-Thailand border conflict: direct activity and recovery path")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(title="Distance to 2011 event")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_b_longntl_ring_event_study.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    frame, pre_sd = load_panel()
    ring_coefficients, joint_tests, ring_result = estimate_ring_model(frame.copy(), pre_sd)
    continuous_coefficients, continuous_result = estimate_continuous_model(frame.copy(), pre_sd)
    ring_coefficients.to_csv(OUT_DIR / "gate_b_longntl_ring_coefficients.csv", index=False)
    joint_tests.to_csv(OUT_DIR / "gate_b_longntl_joint_tests.csv", index=False)
    continuous_coefficients.to_csv(OUT_DIR / "gate_b_longntl_continuous_dose_coefficients.csv", index=False)
    (OUT_DIR / "gate_b_longntl_ring_model_summary.txt").write_text(
        str(ring_result.summary) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "gate_b_longntl_continuous_model_summary.txt").write_text(
        str(continuous_result.summary) + "\n", encoding="utf-8"
    )
    plot_ring_results(ring_coefficients)
    metadata = {
        "status": "first frozen post-conflict Gate B estimate",
        "outcome": OUTCOME,
        "outcome_scale": "asinh annual LongNTL radiance",
        "standardization": "unweighted 2000-2007 outcome SD in the analysis sample",
        "pre_period_outcome_sd": pre_sd,
        "reference_period": REFERENCE_PERIOD,
        "periods": list(PERIOD_ORDER),
        "distance_rings": list(RING_LABELS),
        "reference_ring": "Over 60 km",
        "fixed_effects": "grid cell and one-degree border-sector by calendar year",
        "weather_controls": list(WEATHER_CONTROLS),
        "weights": "frozen outcome-independent binary frontier-support overlap weights",
        "inference": "10 km spatial-block clustered covariance; spatial-placebo inference is reported in a separate frozen artifact",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (OUT_DIR / "gate_b_longntl_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Ring coefficients")
    print(ring_coefficients[["Distance Ring", "Period", "Standardized Estimate", "Standardized Lower 95 CI", "Standardized Upper 95 CI"]].to_string(index=False))
    print("\nJoint tests")
    print(joint_tests.to_string(index=False))
    print("\nContinuous dose")
    print(continuous_coefficients[["Period", "Standardized Estimate", "Standardized Lower 95 CI", "Standardized Upper 95 CI"]].to_string(index=False))


if __name__ == "__main__":
    main()
