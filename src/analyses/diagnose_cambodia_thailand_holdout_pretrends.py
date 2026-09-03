#!/usr/bin/env python3
"""Test pre-conflict trends using weights that exclude all outcome history."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from diagnose_cambodia_thailand_common_support import (
    CLIMATE,
    COVARIATES,
    EXPOSURE,
    FIXED_PREDICTORS,
    PANEL,
    PRE_CLIMATE_PREDICTORS,
    construct_pre_climate,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/exp/experiment-design/cambodia-thailand-common-support"
REFERENCE_YEAR = 2007
CANDIDATES = [
    {"radius": 10, "reservoir": "local_annulus", "control_lower": 20, "control_upper": 40},
    {"radius": 40, "reservoir": "frontier_60km", "control_lower": 60, "control_upper": None},
    {"radius": 60, "reservoir": "frontier_60km", "control_lower": 60, "control_upper": None},
]
OUTCOMES = {
    "Asinh LongNTL": {
        "column": "Asinh Annual NPP-VIIRS-like Radiance",
        "first_year": 2000,
    },
    "Annual NPP": {
        "column": "Annual Land NPP Mean kg C per m2",
        "first_year": 2001,
    },
}


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.sum(weights**2))


def build_design_features() -> pd.DataFrame:
    fixed = pd.read_parquet(COVARIATES)
    exposure = pd.read_parquet(
        EXPOSURE,
        columns=[
            "National Grid Cell ID",
            "Candidate All 2008-2011 Nearest Event Distance km",
            "Candidate Conflict Year 2011 Nearest Event Distance km",
            *[
                f"Candidate Conflict Year 2011 Exposure Within {candidate['radius']} km"
                for candidate in CANDIDATES
            ],
        ],
    )
    links = (
        pd.read_parquet(PANEL, columns=["National Grid Cell ID", "Year", "Climate Cell ID"])
        .loc[lambda frame: frame["Year"].eq(2000), ["National Grid Cell ID", "Climate Cell ID"]]
    )
    climate = construct_pre_climate()
    features = (
        fixed.merge(exposure, on="National Grid Cell ID", validate="one_to_one")
        .merge(links, on="National Grid Cell ID", validate="one_to_one")
        .merge(climate, on="Climate Cell ID", validate="many_to_one")
    )
    predictors = FIXED_PREDICTORS + PRE_CLIMATE_PREDICTORS
    return features.loc[~features[predictors].isna().any(axis=1)].copy()


def overlap_weights(features: pd.DataFrame, candidate: dict[str, object]) -> pd.DataFrame:
    radius = int(candidate["radius"])
    reservoir = str(candidate["reservoir"])
    treated = features[f"Candidate Conflict Year 2011 Exposure Within {radius} km"].astype(bool)
    if reservoir == "local_annulus":
        distance = features["Candidate Conflict Year 2011 Nearest Event Distance km"]
        control = distance.gt(float(candidate["control_lower"])) & distance.le(
            float(candidate["control_upper"])
        )
    else:
        control = features["Candidate All 2008-2011 Nearest Event Distance km"].gt(60) & features[
            "Within 60 km of Cambodia Thailand Border"
        ].astype(bool)
    sample = features.loc[treated | control].copy().reset_index(drop=True)
    treated = sample[f"Candidate Conflict Year 2011 Exposure Within {radius} km"].to_numpy(bool)
    predictors = FIXED_PREDICTORS + PRE_CLIMATE_PREDICTORS
    x = StandardScaler().fit_transform(sample[predictors])
    model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs").fit(x, treated.astype(int))
    propensity = np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    sample["Treated"] = treated
    sample["Propensity Score"] = propensity
    sample["Outcome Independent Overlap Weight"] = np.where(treated, 1 - propensity, propensity)
    sample["Treatment Radius km"] = radius
    sample["Control Reservoir"] = reservoir
    return sample


def estimate_pretrend(
    weights: pd.DataFrame, outcome_name: str, outcome_info: dict[str, object]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    radius = int(weights["Treatment Radius km"].iloc[0])
    reservoir = str(weights["Control Reservoir"].iloc[0])
    first_year = int(outcome_info["first_year"])
    outcome_column = str(outcome_info["column"])
    panel = pd.read_parquet(
        PANEL, columns=["National Grid Cell ID", "Year", outcome_column]
    ).loc[lambda frame: frame["Year"].between(first_year, REFERENCE_YEAR)]
    keep = weights[
        [
            "National Grid Cell ID",
            "Treated",
            "Outcome Independent Overlap Weight",
            "Grid Row",
            "Grid Column",
        ]
    ]
    frame = panel.merge(keep, on="National Grid Cell ID", validate="many_to_one").dropna(
        subset=[outcome_column]
    )
    years = [year for year in range(first_year, REFERENCE_YEAR) if year in frame["Year"].unique()]
    exog_columns = []
    for year in years:
        column = f"TreatedYear{year}"
        frame[column] = frame["Treated"].astype(float) * frame["Year"].eq(year).astype(float)
        exog_columns.append(column)
    frame["Spatial Block"] = (
        (frame["Grid Column"] // 10).astype(str)
        + "_"
        + (frame["Grid Row"] // 10).astype(str)
    )
    outcome_sd = float(frame[outcome_column].std(ddof=1))
    panel_frame = frame.set_index(["National Grid Cell ID", "Year"]).sort_index()
    exog = panel_frame[exog_columns].copy()
    exog.insert(0, "Constant", 1.0)
    model = PanelOLS(
        panel_frame[outcome_column],
        exog,
        weights=panel_frame["Outcome Independent Overlap Weight"],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False,
    )
    clusters = pd.DataFrame(
        {"Spatial Block": pd.Categorical(panel_frame["Spatial Block"]).codes},
        index=panel_frame.index,
    )
    result = model.fit(cov_type="clustered", clusters=clusters)
    restriction = np.zeros((len(exog_columns), len(result.params)), dtype=float)
    for row_index, parameter in enumerate(exog_columns):
        restriction[row_index, result.params.index.get_loc(parameter)] = 1.0
    joint = result.wald_test(restriction)
    rows = []
    for year, parameter in zip(years, exog_columns, strict=True):
        estimate = float(result.params[parameter])
        standard_error = float(result.std_errors[parameter])
        rows.append(
            {
                "Treatment Radius km": radius,
                "Control Reservoir": reservoir,
                "Outcome": outcome_name,
                "Year": year,
                "Reference Year": REFERENCE_YEAR,
                "Difference Relative to Reference": estimate,
                "Standard Error": standard_error,
                "Lower 95 CI": estimate - 1.96 * standard_error,
                "Upper 95 CI": estimate + 1.96 * standard_error,
                "Standardized Difference": estimate / outcome_sd,
                "Standardized Standard Error": standard_error / outcome_sd,
            }
        )
    summary = {
        "Treatment Radius km": radius,
        "Control Reservoir": reservoir,
        "Outcome": outcome_name,
        "Pretrend Years": f"{first_year}-{REFERENCE_YEAR}",
        "Joint Wald Statistic": float(joint.stat),
        "Joint Wald p-value": float(joint.pval),
        "Treated Cell ESS": effective_sample_size(
            weights.loc[weights["Treated"], "Outcome Independent Overlap Weight"].to_numpy()
        ),
        "Control Cell ESS": effective_sample_size(
            weights.loc[~weights["Treated"], "Outcome Independent Overlap Weight"].to_numpy()
        ),
        "Outcome SD": outcome_sd,
    }
    return rows, summary


def plot_coefficients(coefficients: pd.DataFrame) -> None:
    scenarios = [(10, "local_annulus"), (40, "frontier_60km"), (60, "frontier_60km")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=False)
    for column_index, (radius, reservoir) in enumerate(scenarios):
        for row_index, outcome in enumerate(OUTCOMES):
            axis = axes[row_index, column_index]
            part = coefficients[
                (coefficients["Treatment Radius km"] == radius)
                & (coefficients["Control Reservoir"] == reservoir)
                & (coefficients["Outcome"] == outcome)
            ].sort_values("Year")
            axis.errorbar(
                part["Year"],
                part["Standardized Difference"],
                yerr=1.96 * part["Standardized Standard Error"],
                fmt="o-",
                color="#2C6E9B",
                linewidth=1.4,
                markersize=4,
                capsize=2,
            )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.axvline(REFERENCE_YEAR, color="#777777", linewidth=0.8, linestyle="--")
            axis.set_title(f"{radius} km, {reservoir.replace('_', ' ')}")
            axis.set_ylabel(f"{outcome}\nstandardized gap vs 2007")
            axis.set_xlabel("Pre-conflict year")
            axis.grid(alpha=0.2)
    fig.suptitle("Outcome-independent weighted pre-conflict trend diagnostics", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "holdout_pretrend_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = build_design_features()
    all_weights = []
    coefficient_rows = []
    summaries = []
    for candidate in CANDIDATES:
        weights = overlap_weights(features, candidate)
        all_weights.append(weights)
        for outcome_name, outcome_info in OUTCOMES.items():
            rows, summary = estimate_pretrend(weights, outcome_name, outcome_info)
            coefficient_rows.extend(rows)
            summaries.append(summary)
    coefficients = pd.DataFrame(coefficient_rows)
    summary_frame = pd.DataFrame(summaries)
    weights_frame = pd.concat(all_weights, ignore_index=True)
    coefficients.to_csv(OUT_DIR / "holdout_pretrend_coefficients.csv", index=False)
    summary_frame.to_csv(OUT_DIR / "holdout_pretrend_joint_tests.csv", index=False)
    weights_frame[
        [
            "Treatment Radius km",
            "Control Reservoir",
            "National Grid Cell ID",
            "Treated",
            "Propensity Score",
            "Outcome Independent Overlap Weight",
        ]
    ].to_parquet(OUT_DIR / "holdout_pretrend_weights.parquet", index=False)
    plot_coefficients(coefficients)
    metadata = {
        "purpose": "holdout pretrend test independent of outcome-based matching",
        "weight_predictors": "fixed geography and 1991-2007 climate only",
        "outcome_test_periods": {name: f"{info['first_year']}-2007" for name, info in OUTCOMES.items()},
        "reference_year": REFERENCE_YEAR,
        "fixed_effects": "grid cell and calendar year",
        "inference": "10 km spatial-block clustered covariance",
        "post_2007_outcomes_loaded": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (OUT_DIR / "holdout_pretrend_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
