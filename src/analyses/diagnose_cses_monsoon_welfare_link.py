#!/usr/bin/env python3
"""Diagnose whether the frozen monsoon family appears in CSES livelihood outcomes.

This is a prespecified human-link gate, not a promoted causal or mediation model.
It is run because the ecological first stage did not support constructing a
climate-implied NPP/EVI loss exposure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


INPUT = Path("data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet")
OUTPUT = Path("data/exp/analysis/climate-welfare/cses-direct-monsoon-diagnostic")
RAIN = "Last Complete Season May-October Precipitation Anomaly Z"
ONSET = "Last Complete Season Wet-Season Onset Anomaly Z Candidate B"
DRY = "Last Complete Season Longest Dry Spell Anomaly Z Candidate B"
HOT_DRY = "Last Complete Season Hot-Dry Day Count Anomaly Z Candidate B"
EXPOSURES = [RAIN, ONSET, DRY, HOT_DRY]
AG_PARTICIPATION = "Agricultural Participation"
CROP = "Real 2021 Crop Production Value per Cultivated ha Riels"
FOOD = "Real 2021 Food Consumption Value per Household Member Riels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def prepare(root: Path) -> tuple[pd.DataFrame, dict]:
    columns = [
        "Survey Year", "Interview Month", "Household ID", "Village Code", "District Code",
        "Climate Ecology Link Available", "Household Survey Weight", "Point Longitude",
        "Point Latitude", AG_PARTICIPATION, CROP, FOOD,
    ] + EXPOSURES
    frame = pd.read_parquet(root / INPUT, columns=columns)
    frame = frame.loc[frame["Climate Ecology Link Available"].eq(1)].copy()
    frame["Wave by Interview Month"] = (
        frame["Survey Year"].astype(str) + "_" + frame["Interview Month"].astype("Int64").astype(str)
    )
    frame["Spatial Block"] = (
        np.floor((frame["Point Longitude"] - 102.0) / 0.75).astype("Int64").astype(str)
        + "_"
        + np.floor((frame["Point Latitude"] - 10.0) / 0.75).astype("Int64").astype(str)
    )
    crop_positive = pd.to_numeric(frame[CROP], errors="coerce")
    crop_scale = float(crop_positive.loc[crop_positive.gt(0)].median())
    frame["Agricultural Participation Probability"] = frame[AG_PARTICIPATION].astype(float)
    frame["Asinh Real Crop Production per ha"] = np.arcsinh(crop_positive / crop_scale)
    frame["Log Real Food Consumption per Member"] = np.log(pd.to_numeric(frame[FOOD], errors="coerce"))
    transformations = {
        "Agricultural Participation Probability": "linear probability model in levels",
        "Asinh Real Crop Production per ha": (
            f"asinh(real 2021 riels per cultivated ha / {crop_scale:.6f}); scale is the positive "
            "linked-sample median frozen from the marginal outcome distribution"
        ),
        "Log Real Food Consumption per Member": "natural logarithm; all linked observations are positive",
        "outcome_exposure_relationship_read_for_transform_choice": False,
    }
    return frame, transformations


def fit(frame: pd.DataFrame, outcome: str, sample_role: str) -> tuple[object, list[dict], dict]:
    sample = frame.dropna(
        subset=[outcome, "District Code", "Wave by Interview Month", "Household Survey Weight",
                "Spatial Block"] + EXPOSURES
    ).copy()
    absorb = pd.DataFrame(
        {
            "District": sample["District Code"].astype("category"),
            "Wave by Interview Month": sample["Wave by Interview Month"].astype("category"),
        },
        index=sample.index,
    )
    result = AbsorbingLS(
        sample[outcome].astype(float),
        sample[EXPOSURES].astype(float),
        absorb=absorb,
        weights=sample["Household Survey Weight"].astype(float),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample["Spatial Block"]).codes,
        debiased=True,
    )
    interval = result.conf_int(level=0.95)
    rows = [
        {
            "Outcome": outcome,
            "Sample Role": sample_role,
            "Exposure": exposure,
            "Coefficient": float(result.params[exposure]),
            "Clustered Standard Error": float(result.std_errors[exposure]),
            "95 Percent CI Lower": float(interval.loc[exposure, "lower"]),
            "95 Percent CI Upper": float(interval.loc[exposure, "upper"]),
            "Probability Value": float(result.pvalues[exposure]),
            "Observations": int(result.nobs),
            "Villages": int(sample["Village Code"].nunique()),
            "District Fixed Effects": "yes",
            "Survey Wave by Interview Month Fixed Effects": "yes",
            "Survey Weights": "yes",
            "Cluster": "fixed 0.75-degree spatial block",
        }
        for exposure in EXPOSURES
    ]
    restriction = np.zeros((3, len(EXPOSURES)))
    for row, exposure in enumerate((ONSET, DRY, HOT_DRY)):
        restriction[row, EXPOSURES.index(exposure)] = 1
    wald = result.wald_test(restriction=restriction)
    summary = {
        "Outcome": outcome,
        "Sample Role": sample_role,
        "Observations": int(result.nobs),
        "Villages": int(sample["Village Code"].nunique()),
        "Districts": int(sample["District Code"].nunique()),
        "Spatial Blocks": int(sample["Spatial Block"].nunique()),
        "R2": float(result.rsquared),
        "R2 excluding absorbed effects": float(result.absorbed_rsquared),
        "Joint monsoon-structure Wald statistic": float(wald.stat),
        "Joint monsoon-structure Wald probability": float(wald.pval),
    }
    return result, rows, summary


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not (root / INPUT).exists():
        raise FileNotFoundError(root / INPUT)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    frame, transformations = prepare(root)
    (output / "outcome_transformations.json").write_text(
        json.dumps(transformations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rows: list[dict] = []
    summaries: list[dict] = []
    outcomes = (
        ("Agricultural Participation Probability", "all linked households"),
        ("Asinh Real Crop Production per ha", "linked households with valid cultivated-area intensity"),
        ("Log Real Food Consumption per Member", "all linked households"),
    )
    for outcome, role in outcomes:
        _, coefficient_rows, summary = fit(frame, outcome, role)
        rows.extend(coefficient_rows)
        summaries.append(summary)
    coefficients = pd.DataFrame(rows)
    model_summary = pd.DataFrame(summaries)
    coefficients.to_csv(output / "coefficients.csv", index=False)
    model_summary.to_csv(output / "model_summary.csv", index=False)

    dry_rows = coefficients.loc[coefficients["Exposure"].eq(DRY)].set_index("Outcome")
    crop_negative = float(dry_rows.loc["Asinh Real Crop Production per ha", "Coefficient"]) < 0
    food_negative = float(dry_rows.loc["Log Real Food Consumption per Member", "Coefficient"]) < 0
    crop_supported = (
        float(dry_rows.loc["Asinh Real Crop Production per ha", "95 Percent CI Upper"]) < 0
    )
    food_supported = (
        float(dry_rows.loc["Log Real Food Consumption per Member", "95 Percent CI Upper"]) < 0
    )
    gate = (
        "provisional human-link support"
        if crop_negative and food_negative and (crop_supported or food_supported)
        else "human-link gate not passed by the direct monsoon diagnostic"
    )
    result_summary = {
        "experiment": "Direct CSES validation of the frozen last-complete-season monsoon family",
        "status": "diagnostic because the ecological first-stage gate did not pass",
        "dry_spell_crop_direction_negative": crop_negative,
        "dry_spell_food_direction_negative": food_negative,
        "dry_spell_crop_95_percent_interval_below_zero": crop_supported,
        "dry_spell_food_95_percent_interval_below_zero": food_supported,
        "human_link_gate": gate,
        "promotion_rule": "negative crop and food directions, with at least one 95 percent interval below zero, followed by repeated-village and timing validation",
    }
    (output / "results_summary.json").write_text(
        json.dumps(result_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Direct CSES monsoon diagnostic\n\n"
        "This diagnostic asks whether the same frozen monsoon measures appear in CSES agricultural "
        "and food outcomes after the ecological first-stage gate failed. It is not labeled a "
        "mediation model. Models use survey weights, district fixed effects, survey-wave-by-month "
        "fixed effects, and 0.75-degree spatial-block clustered uncertainty.\n\n"
        f"Human-link gate: **{gate}**\n",
        encoding="utf-8",
    )
    print(json.dumps(result_summary, indent=2, ensure_ascii=False))
    print(coefficients.to_string(index=False))
    print(model_summary.to_string(index=False))


if __name__ == "__main__":
    main()
