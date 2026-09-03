#!/usr/bin/env python3
"""Estimate prespecified buffering heterogeneity in the absolute-heat food link."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from statsmodels.stats.multitest import multipletests


INPUT = Path("data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet")
OUTPUT = Path("data/exp/analysis/climate-welfare/absolute-heat-food-buffering")
FOOD = "Real 2021 Food Consumption Value per Household Member Riels"
RAIN = "Last Complete Season May-October Precipitation Anomaly Z"
ONSET = "Last Complete Season Wet-Season Onset Anomaly Z Candidate B"
DRY = "Last Complete Season Longest Dry Spell Anomaly Z Candidate B"
HEAT = "Last Complete Season Absolute Heat Day Count 35 C Candidate B"
CLIMATE = [RAIN, ONSET, DRY, HEAT]
BUFFERS = {
    "Irrigation": "Any Irrigable Parcel",
    "Historical Road Access": "Historical Road Distance km",
    "Baseline Settlement Connectivity": "Log Baseline Population 2000",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def heat_contrast(result: object, interaction: str, value: float) -> dict:
    coefficient = float(result.params[HEAT] + value * result.params[interaction])
    covariance = result.cov
    variance = float(
        covariance.loc[HEAT, HEAT]
        + value**2 * covariance.loc[interaction, interaction]
        + 2 * value * covariance.loc[HEAT, interaction]
    )
    standard_error = float(np.sqrt(max(variance, 0)))
    lower = coefficient - 1.96 * standard_error
    upper = coefficient + 1.96 * standard_error
    return {
        "Modifier Value": value,
        "Absolute Heat Effect Log Points": coefficient,
        "Absolute Heat Effect Percent": float((np.exp(coefficient) - 1) * 100),
        "95 Percent CI Lower Log Points": lower,
        "95 Percent CI Upper Log Points": upper,
        "95 Percent CI Lower Percent": float((np.exp(lower) - 1) * 100),
        "95 Percent CI Upper Percent": float((np.exp(upper) - 1) * 100),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not (root / INPUT).exists():
        raise FileNotFoundError(root / INPUT)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)

    columns = [
        "Survey Year", "Interview Month", "Household ID", "Village Code", "District Code",
        "Climate Ecology Link Available", "Household Survey Weight", "Point Longitude",
        "Point Latitude", FOOD,
    ] + CLIMATE + list(BUFFERS.values())
    frame = pd.read_parquet(root / INPUT, columns=columns)
    frame = frame.loc[frame["Climate Ecology Link Available"].eq(1)].copy()
    frame["Log Real Food Consumption per Member"] = np.log(
        pd.to_numeric(frame[FOOD], errors="coerce")
    )
    frame[HEAT] = pd.to_numeric(frame[HEAT], errors="coerce") / 10.0
    frame["Wave by Interview Month"] = (
        frame["Survey Year"].astype(str)
        + "_"
        + frame["Interview Month"].astype("Int64").astype(str)
    )
    frame["Spatial Block"] = (
        np.floor((frame["Point Longitude"] - 102.0) / 0.75).astype("Int64").astype(str)
        + "_"
        + np.floor((frame["Point Latitude"] - 10.0) / 0.75).astype("Int64").astype(str)
    )

    standardization: dict[str, dict] = {}
    for label, variable in BUFFERS.items():
        if label == "Irrigation":
            frame[f"{label} Modifier"] = pd.to_numeric(frame[variable], errors="coerce")
            standardization[label] = {
                "coding": "0 no observed irrigable parcel; 1 any irrigable parcel"
            }
        else:
            village_values = frame[["Village Code", variable]].drop_duplicates("Village Code")[variable]
            mean = float(village_values.mean())
            sd = float(village_values.std(ddof=1))
            frame[f"{label} Modifier"] = (
                pd.to_numeric(frame[variable], errors="coerce") - mean
            ) / sd
            standardization[label] = {
                "coding": "village-weighted z score",
                "mean_original_units": mean,
                "sd_original_units": sd,
            }

    interaction_rows: list[dict] = []
    marginal_rows: list[dict] = []
    for label in BUFFERS:
        modifier = f"{label} Modifier"
        interaction = f"Absolute Heat by {label}"
        frame[interaction] = frame[HEAT] * frame[modifier]
        regressors = CLIMATE + [modifier, interaction]
        sample = frame.dropna(
            subset=[
                "Log Real Food Consumption per Member", "Household Survey Weight",
                "District Code", "Wave by Interview Month", "Spatial Block",
            ] + regressors
        ).copy()
        absorb = pd.DataFrame(
            {
                "District": sample["District Code"].astype("category"),
                "Wave by Interview Month": sample["Wave by Interview Month"].astype("category"),
            },
            index=sample.index,
        )
        result = AbsorbingLS(
            sample["Log Real Food Consumption per Member"].astype(float),
            sample[regressors].astype(float),
            absorb=absorb,
            weights=sample["Household Survey Weight"].astype(float),
            drop_absorbed=True,
        ).fit(
            cov_type="clustered",
            clusters=pd.Categorical(sample["Spatial Block"]).codes,
            debiased=True,
        )
        interval = result.conf_int(level=0.95)
        expected = "positive" if label != "Historical Road Access" else "negative"
        interaction_rows.append(
            {
                "Buffer": label,
                "Interaction": interaction,
                "Coefficient Log Points": float(result.params[interaction]),
                "Clustered Standard Error": float(result.std_errors[interaction]),
                "95 Percent CI Lower": float(interval.loc[interaction, "lower"]),
                "95 Percent CI Upper": float(interval.loc[interaction, "upper"]),
                "Raw Probability Value": float(result.pvalues[interaction]),
                "Expected Protective Direction": expected,
                "Direction Consistent": bool(
                    result.params[interaction] > 0
                    if expected == "positive" else result.params[interaction] < 0
                ),
                "Observations": int(result.nobs),
                "Villages": int(sample["Village Code"].nunique()),
                "Spatial Blocks": int(sample["Spatial Block"].nunique()),
                "Modifier Mean": float(sample[modifier].mean()),
                "Modifier SD": float(sample[modifier].std(ddof=1)),
                "Modifier Minimum": float(sample[modifier].min()),
                "Modifier Maximum": float(sample[modifier].max()),
            }
        )
        if label == "Irrigation":
            values = [(0.0, "No irrigable parcel"), (1.0, "Any irrigable parcel")]
        else:
            unique_village = sample[["Village Code", modifier]].drop_duplicates("Village Code")
            values = [
                (float(unique_village[modifier].quantile(0.25)), "Village 25th percentile"),
                (float(unique_village[modifier].quantile(0.75)), "Village 75th percentile"),
            ]
        for value, value_label in values:
            row = heat_contrast(result, interaction, value)
            row.update(
                {
                    "Buffer": label,
                    "Modifier Level": value_label,
                    "Observations": int(result.nobs),
                    "Villages": int(sample["Village Code"].nunique()),
                }
            )
            marginal_rows.append(row)

    interactions = pd.DataFrame(interaction_rows)
    interactions["Holm Adjusted Probability Value"] = multipletests(
        interactions["Raw Probability Value"], method="holm"
    )[1]
    interactions["Holm Significant at 0.05"] = (
        interactions["Holm Adjusted Probability Value"] < 0.05
    )
    marginals = pd.DataFrame(marginal_rows)
    interactions.to_csv(output / "buffer_interactions.csv", index=False)
    marginals.to_csv(output / "buffer_marginal_effects.csv", index=False)
    (output / "buffer_standardization.json").write_text(
        json.dumps(standardization, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    supported = interactions.loc[
        interactions["Direction Consistent"]
        & interactions["Holm Significant at 0.05"],
        "Buffer",
    ].tolist()
    summary = {
        "experiment": "Prespecified buffering heterogeneity in absolute-heat food loss",
        "exposure": "10 additional post-onset days at or above 35 C, Candidate B, 5 km",
        "buffers_tested": list(BUFFERS),
        "fixed_effects": "district and survey-wave-by-interview-month",
        "weights": "CSES household survey weights",
        "inference": "0.75-degree spatial-block-clustered uncertainty",
        "multiplicity": "Holm correction across three interaction tests",
        "supported_protective_buffers": supported,
        "constructive_buffer_gate": "pass" if supported else "not passed",
        "interpretation": (
            "heterogeneity evidence only; interaction coefficients are not intervention effects"
        ),
    }
    (output / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Absolute-heat food-consumption buffering\n\n"
        "The exposure is 10 additional post-onset days at or above 35 C under Candidate B "
        "and a 5 km village buffer. The three frozen modifiers are any irrigable parcel, "
        "historical road distance, and baseline population connectivity. Interaction "
        "probabilities receive a Holm correction. Results describe differential sensitivity "
        "and are not intervention-effect estimates.\n\n"
        f"Constructive buffer gate: **{summary['constructive_buffer_gate']}**\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(interactions.to_string(index=False))
    print(marginals.to_string(index=False))


if __name__ == "__main__":
    main()
