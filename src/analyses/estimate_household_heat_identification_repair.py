#!/usr/bin/env python3
"""Estimate household sample-structure and commune pseudo-panel heat models.

Plan: Supply the reopened household heat figure and regression tables with one
authoritative coefficient release.
Framework: AnaSOP Model B household, same-sample, and commune pseudo-panel
specifications with survey weights and spatial-block-clustered uncertainty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS

from validate_cses_hotdry_food_signal import (
    CSES,
    DRY,
    EXPOSURES,
    FOOD,
    HEAT,
    ID,
    ONSET,
    RAIN,
    YEAR,
    ecology_exposures,
    prepare_frame,
)


OUTPUT = Path("data/exp/analysis/climate-welfare/household-heat-identification-repair")
PSEUDO = Path(
    "data/processed/cses_commune_survey_time_climate_food_pseudopanel_preprocessed.parquet"
)

PSEUDO_OUTCOME = "Survey-Weighted Mean Log Real Food Consumption per Member"
PSEUDO_WEIGHT = "Survey Weight Sum"
PSEUDO_TIME = "Survey-Time Fixed-Effect ID"
PSEUDO_AREA = "Commune Code"
PSEUDO_CLUSTER = "Spatial Block ID"
PSEUDO_PRIMARY = "Primary Minimum Five Households"
PSEUDO_ROBUST = "Robustness Minimum Ten Households"
PSEUDO_REPEAT = "Repeated Commune Indicator"

PSEUDO_RENAME = {
    "Survey-Weighted Mean May-October Precipitation Anomaly Z": RAIN,
    "Survey-Weighted Mean Wet-Season Onset Anomaly Z": ONSET,
    "Survey-Weighted Mean Longest Intraseasonal Dry Spell Anomaly Z": DRY,
    "Survey-Weighted Mean Absolute Heat Days 35 C": HEAT,
}
PSEUDO_COMPOSITION = [
    "Survey-Weighted Mean Household Size",
    "Survey-Weighted Mean Female Household Member Share",
    "Survey-Weighted Mean Household Member Age Years",
    "Survey-Weighted Mean Child Age 0-14 Share",
    "Survey-Weighted Mean Older Age 65 Plus Share",
    "Survey-Weighted Mean Household Dependency Ratio",
    "Survey-Weighted Agricultural Participation Share",
    "Survey-Weighted Urban Household Share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def coefficient_rows(
    result: object,
    specification: str,
    evidence_level: str,
    observations: int,
    spatial_units: int,
    clusters: int,
    fixed_effects: str,
    weighting: str,
) -> list[dict[str, object]]:
    interval = result.conf_int(level=0.95)
    rows = []
    for exposure in EXPOSURES:
        rows.append(
            {
                "Specification": specification,
                "Evidence Level": evidence_level,
                "Exposure": exposure,
                "Coefficient Log Points": float(result.params[exposure]),
                "Percent Change for Exposure Unit": float(
                    (np.exp(result.params[exposure]) - 1.0) * 100.0
                ),
                "Exposure Unit": "10 days" if exposure == HEAT else "1 SD",
                "Clustered Standard Error": float(result.std_errors[exposure]),
                "95 Percent CI Lower": float(interval.loc[exposure, "lower"]),
                "95 Percent CI Upper": float(interval.loc[exposure, "upper"]),
                "Probability Value": float(result.pvalues[exposure]),
                "Observations": observations,
                "Spatial Units": spatial_units,
                "Spatial Blocks": clusters,
                "Fixed Effects": fixed_effects,
                "Weighting": weighting,
            }
        )
    return rows


def fit_micro(
    frame: pd.DataFrame,
    specification: str,
    area_column: str,
) -> list[dict[str, object]]:
    required = [
        "Log Real Food Consumption per Member",
        "Household Survey Weight",
        "Spatial Block",
        "Wave by Interview Month",
        area_column,
    ] + EXPOSURES
    sample = frame.dropna(subset=required).copy()
    absorb = pd.DataFrame(
        {
            "Area": sample[area_column].astype("category"),
            "Survey Time": sample["Wave by Interview Month"].astype("category"),
        },
        index=sample.index,
    )
    result = AbsorbingLS(
        sample["Log Real Food Consumption per Member"].astype(float),
        sample[EXPOSURES].astype(float),
        absorb=absorb,
        weights=sample["Household Survey Weight"].astype(float),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample["Spatial Block"]).codes,
        debiased=True,
    )
    return coefficient_rows(
        result=result,
        specification=specification,
        evidence_level="Household",
        observations=int(result.nobs),
        spatial_units=int(sample[area_column].nunique()),
        clusters=int(sample["Spatial Block"].nunique()),
        fixed_effects=f"{area_column} + survey-wave-by-interview-month",
        weighting="Household Survey Weight",
    )


def fit_repeat_interaction(frame: pd.DataFrame) -> dict[str, object]:
    sample = frame.dropna(
        subset=[
            "Log Real Food Consumption per Member",
            "Household Survey Weight",
            "Spatial Block",
            "Wave by Interview Month",
            "District Code",
            "Repeated Village Indicator",
        ]
        + EXPOSURES
    ).copy()
    sample["Heat X Repeated Village"] = sample[HEAT] * sample["Repeated Village Indicator"]
    regressors = EXPOSURES + ["Repeated Village Indicator", "Heat X Repeated Village"]
    absorb = pd.DataFrame(
        {
            "District": sample["District Code"].astype("category"),
            "Survey Time": sample["Wave by Interview Month"].astype("category"),
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
    term = "Heat X Repeated Village"
    return {
        "Specification": "Formal repeated-versus-single-wave heat contrast",
        "Evidence Level": "Household",
        "Exposure": term,
        "Coefficient Log Points": float(result.params[term]),
        "Percent Change for Exposure Unit": float((np.exp(result.params[term]) - 1.0) * 100.0),
        "Exposure Unit": "10 days difference",
        "Clustered Standard Error": float(result.std_errors[term]),
        "95 Percent CI Lower": float(interval.loc[term, "lower"]),
        "95 Percent CI Upper": float(interval.loc[term, "upper"]),
        "Probability Value": float(result.pvalues[term]),
        "Observations": int(result.nobs),
        "Spatial Units": int(sample["Village Code"].nunique()),
        "Spatial Blocks": int(sample["Spatial Block"].nunique()),
        "Fixed Effects": "District Code + survey-wave-by-interview-month",
        "Weighting": "Household Survey Weight",
    }


def fit_pseudo(
    panel: pd.DataFrame,
    specification: str,
    gate: str,
    composition: bool = False,
) -> list[dict[str, object]]:
    sample = panel.loc[panel[PSEUDO_REPEAT] & panel[gate]].copy()
    sample = sample.rename(columns=PSEUDO_RENAME)
    sample[HEAT] = sample[HEAT] / 10.0
    controls = PSEUDO_COMPOSITION if composition else []
    regressors = EXPOSURES + controls
    required = [PSEUDO_OUTCOME, PSEUDO_WEIGHT, PSEUDO_AREA, PSEUDO_TIME, PSEUDO_CLUSTER] + regressors
    sample = sample.dropna(subset=required).copy()
    absorb = pd.DataFrame(
        {
            "Commune": sample[PSEUDO_AREA].astype("category"),
            "Survey Time": sample[PSEUDO_TIME].astype("category"),
        },
        index=sample.index,
    )
    result = AbsorbingLS(
        sample[PSEUDO_OUTCOME].astype(float),
        sample[regressors].astype(float),
        absorb=absorb,
        weights=sample[PSEUDO_WEIGHT].astype(float),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample[PSEUDO_CLUSTER]).codes,
        debiased=True,
    )
    return coefficient_rows(
        result=result,
        specification=specification,
        evidence_level="Commune survey-time pseudo-panel",
        observations=int(result.nobs),
        spatial_units=int(sample[PSEUDO_AREA].nunique()),
        clusters=int(sample[PSEUDO_CLUSTER].nunique()),
        fixed_effects="Commune Code + survey-time",
        weighting=PSEUDO_WEIGHT,
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)

    cses_columns = [
        "Survey Year", "Interview Month", "Household ID", "Village Code", "Commune Code",
        "District Code", "Climate Ecology Link Available", "Household Survey Weight",
        "Point Longitude", "Point Latitude", ID, YEAR, FOOD,
    ]
    base = pd.read_parquet(root / CSES, columns=cses_columns)
    base = prepare_frame(base)
    base = base.loc[base["Climate Ecology Link Available"].eq(1)].copy()
    exposure = ecology_exposures(root, 5, "B", 35)
    primary = base.merge(exposure, left_on=[ID, YEAR], right_on=[ID, "Year"], how="left")

    wave_count = primary.groupby("Village Code", observed=True)["Survey Year"].nunique()
    repeated_ids = wave_count.loc[lambda x: x > 1].index
    primary["Repeated Village Indicator"] = primary["Village Code"].isin(repeated_ids).astype(float)
    repeated = primary.loc[primary["Repeated Village Indicator"].eq(1)].copy()
    single = primary.loc[primary["Repeated Village Indicator"].eq(0)].copy()

    rows: list[dict[str, object]] = []
    rows.extend(fit_micro(primary, "All linked households, district FE", "District Code"))
    rows.extend(fit_micro(primary, "All linked households, commune FE", "Commune Code"))
    rows.extend(fit_micro(repeated, "Repeated-village sample, district FE", "District Code"))
    rows.extend(fit_micro(repeated, "Repeated-village sample, village FE", "Village Code"))
    rows.extend(fit_micro(single, "Single-wave-village sample, district FE", "District Code"))
    rows.append(fit_repeat_interaction(primary))

    panel = pd.read_parquet(root / PSEUDO)
    rows.extend(
        fit_pseudo(
            panel,
            "Commune pseudo-panel, minimum 5 households",
            PSEUDO_PRIMARY,
        )
    )
    rows.extend(
        fit_pseudo(
            panel,
            "Commune pseudo-panel, minimum 10 households",
            PSEUDO_ROBUST,
        )
    )
    rows.extend(
        fit_pseudo(
            panel,
            "Commune pseudo-panel, minimum 5 households, composition adjusted",
            PSEUDO_PRIMARY,
            composition=True,
        )
    )

    coefficients = pd.DataFrame(rows)
    coefficients.to_csv(output / "coefficients.csv", index=False)
    heat = coefficients.loc[
        coefficients["Exposure"].isin([HEAT, "Heat X Repeated Village"])
    ].copy()
    heat.to_csv(output / "heat_identification_summary.csv", index=False)

    indexed = heat.set_index("Specification")
    full_commune = indexed.loc["All linked households, commune FE"]
    pseudo_primary = indexed.loc["Commune pseudo-panel, minimum 5 households"]
    pseudo_robust = indexed.loc["Commune pseudo-panel, minimum 10 households"]
    pass_rule = bool(
        full_commune["95 Percent CI Upper"] < 0
        and pseudo_primary["95 Percent CI Upper"] < 0
        and pseudo_robust["Coefficient Log Points"] < 0
    )
    summary = {
        "household_main_finding_gate": "pass" if pass_rule else "not passed",
        "gate_rule": (
            "negative full-sample commune-FE and primary commune-pseudo-panel intervals, "
            "plus a negative minimum-10-household pseudo-panel coefficient"
        ),
        "full_sample_commune_fe": full_commune.to_dict(),
        "primary_commune_pseudopanel": pseudo_primary.to_dict(),
        "minimum_10_household_pseudopanel": pseudo_robust.to_dict(),
        "same_sample_and_single_wave_results": heat.loc[
            heat["Specification"].str.contains("Repeated-village|Single-wave", regex=True)
        ].to_dict(orient="records"),
        "formal_repeat_status_contrast": indexed.loc[
            "Formal repeated-versus-single-wave heat contrast"
        ].to_dict(),
    }
    (output / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(heat.to_string(index=False))


if __name__ == "__main__":
    main()
