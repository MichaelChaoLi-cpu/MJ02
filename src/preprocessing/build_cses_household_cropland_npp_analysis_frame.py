#!/usr/bin/env python3
"""Build the interview-year-aligned CSES household and cropland-NPP analysis frame."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


CONSUMPTION = Path("data/processed/cses_household_total_consumption_preprocessed.parquet")
WELFARE = Path("data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet")
MEMBERS = Path("data/processed/cses_household_members_preprocessed.parquet")
EDUCATION = Path("data/processed/direction3_education_core_outcomes_preprocessed.parquet")
NPP_BY_RADIUS = {
    2: Path("data/processed/cses_public_village_pixel_cropland_npp_annual_2km_preprocessed.parquet"),
    5: Path("data/processed/cses_public_village_pixel_cropland_npp_annual_preprocessed.parquet"),
    10: Path("data/processed/cses_public_village_pixel_cropland_npp_annual_10km_preprocessed.parquet"),
}
OUTPUT = Path("data/processed/cses_household_cropland_npp_analysis_preprocessed.parquet")
AUDIT = Path("data/exp/data-preprocessing/climate-npp-two-stage/stage2-household")

HOUSEHOLD_KEYS = ["Survey Wave", "Survey Year", "Household ID"]
COMPOSITION = [
    "Household Size",
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
]
SOCIOECONOMIC = [
    "Urban Rural",
    "Agricultural Participation",
    "Household Head Ever Attended School",
]


def numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def household_head_education(root: Path) -> pd.DataFrame:
    members = pd.read_parquet(
        root / MEMBERS,
        columns=HOUSEHOLD_KEYS
        + ["Person ID", "to the head", "Q01AC06 Relationship to the head"],
    )
    relationship = numeric(members["to the head"]).combine_first(
        numeric(members["Q01AC06 Relationship to the head"])
    )
    heads = members.loc[relationship.eq(1), HOUSEHOLD_KEYS + ["Person ID"]].copy()
    if heads.duplicated(HOUSEHOLD_KEYS).any():
        raise ValueError("More than one household head found for a household-wave")

    education = pd.read_parquet(
        root / EDUCATION,
        columns=HOUSEHOLD_KEYS + ["Person ID", "Ever Attended School"],
    )
    if education.duplicated(HOUSEHOLD_KEYS + ["Person ID"]).any():
        raise ValueError("Duplicate person education keys")
    heads = heads.merge(
        education,
        on=HOUSEHOLD_KEYS + ["Person ID"],
        how="left",
        validate="one_to_one",
    )
    return heads[HOUSEHOLD_KEYS + ["Ever Attended School"]].rename(
        columns={"Ever Attended School": "Household Head Ever Attended School"}
    )


def npp_columns(radius: int) -> dict[str, str]:
    suffix = "" if radius == 5 else f" at {radius} km"
    return {
        "Annual Strict-Cropland Mean NPP kg C per m2": (
            f"Prior-Year Strict-Cropland NPP{suffix}"
        ),
        "Annual Inclusive-Agriculture Mean NPP kg C per m2": (
            f"Prior-Year Inclusive-Agriculture NPP{suffix}"
        ),
        "Strict-Cropland Candidate 500m Pixel Count": (
            f"Prior-Year Strict-Cropland Candidate 500m Pixel Count{suffix}"
        ),
        "Strict-Cropland Valid NPP 500m Pixel Count": (
            f"Prior-Year Strict-Cropland Valid NPP 500m Pixel Count{suffix}"
        ),
        "Strict-Cropland Valid NPP Pixel Share": (
            f"Prior-Year Strict-Cropland Valid NPP Pixel Share{suffix}"
        ),
        "Mean Strict-Cropland Recoded NPP QC Filled Days Percent": (
            f"Prior-Year Strict-Cropland Mean Recoded NPP QC Filled Days Percent{suffix}"
        ),
        "Strict-Cropland Original NPP QC Above 100 Pixel Count": (
            f"Prior-Year Strict-Cropland Original NPP QC Above 100 Pixel Count{suffix}"
        ),
        "Inclusive-Agriculture Candidate 500m Pixel Count": (
            f"Prior-Year Inclusive-Agriculture Candidate 500m Pixel Count{suffix}"
        ),
        "Inclusive-Agriculture Valid NPP 500m Pixel Count": (
            f"Prior-Year Inclusive-Agriculture Valid NPP 500m Pixel Count{suffix}"
        ),
        "Inclusive-Agriculture Valid NPP Pixel Share": (
            f"Prior-Year Inclusive-Agriculture Valid NPP Pixel Share{suffix}"
        ),
        "Mean Inclusive-Agriculture Recoded NPP QC Filled Days Percent": (
            f"Prior-Year Inclusive-Agriculture Mean Recoded NPP QC Filled Days Percent{suffix}"
        ),
        "Inclusive-Agriculture Original NPP QC Above 100 Pixel Count": (
            f"Prior-Year Inclusive-Agriculture Original NPP QC Above 100 Pixel Count{suffix}"
        ),
    }


def read_npp(root: Path, radius: int) -> pd.DataFrame:
    renames = npp_columns(radius)
    source_columns = ["National Village Point ID", "Year", *renames]
    npp = pd.read_parquet(root / NPP_BY_RADIUS[radius], columns=source_columns)
    if npp.duplicated(["National Village Point ID", "Year"]).any():
        raise ValueError(f"Duplicate {radius} km village-year NPP keys")
    npp = npp.rename(columns=renames)
    npp["Prior NPP Calendar Year"] = numeric(npp.pop("Year")).astype("Int64")
    return npp


def build_frame(root: Path) -> pd.DataFrame:
    consumption = pd.read_parquet(root / CONSUMPTION)
    welfare_columns = HOUSEHOLD_KEYS + [
        "Interview Calendar Year",
        "Interview Month",
        "Interview Timing Recovered from Raw 2019",
        "Interview Timing Source Warning",
        "Public Village Point Matched",
        "National Village Point ID",
        "Public Village Name",
        "Point Longitude",
        "Point Latitude",
        "Members with Observed Age",
        "Female Household Member Share",
        "Mean Household Member Age Years",
        "Child Age 0-14 Share",
        "Working Age 15-64 Share",
        "Older Age 65 Plus Share",
        "Household Dependency Ratio",
        "Agricultural Participation",
    ]
    welfare = pd.read_parquet(root / WELFARE, columns=welfare_columns)
    if welfare.duplicated(HOUSEHOLD_KEYS).any():
        raise ValueError("Duplicate household-wave keys in the welfare frame")
    frame = consumption.merge(welfare, on=HOUSEHOLD_KEYS, how="left", validate="one_to_one")
    frame = frame.merge(
        household_head_education(root),
        on=HOUSEHOLD_KEYS,
        how="left",
        validate="one_to_one",
    )

    frame["Interview Calendar Year"] = numeric(frame["Interview Calendar Year"]).astype("Int64")
    frame["Prior NPP Calendar Year"] = frame["Interview Calendar Year"] - 1
    for radius in (2, 5, 10):
        frame = frame.merge(
            read_npp(root, radius),
            on=["National Village Point ID", "Prior NPP Calendar Year"],
            how="left",
            validate="many_to_one",
        )

    positive_weight = numeric(frame["Household Survey Weight"]).gt(0)
    primary_npp = frame["Prior-Year Strict-Cropland NPP"].notna()
    composition_complete = frame[COMPOSITION].notna().all(axis=1)
    socioeconomic_complete = frame[SOCIOECONOMIC].notna().all(axis=1)
    total_outcome = frame["Log Real 2021 Annual Total Consumption per Capita"].notna()
    food_outcome = frame["Log Real 2021 Annual Food Consumption per Capita"].notna()
    base_support = (
        frame["Main Linked Sample"].fillna(False)
        & frame["Survey Wave"].ne("2004")
        & frame["Interview Calendar Year"].notna()
        & frame["National Village Point ID"].notna()
        & primary_npp
        & positive_weight
        & composition_complete
    )
    frame["Stage 2 Total Consumption Complete Case"] = base_support & total_outcome
    frame["Stage 2 Food Consumption Complete Case"] = base_support & food_outcome
    frame["Stage 2 Common Outcome Complete Case"] = (
        base_support & total_outcome & food_outcome
    )
    frame["Stage 2 Expanded Socioeconomic Complete Case"] = (
        frame["Stage 2 Common Outcome Complete Case"] & socioeconomic_complete
    )

    if frame.duplicated(HOUSEHOLD_KEYS).any():
        raise ValueError("Duplicate household-wave keys in the stage-2 frame")
    expected_prior = frame["Interview Calendar Year"] - 1
    if not frame["Prior NPP Calendar Year"].equals(expected_prior):
        raise ValueError("Prior NPP year does not equal actual interview year minus one")
    if set(frame.loc[frame["Survey Wave"].eq("2019"), "Interview Calendar Year"].dropna()) != {
        2019,
        2020,
    }:
        raise ValueError("The corrected 2019 survey does not span 2019 and 2020")
    recoded_qc = [column for column in frame if "Mean Recoded NPP QC" in column]
    if frame[recoded_qc].max().max() > 100:
        raise ValueError("An analysis-facing recoded NPP QC value remains above 100")
    return frame.sort_values(["Survey Year", "PSU", "Household ID"]).reset_index(drop=True)


def write_audit(frame: pd.DataFrame, audit: Path) -> dict[str, object]:
    audit.mkdir(parents=True, exist_ok=True)
    variable_rows: list[dict[str, object]] = []
    for column in frame.columns:
        samples = frame[column].dropna().astype(str).drop_duplicates().head(3).tolist()
        variable_rows.append(
            {
                "original_name": column,
                "readable_name": column,
                "dtype": str(frame[column].dtype),
                "null_pct": float(frame[column].isna().mean() * 100),
                "feasibility_status": "unknown",
                "sample_values": json.dumps(samples, ensure_ascii=True),
            }
        )
    pd.DataFrame(variable_rows).to_csv(audit / "variable_list.csv", index=False)

    coverage = frame.groupby(
        ["Survey Wave", "Interview Calendar Year"], dropna=False, observed=True
    ).agg(
        **{
            "Households": ("Household ID", "size"),
            "Original Total-Consumption Eligible": (
                "Total Consumption Main Outcome Eligible",
                "sum",
            ),
            "Public Village Point Matched": ("National Village Point ID", "count"),
            "Prior-Year Strict NPP 2 km Linked": (
                "Prior-Year Strict-Cropland NPP at 2 km",
                "count",
            ),
            "Prior-Year Strict NPP 5 km Linked": ("Prior-Year Strict-Cropland NPP", "count"),
            "Prior-Year Strict NPP 10 km Linked": (
                "Prior-Year Strict-Cropland NPP at 10 km",
                "count",
            ),
            "Stage 2 Total Complete Cases": (
                "Stage 2 Total Consumption Complete Case",
                "sum",
            ),
            "Stage 2 Food Complete Cases": (
                "Stage 2 Food Consumption Complete Case",
                "sum",
            ),
            "Stage 2 Expanded Complete Cases": (
                "Stage 2 Expanded Socioeconomic Complete Case",
                "sum",
            ),
        }
    ).reset_index()
    coverage.to_csv(audit / "stage2_household_coverage_by_wave.csv", index=False)

    linkage_rows: list[dict[str, object]] = []
    eligible = frame["Total Consumption Main Outcome Eligible"].fillna(False)
    for radius in (2, 5, 10):
        suffix = "" if radius == 5 else f" at {radius} km"
        for definition in ("Strict-Cropland", "Inclusive-Agriculture"):
            value = f"Prior-Year {definition} NPP{suffix}"
            linked = eligible & frame[value].notna()
            linkage_rows.append(
                {
                    "Buffer Radius km": radius,
                    "Land-Cover Definition": definition,
                    "Eligible Households": int(eligible.sum()),
                    "Households Linked": int(linked.sum()),
                    "Linked Share": float(linked.sum() / eligible.sum()),
                    "Linked Village Points": int(
                        frame.loc[linked, "National Village Point ID"].nunique()
                    ),
                }
            )
    linkage = pd.DataFrame(linkage_rows)
    linkage.to_csv(audit / "prior_year_npp_linkage_by_radius.csv", index=False)

    dictionary = pd.DataFrame(
        [
            ["Interview Calendar Year", "timing", "year", "Actual interview calendar year, including corrected 2019 survey fieldwork in 2020"],
            ["Prior NPP Calendar Year", "timing", "year", "Interview Calendar Year minus one"],
            ["Prior-Year Strict-Cropland NPP", "primary exposure", "kg C/m2/year", "Prior NPP Calendar Year mean over same-year IGBP class 12 pixels within 5 km"],
            ["Prior-Year Inclusive-Agriculture NPP", "sensitivity exposure", "kg C/m2/year", "Prior NPP Calendar Year mean over same-year IGBP classes 12 and 14 within 5 km"],
            ["Prior-Year Strict-Cropland NPP at 2 km", "scale sensitivity", "kg C/m2/year", "Strict-cropland NPP constructed within 2 km"],
            ["Prior-Year Strict-Cropland NPP at 10 km", "scale sensitivity", "kg C/m2/year", "Strict-cropland NPP constructed within 10 km"],
            ["Household Head Ever Attended School", "socioeconomic control", "binary", "Education response for the unique household member coded as household head"],
            ["Stage 2 Total Consumption Complete Case", "sample flag", "binary", "Positive survey weight, mapped prior-year 5 km strict NPP, complete household composition, and observed log total consumption"],
        ],
        columns=["Variable", "Role", "Unit", "Construction"],
    )
    dictionary.to_csv(audit / "stage2_variable_dictionary.csv", index=False)

    metadata = {
        "output": str(OUTPUT),
        "grain": "one CSES household per survey wave",
        "rows": int(len(frame)),
        "survey_waves": sorted(frame["Survey Wave"].dropna().astype(str).unique().tolist()),
        "main_outcome_eligible_households": int(
            frame["Total Consumption Main Outcome Eligible"].sum()
        ),
        "stage2_total_complete_cases": int(
            frame["Stage 2 Total Consumption Complete Case"].sum()
        ),
        "stage2_food_complete_cases": int(
            frame["Stage 2 Food Consumption Complete Case"].sum()
        ),
        "stage2_expanded_complete_cases": int(
            frame["Stage 2 Expanded Socioeconomic Complete Case"].sum()
        ),
        "timing_rule": "actual Interview Calendar Year minus one",
        "primary_buffer_km": 5,
        "sensitivity_buffers_km": [2, 10],
        "strict_land_cover": "same-year IGBP class 12 Croplands",
        "inclusive_land_cover": "same-year IGBP classes 12 and 14",
        "missing_data": "no imputation",
        "outlier_treatment": "no winsorization or clipping",
        "standardization": "none",
        "processed_at_utc": datetime.now(UTC).isoformat(),
    }
    (audit / "stage2_household_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    frame = build_frame(root)
    destination = root / OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False, compression="zstd")
    metadata = write_audit(frame, root / AUDIT)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(
        pd.read_csv(root / AUDIT / "prior_year_npp_linkage_by_radius.csv").to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
