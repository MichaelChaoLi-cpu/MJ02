#!/usr/bin/env python3
"""Build CSES crop outcomes aligned to the latest complete production year.

The legacy household outcome file aggregates crop records within survey wave.
This repair uses the crop module's explicit reference year and the household's
actual interview calendar year.  A production year is complete only when it is
strictly earlier than the interview year.  The latest complete year within the
preceding three calendar years is retained for each household.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CROPS = ROOT / "data/processed/cses_agriculture_crop_production_preprocessed.parquet"
HOUSEHOLDS = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
CORE = ROOT / "data/processed/direction3_household_core_outcomes_preprocessed.parquet"
OUTPUT = ROOT / "data/processed/cses_household_reference_year_crop_production_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/reference-year-crop-production"

KEYS = ["Survey Wave", "Household ID"]
PRODUCTION = "Q05BC06 How much was produced harvested KG"
PRICE = "Q05BC09 What was the sale price of the crop produced per kg RIELS Kg"
CULTIVATED = "Q05BC04 How big area was cultivated m2"
HARVESTED = "Q05BC05 How big area was harvested m2"
CROP_CODE = "Q05BC03B What crop s have yourhousehold grown on what parcels"
PAST_YEAR = "Past Year What year"
SEASON_YEAR = "Q05BYEAR Q05B Season year"


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def sum_min_one(values: pd.Series) -> float:
    return values.sum(min_count=1)


def main() -> None:
    for path in (CROPS, HOUSEHOLDS, CORE):
        if not path.exists():
            raise FileNotFoundError(path)

    household_columns = [
        *KEYS,
        "Survey Year",
        "Interview Calendar Year",
        "Interview Month",
        "Household Survey Weight",
        "Province Code",
        "District Code",
        "Commune Code",
        "Village Code",
        "National Village Point ID",
        "Point Longitude",
        "Point Latitude",
        "Household Size",
        "Female Household Member Share",
        "Mean Household Member Age Years",
        "Child Age 0-14 Share",
        "Older Age 65 Plus Share",
        "Household Dependency Ratio",
        "Real 2021 Food Consumption Value per Household Member Riels",
        "Real 2021 Own Produced Food Value per Household Member Riels",
    ]
    households = pd.read_parquet(HOUSEHOLDS, columns=household_columns)
    households = households.drop_duplicates(KEYS)
    deflator = pd.read_parquet(
        CORE, columns=[*KEYS, "All Items CPI Deflator to 2021"]
    ).drop_duplicates(KEYS)
    households = households.merge(deflator, on=KEYS, how="left", validate="one_to_one")

    crops = pd.read_parquet(CROPS)
    crops = crops.loc[crops["Survey Wave"].ne("2004")].copy()
    crops = crops.merge(
        households[KEYS + ["Interview Calendar Year"]],
        on=KEYS,
        how="inner",
        validate="many_to_one",
    )
    crops["Crop Production Reference Year"] = numeric(
        crops, SEASON_YEAR
    ).combine_first(numeric(crops, PAST_YEAR))
    reference_year = crops["Crop Production Reference Year"]
    interview_year = numeric(crops, "Interview Calendar Year")
    plausible = reference_year.between(2001, 2021)
    complete = plausible & reference_year.lt(interview_year)
    recent = complete & reference_year.ge(interview_year - 3)
    crops["Reference Year Timing Status"] = np.select(
        [
            recent,
            complete,
            plausible & reference_year.eq(interview_year),
            plausible & reference_year.gt(interview_year),
        ],
        [
            "complete_recent",
            "complete_more_than_three_years_before_interview",
            "same_calendar_year_incomplete",
            "future_or_error",
        ],
        default="missing_or_invalid",
    )

    eligible = crops.loc[recent].copy()
    eligible["Crop Production Reference Year"] = eligible[
        "Crop Production Reference Year"
    ].astype(int)
    latest = eligible.groupby(KEYS, observed=True)[
        "Crop Production Reference Year"
    ].transform("max")
    eligible = eligible.loc[
        eligible["Crop Production Reference Year"].eq(latest)
    ].copy()
    eligible["Crop Production Quantity kg"] = numeric(eligible, PRODUCTION)
    eligible["Cultivated Crop Area m2"] = numeric(eligible, CULTIVATED)
    eligible["Harvested Crop Area m2"] = numeric(eligible, HARVESTED)
    eligible["Nominal Crop Production Value Riels"] = (
        numeric(eligible, PRODUCTION) * numeric(eligible, PRICE)
    )
    eligible["Crop Code"] = eligible[CROP_CODE].astype("string").replace(
        {"": pd.NA, "nan": pd.NA}
    )

    group_keys = KEYS + ["Crop Production Reference Year"]
    outcomes = eligible.groupby(group_keys, observed=True).agg(
        **{
            "Crop Production Quantity kg": (
                "Crop Production Quantity kg",
                sum_min_one,
            ),
            "Cultivated Crop Area m2": ("Cultivated Crop Area m2", sum_min_one),
            "Harvested Crop Area m2": ("Harvested Crop Area m2", sum_min_one),
            "Nominal Crop Production Value Riels": (
                "Nominal Crop Production Value Riels",
                sum_min_one,
            ),
            "Crop Production Observation Count": (
                "Crop Production Quantity kg",
                "count",
            ),
            "Crop Diversity Count": ("Crop Code", "nunique"),
        }
    ).reset_index()
    output = households.merge(outcomes, on=KEYS, how="inner", validate="one_to_one")
    output["Production Year Lag from Interview"] = (
        output["Interview Calendar Year"] - output["Crop Production Reference Year"]
    )
    output["Crop Yield kg per ha"] = (
        output["Crop Production Quantity kg"] * 10_000
        / output["Harvested Crop Area m2"].replace(0, np.nan)
    )
    output["Real 2021 Crop Production Value Riels"] = (
        output["Nominal Crop Production Value Riels"]
        * output["All Items CPI Deflator to 2021"]
    )
    output["Real 2021 Crop Production Value per Cultivated ha Riels"] = (
        output["Real 2021 Crop Production Value Riels"] * 10_000
        / output["Cultivated Crop Area m2"].replace(0, np.nan)
    )
    output = output.sort_values(
        ["Crop Production Reference Year", "Survey Wave", "Household ID"]
    ).reset_index(drop=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    timing = (
        crops.groupby(["Survey Wave", "Reference Year Timing Status"], observed=True)
        .size()
        .rename("Crop Records")
        .reset_index()
    )
    coverage = (
        output.groupby(
            ["Survey Wave", "Crop Production Reference Year"], observed=True
        )
        .agg(
            Households=("Household ID", "nunique"),
            Villages=("Village Code", "nunique"),
            Communes=("Commune Code", "nunique"),
        )
        .reset_index()
    )
    timing.to_csv(AUDIT_DIR / "reference_year_timing_audit.csv", index=False)
    coverage.to_csv(AUDIT_DIR / "aligned_household_coverage.csv", index=False)
    summary = {
        "construction": (
            "Latest explicit crop-production reference year strictly before actual "
            "interview calendar year and no more than three years earlier"
        ),
        "same_calendar_year_records_excluded": True,
        "missing_reference_year_records_excluded": True,
        "households": int(len(output)),
        "villages": int(output["Village Code"].nunique()),
        "survey_waves": sorted(output["Survey Wave"].astype(str).unique().tolist()),
        "production_years": sorted(
            output["Crop Production Reference Year"].astype(int).unique().tolist()
        ),
    }
    (AUDIT_DIR / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
