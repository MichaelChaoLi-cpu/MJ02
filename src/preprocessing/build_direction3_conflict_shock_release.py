"""Build compact Direction 3 household and education analysis releases.

The releases combine current CSES outcomes with place-based historical conflict,
    long-baseline rainfall shocks, local food-price shocks, and satellite-observed
    inundation. Obsolete landmine variables and superseded short-baseline climate
    variables are intentionally excluded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEYS = ["Survey Year", "Survey Wave", "PSU"]

HOUSEHOLD_COLUMNS = [
    "Survey Year",
    "Survey Wave",
    "Household ID",
    "PSU",
    "Province Code Component",
    "District Code Component",
    "Commune Code Component",
    "Village Code Component",
    "Province Name",
    "District Name",
    "Commune Name",
    "Village Name",
    "Urban Rural",
    "Survey Month",
    "Household Size",
    "Children Age 6 to 17 Count",
    "Household Survey Weight",
    "Total Parcel Area m2",
    "Parcel Area Observation Count",
    "Irrigable Parcel Count",
    "Irrigation Status Observation Count",
    "Irrigable Parcel Share",
    "Any Irrigable Parcel",
    "Cultivated Crop Area m2",
    "Harvested Crop Area m2",
    "Crop Production Quantity kg",
    "Post Harvest Crop Loss kg",
    "Crop Rent Quantity kg",
    "Crop Production Observation Count",
    "Crop Diversity Count",
    "Crop Yield kg per ha",
    "Post Harvest Loss Share",
    "Any Crop Production Record",
    "Agricultural Cost Observation Count",
    "Food Item Observation Count",
    "Food Items with Positive Consumption Count",
    "No Food Experience",
    "Went to Sleep Hungry",
    "Went Whole Day Without Eating",
    "Any Severe Food Insecurity Experience",
    "Food Insecurity Severity Sum",
    "Agricultural Household",
    "Real 2021 Crop Production Value Riels",
    "Real 2021 Agricultural Input Cost Riels",
    "Real 2021 Agricultural Input Cost per Cultivated ha Riels",
    "Real 2021 Reported Food Consumption Value Riels",
    "Real 2021 Purchased Food Consumption Value Riels",
    "Real 2021 Own Produced Food Consumption Value Riels",
    "Real 2021 Food Consumption Value per Household Member Riels",
]

EDUCATION_COLUMNS = [
    "Survey Year",
    "Survey Wave",
    "Household ID",
    "Person ID",
    "PSU",
    "Province Code",
    "District Code",
    "Commune Code",
    "Village Code",
    "Urban Rural",
    "Age Years",
    "Female",
    "Ever Attended School",
    "Currently Attending School",
    "Years Attended School",
    "School Age 6 to 17",
    "School Attendance Outcome Eligible",
    "Person Survey Weight",
    "Household Size",
    "Household Survey Weight",
    "Real 2021 Education Expenditure Riels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args()


def assert_unique(data: pd.DataFrame, keys: list[str], label: str) -> None:
    duplicate_count = int(data.duplicated(keys).sum())
    if duplicate_count:
        raise ValueError(f"{label} contains {duplicate_count:,} duplicate key rows")


def prepare_exposures(processed: Path) -> pd.DataFrame:
    conflict = pd.read_parquet(
        processed / "direction3_psu_conflict_exposure_preprocessed.parquet"
    )
    climate = pd.read_parquet(
        processed / "direction3_psu_climate_shocks_preprocessed.parquet"
    )
    prices = pd.read_parquet(
        processed / "direction3_psu_food_price_shocks_preprocessed.parquet"
    )
    floods = pd.read_parquet(
        processed / "direction3_psu_satellite_flood_shocks_preprocessed.parquet"
    )
    for label, data in [
        ("conflict", conflict),
        ("climate", climate),
        ("prices", prices),
        ("floods", floods),
    ]:
        assert_unique(data, KEYS, label)

    climate = climate.drop(
        columns=[
            "Climate Geography Resolution",
            "Climate Geography Code",
            "Climate Link Method",
        ],
        errors="ignore",
    )
    prices = prices.drop(
        columns=[
            "Survey Month Numeric",
            "Matched Climate Province Name",
        ],
        errors="ignore",
    )
    floods = floods.drop(
        columns=[
            "Climate Geography Resolution",
            "Climate Geography Code",
            "Climate Link Method",
            "Survey Month Numeric",
        ],
        errors="ignore",
    )
    exposures = conflict.merge(climate, on=KEYS, how="left", validate="one_to_one")
    exposures = exposures.merge(prices, on=KEYS, how="left", validate="one_to_one")
    exposures = exposures.merge(floods, on=KEYS, how="left", validate="one_to_one")
    assert_unique(exposures, KEYS, "combined exposures")
    return exposures


def select_main_sample(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = sorted(set(columns) - set(data.columns))
    if missing:
        raise KeyError(f"Missing confirmed columns: {missing}")
    return data.loc[data["Main Linked Sample"].fillna(False), columns].copy()


def merge_release(base: pd.DataFrame, exposures: pd.DataFrame, label: str) -> pd.DataFrame:
    before = len(base)
    output = base.merge(exposures, on=KEYS, how="left", validate="many_to_one")
    if len(output) != before:
        raise ValueError(f"{label} merge changed row count from {before:,} to {len(output):,}")
    if output["Historical Conflict Link Matched"].fillna(False).sum() != len(output):
        raise ValueError(f"{label} has unmatched historical-conflict rows")
    if output["Long Baseline Climate Link Matched"].fillna(False).sum() != len(output):
        raise ValueError(f"{label} has unmatched annual-climate rows")
    mine_columns = [column for column in output if "mine" in column.casefold()]
    if mine_columns:
        raise ValueError(f"{label} unexpectedly contains landmine columns: {mine_columns}")
    return output


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    processed = root / "data/processed"
    diagnostics = root / "data/exp/data-preprocessing"
    exposures = prepare_exposures(processed)

    households_source = pd.read_parquet(
        processed / "direction3_household_core_outcomes_preprocessed.parquet"
    )
    education_source = pd.read_parquet(
        processed / "direction3_education_core_outcomes_preprocessed.parquet"
    )
    households = merge_release(
        select_main_sample(households_source, HOUSEHOLD_COLUMNS),
        exposures,
        "household release",
    )
    if households["Agricultural Household"].isna().any():
        raise ValueError("Household release contains missing Agricultural Household values")
    if not households["Agricultural Household"].isin([False, True]).all():
        raise ValueError("Household release contains non-binary Agricultural Household values")
    education = merge_release(
        select_main_sample(education_source, EDUCATION_COLUMNS),
        exposures,
        "education release",
    )

    household_path = processed / "direction3_household_conflict_shock_preprocessed.parquet"
    education_path = processed / "direction3_education_conflict_shock_preprocessed.parquet"
    households.to_parquet(household_path, index=False)
    education.to_parquet(education_path, index=False)

    validation = pd.DataFrame(
        [
            {
                "Dataset": "household",
                "Rows": len(households),
                "Columns": households.shape[1],
                "Survey Years": ", ".join(map(str, sorted(households["Survey Year"].unique()))),
                "PSU Wave Rows": households[KEYS].drop_duplicates().shape[0],
                "Historical Conflict Linked Rows": int(households["Historical Conflict Link Matched"].sum()),
                "Annual Climate Linked Rows": int(households["Long Baseline Climate Link Matched"].sum()),
                "Wholesale Rice Price Linked Rows": int(households["Wholesale Rice Price Linked"].sum()),
                "Broad Retail Food Price Linked Rows": int(households["Broad Retail Food Price Linked"].sum()),
                "Survey Year Satellite Flood Coverage Rows": int(households["Survey Year Satellite Flood Coverage"].sum()),
                "Landmine Columns": 0,
            },
            {
                "Dataset": "education",
                "Rows": len(education),
                "Columns": education.shape[1],
                "Survey Years": ", ".join(map(str, sorted(education["Survey Year"].unique()))),
                "PSU Wave Rows": education[KEYS].drop_duplicates().shape[0],
                "Historical Conflict Linked Rows": int(education["Historical Conflict Link Matched"].sum()),
                "Annual Climate Linked Rows": int(education["Long Baseline Climate Link Matched"].sum()),
                "Wholesale Rice Price Linked Rows": int(education["Wholesale Rice Price Linked"].sum()),
                "Broad Retail Food Price Linked Rows": int(education["Broad Retail Food Price Linked"].sum()),
                "Survey Year Satellite Flood Coverage Rows": int(education["Survey Year Satellite Flood Coverage"].sum()),
                "Landmine Columns": 0,
            },
        ]
    )
    validation.to_csv(
        diagnostics / "direction3_conflict_shock_release_validation.csv", index=False
    )
    print(validation.to_string(index=False))
    print(f"wrote {household_path.name}: {households.shape}")
    print(f"wrote {education_path.name}: {education.shape}")


if __name__ == "__main__":
    main()
