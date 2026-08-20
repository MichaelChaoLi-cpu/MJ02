#!/usr/bin/env python3
"""Construct approved Direction 3 village-level mechanism variables.

The output remains at the village survey PSU-year grain. Village irrigated
agricultural land share is the approved primary candidate. Permanent-market
access is retained only as a two-wave appendix variable. Conditional road
questions and sparse household coping responses are deliberately excluded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/cses_village_infrastructure_preprocessed.parquet"
HOUSEHOLD_PANEL = (
    ROOT / "data/processed/direction3_household_conflict_shock_preprocessed.parquet"
)
OUTPUT = ROOT / "data/processed/direction3_village_mechanisms_preprocessed.parquet"
AUDIT = ROOT / "data/exp/data-preprocessing/direction3_village_mechanism_coverage.csv"

YEAR = "Survey Year"
PSU = "PSU"
TOTAL_LAND_SOURCE = (
    "Q2 1 What is the total area of agricultural land available in this village"
)
IRRIGATED_LAND_SOURCE = "Q2 2 Of which the total irrigated agricultural land is"
MARKET_2011_SOURCE = "15 permanent market in the village S2Q15R4C3"
MARKET_2021_SOURCE = "Is there Permanent market"

ACTIVE_VILLAGE_YEARS = [2007, 2009, 2011, 2014, 2016, 2021]
MARKET_YEARS = [2011, 2021]

IDENTIFIERS = [
    YEAR,
    "Survey Wave",
    "Main Linked Sample",
    PSU,
    "Province Code",
    "District Code",
    "Commune Code",
    "Village Code",
    "Province Name",
    "District Name",
    "Commune Name",
    "Village Name",
    "Urban Rural",
    "Geography Link Matched",
    "Source Dataset",
    "Source Submodule",
    "Module Row ID",
]


def numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def construct_mechanisms(source: pd.DataFrame) -> pd.DataFrame:
    data = source.loc[source[YEAR].isin(ACTIVE_VILLAGE_YEARS)].copy()
    if data.duplicated([YEAR, PSU]).any():
        raise ValueError("Village infrastructure source is not unique by survey year and PSU")

    total_land = numeric(data[TOTAL_LAND_SOURCE])
    irrigated_land = numeric(data[IRRIGATED_LAND_SOURCE])
    valid_irrigation = (
        total_land.gt(0)
        & irrigated_land.ge(0)
        & irrigated_land.le(total_land)
    )

    reason = pd.Series("valid", index=data.index, dtype="string")
    reason.loc[total_land.isna() | irrigated_land.isna()] = "missing_area"
    reason.loc[total_land.notna() & total_land.le(0)] = "nonpositive_total_land"
    reason.loc[irrigated_land.notna() & irrigated_land.lt(0)] = "negative_irrigated_land"
    reason.loc[
        total_land.gt(0) & irrigated_land.notna() & irrigated_land.gt(total_land)
    ] = "irrigated_exceeds_total_land"

    market_raw = pd.Series(np.nan, index=data.index, dtype="float64")
    market_raw.loc[data[YEAR].eq(2011)] = numeric(
        data.loc[data[YEAR].eq(2011), MARKET_2011_SOURCE]
    )
    market_raw.loc[data[YEAR].eq(2021)] = numeric(
        data.loc[data[YEAR].eq(2021), MARKET_2021_SOURCE]
    )
    invalid_market = market_raw.notna() & ~market_raw.isin([1.0, 2.0])
    if invalid_market.any():
        raise ValueError("Permanent-market source contains values outside the verified 1/2 coding")

    output = data[IDENTIFIERS].copy()
    output["Village Agricultural Land Area ha"] = total_land.astype("float64")
    output["Village Irrigated Agricultural Land Area ha"] = irrigated_land.astype(
        "float64"
    )
    output["Village Irrigation Measurement Valid"] = valid_irrigation.astype(bool)
    output["Village Irrigation Missing or Invalid Reason"] = reason
    output["Village Irrigated Agricultural Land Share"] = np.where(
        valid_irrigation, irrigated_land / total_land, np.nan
    )
    output["Permanent Market Measurement Available"] = market_raw.notna()
    output["Permanent Market Access"] = market_raw.map({1.0: 1.0, 2.0: 0.0})
    return output.sort_values([YEAR, PSU], kind="stable").reset_index(drop=True)


def build_coverage_audit(
    mechanisms: pd.DataFrame, household: pd.DataFrame
) -> pd.DataFrame:
    household_keys = household[[YEAR, PSU]].copy()
    joined = household_keys.merge(
        mechanisms[
            [
                YEAR,
                PSU,
                "Village Irrigated Agricultural Land Share",
                "Permanent Market Access",
            ]
        ],
        on=[YEAR, PSU],
        how="left",
        validate="many_to_one",
    )

    records: list[dict[str, int | float]] = []
    for year in sorted(household[YEAR].dropna().astype(int).unique()):
        village_year = mechanisms.loc[mechanisms[YEAR].eq(year)]
        household_year = household_keys.loc[household_keys[YEAR].eq(year)]
        joined_year = joined.loc[joined[YEAR].eq(year)]
        household_psus = set(household_year[PSU].astype(str))
        mechanism_psus = set(village_year[PSU].astype(str))
        records.append(
            {
                YEAR: year,
                "Household Rows": len(household_year),
                "Household PSUs": len(household_psus),
                "Village Mechanism Rows": len(village_year),
                "Village Mechanism PSUs": len(mechanism_psus),
                "Matched Household PSUs": len(household_psus & mechanism_psus),
                "Household Rows with Valid Irrigation Share": int(
                    joined_year["Village Irrigated Agricultural Land Share"].notna().sum()
                ),
                "Household Rows with Market Access": int(
                    joined_year["Permanent Market Access"].notna().sum()
                ),
            }
        )
    return pd.DataFrame(records)


def validate(mechanisms: pd.DataFrame, audit: pd.DataFrame) -> None:
    if mechanisms.duplicated([YEAR, PSU]).any():
        raise ValueError("Processed village mechanisms are not unique by year and PSU")
    if mechanisms[YEAR].drop_duplicates().tolist() != ACTIVE_VILLAGE_YEARS:
        raise ValueError("Processed village mechanism years differ from the approved years")
    share = mechanisms["Village Irrigated Agricultural Land Share"].dropna()
    if not share.between(0, 1, inclusive="both").all():
        raise ValueError("Final irrigation shares fall outside [0, 1]")
    if int(share.shape[0]) != 3005:
        raise ValueError("Unexpected number of valid in-range irrigation shares")
    if int(mechanisms["Permanent Market Access"].notna().sum()) != 1359:
        raise ValueError("Unexpected permanent-market coverage")
    if not set(mechanisms["Permanent Market Access"].dropna().unique()).issubset({0.0, 1.0}):
        raise ValueError("Permanent Market Access is not binary")
    market_nonmissing_years = sorted(
        mechanisms.loc[
            mechanisms["Permanent Market Access"].notna(), YEAR
        ].unique()
    )
    if market_nonmissing_years != MARKET_YEARS:
        raise ValueError("Permanent-market values appear outside the approved waves")
    if audit.loc[audit[YEAR].isin([2013, 2017, 2019]), "Village Mechanism Rows"].sum() != 0:
        raise ValueError("Village mechanism records unexpectedly appear in unavailable waves")


def main() -> None:
    source_columns = [
        *IDENTIFIERS,
        TOTAL_LAND_SOURCE,
        IRRIGATED_LAND_SOURCE,
        MARKET_2011_SOURCE,
        MARKET_2021_SOURCE,
    ]
    source = pd.read_parquet(SOURCE, columns=source_columns)
    household = pd.read_parquet(HOUSEHOLD_PANEL, columns=[YEAR, PSU])
    mechanisms = construct_mechanisms(source)
    audit = build_coverage_audit(mechanisms, household)
    validate(mechanisms, audit)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    mechanisms.to_parquet(OUTPUT, index=False)
    audit.to_csv(AUDIT, index=False)

    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved: {AUDIT.relative_to(ROOT)}")
    print(audit.to_string(index=False))
    print("\nFinal variable coverage")
    print(
        mechanisms.groupby(YEAR, observed=True)
        .agg(
            village_rows=(PSU, "size"),
            irrigation_share=("Village Irrigated Agricultural Land Share", "count"),
            permanent_market=("Permanent Market Access", "count"),
        )
        .to_string()
    )


if __name__ == "__main__":
    main()
