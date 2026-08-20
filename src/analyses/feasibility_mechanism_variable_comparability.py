#!/usr/bin/env python3
"""Targeted feasibility audit for planned mechanism-family variables.

This audit reads the harmonized CSES module parquets because the generic feasibility
scanner does not inspect parquet files or diagnose conditional questionnaire support.
It does not construct final variables or estimate effects.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INFRASTRUCTURE = ROOT / "data/processed/cses_village_infrastructure_preprocessed.parquet"
VULNERABILITY = ROOT / "data/processed/cses_vulnerability_preprocessed.parquet"
OUTPUT = ROOT / "data/exp/feasibility-check/mechanism-variable-comparability"

YEAR = "Survey Year"
PSU = "PSU"
HOUSEHOLD = "Household ID"

TOTAL_AG_LAND = "Q2 1 What is the total area of agricultural land available in this village"
IRRIGATED_AG_LAND = "Q2 2 Of which the total irrigated agricultural land is"
ALL_WEATHER_ROAD = "Q2 10a Does the village have all weather roads"
ALL_WEATHER_ROAD_DISTANCE = (
    "Q2 10b How many kilometers away from an all weather road is this village"
)
MARKET_2011 = "15 permanent market in the village S2Q15R4C3"
MARKET_2021 = "Is there Permanent market"

COPING_FIELDS = {
    "Productive Asset Sale as Coping": (
        "Q01DQ3B Sold productive assets or means of transport sewing machine wheelbar"
    ),
    "Reduced Essential Education or Health Expenditure": (
        "Q01DQ3C Reduced essential non food expenditures such as education health etc"
    ),
    "Formal Lender Borrowing as Coping": (
        "Q01DQ3E Borrowed money food from a formal lender bank"
    ),
    "Child Withdrawn from School as Coping": "Q01DQ3G Withdrew children from school",
    "Adult Work Migration as Coping": (
        "Q01DQ3I Sent an adult household member sought work elsewhere regardless of the"
    ),
}

ACTIVE_YEARS = {2007, 2009, 2011, 2013, 2014, 2016, 2017, 2019, 2021}


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infrastructure_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        YEAR,
        PSU,
        TOTAL_AG_LAND,
        IRRIGATED_AG_LAND,
        ALL_WEATHER_ROAD,
        ALL_WEATHER_ROAD_DISTANCE,
        MARKET_2011,
        MARKET_2021,
    ]
    data = pd.read_parquet(INFRASTRUCTURE, columns=columns)
    data = data.loc[data[YEAR].isin(ACTIVE_YEARS)].copy()
    data["_total_land"] = numeric(data[TOTAL_AG_LAND])
    data["_irrigated_land"] = numeric(data[IRRIGATED_AG_LAND])
    data["_valid_irrigation_denominator"] = data["_total_land"].gt(0) & data[
        "_irrigated_land"
    ].ge(0)
    data["_irrigation_share"] = np.where(
        data["_valid_irrigation_denominator"],
        data["_irrigated_land"] / data["_total_land"],
        np.nan,
    )
    data["_irrigation_share_out_of_range"] = data["_irrigation_share"].gt(1)

    irrigation = (
        data.groupby(YEAR, observed=True)
        .agg(
            village_rows=(PSU, "size"),
            psus=(PSU, "nunique"),
            valid_denominators=("_valid_irrigation_denominator", "sum"),
            out_of_range_shares=("_irrigation_share_out_of_range", "sum"),
            median_share=("_irrigation_share", "median"),
            p95_share=("_irrigation_share", lambda values: values.quantile(0.95)),
        )
        .reset_index()
    )

    infrastructure = []
    for year, group in data.groupby(YEAR, observed=True):
        market_column = MARKET_2011 if int(year) == 2011 else MARKET_2021 if int(year) == 2021 else None
        market = group[market_column] if market_column else pd.Series(index=group.index, dtype=float)
        infrastructure.append(
            {
                YEAR: int(year),
                "Village Rows": int(len(group)),
                "All-Weather Road Nonmissing": int(group[ALL_WEATHER_ROAD].notna().sum()),
                "All-Weather Road Distance Nonmissing": int(
                    group[ALL_WEATHER_ROAD_DISTANCE].notna().sum()
                ),
                "Permanent Market Nonmissing": int(market.notna().sum()),
                "Permanent Market Yes": int(numeric(market).eq(1).sum()),
                "Permanent Market No": int(numeric(market).eq(2).sum()),
            }
        )
    infrastructure_frame = pd.DataFrame(infrastructure)

    availability = pd.DataFrame(
        [
            {
                "Dataset": "Village infrastructure module",
                "Rows": int(len(data)),
                "Columns Audited": len(columns),
                "Years": ", ".join(map(str, sorted(data[YEAR].dropna().astype(int).unique()))),
                "Unit": "Village survey PSU-year",
            }
        ]
    )
    return irrigation, infrastructure_frame, availability


def coping_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [YEAR, HOUSEHOLD, *COPING_FIELDS.values()]
    data = pd.read_parquet(VULNERABILITY, columns=columns)
    data = data.loc[data[YEAR].eq(2021)].copy()
    records: list[dict[str, object]] = []
    for variable, source in COPING_FIELDS.items():
        values = data[[HOUSEHOLD, source]].dropna().copy()
        values["_binary"] = numeric(values[source]).map({1.0: 0.0, 2.0: 0.0, 3.0: 1.0})
        values = values.dropna(subset=["_binary"])
        grouped = values.groupby(HOUSEHOLD, observed=True)["_binary"]
        any_yes = grouped.max()
        all_yes = grouped.min()
        binary_conflict = grouped.nunique().gt(1)
        consistent_yes = any_yes.eq(1) & ~binary_conflict
        records.append(
            {
                "Variable": variable,
                "Raw Rows": int(len(values)),
                "Households Observed": int(grouped.ngroups),
                "Any-Yes Households": int(any_yes.eq(1).sum()),
                "Any-Yes Rate": float(any_yes.mean()),
                "Consistent-Yes Households": int(consistent_yes.sum()),
                "Consistent-Yes Rate": float(consistent_yes.mean()),
                "Binary-Conflict Households": int(binary_conflict.sum()),
                "All-Yes Households": int(all_yes.eq(1).sum()),
            }
        )
    availability = pd.DataFrame(
        [
            {
                "Dataset": "Household vulnerability module",
                "Rows": int(len(data)),
                "Columns Audited": len(columns),
                "Years": "2021",
                "Unit": "Repeated module rows within household",
            }
        ]
    )
    return pd.DataFrame(records), availability


def feasibility_table(
    irrigation: pd.DataFrame,
    infrastructure: pd.DataFrame,
    coping: pd.DataFrame,
) -> pd.DataFrame:
    irrigation_units = int(irrigation["valid_denominators"].sum())
    irrigation_years = ", ".join(map(str, irrigation[YEAR].astype(int)))
    market_rows = int(infrastructure["Permanent Market Nonmissing"].sum())
    market_years = ", ".join(
        map(
            str,
            infrastructure.loc[
                infrastructure["Permanent Market Nonmissing"].gt(0), YEAR
            ].astype(int),
        )
    )
    road_rows = int(infrastructure["All-Weather Road Nonmissing"].sum())
    records = [
        {
            "Question Component": "Infrastructure and agricultural capacity",
            "Candidate Variable": "Village Irrigated Agricultural Land Share",
            "Visible Support": f"{irrigation_units:,} valid village-years across {irrigation_years}",
            "Feasibility Status": "partly-testable",
            "Primary Risk": "Zero agricultural-land denominators and one active-period share above one",
            "Recommendation": "Advance to preprocessing; set invalid denominators and shares outside [0,1] to missing",
        },
        {
            "Question Component": "Infrastructure and agricultural capacity",
            "Candidate Variable": "Permanent Market Access",
            "Visible Support": f"{market_rows:,} village-years in {market_years}",
            "Feasibility Status": "weakly-testable",
            "Primary Risk": "Only two comparable active-release waves",
            "Recommendation": "Retain as secondary wave-limited candidate; do not describe as a full-period mechanism",
        },
        {
            "Question Component": "Infrastructure and agricultural capacity",
            "Candidate Variable": "All-Weather Road Access",
            "Visible Support": f"{road_rows:,} nonmissing village-years under conditional questionnaire routing",
            "Feasibility Status": "not-yet-testable",
            "Primary Risk": "Question is conditionally observed and missingness cannot be coded as no access",
            "Recommendation": "Defer; do not substitute motorable-road access without a new human decision",
        },
    ]
    for _, row in coping.iterrows():
        records.append(
            {
                "Question Component": "Costly household coping",
                "Candidate Variable": row["Variable"],
                "Visible Support": (
                    f"2021 only; {int(row['Any-Yes Households']):,} any-yes and "
                    f"{int(row['Consistent-Yes Households']):,} consistent-yes households"
                ),
                "Feasibility Status": "not-yet-testable",
                "Primary Risk": "Single wave, very rare positive events, and repeated-record inconsistencies",
                "Recommendation": "Retain for descriptive/reference use only; do not activate interaction estimation",
            }
        )
    return pd.DataFrame(records)


def write_readme(feasibility: pd.DataFrame, coping: pd.DataFrame) -> None:
    irrigation = feasibility.loc[
        feasibility["Candidate Variable"].eq("Village Irrigated Agricultural Land Share")
    ].iloc[0]
    market = feasibility.loc[
        feasibility["Candidate Variable"].eq("Permanent Market Access")
    ].iloc[0]
    text = f"""# Mechanism-variable comparability feasibility

This is an exploratory feasibility audit. It does not construct final variables or estimate
mechanism effects.

## Findings

- Village irrigated agricultural land share is **{irrigation['Feasibility Status']}**:
  {irrigation['Visible Support']}.
- Permanent-market access is **{market['Feasibility Status']}**: {market['Visible Support']}.
- All-weather-road access is not currently testable because the field is conditionally routed;
  missing values are not valid zeros.
- The five costly-coping responses are observed in 2021 only. Under the questionnaire coding,
  code 3 is yes and codes 1 and 2 are distinct no responses. Any-yes household counts range from
  {int(coping['Any-Yes Households'].min())} to {int(coping['Any-Yes Households'].max())} among
  {int(coping['Households Observed'].max()):,} households; positive-event support is too sparse for
  planned interaction estimation.

## Decision implication

Only village irrigated agricultural land share should advance as a primary mechanism candidate.
Permanent-market access may advance as a secondary, wave-limited variable. The all-weather-road
and costly-coping variables should remain non-final unless new evidence or a revised estimand is
approved. Any later mechanism result is channel-consistent evidence, not causal mediation.
"""
    (OUTPUT / "README.md").write_text(text, encoding="utf-8")


def validate_outputs(
    irrigation: pd.DataFrame,
    infrastructure: pd.DataFrame,
    coping: pd.DataFrame,
    feasibility: pd.DataFrame,
) -> None:
    assert irrigation[YEAR].tolist() == [2007, 2009, 2011, 2014, 2016, 2021]
    assert int(irrigation["valid_denominators"].sum()) == 3006
    assert int(irrigation["out_of_range_shares"].sum()) == 1
    assert int(infrastructure["Permanent Market Nonmissing"].sum()) == 1359
    assert int(coping["Households Observed"].max()) == 10080
    assert set(feasibility["Feasibility Status"]) == {
        "partly-testable",
        "weakly-testable",
        "not-yet-testable",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    irrigation, infrastructure, infrastructure_availability = infrastructure_audit()
    coping, coping_availability = coping_audit()
    availability = pd.concat(
        [infrastructure_availability, coping_availability], ignore_index=True
    )
    feasibility = feasibility_table(irrigation, infrastructure, coping)

    availability.to_csv(OUTPUT / "dataset_availability.csv", index=False)
    feasibility.to_csv(OUTPUT / "question_feasibility.csv", index=False)
    feasibility.rename(
        columns={
            "Candidate Variable": "Variable",
            "Visible Support": "Availability",
            "Feasibility Status": "Status",
        }
    ).to_csv(OUTPUT / "variable_inventory.csv", index=False)
    irrigation.to_csv(OUTPUT / "irrigation_validity_by_year.csv", index=False)
    infrastructure.to_csv(OUTPUT / "infrastructure_coverage_by_year.csv", index=False)
    coping.to_csv(OUTPUT / "coping_household_consistency.csv", index=False)
    write_readme(feasibility, coping)
    validate_outputs(irrigation, infrastructure, coping, feasibility)

    print(f"Saved feasibility outputs: {OUTPUT.relative_to(ROOT)}")
    print(feasibility.to_string(index=False))


if __name__ == "__main__":
    main()
