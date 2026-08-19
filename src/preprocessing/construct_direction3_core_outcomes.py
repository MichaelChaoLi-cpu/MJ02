#!/usr/bin/env python3
"""Construct concept-level Direction 3 household and education outcomes.

The script aggregates only variables with stable question wording and units in
the 2007-2021 main sample. It preserves nominal monetary values, does not impute
missing observations, and does not winsorize. CSES 2004 remains in the household
spine but is not forced into incompatible post-2007 constructions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HOUSEHOLD_KEYS = ["Survey Wave", "Household ID"]
PERSON_KEYS = ["Survey Wave", "Person ID"]


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def group_sum(frame: pd.DataFrame, values: pd.Series, name: str) -> pd.DataFrame:
    work = frame[HOUSEHOLD_KEYS].copy()
    # Keep aggregated outputs Arrow-safe and numerically comparable.  Pandas can
    # otherwise retain a mixed object column when nullable booleans are summed.
    work[name] = pd.to_numeric(values, errors="coerce").astype("Float64")
    return work.groupby(HOUSEHOLD_KEYS, dropna=False)[name].sum(min_count=1).reset_index()


def group_count(frame: pd.DataFrame, values: pd.Series, name: str) -> pd.DataFrame:
    work = frame[HOUSEHOLD_KEYS].copy()
    work[name] = values
    return work.groupby(HOUSEHOLD_KEYS, dropna=False)[name].count().reset_index()


def merge_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    output = parts[0]
    for part in parts[1:]:
        output = output.merge(part, on=HOUSEHOLD_KEYS, how="outer", validate="one_to_one")
    return output


def binary_yes_no(values: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [numeric_values.eq(1), numeric_values.eq(2)],
            [1.0, 0.0],
            default=np.nan,
        ),
        index=values.index,
    )


def first_nonmissing(values: pd.Series):
    present = values.dropna()
    return present.iloc[0] if len(present) else np.nan


def household_size_and_demographics(processed: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    members = pd.read_parquet(processed / "cses_household_members_preprocessed.parquet")
    age_early = numeric(members, "Q01AC05")
    age_late = numeric(members, "Q01AC05A What is NAME s age in completed years")
    age_2004 = numeric(members, "Age").combine_first(numeric(members, "Q01A05 age completed"))
    members["Age Years"] = age_late.combine_first(age_early).combine_first(age_2004)
    members["Female"] = binary_yes_no(numeric(members, "Q01AC03 Sex").replace({1: 2, 2: 1}))
    members.loc[members["Survey Wave"].eq("2004"), "Female"] = binary_yes_no(
        numeric(members.loc[members["Survey Wave"].eq("2004")], "Sex").replace({1: 2, 2: 1})
    ).values
    members["Child Age 6 to 17"] = members["Age Years"].between(6, 17, inclusive="both")

    household = members.groupby(HOUSEHOLD_KEYS, dropna=False).agg(
        **{
            "Household Size": ("Module Row ID", "size"),
            "Children Age 6 to 17 Count": ("Child Age 6 to 17", "sum"),
        }
    ).reset_index()
    person = members[
        HOUSEHOLD_KEYS + ["Person ID", "Age Years", "Female"]
    ].drop_duplicates(PERSON_KEYS, keep="first")
    return household, person


def household_weights(processed: Path) -> pd.DataFrame:
    weights = pd.read_parquet(processed / "cses_household_weights_preprocessed.parquet")
    mapping = {
        "2007": "Hhweightadjusted",
        "2009": "Hw09A",
        "2011-12": "Hw11A",
        "2013": "Hw13A",
        "2014": "Hw14A",
        "2016": "Hw16A",
        "2017": "Hw17A",
        "2021": "HW2021",
    }
    weights["Household Survey Weight"] = np.nan
    for wave, column in mapping.items():
        mask = weights["Survey Wave"].eq(wave)
        weights.loc[mask, "Household Survey Weight"] = numeric(weights.loc[mask], column).values
    output = weights.groupby(HOUSEHOLD_KEYS, dropna=False)["Household Survey Weight"].agg(
        first_nonmissing
    ).reset_index()

    members = pd.read_parquet(
        processed / "cses_household_members_preprocessed.parquet",
        columns=["Survey Wave", "Household ID", "Hw20A", "Household weight"],
    )
    members["Embedded Household Survey Weight"] = numeric(members, "Hw20A").combine_first(
        numeric(members, "Household weight")
    )
    embedded = members.groupby(HOUSEHOLD_KEYS, dropna=False)["Embedded Household Survey Weight"].agg(
        first_nonmissing
    ).reset_index()
    output = output.merge(embedded, on=HOUSEHOLD_KEYS, how="outer", validate="one_to_one")
    output["Household Survey Weight"] = output["Household Survey Weight"].combine_first(
        output["Embedded Household Survey Weight"]
    )
    return output.drop(columns="Embedded Household Survey Weight")


def land_outcomes(processed: Path) -> pd.DataFrame:
    land = pd.read_parquet(processed / "cses_agriculture_land_preprocessed.parquet")
    land = land[land["Survey Wave"].ne("2004")].copy()
    area = numeric(land, "Q05AC02 What is the area of the parcel in square meters m2")
    irrigable = binary_yes_no(numeric(land, "Q05AC17 Can you add water to this parcel with irrigation and or water pumped f"))
    parts = [
        group_sum(land, area, "Total Parcel Area m2"),
        group_count(land, area, "Parcel Area Observation Count"),
        group_sum(land, irrigable, "Irrigable Parcel Count"),
        group_count(land, irrigable, "Irrigation Status Observation Count"),
    ]
    output = merge_parts(parts)
    output["Irrigable Parcel Share"] = output["Irrigable Parcel Count"] / output[
        "Irrigation Status Observation Count"
    ].replace(0, np.nan)
    output["Any Irrigable Parcel"] = np.where(
        output["Irrigation Status Observation Count"].gt(0),
        output["Irrigable Parcel Count"].gt(0).astype(float),
        np.nan,
    )
    return output


def crop_outcomes(processed: Path) -> pd.DataFrame:
    crops = pd.read_parquet(processed / "cses_agriculture_crop_production_preprocessed.parquet")
    crops = crops[crops["Survey Wave"].ne("2004")].copy()
    cultivated = numeric(crops, "Q05BC04 How big area was cultivated m2")
    harvested = numeric(crops, "Q05BC05 How big area was harvested m2")
    production = numeric(crops, "Q05BC06 How much was produced harvested KG")
    loss = numeric(crops, "Q05BC07 How much has been the post harvest loss until the day of interview KG")
    rent = numeric(crops, "Q05BC08 How much quantity was given as crop rent KG")
    price = numeric(crops, "Q05BC09 What was the sale price of the crop produced per kg RIELS Kg")
    value = production * price
    crop_code_column = "Q05BC03B What crop s have yourhousehold grown on what parcels"
    crop_code = crops[crop_code_column] if crop_code_column in crops else pd.Series(pd.NA, index=crops.index)
    crop_code = crop_code.astype("string").replace({"": pd.NA, "nan": pd.NA})

    parts = [
        group_sum(crops, cultivated, "Cultivated Crop Area m2"),
        group_sum(crops, harvested, "Harvested Crop Area m2"),
        group_sum(crops, production, "Crop Production Quantity kg"),
        group_sum(crops, loss, "Post Harvest Crop Loss kg"),
        group_sum(crops, rent, "Crop Rent Quantity kg"),
        group_sum(crops, value, "Nominal Crop Production Value Riels"),
        group_count(crops, production, "Crop Production Observation Count"),
    ]
    output = merge_parts(parts)
    diversity = crops.assign(**{"Crop Code": crop_code}).groupby(HOUSEHOLD_KEYS, dropna=False)[
        "Crop Code"
    ].nunique(dropna=True).rename("Crop Diversity Count").reset_index()
    output = output.merge(diversity, on=HOUSEHOLD_KEYS, how="outer", validate="one_to_one")
    output["Crop Yield kg per ha"] = (
        output["Crop Production Quantity kg"] * 10000
        / output["Harvested Crop Area m2"].replace(0, np.nan)
    )
    output["Post Harvest Loss Share"] = output["Post Harvest Crop Loss kg"] / (
        output["Crop Production Quantity kg"] + output["Post Harvest Crop Loss kg"]
    ).replace(0, np.nan)
    output["Any Crop Production Record"] = output["Crop Production Observation Count"].gt(0)
    return output


def crop_cost_outcomes(processed: Path) -> pd.DataFrame:
    costs = pd.read_parquet(processed / "cses_agriculture_crop_costs_preprocessed.parquet")
    costs = costs[costs["Survey Wave"].ne("2004")].copy()
    total = numeric(costs, "Q05CC16 Total Col 3 15 RIELS")
    return merge_parts(
        [
            group_sum(costs, total, "Nominal Agricultural Input Cost Riels"),
            group_count(costs, total, "Agricultural Cost Observation Count"),
        ]
    )


def food_consumption_outcomes(processed: Path) -> pd.DataFrame:
    food = pd.read_parquet(processed / "cses_food_consumption_preprocessed.parquet")
    total = pd.Series(np.nan, index=food.index)
    purchased = pd.Series(np.nan, index=food.index)
    own = pd.Series(np.nan, index=food.index)
    mappings = [
        (
            ["2004"],
            "Q01D04 total consumption",
            "Q01D02 value of consumption purchased riels",
            "Q01D03 value of consumption own produce riels",
        ),
        (
            ["2007", "2009", "2011-12", "2013", "2014", "2016", "2017"],
            "Q01BC05",
            "Q01BC03",
            "Q01BC04",
        ),
        (
            ["2019", "2021"],
            "Q01BC7 Total consumption column 5 column 6",
            "Q01BC5 Purchased in cash RIELS",
            "Q01BC6 Own production wages in kind gifts free collections imputed value In",
        ),
    ]
    for waves, total_col, purchased_col, own_col in mappings:
        mask = food["Survey Wave"].isin(waves)
        total.loc[mask] = numeric(food.loc[mask], total_col).values
        purchased.loc[mask] = numeric(food.loc[mask], purchased_col).values
        own.loc[mask] = numeric(food.loc[mask], own_col).values
    return merge_parts(
        [
            group_sum(food, total, "Reported Food Consumption Value Riels"),
            group_sum(food, purchased, "Purchased Food Consumption Value Riels"),
            group_sum(food, own, "Own Produced Food Consumption Value Riels"),
            group_count(food, total, "Food Item Observation Count"),
            group_sum(food, total.gt(0).where(total.notna()), "Food Items with Positive Consumption Count"),
        ]
    )


def food_security_outcomes(processed: Path) -> pd.DataFrame:
    dedicated = pd.read_parquet(processed / "cses_food_security_preprocessed.parquet")
    vulnerability = pd.read_parquet(processed / "cses_vulnerability_preprocessed.parquet")
    vulnerability = vulnerability[vulnerability["Survey Wave"].isin(["2014", "2016", "2017"])]
    frame = pd.concat([vulnerability, dedicated], ignore_index=True, sort=False)
    q4 = numeric(frame, "Q01DQ4 In the past 30 days how often has your household ever no food to eat of")
    q5 = numeric(frame, "Q01DQ5 In the past 30 days how often did you or any household member go to slee")
    q6 = numeric(frame, "Q01DQ6 In the past 30 days how often did you or any household member go a whole")
    is_2014 = frame["Survey Wave"].eq("2014")

    def occurrence(values: pd.Series) -> pd.Series:
        result = pd.Series(np.nan, index=values.index)
        result.loc[is_2014 & values.notna()] = values.loc[is_2014 & values.notna()].eq(1).astype(float)
        result.loc[~is_2014 & values.notna()] = values.loc[~is_2014 & values.notna()].gt(1).astype(float)
        return result

    no_food = occurrence(q4)
    sleep_hungry = occurrence(q5)
    whole_day = occurrence(q6)
    experience = pd.concat([no_food, sleep_hungry, whole_day], axis=1).max(axis=1, skipna=True)
    experience[pd.concat([no_food, sleep_hungry, whole_day], axis=1).notna().sum(axis=1).eq(0)] = np.nan
    severity = pd.concat([(q4 - 1).clip(lower=0), (q5 - 1).clip(lower=0), (q6 - 1).clip(lower=0)], axis=1).sum(
        axis=1, min_count=1
    )
    severity.loc[is_2014] = np.nan
    work = frame[HOUSEHOLD_KEYS].copy()
    work["No Food Experience"] = no_food
    work["Went to Sleep Hungry"] = sleep_hungry
    work["Went Whole Day Without Eating"] = whole_day
    work["Any Severe Food Insecurity Experience"] = experience
    work["Food Insecurity Severity Sum"] = severity
    return work.groupby(HOUSEHOLD_KEYS, dropna=False).agg(
        **{
            "No Food Experience": ("No Food Experience", first_nonmissing),
            "Went to Sleep Hungry": ("Went to Sleep Hungry", first_nonmissing),
            "Went Whole Day Without Eating": ("Went Whole Day Without Eating", first_nonmissing),
            "Any Severe Food Insecurity Experience": ("Any Severe Food Insecurity Experience", first_nonmissing),
            "Food Insecurity Severity Sum": ("Food Insecurity Severity Sum", first_nonmissing),
        }
    ).reset_index()


def enhanced_climate(processed: Path) -> pd.DataFrame:
    climate = pd.read_parquet(processed / "direction3_psu_year_climate_enhanced_preprocessed.parquet")
    metadata = [
        "Survey Wave",
        "PSU",
        "Climate Geography Resolution",
        "Climate Geography Code",
        "Climate Link Method",
        "Climate Match Score",
        "Climate Match Margin",
        "Matched Climate Province Name",
        "Matched Climate District Name",
        "Matched Climate Commune Name",
        "Exact Commune Climate Link Matched",
        "Enhanced Climate Link Matched",
    ]
    climate_values = [
        "Annual Rainfall mm",
        "Observed Climate Months",
        "Climate Grid Cell Count",
        "Climate Extraction Method",
        "May October Rainfall mm",
        "Annual Rainfall Anomaly Z",
        "Annual Rainfall Bottom Decile",
        "Annual Rainfall Top Decile",
        "May October Rainfall Anomaly Z",
        "May October Rainfall Bottom Decile",
        "May October Rainfall Top Decile",
    ]
    output = climate[metadata + climate_values].copy()
    output = output.rename(columns={column: f"Enhanced {column}" for column in climate_values})
    return output


def annual_cpi_component(processed: Path, component: str, prefix: str) -> pd.DataFrame:
    annual = pd.read_parquet(processed / "cambodia_cpi_annual_preprocessed.parquet")
    annual = annual.loc[annual["CPI Component Code"].eq(component)].copy()
    annual = annual[
        ["Year", "Annual Mean CPI Index", "CPI 2021 Annual Mean", "Annual Deflator to 2021"]
    ].rename(
        columns={
            "Year": "Survey Year",
            "Annual Mean CPI Index": f"{prefix} Annual Mean CPI Index",
            "CPI 2021 Annual Mean": f"{prefix} CPI 2021 Annual Mean",
            "Annual Deflator to 2021": f"{prefix} CPI Deflator to 2021",
        }
    )
    return annual


def add_real_household_values(household: pd.DataFrame, processed: Path) -> pd.DataFrame:
    output = household.merge(
        annual_cpi_component(processed, "_T", "All Items"),
        on="Survey Year",
        how="left",
        validate="many_to_one",
    )
    output["Agricultural Monetary Deflation Method"] = np.where(
        output["All Items CPI Deflator to 2021"].notna(),
        "annual all-items CPI",
        pd.NA,
    )
    for nominal, real in [
        ("Nominal Crop Production Value Riels", "Real 2021 Crop Production Value Riels"),
        ("Nominal Agricultural Input Cost Riels", "Real 2021 Agricultural Input Cost Riels"),
        (
            "Agricultural Input Cost per Cultivated ha Riels",
            "Real 2021 Agricultural Input Cost per Cultivated ha Riels",
        ),
    ]:
        output[real] = numeric(output, nominal) * numeric(output, "All Items CPI Deflator to 2021")

    food_annual = annual_cpi_component(processed, "CP01", "Food")
    output = output.merge(food_annual, on="Survey Year", how="left", validate="many_to_one")
    food_monthly = pd.read_parquet(processed / "cambodia_cpi_monthly_preprocessed.parquet")
    food_monthly = food_monthly.loc[food_monthly["CPI Component Code"].eq("CP01")].copy()
    food_monthly = food_monthly[
        ["Year", "Month", "CPI Index", "CPI 2021 Annual Mean", "Deflator to 2021"]
    ].rename(
        columns={
            "Year": "Survey Year",
            "Month": "CPI Match Month",
            "CPI Index": "Monthly Food CPI Index",
            "CPI 2021 Annual Mean": "Monthly Food CPI 2021 Annual Mean",
            "Deflator to 2021": "Monthly Food CPI Deflator to 2021",
        }
    )
    output["CPI Match Month"] = numeric(output, "Survey Month").astype("Int64")
    output = output.merge(
        food_monthly,
        on=["Survey Year", "CPI Match Month"],
        how="left",
        validate="many_to_one",
    )
    output["Food CPI Index Used"] = output["Monthly Food CPI Index"].combine_first(
        output["Food Annual Mean CPI Index"]
    )
    output["Food CPI 2021 Annual Mean Used"] = output[
        "Monthly Food CPI 2021 Annual Mean"
    ].combine_first(output["Food CPI 2021 Annual Mean"])
    output["Food CPI Deflator to 2021 Used"] = output[
        "Monthly Food CPI Deflator to 2021"
    ].combine_first(output["Food CPI Deflator to 2021"])
    output["Food Monetary Deflation Method"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output.loc[output["Monthly Food CPI Index"].notna(), "Food Monetary Deflation Method"] = (
        "monthly food CPI"
    )
    annual_fallback = output["Monthly Food CPI Index"].isna() & output[
        "Food Annual Mean CPI Index"
    ].notna()
    output.loc[annual_fallback, "Food Monetary Deflation Method"] = "annual food CPI fallback"
    for nominal, real in [
        ("Reported Food Consumption Value Riels", "Real 2021 Reported Food Consumption Value Riels"),
        ("Purchased Food Consumption Value Riels", "Real 2021 Purchased Food Consumption Value Riels"),
        ("Own Produced Food Consumption Value Riels", "Real 2021 Own Produced Food Consumption Value Riels"),
        (
            "Food Consumption Value per Household Member Riels",
            "Real 2021 Food Consumption Value per Household Member Riels",
        ),
    ]:
        output[real] = numeric(output, nominal) * numeric(output, "Food CPI Deflator to 2021 Used")
    main = output["Main Linked Sample"].fillna(False)
    for nominal, real in [
        ("Nominal Crop Production Value Riels", "Real 2021 Crop Production Value Riels"),
        ("Nominal Agricultural Input Cost Riels", "Real 2021 Agricultural Input Cost Riels"),
        ("Reported Food Consumption Value Riels", "Real 2021 Reported Food Consumption Value Riels"),
        ("Purchased Food Consumption Value Riels", "Real 2021 Purchased Food Consumption Value Riels"),
        ("Own Produced Food Consumption Value Riels", "Real 2021 Own Produced Food Consumption Value Riels"),
    ]:
        if (main & output[nominal].notna() & output[real].isna()).any():
            raise ValueError(f"Missing real-price conversion for observed main-sample {nominal}")
    return output.drop(
        columns=[
            "CPI Match Month",
            "Monthly Food CPI Index",
            "Monthly Food CPI 2021 Annual Mean",
            "Monthly Food CPI Deflator to 2021",
        ]
    )


def person_weights(processed: Path, education: pd.DataFrame) -> pd.DataFrame:
    weights = pd.read_parquet(processed / "cses_person_weights_preprocessed.parquet")
    mapping = {
        "2007": "Personweightadjusted",
        "2009": "Pw09A",
        "2011-12": "Pw11A",
        "2013": "Pw13A",
        "2014": "Pw14A",
        "2016": "Pw16A",
        "2017": "Pw17A",
        "2021": "Person weight",
    }
    weights["Person Survey Weight"] = np.nan
    for wave, column in mapping.items():
        mask = weights["Survey Wave"].eq(wave)
        weights.loc[mask, "Person Survey Weight"] = numeric(weights.loc[mask], column).values
    output = weights.groupby(PERSON_KEYS, dropna=False)["Person Survey Weight"].agg(first_nonmissing).reset_index()
    embedded = education[PERSON_KEYS].copy()
    embedded["Embedded Person Survey Weight"] = numeric(education, "Pw20A").combine_first(
        numeric(education, "Person weight")
    )
    embedded = embedded.groupby(PERSON_KEYS, dropna=False)["Embedded Person Survey Weight"].agg(
        first_nonmissing
    ).reset_index()
    output = output.merge(embedded, on=PERSON_KEYS, how="outer", validate="one_to_one")
    output["Person Survey Weight"] = output["Person Survey Weight"].combine_first(
        output["Embedded Person Survey Weight"]
    )
    return output.drop(columns="Embedded Person Survey Weight")


def education_outcomes(
    processed: Path,
    person_demographics: pd.DataFrame,
    household: pd.DataFrame,
) -> pd.DataFrame:
    education = pd.read_parquet(processed / "cses_education_preprocessed.parquet")
    line = numeric(education, "Q02C01 Line number").astype("Int64").astype("string").str.zfill(2)
    derived_person = education["Household ID"].astype("string") + line
    education["Person ID"] = education["Person ID"].astype("string").fillna(derived_person)
    education = education.drop_duplicates(PERSON_KEYS, keep="first").reset_index(drop=True)
    output = education[
        [
            "Survey Year",
            "Survey Wave",
            "Main Linked Sample",
            "Household ID",
            "Person ID",
            "PSU",
            "Province Code",
            "District Code",
            "Commune Code",
            "Village Code",
            "Urban Rural",
            "Geography Link Matched",
        ]
    ].copy()
    output = output.merge(person_demographics, on=HOUSEHOLD_KEYS + ["Person ID"], how="left", validate="one_to_one")
    output["Ever Attended School"] = binary_yes_no(numeric(education, "Q02C04 Has NAME ever attended school"))
    output["Currently Attending School"] = binary_yes_no(
        numeric(education, "Q02C07 Is NAME currently in the school system")
    )
    output["Years Attended School"] = numeric(education, "Q02C05 How many years has NAME attended school")
    output["Nominal Education Expenditure Riels"] = numeric(education, "Q02C16H TOTAL Col 16a 16g")
    output["School Age 6 to 17"] = output["Age Years"].between(6, 17, inclusive="both")
    output["School Attendance Outcome Eligible"] = output["School Age 6 to 17"] & output[
        "Currently Attending School"
    ].notna()

    weights = person_weights(processed, education)
    output = output.merge(weights, on=PERSON_KEYS, how="left", validate="one_to_one")
    household_context = household[
        [
            "Survey Wave",
            "Household ID",
            "Household Size",
            "Household Survey Weight",
            "Mine Baseline Recorded",
            "Mine Baseline Record Count",
            "Log Mine Baseline Record Count",
            "Climate Geography Resolution",
            "Climate Link Method",
            "Exact Commune Climate Link Matched",
            "Enhanced Climate Link Matched",
            "Enhanced Annual Rainfall mm",
            "Enhanced Annual Rainfall Anomaly Z",
            "Enhanced Annual Rainfall Bottom Decile",
            "Enhanced Annual Rainfall Top Decile",
            "Enhanced May October Rainfall mm",
            "Enhanced May October Rainfall Anomaly Z",
            "Enhanced May October Rainfall Bottom Decile",
            "Enhanced May October Rainfall Top Decile",
        ]
    ]
    output = output.merge(household_context, on=HOUSEHOLD_KEYS, how="left", validate="many_to_one")
    output = output.merge(
        annual_cpi_component(processed, "CP10", "Education"),
        on="Survey Year",
        how="left",
        validate="many_to_one",
    )
    output["Education Monetary Deflation Method"] = np.where(
        output["Education CPI Deflator to 2021"].notna(),
        "annual education CPI",
        pd.NA,
    )
    output["Real 2021 Education Expenditure Riels"] = numeric(
        output, "Nominal Education Expenditure Riels"
    ) * numeric(output, "Education CPI Deflator to 2021")
    return output.sort_values(["Survey Year", "PSU", "Household ID", "Person ID"]).reset_index(drop=True)


def variable_dictionary() -> pd.DataFrame:
    rows = [
        ("Household Size", "Number of household-member records", "control", "persons", "Count of member records within household-wave"),
        ("Children Age 6 to 17 Count", "Number of children age 6 to 17", "control", "persons", "Count of member records with age from 6 through 17"),
        ("Household Survey Weight", "Household survey weight", "weight", "survey-weight units", "Wave-specific household weight, using embedded weights in later waves"),
        ("Total Parcel Area m2", "Total reported parcel area", "outcome", "square metres", "Sum of parcel area across post-2007 land records"),
        ("Parcel Area Observation Count", "Observed parcel-area records", "quality", "count", "Number of nonmissing parcel-area records"),
        ("Irrigable Parcel Count", "Number of irrigable parcels", "mechanism", "count", "Sum of yes-coded irrigation status records"),
        ("Irrigable Parcel Share", "Share of parcels that can receive irrigation", "mechanism", "share", "Irrigable parcel count divided by observed irrigation-status count"),
        ("Any Irrigable Parcel", "Any irrigable parcel", "mechanism", "binary", "One if at least one observed parcel is irrigable"),
        ("Cultivated Crop Area m2", "Total cultivated crop area", "outcome", "square metres", "Sum across crop-production records"),
        ("Harvested Crop Area m2", "Total harvested crop area", "outcome", "square metres", "Sum across crop-production records"),
        ("Crop Production Quantity kg", "Total crop production quantity", "outcome", "kilograms", "Sum across crop-production records"),
        ("Crop Yield kg per ha", "Crop yield", "outcome", "kilograms per hectare", "Production quantity times 10000 divided by harvested area"),
        ("Crop Diversity Count", "Number of distinct reported crop codes", "mechanism", "count", "Distinct crop codes within household-wave"),
        ("Post Harvest Crop Loss kg", "Post-harvest crop loss", "outcome", "kilograms", "Sum of reported post-harvest loss across crop records"),
        ("Post Harvest Loss Share", "Share of crop output lost after harvest", "outcome", "share", "Loss divided by production plus loss"),
        ("Crop Rent Quantity kg", "Crop quantity paid as rent", "mechanism", "kilograms", "Sum of reported crop-rent quantity"),
        ("Nominal Crop Production Value Riels", "Production quantity valued at reported crop price", "outcome", "nominal riels", "Crop quantity times reported price, summed within household-wave"),
        ("Real 2021 Crop Production Value Riels", "Crop production value at 2021 prices", "outcome", "2021 riels", "Nominal crop production value multiplied by the annual all-items CPI deflator to the 2021 annual mean"),
        ("Any Crop Production Record", "Any crop-production record", "sample", "binary", "One if at least one production quantity is observed"),
        ("Nominal Agricultural Input Cost Riels", "Reported agricultural input costs", "outcome", "nominal riels", "Sum of reported total crop costs"),
        ("Real 2021 Agricultural Input Cost Riels", "Agricultural input costs at 2021 prices", "outcome", "2021 riels", "Nominal agricultural input costs multiplied by the annual all-items CPI deflator to the 2021 annual mean"),
        ("Agricultural Input Cost per Cultivated ha Riels", "Agricultural input cost per cultivated hectare", "outcome", "nominal riels per hectare", "Input costs times 10000 divided by cultivated area"),
        ("Real 2021 Agricultural Input Cost per Cultivated ha Riels", "Agricultural input cost per cultivated hectare at 2021 prices", "outcome", "2021 riels per hectare", "Nominal cost per cultivated hectare multiplied by the annual all-items CPI deflator"),
        ("Agricultural Household", "Agricultural-household indicator", "sample", "binary", "Any positive parcel area or observed crop-production record"),
        ("All Items CPI Deflator to 2021", "Annual all-items CPI deflator to 2021", "deflator", "ratio", "2021 annual mean all-items CPI divided by the survey-year annual mean all-items CPI"),
        ("Food CPI Deflator to 2021 Used", "Food CPI deflator to 2021", "deflator", "ratio", "2021 annual mean food CPI divided by interview-month food CPI; annual-mean fallback when month is unavailable"),
        ("Food Monetary Deflation Method", "Food monetary deflation method", "quality", "category", "Records monthly food CPI or annual food CPI fallback"),
        ("Reported Food Consumption Value Riels", "Reported food consumption value", "outcome", "nominal riels", "Sum of item-level total food consumption values using wave-specific equivalent fields"),
        ("Real 2021 Reported Food Consumption Value Riels", "Reported food consumption value at 2021 prices", "outcome", "2021 riels", "Nominal value multiplied by monthly food CPI deflator; annual food CPI fallback when interview month is unavailable"),
        ("Purchased Food Consumption Value Riels", "Purchased component of food consumption", "outcome", "nominal riels", "Sum of item-level purchased food values"),
        ("Real 2021 Purchased Food Consumption Value Riels", "Purchased food consumption at 2021 prices", "outcome", "2021 riels", "Nominal purchased food value multiplied by the applicable food CPI deflator"),
        ("Own Produced Food Consumption Value Riels", "Own-production component of food consumption", "outcome", "nominal riels", "Sum of item-level own-production and in-kind food values"),
        ("Real 2021 Own Produced Food Consumption Value Riels", "Own-produced food consumption at 2021 prices", "outcome", "2021 riels", "Nominal own-produced and in-kind food value multiplied by the applicable food CPI deflator"),
        ("Food Consumption Value per Household Member Riels", "Reported food consumption per household member", "outcome", "nominal riels per person", "Household food value divided by household size"),
        ("Real 2021 Food Consumption Value per Household Member Riels", "Food consumption per household member at 2021 prices", "outcome", "2021 riels per person", "Nominal per-member food value multiplied by the applicable food CPI deflator"),
        ("Food Item Observation Count", "Observed food-item records", "quality", "count", "Number of nonmissing item-level total food values"),
        ("Food Items with Positive Consumption Count", "Food items with positive consumption", "outcome", "count", "Count of observed item values greater than zero"),
        ("No Food Experience", "Household had no food to eat", "outcome", "binary", "Any reported occurrence during the direct-question recall period"),
        ("Went to Sleep Hungry", "Household member went to sleep hungry", "outcome", "binary", "Any reported occurrence during the direct-question recall period"),
        ("Went Whole Day Without Eating", "Household member went a whole day without eating", "outcome", "binary", "Any reported occurrence during the direct-question recall period"),
        ("Any Severe Food Insecurity Experience", "Any severe food insecurity experience", "outcome", "binary", "Any no-food, sleep-hungry, or whole-day-without-food experience"),
        ("Food Insecurity Severity Sum", "Ordinal food insecurity severity sum", "outcome", "index", "Sum of three 0-3 frequency codings for 2016, 2017, 2019, and 2021"),
        ("Age Years", "Age in completed years", "control", "years", "Wave-equivalent household-member age field"),
        ("Female", "Female indicator", "control", "binary", "One for female and zero for male"),
        ("Ever Attended School", "Ever attended school", "outcome", "binary", "One for yes and zero for no"),
        ("Currently Attending School", "Currently attending school", "outcome", "binary", "One for yes and zero for no"),
        ("School Age 6 to 17", "School-age indicator", "sample", "binary", "Age from 6 through 17 inclusive"),
        ("School Attendance Outcome Eligible", "School-attendance outcome eligibility", "sample", "binary", "School-age person with an observed attendance response"),
        ("Years Attended School", "Years attended school", "outcome", "years", "Reported years attended"),
        ("Nominal Education Expenditure Riels", "Reported education expenditure", "outcome", "nominal riels", "Reported total education expenditure; comparable field begins in 2009"),
        ("Real 2021 Education Expenditure Riels", "Education expenditure at 2021 prices", "outcome", "2021 riels", "Nominal education expenditure multiplied by the annual education CPI deflator to the 2021 annual mean"),
        ("Education CPI Deflator to 2021", "Annual education CPI deflator to 2021", "deflator", "ratio", "2021 annual mean education CPI divided by the survey-year annual mean education CPI"),
        ("Person Survey Weight", "Person survey weight", "weight", "survey-weight units", "Wave-specific person weight, using embedded weights in later waves"),
        ("Climate Geography Resolution", "Spatial resolution of enhanced climate value", "quality", "category", "Commune, district, or province"),
        ("Climate Link Method", "Historical climate linkage method", "quality", "category", "Exact, unique-name, high-confidence fuzzy, or explicitly flagged fallback method"),
        ("Enhanced Annual Rainfall Anomaly Z", "Enhanced annual rainfall anomaly", "exposure", "standard deviations", "Commune rainfall after audited name repair, with district or province fallback explicitly flagged"),
        ("Enhanced May October Rainfall Anomaly Z", "Enhanced growing-season rainfall anomaly", "exposure", "standard deviations", "May-October rainfall anomaly at the recorded climate geography resolution"),
    ]
    return pd.DataFrame(rows, columns=["variable_name", "full_name", "role", "unit", "construction"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    processed = root / "data" / "processed"
    exp = root / "data" / "exp" / "data-preprocessing"

    household = pd.read_parquet(processed / "direction3_household_year_spine_preprocessed.parquet")
    size, person_demographics = household_size_and_demographics(processed)
    household = household.merge(size, on=HOUSEHOLD_KEYS, how="left", validate="one_to_one")
    for part in [
        household_weights(processed),
        land_outcomes(processed),
        crop_outcomes(processed),
        crop_cost_outcomes(processed),
        food_consumption_outcomes(processed),
        food_security_outcomes(processed),
    ]:
        household = household.merge(part, on=HOUSEHOLD_KEYS, how="left", validate="one_to_one")

    household["Food Consumption Value per Household Member Riels"] = household[
        "Reported Food Consumption Value Riels"
    ] / household["Household Size"].replace(0, np.nan)
    household["Agricultural Input Cost per Cultivated ha Riels"] = (
        household["Nominal Agricultural Input Cost Riels"] * 10000
        / household["Cultivated Crop Area m2"].replace(0, np.nan)
    )
    household["Agricultural Household"] = (
        household["Total Parcel Area m2"].fillna(0).gt(0)
        | household["Any Crop Production Record"].fillna(False).astype(bool)
    ).astype("boolean")
    if household["Agricultural Household"].isna().any():
        raise ValueError("Agricultural Household must be a complete binary indicator")
    household = household.merge(enhanced_climate(processed), on=["Survey Wave", "PSU"], how="left", validate="many_to_one")
    household = add_real_household_values(household, processed)
    household = household.sort_values(["Survey Year", "PSU", "Household ID"]).reset_index(drop=True)
    household_path = processed / "direction3_household_core_outcomes_preprocessed.parquet"
    household.to_parquet(household_path, index=False)

    education = education_outcomes(processed, person_demographics, household)
    education_path = processed / "direction3_education_core_outcomes_preprocessed.parquet"
    education.to_parquet(education_path, index=False)

    dictionary = variable_dictionary()
    dictionary.to_csv(exp / "direction3_core_outcome_dictionary.csv", index=False)

    household_validation = household.groupby("Survey Wave").agg(
        **{
            "Households": ("Household ID", "size"),
            "Geography Matched": ("Geography Link Matched", "sum"),
            "Enhanced Climate Matched": ("Enhanced Climate Link Matched", "sum"),
            "Parcel Area Observed": ("Total Parcel Area m2", "count"),
            "Crop Production Observed": ("Crop Production Quantity kg", "count"),
            "Food Consumption Observed": ("Reported Food Consumption Value Riels", "count"),
            "Food Security Observed": ("Any Severe Food Insecurity Experience", "count"),
            "Household Weight Observed": ("Household Survey Weight", "count"),
            "Real Food Consumption Observed": ("Real 2021 Reported Food Consumption Value Riels", "count"),
            "Real Agricultural Input Cost Observed": ("Real 2021 Agricultural Input Cost Riels", "count"),
            "Monthly Food CPI Rows": ("Food Monetary Deflation Method", lambda values: int(values.eq("monthly food CPI").sum())),
            "Annual Food CPI Fallback Rows": ("Food Monetary Deflation Method", lambda values: int(values.eq("annual food CPI fallback").sum())),
        }
    ).reset_index()
    education_validation = education.groupby("Survey Wave").agg(
        **{
            "Education Person Rows": ("Person ID", "size"),
            "School Age Rows": ("School Age 6 to 17", "sum"),
            "Attendance Outcome Observed": ("Currently Attending School", "count"),
            "Eligible Attendance Outcomes": ("School Attendance Outcome Eligible", "sum"),
            "Person Weight Observed": ("Person Survey Weight", "count"),
            "Real Education Expenditure Observed": ("Real 2021 Education Expenditure Riels", "count"),
        }
    ).reset_index()
    validation = household_validation.merge(education_validation, on="Survey Wave", how="outer")
    validation.to_csv(exp / "direction3_core_outcome_validation.csv", index=False)

    print(f"household_rows={len(household):,}, columns={len(household.columns)}")
    print(f"education_rows={len(education):,}, columns={len(education.columns)}")
    print(f"dictionary_variables={len(dictionary)}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
