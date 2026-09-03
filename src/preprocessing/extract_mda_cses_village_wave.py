#!/usr/bin/env python3
"""Extract harmonized CSES modules from PostgreSQL and aggregate to PSU-wave.

The output keeps the survey's repeated-cross-section grain.  A PSU-wave is
treated as a surveyed village/community record; it is not treated as a stable
longitudinal village panel.  Village-questionnaire totals and survey-weighted
sample estimates are kept as separate, explicitly named fields.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path("data/processed/cses_mda_village_wave_preprocessed.parquet")
VILLAGE_OUTPUT = Path("data/processed/cses_mda_village_year_preprocessed.parquet")
AUDIT_DIR = Path("data/exp/data-preprocessing/cses-mda-village-wave")
CPI = Path("data/processed/cambodia_cpi_annual_preprocessed.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", default="mda")
    return parser.parse_args()


def weighted_mean(value: str, weight: str) -> str:
    valid = f"{value} IS NOT NULL AND {weight} IS NOT NULL AND {weight} > 0"
    return (
        f"sum(({value})::double precision * {weight}) FILTER (WHERE {valid}) "
        f"/ nullif(sum({weight}) FILTER (WHERE {valid}), 0)"
    )


def query() -> str:
    hh_weight = "household_weight"
    person_weight = "coalesce(person_weight, household_weight)"
    hh = {
        "Weighted Mean Household Size": weighted_mean("household_member_count", hh_weight),
        "Weighted Female Household Member Share": (
            "sum(household_weight * female_member_count) FILTER (WHERE household_weight > 0 "
            "AND female_member_count IS NOT NULL) / nullif(sum(household_weight * "
            "household_member_count) FILTER (WHERE household_weight > 0 AND "
            "female_member_count IS NOT NULL AND household_member_count > 0), 0)"
        ),
        "Weighted Child Household Member Share 0-14": (
            "sum(household_weight * child_member_count_0_14) FILTER (WHERE household_weight > 0 "
            "AND child_member_count_0_14 IS NOT NULL) / nullif(sum(household_weight * "
            "household_member_count) FILTER (WHERE household_weight > 0 AND "
            "child_member_count_0_14 IS NOT NULL AND household_member_count > 0), 0)"
        ),
        "Weighted Working Age Household Member Share 15-64": (
            "sum(household_weight * working_age_member_count_15_64) FILTER (WHERE household_weight > 0 "
            "AND working_age_member_count_15_64 IS NOT NULL) / nullif(sum(household_weight * "
            "household_member_count) FILTER (WHERE household_weight > 0 AND "
            "working_age_member_count_15_64 IS NOT NULL AND household_member_count > 0), 0)"
        ),
        "Weighted Older Household Member Share 65 Plus": (
            "sum(household_weight * older_member_count_65_plus) FILTER (WHERE household_weight > 0 "
            "AND older_member_count_65_plus IS NOT NULL) / nullif(sum(household_weight * "
            "household_member_count) FILTER (WHERE household_weight > 0 AND "
            "older_member_count_65_plus IS NOT NULL AND household_member_count > 0), 0)"
        ),
        "Weighted Female Household Head Share": weighted_mean(
            "case when household_head_sex=2 then 1.0 when household_head_sex=1 then 0.0 end",
            hh_weight,
        ),
        "Weighted Mean Household Head Age": weighted_mean("household_head_age", hh_weight),
        "Weighted Mean Household Head Years Schooling": weighted_mean(
            "household_head_years_attended_school", hh_weight
        ),
        "Weighted Literate Household Head Share": weighted_mean(
            "case when household_head_can_read=1 and household_head_can_write=1 then 1.0 "
            "when household_head_can_read is not null and household_head_can_write is not null "
            "then 0.0 end",
            hh_weight,
        ),
    }
    hl = {
        "Weighted Mean Person Age": weighted_mean("age", person_weight),
        "Weighted Female Person Share": weighted_mean(
            "case when sex=2 then 1.0 when sex=1 then 0.0 end", person_weight
        ),
        "Weighted Child Person Share 0-14": weighted_mean(
            "case when age between 0 and 14 then 1.0 when age is not null then 0.0 end",
            person_weight,
        ),
        "Weighted Working Age Person Share 15-64": weighted_mean(
            "case when age between 15 and 64 then 1.0 when age is not null then 0.0 end",
            person_weight,
        ),
        "Weighted Older Person Share 65 Plus": weighted_mean(
            "case when age>=65 then 1.0 when age is not null then 0.0 end", person_weight
        ),
        "Weighted Absent Household Member Share": weighted_mean(
            "absent_from_household", person_weight
        ),
    }
    ed = {
        "Weighted Literate Person Share": weighted_mean(
            "case when can_read=1 and can_write=1 then 1.0 "
            "when can_read is not null and can_write is not null then 0.0 end",
            person_weight,
        ),
        "Weighted Mean Years Attended School": weighted_mean(
            "years_attended_school", person_weight
        ),
        "Weighted Current School Attendance Share Age 6-17": weighted_mean(
            "case when age between 6 and 17 then currently_attending_school::double precision end",
            person_weight,
        ),
    }
    ec = {
        "Weighted Worked Past 7 Days Share Age 15-64": weighted_mean(
            "case when age between 15 and 64 then worked_at_least_one_hour_past_7_days::double precision end",
            person_weight,
        ),
        "Weighted Mean Weekly Work Hours Age 15-64": weighted_mean(
            "case when age between 15 and 64 then total_hours_worked_past_7_days::double precision end",
            person_weight,
        ),
        "Weighted Mean Monthly Salary Wages Riel": weighted_mean(
            "monthly_salary_wages_riel", person_weight
        ),
        "Weighted Actively Seeking Work Share Age 15-64": weighted_mean(
            "case when age between 15 and 64 then actively_seeking_work::double precision end",
            person_weight,
        ),
    }
    ho = {
        "Weighted Mean Floor Area Square Meters": weighted_mean(
            "floor_area_square_meters", hh_weight
        ),
        "Weighted Mean Rooms Used": weighted_mean("rooms_used", hh_weight),
        "Weighted Household Toilet Facility Share": weighted_mean(
            "has_toilet_facility", hh_weight
        ),
        "Weighted Household Drinking Water Treatment Share": weighted_mean(
            "treats_drinking_water", hh_weight
        ),
        "Weighted Mean Monthly Water Charges Riel": weighted_mean(
            "monthly_water_charges_riel", hh_weight
        ),
        "Weighted Mean Monthly Electricity Expense Riel": weighted_mean(
            "monthly_electricity_expense_riel", hh_weight
        ),
        "Weighted Mean Monthly Total Energy Expense Riel": weighted_mean(
            "coalesce(monthly_electricity_expense_riel,0)+coalesce(monthly_gas_expense_riel,0)+"
            "coalesce(monthly_kerosene_expense_riel,0)+coalesce(monthly_firewood_expense_riel,0)+"
            "coalesce(monthly_charcoal_expense_riel,0)+coalesce(monthly_battery_expense_riel,0)+"
            "coalesce(monthly_other_energy_expense_riel,0)",
            hh_weight,
        ),
        "Weighted Mean Monthly Rent Paid Riel": weighted_mean(
            "monthly_rent_paid_riel", hh_weight
        ),
        "Weighted Mean Monthly Imputed Rent Riel": weighted_mean(
            "monthly_imputed_rent_riel", hh_weight
        ),
    }

    def selections(items: dict[str, str]) -> str:
        return ",\n        ".join(f'{expression} AS "{name}"' for name, expression in items.items())

    return f"""
WITH hh AS (
    SELECT survey_wave, survey_year, psu,
        min(province_code) AS province_code,
        min(district_code) AS district_code,
        min(commune_code) AS commune_code,
        min(village_code) AS village_code,
        min(urban_rural) AS urban_rural,
        min(survey_month) AS survey_month_min,
        max(survey_month) AS survey_month_max,
        count(*) AS sample_households_database,
        count(household_weight) AS households_with_weight,
        sum(household_weight) FILTER (WHERE household_weight > 0) AS household_weight_sum,
        {selections(hh)}
    FROM "final_HH_CSES"
    GROUP BY survey_wave, survey_year, psu
),
hl AS (
    SELECT survey_wave, survey_year, psu,
        count(*) AS sample_people_database,
        count({person_weight}) AS people_with_weight,
        sum({person_weight}) FILTER (WHERE {person_weight} > 0) AS person_weight_sum,
        {selections(hl)}
    FROM "final_HL_CSES"
    GROUP BY survey_wave, survey_year, psu
),
ed AS (
    SELECT survey_wave, survey_year, psu,
        count(*) AS education_records_database,
        count(*) FILTER (WHERE age BETWEEN 6 AND 17 AND currently_attending_school IS NOT NULL)
            AS school_age_attendance_observations,
        {selections(ed)}
    FROM "final_ED_CSES"
    GROUP BY survey_wave, survey_year, psu
),
ec AS (
    SELECT survey_wave, survey_year, psu,
        count(*) AS employment_records_database,
        count(monthly_salary_wages_riel) AS observed_salary_records,
        {selections(ec)}
    FROM "final_EC_CSES"
    GROUP BY survey_wave, survey_year, psu
),
ho AS (
    SELECT survey_wave, survey_year, psu,
        count(*) AS housing_records_database,
        {selections(ho)}
    FROM "final_HO_CSES"
    GROUP BY survey_wave, survey_year, psu
),
vl AS (
    SELECT survey_wave, survey_year, psu,
        province_name, district_name, commune_name, village_name,
        sample_household_count, sample_person_count,
        village_reference_day, village_reference_month, village_reference_year,
        village_household_count, enumeration_area_count,
        households_in_surveyed_enumeration_area,
        village_person_count, village_male_count, village_female_count,
        population_below_18_count, boys_below_18_count, girls_below_18_count,
        population_18_plus_count, men_18_plus_count, women_18_plus_count,
        village_land_area_square_kilometers,
        five_year_population_movement_source_code,
        village_household_count_five_years_ago,
        village_person_count_five_years_ago
    FROM "final_VL_CSES"
)
SELECT
    hh.survey_year AS "Survey Year",
    hh.survey_wave AS "Survey Wave",
    hh.psu AS "PSU",
    hh.province_code AS "Province Code Component",
    hh.district_code AS "District Code Component",
    hh.commune_code AS "Commune Code Component",
    hh.village_code AS "Village Code Component",
    vl.province_name AS "Province Name",
    vl.district_name AS "District Name",
    vl.commune_name AS "Commune Name",
    vl.village_name AS "Village Name",
    hh.urban_rural AS "Urban Rural",
    hh.survey_month_min AS "Survey Month",
    (hh.survey_month_min IS NOT DISTINCT FROM hh.survey_month_max)::int AS "Single Survey Month Within PSU",
    hh.sample_households_database AS "Sample Households Database",
    hh.households_with_weight AS "Households With Survey Weight",
    hh.household_weight_sum AS "Household Survey Weight Sum",
    hl.sample_people_database AS "Sample People Database",
    hl.people_with_weight AS "People With Survey Weight",
    hl.person_weight_sum AS "Person Survey Weight Sum",
    ed.education_records_database AS "Education Records Database",
    ed.school_age_attendance_observations AS "School Age Attendance Observations",
    ec.employment_records_database AS "Employment Records Database",
    ec.observed_salary_records AS "Observed Salary Records",
    ho.housing_records_database AS "Housing Records Database",
    {', '.join(f'hh."{name}"' for name in hh)},
    {', '.join(f'hl."{name}"' for name in hl)},
    {', '.join(f'ed."{name}"' for name in ed)},
    {', '.join(f'ec."{name}"' for name in ec)},
    {', '.join(f'ho."{name}"' for name in ho)},
    vl.sample_household_count AS "Village Questionnaire Sample Household Count",
    vl.sample_person_count AS "Village Questionnaire Sample Person Count",
    vl.village_reference_day AS "Village Reference Day",
    vl.village_reference_month AS "Village Reference Month",
    vl.village_reference_year AS "Village Reference Year",
    vl.village_household_count AS "Reported Village Household Count",
    vl.enumeration_area_count AS "Reported Village Enumeration Area Count",
    vl.households_in_surveyed_enumeration_area AS "Reported Households in Surveyed Enumeration Area",
    vl.village_person_count AS "Reported Village Person Count",
    vl.village_male_count AS "Reported Village Male Count",
    vl.village_female_count AS "Reported Village Female Count",
    vl.population_below_18_count AS "Reported Village Population Below 18 Count",
    vl.boys_below_18_count AS "Reported Village Boys Below 18 Count",
    vl.girls_below_18_count AS "Reported Village Girls Below 18 Count",
    vl.population_18_plus_count AS "Reported Village Population 18 Plus Count",
    vl.men_18_plus_count AS "Reported Village Men 18 Plus Count",
    vl.women_18_plus_count AS "Reported Village Women 18 Plus Count",
    vl.village_land_area_square_kilometers AS "Reported Village Land Area Square Kilometers",
    vl.five_year_population_movement_source_code AS "Five Year Population Movement Source Code",
    vl.village_household_count_five_years_ago AS "Reported Village Household Count Five Years Ago",
    vl.village_person_count_five_years_ago AS "Reported Village Person Count Five Years Ago"
FROM hh
LEFT JOIN hl USING (survey_wave, survey_year, psu)
LEFT JOIN ed USING (survey_wave, survey_year, psu)
LEFT JOIN ec USING (survey_wave, survey_year, psu)
LEFT JOIN ho USING (survey_wave, survey_year, psu)
LEFT JOIN vl USING (survey_wave, survey_year, psu)
ORDER BY hh.survey_year, hh.psu
"""


def run_psql(sql: str, host: str, port: int, database: str) -> pd.DataFrame:
    command = [
        "psql", "-h", host, "-p", str(port), "-d", database,
        "-v", "ON_ERROR_STOP=1", "-q", "-c", f"COPY ({sql}) TO STDOUT WITH CSV HEADER",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    return pd.read_csv(io.BytesIO(result.stdout), low_memory=False)


def safe_code(frame: pd.DataFrame, column: str, width: int = 2) -> pd.Series:
    value = frame[column].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return value.where(value.notna() & value.ne("")).str.zfill(width)


def add_constructed_variables(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    components = [
        "Province Code Component", "District Code Component",
        "Commune Code Component", "Village Code Component",
    ]
    for column in components:
        frame[column] = safe_code(frame, column)
    complete = frame[components].notna().all(axis=1)
    frame["Commune Code"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["Village Code"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame.loc[complete, "Commune Code"] = frame.loc[complete, components[:3]].agg("".join, axis=1)
    frame.loc[complete, "Village Code"] = frame.loc[complete, components].agg("".join, axis=1)

    denominator = pd.to_numeric(frame["Reported Village Person Count"], errors="coerce")
    female_count = pd.to_numeric(frame["Reported Village Female Count"], errors="coerce")
    below_18_count = pd.to_numeric(
        frame["Reported Village Population Below 18 Count"], errors="coerce"
    )
    female_valid = denominator.gt(0) & female_count.between(0, denominator)
    below_18_valid = denominator.gt(0) & below_18_count.between(0, denominator)
    frame["Reported Village Female Count Plausible"] = female_valid.where(
        denominator.notna() & female_count.notna()
    ).astype("boolean")
    frame["Reported Village Population Below 18 Count Plausible"] = below_18_valid.where(
        denominator.notna() & below_18_count.notna()
    ).astype("boolean")
    frame["Reported Village Female Share"] = (female_count / denominator).where(female_valid)
    frame["Reported Village Population Below 18 Share"] = (
        below_18_count / denominator
    ).where(below_18_valid)
    previous = pd.to_numeric(
        frame["Reported Village Person Count Five Years Ago"], errors="coerce"
    )
    frame["Reported Five Year Village Population Growth Rate"] = (
        denominator / previous - 1
    ).where(previous > 0)

    cpi = pd.read_parquet(root / CPI)
    cpi = cpi.loc[cpi["CPI Component Code"].eq("_T"), ["Year", "Annual Deflator to 2021"]]
    cpi = cpi.rename(columns={"Year": "Survey Year"})
    frame = frame.merge(cpi, on="Survey Year", how="left", validate="many_to_one")
    nominal = [column for column in frame.columns if column.endswith(" Riel")]
    for column in nominal:
        real_name = column.removesuffix(" Riel") + " Real 2021 Riel"
        frame[real_name] = (
            pd.to_numeric(frame[column], errors="coerce") * frame["Annual Deflator to 2021"]
        )
    frame = frame.drop(columns=nominal)
    return frame


def first_observed(series: pd.Series) -> object:
    observed = series.dropna()
    return observed.iloc[0] if len(observed) else pd.NA


def collapse_to_village_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the small number of within-wave multi-PSU villages correctly."""
    source = frame.loc[frame["Village Code"].notna()].copy()
    keys = ["Survey Year", "Survey Wave", "Village Code"]
    grouped = source.groupby(keys, observed=True, sort=True, dropna=False)
    village = grouped.size().rename("PSU Count").to_frame()
    village["PSUs"] = grouped["PSU"].agg(
        lambda values: " | ".join(sorted(set(values.dropna().astype(str))))
    )

    reference = [
        "Province Code Component", "District Code Component", "Commune Code Component",
        "Village Code Component", "Commune Code", "Province Name", "District Name",
        "Commune Name", "Village Name", "Urban Rural", "Annual Deflator to 2021",
    ]
    for column in reference:
        village[column] = grouped[column].agg(first_observed)
    village["Survey Month Minimum"] = grouped["Survey Month"].min()
    village["Survey Month Maximum"] = grouped["Survey Month"].max()
    village["Single Survey Month Within Village"] = (
        village["Survey Month Minimum"].eq(village["Survey Month Maximum"])
    ).astype("Int8")

    additive = [
        column for column in source.columns
        if column.startswith("Sample ")
        or column.endswith(" Records Database")
        or column.endswith(" Observations")
        or column in {
            "Households With Survey Weight", "Household Survey Weight Sum",
            "People With Survey Weight", "Person Survey Weight Sum",
        }
    ]
    for column in additive:
        village[column] = grouped[column].sum(min_count=1)

    person_weighted_markers = (
        "Person", "School Attendance", "Years Attended School", "Worked Past",
        "Work Hours", "Salary Wages", "Seeking Work",
    )
    weighted = [column for column in source.columns if column.startswith("Weighted ")]
    group_index = pd.MultiIndex.from_frame(source[keys])
    for column in weighted:
        weight_column = (
            "Person Survey Weight Sum"
            if any(marker in column for marker in person_weighted_markers)
            else "Household Survey Weight Sum"
        )
        values = pd.to_numeric(source[column], errors="coerce")
        weights = pd.to_numeric(source[weight_column], errors="coerce")
        valid = values.notna() & weights.gt(0)
        numerator = (values.where(valid) * weights.where(valid)).groupby(group_index).sum(min_count=1)
        denominator = weights.where(valid).groupby(group_index).sum(min_count=1)
        village[column] = numerator / denominator

    reported = [
        column for column in source.columns
        if column.startswith("Village Questionnaire ")
        or column.startswith("Reported Village ")
        or column.startswith("Reported Five Year ")
        or column == "Five Year Population Movement Source Code"
    ]
    reported = [
        column for column in reported
        if not column.endswith(" Share")
        and not column.endswith(" Growth Rate")
        and not column.endswith(" Plausible")
    ]
    for column in reported:
        numeric = pd.to_numeric(source[column], errors="coerce")
        village[column] = numeric.groupby(group_index).median()

    village = village.reset_index()
    denominator = pd.to_numeric(village["Reported Village Person Count"], errors="coerce")
    female = pd.to_numeric(village["Reported Village Female Count"], errors="coerce")
    below_18 = pd.to_numeric(
        village["Reported Village Population Below 18 Count"], errors="coerce"
    )
    female_valid = denominator.gt(0) & female.between(0, denominator)
    below_18_valid = denominator.gt(0) & below_18.between(0, denominator)
    village["Reported Village Female Count Plausible"] = female_valid.where(
        denominator.notna() & female.notna()
    ).astype("boolean")
    village["Reported Village Population Below 18 Count Plausible"] = below_18_valid.where(
        denominator.notna() & below_18.notna()
    ).astype("boolean")
    village["Reported Village Female Share"] = (female / denominator).where(female_valid)
    village["Reported Village Population Below 18 Share"] = (
        below_18 / denominator
    ).where(below_18_valid)
    previous = pd.to_numeric(
        village["Reported Village Person Count Five Years Ago"], errors="coerce"
    )
    village["Reported Five Year Village Population Growth Rate"] = (
        denominator / previous - 1
    ).where(previous > 0)
    return village.sort_values(keys).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / OUTPUT
    audit_dir = root / AUDIT_DIR
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    frame = add_constructed_variables(
        run_psql(query(), args.host, args.port, args.database), root
    )
    if frame.duplicated(["Survey Wave", "PSU"]).any():
        raise RuntimeError("CSES PSU-wave key is not unique after aggregation")
    frame.to_parquet(output, index=False)
    village = collapse_to_village_year(frame)
    if village.duplicated(["Survey Wave", "Village Code"]).any():
        raise RuntimeError("CSES village-year key is not unique after collapsing PSUs")
    village.to_parquet(root / VILLAGE_OUTPUT, index=False)

    coverage = frame.groupby("Survey Year", observed=True).agg(
        **{
            "PSU-Wave Rows": ("PSU", "size"),
            "Rows With Full Village Code": ("Village Code", "count"),
            "Rows With Village Questionnaire": ("Reported Village Person Count", "count"),
            "Household Sample Records": ("Sample Households Database", "sum"),
            "Person Sample Records": ("Sample People Database", "sum"),
            "Rows With Education Estimate": ("Weighted Current School Attendance Share Age 6-17", "count"),
            "Rows With Employment Estimate": ("Weighted Worked Past 7 Days Share Age 15-64", "count"),
            "Rows With Housing Estimate": ("Weighted Household Toilet Facility Share", "count"),
        }
    ).reset_index()
    coverage.to_csv(audit_dir / "coverage_by_survey_year.csv", index=False)
    variable_dictionary = pd.DataFrame(
        {
            "Variable": frame.columns,
            "Data Type": [str(frame[column].dtype) for column in frame.columns],
            "Missing Percent": [float(frame[column].isna().mean() * 100) for column in frame.columns],
            "Construction": [
                "CSES village questionnaire report" if column.startswith("Reported Village")
                else "Survey-weighted aggregation from harmonized CSES microdata"
                if column.startswith("Weighted")
                else "Database linkage, sample-size, or constructed reference field"
                for column in frame.columns
            ],
        }
    )
    variable_dictionary.to_csv(audit_dir / "variable_dictionary.csv", index=False)
    metadata = {
        "source": "PostgreSQL mda public final_VL/HH/HL/ED/EC/HO_CSES",
        "database_host": args.host,
        "database_port": args.port,
        "database_name": args.database,
        "output": str(OUTPUT),
        "village_output": str(VILLAGE_OUTPUT),
        "grain": "one CSES PSU by survey wave",
        "rows": int(len(frame)),
        "survey_years": sorted(frame["Survey Year"].dropna().astype(int).unique().tolist()),
        "unique_village_codes": int(frame["Village Code"].nunique()),
        "village_year_rows": int(len(village)),
        "rules": [
            "No missing value imputation",
            "No winsorization",
            "Internally impossible village-questionnaire component shares remain missing and are flagged; source counts remain unchanged",
            "Survey-weighted sample estimates are distinct from village-questionnaire totals",
            "Monetary aggregates use the all-items annual CPI deflator to 2021 riel",
            "CSES is a repeated cross-section and is not declared a stable village panel",
        ],
    }
    (audit_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {OUTPUT}; rows={len(frame):,}; columns={len(frame.columns):,}")
    print(f"Saved {VILLAGE_OUTPUT}; rows={len(village):,}; columns={len(village.columns):,}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
