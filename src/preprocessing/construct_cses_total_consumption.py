#!/usr/bin/env python3
"""Construct annual real per-capita CSES consumption outcomes.

The construction follows the recall-period logic documented in the CSES
questionnaires.  Food is reported for the prior seven days and is annualized
by 52.  Non-food items are annualized item by item.  Housing services are
added separately after 2007 because the 2007 non-food instrument already
contains rent, water, and fuel.  The 2019 and 2021 instruments move education
out of the recall non-food module, so household education expenditure is
added from the person-level education module in those waves.

No outlier trimming or missing-value imputation is performed.  CSES 2004 is
retained as a nominal diagnostic but is ineligible for the real-price main
outcome because the available CPI series and imputed-rent construction do not
support a comparable value.
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["Survey Wave", "Household ID"]
OUTPUT = Path("data/processed/cses_household_total_consumption_preprocessed.parquet")
AUDIT = Path("data/exp/data-preprocessing/climate-npp-two-stage/consumption")


HOUSING_SOURCES: dict[str, tuple[str, tuple[str, ...]]] = {
    "2004": (
        "data/raw/CSE/CSES 2004.zip",
        ("CSES 2004/Stata 2004/2004hh_s03_housing.dta",),
    ),
    "2009": (
        "data/raw/CSE/CSES 2009.zip",
        ("CSES 2009/Stata CSES09/hhhousing.dta",),
    ),
    "2011-12": (
        "data/raw/CSE/CSES 2011-12.zip",
        ("CSES 2011-12/11housing.dta",),
    ),
    "2013": (
        "data/raw/CSE/CSES2013.zip",
        ("CSES2013/CSES2013/CSES 2013.zip", "HHHousing.dta"),
    ),
    "2014": (
        "data/raw/CSE/CSES 2014.zip",
        ("CSES 2014/CSES2014/hh housing.dta",),
    ),
    "2016": (
        "data/raw/CSE/CSES2016.zip",
        ("CSES2016/CSES2016/hhhousing.dta",),
    ),
    "2017": (
        "data/raw/CSE/CSES2017.zip",
        ("CSES2017/CSES2017/2017hh_s04_housing.dta",),
    ),
    "2019": (
        "data/raw/CSE/CSES2019.zip",
        ("CSES2019/S04_hhhousing.dta",),
    ),
    "2021": (
        "data/raw/CSE/Data of CSES2021.zip",
        ("Data of CSES2021/S04_HHhousing.dta",),
    ),
}


def numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("float64")


def canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [str(column).lower() for column in output.columns]
    return output


def normalize_hhid(values: pd.Series, wave: str) -> pd.Series:
    width = 6 if wave == "2004" else 7
    result = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return result.replace({"": pd.NA, "nan": pd.NA}).str.zfill(width)


def read_nested_stata(root: Path, archive: str, members: tuple[str, ...]) -> pd.DataFrame:
    with zipfile.ZipFile(root / archive) as outer:
        payload = outer.read(members[0])
    for member in members[1:]:
        with zipfile.ZipFile(io.BytesIO(payload)) as inner:
            payload = inner.read(member)
    return canonical_columns(pd.read_stata(io.BytesIO(payload), convert_categoricals=False))


def annualization_factor(wave: str, code: int) -> float:
    if wave == "2004":
        return 2.0 if code in {1, 2} else 1.0
    if wave == "2007":
        if 1 <= code <= 8:
            return 12.0
        if code == 9:
            return 2.0
        return 1.0
    if wave in {"2009", "2011-12", "2013"}:
        if 1 <= code <= 4:
            return 12.0
        if code == 5:
            return 2.0
        return 1.0
    if wave in {"2014", "2016", "2017"}:
        if code in {1, 2, 4, 5, 6, 8}:
            return 12.0
        if code == 9:
            return 2.0
        return 1.0
    if wave in {"2019", "2021"}:
        one_month = {8, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 31}
        three_month = {9, 10}
        six_month = {1, 2, 3, 4, 5, 6, 11, 12, 24, 25, 26, 27, 32, 33}
        if code in one_month:
            return 12.0
        if code in three_month:
            return 4.0
        if code in six_month:
            return 2.0
        return 1.0
    raise ValueError(f"Unsupported survey wave: {wave}")


def nonfood_component(processed: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(processed / "cses_nonfood_consumption_preprocessed.parquet")
    frame["Item Code"] = numeric(
        frame["Q07B01 serial number"].combine_first(frame["Q01CC01 NON FOOD ITEMS"])
    )
    reported_total = numeric(
        frame["Q07B05 total expenditure riels"].combine_first(
            frame["Q01CC06 Total expenditure Col 4 Col 5"]
        )
    )
    cash = numeric(
        frame["Q07B03 value in cash expenditure riels"].combine_first(
            frame["Q01CC04 In cash expenditure In Riels"]
        )
    )
    in_kind = numeric(
        frame["Q07B04 value in kind expenditure riels"].combine_first(
            frame["Q01CC05 In kind expenditure or gifts given away In Riels"]
        )
    )
    cash_kind_observed = cash.notna() | in_kind.notna()
    cash_kind = cash.fillna(0) + in_kind.fillna(0)
    frame["Reported Recall Nonfood Expenditure Riels"] = reported_total.where(
        reported_total.notna(), cash_kind.where(cash_kind_observed)
    )
    # In 2019, zero-valued item totals are stored as missing while cash and in-kind
    # values are explicit zeros.  Prefer their sum when the reported total is absent.
    frame.loc[
        frame["Reported Recall Nonfood Expenditure Riels"].isna() & cash_kind_observed,
        "Reported Recall Nonfood Expenditure Riels",
    ] = cash_kind
    frame = frame.loc[~(
        frame["Survey Wave"].isin(["2019", "2021"]) & frame["Item Code"].eq(41)
    )].copy()
    # A handful of source rows carry neither an item code nor an expenditure.
    # They cannot be annualized and are retained only in the upstream raw-preserving
    # module, not in this household consumption aggregation.
    frame = frame.loc[frame["Item Code"].notna()].copy()
    frame["Recall Period Annualization Factor"] = [
        annualization_factor(str(wave), int(code))
        for wave, code in zip(frame["Survey Wave"], frame["Item Code"], strict=True)
    ]
    frame["Annualized Recall Nonfood Expenditure Riels"] = (
        frame["Reported Recall Nonfood Expenditure Riels"]
        * frame["Recall Period Annualization Factor"]
    )
    if (frame["Annualized Recall Nonfood Expenditure Riels"].dropna() < 0).any():
        raise ValueError("Negative non-food expenditures found")
    household = frame.groupby(KEYS, dropna=False).agg(
        **{
            "Nominal Annual Recall Nonfood Expenditure Riels": (
                "Annualized Recall Nonfood Expenditure Riels",
                lambda values: values.sum(min_count=1),
            ),
            "Nonfood Item Rows Observed": (
                "Reported Recall Nonfood Expenditure Riels",
                "count",
            ),
            "Nonfood Distinct Item Codes Observed": ("Item Code", "nunique"),
        }
    ).reset_index()
    recall_audit = frame.groupby(["Survey Wave", "Item Code"], dropna=False).agg(
        **{
            "Rows": ("Household ID", "size"),
            "Observed Expenditure Rows": ("Reported Recall Nonfood Expenditure Riels", "count"),
            "Annualization Factor": ("Recall Period Annualization Factor", "first"),
        }
    ).reset_index()
    return household, recall_audit


def first_available(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype="float64")
    for name in names:
        if name in frame:
            result = result.combine_first(numeric(frame[name]))
    return result


def sum_available(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    pieces = [numeric(frame[name]) for name in names if name in frame]
    if not pieces:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.concat(pieces, axis=1).sum(axis=1, min_count=1)


def housing_component(root: Path) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for wave, (archive, members) in HOUSING_SOURCES.items():
        frame = read_nested_stata(root, archive, members)
        frame["Household ID"] = normalize_hhid(frame["hhid"], wave)
        if wave == "2004":
            rent = first_available(frame, ["q03_28"])
            utilities = sum_available(
                frame,
                ["q03_17", "q03_21", "q03_22", "q03_24a", "q03_24b", "q03_24c", "q03_24d", "q03_24e", "q03_24f", "q03_24g"],
            )
            maintenance = first_available(frame, ["q03_29"])
            rent_method = "actual rent only; imputed rent unavailable"
        elif wave in {"2009", "2011-12", "2013"}:
            rent = first_available(frame, ["q04_25a", "q04_25b"])
            utilities = sum_available(
                frame,
                ["q04_16", "q04_20", "q04_21", "q04_23a", "q04_23b", "q04_23c", "q04_23d", "q04_23e", "q04_23f", "q04_23g"],
            )
            maintenance = first_available(frame, ["q04_26"])
            rent_method = "actual rent for renters; estimated equivalent rent otherwise"
        else:
            rent = first_available(frame, ["q04_29a", "q04_29b"])
            utilities = sum_available(
                frame,
                ["q04_16", "q04_20", "q04_21", "q04_27a", "q04_27b", "q04_27c", "q04_27d", "q04_27e", "q04_27f", "q04_27g"],
            )
            # The 2019/2021 non-food item 28 already includes dwelling
            # insurance and maintenance.  Excluding q04_30 avoids duplication.
            maintenance = (
                pd.Series(0.0, index=frame.index)
                if wave in {"2019", "2021"}
                else first_available(frame, ["q04_30"])
            )
            rent_method = "actual rent for renters; estimated equivalent rent otherwise"
        monthly = pd.concat(
            [rent.rename("rent"), utilities.rename("utilities"), maintenance.rename("maintenance")],
            axis=1,
        ).sum(axis=1, min_count=1)
        output = pd.DataFrame(
            {
                "Survey Wave": wave,
                "Household ID": frame["Household ID"],
                "Nominal Annual Housing Services Riels": monthly * 12,
                "Housing Rent Component Monthly Riels": rent,
                "Housing Utilities Component Monthly Riels": utilities,
                "Housing Maintenance Component Monthly Riels": maintenance,
                "Housing Rent Construction": rent_method,
            }
        )
        if output.duplicated(KEYS).any():
            raise ValueError(f"Duplicate housing household keys in {wave}")
        pieces.append(output)
    return pd.concat(pieces, ignore_index=True)


def education_supplement(processed: Path) -> pd.DataFrame:
    education = pd.read_parquet(
        processed / "direction3_education_core_outcomes_preprocessed.parquet",
        columns=[
            "Survey Wave",
            "Household ID",
            "Currently Attending School",
            "Nominal Education Expenditure Riels",
            "Real 2021 Education Expenditure Riels",
        ],
    )
    education = education.loc[education["Survey Wave"].isin(["2019", "2021"])].copy()
    education["Attending"] = numeric(education["Currently Attending School"]).eq(1)
    grouped = education.groupby(KEYS, dropna=False).agg(
        **{
            "Household Members Currently Attending School": ("Attending", "sum"),
            "Education Expenditure Records": ("Nominal Education Expenditure Riels", "count"),
            "Nominal Annual Supplemental Education Expenditure Riels": (
                "Nominal Education Expenditure Riels",
                lambda values: values.sum(min_count=1),
            ),
            "Real 2021 Annual Supplemental Education Expenditure Riels": (
                "Real 2021 Education Expenditure Riels",
                lambda values: values.sum(min_count=1),
            ),
        }
    ).reset_index()
    no_attendance = grouped["Household Members Currently Attending School"].eq(0)
    grouped.loc[
        no_attendance & grouped["Nominal Annual Supplemental Education Expenditure Riels"].isna(),
        "Nominal Annual Supplemental Education Expenditure Riels",
    ] = 0.0
    grouped.loc[
        no_attendance & grouped["Real 2021 Annual Supplemental Education Expenditure Riels"].isna(),
        "Real 2021 Annual Supplemental Education Expenditure Riels",
    ] = 0.0
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    processed = root / "data/processed"
    audit = root / AUDIT
    audit.mkdir(parents=True, exist_ok=True)

    base = pd.read_parquet(processed / "direction3_household_core_outcomes_preprocessed.parquet")
    nonfood, recall_audit = nonfood_component(processed)
    housing = housing_component(root)
    education = education_supplement(processed)
    frame = base.merge(nonfood, on=KEYS, how="left", validate="one_to_one")
    frame = frame.merge(housing, on=KEYS, how="left", validate="one_to_one")
    frame = frame.merge(education, on=KEYS, how="left", validate="one_to_one")

    frame["Nominal Annual Food Consumption Riels"] = (
        numeric(frame["Reported Food Consumption Value Riels"]) * 52
    )
    frame["Real 2021 Annual Food Consumption Riels"] = (
        numeric(frame["Real 2021 Reported Food Consumption Value Riels"]) * 52
    )
    # Housing is already embedded in the 2007 recall non-food instrument.
    housing_needed = ~frame["Survey Wave"].eq("2007")
    frame.loc[~housing_needed, "Nominal Annual Housing Services Riels"] = 0.0
    frame.loc[~housing_needed, "Housing Rent Construction"] = (
        "housing, water, and fuel embedded in 2007 recall non-food items 1-4"
    )
    education_needed = frame["Survey Wave"].isin(["2019", "2021"])
    frame.loc[
        ~education_needed,
        "Nominal Annual Supplemental Education Expenditure Riels",
    ] = 0.0
    frame.loc[
        ~education_needed,
        "Real 2021 Annual Supplemental Education Expenditure Riels",
    ] = 0.0

    all_items_deflator = numeric(frame["All Items CPI Deflator to 2021"])
    frame["Real 2021 Annual Recall Nonfood Expenditure Riels"] = (
        numeric(frame["Nominal Annual Recall Nonfood Expenditure Riels"]) * all_items_deflator
    )
    frame["Real 2021 Annual Housing Services Riels"] = (
        numeric(frame["Nominal Annual Housing Services Riels"]) * all_items_deflator
    )
    nominal_components = [
        "Nominal Annual Food Consumption Riels",
        "Nominal Annual Recall Nonfood Expenditure Riels",
        "Nominal Annual Housing Services Riels",
        "Nominal Annual Supplemental Education Expenditure Riels",
    ]
    real_components = [
        "Real 2021 Annual Food Consumption Riels",
        "Real 2021 Annual Recall Nonfood Expenditure Riels",
        "Real 2021 Annual Housing Services Riels",
        "Real 2021 Annual Supplemental Education Expenditure Riels",
    ]
    frame["Nominal Annual Total Consumption Riels"] = frame[nominal_components].sum(
        axis=1, min_count=len(nominal_components)
    )
    frame["Real 2021 Annual Total Consumption Riels"] = frame[real_components].sum(
        axis=1, min_count=len(real_components)
    )
    household_size = numeric(frame["Household Size"]).replace(0, np.nan)
    frame["Real 2021 Annual Total Consumption per Capita Riels"] = (
        frame["Real 2021 Annual Total Consumption Riels"] / household_size
    )
    frame["Real 2021 Daily Total Consumption per Capita Riels"] = (
        frame["Real 2021 Annual Total Consumption per Capita Riels"] / 365.25
    )
    frame["Real 2021 Annual Food Consumption per Capita Riels"] = (
        frame["Real 2021 Annual Food Consumption Riels"] / household_size
    )
    frame["Real 2021 Daily Food Consumption per Capita Riels"] = (
        frame["Real 2021 Annual Food Consumption per Capita Riels"] / 365.25
    )
    frame["Log Real 2021 Annual Total Consumption per Capita"] = np.log(
        frame["Real 2021 Annual Total Consumption per Capita Riels"].where(
            frame["Real 2021 Annual Total Consumption per Capita Riels"].gt(0)
        )
    )
    frame["Log Real 2021 Annual Food Consumption per Capita"] = np.log(
        frame["Real 2021 Annual Food Consumption per Capita Riels"].where(
            frame["Real 2021 Annual Food Consumption per Capita Riels"].gt(0)
        )
    )
    frame["Total Consumption Instrument Regime"] = frame["Survey Wave"].map(
        {
            "2004": "2004 diagnostic",
            "2007": "2007 integrated housing recall",
            "2009": "2009-2013 recall plus housing",
            "2011-12": "2009-2013 recall plus housing",
            "2013": "2009-2013 recall plus housing",
            "2014": "2014-2017 expanded recall plus housing",
            "2016": "2014-2017 expanded recall plus housing",
            "2017": "2014-2017 expanded recall plus housing",
            "2019": "2019-2021 itemized recall plus housing and education",
            "2021": "2019-2021 itemized recall plus housing and education",
        }
    )
    component_complete = frame[real_components].notna().all(axis=1)
    frame["Total Consumption Main Outcome Eligible"] = (
        frame["Main Linked Sample"].fillna(False)
        & frame["Survey Wave"].ne("2004")
        & component_complete
        & frame["Real 2021 Annual Total Consumption per Capita Riels"].gt(0)
    )
    if frame.duplicated(KEYS).any():
        raise ValueError("Duplicate household-wave keys in final consumption release")
    if (
        frame.loc[frame["Total Consumption Main Outcome Eligible"], "Real 2021 Annual Total Consumption per Capita Riels"]
        <= 0
    ).any():
        raise ValueError("Non-positive eligible total consumption values")

    selected = [
        "Survey Year", "Survey Wave", "Household ID", "PSU", "Province Code",
        "District Code", "Commune Code", "Village Code", "Province Name",
        "District Name", "Commune Name", "Village Name", "Urban Rural",
        "Survey Month", "Main Linked Sample", "Geography Link Matched",
        "Household Size", "Household Survey Weight", "Food CPI Deflator to 2021 Used",
        "All Items CPI Deflator to 2021", "Nominal Annual Food Consumption Riels",
        "Nominal Annual Recall Nonfood Expenditure Riels", "Nominal Annual Housing Services Riels",
        "Nominal Annual Supplemental Education Expenditure Riels",
        "Nominal Annual Total Consumption Riels", "Real 2021 Annual Food Consumption Riels",
        "Real 2021 Annual Recall Nonfood Expenditure Riels", "Real 2021 Annual Housing Services Riels",
        "Real 2021 Annual Supplemental Education Expenditure Riels",
        "Real 2021 Annual Total Consumption Riels",
        "Real 2021 Annual Total Consumption per Capita Riels",
        "Real 2021 Daily Total Consumption per Capita Riels",
        "Real 2021 Annual Food Consumption per Capita Riels",
        "Real 2021 Daily Food Consumption per Capita Riels",
        "Log Real 2021 Annual Total Consumption per Capita",
        "Log Real 2021 Annual Food Consumption per Capita",
        "Nonfood Item Rows Observed", "Nonfood Distinct Item Codes Observed",
        "Housing Rent Construction", "Household Members Currently Attending School",
        "Education Expenditure Records", "Total Consumption Instrument Regime",
        "Total Consumption Main Outcome Eligible",
    ]
    output = frame[selected].sort_values(["Survey Year", "PSU", "Household ID"])
    destination = root / OUTPUT
    output.to_parquet(destination, index=False)

    coverage = output.groupby("Survey Wave", observed=True).agg(
        **{
            "Households": ("Household ID", "size"),
            "Nominal Total Observed": ("Nominal Annual Total Consumption Riels", "count"),
            "Real Total Observed": ("Real 2021 Annual Total Consumption Riels", "count"),
            "Main Outcome Eligible": ("Total Consumption Main Outcome Eligible", "sum"),
            "Real Per Capita Median": ("Real 2021 Annual Total Consumption per Capita Riels", "median"),
            "Real Per Capita P10": ("Real 2021 Annual Total Consumption per Capita Riels", lambda x: x.quantile(0.10)),
            "Real Per Capita P90": ("Real 2021 Annual Total Consumption per Capita Riels", lambda x: x.quantile(0.90)),
        }
    ).reset_index()
    coverage.to_csv(audit / "consumption_coverage_by_wave.csv", index=False)
    recall_audit.to_csv(audit / "nonfood_recall_annualization_audit.csv", index=False)
    dictionary = pd.DataFrame(
        [
            ["Real 2021 Annual Total Consumption per Capita Riels", "main outcome", "2021 riels/person/year", "Annualized food, recall non-food, housing services, and where required education; component-specific CPI deflation; divided by household size"],
            ["Real 2021 Annual Food Consumption per Capita Riels", "secondary outcome", "2021 riels/person/year", "Seven-day food consumption multiplied by 52, deflated with interview-month food CPI, divided by household size"],
            ["Heat Days at or Above 35 C", "stage-1 exposure", "days/year", "Constructed separately in the annual climate-shock release"],
        ],
        columns=["Variable", "Role", "Unit", "Construction"],
    )
    dictionary.to_csv(audit / "consumption_variable_dictionary.csv", index=False)
    metadata = {
        "output": str(OUTPUT),
        "grain": "one CSES household per survey wave",
        "food_annualization": "reported prior-seven-day value multiplied by 52",
        "nonfood_annualization": "item-specific questionnaire recall period",
        "housing": "monthly rent service, utilities, and non-duplicated maintenance multiplied by 12; already embedded in 2007 non-food",
        "education": "added from person education records only in 2019 and 2021 when absent from recall non-food",
        "real_price_base": "2021 riels",
        "main_outcome": "Real 2021 Annual Total Consumption per Capita Riels",
        "secondary_outcome": "Real 2021 Annual Food Consumption per Capita Riels",
        "main_eligible_waves": ["2007", "2009", "2011-12", "2013", "2014", "2016", "2017", "2019", "2021"],
        "outlier_treatment": "none",
        "imputation": "none; structural skips only converted to zero for the 2019/2021 education supplement when no household member attends school",
    }
    (audit / "consumption_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
