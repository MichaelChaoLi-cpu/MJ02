#!/usr/bin/env python3
"""Household and Estimation Variable Definitions.

Plan: Standalone Appendix dictionary for every household-level and commune
pseudo-panel variable used by Model B, including weights, fixed effects,
sample gates, composition sensitivities, and spatial clustering.
Framework: AnaSOP Sections 4-7 household and commune pseudo-panel models.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from _variable_definition_workbook import validate_variable_dictionary, write_variable_dictionary
from table_variable_definitions_and_harmonization import COLUMNS, record


ROOT = Path(__file__).resolve().parents[2]
HOUSEHOLD = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
PSEUDO = ROOT / "data/processed/cses_commune_survey_time_climate_food_pseudopanel_preprocessed.parquet"
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/tables/Table_household_and_estimation_variable_definitions.xlsx"
ANALYSIS = ROOT / "data/exp/analysis/climate-welfare/household-estimation-variable-definitions/table_rows.csv"

FOOD = "Real 2021 Food Consumption Value per Household Member Riels"
WEIGHT = "Household Survey Weight"
HEAT = "Last Complete Season Absolute Heat Day Count 35 C Candidate B"
RAIN = "Last Complete Season May-October Precipitation Anomaly Z"
ONSET = "Last Complete Season Wet-Season Onset Anomaly Z Candidate B"
DRY = "Last Complete Season Longest Dry Spell Anomaly Z Candidate B"


def missing_range(frame: pd.DataFrame, columns: list[str]) -> str:
    missing = frame[columns].isna().mean().mul(100)
    if len(columns) == 1 or missing.max() - missing.min() < 0.05:
        return f"{missing.mean():.1f}%"
    return f"{missing.min():.1f}–{missing.max():.1f}% across components"


def declared_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    household_columns = [
        "Climate Ecology Link Available", FOOD, WEIGHT, HEAT, RAIN, ONSET, DRY,
        "Household Size", "Female Household Member Share",
        "Mean Household Member Age Years", "Child Age 0-14 Share",
        "Older Age 65 Plus Share", "Household Dependency Ratio",
        "Agricultural Participation", "Urban Rural", "District Code",
        "Commune Code", "Village Code", "Survey Year", "Interview Month",
        "Point Longitude", "Point Latitude",
    ]
    household = pd.read_parquet(HOUSEHOLD, columns=household_columns)
    household = household.loc[household["Climate Ecology Link Available"].eq(1)].copy()
    required = [
        FOOD, WEIGHT, HEAT, RAIN, ONSET, DRY, "District Code", "Commune Code",
        "Village Code", "Survey Year", "Interview Month", "Point Longitude",
        "Point Latitude",
    ]
    complete = household[required].notna().all(axis=1)
    complete &= household[FOOD].gt(0) & household[WEIGHT].gt(0)
    household = household.loc[complete].copy()
    pseudo = pd.read_parquet(PSEUDO)
    assert len(household) == 46_445
    assert len(pseudo) == 3_922
    return household, pseudo


def build_table() -> pd.DataFrame:
    household, pseudo = declared_frames()
    household_n = f"46,445 eligible linked households"
    pseudo_n = "3,922 commune survey-time cells"
    composition = [
        "Household Size", "Female Household Member Share",
        "Mean Household Member Age Years", "Child Age 0-14 Share",
        "Older Age 65 Plus Share", "Household Dependency Ratio",
        "Agricultural Participation", "Urban Rural",
    ]
    pseudo_composition = [
        "Survey-Weighted Mean Household Size",
        "Survey-Weighted Mean Female Household Member Share",
        "Survey-Weighted Mean Household Member Age Years",
        "Survey-Weighted Mean Child Age 0-14 Share",
        "Survey-Weighted Mean Older Age 65 Plus Share",
        "Survey-Weighted Mean Household Dependency Ratio",
        "Survey-Weighted Agricultural Participation Share",
        "Survey-Weighted Urban Household Share",
    ]

    rows = [
        record(
            "Real 2021 Food Consumption per Household Member",
            "Reported household food-consumption value divided by household size and expressed in constant 2021 riels.",
            "2021 riels person⁻¹", "Household-wave", "2007–2021",
            "Interview month; household-member denominator",
            "Component-matched food CPI with documented annual fallback when monthly CPI is unavailable.",
            missing_range(household, [FOOD]), "Natural-unit welfare outcome; final",
            "Consumption value is not calories, nutrition, or individual intake.",
        ),
        record(
            "Log Real Food Consumption per Household Member",
            "Natural logarithm of positive real food consumption per household member.",
            "Log points", "Household-wave", "2007–2021",
            household_n,
            "Natural log; coefficients are translated as 100 × [exp(beta) − 1].",
            missing_range(household, [FOOD]), "Primary household outcome; final",
            "Percentage interpretation remains conditional on the fixed-effects design.",
        ),
        record(
            "Completed-Season Absolute Heat Days 35 C",
            "Post-onset days with daily maximum air temperature at or above 35°C in the last fully completed May–October season.",
            "Days; coefficient per 10 days", "Linked village-season / household", "2007–2021 interviews",
            "Last complete production season before interview",
            "Approved monsoon definition; 5 km village-buffer mean; divided by ten only in estimation.",
            missing_range(household, [HEAT]), "Primary household exposure; final",
            "An air-temperature count, not land-surface temperature or individual exposure.",
        ),
        record(
            "Completed-Season May-October Precipitation Anomaly",
            "Seasonal rainfall total relative to the local 1991–2020 distribution.",
            "Local SD", "Linked village-season / household", "2007–2021 interviews",
            "Last complete production season before interview",
            "5 km village-buffer mean of the local standardized anomaly.",
            missing_range(household, [RAIN]), "Household rainfall control; final",
            "Seasonal quantity does not measure rainfall timing or flood exposure.",
        ),
        record(
            "Completed-Season Wet-Season Onset Anomaly",
            "Sustained wet-season onset timing relative to the local 1991–2020 distribution.",
            "Local SD", "Linked village-season / household", "2007–2021 interviews",
            "Last complete production season before interview",
            "Approved 30 mm / 3-day onset rule; 5 km village-buffer mean.",
            missing_range(household, [ONSET]), "Household timing control; final",
            "A timing measure, not rainfall volume.",
        ),
        record(
            "Completed-Season Longest Intraseasonal Dry-Spell Anomaly",
            "Longest post-onset run below 1 mm rainfall relative to local climatology.",
            "Local SD", "Linked village-season / household", "2007–2021 interviews",
            "Last complete production season before interview",
            "Approved definition; 5 km village-buffer mean standardized to 1991–2020.",
            missing_range(household, [DRY]), "Household sequencing control; final",
            "Meteorological interruption, not hydrological drought.",
        ),
        record(
            "Household Composition Vector",
            "Household size, female-member share, mean age, child share, older-age share, dependency ratio, agricultural participation, and urban residence.",
            "Counts, shares, years, binary", "Household-wave", "2007–2021",
            household_n,
            "Entered jointly only in the declared composition sensitivity.",
            missing_range(household, composition), "Household composition sensitivity; final",
            "Contemporary composition may respond to climate and is excluded from the minimal model.",
        ),
        record(
            "Household Survey Weight",
            "Released CSES household sampling weight.",
            "Survey weight", "Household-wave", "2007–2021",
            household_n,
            "Used in all household regressions and weighted household coverage summaries.",
            missing_range(household, [WEIGHT]), "Primary household estimation weight; final",
            "Does not repair missing public-village linkage by itself.",
        ),
        record(
            "Linkage-Response Adjusted Survey Weight",
            "Survey weight multiplied by a stabilized inverse predicted public-point linkage probability.",
            "Stabilized weight", "Household-wave", "2007–2021",
            "Linked households in the selection sensitivity",
            "Outcome-blind regularized logistic response model; probabilities clipped to 0.02–0.98 and multipliers clipped at P1/P99.",
            "0.0% after construction", "Linkage-selection sensitivity; final",
            "Balances observed predictors only and cannot locate unlinked villages.",
        ),
        record(
            "Household Location Fixed-Effect Identifiers",
            "District Code, Commune Code, and Village Code attached to each linked household.",
            "Categories", "Household-wave", "2007–2021",
            household_n,
            "District and commune fixed effects use the full linked sample; village fixed effects use the repeated-village sensitivity sample.",
            missing_range(household, ["District Code", "Commune Code", "Village Code"]),
            "Declared location fixed effects; final",
            "Changing the area identifier changes identifying variation and sometimes the sample.",
        ),
        record(
            "Household Survey-Time Fixed-Effect ID",
            "Interaction of CSES survey wave and interview month used in household regressions.",
            "Category", "Household-wave", "2007–2021",
            household_n,
            "Constructed after restoring the 2019–2020 interview calendar and entered as a complete fixed-effect set.",
            "0.0% after construction", "Household survey-timing fixed effect; final",
            "Controls common survey timing but does not create a household panel.",
        ),
        record(
            "Repeated Village Indicator",
            "Indicator that a CSES village code appears in more than one survey wave.",
            "Binary", "Village / household", "2007–2021",
            household_n,
            "Constructed before splitting repeated- and single-wave-village samples and their formal heat-slope contrast.",
            "0.0% after construction", "Sample-structure diagnostic; final",
            "Repeated villages are not repeated households.",
        ),
        record(
            "Household Spatial Block ID",
            "Fixed 0.75-degree latitude-longitude block containing the linked public village point.",
            "Category", "Village / household", "Time invariant",
            household_n,
            "Constructed from fixed longitude and latitude bins for cluster-robust uncertainty.",
            "0.0% after construction", "Household inference cluster; final",
            "Thirty-seven household blocks limit fine-grained spatial-dependence modelling.",
        ),
        record(
            "Survey-Weighted Mean Log Real Food Consumption per Member",
            "Survey-weighted mean of the household log food outcome within each commune survey-time cell.",
            "Mean log points", "Commune survey-time cell", "2007–2021",
            pseudo_n,
            "Weighted within Commune Code × survey wave × interview calendar year-month.",
            missing_range(pseudo, ["Survey-Weighted Mean Log Real Food Consumption per Member"]),
            "Primary pseudo-panel outcome; final",
            "A cell mean from repeated cross-sections, not household-panel consumption.",
        ),
        record(
            "Survey-Weighted Mean Absolute Heat Days 35 C",
            "Survey-weighted cell mean of village-assigned completed-season absolute heat days.",
            "Days; coefficient per 10 days", "Commune survey-time cell", "2007–2021",
            pseudo_n,
            "Same approved 5 km household exposure, averaged with released survey weights; divided by ten in estimation.",
            missing_range(pseudo, ["Survey-Weighted Mean Absolute Heat Days 35 C"]),
            "Primary pseudo-panel exposure; final",
            "Within-cell household weights do not turn assigned village weather into individual exposure.",
        ),
        record(
            "Survey-Weighted Mean May-October Precipitation Anomaly",
            "Survey-weighted cell mean of the completed-season rainfall anomaly.",
            "Local SD", "Commune survey-time cell", "2007–2021",
            pseudo_n, "Weighted mean of the 5 km village-assigned household control.",
            missing_range(pseudo, ["Survey-Weighted Mean May-October Precipitation Anomaly Z"]),
            "Pseudo-panel rainfall control; final",
            "Does not capture within-season timing or floods.",
        ),
        record(
            "Survey-Weighted Mean Wet-Season Onset Anomaly",
            "Survey-weighted cell mean of the approved-definition onset anomaly.",
            "Local SD", "Commune survey-time cell", "2007–2021",
            pseudo_n, "Weighted mean of the 5 km village-assigned household control.",
            missing_range(pseudo, ["Survey-Weighted Mean Wet-Season Onset Anomaly Z"]),
            "Pseudo-panel timing control; final",
            "A timing control rather than rainfall quantity.",
        ),
        record(
            "Survey-Weighted Mean Longest Intraseasonal Dry-Spell Anomaly",
            "Survey-weighted cell mean of the approved-definition dry-spell anomaly.",
            "Local SD", "Commune survey-time cell", "2007–2021",
            pseudo_n, "Weighted mean of the 5 km village-assigned household control.",
            missing_range(pseudo, ["Survey-Weighted Mean Longest Intraseasonal Dry Spell Anomaly Z"]),
            "Pseudo-panel sequencing control; final",
            "A meteorological interruption rather than hydrological drought.",
        ),
        record(
            "Survey-Weighted Household Composition Vector",
            "Weighted cell means of household size, female share, mean age, child share, older-age share, dependency ratio, agricultural participation, and urban share.",
            "Mixed", "Commune survey-time cell", "2007–2021",
            pseudo_n,
            "Eight components entered jointly only in the composition-adjusted pseudo-panel sensitivity.",
            missing_range(pseudo, pseudo_composition), "Pseudo-panel composition sensitivity; final",
            "Composition adjustment is secondary because composition may itself respond to climate.",
        ),
        record(
            "Survey Weight Sum",
            "Sum of released household survey weights represented by a commune survey-time cell.",
            "Weight", "Commune survey-time cell", "2007–2021",
            pseudo_n, "Used as the analytic weight in all pseudo-panel regressions.",
            missing_range(pseudo, ["Survey Weight Sum"]), "Pseudo-panel estimation weight; final",
            "Large cells receive more weight; this does not remove remaining within-cell sampling error.",
        ),
        record(
            "Household Count and Cell-Size Gates",
            "Number of eligible linked households in the cell, with indicators for at least five and at least ten households.",
            "Count and binary", "Commune survey-time cell", "2007–2021",
            "3,922 cells; 3,884 pass five households and 3,840 pass ten",
            "Primary Minimum Five Households defines the main gate; Robustness Minimum Ten Households defines the stricter sensitivity.",
            missing_range(pseudo, ["Household Count", "Primary Minimum Five Households", "Robustness Minimum Ten Households"]),
            "Pseudo-panel sample gates; final",
            "After requiring repeated communes, the estimation samples contain 3,580 and 3,538 cells.",
        ),
        record(
            "Commune Code and Repeated Commune Indicator",
            "Stable commune identifier and indicator that the commune appears in at least two CSES waves.",
            "Category and binary", "Commune survey-time cell", "2007–2021",
            "1,298 communes overall; 994 repeated communes",
            "Commune Code is absorbed as the area fixed effect; only repeated communes enter the pseudo-panel estimates.",
            missing_range(pseudo, ["Commune Code", "Repeated Commune Indicator"]),
            "Pseudo-panel area support; final",
            "Repeated communes still contain different sampled households across waves.",
        ),
        record(
            "Pseudo-Panel Survey-Time and Spatial Block IDs",
            "Survey-wave × interview-calendar-year-month fixed-effect ID and the commune-reference 0.75-degree spatial block ID.",
            "Categories", "Commune survey-time cell", "2007–2021",
            pseudo_n,
            "Survey-Time Fixed-Effect ID absorbs common fieldwork timing; Spatial Block ID clusters uncertainty.",
            missing_range(pseudo, ["Survey-Time Fixed-Effect ID", "Spatial Block ID"]),
            "Pseudo-panel time fixed effect and inference cluster; final",
            "Thirty-six spatial blocks remain after the repeated-commune and primary cell gates.",
        ),
    ]
    table = pd.DataFrame(rows, columns=COLUMNS)
    assert len(table) == 23
    assert table["Readable variable name"].is_unique
    return table


def main() -> None:
    table = build_table()
    ANALYSIS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(ANALYSIS, index=False)
    write_variable_dictionary(
        table,
        title="Household and Estimation Variable Definitions",
        sheet_name="Household variables",
        output=OUTPUT,
        notes=[
            "Notes: Household models use released survey weights; pseudo-panel cell means and Survey Weight Sum use the same released weights.",
            "The pseudo-panel contains repeated cross-sections aggregated to stable commune survey-time cells; it is not a household panel.",
            "Climate exposure uses the last fully completed production season before interview and the approved 5 km monsoon definition.",
        ],
        fit_to_height=0,
        page_break_after_data_rows=[13],
    )
    validate_variable_dictionary(OUTPUT, sheet_name="Household variables", expected_rows=23)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Rows: {len(table)}; household rows: 13; pseudo-panel rows: 10")


if __name__ == "__main__":
    main()
