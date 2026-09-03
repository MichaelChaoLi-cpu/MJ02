#!/usr/bin/env python3
"""Audit whether local welfare outcomes can strengthen the boundary study.

This is a support and availability audit only.  It does not estimate historical
side coefficients or revise the research plan.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/exp/feasibility-check/local-welfare-extension"
DESIGN = ROOT / "data/processed/historical_boundary_design_preprocessed.parquet"
HOUSEHOLD = ROOT / "data/processed/direction3_household_core_outcomes_preprocessed.parquet"
EDUCATION = ROOT / "data/processed/direction3_education_core_outcomes_preprocessed.parquet"
MIGRATION = ROOT / "data/processed/cses_migration_preprocessed.parquet"


def normalized_psu(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["PSU"] = result["PSU"].astype("string").str.replace(r"\.0$", "", regex=True)
    return result


def outcome_row(
    frame: pd.DataFrame,
    domain: str,
    outcome: str,
    status: str,
    role: str,
    limitation: str,
) -> dict[str, object]:
    observed = frame[outcome].notna()
    return {
        "domain": domain,
        "outcome": outcome,
        "nonmissing_rows": int(observed.sum()),
        "psu_years": int(frame.loc[observed, ["Survey Year", "PSU"]].drop_duplicates().shape[0]),
        "survey_waves": int(frame.loc[observed, "Survey Year"].nunique()),
        "southwest_psu_years": int(
            frame.loc[
                observed & frame["Higher-Repression Southwest Zone"].eq(1),
                ["Survey Year", "PSU"],
            ].drop_duplicates().shape[0]
        ),
        "west_psu_years": int(
            frame.loc[
                observed & frame["Higher-Repression Southwest Zone"].eq(0),
                ["Survey Year", "PSU"],
            ].drop_duplicates().shape[0]
        ),
        "feasibility_status": status,
        "admissible_role": role,
        "binding_limitation": limitation,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    design = normalized_psu(pd.read_parquet(DESIGN))
    design = design.loc[design["Historical-Boundary Common Support 5 km"].eq(1)].copy()
    design_keys = [
        "Survey Year", "PSU", "Historical Repression Side",
        "Higher-Repression Southwest Zone",
        "Signed Distance to Historical Repression Boundary km",
        "Commune Code Normalized", "Public Village Point Link Method",
    ]
    if design.duplicated(["Survey Year", "PSU"]).any():
        raise ValueError("Boundary design is not unique by survey year and PSU")

    wave_support = (
        design.groupby(["Survey Year", "Historical Repression Side"], observed=True)
        .agg(
            psu_years=("PSU", "size"),
            unique_psus=("PSU", "nunique"),
            communes=("Commune Code Normalized", "nunique"),
        )
        .reset_index()
    )
    wave_support.to_csv(OUTPUT / "current_support_by_wave_and_side.csv", index=False)

    commune_year = (
        design.groupby(["Survey Year", "Commune Code Normalized"], observed=True)
        .agg(
            historical_sides=("Higher-Repression Southwest Zone", "nunique"),
            psu_years=("PSU", "size"),
        )
        .reset_index()
    )
    commune_year["cross_side_within_commune"] = commune_year["historical_sides"].eq(2)
    commune_year.to_csv(OUTPUT / "within_commune_shock_support.csv", index=False)

    household = normalized_psu(pd.read_parquet(HOUSEHOLD)).merge(
        design[design_keys], on=["Survey Year", "PSU"], how="inner", validate="many_to_one"
    )
    education = normalized_psu(pd.read_parquet(EDUCATION)).merge(
        design[design_keys], on=["Survey Year", "PSU"], how="inner", validate="many_to_one"
    )
    migration = normalized_psu(pd.read_parquet(MIGRATION)).merge(
        design[design_keys], on=["Survey Year", "PSU"], how="inner", validate="many_to_one"
    )

    rows = [
        outcome_row(
            household, "Consumption", "Real 2021 Food Consumption Value per Household Member Riels",
            "feasible-supplementary", "same-support level validation",
            "86 PSU-years but only one cross-side commune-year for a local shock interaction",
        ),
        outcome_row(
            household, "Agriculture", "Crop Yield kg per ha",
            "feasible-supplementary", "same-support level validation",
            "agricultural-household selection and insufficient shared local shocks",
        ),
        outcome_row(
            household, "Agriculture", "Real 2021 Crop Production Value Riels",
            "feasible-supplementary", "same-support level validation",
            "agricultural-household selection and insufficient shared local shocks",
        ),
        outcome_row(
            household, "Agriculture", "Post Harvest Loss Share",
            "feasible-supplementary", "same-support level validation",
            "agricultural-household selection and insufficient shared local shocks",
        ),
        outcome_row(
            household, "Food security", "Any Severe Food Insecurity Experience",
            "marginal-supplementary", "descriptive or level robustness only",
            "available in five waves and too few cross-side commune-years",
        ),
        outcome_row(
            household, "Food security", "Food Insecurity Severity Sum",
            "marginal-supplementary", "descriptive or level robustness only",
            "available in four waves and too few cross-side commune-years",
        ),
        outcome_row(
            household, "Agricultural adaptation", "Any Irrigable Parcel",
            "marginal-supplementary", "mechanism screen only",
            "post-treatment mechanism, 216 households, and insufficient shared local shocks",
        ),
    ]
    school_age = education.loc[education["School Age 6 to 17"].eq(1)].copy()
    rows.append(
        outcome_row(
            school_age, "Education", "Currently Attending School",
            "feasible-supplementary", "same-support level validation",
            "repeated cross-section supports levels but not a local shock-response design",
        )
    )
    rows.append(
        outcome_row(
            school_age, "Education", "Years Attended School",
            "feasible-supplementary", "same-support level validation",
            "school-age measure is not a cohort-level historical exposure design",
        )
    )
    availability = pd.DataFrame(rows)
    availability.to_csv(OUTPUT / "current_outcome_availability.csv", index=False)

    identification = pd.DataFrame(
        [
            {
                "target_estimand": "Same-support household welfare level discontinuity",
                "support": f"{len(household)} household-wave rows; {design[['Survey Year', 'PSU']].drop_duplicates().shape[0]} PSU-years; {design['PSU'].nunique()} unique PSU codes; {design['Survey Year'].nunique()} waves",
                "status": "potentially feasible after an outcome-blind precision gate",
                "reason": "both historical sides are represented in eight waves, but inference must use PSU-level effective support",
            },
            {
                "target_estimand": "Historical-side difference in household rainfall sensitivity",
                "support": f"{int(commune_year['cross_side_within_commune'].sum())} cross-side commune-year",
                "status": "not feasible as headline local evidence",
                "reason": "the mandatory modern-commune/shared-shock comparison has only one identifying group",
            },
            {
                "target_estimand": "Migration level or composition difference",
                "support": f"{len(migration)} person-module rows on the 5 km support",
                "status": "needs harmonization before feasibility can be judged",
                "reason": "migration questions and reference periods change across waves",
            },
            {
                "target_estimand": "Child nutrition or anthropometry difference",
                "support": "health modules contain infant feeding and vaccination but no height/weight anthropometry",
                "status": "not available in current CSES files",
                "reason": "a new nutrition source is required",
            },
        ]
    )
    identification.to_csv(OUTPUT / "identification_feasibility.csv", index=False)

    external = pd.DataFrame(
        [
            {
                "priority": 1,
                "source": "NIS custom village-level 2019 Population Census tabulation",
                "outcomes": "education, migration, housing, disability, utilities, occupation",
                "public_geography": "IPUMS public sample reaches communes with 2,000+ population but suppresses detailed geography",
                "required_access": "official village-code tabulations for the frozen 291-village list; individual microdata not required",
                "potential_role": "strong independent same-support level validation",
            },
            {
                "priority": 2,
                "source": "Cambodia Agriculture Survey 2021-2023 panel / CAS 2023",
                "outcomes": "crop area, harvest, production, inputs, irrigation, shocks, FIES, savings and remittances",
                "public_geography": "public CAS 2023 core file exposes province only",
                "required_access": "stable anonymized EA/village crosswalk, secure linkage, or official boundary-corridor tabulations",
                "potential_role": "best route to agricultural welfare and reported shock impacts",
            },
            {
                "priority": 3,
                "source": "Cambodia DHS 2021-22",
                "outcomes": "child anthropometry, nutrition, health, household wealth, education",
                "public_geography": "registered GPS clusters are randomly displaced; rural points by up to 5 km and 1% by up to 10 km",
                "required_access": "DHS registration plus probabilistic side assignment and wide-band sensitivity",
                "potential_role": "nutrition validation only; unsuitable as clean 5 km boundary evidence",
            },
            {
                "priority": 4,
                "source": "IPUMS Cambodia 2019 Census public-use sample",
                "outcomes": "education, migration, work, housing and disability",
                "public_geography": "communes with 2,000+ population; detailed geography suppressed and records may be swapped",
                "required_access": "standard IPUMS research application",
                "potential_role": "coarse supplementary level analysis, not village-side assignment",
            },
        ]
    )
    external.to_csv(OUTPUT / "external_source_candidates.csv", index=False)

    print("Current outcome availability")
    print(availability.to_string(index=False))
    print("\nIdentification feasibility")
    print(identification.to_string(index=False))
    print("\nExternal candidates")
    print(external.to_string(index=False))


if __name__ == "__main__":
    main()
