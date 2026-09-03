#!/usr/bin/env python3
"""Audit outcome-blind support for nested and mirrored boundary comparisons.

The audit deliberately does not load any outcome columns.  It separates record
counts from the PSU, village, climate-commune, and boundary-segment units that
govern effective information in the proposed spatial-reach experiment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/exp/feasibility-check/boundary-mirrored-strips"
DESIGN = ROOT / "data/processed/historical_boundary_design_preprocessed.parquet"
HOUSEHOLDS = ROOT / "data/processed/direction3_household_conflict_shock_preprocessed.parquet"
EDUCATION = ROOT / "data/processed/direction3_education_conflict_shock_preprocessed.parquet"
VIIRS = ROOT / "data/processed/viirs_historical_boundary_climate_preprocessed.parquet"

KEYS = ["Survey Year", "Survey Wave", "PSU"]
SIDE = "Historical Repression Side"
DISTANCE = "Absolute Distance to Historical Repression Boundary km"
SEGMENT = "Historical Boundary Segment"
VILLAGE = "Matched Public Village Code"

FINE_STRIPS = (
    (0, 2, "0-2 km"),
    (2, 5, "2-5 km"),
    (5, 10, "5-10 km"),
    (10, 15, "10-15 km"),
    (15, 20, "15-20 km"),
    (20, 30, "20-30 km"),
)
SURVEY_ANALYSIS_STRIPS = (
    (0, 5, "0-5 km"),
    (5, 15, "5-15 km"),
    (15, 30, "15-30 km"),
)
CUMULATIVE_BANDWIDTHS = (2, 5, 10, 15, 20, 30)


def in_strip(distance: pd.Series, lower: int, upper: int) -> pd.Series:
    """Return non-overlapping intervals: [0, upper] first, then (lower, upper]."""
    mask = distance.le(upper)
    if lower > 0:
        mask &= distance.gt(lower)
    return mask


def side_count(frame: pd.DataFrame, side: str, unique: list[str] | None = None) -> int:
    selected = frame.loc[frame[SIDE].eq(side)]
    if unique:
        return int(selected[unique].drop_duplicates().shape[0])
    return int(len(selected))


def survey_row(frame: pd.DataFrame, unit: str, lower: int, upper: int, label: str) -> dict:
    selected = frame.loc[in_strip(frame[DISTANCE], lower, upper)].copy()
    return {
        "unit": unit,
        "distance_band": label,
        "lower_km_exclusive_except_zero": lower,
        "upper_km_inclusive": upper,
        "records": len(selected),
        "southwest_records": side_count(selected, "Southwest"),
        "west_records": side_count(selected, "West"),
        "psu_wave_units": selected[KEYS].drop_duplicates().shape[0],
        "southwest_psu_wave_units": side_count(selected, "Southwest", KEYS),
        "west_psu_wave_units": side_count(selected, "West", KEYS),
        "public_villages": selected[VILLAGE].nunique(),
        "survey_years": selected["Survey Year"].nunique(),
        "boundary_segments": selected[SEGMENT].nunique(),
    }


def survey_support(
    design: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    design_columns = KEYS + [VILLAGE, SIDE, DISTANCE, SEGMENT]
    design_small = design[design_columns]
    households = pd.read_parquet(HOUSEHOLDS).merge(
        design_small, on=KEYS, how="inner", validate="many_to_one"
    )
    education = pd.read_parquet(EDUCATION)
    education = education.loc[education["School Age 6 to 17"].eq(1)].merge(
        design_small, on=KEYS, how="inner", validate="many_to_one"
    )

    rows = []
    for unit, frame in (("household", households), ("school-age person", education)):
        for lower, upper, label in FINE_STRIPS:
            row = survey_row(frame, unit, lower, upper, label)
            row["planned_role"] = "support diagnostic only"
            rows.append(row)
        for lower, upper, label in SURVEY_ANALYSIS_STRIPS:
            row = survey_row(frame, unit, lower, upper, label)
            row["planned_role"] = "formal survey spatial-reach band"
            rows.append(row)
    support = pd.DataFrame(rows)

    wave_rows = []
    for unit, frame in (("household", households), ("school-age person", education)):
        for lower, upper, label in SURVEY_ANALYSIS_STRIPS:
            selected = frame.loc[in_strip(frame[DISTANCE], lower, upper)]
            for year, group in selected.groupby("Survey Year", observed=True):
                wave_rows.append(
                    {
                        "unit": unit,
                        "distance_band": label,
                        "survey_year": int(year),
                        "records": len(group),
                        "southwest_records": side_count(group, "Southwest"),
                        "west_records": side_count(group, "West"),
                        "psu_wave_units": group[KEYS].drop_duplicates().shape[0],
                    }
                )
    by_wave = pd.DataFrame(wave_rows)

    outcome_rows = []
    outcome_sets = {
        "household": (
            households,
            (
                ("Consumption", "Real 2021 Food Consumption Value per Household Member Riels"),
                ("Agriculture", "Crop Yield kg per ha"),
                ("Agriculture", "Real 2021 Crop Production Value Riels"),
                ("Agriculture", "Post Harvest Loss Share"),
                ("Food security", "Any Severe Food Insecurity Experience"),
                ("Food security", "Food Insecurity Severity Sum"),
                ("Agricultural adaptation", "Any Irrigable Parcel"),
            ),
        ),
        "school-age person": (
            education,
            (
                ("Education", "Currently Attending School"),
                ("Education", "Years Attended School"),
            ),
        ),
    }
    for unit, (frame, outcomes) in outcome_sets.items():
        for lower, upper, label in SURVEY_ANALYSIS_STRIPS:
            selected = frame.loc[in_strip(frame[DISTANCE], lower, upper)]
            for domain, outcome in outcomes:
                observed = selected.loc[selected[outcome].notna()]
                outcome_rows.append(
                    {
                        "unit": unit,
                        "domain": domain,
                        "outcome": outcome,
                        "distance_band": label,
                        "nonmissing_records": len(observed),
                        "southwest_records": side_count(observed, "Southwest"),
                        "west_records": side_count(observed, "West"),
                        "psu_wave_units": observed[KEYS].drop_duplicates().shape[0],
                        "southwest_psu_wave_units": side_count(observed, "Southwest", KEYS),
                        "west_psu_wave_units": side_count(observed, "West", KEYS),
                        "survey_years": observed["Survey Year"].nunique(),
                    }
                )
    outcome_availability = pd.DataFrame(outcome_rows)

    flow = pd.DataFrame(
        [
            {
                "stage": "National georeferenced 2007-2021 survey release",
                "households": 62_920,
                "school_age_people": int(
                    pd.read_parquet(EDUCATION, columns=["School Age 6 to 17"])[
                        "School Age 6 to 17"
                    ].eq(1).sum()
                ),
                "reason_for_restriction": "None; full supporting national release",
            },
            {
                "stage": "Kampong Speu survey observations",
                "households": len(households),
                "school_age_people": len(education),
                "reason_for_restriction": "Province containing the reconstructed historical boundary",
            },
            {
                "stage": "Frozen 0-5 km local corridor",
                "households": int(in_strip(households[DISTANCE], 0, 5).sum()),
                "school_age_people": int(in_strip(education[DISTANCE], 0, 5).sum()),
                "reason_for_restriction": "Outcome-blind local boundary support; unrelated to VIIRS dates",
            },
            {
                "stage": "Expanded 0-30 km spatial-reach corridor",
                "households": int(in_strip(households[DISTANCE], 0, 30).sum()),
                "school_age_people": int(in_strip(education[DISTANCE], 0, 30).sum()),
                "reason_for_restriction": "Secondary spatial-reach ceiling; wider contrasts are not local RD",
            },
            {
                "stage": "Hypothetical 2013-2021 VIIRS-year intersection (not used)",
                "households": int(households["Survey Year"].between(2013, 2021).sum()),
                "school_age_people": int(education["Survey Year"].between(2013, 2021).sum()),
                "reason_for_restriction": "Diagnostic only; survey and VIIRS panels are not intersected",
            },
        ]
    )
    return support, by_wave, flow, outcome_availability


def viirs_support() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(VIIRS)
    first_year = int(panel["Year"].min())
    cells = panel.loc[panel["Year"].eq(first_year)].copy()
    cells = cells.loc[cells["Published Kampong Speu Replication Frame"].eq(1)]

    strip_rows = []
    for lower, upper, label in FINE_STRIPS:
        selected = cells.loc[in_strip(cells[DISTANCE], lower, upper)].copy()
        commune_sides = selected.groupby("Linked Climate Commune Code")[SIDE].nunique()
        strip_rows.append(
            {
                "distance_band": label,
                "lower_km_exclusive_except_zero": lower,
                "upper_km_inclusive": upper,
                "grid_cells": len(selected),
                "cell_years_2013_2021": len(selected) * panel["Year"].nunique(),
                "southwest_grid_cells": side_count(selected, "Southwest"),
                "west_grid_cells": side_count(selected, "West"),
                "climate_communes": selected["Linked Climate Commune Code"].nunique(),
                "cross_side_climate_communes": int(commune_sides.eq(2).sum()),
                "boundary_segments": selected[SEGMENT].nunique(),
                "planned_role": "formal satellite spatial-reach band",
            }
        )
    strips = pd.DataFrame(strip_rows)

    cumulative_rows = []
    for bandwidth in CUMULATIVE_BANDWIDTHS:
        selected = cells.loc[cells[DISTANCE].le(bandwidth)].copy()
        commune_sides = selected.groupby("Linked Climate Commune Code")[SIDE].nunique()
        cumulative_rows.append(
            {
                "bandwidth_km": bandwidth,
                "grid_cells": len(selected),
                "cell_years_2013_2021": len(selected) * panel["Year"].nunique(),
                "southwest_grid_cells": side_count(selected, "Southwest"),
                "west_grid_cells": side_count(selected, "West"),
                "climate_communes": selected["Linked Climate Commune Code"].nunique(),
                "cross_side_climate_communes": int(commune_sides.eq(2).sum()),
                "boundary_segments": selected[SEGMENT].nunique(),
            }
        )
    return strips, pd.DataFrame(cumulative_rows)


def write_readme(survey: pd.DataFrame, viirs: pd.DataFrame, flow: pd.DataFrame) -> None:
    household_formal = survey.loc[
        survey["unit"].eq("household")
        & survey["planned_role"].eq("formal survey spatial-reach band")
    ]
    readme = f"""# Outcome-blind mirrored-strip support audit

This audit answers whether the local boundary design discarded survey records because VIIRS
starts in 2013. It did not. The household/person surveys and VIIRS are separate outcome systems;
their years are not intersected. The 5 km restriction is a geographic identification choice.

## Frozen hierarchy

1. Keep 0-5 km as the primary local boundary/RD corridor.
2. Use cumulative 2, 5, 10, 15, 20, and 30 km estimates as specification sensitivity.
3. Use non-overlapping 0-2, 2-5, 5-10, 10-15, 15-20, and 20-30 km mirrored strips for satellite
   spatial reach. These are increasingly descriptive away from the boundary.
4. Pool survey records into 0-5, 5-15, and 15-30 km formal bands because the finer strips contain
   too few independent PSU-wave units for headline inference. Fine survey counts remain visible.
5. Use all compatible survey waves (2007-2021) in survey models and all available VNL V2.1 years
   (2013-2021) in VIIRS models. Do not impose a common calendar window.

## Key counts

- Kampong Speu contains {int(flow.loc[flow['stage'].eq('Kampong Speu survey observations'), 'households'].iloc[0]):,} released households.
- The frozen 0-5 km corridor contains {int(flow.loc[flow['stage'].eq('Frozen 0-5 km local corridor'), 'households'].iloc[0]):,} households.
- The expanded 0-30 km corridor contains {int(flow.loc[flow['stage'].eq('Expanded 0-30 km spatial-reach corridor'), 'households'].iloc[0]):,} households.
- Formal survey-band household counts range from {household_formal['records'].min():,} to {household_formal['records'].max():,}; inference must still respect PSU-wave clustering.
- Every fine VIIRS strip contains both historical sides and all {int(viirs['boundary_segments'].min())} boundary segments.
- Cross-side modern-commune overlap declines from {int(viirs.iloc[0]['cross_side_climate_communes'])} in 0-2 km to {int(viirs.iloc[-1]['cross_side_climate_communes'])} in 20-30 km. Therefore the mandatory within-commune confirmation remains meaningful locally but cannot identify the outermost strips.

No outcome variable was loaded or inspected in producing these support files.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    design = pd.read_parquet(DESIGN)
    survey, by_wave, flow, outcome_availability = survey_support(design)
    viirs, cumulative = viirs_support()
    survey.to_csv(OUTPUT / "survey_support_by_strip.csv", index=False)
    by_wave.to_csv(OUTPUT / "survey_support_by_wave_and_strip.csv", index=False)
    flow.to_csv(OUTPUT / "survey_sample_flow.csv", index=False)
    outcome_availability.to_csv(
        OUTPUT / "survey_outcome_availability_by_formal_band.csv", index=False
    )
    viirs.to_csv(OUTPUT / "viirs_support_by_strip.csv", index=False)
    cumulative.to_csv(OUTPUT / "viirs_support_by_cumulative_bandwidth.csv", index=False)
    write_readme(survey, viirs, flow)


if __name__ == "__main__":
    main()
