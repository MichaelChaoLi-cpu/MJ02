#!/usr/bin/env python3
"""Analytical Samples and Linkage.

Plan: Present the national climate, vegetation, village, ecological, and
household analytical samples and their linkage and support boundaries.
Framework: AnaSOP workflow steps 1 and 3; Model A uses the 5 km 2001-2023
village-season sample and Model B uses linked households with completed prior
May-October exposure.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from _sample_linkage_workbook import validate_workbook, write_single_table
from table_analytical_samples_variable_definitions_and_linkage import (
    COLUMNS,
    CSES,
    build_table,
    spatial_block,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/tables/Table_analytical_samples_and_linkage.xlsx"
PSEUDO = ROOT / (
    "data/processed/"
    "cses_commune_survey_time_climate_food_pseudopanel_preprocessed.parquet"
)


def main() -> None:
    combined, section_breaks = build_table()
    table = combined.iloc[: section_breaks[0]].copy()
    table = table.loc[
        ~table["Evidence family / analytical sample"].isin(
            {
                "Season-aligned vegetation frame",
                "Strictly ordered NDVI mediation diagnostic",
            }
        )
    ].copy()

    cses = pd.read_parquet(
        CSES,
        columns=[
            "Village Code",
            "Survey Year",
            "Climate Ecology Link Available",
            "Point Longitude",
            "Point Latitude",
        ],
    )
    linked = cses.loc[cses["Climate Ecology Link Available"].eq(1)].copy()
    repeated_ids = (
        linked.groupby("Village Code", observed=True)["Survey Year"]
        .nunique()
        .loc[lambda values: values > 1]
        .index
    )
    single = linked.loc[~linked["Village Code"].isin(repeated_ids)].copy()
    single["Spatial Block"] = spatial_block(single)

    pseudo = pd.read_parquet(PSEUDO)
    repeated_cells = pseudo.loc[pseudo["Repeated Commune Indicator"]].copy()
    primary_cells = repeated_cells.loc[
        repeated_cells["Primary Minimum Five Households"]
    ].copy()
    robust_cells = repeated_cells.loc[
        repeated_cells["Robustness Minimum Ten Households"]
    ].copy()

    additions = pd.DataFrame(
        [
            {
                COLUMNS[0]: "Single-wave-village comparison sample",
                COLUMNS[1]: "2007-2021 waves",
                COLUMNS[2]: "Linked household",
                COLUMNS[3]: len(single),
                COLUMNS[4]: f"{single['Village Code'].nunique():,} villages",
                COLUMNS[5]: 1.0,
                COLUMNS[6]: "Food and approved climate family complete: 100.0%",
                COLUMNS[7]: f"{single['Spatial Block'].nunique()} spatial blocks",
                COLUMNS[8]: "Village appears in only one survey wave",
                COLUMNS[9]: "Sample-structure comparison",
            },
            {
                COLUMNS[0]: "Primary commune pseudo-panel sample",
                COLUMNS[1]: "2007-2021 survey-time cells",
                COLUMNS[2]: "Commune-survey-time cell",
                COLUMNS[3]: len(primary_cells),
                COLUMNS[4]: f"{primary_cells['Commune Code'].nunique():,} repeated communes",
                COLUMNS[5]: primary_cells["Household Count"].sum() / len(linked),
                COLUMNS[6]: "Survey-weighted food, climate, and cell weights complete",
                COLUMNS[7]: f"{primary_cells['Spatial Block ID'].nunique()} spatial blocks",
                COLUMNS[8]: "Repeated commune; at least 5 households per cell",
                COLUMNS[9]: "Primary within-commune Model B estimation",
            },
            {
                COLUMNS[0]: "Ten-household commune pseudo-panel sensitivity",
                COLUMNS[1]: "2007-2021 survey-time cells",
                COLUMNS[2]: "Commune-survey-time cell",
                COLUMNS[3]: len(robust_cells),
                COLUMNS[4]: f"{robust_cells['Commune Code'].nunique():,} repeated communes",
                COLUMNS[5]: robust_cells["Household Count"].sum() / len(linked),
                COLUMNS[6]: "Survey-weighted food, climate, and cell weights complete",
                COLUMNS[7]: f"{robust_cells['Spatial Block ID'].nunique()} spatial blocks",
                COLUMNS[8]: "Repeated commune; at least 10 households per cell",
                COLUMNS[9]: "Cell-size robustness check",
            },
        ],
        columns=COLUMNS,
    )
    table = pd.concat([table, additions], ignore_index=True)
    table = table.replace(
        {
            "Candidate B": "approved",
            "2007–2021": "2007-2021",
            "spatial–temporal": "spatial-temporal",
        },
        regex=True,
    )
    assert table.shape == (10, 10), table.shape
    assert len(single) == 24_721
    assert len(primary_cells) == 3_580
    assert len(robust_cells) == 3_538
    write_single_table(
        table,
        title="Analytical Samples and Linkage",
        sheet_name="Analytical samples",
        output=OUTPUT,
        notes=[
            "Notes: Counts and linkage shares are unweighted. Linked share is the proportion of households with an unambiguous public-village climate–ecology link.",
            "Effective spatial clusters use fixed 0.75-degree blocks. The ecological comparison uses 25 strict spatial-by-temporal held-out cells.",
            "For pseudo-panel rows, linked share is the share of the 46,445 linked households represented by eligible repeated-commune cells.",
            "The ecological and household samples support two parallel, phase-specific response models; pseudo-panel cells do not create a household panel.",
        ],
        highlighted_rows={
            "Primary post-monsoon ecological sample",
            "Strict spatial-temporal cross-fit sample",
            "Linked primary food-consumption sample",
            "Primary commune pseudo-panel sample",
        },
    )
    validate_workbook(OUTPUT, "Analytical samples", 10)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
