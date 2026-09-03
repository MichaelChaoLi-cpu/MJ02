#!/usr/bin/env python3
"""Survey-Wave Linkage and Interview Timing.

Plan: Report each CSES wave's household and village counts, public-point linkage,
linked outcome support, effective spatial blocks, and actual fieldwork timing.
Framework: AnaSOP workflow step 1 linkage audit and Model B's use of the last
fully completed May-October season before interview.
"""

from __future__ import annotations

from pathlib import Path

from _sample_linkage_workbook import validate_workbook, write_single_table
from table_analytical_samples_variable_definitions_and_linkage import build_table


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/tables/Table_survey_wave_linkage_and_interview_timing.xlsx"


def main() -> None:
    combined, section_breaks = build_table()
    table = combined.iloc[section_breaks[0] :].copy()
    assert table.shape == (9, 10)
    write_single_table(
        table,
        title="Survey-Wave Linkage and Interview Timing",
        sheet_name="Survey-wave linkage",
        output=OUTPUT,
        notes=[
            "Notes: Counts and linkage shares are unweighted. Outcome support is evaluated among households with an unambiguous public-village climate–ecology link.",
            "Effective spatial clusters use fixed 0.75-degree blocks and vary across survey waves with linked geographic coverage.",
            "The 2019 survey ran from July 2019 through June 2020 after raw-date recovery; one household retains a source timing warning.",
        ],
    )
    validate_workbook(OUTPUT, "Survey-wave linkage", 9)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
