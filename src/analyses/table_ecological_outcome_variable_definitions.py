#!/usr/bin/env python3
"""Ecological Outcome Variable Definitions.

Plan: Standalone Appendix dictionary for post-monsoon vegetation and annual
NPP outcomes.
Framework: AnaSOP Sections 4-7 ecological timing and interpretation gates.
"""

from pathlib import Path

from _variable_definition_workbook import validate_variable_dictionary, write_variable_dictionary
from table_variable_definitions_and_harmonization import build_table


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/tables/Table_ecological_outcome_variable_definitions.xlsx"
ANALYSIS = ROOT / "data/exp/analysis/climate-welfare/ecological-outcome-variable-definitions/table_rows.csv"


def main() -> None:
    table = build_table()[0].iloc[10:12].copy()
    assert len(table) == 2
    ANALYSIS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(ANALYSIS, index=False)
    write_variable_dictionary(
        table,
        title="Ecological Outcome Variable Definitions",
        sheet_name="Ecological outcomes",
        output=OUTPUT,
        notes=[
            "Notes: Post-monsoon EVI is primary and NDVI is the cross-sensor confirmation; both are assigned to the preceding production season.",
            "Only the two post-monsoon vegetation outcomes retained in the final analytical story are listed.",
        ],
    )
    validate_variable_dictionary(OUTPUT, sheet_name="Ecological outcomes", expected_rows=2)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
