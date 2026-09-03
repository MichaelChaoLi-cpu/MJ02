#!/usr/bin/env python3
"""National Mapping and Transport-Support Variable Definitions.

Plan: Standalone Appendix dictionary for current national exposure layers
and the survey transport-support boundary.
Framework: AnaSOP Sections 4-7 national geography and extrapolation gate.
"""

from pathlib import Path

from _variable_definition_workbook import validate_variable_dictionary, write_variable_dictionary
from table_variable_definitions_and_harmonization import build_table


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/tables/Table_national_mapping_and_transport_support_variable_definitions.xlsx"
ANALYSIS = ROOT / "data/exp/analysis/climate-welfare/national-mapping-variable-definitions/table_rows.csv"


def main() -> None:
    table = build_table()[0].iloc[25:32].copy()
    assert len(table) == 7
    ANALYSIS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(ANALYSIS, index=False)
    write_variable_dictionary(
        table,
        title="National Mapping and Transport-Support Variable Definitions",
        sheet_name="Mapping variables",
        output=OUTPUT,
        notes=[
            "Notes: The national map uses historical mean dry-spell days, mean absolute-heat days, their percentile ranks, and a continuous overlap rank.",
            "Survey common support restricts household interpretation only; every national village remains in the hazard maps.",
        ],
    )
    validate_variable_dictionary(OUTPUT, sheet_name="Mapping variables", expected_rows=7)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
