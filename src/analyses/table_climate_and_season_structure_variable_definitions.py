#!/usr/bin/env python3
"""Climate and Season-Structure Variable Definitions.

Plan: Standalone Appendix dictionary for the approved and non-promoted climate
and monsoon-structure fields.
Framework: AnaSOP Sections 4-7 climate construction and definition gates.
"""

from pathlib import Path

from _variable_definition_workbook import validate_variable_dictionary, write_variable_dictionary
from table_variable_definitions_and_harmonization import build_table


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/tables/Table_climate_and_season_structure_variable_definitions.xlsx"
ANALYSIS = ROOT / "data/exp/analysis/climate-welfare/climate-season-variable-definitions/table_rows.csv"


def main() -> None:
    table = build_table()[0].iloc[0:10].copy()
    assert len(table) == 10
    ANALYSIS.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(ANALYSIS, index=False)
    write_variable_dictionary(
        table,
        title="Climate and Season-Structure Variable Definitions",
        sheet_name="Climate variables",
        output=OUTPUT,
        notes=[
            "Notes: Missingness is calculated in the declared processed frame and period; no missing climate value is imputed.",
            "One outcome-blind monsoon timing rule is used throughout. The 33°C and 37°C counts are threshold checks around the 35°C primary exposure.",
        ],
    )
    validate_variable_dictionary(OUTPUT, sheet_name="Climate variables", expected_rows=10)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
