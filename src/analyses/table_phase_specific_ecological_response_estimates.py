#!/usr/bin/env python3
"""Phase-Specific Ecological Response Estimates.

Plan: Present the validated vegetation regressions in a compact Stargazer-style table.
Framework: AnaSOP Model A, with village and year fixed effects and 0.75-degree
spatial-block-clustered standard errors; strict cross-fit performance is reported separately.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data/exp/analysis/climate-welfare/postmonsoon-vegetation-validation"
COEFFICIENTS = INPUT_DIR / "coefficients.csv"
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/tables/Table_phase_specific_ecological_response_estimates.xlsx"

EVI = "Village Buffer Mean November-February Mean EVI Anomaly Z"
NDVI = "Village Buffer Mean November-February Mean NDVI Anomaly Z"

MODELS = [
    ("(1)", "EVI", "Candidate B, 35 C, 2 km", EVI),
    ("(2)", "EVI", "Candidate B, 35 C, 5 km", EVI),
    ("(3)", "EVI", "Candidate B, 35 C, 10 km", EVI),
    ("(4)", "NDVI", "Candidate B, 35 C, 5 km", NDVI),
]

EXPOSURES = [
    ("May–October precipitation total", "May October Precipitation Total"),
    ("Wet-season onset date", "Wet-Season Onset DOY"),
    ("Longest intraseasonal dry spell", "Longest Intraseasonal Dry Spell Days"),
    ("Absolute heat days, 35 C threshold (per 10)", "Post-Onset Absolute Heat Day Count 35 C"),
]


def stars(probability: float) -> str:
    if probability < 0.01:
        return "***"
    if probability < 0.05:
        return "**"
    if probability < 0.10:
        return "*"
    return ""


def regression_row(coefficients: pd.DataFrame, specification: str, outcome: str, exposure: str) -> pd.Series:
    rows = coefficients.loc[
        coefficients["Specification"].eq(specification)
        & coefficients["Outcome"].eq(outcome)
        & coefficients["Exposure"].str.contains(exposure, regex=False)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one result for specification={specification!r}, outcome={outcome!r}, "
            f"exposure={exposure!r}; found {len(rows)}"
        )
    return rows.iloc[0]


def add_value(ws, row: int, column: int, value, *, bold: bool = False, italic: bool = False) -> None:
    cell = ws.cell(row=row, column=column, value=value)
    cell.font = Font(name="Times New Roman", size=10.5, bold=bold, italic=italic, color="000000")
    cell.alignment = Alignment(horizontal="center", vertical="center")


def main() -> None:
    coefficients = pd.read_csv(COEFFICIENTS)
    wb = Workbook()
    ws = wb.active
    ws.title = "Regression results"
    ws.sheet_view.showGridLines = False
    ws.freeze_panes = "B7"

    thin_black = Side(style="thin", color="000000")
    medium_black = Side(style="medium", color="000000")
    light_gray = PatternFill("solid", fgColor="EDEDED")
    pale_green = PatternFill("solid", fgColor="EAF3EE")

    last_column = 1 + len(MODELS)
    last_letter = get_column_letter(last_column)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last_column)
    ws["A1"] = "Phase-Specific Ecological Response Estimates"
    ws["A1"].font = Font(name="Times New Roman", size=14, bold=True)
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 24

    ws["A3"] = "Dependent variable"
    ws["A4"] = "Monsoon definition / radius"
    for cell in (ws["A3"], ws["A4"]):
        cell.font = Font(name="Times New Roman", size=10.5)
        cell.alignment = Alignment(horizontal="left", vertical="center")

    for column, (model_number, outcome_short, specification, _) in enumerate(MODELS, start=2):
        _, threshold, radius = specification.split(", ")
        add_value(ws, 2, column, model_number, bold=True)
        add_value(ws, 3, column, f"Post-monsoon {outcome_short}", bold=True)
        add_value(
            ws, 4, column,
            f"Approved / {threshold.replace(' C', '')} C / {radius}",
        )

    for cell in ws[2]:
        cell.fill = light_gray
    for cell in ws[4]:
        cell.border = Border(bottom=thin_black)

    current_row = 6
    selected_rows: dict[int, list[pd.Series]] = {
        column: [] for column in range(2, last_column + 1)
    }
    for exposure_label, exposure_match in EXPOSURES:
        ws.cell(current_row, 1, exposure_label)
        ws.cell(current_row, 1).font = Font(name="Times New Roman", size=10.5)
        ws.cell(current_row, 1).alignment = Alignment(horizontal="left", vertical="center")
        if exposure_match == "Longest Intraseasonal Dry Spell Days":
            for cell in ws[current_row]:
                cell.fill = pale_green

        for column, (_, _, specification, outcome) in enumerate(MODELS, start=2):
            result = regression_row(coefficients, specification, outcome, exposure_match)
            selected_rows[column].append(result)
            estimate = float(result["Coefficient Outcome SD per Exposure Unit"])
            probability = float(result["Probability Value"])
            add_value(ws, current_row, column, f"{estimate:.3f}{stars(probability)}")
            add_value(
                ws,
                current_row + 1,
                column,
                f'({float(result["Clustered Standard Error"]):.3f})',
                italic=True,
            )

        ws.cell(current_row + 1, 1, "")
        if exposure_match == "Longest Intraseasonal Dry Spell Days":
            for cell in ws[current_row + 1]:
                cell.fill = pale_green
        current_row += 2

    summary_start = current_row + 1
    summary_labels = [
        "Village fixed effects",
        "Year fixed effects",
        "Observations",
        "Villages",
        "Within R²",
        "SE clustered by 0.75° spatial block",
    ]
    for offset, label in enumerate(summary_labels):
        row = summary_start + offset
        ws.cell(row, 1, label)
        ws.cell(row, 1).font = Font(name="Times New Roman", size=10.5)
        ws.cell(row, 1).alignment = Alignment(horizontal="left", vertical="center")
        for column in range(2, last_column + 1):
            representative = selected_rows[column][0]
            if label in {"Village fixed effects", "Year fixed effects", "SE clustered by 0.75° spatial block"}:
                value = "Yes"
            elif label == "Observations":
                value = int(representative["Observations"])
            elif label == "Villages":
                value = int(representative["Villages"])
            else:
                value = float(representative["Within R2"])
            add_value(ws, row, column, value)
            if label in {"Observations", "Villages"}:
                ws.cell(row, column).number_format = "#,##0"
            elif label == "Within R²":
                ws.cell(row, column).number_format = "0.000"

    ws.cell(summary_start - 1, 1).border = Border(top=thin_black)
    for column in range(1, last_column + 1):
        ws.cell(summary_start - 1, column).border = Border(top=thin_black)
        ws.cell(summary_start + len(summary_labels) - 1, column).border = Border(bottom=medium_black)

    notes_row = summary_start + len(summary_labels) + 2
    notes = [
        "Notes: Heat coefficients are vegetation-index SD per 10 additional days at or above 35 C; other climate coefficients are per 1 SD.",
        "All regressions include village and year fixed effects. ***, **, and * denote p < 0.01, p < 0.05, and p < 0.10.",
        "Post-monsoon outcomes are November–February averages; climate exposures use the completed May–October season.",
    ]
    for offset, note in enumerate(notes):
        ws.merge_cells(
            start_row=notes_row + offset,
            start_column=1,
            end_row=notes_row + offset,
            end_column=last_column,
        )
        cell = ws.cell(notes_row + offset, 1, note)
        cell.font = Font(name="Times New Roman", size=9, italic=True, color="333333")
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        ws.row_dimensions[notes_row + offset].height = 18

    widths = {"A": 38, "B": 18, "C": 18, "D": 18, "E": 18}
    for column, width in widths.items():
        ws.column_dimensions[column].width = width
    for row in range(2, notes_row):
        if ws.row_dimensions[row].height is None:
            ws.row_dimensions[row].height = 19

    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1
    ws.print_options.horizontalCentered = True
    ws.sheet_properties.pageSetUpPr.autoPageBreaks = False
    ws.page_margins.left = 0.25
    ws.page_margins.right = 0.25
    ws.page_margins.top = 0.35
    ws.page_margins.bottom = 0.35
    ws.print_area = f"A1:{last_letter}{notes_row + len(notes) - 1}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
