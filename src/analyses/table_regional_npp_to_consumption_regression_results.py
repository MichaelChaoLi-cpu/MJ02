#!/usr/bin/env python3
"""Regional NPP-to-Consumption Regression Results.

Plan: Compare the national expanded-control Stage-2 NPP slopes with the frozen six-region
SKATER slopes for total and food consumption and report joint equality tests.
Framework: AnaSOP Sections 5-7 survey-weighted expanded-control models,
exact survey-time controls, a common regional interaction model with region
fixed effects, village-clustered inference, and pre-specified joint Wald tests.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from workbook_layout import unmerge_display_spans

from figure_cropland_npp_and_household_consumption import (
    COMPOSITION_CONTROLS,
    FOOD_FLAG,
    FOOD_OUTCOME,
    ID,
    NPP,
    REGION,
    SOCIOECONOMIC_CONTROLS,
    TOTAL_FLAG,
    TOTAL_OUTCOME,
    WEIGHT,
    estimate_national_ladder,
    estimate_regional_slopes,
    load_data,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/results/tables/Table_regional_npp_to_consumption_regression_results.xlsx"
EVIDENCE = ROOT / "data/exp/analysis/climate-npp/regional-npp-to-consumption-regression-results"
NPP_INCREMENT = 0.1


def stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def coefficient_text(row: pd.Series) -> str:
    return (
        f"{NPP_INCREMENT * row['NPP Coefficient']:.5f}{stars(float(row['Probability Value']))}\n"
        f"({NPP_INCREMENT * row['Clustered Standard Error']:.5f})"
    )


def percent_text(row: pd.Series) -> str:
    return (
        f"{row['Percent Difference per 0.1 NPP']:.2f}%{stars(float(row['Probability Value']))}\n"
        f"[{row['Percent Difference 95 Percent CI Lower']:.2f}, "
        f"{row['Percent Difference 95 Percent CI Upper']:.2f}]"
    )


def p_text(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def regional_support(frame: pd.DataFrame, outcome: str, flag: str) -> pd.DataFrame:
    required = [
        ID,
        REGION,
        WEIGHT,
        "Exact Survey Time",
        NPP,
        outcome,
        *COMPOSITION_CONTROLS,
        *SOCIOECONOMIC_CONTROLS,
    ]
    sample = frame.loc[frame[flag].fillna(False).astype(bool)].dropna(subset=required).copy()
    return (
        sample.groupby(REGION, as_index=False)
        .agg(Observations=(ID, "size"), Villages=(ID, "nunique"))
        .rename(columns={REGION: "Region ID"})
    )


def assemble_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = load_data()
    national = pd.concat(
        [
            estimate_national_ladder(frame, TOTAL_OUTCOME, TOTAL_FLAG, "Total consumption"),
            estimate_national_ladder(frame, FOOD_OUTCOME, FOOD_FLAG, "Food consumption"),
        ],
        ignore_index=True,
    )
    national = national.loc[national["Specification"].eq("+ Socioeconomic controls")].copy()
    national["Geography"] = "National"

    total_regional, total_test = estimate_regional_slopes(
        frame,
        TOTAL_OUTCOME,
        TOTAL_FLAG,
        "Total consumption",
        controls=[*COMPOSITION_CONTROLS, *SOCIOECONOMIC_CONTROLS],
        specification="Frozen SKATER regional slope + socioeconomic controls",
    )
    food_regional, food_test = estimate_regional_slopes(
        frame,
        FOOD_OUTCOME,
        FOOD_FLAG,
        "Food consumption",
        controls=[*COMPOSITION_CONTROLS, *SOCIOECONOMIC_CONTROLS],
        specification="Frozen SKATER regional slope + socioeconomic controls",
    )
    regional = pd.concat([total_regional, food_regional], ignore_index=True)
    regional["Geography"] = "R" + regional["Region ID"].astype(int).astype(str)
    coefficients = pd.concat([national, regional], ignore_index=True, sort=False)
    tests = pd.concat([total_test, food_test], ignore_index=True)

    total_support = regional_support(frame, TOTAL_OUTCOME, TOTAL_FLAG)
    total_support.insert(0, "Outcome", "Total consumption")
    food_support = regional_support(frame, FOOD_OUTCOME, FOOD_FLAG)
    food_support.insert(0, "Outcome", "Food consumption")
    support = pd.concat([total_support, food_support], ignore_index=True)
    return coefficients, tests, support, frame


def get_row(coefficients: pd.DataFrame, outcome: str, geography: str) -> pd.Series:
    return coefficients.loc[
        coefficients["Outcome"].eq(outcome) & coefficients["Geography"].eq(geography)
    ].iloc[0]


def write_workbook(
    coefficients: pd.DataFrame,
    tests: pd.DataFrame,
    support: pd.DataFrame,
) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Regional Stage 2"
    ws.sheet_view.showGridLines = False

    navy = "1F4E78"
    mid_blue = "5B9BD5"
    light_gray = "F4F6F7"
    white = "FFFFFF"
    thin = Side(style="thin", color="B7C9D6")
    medium = Side(style="medium", color="7F9DB9")
    vertical = Side(style="thin", color="D7E0E6")
    region_colors = ["173F5F", "2F80A2", "3A9D8F", "D4A72C", "D97757", "9B4A63"]

    ws.merge_cells("A1:I1")
    ws["A1"] = "Regional NPP-to-Consumption Regression Results"
    ws["A1"].font = Font(name="Times New Roman", size=15, bold=True)
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28

    ws["A3"] = "Exposure"
    ws.merge_cells("B3:I3")
    ws["B3"] = "Prior-year 5 km strict-cropland NPP (per 0.1 kg C m⁻²)"
    for cell in ws[3]:
        cell.fill = PatternFill("solid", fgColor=navy)
        cell.font = Font(name="Times New Roman", size=10, bold=True, color=white)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(top=medium, bottom=medium)
    ws["A3"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.row_dimensions[3].height = 25

    headers = ["Measure", "National", "R1", "R2", "R3", "R4", "R5", "R6", "Equality p"]
    for column, label in enumerate(headers, start=1):
        fill = mid_blue if column < 3 or column == 9 else region_colors[column - 3]
        cell = ws.cell(4, column, label)
        cell.fill = PatternFill("solid", fgColor=fill)
        cell.font = Font(name="Times New Roman", size=10, bold=True, color=white)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(left=vertical if column > 1 else None)
    ws["A4"].alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.row_dimensions[4].height = 28

    geographies = ["National", "R1", "R2", "R3", "R4", "R5", "R6"]
    row_number = 5
    for outcome_index, outcome in enumerate(["Total consumption", "Food consumption"]):
        ws.merge_cells(start_row=row_number, start_column=1, end_row=row_number, end_column=9)
        section = ws.cell(row_number, 1, outcome)
        section.fill = PatternFill("solid", fgColor=mid_blue)
        section.font = Font(name="Times New Roman", size=9.6, bold=True, color=white)
        section.alignment = Alignment(horizontal="left", vertical="center", indent=1)
        ws.row_dimensions[row_number].height = 20
        row_number += 1

        coefficient_label = ws.cell(row_number, 1, "Log coefficient (standard error)")
        coefficient_label.font = Font(name="Times New Roman", size=9.4, bold=True)
        coefficient_label.alignment = Alignment(horizontal="left", vertical="center")
        coefficient_label.border = Border(bottom=thin)
        for column, geography in enumerate(geographies, start=2):
            cell = ws.cell(row_number, column, coefficient_text(get_row(coefficients, outcome, geography)))
            cell.font = Font(name="Times New Roman", size=9.3, bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = Border(left=vertical, bottom=thin)
        joint_p = float(tests.loc[tests["Outcome"].eq(outcome), "Probability Value"].iloc[0])
        cell = ws.cell(row_number, 9, p_text(joint_p))
        cell.font = Font(name="Times New Roman", size=9.4, bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(left=vertical, bottom=thin)
        ws.row_dimensions[row_number].height = 38
        row_number += 1

        percent_label = ws.cell(row_number, 1, "Implied consumption difference (%) [95% CI]")
        percent_label.font = Font(name="Times New Roman", size=9.4, bold=True)
        percent_label.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        percent_label.border = Border(bottom=thin)
        percent_label.fill = PatternFill("solid", fgColor=light_gray)
        for column, geography in enumerate(geographies, start=2):
            cell = ws.cell(row_number, column, percent_text(get_row(coefficients, outcome, geography)))
            cell.font = Font(name="Times New Roman", size=9.2, bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = Border(left=vertical, bottom=thin)
            cell.fill = PatternFill("solid", fgColor=light_gray)
        ws.cell(row_number, 9).border = Border(left=vertical, bottom=thin)
        ws.cell(row_number, 9).fill = PatternFill("solid", fgColor=light_gray)
        ws.row_dimensions[row_number].height = 38
        row_number += 1

    ws.merge_cells(start_row=row_number, start_column=1, end_row=row_number, end_column=9)
    ws.cell(row_number, 1, "Model structure and regional support")
    ws.cell(row_number, 1).fill = PatternFill("solid", fgColor=mid_blue)
    ws.cell(row_number, 1).font = Font(name="Times New Roman", size=9.5, bold=True, color=white)
    ws.cell(row_number, 1).alignment = Alignment(horizontal="left", vertical="center", indent=1)
    ws.row_dimensions[row_number].height = 20
    row_number += 1

    total_support = support.loc[support["Outcome"].eq("Total consumption")].set_index("Region ID")
    food_support = support.loc[support["Outcome"].eq("Food consumption")].set_index("Region ID")
    structure_rows = [
        ("Single common regional-interaction model", ["Reference"] + ["Yes"] * 6 + [None], None),
        ("Exact survey-time fixed effects", ["Yes"] * 7 + [None], None),
        ("Household composition controls", ["Yes"] * 7 + [None], None),
        ("Socioeconomic controls", ["Yes"] * 7 + [None], None),
        ("Region fixed effects", ["No"] + ["Yes"] * 6 + [None], None),
        ("Survey weights", ["Yes"] * 7 + [None], None),
        ("Standard errors clustered by village", ["Yes"] * 7 + [None], None),
        (
            "Households: total consumption",
            [43120] + [int(total_support.loc[i, "Observations"]) for i in range(1, 7)] + [None],
            "#,##0",
        ),
        (
            "Households: food consumption",
            [43365] + [int(food_support.loc[i, "Observations"]) for i in range(1, 7)] + [None],
            "#,##0",
        ),
        (
            "Villages: total consumption",
            [2848] + [int(total_support.loc[i, "Villages"]) for i in range(1, 7)] + [None],
            "#,##0",
        ),
        (
            "Villages: food consumption",
            [2848] + [int(food_support.loc[i, "Villages"]) for i in range(1, 7)] + [None],
            "#,##0",
        ),
        ("Joint-test degrees of freedom", [None] * 7 + [5], "0"),
    ]
    for index, (label, values, number_format) in enumerate(structure_rows):
        label_cell = ws.cell(row_number, 1, label)
        label_cell.font = Font(name="Times New Roman", size=8.9, bold=index < 6)
        label_cell.alignment = Alignment(horizontal="left", vertical="center")
        label_cell.border = Border(bottom=thin)
        if index % 2:
            label_cell.fill = PatternFill("solid", fgColor=light_gray)
        for column, value in enumerate(values, start=2):
            cell = ws.cell(row_number, column, value)
            cell.font = Font(name="Times New Roman", size=8.9)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = Border(left=vertical, bottom=thin)
            if number_format:
                cell.number_format = number_format
            if index % 2:
                cell.fill = PatternFill("solid", fgColor=light_gray)
        ws.row_dimensions[row_number].height = 23
        row_number += 1

    row_number += 1
    notes = [
        "Notes: National values reproduce the expanded-control models in the national NPP-to-consumption results. R1–R6 are survey-weighted slopes from one common regional-interaction model with the same controls.",
        "Coefficient rows report the log-consumption coefficient and village-clustered standard error per 0.1 kg C m⁻² NPP. Percentage rows report 100[exp(0.1β)−1] with 95% confidence interval.",
        "Equality p tests the null that all six regional NPP slopes are equal (5 df). *** p<0.01; ** p<0.05; * p<0.10. Regions were frozen using outcome-blind features before coefficient inspection.",
    ]
    for note in notes:
        ws.merge_cells(start_row=row_number, start_column=1, end_row=row_number, end_column=9)
        cell = ws.cell(row_number, 1, note)
        cell.font = Font(name="Times New Roman", size=8.1, italic=True, color="333333")
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        ws.row_dimensions[row_number].height = 22
        row_number += 1

    widths = [42, 18, 17, 17, 17, 17, 17, 17, 17]
    for column, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(column)].width = width
    ws.freeze_panes = "B5"
    ws.sheet_view.zoomScale = 70
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1
    ws.print_options.horizontalCentered = True
    ws.page_margins.left = ws.page_margins.right = 0.18
    ws.page_margins.top = ws.page_margins.bottom = 0.22
    ws.print_area = f"A1:I{row_number - 1}"
    wb.properties.title = "Regional NPP-to-Consumption Regression Results"
    wb.properties.subject = "Frozen six-region Stage-2 interaction models"
    wb.properties.creator = "Mike Li"
    # Preserve cell content while removing presentation-only merges for DOCX.
    unmerge_display_spans(ws)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws["B3"].alignment = Alignment(horizontal="left", vertical="center", wrap_text=False)
    for note_row in range(row_number - len(notes), row_number):
        ws.cell(note_row, 1).alignment = Alignment(horizontal="left", vertical="center", wrap_text=False)
        ws.row_dimensions[note_row].height = 17
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT)


def validate() -> None:
    wb = load_workbook(OUTPUT, data_only=False)
    assert wb.sheetnames == ["Regional Stage 2"]
    ws = wb["Regional Stage 2"]
    assert ws.max_column == 9
    assert not ws.merged_cells.ranges
    assert [ws.cell(4, column).value for column in range(1, 10)] == [
        "Measure", "National", "R1", "R2", "R3", "R4", "R5", "R6", "Equality p"
    ]
    for sheet_row in ws.iter_rows():
        for cell in sheet_row:
            if isinstance(cell.value, str):
                assert not cell.value.startswith(("#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"))


def main() -> None:
    coefficients, tests, support, _ = assemble_results()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(EVIDENCE / "regional_npp_to_consumption_coefficients.csv", index=False)
    tests.to_csv(EVIDENCE / "regional_npp_slope_equality_tests.csv", index=False)
    support.to_csv(EVIDENCE / "regional_npp_to_consumption_support.csv", index=False)
    write_workbook(coefficients, tests, support)
    validate()
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(coefficients[["Outcome", "Geography", "NPP Coefficient", "Clustered Standard Error", "Probability Value", "Percent Difference per 0.1 NPP"]].to_string(index=False))
    print(tests.to_string(index=False))
    print(support.to_string(index=False))


if __name__ == "__main__":
    main()
