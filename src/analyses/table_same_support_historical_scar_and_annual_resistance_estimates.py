#!/usr/bin/env python3
"""Same-Support Historical Scar and Annual Resistance Estimates.

Plan: Report local level/trend discontinuities, spatial-HAC and bandwidth sensitivity,
and completed annual rainfall-response compatibility frontiers in one review sheet.
Framework: AnaSOP Sections 5.7, 6.13, 6.15, and the corresponding Section 7 steps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

from figure_persistent_historical_scar_and_annual_resistance import (
    ROOT,
    build_annual_estimates,
    fit_rd,
    prepare_npp,
    prepare_social,
    prepare_viirs,
)


OUTPUT = ROOT / "data/exp/legacy-results/tables/Table_same_support_historical_scar_and_annual_resistance_estimates.xlsx"
EXP_DIR = ROOT / "data/exp/same-support-scar-resistance"
SHEET = "Scar and Resistance"


def level_rows() -> pd.DataFrame:
    rows = []
    for data, domain in ((prepare_npp(), "Land NPP"), (prepare_viirs(), "Nighttime activity")):
        for outcome, estimand in (("level", "Long-run level"), ("trend", "Linear trend per decade")):
            for confirmation, specification in ((False, "Primary"), (True, "Within commune")):
                for bandwidth_km in (2, 5, 10):
                    fit = fit_rd(
                        data,
                        outcome,
                        confirmation,
                        cutoff_km=20.0,
                        bandwidth_km=bandwidth_km,
                    )
                    if fit.standardized_ci_low > 0:
                        interpretation = "Positive difference; verify across design sensitivities"
                    elif fit.standardized_ci_high < 0:
                        interpretation = "Negative difference; verify across design sensitivities"
                    else:
                        interpretation = "Confidence interval includes zero"
                    rows.append({
                        "Component": "Persistent scar",
                        "Domain": domain,
                        "Estimand": estimand,
                        "Specification": specification,
                        "Period": "2001-2020" if domain == "Land NPP" and outcome == "level" else
                                  "2001-2021" if domain == "Land NPP" else "2013-2021",
                        "Bandwidth km": bandwidth_km,
                        "Scale": "Outcome SD",
                        "Estimate": fit.standardized_estimate,
                        "95% CI low": fit.standardized_ci_low,
                        "95% CI high": fit.standardized_ci_high,
                        "Compatibility frontier SD": max(abs(fit.standardized_ci_low), abs(fit.standardized_ci_high)),
                        "Units": f"{fit.observations:,} units ({fit.southwest_units:,} Southwest; {fit.west_units:,} West)",
                        "Inference": "Conley spatial HAC; Bartlett 20 km",
                        "Interpretation": interpretation,
                    })
    social = prepare_social()
    for outcome, estimand in (
        ("Village Poverty Rate Percent", "Village poverty rate"),
        ("Mean Years of Schooling", "Mean years of schooling"),
        ("Adult Literacy Rate Percent", "Adult literacy rate"),
    ):
        for confirmation, specification in ((False, "Primary"), (True, "Within commune")):
            for bandwidth_km in (2, 5, 10):
                fit = fit_rd(social, outcome, confirmation, cutoff_km=20.0, bandwidth_km=bandwidth_km)
                if fit.standardized_ci_low > 0:
                    interpretation = "Positive difference; verify across design sensitivities"
                elif fit.standardized_ci_high < 0:
                    interpretation = "Negative difference; verify across design sensitivities"
                else:
                    interpretation = "Confidence interval includes zero"
                rows.append({
                    "Component": "Persistent scar",
                    "Domain": "Village welfare",
                    "Estimand": estimand,
                    "Specification": specification,
                    "Period": "Public replication cross-section",
                    "Bandwidth km": bandwidth_km,
                    "Scale": "Outcome SD",
                    "Estimate": fit.standardized_estimate,
                    "95% CI low": fit.standardized_ci_low,
                    "95% CI high": fit.standardized_ci_high,
                    "Compatibility frontier SD": max(abs(fit.standardized_ci_low), abs(fit.standardized_ci_high)),
                    "Units": f"{fit.observations:,} villages ({fit.southwest_units:,} Southwest; {fit.west_units:,} West)",
                    "Inference": "Conley spatial HAC; Bartlett 20 km",
                    "Interpretation": interpretation,
                })
    return pd.DataFrame(rows)


def annual_rows() -> pd.DataFrame:
    annual = build_annual_estimates()
    rows = []
    for _, row in annual.iterrows():
        rows.append({
            "Component": "Annual resistance",
            "Domain": row["domain"],
            "Estimand": row["estimand"],
            "Specification": row["specification"],
            "Period": row["period"],
            "Bandwidth km": row["bandwidth_km"],
            "Scale": "Outcome SD per 1-SD rainfall shock",
            "Estimate": row["standardized_estimate"],
            "95% CI low": row["standardized_ci_low"],
            "95% CI high": row["standardized_ci_high"],
            "Compatibility frontier SD": row["compatibility_frontier_sd"],
            "Units": "See original annual-response table",
            "Inference": row["inference"],
            "Interpretation": "95% CI inside pre-specified +/-0.20 SD region",
        })
    return pd.DataFrame(rows)


def validate(table: pd.DataFrame) -> None:
    numeric = ["Estimate", "95% CI low", "95% CI high", "Compatibility frontier SD"]
    if table[numeric].isna().any().any():
        raise ValueError("Unexpected missing estimate or confidence interval")
    if not (table["95% CI low"] <= table["Estimate"]).all():
        raise ValueError("At least one estimate lies below its confidence interval")
    if not (table["Estimate"] <= table["95% CI high"]).all():
        raise ValueError("At least one estimate lies above its confidence interval")
    frontier = table[["95% CI low", "95% CI high"]].abs().max(axis=1)
    if not np.allclose(frontier, table["Compatibility frontier SD"]):
        raise ValueError("Compatibility frontier does not match confidence interval")


def write_workbook(table: pd.DataFrame) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name=SHEET, index=False, startrow=3)
        ws = writer.book[SHEET]
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(table.columns))
        ws.cell(1, 1, "Same-Support Historical Scar and Annual Resistance Estimates")
        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(table.columns))
        ws.cell(2, 1, "Level and trend rows use side-specific local linear boundary models. Annual rows preserve the original pre-specified +/-0.20 SD equivalence region; the compatibility frontier is descriptive, not a revised threshold.")

        navy = "1F4E78"
        light_blue = "D9EAF7"
        thin = Side(style="thin", color="B8C2CC")
        ws.cell(1, 1).font = Font(name="Arial", size=14, bold=True, color="FFFFFF")
        ws.cell(1, 1).fill = PatternFill("solid", fgColor=navy)
        ws.cell(1, 1).alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[1].height = 24
        ws.cell(2, 1).font = Font(name="Arial", size=9, italic=True, color="444444")
        ws.cell(2, 1).alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[2].height = 32

        header_row = 4
        for cell in ws[header_row]:
            cell.font = Font(name="Arial", size=9, bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor=navy)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = Border(bottom=thin)
        ws.row_dimensions[header_row].height = 36

        for row in range(5, 5 + len(table)):
            fill = PatternFill("solid", fgColor="FFFFFF" if row % 2 else light_blue)
            for col in range(1, len(table.columns) + 1):
                cell = ws.cell(row, col)
                cell.font = Font(name="Arial", size=8.5)
                cell.fill = fill
                cell.alignment = Alignment(vertical="top", wrap_text=col in (2, 3, 4, 12, 13, 14))
                cell.border = Border(bottom=Side(style="hair", color="D8DEE4"))
            for col in range(8, 12):
                ws.cell(row, col).number_format = "0.000"

        widths = [17, 18, 24, 16, 12, 12, 22, 12, 12, 12, 19, 31, 31, 40]
        for index, width in enumerate(widths, start=1):
            ws.column_dimensions[ws.cell(4, index).column_letter].width = width
        ws.freeze_panes = "A5"
        ws.auto_filter.ref = f"A4:{ws.cell(4 + len(table), len(table.columns)).coordinate}"
        ws.sheet_view.showGridLines = False
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 2
        ws.sheet_properties.pageSetUpPr.fitToPage = True
        ws.print_title_rows = "1:4"


def main() -> None:
    table = pd.concat([level_rows(), annual_rows()], ignore_index=True)
    validate(table)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(EXP_DIR / "same_support_scar_and_annual_resistance_table.csv", index=False)
    write_workbook(table)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Rows: {len(table)}; sheets: 1")
    primary = table.loc[table["Bandwidth km"].eq(5)]
    print(primary[["Component", "Domain", "Estimand", "Specification", "Estimate", "95% CI low", "95% CI high", "Compatibility frontier SD"]].to_string(index=False))


if __name__ == "__main__":
    main()
