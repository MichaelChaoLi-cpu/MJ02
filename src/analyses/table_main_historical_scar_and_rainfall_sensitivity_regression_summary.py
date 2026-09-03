#!/usr/bin/env python3
"""Main Historical Scar and Rainfall-Sensitivity Regression Summary.

Plan: Present the completed persistent-scar and annual rainfall-sensitivity results in a
single conventional side-by-side regression table.
Framework: AnaSOP Sections 5.7, 6.9-6.10, 6.12-6.13, and the Section 7 integration step.
This script only summarizes frozen estimates; it does not alter any estimator or sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from figure_historical_repression_and_contemporary_shock_sensitivity import (
    STANDARDIZED_OUTCOME,
    YEAR,
    fit_model,
    prepared_panel,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "data/exp/legacy-results/tables/"
    / "Table_main_historical_scar_and_rainfall_sensitivity_regression_summary.xlsx"
)
AUDIT_OUTPUT = (
    ROOT
    / "data/exp/main-regression-summary/"
    / "main_historical_scar_and_rainfall_sensitivity_regression_summary.csv"
)
SCAR_SOURCE = (
    ROOT
    / "data/exp/same-support-scar-resistance/"
    / "same_support_scar_and_annual_resistance_estimates.csv"
)
VIIRS_SOURCE = (
    ROOT
    / "data/exp/viirs-boundary-experiment/"
    / "viirs_boundary_shock_response_estimates.csv"
)
LONGNTL_SOURCE = (
    ROOT
    / "data/exp/longntl-v2-boundary-experiment/"
    / "longntl_boundary_response_estimates.csv"
)
SHEET = "Main Regression Summary"
SESOI = 0.20


@dataclass(frozen=True)
class Result:
    panel: str
    model: str
    outcome: str
    specification: str
    period: str
    coefficient: float
    standard_error: float
    ci_low: float
    ci_high: float
    p_value: float
    observations: int
    units: int
    bandwidth_km: int
    unit_fixed_effects: bool
    segment_time_effects: bool
    commune_time_effects: bool
    equivalence: str
    inference: str


def stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def p_text(p_value: float) -> str:
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def standardized_se(ci_low: float, ci_high: float) -> float:
    return (ci_high - ci_low) / (2 * 1.96)


def select_one(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in conditions.items():
        mask &= frame[column].eq(value)
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {conditions}, found {len(selected)}")
    return selected.iloc[0]


def build_scar_results() -> list[Result]:
    source = pd.read_csv(SCAR_SOURCE)
    definitions = [
        ("Poverty", "Village welfare", "Village poverty rate"),
        ("Land NPP level", "Land NPP", "Long-run level"),
        ("Observed VIIRS level", "Nighttime activity", "Long-run level"),
    ]
    results: list[Result] = []
    for outcome_label, domain, estimand in definitions:
        for specification in ("Primary", "Within commune"):
            row = select_one(
                source,
                evidence_component="Persistent scar",
                domain=domain,
                estimand=estimand,
                specification=specification,
                bandwidth_km=5.0,
            )
            ci_low = float(row["standardized_ci_low"])
            ci_high = float(row["standardized_ci_high"])
            results.append(
                Result(
                    panel="A. Persistent historical scar",
                    model=f"{outcome_label}: {specification}",
                    outcome=outcome_label,
                    specification=specification,
                    period=str(row["period"]),
                    coefficient=float(row["standardized_estimate"]),
                    standard_error=standardized_se(ci_low, ci_high),
                    ci_low=ci_low,
                    ci_high=ci_high,
                    p_value=float(row["p_value"]),
                    observations=int(row["observations"]),
                    units=int(row["observations"]),
                    bandwidth_km=int(row["bandwidth_km"]),
                    unit_fixed_effects=False,
                    segment_time_effects=False,
                    commune_time_effects=specification == "Within commune",
                    equivalence="Not an equivalence estimand",
                    inference="Conley spatial HAC (20 km)",
                )
            )
    return results


def build_npp_results() -> list[Result]:
    panel = prepared_panel()
    # The frozen confirmatory NPP window is 2001-2021. The source panel now also
    # contains later validation years, so the table must impose the approved period
    # explicitly rather than inherit the materialized panel maximum.
    panel = panel.loc[panel[YEAR].between(2001, 2021)].copy()
    results: list[Result] = []
    for confirmation, specification in ((False, "Primary"), (True, "Within commune")):
        fitted = fit_model(
            panel,
            label=f"Regression summary: NPP {specification}",
            outcome=STANDARDIZED_OUTCOME,
            confirmation=confirmation,
        ).estimate
        results.append(
            Result(
                panel="B. Annual rainfall-sensitivity difference",
                model=f"Land NPP: {specification}",
                outcome="Land NPP",
                specification=specification,
                period="2001-2021",
                coefficient=fitted.estimate,
                standard_error=fitted.standard_error,
                ci_low=fitted.ci_low,
                ci_high=fitted.ci_high,
                p_value=fitted.p_value,
                observations=fitted.sample_size,
                units=fitted.villages,
                bandwidth_km=fitted.bandwidth_km,
                unit_fixed_effects=True,
                segment_time_effects=True,
                commune_time_effects=confirmation,
                equivalence="Yes: 95% CI inside +/-0.20 SD",
                inference="Village + district-by-year clustered",
            )
        )
    return results


def build_viirs_results() -> list[Result]:
    source = pd.read_csv(VIIRS_SOURCE)
    results: list[Result] = []
    for source_specification, specification in (
        ("Primary 5 km", "Primary"),
        ("Within-commune confirmation 5 km", "Within commune"),
    ):
        row = select_one(source, specification=source_specification)
        results.append(
            Result(
                panel="B. Annual rainfall-sensitivity difference",
                model=f"Observed VIIRS: {specification}",
                outcome="Observed VIIRS",
                specification=specification,
                period="2013-2021",
                coefficient=float(row["standardized_estimate"]),
                standard_error=float(row["standard_error"] / row["outcome_sd_reference"]),
                ci_low=float(row["standardized_ci_low"]),
                ci_high=float(row["standardized_ci_high"]),
                p_value=float(row["p_value"]),
                observations=int(row["observations"]),
                units=int(row["grid_cells"]),
                bandwidth_km=int(row["bandwidth_km"]),
                unit_fixed_effects=True,
                segment_time_effects=True,
                commune_time_effects=bool(row["confirmation_model"]),
                equivalence="Yes: 95% CI inside +/-0.20 SD",
                inference="Grid cell + district-by-year clustered",
            )
        )
    return results


def build_longntl_results() -> list[Result]:
    source = pd.read_csv(LONGNTL_SOURCE)
    results: list[Result] = []
    for source_specification, specification in (
        ("Full period primary 5 km", "Primary"),
        ("Full period within-commune 5 km", "Within commune"),
    ):
        row = select_one(source, specification=source_specification)
        results.append(
            Result(
                panel="B. Annual rainfall-sensitivity difference",
                model=f"LongNTL: {specification}",
                outcome="LongNTL",
                specification=specification,
                period=str(row["period"]),
                coefficient=float(row["common_standardized_estimate"]),
                standard_error=float(row["standard_error"] / row["common_sd_reference"]),
                ci_low=float(row["common_standardized_ci_low"]),
                ci_high=float(row["common_standardized_ci_high"]),
                p_value=float(row["p_value"]),
                observations=int(row["observations"]),
                units=int(row["grid_cells"]),
                bandwidth_km=int(row["bandwidth_km"]),
                unit_fixed_effects=True,
                segment_time_effects=True,
                commune_time_effects=bool(row["confirmation_model"]),
                equivalence="Yes: 95% CI inside +/-0.20 SD",
                inference="Grid cell + district-by-year clustered",
            )
        )
    return results


def validate(results: list[Result]) -> None:
    if len(results) != 12:
        raise RuntimeError(f"Expected 12 displayed estimates, found {len(results)}")
    for result in results:
        numeric = np.array(
            [
                result.coefficient,
                result.standard_error,
                result.ci_low,
                result.ci_high,
                result.p_value,
            ],
            dtype=float,
        )
        if not np.isfinite(numeric).all():
            raise RuntimeError(f"Non-finite result in {result.model}")
        if not result.ci_low <= result.coefficient <= result.ci_high:
            raise RuntimeError(f"Estimate outside confidence interval in {result.model}")
        if result.standard_error <= 0 or result.observations <= 0 or result.units <= 0:
            raise RuntimeError(f"Invalid uncertainty or sample support in {result.model}")
    for result in results:
        if result.panel.startswith("B") and not (
            result.ci_low >= -SESOI and result.ci_high <= SESOI
        ):
            raise RuntimeError(f"Unexpected equivalence failure in {result.model}")


def audit_frame(results: list[Result]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "panel": result.panel,
                "model": result.model,
                "outcome": result.outcome,
                "specification": result.specification,
                "period": result.period,
                "standardized_coefficient": result.coefficient,
                "standardized_standard_error": result.standard_error,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "p_value": result.p_value,
                "observations": result.observations,
                "units": result.units,
                "bandwidth_km": result.bandwidth_km,
                "unit_fixed_effects": result.unit_fixed_effects,
                "segment_time_effects": result.segment_time_effects,
                "commune_time_effects": result.commune_time_effects,
                "equivalence": result.equivalence,
                "inference": result.inference,
            }
            for result in results
        ]
    )


def model_columns(results: list[Result], panel: str) -> list[Result]:
    selected = [result for result in results if result.panel == panel]
    if len(selected) != 6:
        raise RuntimeError(f"Expected six columns in {panel}, found {len(selected)}")
    return selected


def write_panel(
    ws,
    *,
    start_row: int,
    panel_title: str,
    coefficient_label: str,
    models: list[Result],
) -> int:
    navy = "1F4E78"
    panel_fill = "D9EAF7"
    light_fill = "F4F7FA"
    grey = "555555"
    thin = Side(style="thin", color="AEB8C2")
    hair = Side(style="hair", color="D8DEE4")

    ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=7)
    panel_cell = ws.cell(start_row, 1, panel_title)
    panel_cell.font = Font(name="Arial", size=10, bold=True, color=navy)
    panel_cell.fill = PatternFill("solid", fgColor=panel_fill)
    panel_cell.alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[start_row].height = 20

    header_row = start_row + 1
    ws.cell(header_row, 1, "Variable / model statistic")
    for index, model in enumerate(models, start=2):
        ws.cell(header_row, index, f"({index - 1})\n{model.outcome}\n{model.specification}")
    for cell in ws[header_row]:
        cell.font = Font(name="Arial", size=8.5, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=navy)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(bottom=thin)
    ws.row_dimensions[header_row].height = 45

    rows = [
        (coefficient_label, [f"{m.coefficient:.3f}{stars(m.p_value)}" for m in models]),
        ("Standard error", [f"({m.standard_error:.3f})" for m in models]),
        ("95% confidence interval", [f"[{m.ci_low:.3f}, {m.ci_high:.3f}]" for m in models]),
        ("p-value", [p_text(m.p_value) for m in models]),
        ("Outcome period", [m.period for m in models]),
        ("Observations", [f"{m.observations:,}" for m in models]),
        ("Location units", [f"{m.units:,}" for m in models]),
        ("Bandwidth", [f"{m.bandwidth_km} km" for m in models]),
        ("Location fixed effects", ["Yes" if m.unit_fixed_effects else "No" for m in models]),
        ("Boundary-segment/time controls", ["Yes" if m.segment_time_effects else "Boundary segment" for m in models]),
        ("Modern commune safeguard", ["Yes" if m.commune_time_effects else "No" for m in models]),
        ("SESOI equivalence", [m.equivalence.replace("Yes: ", "") for m in models]),
        ("Inference", [m.inference for m in models]),
    ]

    for offset, (label, values) in enumerate(rows, start=2):
        row_index = start_row + offset
        fill = PatternFill("solid", fgColor="FFFFFF" if offset % 2 == 0 else light_fill)
        ws.cell(row_index, 1, label)
        for column_index, value in enumerate(values, start=2):
            ws.cell(row_index, column_index, value)
        for column_index in range(1, 8):
            cell = ws.cell(row_index, column_index)
            cell.font = Font(
                name="Arial",
                size=8.5,
                bold=offset == 2 and column_index >= 2,
                color="000000" if offset != 3 else grey,
                italic=offset in (3, 4),
            )
            cell.fill = fill
            cell.alignment = Alignment(
                horizontal="left" if column_index == 1 else "center",
                vertical="center",
                wrap_text=True,
            )
            cell.border = Border(bottom=hair)
        ws.row_dimensions[row_index].height = 30 if label in {"SESOI equivalence", "Inference"} else 19
    return start_row + len(rows) + 2


def write_workbook(results: list[Result]) -> None:
    workbook = Workbook()
    ws = workbook.active
    ws.title = SHEET
    ws.sheet_view.showGridLines = False

    navy = "1F4E78"
    ws.merge_cells("A1:G1")
    ws["A1"] = "Main Historical Scar and Rainfall-Sensitivity Regression Summary"
    ws["A1"].font = Font(name="Arial", size=14, bold=True, color="FFFFFF")
    ws["A1"].fill = PatternFill("solid", fgColor=navy)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 25

    ws.merge_cells("A2:G2")
    ws["A2"] = (
        "All coefficients are standardized. Panel A reports Southwest-minus-West level discontinuities; "
        "Panel B reports Southwest-minus-West differences in the response to a one-standard-deviation "
        "May-October rainfall anomaly."
    )
    ws["A2"].font = Font(name="Arial", size=9, italic=True, color="444444")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[2].height = 34

    next_row = write_panel(
        ws,
        start_row=4,
        panel_title="Panel A. Persistent historical scar (Southwest minus West, outcome SD)",
        coefficient_label="Higher-repression Southwest assignment",
        models=model_columns(results, "A. Persistent historical scar"),
    )
    next_row = write_panel(
        ws,
        start_row=next_row,
        panel_title="Panel B. Annual rainfall-sensitivity difference (outcome SD per 1-SD rainfall shock)",
        coefficient_label="Southwest assignment x rainfall anomaly",
        models=model_columns(results, "B. Annual rainfall-sensitivity difference"),
    )

    note_row = next_row + 1
    ws.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=7)
    ws.cell(note_row, 1, "Notes: * p<0.10, ** p<0.05, *** p<0.01. Panel A uses local linear boundary models with side-specific distance terms and Conley spatial-HAC inference. Panel B absorbs location and boundary-segment-by-year effects; within-commune models additionally absorb commune-by-year effects. The +/-0.20-SD equivalence threshold applies only to Panel B.")
    ws.cell(note_row, 1).font = Font(name="Arial", size=8, italic=True, color="444444")
    ws.cell(note_row, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[note_row].height = 46

    source_row = note_row + 1
    ws.merge_cells(start_row=source_row, start_column=1, end_row=source_row, end_column=7)
    ws.cell(source_row, 1, "Source: frozen estimates from the same-support scar, annual land-NPP, observed-VIIRS, and LongNTL result releases.")
    ws.cell(source_row, 1).font = Font(name="Arial", size=8, italic=True, color="666666")
    ws.cell(source_row, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[source_row].height = 25

    widths = [36, 20, 20, 20, 20, 20, 20]
    for index, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width
    ws.freeze_panes = "B6"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_area = f"A1:G{source_row}"
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.oddFooter.center.text = "Main regression summary"
    ws.oddFooter.right.text = "Page &P of &N"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(OUTPUT)


def main() -> None:
    results = (
        build_scar_results()
        + build_npp_results()
        + build_viirs_results()
        + build_longntl_results()
    )
    validate(results)
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    audit_frame(results).to_csv(AUDIT_OUTPUT, index=False)
    write_workbook(results)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Rows displayed: 2 panels; model columns: 6 per panel; sheets: 1")
    print(
        audit_frame(results)[
            [
                "panel",
                "model",
                "standardized_coefficient",
                "standardized_standard_error",
                "ci_low",
                "ci_high",
                "p_value",
                "observations",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
