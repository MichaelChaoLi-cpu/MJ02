"""Shared presentation helpers for article-facing XLSX workbooks."""
from __future__ import annotations

from openpyxl.styles import Alignment
from openpyxl.worksheet.worksheet import Worksheet


DISPLAY_LABELS = {
    "A. Natural-unit distributions on the common complete sample": "A. Distributions",
    (
        "Common rules: food prior-7-day value ×52 · divide by household size · "
        "log outcome in regressions · no winsorisation or routine imputation"
    ): "Construction",
    "A. Adaptive bandwidth, prediction, and effective local sample": "A. Bandwidth and support",
    "B. Local coefficient dispersion, uncertainty, and multiplicity": "B. Local slopes",
    "Released and eligible samples": "Eligible sample",
    "Village and NPP linkage": "NPP linkage",
    "Outcome support and retention": "Outcome support",
    "Annual strict-cropland mean NPP (kg C m⁻² year⁻¹)": "Mean annual NPP",
    "Log real per-capita total consumption": "Total consumption",
    "Log real per-capita food consumption": "Food consumption",
    "Real per-capita total consumption": "Total consumption",
    "Real per-capita food consumption": "Food consumption",
    "A. Model and common-sample checks": "A. Model checks",
    "B. Exposure-definition and timing checks": "B. Exposure checks",
    "C. Questionnaire-regime slope interactions": "C. Regime checks",
    "Prior-year 5 km strict-cropland NPP (per 0.1 kg C m⁻²)": "Prior-year NPP",
    "Adjacent-bandwidth\nsign stable": "Sign stable",
}


def unmerge_display_spans(ws: Worksheet) -> None:
    """Remove merges without clipping the text that occupied each display span.

    MiliFrame workbooks must remain rectangular for deterministic DOCX export.
    When a merged range is removed, however, LibreOffice otherwise keeps the
    original wrapped/centred alignment in the upper-left cell and constrains
    the text to that single column.  Left-aligned, unwrapped text can flow over
    the now-empty cells that belonged to the former display span.
    """

    horizontal_spans: list[tuple[int, int]] = []
    for merged_range in list(ws.merged_cells.ranges):
        if merged_range.max_col > merged_range.min_col:
            horizontal_spans.append((merged_range.min_row, merged_range.min_col))
        ws.unmerge_cells(str(merged_range))

    for row, column in horizontal_spans:
        cell = ws.cell(row, column)
        if isinstance(cell.value, str):
            cell.value = DISPLAY_LABELS.get(cell.value, cell.value)
            horizontal = "left"
        else:
            horizontal = "center"
        cell.alignment = Alignment(
            horizontal=horizontal,
            vertical="center",
            wrap_text=False,
            indent=1 if cell.alignment.indent else 0,
        )
