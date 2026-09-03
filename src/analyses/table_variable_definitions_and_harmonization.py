#!/usr/bin/env python3
"""Variable Definitions and Harmonization.

Plan: Provide an auditable dictionary of the climate, vegetation, household,
predetermined-context, mapping, and support variables used or explicitly
excluded by the current analysis.
Framework: AnaSOP Section 4 variable construction, Sections 5-7 timing and
interpretation gates, and workflow step 12 reproducibility audit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.pagebreak import Break
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[2]
CLIMATE = ROOT / "data/processed/cambodia_national_monsoon_timing_preprocessed.parquet"
ECOLOGY = ROOT / "data/processed/cambodia_public_village_monsoon_ecology_panel_preprocessed.parquet"
CSES = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
RISK = ROOT / "data/exp/analysis/climate-welfare/national-dryspell-absolute-heat-risk/village_risk_metrics_5km.parquet"
SUPPORT = ROOT / "data/exp/analysis/climate-welfare/survey-linkage-transport-support/national_village_support.parquet"
ANALYSIS_DIR = ROOT / "data/exp/analysis/climate-welfare/variable-definitions-harmonization"
OUTPUT = ROOT / "data/exp/internal_output_archive/tables/Table_variable_definitions_and_harmonization_superseded.xlsx"

COLUMNS = [
    "Readable variable name",
    "Definition",
    "Unit",
    "Analytical level",
    "Valid period",
    "Timing / denominator",
    "Transformation / harmonization",
    "Missingness in declared frame",
    "Final role / status",
    "Limitation",
]


def missing_text(
    path: Path,
    variables: list[str],
    *,
    filters: dict[str, object] | None = None,
) -> str:
    filter_columns = list((filters or {}).keys())
    frame = pd.read_parquet(path, columns=list(dict.fromkeys(filter_columns + variables)))
    for column, rule in (filters or {}).items():
        if isinstance(rule, tuple):
            frame = frame.loc[frame[column].between(rule[0], rule[1])]
        else:
            frame = frame.loc[frame[column].eq(rule)]
    values = 100 * frame[variables].isna().mean()
    if len(values) == 1 or values.max() - values.min() < 0.05:
        return f"{values.mean():.1f}%"
    return f"{values.min():.1f}–{values.max():.1f}% across components"


def record(
    name: str,
    definition: str,
    unit: str,
    level: str,
    period: str,
    timing: str,
    transformation: str,
    missingness: str,
    status: str,
    limitation: str,
) -> dict[str, object]:
    return dict(zip(COLUMNS, [
        name, definition, unit, level, period, timing, transformation,
        missingness, status, limitation,
    ]))


def build_table() -> tuple[pd.DataFrame, list[int]]:
    climate_filters = {"Year": (1991, 2024)}
    ecology_5 = {"Buffer Radius km": 5}
    ecology_5_complete = {"Buffer Radius km": 5, "Year": (2001, 2023)}

    rows: list[dict[str, object]] = []

    # A. Climate and season structure
    rows.extend([
        record(
            "Wet-Season Onset Anomaly Z",
            "Timing of the sustained local wet-season onset; positive values are later than usual.",
            "Local SD", "Climate cell-year; village-buffer mean", "1991–2024",
            "First qualifying event from 15 Apr to 31 Aug",
            "Approved rule: first 3-day rain ≥30 mm and no 14-day total <5 mm in the next 30 days; standardized to the cell's 1991–2020 mean and SD.",
            missing_text(CLIMATE, ["Wet-Season Onset DOY Candidate B Anomaly Z"], filters=climate_filters),
            "Primary climate timing; final",
            "A timing measure, not rainfall volume.",
        ),
        record(
            "Longest Intraseasonal Dry Spell Anomaly Z",
            "Longest run of days with rainfall below 1 mm after sustained onset.",
            "Local SD", "Climate cell-year; village-buffer mean", "1991–2024",
            "Approved onset through 31 Oct",
            "Dry-spell days standardized to the cell's 1991–2020 mean and SD; 5 km is primary, 2/10 km are sensitivities.",
            missing_text(CLIMATE, ["Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z"], filters=climate_filters),
            "Primary ecological exposure; final",
            "Captures within-season interruption, not hydrological drought.",
        ),
        record(
            "Post-Onset Absolute Heat Day Count 35 C",
            "Number of post-onset days with daily maximum air temperature at or above 35°C.",
            "Days", "Climate cell-year; village-buffer mean", "1991–2024",
            "Approved onset through 31 Oct; last fully completed season for CSES",
            "Natural count from CHIRTS-ERA5; regression coefficient reported per 10 additional days.",
            missing_text(CLIMATE, ["Post-Onset Absolute Heat Day Count 35 C Candidate B"], filters=climate_filters),
            "Primary household exposure; final",
            "Air temperature threshold; not MODIS land-surface temperature.",
        ),
        record(
            "Post-Onset Heat Degree-Days Above 35 C",
            "Cumulative degrees above 35°C across post-onset days.",
            "°C-days", "Climate cell-year; village-buffer mean", "1991–2024",
            "Approved onset through 31 Oct",
            "Sum of max(Tmax − 35°C, 0); retained in natural units.",
            missing_text(CLIMATE, ["Post-Onset Heat Degree-Days Above 35 C Candidate B"], filters=climate_filters),
            "Secondary intensity diagnostic; final",
            "Not the main estimand and not interchangeable with heat-day counts.",
        ),
        record(
            "Post-Onset Absolute Heat Day Counts 33 C and 37 C",
            "Alternative counts of post-onset days crossing 33°C or 37°C.",
            "Days", "Climate cell-year; village-buffer mean", "1991–2024",
            "Approved onset through 31 Oct",
            "Same construction as the 35°C count; thresholds fixed before comparison.",
            missing_text(CLIMATE, ["Post-Onset Absolute Heat Day Count 33 C Candidate B", "Post-Onset Absolute Heat Day Count 37 C Candidate B"], filters=climate_filters),
            "Threshold robustness; final",
            "The 37°C measure is sparse and does not support the main claim alone.",
        ),
        record(
            "Wet-Season Onset Day of Year",
            "Calendar day of the sustained local wet-season onset under the approved rule.",
            "Day of year", "Climate cell-year; village-buffer mean", "1991–2024",
            "First qualifying event from 15 Apr to 31 Aug",
            "Retained in natural days for exposure-support graphics; the regression uses the local anomaly.",
            missing_text(CLIMATE, ["Wet-Season Onset DOY Candidate B"], filters=climate_filters),
            "Natural-unit support diagnostic; final",
            "Calendar timing is locally heterogeneous and should not replace the local anomaly in regression.",
        ),
        record(
            "Longest Intraseasonal Dry Spell Days",
            "Maximum number of consecutive post-onset days with rainfall below 1 mm.",
            "Days", "Climate cell-year; village-buffer mean", "1991–2024",
            "Approved onset through 31 Oct",
            "Retained in natural days for interpretation and mapping; the regression uses the local anomaly.",
            missing_text(CLIMATE, ["Longest Intraseasonal Dry Spell Days Candidate B"], filters=climate_filters),
            "Natural-unit mapping variable; final",
            "This meteorological interruption is not a hydrological drought measure.",
        ),
        record(
            "May-October Precipitation Anomaly Z",
            "Total May–October rainfall relative to the local reference distribution.",
            "Local SD", "Village-buffer year", "2001–2024",
            "Fixed May–October season",
            "Climate-cell total standardized to 1991–2020, then area-weighted within 2/5/10 km buffers.",
            missing_text(ECOLOGY, ["Village Buffer Mean May October Precipitation Total mm Anomaly Z"], filters=ecology_5),
            "Benchmark exposure; final",
            "Seasonal total cannot identify timing or within-season dry spells.",
        ),
        record(
            "False Onset Share",
            "Share of a village buffer with an apparent onset followed by the defined dry reversal.",
            "Proportion", "Village-buffer year", "2001–2024",
            "Approved onset diagnostic",
            "Area-weighted mean of the climate-cell false-onset indicator.",
            missing_text(ECOLOGY, ["Village Buffer Mean False Onset Indicator Candidate B"], filters=ecology_5),
            "Climate diagnostic; final",
            "Rare event; not promoted as a main exposure.",
        ),
        record(
            "Post-Onset Absolute Heat Frequency 35 C",
            "Share of observed post-onset days with daily maximum air temperature at or above 35°C.",
            "Proportion", "Climate cell-year; village-buffer mean", "1991–2024",
            "Approved onset through 31 Oct",
            "The heat-day count divided by the observed post-onset day count; used only to check exposure support.",
            missing_text(CLIMATE, ["Post-Onset Absolute Heat Day Count 35 C Candidate B"], filters=climate_filters),
            "Exposure-support diagnostic; final",
            "The main regression remains in count days so its coefficient is directly interpretable.",
        ),
    ])
    section_breaks = [0, len(rows)]

    # B. Ecological outcomes
    rows.extend([
        record(
            "November-February Mean EVI Anomaly Z",
            "Mean post-monsoon Enhanced Vegetation Index anomaly assigned to the preceding production season.",
            "Local SD", "Village-buffer season", "2001–2023",
            "November–February following May–October",
            "Standardized within climate cell and 16-day calendar slot against 2001–2020, then area-weighted.",
            missing_text(ECOLOGY, ["Village Buffer Mean November-February Mean EVI Anomaly Z"], filters=ecology_5_complete),
            "Primary ecological outcome; final",
            "Vegetation condition, not crop yield or carbon productivity.",
        ),
        record(
            "November-February Mean NDVI Anomaly Z",
            "Mean post-monsoon Normalized Difference Vegetation Index anomaly over the same window.",
            "Local SD", "Village-buffer season", "2001–2023",
            "November–February following May–October",
            "Same reference, composite support, and spatial aggregation as EVI.",
            missing_text(ECOLOGY, ["Village Buffer Mean November-February Mean NDVI Anomaly Z"], filters=ecology_5_complete),
            "Cross-sensor confirmation; final",
            "Confirms the vegetation direction but does not identify a household mechanism.",
        ),
        record(
            "Annual Land NPP Anomaly kg C per m2",
            "Annual land net primary production minus the village buffer's 2001–2020 mean.",
            "kg C m⁻² year⁻¹", "Village-buffer year", "2001–2024",
            "Same calendar year",
            "MODIS annual NPP area-weighted within 2/5/10 km buffers; anomaly remains in natural units.",
            missing_text(ECOLOGY, ["Village Buffer Mean Annual Land NPP Anomaly kg C per m2"], filters=ecology_5),
            "Appendix failure-mode outcome; final",
            "Failed the incremental-prediction gate; not a main outcome.",
        ),
        record(
            "Baseline-Cropland-Weighted Annual Land NPP Anomaly kg C per m2",
            "Annual NPP anomaly weighted by fixed baseline cropland share.",
            "kg C m⁻² year⁻¹", "Village-buffer year", "2001–2024",
            "Same calendar year; fixed baseline land cover",
            "Grid-cell NPP weighted by baseline cropland share before village aggregation.",
            missing_text(ECOLOGY, ["Baseline-Cropland-Weighted Annual Land NPP Anomaly kg C per m2"], filters=ecology_5),
            "Appendix composition sensitivity; final",
            "Not an exact crop-pixel mask and did not pass the promotion gate.",
        ),
    ])
    section_breaks.append(len(rows))

    # C. Household outcomes, modifiers, and estimation fields
    rows.extend([
        record(
            "Agricultural Participation", "Household reports agricultural production activity.",
            "Binary", "Household-wave", "2007–2021", "All harmonized households",
            "Coded 1 for participation and 0 otherwise.",
            missing_text(CSES, ["Agricultural Participation"]),
            "Appendix household outcome; final", "Did not pass the family-level agricultural outcome gate.",
        ),
        record(
            "Real 2021 Crop Production Value per Cultivated ha Riels",
            "Constant-price crop production value divided by cultivated hectares.",
            "2021 riels ha⁻¹", "Household-wave", "2007–2021", "Positive cultivated area",
            "Nominal value multiplied by annual all-items CPI deflator; divided by cultivated area.",
            missing_text(CSES, ["Real 2021 Crop Production Value per Cultivated ha Riels"]),
            "Appendix household outcome; final", "Structurally undefined for households without positive cultivated area.",
        ),
        record(
            "Real 2021 Food Consumption Value per Household Member Riels",
            "Constant-price reported food consumption divided by household size.",
            "2021 riels person⁻¹", "Household-wave", "2007–2021", "Household members",
            "Interview-month food CPI; documented annual food-CPI fallback when month is unavailable.",
            missing_text(CSES, ["Real 2021 Food Consumption Value per Household Member Riels"]),
            "Primary welfare outcome; final", "Consumption value is not nutrition, calories, or individual intake.",
        ),
        record(
            "Log Real Food Consumption per Household Member",
            "Natural logarithm of positive real food consumption per household member.",
            "Log points", "Household-wave", "2007–2021", "Strictly positive per-member values",
            "Natural log; coefficients translated as 100 × [exp(beta) − 1].",
            missing_text(CSES, ["Real 2021 Food Consumption Value per Household Member Riels"]),
            "Primary modeled welfare outcome; final", "Percentage interpretation is conditional on the regression design.",
        ),
        record(
            "Real 2021 Own Produced Food Value per Household Member Riels",
            "Constant-price own-produced food consumed per household member.",
            "2021 riels person⁻¹", "Household-wave", "2007–2021", "Household members",
            "Same interview-timed food CPI and annual fallback as total food consumption.",
            missing_text(CSES, ["Real 2021 Own Produced Food Value per Household Member Riels"]),
            "Appendix welfare outcome; final", "Does not cover food acquired outside own production.",
        ),
        record(
            "Any Severe Food Insecurity Experience",
            "Household reports at least one harmonized severe food-insecurity experience.",
            "Binary", "Household-wave", "Module-supported waves", "Available module respondents",
            "Harmonized binary indicator; unavailable waves remain missing.",
            missing_text(CSES, ["Any Severe Food Insecurity Experience"]),
            "Appendix welfare outcome; final", "Wave-limited module prevents full-period comparison.",
        ),
        record(
            "Any Irrigable Parcel", "Household has at least one observed agricultural parcel reported as irrigable.",
            "Binary", "Household-wave", "2007–2021", "Households with observed parcel irrigation",
            "Maximum irrigability indicator across observed parcels; unavailable parcel information remains missing.",
            missing_text(CSES, ["Any Irrigable Parcel"]),
            "Appendix buffering modifier; final", "Interaction failed Holm adjustment and is not an intervention effect.",
        ),
        record(
            "Historical Road Distance km", "Mean distance from cells in the 5 km village buffer to the historical road proxy.",
            "km", "Linked household / village", "Time-invariant", "5 km linked-village buffer",
            "Predetermined road layer excluding identified post-2007 AidData corridors.",
            missing_text(CSES, ["Historical Road Distance km"]),
            "Appendix modifier and support context; final", "Unavailable for unlinked households; road proxy is not a complete road census.",
        ),
        record(
            "Log Baseline Population 2000", "Mean log baseline population within the 5 km village buffer.",
            "Log persons", "Linked household / village", "Baseline 2000", "5 km linked-village buffer",
            "Fixed baseline settlement measure aggregated before contemporary outcomes.",
            missing_text(CSES, ["Log Baseline Population 2000"]),
            "Appendix modifier and mapping context; final", "A connectivity proxy, not contemporary population.",
        ),
        record(
            "Household Composition Vector",
            "Female share, mean age, child share, older-age share, and dependency ratio.",
            "Mixed", "Household-wave", "2007–2021", "Members with observed age/sex",
            "Jointly entered only in the declared composition sensitivity.",
            missing_text(CSES, ["Female Household Member Share", "Mean Household Member Age Years", "Child Age 0-14 Share", "Older Age 65 Plus Share", "Household Dependency Ratio"]),
            "Sensitivity controls; final", "Contemporary composition may itself respond to climate and is excluded from the minimal model.",
        ),
        record(
            "Household Survey Weight", "Released CSES household sampling weight.",
            "Weight", "Household-wave", "2007–2021", "Released survey sample",
            "Used in household estimation and weighted coverage; never applied to ecological observations.",
            missing_text(CSES, ["Household Survey Weight"]),
            "Survey estimation weight; final", "Does not repair missing public-village linkage by itself.",
        ),
    ])
    section_breaks.append(len(rows))

    # D. National mapping and transport support
    rows.extend([
        record(
            "Historical Mean Longest Dry Spell Days",
            "Mean annual approved-definition longest intraseasonal dry-spell duration around each village.",
            "Days", "National village, 5 km", "1991–2024", "Valid climate years",
            "Area-weighted 5 km climate history; retained continuously.",
            missing_text(RISK, ["Historical Mean Longest Dry Spell Days"]),
            "Current national risk-map exposure; analysis-derived",
            "Current map uses a historical mean, not the earlier planned exceedance frequency.",
        ),
        record(
            "Historical Mean Post-Onset Heat Days 35 C",
            "Mean annual count of approved-definition post-onset days at or above 35°C around each village.",
            "Days year⁻¹", "National village, 5 km", "1991–2024", "Valid climate years",
            "Area-weighted 5 km climate history; retained continuously.",
            missing_text(RISK, ["Historical Mean Post-Onset Heat Days 35 C"]),
            "Current national risk-map exposure; analysis-derived",
            "Descriptive exposure surface; not a causal welfare-loss prediction.",
        ),
        record(
            "Continuous Joint Exposure Rank",
            "Geometric mean of the national dry-spell and heat percentile ranks.",
            "0–1 rank", "National village, 5 km", "1991–2024 summary", "All national villages",
            "Square root of dry-spell percentile × heat percentile; no regression coefficient applied.",
            missing_text(RISK, ["Continuous Joint Exposure Rank"]),
            "Current descriptive overlap layer; analysis-derived",
            "A relative exposure rank, not an adaptation-priority or causal loss score.",
        ),
        record(
            "Dry-Spell National Percentile",
            "National percentile rank of each village's historical mean longest dry-spell duration.",
            "0–1 rank", "National village, 5 km", "1991–2024 summary", "13,042 national villages",
            "Average-rank percentile with ties retained; used only to describe relative exposure.",
            missing_text(RISK, ["Dry-Spell National Percentile"]),
            "Current descriptive map layer; analysis-derived",
            "A national relative rank, not an absolute drought threshold.",
        ),
        record(
            "Heat National Percentile",
            "National percentile rank of each village's historical mean post-onset 35°C heat days.",
            "0–1 rank", "National village, 5 km", "1991–2024 summary", "13,042 national villages",
            "Average-rank percentile with ties retained; used only to describe relative exposure.",
            missing_text(RISK, ["Heat National Percentile"]),
            "Current descriptive map layer; analysis-derived",
            "A national relative rank, not a household welfare prediction.",
        ),
        record(
            "CSES Linked Village",
            "Indicator that the national public village point has at least one unambiguous CSES village-code link.",
            "Binary", "National village", "2007–2021 survey linkage", "13,042 national villages",
            "Derived from exact public-code or unique normalised-name-within-commune linkage.",
            missing_text(RISK, ["CSES Linked Village"]),
            "Observed survey-linkage layer; final",
            "Linkage indicates observed survey geography, not representativeness by itself.",
        ),
        record(
            "Survey Common Support",
            "Similarity of each national village to the observed CSES-linked village context.",
            "0–1 score and indicator", "National village, 5 km", "Time-invariant transport boundary", "13,042 national villages",
            "exp(−nearest-linked distance) in seven robust-IQR-scaled predetermined variables; P95 primary and P99 sensitivity masks.",
            missing_text(SUPPORT, ["Survey Support Score", "Primary 95 Percent Support", "Sensitivity 99 Percent Support"]),
            "Household-interpretation boundary; analysis-derived",
            "Restricts welfare interpretation only; national hazard mapping remains complete.",
        ),
    ])

    return pd.DataFrame(rows, columns=COLUMNS), section_breaks


def write_workbook(table: pd.DataFrame, section_breaks: list[int]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Variable dictionary"
    ws.sheet_view.showGridLines = False

    navy, blue, light, green, yellow, gray, white = (
        "1F4E78", "5B9BD5", "F3F3F3", "E2F0D9", "FFF2CC", "E7E6E6", "FFFFFF"
    )
    thin = Side(style="thin", color="B7C9D6")
    medium = Side(style="medium", color="7F9DB9")
    vertical = Side(style="thin", color="D7E0E6")

    ws.merge_cells("A1:J1")
    ws["A1"] = "Variable Definitions and Harmonization"
    ws["A1"].font = Font(name="Times New Roman", size=14, bold=True)
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 25

    for column, label in enumerate(COLUMNS, 1):
        cell = ws.cell(3, column, label)
        cell.fill = PatternFill("solid", fgColor=navy)
        cell.font = Font(name="Times New Roman", size=8.8, bold=True, color=white)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(top=medium, bottom=medium, left=vertical if column in {4, 7, 9} else None)
    ws.row_dimensions[3].height = 43

    labels = {
        section_breaks[0]: "A. Climate and season structure",
        section_breaks[1]: "B. Ecological outcomes",
        section_breaks[2]: "C. Household outcomes, modifiers, and estimation fields",
        section_breaks[3]: "D. National mapping and transport support",
    }
    cursor = 4
    section_excel_rows: list[int] = []
    continuation_start = 0
    for index, values in table.iterrows():
        if index == section_breaks[2]:
            continuation_start = cursor
            ws.merge_cells(start_row=cursor, start_column=1, end_row=cursor, end_column=10)
            cell = ws.cell(cursor, 1, "Variable Definitions and Harmonization (continued)")
            cell.font = Font(name="Times New Roman", size=12, bold=True)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            ws.row_dimensions[cursor].height = 24
            cursor += 1
            for column, label in enumerate(COLUMNS, 1):
                cell = ws.cell(cursor, column, label)
                cell.fill = PatternFill("solid", fgColor=navy)
                cell.font = Font(name="Times New Roman", size=8.8, bold=True, color=white)
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                cell.border = Border(top=medium, bottom=medium, left=vertical if column in {4, 7, 9} else None)
            ws.row_dimensions[cursor].height = 43
            cursor += 1
        if index in labels:
            section_excel_rows.append(cursor)
            ws.merge_cells(start_row=cursor, start_column=1, end_row=cursor, end_column=10)
            cell = ws.cell(cursor, 1, labels[index])
            cell.fill = PatternFill("solid", fgColor=blue)
            cell.font = Font(name="Times New Roman", size=9.5, bold=True, color=white)
            cell.alignment = Alignment(horizontal="left", vertical="center")
            cell.border = Border(top=medium, bottom=thin)
            ws.row_dimensions[cursor].height = 20
            cursor += 1

        status = str(values[COLUMNS[8]])
        for column, value in enumerate(values, 1):
            cell = ws.cell(cursor, column, None if pd.isna(value) else value)
            cell.font = Font(name="Times New Roman", size=8.1)
            cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
            cell.border = Border(bottom=thin, left=vertical if column in {4, 7, 9} else None)
            if index % 2:
                cell.fill = PatternFill("solid", fgColor=light)
        if "Primary" in status or "Current national" in status:
            for cell in ws[cursor]:
                cell.fill = PatternFill("solid", fgColor=green)
        elif "not final" in status.lower() or "not current" in status.lower():
            for cell in ws[cursor]:
                cell.fill = PatternFill("solid", fgColor=gray)
        elif "Legacy" in status or "Planned wording" in status:
            for cell in ws[cursor]:
                cell.fill = PatternFill("solid", fgColor=yellow)
        ws.row_dimensions[cursor].height = 38
        cursor += 1

    notes = [
        "Notes: Missingness is calculated in each variable's declared processed frame and period; structural non-applicability is not imputed. Monetary values are converted to 2021 riels using component-matched CPI series.",
        "Green rows identify current primary or national-map fields; grey/yellow rows identify non-final, legacy, or wording-only concepts. Status text, rather than colour, is authoritative.",
        "The current national map uses historical mean dry-spell days, mean absolute-heat days, and their continuous rank—not the earlier unmaterialized threshold-frequency wording.",
    ]
    notes_start = cursor + 1
    for offset, note in enumerate(notes):
        row = notes_start + offset
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=10)
        cell = ws.cell(row, 1, note)
        cell.font = Font(name="Times New Roman", size=8.0, italic=True, color="333333")
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        ws.row_dimensions[row].height = 19

    widths = [31, 39, 17, 23, 18, 30, 46, 24, 31, 42]
    for column, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(column)].width = width
    ws.freeze_panes = "A4"
    ws.auto_filter.ref = f"A3:J{cursor - 1}"
    ws.sheet_view.zoomScale = 52
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.print_options.horizontalCentered = True
    ws.page_margins.left = ws.page_margins.right = 0.14
    ws.page_margins.top = ws.page_margins.bottom = 0.20
    ws.print_area = f"A1:J{notes_start + len(notes) - 1}"
    # Keep the household and national-mapping sections together on the second page.
    ws.row_breaks.append(Break(id=continuation_start - 1))

    wb.properties.title = "Variable Definitions and Harmonization"
    wb.properties.subject = "Climate-welfare variable dictionary and harmonization audit"
    wb.properties.creator = "Mike Li"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUTPUT)


def validate(table: pd.DataFrame) -> None:
    assert table.shape == (32, 10), table.shape
    assert table[COLUMNS[0]].is_unique
    assert table.loc[table[COLUMNS[0]].eq("Compound Absolute Heat-Drought Exposure"), COLUMNS[8]].iloc[0] == "Not final"
    assert "historical-mean" in table.loc[table[COLUMNS[0]].eq("Historical Dry-Spell Frequency"), COLUMNS[9]].iloc[0]
    wb = load_workbook(OUTPUT, data_only=False)
    assert wb.sheetnames == ["Variable dictionary"]
    assert wb["Variable dictionary"].max_column == 10
    for row in wb["Variable dictionary"].iter_rows():
        for cell in row:
            if isinstance(cell.value, str):
                assert not cell.value.startswith(("#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"))


def main() -> None:
    table, section_breaks = build_table()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(ANALYSIS_DIR / "table_rows.csv", index=False)
    write_workbook(table, section_breaks)
    validate(table)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Rows={len(table)}; current or final rows={(~table[COLUMNS[8]].str.contains('not final|not current|Legacy|Planned wording', case=False)).sum()}")


if __name__ == "__main__":
    main()
