#!/usr/bin/env python3
"""Run the first sequential Gate C NPP validation: spatial placebo sites."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mj02-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from pyproj import Transformer

from estimate_cambodia_thailand_gate_c_npp import (
    MAIN_PERIOD,
    NPP_NATURAL,
    POST,
    TARGET,
    fit_model,
)
from freeze_cambodia_thailand_gate_b_protocol import OUT_DIR as GATE_B_DIR, load_features
from test_cambodia_thailand_gate_b_placebo_sites import make_weights


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PANEL = ROOT / "data/processed/cambodia_national_annual_satellite_climate_panel_preprocessed.parquet"
ACTUAL_RESULTS = ROOT / "data/exp/experiments/cambodia-thailand-gate-c-npp/gate_c_npp_main_model_tidy.csv"
PLACEBO_UNIVERSE = GATE_B_DIR / "gate_b_placebo_site_universe.csv"
OUT = ROOT / "data/exp/experiments/cambodia-thailand-gate-c-npp/spatial-placebo"
ESTIMATES = OUT / "gate_c_npp_spatial_placebo_estimates.csv"
SUMMARY = OUT / "gate_c_npp_spatial_placebo_inference.csv"
WORKBOOK = OUT / "gate_c_npp_spatial_placebo_summary.xlsx"
FIGURE = OUT / "gate_c_npp_spatial_placebo_distribution.png"
METADATA = OUT / "gate_c_npp_spatial_placebo_metadata.json"

DRY_RAIN = "May October Dry Rainfall Intensity"
BALANCE_THRESHOLD = 0.10
MIN_TREATED_BLOCK_ESS = 10


def estimate_placement(
    base: pd.DataFrame,
    panel: pd.DataFrame,
    first_easting: float,
    first_northing: float,
    second_easting: float,
    second_northing: float,
) -> dict[str, object]:
    distance = np.minimum(
        np.hypot(
            base["Grid Centre Easting m"] - first_easting,
            base["Grid Centre Northing m"] - first_northing,
        ),
        np.hypot(
            base["Grid Centre Easting m"] - second_easting,
            base["Grid Centre Northing m"] - second_northing,
        ),
    ) / 1000
    candidate = base.copy()
    treated = distance <= 60
    weights, max_smd, treated_block_ess = make_weights(candidate, treated)
    candidate["Gate B Binary Support Overlap Weight"] = weights
    candidate["Continuous Conflict Dose"] = np.maximum(0.0, 1 - distance / 60)
    frame = panel.merge(
        candidate[
            [
                "National Grid Cell ID",
                "Spatial Block ID",
                "Gate B Binary Support Overlap Weight",
                "Continuous Conflict Dose",
            ]
        ],
        on="National Grid Cell ID",
        validate="many_to_one",
    )
    frame[POST] = frame["Year"].between(2012, 2024)
    frame[MAIN_PERIOD] = frame["Year"].between(2001, 2007) | frame["Year"].between(2012, 2024)
    results, diagnostics = fit_model(
        frame,
        NPP_NATURAL,
        DRY_RAIN,
        "Spatial placebo: natural-unit NPP x dry-rainfall intensity",
    )
    target = results.loc[results["Term"].eq(TARGET)].iloc[0]
    return {
        "Maximum Absolute Weighted SMD": max_smd,
        "Treated 10 km Block ESS": treated_block_ess,
        "Target Estimate kg C per m2": float(target["Estimate"]),
        "Clustered Standard Error": float(target["Clustered Standard Error"]),
        "Model-based p-value": float(target["p-value"]),
        "Observations": int(target["Observations"]),
        "Spatial Blocks": int(target["Spatial Blocks"]),
        "Grid Cells": int(target["Grid Cells"]),
        "Climate Cells": int(target["Climate Cells"]),
        "Included Years": diagnostics["included_year_count"],
    }


def write_workbook(estimates: pd.DataFrame, summary: pd.DataFrame) -> None:
    with pd.ExcelWriter(WORKBOOK, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Inference Summary", startrow=0)
        estimates.to_excel(writer, index=False, sheet_name="Placebo Estimates", startrow=0)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.sheet_view.showGridLines = False
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for column_cells in sheet.columns:
                width = min(34, max(12, max(len(str(cell.value or "")) for cell in column_cells) + 2))
                sheet.column_dimensions[column_cells[0].column_letter].width = width


def plot_distribution(valid: pd.DataFrame, actual: float) -> None:
    ordered = valid.sort_values("Target Estimate kg C per m2").reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(7.4, 4.4))
    axis.scatter(
        ordered["Target Estimate kg C per m2"],
        np.arange(1, len(ordered) + 1),
        color="#6D8299",
        s=38,
        label="Eligible placebo placement",
    )
    axis.axvline(actual, color="#B23A48", linewidth=1.8, label="Actual conflict placement")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Conflict dose x post x drought estimate (kg C per m2)")
    axis.set_ylabel("Ordered placebo placement")
    axis.set_yticks(np.arange(1, len(ordered) + 1))
    axis.set_yticklabels(ordered["Placebo Placement ID"])
    axis.grid(axis="x", alpha=0.2)
    axis.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_features()
    base = base.loc[
        base["Candidate All 2008-2011 Nearest Event Distance km"].gt(60)
    ].copy()
    panel = pd.read_parquet(
        SOURCE_PANEL,
        columns=[
            "National Grid Cell ID",
            "Year",
            "Climate Cell ID",
            NPP_NATURAL,
            DRY_RAIN,
        ],
    ).loc[lambda frame: frame["Year"].between(2001, 2024)]
    panel = panel.dropna(subset=[NPP_NATURAL, DRY_RAIN]).copy()
    placebos = pd.read_csv(PLACEBO_UNIVERSE).loc[lambda frame: frame["Eligible"]].copy()
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    rows = []
    for _, placebo in placebos.iterrows():
        east1, north1 = to_utm.transform(
            placebo["First Longitude"], placebo["First Latitude"]
        )
        east2, north2 = to_utm.transform(
            placebo["Second Longitude"], placebo["Second Latitude"]
        )
        print(f"Estimating {placebo['Placebo Placement ID']}", flush=True)
        estimate = estimate_placement(base, panel, east1, north1, east2, north2)
        rows.append({**placebo.to_dict(), **estimate})
    estimates = pd.DataFrame(rows)
    estimates["Passes Estimation Support"] = (
        estimates["Maximum Absolute Weighted SMD"].le(BALANCE_THRESHOLD)
        & estimates["Treated 10 km Block ESS"].ge(MIN_TREATED_BLOCK_ESS)
    )
    actual_results = pd.read_csv(ACTUAL_RESULTS)
    actual = float(
        actual_results.loc[
            actual_results["Specification"].eq(
                "Primary: natural-unit NPP x dry-rainfall intensity"
            )
            & actual_results["Term"].eq(TARGET),
            "Estimate",
        ].iloc[0]
    )
    valid = estimates.loc[estimates["Passes Estimation Support"]].copy()
    more_negative = int((valid["Target Estimate kg C per m2"] <= actual).sum())
    more_absolute = int((valid["Target Estimate kg C per m2"].abs() >= abs(actual)).sum())
    one_sided_p = (1 + more_negative) / (1 + len(valid))
    two_sided_p = (1 + more_absolute) / (1 + len(valid))
    summary = pd.DataFrame(
        [
            {
                "Actual Target Estimate kg C per m2": actual,
                "Eligible Placebo Placements": len(placebos),
                "Support-Passing Placebo Placements": len(valid),
                "Placebos At Least As Negative As Actual": more_negative,
                "One-Sided Negative-Tail Rank p-value": one_sided_p,
                "Placebos At Least As Large in Absolute Value": more_absolute,
                "Two-Sided Absolute Rank p-value": two_sided_p,
                "Smallest Attainable One-Sided p-value": 1 / (1 + len(valid)),
                "Passes One-Sided 5%": bool(one_sided_p < 0.05),
            }
        ]
    )
    estimates.to_csv(ESTIMATES, index=False)
    summary.to_csv(SUMMARY, index=False)
    write_workbook(estimates, summary)
    plot_distribution(valid, actual)
    metadata = {
        "status": "first sequential NPP validation experiment",
        "human_approval_record": "MILI-D-20260822-015",
        "placebo_universe": "frozen Gate B two-sector translations along the Cambodia-Thailand frontier",
        "placebo_sample": "frontier cells more than 60 km from every actual 2008-2011 event",
        "model": "same NPP natural-unit dry-rainfall Gate C model with placement-specific conflict dose and overlap weights",
        "support_thresholds": {
            "maximum_absolute_weighted_SMD": BALANCE_THRESHOLD,
            "minimum_treated_10km_block_ESS": MIN_TREATED_BLOCK_ESS,
        },
        "rank_inference": "add-one finite-placebo correction",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nSpatial placebo inference")
    print(summary.to_string(index=False))
    print("\nPlacebo estimates")
    print(
        estimates[
            [
                "Placebo Placement ID",
                "Maximum Absolute Weighted SMD",
                "Treated 10 km Block ESS",
                "Target Estimate kg C per m2",
                "Model-based p-value",
                "Passes Estimation Support",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved {SUMMARY.relative_to(ROOT)}")
    print(f"Saved {WORKBOOK.relative_to(ROOT)}")
    print(f"Saved {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
