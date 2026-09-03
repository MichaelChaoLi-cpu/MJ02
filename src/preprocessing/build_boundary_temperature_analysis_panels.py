#!/usr/bin/env python3
"""Attach prospective temperature shocks to the existing local outcome panels.

The script preserves the existing rainfall and outcome releases.  It creates new
temperature-enhanced files so frozen rainfall analyses remain reproducible.
Only exact, pre-existing keys are used: CHIRPS cell/year/MODIS slot for the
high-frequency vegetation panel and village/year for the five-kilometre annual
spatial panel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VEGETATION_INPUT = Path(
    "data/processed/historical_boundary_16day_climate_vegetation_preprocessed.parquet"
)
INTERVAL_TEMPERATURE_INPUT = Path(
    "data/processed/historical_boundary_16day_temperature_shocks_preprocessed.parquet"
)
ANNUAL_SPATIAL_INPUT = Path(
    "data/processed/historical_boundary_annual_spatial_climate_preprocessed.parquet"
)
VILLAGE_TEMPERATURE_INPUT = Path(
    "data/processed/historical_boundary_village_year_temperature_shocks_preprocessed.parquet"
)
VEGETATION_OUTPUT = Path(
    "data/processed/"
    "historical_boundary_16day_climate_vegetation_temperature_preprocessed.parquet"
)
ANNUAL_OUTPUT = Path(
    "data/processed/"
    "historical_boundary_annual_local_temperature_analysis_preprocessed.parquet"
)
EXP_DIR = Path("data/exp/data-preprocessing/temperature-analysis-panels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args()


def project_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_high_frequency(root: Path) -> tuple[pd.DataFrame, dict[str, float | int]]:
    vegetation = pd.read_parquet(project_path(root, VEGETATION_INPUT))
    temperature = pd.read_parquet(project_path(root, INTERVAL_TEMPERATURE_INPUT))
    keys = ["CHIRPS Cell ID", "Year", "MODIS Composite Slot"]
    vegetation_windows = vegetation[
        keys
        + [
            "Composite Date",
            "Interval Rainfall mm",
            "Interval-Aligned Rainfall Anomaly Z",
        ]
    ].drop_duplicates(keys)
    comparison = vegetation_windows.merge(
        temperature[
            keys
            + [
                "Interval Start Date",
                "Interval Rainfall mm",
                "Interval Rainfall Anomaly Z",
            ]
        ],
        on=keys,
        how="left",
        suffixes=(" Existing", " Temperature"),
        validate="one_to_one",
    )
    if comparison["Interval Start Date"].isna().any():
        raise ValueError("At least one vegetation window lacks temperature exposure")
    date_mismatches = int(
        (comparison["Composite Date"] != comparison["Interval Start Date"]).sum()
    )
    rainfall_difference = (
        comparison["Interval Rainfall mm Existing"]
        - comparison["Interval Rainfall mm Temperature"]
    ).abs()
    rainfall_z_difference = (
        comparison["Interval-Aligned Rainfall Anomaly Z"]
        - comparison["Interval Rainfall Anomaly Z"]
    ).abs()
    if date_mismatches:
        raise ValueError(f"MODIS interval-date mismatches: {date_mismatches}")
    if float(rainfall_difference.max()) > 1e-10:
        raise ValueError("Temperature and vegetation releases use different interval rainfall")
    if float(rainfall_z_difference.max()) > 1e-10:
        raise ValueError("Temperature and vegetation releases use different rainfall z-scores")

    duplicate_temperature_columns = {
        "CHIRPS Cell Longitude",
        "CHIRPS Cell Latitude",
        "Interval Start Date",
        "Interval End Date",
        "Interval Rainfall mm",
        "Interval Rainfall mm Reference Mean",
        "Interval Rainfall mm Reference SD",
        "Interval Rainfall mm Reference Observations",
        "Interval Rainfall Anomaly Z",
    }
    keep = [
        column
        for column in temperature.columns
        if column in keys or column not in duplicate_temperature_columns
    ]
    enhanced = vegetation.merge(
        temperature[keep], on=keys, how="left", validate="many_to_one"
    )
    required = [
        "Interval Hot Day Count Anomaly Z",
        "Interval Heatwave Day Count Anomaly Z",
        "Interval Hot Night Count Anomaly Z",
        "Compound Hot-Dry Intensity",
    ]
    if enhanced[required].isna().any().any():
        raise ValueError("Temperature-enhanced vegetation panel has missing core shocks")
    diagnostics = {
        "vegetation_rows": len(vegetation),
        "enhanced_rows": len(enhanced),
        "unique_vegetation_windows": len(vegetation_windows),
        "matched_vegetation_windows": int(comparison["Interval Start Date"].notna().sum()),
        "date_mismatches": date_mismatches,
        "maximum_interval_rainfall_difference_mm": float(rainfall_difference.max()),
        "maximum_interval_rainfall_z_difference": float(rainfall_z_difference.max()),
    }
    return enhanced, diagnostics


def build_annual_local(root: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    spatial = pd.read_parquet(project_path(root, ANNUAL_SPATIAL_INPUT))
    temperature = pd.read_parquet(project_path(root, VILLAGE_TEMPERATURE_INPUT))
    local = spatial.loc[
        spatial["Historical-Boundary Common Support 5 km"].eq(1)
    ].copy()
    rename = {
        "May October Rainfall mm": "Temperature Grid May October Rainfall mm",
        "May October Rainfall mm Reference Mean": "Temperature Grid May October Rainfall Reference Mean",
        "May October Rainfall mm Reference SD": "Temperature Grid May October Rainfall Reference SD",
        "May October Rainfall mm Reference Observations": "Temperature Grid May October Rainfall Reference Observations",
        "May October Rainfall Anomaly Z": "Temperature Grid May October Rainfall Anomaly Z",
        "May October Dry Rainfall Intensity": "Temperature Grid May October Dry Rainfall Intensity",
    }
    temperature = temperature.drop(
        columns=["CHIRPS Cell Longitude", "CHIRPS Cell Latitude"]
    ).rename(columns=rename)
    enhanced = local.merge(
        temperature,
        on=["Village Code", "Year"],
        how="left",
        validate="many_to_one",
    )
    required = [
        "CHIRPS Cell ID",
        "May October Hot Day Count Anomaly Z",
        "May October Heatwave Day Count Anomaly Z",
        "May October Hot Night Count Anomaly Z",
        "May October Compound Hot-Dry Intensity",
    ]
    if enhanced[required].isna().any().any():
        missing = enhanced.loc[enhanced[required].isna().any(axis=1), ["Village Code", "Year"]]
        raise ValueError(
            f"Local annual panel has {len(missing)} rows without temperature exposure"
        )
    if len(enhanced) != len(local):
        raise ValueError("Annual temperature merge changed the frozen local row count")
    diagnostics = {
        "source_spatial_rows": len(spatial),
        "local_five_km_rows": len(local),
        "enhanced_rows": len(enhanced),
        "local_villages": int(enhanced["Village Code"].nunique()),
        "minimum_year": int(enhanced["Year"].min()),
        "maximum_year": int(enhanced["Year"].max()),
        "missing_core_temperature_rows": int(enhanced[required].isna().any(axis=1).sum()),
    }
    return enhanced, diagnostics


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    high_frequency, high_frequency_diagnostics = build_high_frequency(root)
    annual, annual_diagnostics = build_annual_local(root)
    outputs = {
        VEGETATION_OUTPUT: high_frequency,
        ANNUAL_OUTPUT: annual,
    }
    for relative, frame in outputs.items():
        path = project_path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        print(f"Saved: {path.relative_to(root)} rows={len(frame):,}")
    exp_dir = project_path(root, EXP_DIR)
    exp_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "high_frequency": high_frequency_diagnostics,
        "annual_local": annual_diagnostics,
        "scope_note": (
            "Temperature extraction is complete for the frozen five-kilometre "
            "village/vegetation support. Wider LongNTL grid cells require a "
            "separate outcome-blind grid expansion before temperature estimation."
        ),
    }
    (exp_dir / "linkage_validation.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    readme = """# Temperature-enhanced local analysis panels

These files attach prospective CHIRTS-ERA5 temperature shocks to the existing
five-kilometre historical-boundary outcome panels without overwriting the frozen
rainfall releases. The 16-day merge uses exact CHIRPS-cell, year, and MODIS-slot
keys. The annual merge uses exact village and year keys after restricting the
existing annual spatial panel to its pre-existing five-kilometre support.

The temperature-grid rainfall anomaly is explicitly labelled in the annual
panel because the existing annual spatial release uses a separate audited
climate-geography aggregation. Compound hot-dry intensity always uses the
temperature-grid rainfall series that was aligned day by day before aggregation.

This release does not yet cover the full 30-kilometre LongNTL grid. Expanding
temperature to that support must use pre-existing pixel coordinates and a new
source-window audit, not outcome estimates.
"""
    (exp_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
