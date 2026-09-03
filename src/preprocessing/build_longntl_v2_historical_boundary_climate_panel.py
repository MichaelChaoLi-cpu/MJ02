#!/usr/bin/env python3
"""Link the LongNTL Version 2 panel to the frozen boundary and rainfall design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_viirs_historical_boundary_climate_panel import (
    ABS_DISTANCE,
    BANDWIDTHS_KM,
    COMMUNE,
    SEGMENT,
    SIDE,
    TREATMENT,
    build_spatial_crosswalk,
)


YEARS = tuple(range(2000, 2025))
CELL = "Grid Cell ID"
LONG_CELL = "LongNTL Grid Cell ID"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--longntl",
        type=Path,
        default=Path("data/processed/longntl_v2_pixel_year_preprocessed.parquet"),
    )
    parser.add_argument(
        "--climate",
        type=Path,
        default=Path("data/processed/chirps_long_baseline_commune_year_preprocessed.parquet"),
    )
    parser.add_argument(
        "--boundary-source",
        type=Path,
        default=Path("data/exp/data-preprocessing/historical-boundary-source"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/longntl_v2_historical_boundary_climate_preprocessed.parquet"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/exp/data-preprocessing/longntl-v2-boundary-climate"),
    )
    return parser.parse_args()


def resolved(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def make_audits(crosswalk: pd.DataFrame, panel: pd.DataFrame, output: Path) -> None:
    support_records = []
    for bandwidth in BANDWIDTHS_KM:
        supported = crosswalk.loc[
            crosswalk[f"Historical-Boundary Common Support {bandwidth} km"].eq(1)
        ]
        commune_sides = supported.groupby(COMMUNE, observed=True)[TREATMENT].nunique()
        for side, group in supported.groupby(SIDE, observed=True):
            support_records.append(
                {
                    "Bandwidth km": bandwidth,
                    "Historical Repression Side": side,
                    "Unique Grid Cells": group[CELL].nunique(),
                    "Boundary Segments": group[SEGMENT].nunique(),
                    "Climate Communes": group[COMMUNE].nunique(),
                    "Cross-Side Climate Communes": int(commune_sides.eq(2).sum()),
                }
            )
    pd.DataFrame(support_records).to_csv(
        output / "longntl_boundary_linkage_coverage.csv", index=False
    )

    primary = panel.loc[panel["Historical-Boundary Common Support 5 km"].eq(1)]
    year_coverage = (
        primary.groupby(["Year", "LongNTL Source Stage"], observed=True)
        .agg(
            cell_years=(CELL, "size"),
            grid_cells=(CELL, "nunique"),
            rainfall_linked=("Annual Climate Shock Available", "sum"),
            observed_radiance=("Annual NPP-VIIRS-like Radiance", "count"),
        )
        .reset_index()
    )
    year_coverage["rainfall_linkage_share"] = (
        year_coverage["rainfall_linked"] / year_coverage["cell_years"]
    )
    year_coverage.to_csv(output / "primary_support_by_year.csv", index=False)

    metadata = {
        "spatial_unit": "fixed approximately 500 metre LongNTL grid-cell centre",
        "period": [YEARS[0], YEARS[-1]],
        "reconstructed_period": [2000, 2012],
        "observed_composite_period": [2013, 2024],
        "rainfall_available_period": [2000, 2024],
        "primary_bandwidth_km": 5,
        "alternative_bandwidths_km": [2, 10, 15, 20, 30],
        "outcome_coefficients_inspected": False,
        "interpretation": (
            "Full 2000-2024 panel supports levels, trends, and rainfall-response models "
            "on one balanced grid-cell-by-year panel."
        ),
    }
    (output / "decisions.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    longntl_path = resolved(root, args.longntl)
    climate_path = resolved(root, args.climate)
    source = resolved(root, args.boundary_source)
    output_path = resolved(root, args.output)
    audit_output = resolved(root, args.audit_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_output.mkdir(parents=True, exist_ok=True)

    longntl = pd.read_parquet(longntl_path).rename(columns={LONG_CELL: CELL})
    cells = longntl[[CELL, "Grid Row", "Grid Column", "Longitude", "Latitude"]].drop_duplicates(
        CELL
    )
    if len(cells) * len(YEARS) != len(longntl):
        raise RuntimeError("LongNTL input is not a balanced 2000-2024 cell-year panel")
    crosswalk = build_spatial_crosswalk(root, cells, source)

    climate = pd.read_parquet(climate_path)
    climate["Climate Geography Code"] = climate["Climate Geography Code"].astype("string").str.zfill(6)
    climate = climate.loc[climate["Year"].isin(YEARS)].copy()
    if climate.duplicated(["Climate Geography Code", "Year"]).any():
        raise RuntimeError("Climate commune-year keys are not unique")

    panel = longntl.merge(
        crosswalk.drop(columns=["Grid Row", "Grid Column", "Longitude", "Latitude"]),
        on=CELL,
        how="left",
        validate="many_to_one",
    )
    panel = panel.merge(
        climate,
        left_on=[COMMUNE, "Year"],
        right_on=["Climate Geography Code", "Year"],
        how="left",
        validate="many_to_one",
    ).drop(columns="Climate Geography Code")
    panel["Annual Climate Shock Available"] = (
        panel["Annual Rainfall Anomaly Z (1991-2020)"].notna().astype("Int8")
    )
    if len(panel) != len(longntl) or panel.duplicated([CELL, "Year"]).any():
        raise RuntimeError("Spatial-climate linkage changed LongNTL panel keys")

    primary = panel["Historical-Boundary Common Support 5 km"].eq(1)
    expected_rainfall = primary & panel["Year"].between(2000, 2024)
    if not panel.loc[expected_rainfall, "Annual Climate Shock Available"].eq(1).all():
        raise RuntimeError("Rainfall linkage is incomplete inside 5 km support in 2000-2024")

    panel.to_parquet(output_path, index=False)
    crosswalk.to_csv(audit_output / "longntl_spatial_crosswalk.csv", index=False)
    make_audits(crosswalk, panel, audit_output)
    print(f"Wrote {len(panel):,} rows to {output_path}")
    print(f"Five-kilometre support: {int(primary.sum()):,} cell-years")
    print(f"Rainfall-linked five-kilometre support: {int(expected_rainfall.sum()):,} cell-years")


if __name__ == "__main__":
    main()
