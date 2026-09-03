#!/usr/bin/env python3
"""Link the Cambodia 1 km analysis grid to the native 0.05 degree climate grid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


GRID = Path("data/processed/cambodia_national_1km_grid_preprocessed.parquet")
CELL_OUTPUT = Path("data/processed/cambodia_national_005deg_climate_cells_preprocessed.parquet")
CROSSWALK_OUTPUT = Path("data/processed/cambodia_national_1km_to_climate_cell_preprocessed.parquet")
AUDIT_DIR = Path("data/exp/data-preprocessing/national-satellite")
GRID_SIZE = 0.05
LAT_ORIGIN = -50.0
LON_ORIGIN = -180.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    grid = pd.read_parquet(root / GRID, columns=["National Grid Cell ID", "Longitude", "Latitude"])
    grid["Climate Longitude Index"] = ((grid["Longitude"] - LON_ORIGIN) / GRID_SIZE).map(math.floor)
    grid["Climate Latitude Index"] = ((grid["Latitude"] - LAT_ORIGIN) / GRID_SIZE).map(math.floor)
    grid["Climate Cell Longitude"] = LON_ORIGIN + (grid["Climate Longitude Index"] + 0.5) * GRID_SIZE
    grid["Climate Cell Latitude"] = LAT_ORIGIN + (grid["Climate Latitude Index"] + 0.5) * GRID_SIZE
    grid["Climate Cell ID"] = (
        grid["Climate Longitude Index"].astype(str)
        + "_"
        + grid["Climate Latitude Index"].astype(str)
    )
    crosswalk = grid[
        [
            "National Grid Cell ID",
            "Climate Cell ID",
            "Climate Cell Longitude",
            "Climate Cell Latitude",
        ]
    ].copy()
    cells = (
        grid[
            [
                "Climate Cell ID",
                "Climate Longitude Index",
                "Climate Latitude Index",
                "Climate Cell Longitude",
                "Climate Cell Latitude",
            ]
        ]
        .drop_duplicates("Climate Cell ID")
        .sort_values("Climate Cell ID")
        .reset_index(drop=True)
    )
    (root / CELL_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(root / CELL_OUTPUT, index=False)
    crosswalk.to_parquet(root / CROSSWALK_OUTPUT, index=False)
    audit = root / AUDIT_DIR
    audit.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_grid_cells": len(grid),
        "unique_climate_cells": len(cells),
        "spatial_resolution_degrees": GRID_SIZE,
        "longitude_range": [float(cells["Climate Cell Longitude"].min()), float(cells["Climate Cell Longitude"].max())],
        "latitude_range": [float(cells["Climate Cell Latitude"].min()), float(cells["Climate Cell Latitude"].max())],
        "link_rule": "containing 0.05 degree climate cell based on the national 1 km cell centre",
    }
    (audit / "national_climate_grid_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved cells: {CELL_OUTPUT}; rows={len(cells):,}")
    print(f"Saved crosswalk: {CROSSWALK_OUTPUT}; rows={len(crosswalk):,}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
