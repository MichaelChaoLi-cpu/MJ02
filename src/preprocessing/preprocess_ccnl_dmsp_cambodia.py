#!/usr/bin/env python3
"""Aggregate public CCNL Cambodia crops to the stable national 1 km grid."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


GRID = Path("data/processed/cambodia_national_1km_grid_preprocessed.parquet")
SOURCE_DIR = Path("data/raw/independent_validation/ccnl_dmsp_cambodia")
OUTPUT = Path("data/processed/cambodia_national_ccnl_dmsp_annual_preprocessed.parquet")
AUDIT_DIR = Path("data/exp/data-preprocessing/ccnl-dmsp")
YEARS = (2010, 2011, 2012, 2013)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    grid = pd.read_parquet(root / GRID)
    min_column = int(grid["Grid Column"].min())
    max_column = int(grid["Grid Column"].max())
    min_row = int(grid["Grid Row"].min())
    max_row = int(grid["Grid Row"].max())
    width = max_column - min_column + 1
    height = max_row - min_row + 1
    transform = from_origin(min_column * 1_000, (max_row + 1) * 1_000, 1_000, 1_000)
    raster_rows = max_row - grid["Grid Row"].to_numpy(int)
    raster_columns = grid["Grid Column"].to_numpy(int) - min_column
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []

    for year in YEARS:
        source_path = root / SOURCE_DIR / f"CCNL_DMSP_{year}_V1_cambodia.tif"
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        destination = np.full((height, width), np.nan, dtype="float64")
        with rasterio.open(source_path) as source:
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=transform,
                dst_crs="EPSG:32648",
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
        values = destination[raster_rows, raster_columns]
        positive = np.where(np.isnan(values), np.nan, (values > 0).astype("float32"))
        frame = pd.DataFrame(
            {
                "National Grid Cell ID": grid["National Grid Cell ID"].astype("string"),
                "Year": np.full(len(grid), year, dtype="int16"),
                "CCNL DMSP Corrected DN": values,
                "Asinh CCNL DMSP Corrected DN": np.arcsinh(values),
                "Any Positive CCNL DMSP Corrected DN": positive,
                "CCNL Product Version": "V1",
            }
        )
        frames.append(frame)
        coverage_rows.append(
            {
                "Year": year,
                "Grid Cells": len(frame),
                "Observed Cells": int(frame["CCNL DMSP Corrected DN"].notna().sum()),
                "Positive Share": float(frame["Any Positive CCNL DMSP Corrected DN"].mean()),
                "Mean Corrected DN": float(frame["CCNL DMSP Corrected DN"].mean()),
                "Maximum Corrected DN": float(frame["CCNL DMSP Corrected DN"].max()),
            }
        )
        print(f"Processed CCNL {year}", flush=True)

    output = pd.concat(frames, ignore_index=True)
    output_path = root / OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    audit_dir = root / AUDIT_DIR
    audit_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(audit_dir / "coverage_and_distribution_by_year.csv", index=False)
    decisions = {
        "status": "interim independent-product validation; not promoted into AnaSOP",
        "selected_variables": [
            "National Grid Cell ID",
            "Year",
            "CCNL DMSP Corrected DN",
            "Asinh CCNL DMSP Corrected DN",
            "Any Positive CCNL DMSP Corrected DN",
        ],
        "readable_name_rule": "CCNL and DMSP are retained as published product/platform names; DN is identified as corrected digital number",
        "aggregation": "area-average reprojection from 30 arc-second WGS84 pixels to stable EPSG:32648 national 1 km cells",
        "zeros": "retained as observed zero light",
        "missing": "source nodata remains missing",
        "omission": "CCNL does not distribute the native F18 cloud-free observation count; native coverage validation remains deferred",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (audit_dir / "decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "source": "Zenodo CCNL V1, DOI 10.5281/zenodo.6644980",
        "years": list(YEARS),
        "national_grid_cells": len(grid),
        "rows": len(output),
        "output": str(OUTPUT),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (audit_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Saved {OUTPUT}; rows={len(output):,}")


if __name__ == "__main__":
    main()
