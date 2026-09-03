#!/usr/bin/env python3
"""Aggregate nationwide native LongNTL crops to the stable Cambodia 1 km grid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Prevent an unrelated user-level Anaconda PROJ database from entering rasterio.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject


GRID = Path("data/processed/cambodia_national_1km_grid_preprocessed.parquet")
SOURCE = Path("data/raw/independent_validation/longntl_v2_cambodia")
OUTPUT = Path("data/processed/cambodia_national_longntl_annual_preprocessed.parquet")
AUDIT_DIR = Path("data/exp/data-preprocessing/national-satellite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
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

    years = list(range(args.start_year, args.end_year + 1))
    paths = {year: root / SOURCE / f"longntl_v2_{year}_cambodia.tif" for year in years}
    missing = [str(path.relative_to(root)) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} nationwide LongNTL crops: {missing[:3]}")

    output_path = root / OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None
    coverage_rows = []
    try:
        for year in years:
            destination = np.full((height, width), np.nan, dtype="float32")
            with rasterio.open(paths[year]) as source:
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
            radiance = destination[raster_rows, raster_columns]
            frame = pd.DataFrame(
                {
                    "National Grid Cell ID": grid["National Grid Cell ID"].astype("string"),
                    "Year": np.full(len(grid), year, dtype="int16"),
                    "Annual NPP-VIIRS-like Radiance": radiance,
                    "Asinh Annual NPP-VIIRS-like Radiance": np.arcsinh(radiance),
                    "Any Nonzero Annual NPP-VIIRS-like Radiance": np.where(
                        np.isnan(radiance), np.nan, (radiance != 0).astype("float32")
                    ),
                    "LongNTL Source Stage": (
                        "Reconstructed" if year <= 2012 else "Observed composite"
                    ),
                    "Reconstructed LongNTL Indicator": np.int8(year <= 2012),
                }
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=100_000)
            coverage_rows.append(
                {
                    "Year": year,
                    "Grid Cells": len(frame),
                    "Observed Radiance": int(frame["Annual NPP-VIIRS-like Radiance"].notna().sum()),
                    "Nonzero Share": float(frame["Any Nonzero Annual NPP-VIIRS-like Radiance"].mean()),
                    "Mean Radiance": float(frame["Annual NPP-VIIRS-like Radiance"].mean()),
                }
            )
            print(f"Processed {year}: {coverage_rows[-1]['Observed Radiance']:,} observed cells", flush=True)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)

    audit = root / AUDIT_DIR
    audit.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(audit / "national_longntl_coverage_by_year.csv", index=False)
    metadata = {
        "source": "LongNTL Version 2 nationwide Cambodia native crops",
        "analysis_grid": "stable Cambodia 1 km EPSG:32648 grid",
        "years": years,
        "rows": int(len(grid) * len(years)),
        "grid_cells": len(grid),
        "aggregation": "area-average reprojection from native approximately 500 m cells",
        "missing_rule": "source nodata remains missing; zeros are retained",
    }
    (audit / "national_longntl_preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved: {OUTPUT}; rows={metadata['rows']:,}")


if __name__ == "__main__":
    main()
