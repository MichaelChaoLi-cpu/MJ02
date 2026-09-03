#!/usr/bin/env python3
"""Aggregate nationwide MODIS annual NPP and quality to the Cambodia 1 km grid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
SOURCE = Path("data/exp/data-preprocessing/national-satellite-source/modis-npp")
OUTPUT = Path("data/processed/cambodia_national_modis_npp_annual_preprocessed.parquet")
AUDIT_DIR = Path("data/exp/data-preprocessing/national-satellite")
NPP_SCALE = 0.0001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2024)
    return parser.parse_args()


def grid_layout(grid: pd.DataFrame) -> tuple[object, int, int, np.ndarray, np.ndarray]:
    min_column, max_column = int(grid["Grid Column"].min()), int(grid["Grid Column"].max())
    min_row, max_row = int(grid["Grid Row"].min()), int(grid["Grid Row"].max())
    transform = from_origin(min_column * 1_000, (max_row + 1) * 1_000, 1_000, 1_000)
    rows = max_row - grid["Grid Row"].to_numpy(int)
    columns = grid["Grid Column"].to_numpy(int) - min_column
    return transform, max_column - min_column + 1, max_row - min_row + 1, rows, columns


def reproject_tiles(
    paths: list[Path],
    transform: object,
    width: int,
    height: int,
) -> np.ndarray:
    destination = np.full((height, width), np.nan, dtype="float32")
    for path in paths:
        with rasterio.open(path) as source:
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
                init_dest_nodata=False,
            )
    return destination


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    grid = pd.read_parquet(root / GRID)
    transform, width, height, raster_rows, raster_columns = grid_layout(grid)
    years = list(range(args.start_year, args.end_year + 1))
    source = root / SOURCE / "clips"
    npp_paths = {year: sorted(source.glob(f"{year}_h??v??_Npp_500m.tif")) for year in years}
    qc_paths = {year: sorted(source.glob(f"{year}_h??v??_Npp_QC_500m.tif")) for year in years}
    missing = [year for year in years if not npp_paths[year] or not qc_paths[year]]
    if missing:
        raise FileNotFoundError(f"Missing nationwide NPP/QC tiles for years: {missing}")

    baseline_years = [year for year in years if 2001 <= year <= 2020]
    if len(baseline_years) != 20:
        raise ValueError("The requested range must contain the complete 2001-2020 NPP baseline")
    baseline_values = []
    for year in baseline_years:
        raster = reproject_tiles(npp_paths[year], transform, width, height)
        baseline_values.append(raster[raster_rows, raster_columns] * NPP_SCALE)
        print(f"Baseline pass {year}", flush=True)
    baseline = np.vstack(baseline_values)
    valid_counts = np.sum(~np.isnan(baseline), axis=0)
    baseline_mean = np.nanmean(baseline, axis=0)
    baseline_sd = np.nanstd(baseline, axis=0, ddof=1)

    output_path = root / OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".parquet.tmp")
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None
    coverage_rows = []
    try:
        for year in years:
            npp_raster = reproject_tiles(npp_paths[year], transform, width, height)
            qc_raster = reproject_tiles(qc_paths[year], transform, width, height)
            npp = npp_raster[raster_rows, raster_columns] * NPP_SCALE
            quality = qc_raster[raster_rows, raster_columns]
            anomaly = npp - baseline_mean
            anomaly_z = np.divide(
                anomaly,
                baseline_sd,
                out=np.full_like(anomaly, np.nan),
                where=baseline_sd > 0,
            )
            frame = pd.DataFrame(
                {
                    "National Grid Cell ID": grid["National Grid Cell ID"].astype("string"),
                    "Year": np.full(len(grid), year, dtype="int16"),
                    "Annual Land NPP Mean kg C per m2": npp,
                    "Mean NPP QC Filled Growing-Season Days Percent": quality,
                    "Grid 2001-2020 Mean Annual Land NPP kg C per m2": baseline_mean,
                    "Grid 2001-2020 SD Annual Land NPP kg C per m2": baseline_sd,
                    "Grid 2001-2020 Valid NPP Year Count": valid_counts.astype("int8"),
                    "Annual Land NPP Anomaly kg C per m2": anomaly,
                    "Annual Land NPP Anomaly Z 2001-2020": anomaly_z,
                    "NPP Complete 2001-2020 Baseline": (valid_counts == 20).astype("int8"),
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
                    "Observed NPP": int(frame["Annual Land NPP Mean kg C per m2"].notna().sum()),
                    "Complete Baseline Cells": int(frame["NPP Complete 2001-2020 Baseline"].sum()),
                    "Mean NPP kg C per m2": float(frame["Annual Land NPP Mean kg C per m2"].mean()),
                }
            )
            print(f"Output pass {year}: {coverage_rows[-1]['Observed NPP']:,} observed cells", flush=True)
    finally:
        if writer is not None:
            writer.close()
    temporary.replace(output_path)

    audit = root / AUDIT_DIR
    audit.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(audit / "national_modis_npp_coverage_by_year.csv", index=False)
    metadata = {
        "source": "MOD17A3HGF Version 6.1 nationwide native clips",
        "years": years,
        "rows": int(len(grid) * len(years)),
        "grid_cells": len(grid),
        "aggregation": "area-average reprojection to the stable Cambodia 1 km grid",
        "npp_scale": NPP_SCALE,
        "reference_period": [2001, 2020],
        "quality_rule": "retain the annual percentage of filled growing-season days as a continuous quality field",
    }
    (audit / "national_modis_npp_preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved: {OUTPUT}; rows={metadata['rows']:,}")


if __name__ == "__main__":
    main()
