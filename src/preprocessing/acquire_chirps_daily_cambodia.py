#!/usr/bin/env python3
"""Extract nationwide Cambodia daily CHIRPS v2 precipitation by 0.05 degree cell.

Annual global NetCDF files are accessed with HTTP range requests. Only the
Cambodia bounding window is read; no global annual file is materialized locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


CELL_FILE = Path("data/processed/cambodia_national_005deg_climate_cells_preprocessed.parquet")
OUTPUT_DIR = Path("data/processed/cambodia_national_daily_precipitation")
AUDIT_DIR = Path("data/exp/data-preprocessing/national-satellite/chirps")
WEST_CENTER = -179.975
SOUTH_CENTER = -49.975
GRID_SIZE = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def source_url(year: int) -> str:
    return (
        "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/"
        f"chirps-v2.0.{year}.days_p05.nc"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cells(root: Path) -> pd.DataFrame:
    cells = pd.read_parquet(root / CELL_FILE).rename(
        columns={
            "Climate Cell Longitude": "CHIRPS Cell Longitude",
            "Climate Cell Latitude": "CHIRPS Cell Latitude",
        }
    )
    cells["CHIRPS Column Index"] = np.rint(
        (cells["CHIRPS Cell Longitude"] - WEST_CENTER) / GRID_SIZE
    ).astype(int)
    cells["CHIRPS Row Index"] = np.rint(
        (cells["CHIRPS Cell Latitude"] - SOUTH_CENTER) / GRID_SIZE
    ).astype(int)
    reconstructed_lon = WEST_CENTER + cells["CHIRPS Column Index"] * GRID_SIZE
    reconstructed_lat = SOUTH_CENTER + cells["CHIRPS Row Index"] * GRID_SIZE
    if not np.allclose(reconstructed_lon, cells["CHIRPS Cell Longitude"], atol=1e-8):
        raise ValueError("National climate and CHIRPS longitude grids do not align")
    if not np.allclose(reconstructed_lat, cells["CHIRPS Cell Latitude"], atol=1e-8):
        raise ValueError("National climate and CHIRPS latitude grids do not align")
    return cells[[
        "Climate Cell ID", "CHIRPS Cell Longitude", "CHIRPS Cell Latitude",
        "CHIRPS Column Index", "CHIRPS Row Index",
    ]].sort_values("Climate Cell ID").reset_index(drop=True)


def extract_year(cells: pd.DataFrame, year: int) -> pd.DataFrame:
    row_min = int(cells["CHIRPS Row Index"].min())
    row_max = int(cells["CHIRPS Row Index"].max())
    col_min = int(cells["CHIRPS Column Index"].min())
    col_max = int(cells["CHIRPS Column Index"].max())
    window = Window(col_min, row_min, col_max - col_min + 1, row_max - row_min + 1)
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    with rasterio.Env(
        GDAL_SKIP="netCDF",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".nc",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_TIMEOUT="180",
    ):
        with rasterio.open(f"/vsicurl/{source_url(year)}") as dataset:
            if (dataset.width, dataset.height) != (7200, 2000):
                raise ValueError(f"Unexpected CHIRPS grid for {year}: {dataset.width} x {dataset.height}")
            if dataset.count != len(dates):
                raise ValueError(f"Unexpected CHIRPS band count for {year}: {dataset.count}")
            values = dataset.read(window=window).astype("float32")
            nodata = dataset.nodata
    if nodata is not None:
        values[values == nodata] = np.nan
    parts: list[pd.DataFrame] = []
    for cell_id, longitude, latitude, column_index, row_index in cells.itertuples(index=False, name=None):
        local_row = int(row_index) - row_min
        local_col = int(column_index) - col_min
        parts.append(pd.DataFrame({
            "Climate Cell ID": cell_id,
            "Climate Cell Longitude": float(longitude),
            "Climate Cell Latitude": float(latitude),
            "Date": dates,
            "Daily Precipitation mm": values[:, local_row, local_col],
        }))
    return pd.concat(parts, ignore_index=True).sort_values(["Climate Cell ID", "Date"]).reset_index(drop=True)


def partition_valid(path: Path, year: int, cells: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path, columns=["Climate Cell ID", "Date"])
    except Exception:
        return False
    days = len(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"))
    return (
        len(frame) == cells * days
        and frame["Climate Cell ID"].nunique() == cells
        and not frame.duplicated(["Climate Cell ID", "Date"]).any()
    )


def acquire_year(cells: pd.DataFrame, year: int, path: Path, force: bool) -> tuple[int, Path, str]:
    if not force and partition_valid(path, year, len(cells)):
        return year, path, "reused"
    frame = extract_year(cells, year)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return year, path, "extracted"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    cells = load_cells(root)
    years = list(range(args.start_year, args.end_year + 1))
    paths = {year: root / OUTPUT_DIR / f"year={year}" / "precipitation.parquet" for year in years}
    manifest_rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(years))) as executor:
        futures = {
            executor.submit(acquire_year, cells, year, paths[year], args.force): year
            for year in years
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            year, path, status = future.result()
            precipitation = pd.read_parquet(path, columns=["Daily Precipitation mm"])["Daily Precipitation mm"]
            manifest_rows.append({
                "Year": year, "Source URL": source_url(year),
                "Local Partition": str(path.relative_to(root)), "SHA256": sha256(path),
                "Rows": len(precipitation), "Missing Rows": int(precipitation.isna().sum()),
                "Negative Rows": int((precipitation < 0).sum()), "Status": status,
            })
            print(f"Completed {completed}/{len(years)}: {year} {status}; rows={len(precipitation):,}", flush=True)
    audit = root / AUDIT_DIR
    audit.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifest_rows).sort_values("Year")
    manifest.to_csv(audit / "source_manifest.csv", index=False)
    metadata = {
        "dataset": "CHIRPS v2 daily precipitation",
        "coverage": "Cambodia national 0.05 degree climate cells",
        "years": years, "cells": len(cells),
        "source_access": "HTTP range extraction of Cambodia window from annual global NetCDF",
        "partitioning": "one Parquet file per calendar year",
        "missing_rule": "no imputation; structural source nodata retained",
    }
    (audit / "README.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
