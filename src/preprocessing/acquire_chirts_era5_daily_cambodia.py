#!/usr/bin/env python3
"""Extract nationwide Cambodia daily CHIRTS-ERA5 Tmax and Tmin by 0.05 degree cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from acquire_chirts_era5_daily_boundary_panel import (
    GRID_SIZE,
    SOUTH_CENTER,
    WEST_CENTER,
    extract_year,
    source_url,
)


CELL_FILE = Path("data/processed/cambodia_national_005deg_climate_cells_preprocessed.parquet")
OUTPUT_DIR = Path("data/processed/cambodia_national_daily_temperature")
AUDIT_DIR = Path("data/exp/data-preprocessing/national-satellite/chirts-era5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cells(root: Path) -> pd.DataFrame:
    cells = pd.read_parquet(root / CELL_FILE).rename(
        columns={
            "Climate Cell ID": "CHIRPS Cell ID",
            "Climate Cell Longitude": "CHIRPS Cell Longitude",
            "Climate Cell Latitude": "CHIRPS Cell Latitude",
        }
    )
    cells["CHIRTS Column Index"] = np.rint(
        (cells["CHIRPS Cell Longitude"] - WEST_CENTER) / GRID_SIZE
    ).astype(int)
    cells["CHIRTS Row Index"] = np.rint(
        (cells["CHIRPS Cell Latitude"] - SOUTH_CENTER) / GRID_SIZE
    ).astype(int)
    reconstructed_lon = WEST_CENTER + cells["CHIRTS Column Index"] * GRID_SIZE
    reconstructed_lat = SOUTH_CENTER + cells["CHIRTS Row Index"] * GRID_SIZE
    if not np.allclose(reconstructed_lon, cells["CHIRPS Cell Longitude"], atol=1e-8):
        raise ValueError("National climate and CHIRTS longitude grids do not align")
    if not np.allclose(reconstructed_lat, cells["CHIRPS Cell Latitude"], atol=1e-8):
        raise ValueError("National climate and CHIRTS latitude grids do not align")
    return cells[
        [
            "CHIRPS Cell ID",
            "CHIRPS Cell Longitude",
            "CHIRPS Cell Latitude",
            "CHIRTS Column Index",
            "CHIRTS Row Index",
        ]
    ].sort_values("CHIRPS Cell ID").reset_index(drop=True)


def annual_valid(path: Path, year: int, cell_count: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    days = len(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"))
    return (
        len(frame) == days * cell_count
        and frame["Climate Cell ID"].nunique() == cell_count
        and not frame.duplicated(["Climate Cell ID", "Date"]).any()
    )


def acquire_year(cells: pd.DataFrame, year: int, path: Path, force: bool) -> tuple[int, Path, str]:
    if not force and annual_valid(path, year, len(cells)):
        return year, path, "reused"
    frame = extract_year(cells, year, allow_structural_missing=True).rename(
        columns={
            "CHIRPS Cell ID": "Climate Cell ID",
            "CHIRPS Cell Longitude": "Climate Cell Longitude",
            "CHIRPS Cell Latitude": "Climate Cell Latitude",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return year, path, "extracted"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    cells = load_cells(root)
    years = list(range(args.start_year, args.end_year + 1))
    paths = {
        year: root / OUTPUT_DIR / f"year={year}" / "temperature.parquet"
        for year in years
    }
    manifests = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(years))) as executor:
        futures = {
            executor.submit(acquire_year, cells, year, paths[year], args.force): year
            for year in years
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            year, path, status = future.result()
            rows = len(pd.read_parquet(path, columns=["Date"]))
            manifests.append(
                {
                    "Year": year,
                    "Tmax URL": source_url("tmax", year),
                    "Tmin URL": source_url("tmin", year),
                    "Local Partition": str(path.relative_to(root)),
                    "SHA256": sha256(path),
                    "Rows": rows,
                    "Missing Tmax Rows": int(pd.read_parquet(path, columns=["Daily Maximum Temperature C"])["Daily Maximum Temperature C"].isna().sum()),
                    "Missing Tmin Rows": int(pd.read_parquet(path, columns=["Daily Minimum Temperature C"])["Daily Minimum Temperature C"].isna().sum()),
                    "Status": status,
                }
            )
            print(f"Completed {completed}/{len(years)}: {year} {status}; rows={rows:,}", flush=True)
    manifest = pd.DataFrame(manifests).sort_values("Year")
    audit = root / AUDIT_DIR
    audit.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(audit / "source_manifest.csv", index=False)
    metadata = {
        "dataset": "CHIRTS-ERA5 daily Tmax and Tmin",
        "coverage": "Cambodia national 0.05 degree climate cells",
        "years": years,
        "cells": len(cells),
        "partitioning": "one Parquet file per calendar year",
        "rows": int(manifest["Rows"].sum()),
        "missing_rule": "no imputation; structural source nodata is retained and counted in the manifest",
    }
    (audit / "README.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
