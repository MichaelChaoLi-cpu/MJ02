#!/usr/bin/env python3
"""Aggregate nationwide MOD13Q1 EVI/NDVI clips to the stable Cambodia 1 km grid.

The output is partitioned by calendar year so that the roughly 100 million-row
high-frequency panel remains practical to query. Source pixels with MODIS pixel
reliability 0 (good) or 1 (marginal) are retained; no temporal imputation is
performed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


GRID_FILE = Path("data/processed/cambodia_national_1km_grid_preprocessed.parquet")
SOURCE_MANIFEST = Path("data/exp/data-preprocessing/national-satellite-source/modis-13Q1/source_manifest.csv")
OUTPUT_DIR = Path("data/processed/cambodia_national_modis_vegetation_16day")
AUDIT_DIR = Path("data/exp/data-preprocessing/national-satellite/modis-vegetation")

EVI = "250m_16_days_EVI"
NDVI = "250m_16_days_NDVI"
RELIABILITY = "250m_16_days_pixel_reliability"
ASSETS = {EVI, NDVI, RELIABILITY}
MIN_VALID_PIXELS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def target_geometry(grid: pd.DataFrame) -> tuple[object, int, int, np.ndarray, np.ndarray]:
    west = float(grid["Grid West m"].min())
    east = float(grid["Grid East m"].max())
    south = float(grid["Grid South m"].min())
    north = float(grid["Grid North m"].max())
    width = int(round((east - west) / 1000.0))
    height = int(round((north - south) / 1000.0))
    transform = from_origin(west, north, 1000.0, 1000.0)
    rows = np.rint((north - grid["Grid Centre Northing m"].to_numpy(float)) / 1000.0 - 0.5).astype(int)
    cols = np.rint((grid["Grid Centre Easting m"].to_numpy(float) - west) / 1000.0 - 0.5).astype(int)
    if rows.min() < 0 or rows.max() >= height or cols.min() < 0 or cols.max() >= width:
        raise ValueError("National grid indices fall outside the target raster")
    return transform, height, width, rows, cols


def add_reprojected_sum(
    source: np.ndarray,
    source_transform: object,
    source_crs: object,
    destination: np.ndarray,
    target_transform: object,
) -> None:
    temporary = np.zeros(destination.shape, dtype=np.float32)
    reproject(
        source=source.astype(np.float32, copy=False),
        destination=temporary,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs="EPSG:32648",
        resampling=Resampling.sum,
        init_dest_nodata=False,
    )
    destination += temporary


def aggregate_composite(
    date_rows: pd.DataFrame,
    target_transform: object,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    totals = {
        "candidate": np.zeros((height, width), dtype=np.float32),
        "good": np.zeros((height, width), dtype=np.float32),
        "marginal": np.zeros((height, width), dtype=np.float32),
        "evi_count": np.zeros((height, width), dtype=np.float32),
        "evi_sum": np.zeros((height, width), dtype=np.float32),
        "ndvi_count": np.zeros((height, width), dtype=np.float32),
        "ndvi_sum": np.zeros((height, width), dtype=np.float32),
    }
    for tile, tile_rows in date_rows.groupby(["horizontal_tile", "vertical_tile"], observed=True):
        if set(tile_rows["asset"]) != ASSETS or len(tile_rows) != 3:
            raise ValueError(f"Incomplete or duplicate assets for tile {tile}")
        paths = {row.asset: Path(row.absolute_path) for row in tile_rows.itertuples(index=False)}
        with rasterio.open(paths[RELIABILITY]) as rel_ds:
            reliability = rel_ds.read(1)
            source_transform, source_crs = rel_ds.transform, rel_ds.crs
            candidate = reliability != rel_ds.nodata
            good = np.isin(reliability, [0, 1])
            marginal = reliability == 1
            add_reprojected_sum(candidate, source_transform, source_crs, totals["candidate"], target_transform)
            add_reprojected_sum(good, source_transform, source_crs, totals["good"], target_transform)
            add_reprojected_sum(marginal, source_transform, source_crs, totals["marginal"], target_transform)
        for asset, prefix in ((EVI, "evi"), (NDVI, "ndvi")):
            with rasterio.open(paths[asset]) as dataset:
                raw = dataset.read(1)
                if dataset.transform != source_transform or dataset.crs != source_crs:
                    raise ValueError(f"Asset grid mismatch: {paths[asset]}")
                valid = good & (raw >= -2000) & (raw <= 10000)
                values = np.where(valid, raw.astype(np.float32) * 0.0001, 0.0)
                add_reprojected_sum(valid, source_transform, source_crs, totals[f"{prefix}_count"], target_transform)
                add_reprojected_sum(values, source_transform, source_crs, totals[f"{prefix}_sum"], target_transform)
    return totals


def build_year(
    year: int,
    manifest: pd.DataFrame,
    grid: pd.DataFrame,
    target_transform: object,
    height: int,
    width: int,
    land_rows: np.ndarray,
    land_cols: np.ndarray,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    year_manifest = manifest.loc[manifest["year"].eq(year)]
    for index, (date, date_rows) in enumerate(year_manifest.groupby("composite_date", observed=True, sort=True), start=1):
        totals = aggregate_composite(date_rows, target_transform, height, width)
        candidate = totals["candidate"][land_rows, land_cols]
        good = totals["good"][land_rows, land_cols]
        marginal = totals["marginal"][land_rows, land_cols]
        evi_count = totals["evi_count"][land_rows, land_cols]
        ndvi_count = totals["ndvi_count"][land_rows, land_cols]
        evi = np.divide(
            totals["evi_sum"][land_rows, land_cols], evi_count,
            out=np.full(len(grid), np.nan, dtype=np.float32), where=evi_count >= MIN_VALID_PIXELS,
        )
        ndvi = np.divide(
            totals["ndvi_sum"][land_rows, land_cols], ndvi_count,
            out=np.full(len(grid), np.nan, dtype=np.float32), where=ndvi_count >= MIN_VALID_PIXELS,
        )
        part = grid[[
            "National Grid Cell ID", "Longitude", "Latitude", "Province Code", "Province Name",
            "District Code", "District Name", "Commune Code", "Commune Name",
        ]].copy()
        part["Composite Date"] = pd.Timestamp(date)
        part["Year"] = year
        part["Day of Year"] = int(pd.Timestamp(date).dayofyear)
        part["MODIS Composite Slot"] = int((pd.Timestamp(date).dayofyear - 1) // 16 + 1)
        part["Candidate Pixel Count"] = np.rint(candidate).astype(np.int16)
        part["Good or Marginal Reliability Pixel Count"] = np.rint(good).astype(np.int16)
        part["Marginal Reliability Pixel Share"] = np.divide(
            marginal, candidate, out=np.full(len(grid), np.nan, dtype=np.float32), where=candidate > 0
        )
        part["EVI Valid Pixel Count"] = np.rint(evi_count).astype(np.int16)
        part["NDVI Valid Pixel Count"] = np.rint(ndvi_count).astype(np.int16)
        part["EVI Valid Pixel Share"] = np.divide(
            evi_count, candidate, out=np.full(len(grid), np.nan, dtype=np.float32), where=candidate > 0
        )
        part["NDVI Valid Pixel Share"] = np.divide(
            ndvi_count, candidate, out=np.full(len(grid), np.nan, dtype=np.float32), where=candidate > 0
        )
        part["Mean EVI"] = evi
        part["Mean NDVI"] = ndvi
        pieces.append(part)
        if index % 6 == 0 or index == year_manifest["composite_date"].nunique():
            print(f"{year}: aggregated {index}/{year_manifest['composite_date'].nunique()} composites", flush=True)
    if not pieces:
        raise ValueError(f"No source composites found for {year}")
    return pd.concat(pieces, ignore_index=True)


def valid_partition(path: Path, year: int, cells: int, composites: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path, columns=["National Grid Cell ID", "Composite Date", "Year"])
    except Exception:
        return False
    return (
        len(frame) == cells * composites
        and frame["National Grid Cell ID"].nunique() == cells
        and frame["Composite Date"].nunique() == composites
        and frame["Year"].eq(year).all()
        and not frame.duplicated(["National Grid Cell ID", "Composite Date"]).any()
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    grid = pd.read_parquet(root / GRID_FILE).sort_values("National Grid Cell ID").reset_index(drop=True)
    manifest = pd.read_csv(root / SOURCE_MANIFEST)
    manifest["composite_date"] = pd.to_datetime(manifest["composite_date"])
    manifest["absolute_path"] = manifest["local_path"].map(lambda value: root / str(value))
    if not manifest["absolute_path"].map(Path.exists).all():
        raise FileNotFoundError("At least one MODIS vegetation clip is missing")
    if set(manifest["asset"].unique()) != ASSETS:
        raise ValueError("Manifest does not contain the frozen EVI, NDVI, and reliability assets")
    target_transform, height, width, land_rows, land_cols = target_geometry(grid)
    audit_rows: list[dict[str, object]] = []
    for year in range(args.start_year, args.end_year + 1):
        expected = int(manifest.loc[manifest["year"].eq(year), "composite_date"].nunique())
        output = root / OUTPUT_DIR / f"year={year}" / "vegetation.parquet"
        if expected == 0:
            raise ValueError(f"Manifest has no composites for {year}")
        status = "reused"
        if args.force or not valid_partition(output, year, len(grid), expected):
            panel = build_year(year, manifest, grid, target_transform, height, width, land_rows, land_cols)
            output.parent.mkdir(parents=True, exist_ok=True)
            panel.to_parquet(output, index=False, compression="zstd")
            status = "aggregated"
        check = pd.read_parquet(output, columns=["National Grid Cell ID", "Composite Date", "Mean EVI", "Mean NDVI"])
        audit_rows.append({
            "Year": year, "Composites": expected, "Rows": len(check),
            "EVI Available Share": float(check["Mean EVI"].notna().mean()),
            "NDVI Available Share": float(check["Mean NDVI"].notna().mean()),
            "Status": status, "Partition": str(output.relative_to(root)),
        })
        print(f"Completed {year}: {status}; rows={len(check):,}", flush=True)
    audit = root / AUDIT_DIR
    audit.mkdir(parents=True, exist_ok=True)
    coverage = pd.DataFrame(audit_rows)
    coverage.to_csv(audit / "coverage_by_year.csv", index=False)
    metadata = {
        "dataset": "MODIS MOD13Q1.061 nationwide Cambodia EVI and NDVI",
        "target_grid": "stable Cambodia EPSG:32648 1 km land-cell grid",
        "years": [args.start_year, args.end_year],
        "quality_rule": "pixel reliability 0 or 1; raw index range -2000 to 10000; at least four valid pixels",
        "temporal_imputation": "none",
        "partitioning": "one Parquet file per calendar year",
    }
    (audit / "README.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
