#!/usr/bin/env python3
"""Acquire exact Cambodia pixel windows from the public Zenodo CCNL GeoTIFFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds


GRID = Path("data/processed/cambodia_national_1km_grid_preprocessed.parquet")
OUTPUT_DIR = Path("data/raw/independent_validation/ccnl_dmsp_cambodia")
ZENODO_RECORD = 6644980
ZENODO_DOI = "10.5281/zenodo.6644980"
YEARS = (2010, 2011, 2012, 2013)
SOURCE_MD5 = {
    2010: "30e35cc3d9f6c3a49d3013f4fe916bd8",
    2011: "c1caeb7fff1bec24f337d34255cd4d01",
    2012: "7cd361849cd6b7412dbf9eee37395447",
    2013: "bbce4a219b5806d6cfb1ce9f7f40d994",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enclosing_window(window: Window) -> Window:
    left = math.floor(window.col_off)
    top = math.floor(window.row_off)
    right = math.ceil(window.col_off + window.width)
    bottom = math.ceil(window.row_off + window.height)
    return Window(left, top, right - left, bottom - top)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    grid = pd.read_parquet(root / GRID, columns=["Longitude", "Latitude"])
    # A 0.05-degree guard band covers every national 1 km cell footprint and
    # the source pixels needed for area-average reprojection at the coastline.
    bounds = (
        float(grid["Longitude"].min() - 0.05),
        float(grid["Latitude"].min() - 0.05),
        float(grid["Longitude"].max() + 0.05),
        float(grid["Latitude"].max() + 0.05),
    )
    output_dir = root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for year in YEARS:
        filename = f"CCNL_DMSP_{year}_V1.tif"
        source_url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{filename}?download=1"
        destination = output_dir / f"CCNL_DMSP_{year}_V1_cambodia.tif"
        virtual_path = f"/vsicurl/{source_url}"
        print(f"Reading {year} Cambodia window from Zenodo", flush=True)
        with rasterio.Env(
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MAX_RETRY="5",
            GDAL_HTTP_RETRY_DELAY="2",
        ):
            with rasterio.open(virtual_path) as source:
                window = enclosing_window(from_bounds(*bounds, transform=source.transform))
                data = source.read(1, window=window)
                profile = source.profile.copy()
                profile.update(
                    driver="GTiff",
                    height=data.shape[0],
                    width=data.shape[1],
                    transform=source.window_transform(window),
                    count=1,
                    compress="DEFLATE",
                    predictor=3,
                    tiled=True,
                    blockxsize=128,
                    blockysize=128,
                )
                with rasterio.open(destination, "w", **profile) as target:
                    target.write(data, 1)
                record = {
                    "year": year,
                    "source_filename": filename,
                    "source_url": source_url,
                    "source_reported_md5": SOURCE_MD5[year],
                    "source_global_width": source.width,
                    "source_global_height": source.height,
                    "source_crs": source.crs.to_string(),
                    "source_transform": list(source.transform)[:6],
                    "source_nodata": source.nodata,
                    "subset_window": {
                        "column_offset": int(window.col_off),
                        "row_offset": int(window.row_off),
                        "width": int(window.width),
                        "height": int(window.height),
                    },
                    "local_file": str(destination.relative_to(root)),
                    "local_sha256": sha256(destination),
                    "local_bytes": destination.stat().st_size,
                    "pixel_values_modified": False,
                }
        records.append(record)
        print(f"Saved {destination.relative_to(root)} ({destination.stat().st_size / 1e6:.1f} MB)")

    metadata = {
        "dataset": "Consistent and Corrected Nighttime Light dataset (CCNL 1992-2013)",
        "record": ZENODO_RECORD,
        "doi": ZENODO_DOI,
        "license": "CC BY 4.0",
        "role": "interim independent-product validation of the Gate B LongNTL signal",
        "scope": "exact source pixel values in a guarded Cambodia window; GeoTIFF container recompressed",
        "requested_bounds_wgs84": list(bounds),
        "years": list(YEARS),
        "records": records,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (output_dir / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
