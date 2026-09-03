#!/usr/bin/env python3
"""Aggregate GHS-POP R2023A year-2000 population to Cambodia's 1 km grid."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "data/processed/cambodia_national_1km_grid_preprocessed.parquet"
SOURCE_DIR = ROOT / "data/raw/population/ghs_pop_r2023a"
ARCHIVE = SOURCE_DIR / "GHS_POP_E2000_GLOBE_R2023A_54009_1000_V1_0.zip"
EXTRACTED_DIR = SOURCE_DIR / "extracted"
OUTPUT = ROOT / "data/processed/cambodia_national_baseline_population_2000_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/national-baseline-population-2000"

SOURCE_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_POP_GLOBE_R2023A/GHS_POP_E2000_GLOBE_R2023A_54009_1000/V1-0/"
    "GHS_POP_E2000_GLOBE_R2023A_54009_1000_V1_0.zip"
)
SOURCE_DATASET_DOI = "10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_source_raster() -> Path:
    candidates = [
        path
        for path in EXTRACTED_DIR.rglob("*.tif")
        if "GHS_POP_E2000_GLOBE_R2023A_54009_1000" in path.name
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one GHS-POP 2000 raster, found {candidates}")
    return candidates[0]


def target_grid(grid: pd.DataFrame) -> tuple[object, int, int, np.ndarray, np.ndarray]:
    west = float(grid["Grid West m"].min())
    north = float(grid["Grid North m"].max())
    width = int(round((grid["Grid East m"].max() - west) / 1000))
    height = int(round((north - grid["Grid South m"].min()) / 1000))
    transform = from_origin(west, north, 1000, 1000)
    row_index = (grid["Grid Row"].max() - grid["Grid Row"]).to_numpy(dtype=int)
    col_index = (grid["Grid Column"] - grid["Grid Column"].min()).to_numpy(dtype=int)
    return transform, width, height, row_index, col_index


def main() -> None:
    if not GRID.exists() or not ARCHIVE.exists():
        raise FileNotFoundError("National grid or GHS-POP archive is missing")
    source_raster = locate_source_raster()
    grid = pd.read_parquet(GRID)
    target_transform, width, height, row_index, col_index = target_grid(grid)

    destination = np.full((height, width), np.nan, dtype="float32")
    with rasterio.open(source_raster) as source:
        population = source.read(1).astype("float32")
        nodata = source.nodata if source.nodata is not None else -200
        population[(population == nodata) | (population < 0)] = np.nan
        reproject(
            source=population,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=np.nan,
            dst_transform=target_transform,
            dst_crs="EPSG:32648",
            dst_nodata=np.nan,
            resampling=Resampling.sum,
        )
        source_metadata = {
            "source_crs": source.crs.to_string(),
            "source_resolution": list(source.res),
            "source_nodata": source.nodata,
            "source_width": source.width,
            "source_height": source.height,
        }

    values = destination[row_index, col_index]
    output = grid[
        [
            "National Grid Cell ID",
            "Province Code",
            "Province Name",
            "District Code",
            "District Name",
            "Commune Code",
            "Commune Name",
        ]
    ].copy()
    output["Baseline Population 2000"] = values.astype("float32")
    output["Baseline Population Density per km2"] = values.astype("float32")
    output["Log Baseline Population 2000"] = np.log1p(values).astype("float32")
    output["Baseline Population Observed"] = np.isfinite(values)
    if output["National Grid Cell ID"].duplicated().any() or len(output) != 179_072:
        raise RuntimeError("Population output failed national grid-key validation")
    if output["Baseline Population Observed"].mean() < 0.99:
        raise RuntimeError("GHS-POP coverage unexpectedly falls below 99 percent")
    if np.nanmin(values) < 0:
        raise RuntimeError("GHS-POP output contains negative population")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False)
    observed = output.loc[output["Baseline Population Observed"], "Baseline Population 2000"]
    summary = pd.DataFrame(
        {
            "Metric": [
                "Grid cells",
                "Observed cells",
                "Observed share",
                "Estimated population sum",
                "Mean population per 1 km cell",
                "Median population per 1 km cell",
                "P90 population per 1 km cell",
                "P99 population per 1 km cell",
                "Maximum population per 1 km cell",
            ],
            "Value": [
                len(output),
                observed.size,
                observed.size / len(output),
                observed.sum(),
                observed.mean(),
                observed.median(),
                observed.quantile(0.90),
                observed.quantile(0.99),
                observed.max(),
            ],
        }
    )
    summary.to_csv(AUDIT_DIR / "baseline_population_summary.csv", index=False)
    metadata = {
        "panel_unit": "Cambodia national 1 km grid cell",
        "grid_cells": int(len(output)),
        "source": "GHS-POP R2023A year 2000 population grid derived from GPW4.11",
        "source_url": SOURCE_URL,
        "source_dataset_doi": SOURCE_DATASET_DOI,
        "source_archive_sha256": sha256(ARCHIVE),
        "source_raster": source_raster.relative_to(ROOT).as_posix(),
        **source_metadata,
        "target_crs": "EPSG:32648",
        "target_resolution_m": 1000,
        "aggregation_rule": "mass-preserving sum resampling of source population counts onto the aligned Cambodia 1 km grid",
        "timing_rule": "year-2000 predetermined baseline; later modeled population is not a time-varying control",
        "outcome_blind": True,
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
    }
    (AUDIT_DIR / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
