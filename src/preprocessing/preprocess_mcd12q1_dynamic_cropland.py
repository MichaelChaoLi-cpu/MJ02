#!/usr/bin/env python3
"""Aggregate annual MCD12Q1 strict and inclusive cropland shares to the 1 km grid."""

from __future__ import annotations

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
MANIFEST = ROOT / "data/raw/land_cover/modis_mcd12q1_v061_dynamic/source_manifest.csv"
OUTPUT = ROOT / "data/processed/cambodia_national_annual_cropland_share_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/dynamic-modis-cropland"
STRICT_CODES = {12}
INCLUSIVE_CODES = {12, 14}


def target_grid(grid: pd.DataFrame) -> tuple[object, int, int, np.ndarray, np.ndarray]:
    west = float(grid["Grid West m"].min())
    north = float(grid["Grid North m"].max())
    width = int(round((grid["Grid East m"].max() - west) / 1000))
    height = int(round((north - grid["Grid South m"].min()) / 1000))
    transform = from_origin(west, north, 1000, 1000)
    row_index = (grid["Grid Row"].max() - grid["Grid Row"]).to_numpy(dtype=int)
    col_index = (grid["Grid Column"] - grid["Grid Column"].min()).to_numpy(dtype=int)
    return transform, width, height, row_index, col_index


def project_average(
    values: np.ndarray,
    source_transform: object,
    source_crs: object,
    target_transform: object,
    width: int,
    height: int,
) -> np.ndarray:
    destination = np.full((height, width), np.nan, dtype="float32")
    reproject(
        source=values.astype("float32", copy=False),
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=np.nan,
        dst_transform=target_transform,
        dst_crs="EPSG:32648",
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return destination


def main() -> None:
    if not GRID.exists() or not MANIFEST.exists():
        raise FileNotFoundError("National grid or dynamic MCD12Q1 manifest is missing")
    grid = pd.read_parquet(GRID).sort_values("National Grid Cell ID").reset_index(drop=True)
    manifest = pd.read_csv(MANIFEST).sort_values("year")
    years = tuple(manifest["year"].astype(int))
    if years != tuple(range(2001, 2022)):
        raise ValueError("Dynamic MCD12Q1 manifest must contain exactly 2001-2021")

    target_transform, width, height, row_index, col_index = target_grid(grid)
    identity = grid[["National Grid Cell ID"]].copy()
    pieces: list[pd.DataFrame] = []
    profile: tuple[object, object, int, int] | None = None
    for row in manifest.itertuples(index=False):
        path = ROOT / str(row.local_path)
        with rasterio.open(path) as dataset:
            observed = (dataset.crs, dataset.transform, dataset.width, dataset.height)
            if profile is None:
                profile = observed
            elif observed != profile:
                raise RuntimeError(f"MCD12Q1 clip grid changed across years: {path}")
            classification = dataset.read(1)
            valid = (classification >= 1) & (classification <= 17)
            strict = np.full(classification.shape, np.nan, dtype="float32")
            inclusive = np.full(classification.shape, np.nan, dtype="float32")
            strict[valid] = np.isin(classification[valid], list(STRICT_CODES)).astype("float32")
            inclusive[valid] = np.isin(
                classification[valid], list(INCLUSIVE_CODES)
            ).astype("float32")
            strict_grid = project_average(
                strict, dataset.transform, dataset.crs, target_transform, width, height
            )
            inclusive_grid = project_average(
                inclusive, dataset.transform, dataset.crs, target_transform, width, height
            )
        part = identity.copy()
        part["Year"] = int(row.year)
        part["Strict Cropland Share"] = strict_grid[row_index, col_index]
        part["Inclusive Agricultural Share"] = inclusive_grid[row_index, col_index]
        pieces.append(part)
        print(f"Aggregated dynamic MCD12Q1 {int(row.year)}", flush=True)

    output = pd.concat(pieces, ignore_index=True)
    if output.duplicated(["National Grid Cell ID", "Year"]).any():
        raise RuntimeError("Dynamic cropland output has duplicate cell-year keys")
    if not output[["Strict Cropland Share", "Inclusive Agricultural Share"]].apply(
        lambda series: series.dropna().between(0, 1).all()
    ).all():
        raise RuntimeError("Dynamic cropland shares fall outside zero to one")
    if (output["Inclusive Agricultural Share"] + 1e-6 < output["Strict Cropland Share"]).any():
        raise RuntimeError("Inclusive agricultural share is below strict cropland share")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False, compression="zstd")
    summary = (
        output.groupby("Year", observed=True)
        .agg(
            Cells=("National Grid Cell ID", "size"),
            Observed_Strict_Share=("Strict Cropland Share", lambda s: float(s.notna().mean())),
            Mean_Strict_Cropland_Share=("Strict Cropland Share", "mean"),
            Cells_With_Strict_Cropland=("Strict Cropland Share", lambda s: int(s.gt(0).sum())),
            Mean_Inclusive_Agricultural_Share=("Inclusive Agricultural Share", "mean"),
            Cells_With_Inclusive_Agriculture=(
                "Inclusive Agricultural Share", lambda s: int(s.gt(0).sum())
            ),
        )
        .reset_index()
    )
    summary.to_csv(AUDIT_DIR / "coverage_by_year.csv", index=False)
    metadata = {
        "dataset": "MODIS MCD12Q1.061 LC_Type1 annual cropland shares",
        "unit": "Cambodia national 1 km grid cell by year",
        "years": [2001, 2021],
        "strict_definition": "IGBP class 12 Croplands",
        "inclusive_definition": "IGBP classes 12 Croplands and 14 Cropland/Natural Vegetation Mosaics",
        "aggregation": "area-average 500 m class indicators to the stable 1 km grid",
        "temporal_rule": "same calendar year as each MOD13Q1 composite",
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "processed_at_utc": datetime.now(UTC).isoformat(),
    }
    (AUDIT_DIR / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
