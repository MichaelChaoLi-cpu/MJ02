#!/usr/bin/env python3
"""Aggregate 2001-2007 MCD12Q1 land-cover shares to Cambodia's 1 km grid."""

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
MANIFEST = ROOT / "data/raw/land_cover/modis_mcd12q1_v061/source_manifest.csv"
OUTPUT = ROOT / "data/processed/cambodia_national_baseline_land_cover_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/national-baseline-land-cover"
YEARS = tuple(range(2001, 2008))

CLASS_GROUPS = {
    "Cropland": {12, 14},
    "Forest": {1, 2, 3, 4, 5},
    "Grass Shrub": {6, 7, 8, 9, 10},
    "Built": {13},
    "Water Wetland": {11, 17},
    "Other": {15, 16},
}


def target_grid(grid: pd.DataFrame) -> tuple[object, int, int, np.ndarray, np.ndarray]:
    west = float(grid["Grid West m"].min())
    north = float(grid["Grid North m"].max())
    width = int(round((grid["Grid East m"].max() - west) / 1000))
    height = int(round((north - grid["Grid South m"].min()) / 1000))
    transform = from_origin(west, north, 1000, 1000)
    row_index = (grid["Grid Row"].max() - grid["Grid Row"]).to_numpy(dtype=int)
    col_index = (grid["Grid Column"] - grid["Grid Column"].min()).to_numpy(dtype=int)
    if row_index.min() < 0 or row_index.max() >= height:
        raise RuntimeError("Grid row indices fall outside target raster")
    if col_index.min() < 0 or col_index.max() >= width:
        raise RuntimeError("Grid column indices fall outside target raster")
    return transform, width, height, row_index, col_index


def project_average(
    values: np.ndarray,
    source_crs: object,
    source_transform: object,
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


def detailed_stability(classifications: np.ndarray) -> np.ndarray:
    valid = (classifications >= 1) & (classifications <= 17)
    valid_count = valid.sum(axis=0)
    maximum_count = np.zeros(valid_count.shape, dtype="uint8")
    for code in range(1, 18):
        maximum_count = np.maximum(maximum_count, (classifications == code).sum(axis=0))
    stability = np.full(valid_count.shape, np.nan, dtype="float32")
    np.divide(maximum_count, valid_count, out=stability, where=valid_count > 0)
    return stability


def main() -> None:
    if not GRID.exists() or not MANIFEST.exists():
        raise FileNotFoundError("National grid or MCD12Q1 manifest is missing")
    grid = pd.read_parquet(GRID)
    manifest = pd.read_csv(MANIFEST).sort_values("year")
    if tuple(manifest["year"].astype(int)) != YEARS:
        raise ValueError(f"MCD12Q1 manifest must contain exactly {YEARS}")

    target_transform, width, height, row_index, col_index = target_grid(grid)
    annual_group_grids: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_GROUPS}
    annual_source: list[np.ndarray] = []
    source_profile: tuple[object, object, int, int] | None = None

    for row in manifest.itertuples(index=False):
        path = ROOT / str(row.local_path)
        with rasterio.open(path) as dataset:
            observed_profile = (dataset.crs, dataset.transform, dataset.width, dataset.height)
            if source_profile is None:
                source_profile = observed_profile
            elif observed_profile != source_profile:
                raise RuntimeError(f"MCD12Q1 clip grid changed across years: {path}")
            classification = dataset.read(1)
            valid = (classification >= 1) & (classification <= 17)
            annual_source.append(classification)
            for group, codes in CLASS_GROUPS.items():
                mask = np.full(classification.shape, np.nan, dtype="float32")
                mask[valid] = np.isin(classification[valid], list(codes)).astype("float32")
                annual_group_grids[group].append(
                    project_average(
                        mask,
                        dataset.crs,
                        dataset.transform,
                        target_transform,
                        width,
                        height,
                    )
                )
        print(f"Aggregated MCD12Q1 {int(row.year)}")

    if source_profile is None:
        raise RuntimeError("No MCD12Q1 rasters were read")
    source_crs, source_transform, _, _ = source_profile
    source_stack = np.stack(annual_source)
    stability_grid = project_average(
        detailed_stability(source_stack),
        source_crs,
        source_transform,
        target_transform,
        width,
        height,
    )

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
    share_columns: list[str] = []
    for group, grids in annual_group_grids.items():
        baseline = np.nanmean(np.stack(grids), axis=0)
        column = f"Baseline {group} Share"
        output[column] = baseline[row_index, col_index].astype("float32")
        share_columns.append(column)
    output["Baseline Land Cover Stability Share"] = stability_grid[
        row_index, col_index
    ].astype("float32")
    output["Baseline Land Cover Observed"] = output[share_columns].notna().all(axis=1)
    output["Baseline Classified Share Sum"] = output[share_columns].sum(axis=1, min_count=1)

    main_groups = [
        "Baseline Cropland Share",
        "Baseline Forest Share",
        "Baseline Grass Shrub Share",
        "Baseline Built Share",
        "Baseline Water Wetland Share",
        "Baseline Other Share",
    ]
    output["Baseline Dominant Land Cover Group"] = (
        output[main_groups]
        .idxmax(axis=1)
        .str.removeprefix("Baseline ")
        .str.removesuffix(" Share")
        .astype("string")
    )

    if output["National Grid Cell ID"].duplicated().any() or len(output) != len(grid):
        raise RuntimeError("Land-cover output failed national grid-key validation")
    observed = output["Baseline Land Cover Observed"]
    if observed.mean() < 0.99:
        raise RuntimeError("Land-cover coverage unexpectedly falls below 99 percent")
    share_error = (output.loc[observed, "Baseline Classified Share Sum"] - 1).abs()
    if share_error.quantile(0.99) > 0.02:
        raise RuntimeError("Land-cover class shares do not sum to one within tolerance")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False)

    summary_rows = []
    for column in share_columns + ["Baseline Land Cover Stability Share"]:
        series = output[column]
        summary_rows.append(
            {
                "Variable": column,
                "Observed Cells": int(series.notna().sum()),
                "Mean": float(series.mean()),
                "SD": float(series.std()),
                "P10": float(series.quantile(0.10)),
                "Median": float(series.median()),
                "P90": float(series.quantile(0.90)),
                "Minimum": float(series.min()),
                "Maximum": float(series.max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(AUDIT_DIR / "baseline_land_cover_summary.csv", index=False)
    (
        output["Baseline Dominant Land Cover Group"]
        .value_counts(dropna=False)
        .rename_axis("Dominant Land Cover Group")
        .reset_index(name="Grid Cells")
        .to_csv(AUDIT_DIR / "dominant_land_cover_group_counts.csv", index=False)
    )
    metadata = {
        "panel_unit": "Cambodia national 1 km grid cell",
        "grid_cells": int(len(output)),
        "observed_cells": int(observed.sum()),
        "observed_share": float(observed.mean()),
        "source": "MCD12Q1 Version 6.1 LC_Type1 IGBP annual classification",
        "source_years": list(YEARS),
        "source_manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "target_crs": "EPSG:32648",
        "target_resolution_m": 1000,
        "share_rule": "area-average annual class indicators to 1 km, then mean over 2001-2007",
        "stability_rule": "mean native-pixel modal IGBP-class frequency over seven pre-conflict years",
        "class_groups": {name: sorted(codes) for name, codes in CLASS_GROUPS.items()},
        "timing_rule": "predetermined 2001-2007 baseline; no post-conflict land cover used as a control",
        "outcome_blind": True,
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
    }
    (AUDIT_DIR / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
