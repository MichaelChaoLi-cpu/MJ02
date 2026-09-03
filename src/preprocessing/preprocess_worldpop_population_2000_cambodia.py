#!/usr/bin/env python3
"""Prepare WorldPop 2000 as a robustness baseline on Cambodia's 1 km grid."""

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
SOURCE = ROOT / "data/raw/population/worldpop_2000/khm_ppp_2000_1km_Aggregated_UNadj.tif"
OUTPUT = ROOT / "data/processed/cambodia_national_worldpop_2000_robustness_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/national-worldpop-2000-robustness"

SOURCE_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/"
    "2000/KHM/khm_ppp_2000_1km_Aggregated_UNadj.tif"
)
SOURCE_DOI = "10.5258/SOTON/WP00671"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not GRID.exists() or not SOURCE.exists():
        raise FileNotFoundError("National grid or WorldPop 2000 source is missing")
    grid = pd.read_parquet(GRID)
    west = float(grid["Grid West m"].min())
    north = float(grid["Grid North m"].max())
    width = int(round((grid["Grid East m"].max() - west) / 1000))
    height = int(round((north - grid["Grid South m"].min()) / 1000))
    target_transform = from_origin(west, north, 1000, 1000)
    row_index = (grid["Grid Row"].max() - grid["Grid Row"]).to_numpy(dtype=int)
    col_index = (grid["Grid Column"] - grid["Grid Column"].min()).to_numpy(dtype=int)

    destination = np.full((height, width), np.nan, dtype="float32")
    with rasterio.open(SOURCE) as source:
        population = source.read(1).astype("float32")
        nodata = source.nodata
        if nodata is not None:
            population[population == nodata] = np.nan
        population[population < 0] = np.nan
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
    output["WorldPop Baseline Population 2000"] = values.astype("float32")
    output["Log WorldPop Baseline Population 2000"] = np.log1p(values).astype("float32")
    output["WorldPop Baseline Population Observed"] = np.isfinite(values)
    if output["National Grid Cell ID"].duplicated().any() or len(output) != 179_072:
        raise RuntimeError("WorldPop output failed national grid-key validation")
    observed = output["WorldPop Baseline Population Observed"]
    if observed.mean() < 0.99 or np.nanmin(values) < 0:
        raise RuntimeError("WorldPop output failed coverage or nonnegative-value validation")
    estimated_total = float(np.nansum(values))
    if not 8_000_000 <= estimated_total <= 16_000_000:
        raise RuntimeError(f"Implausible WorldPop 2000 Cambodia total: {estimated_total:,.0f}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT, index=False)
    series = output.loc[observed, "WorldPop Baseline Population 2000"]
    summary = pd.DataFrame(
        {
            "Metric": [
                "Grid cells",
                "Observed cells",
                "Estimated population sum",
                "Median population per 1 km cell",
                "P90 population per 1 km cell",
                "P99 population per 1 km cell",
            ],
            "Value": [
                len(output),
                observed.sum(),
                series.sum(),
                series.median(),
                series.quantile(0.90),
                series.quantile(0.99),
            ],
        }
    )
    summary.to_csv(AUDIT_DIR / "worldpop_2000_summary.csv", index=False)
    metadata = {
        "panel_unit": "Cambodia national 1 km grid cell",
        "source": "WorldPop unconstrained population count 2000, 1 km aggregated, UN-unadjusted",
        "source_url": SOURCE_URL,
        "source_doi": SOURCE_DOI,
        "source_sha256": sha256(SOURCE),
        **source_metadata,
        "target_crs": "EPSG:32648",
        "target_resolution_m": 1000,
        "aggregation_rule": "mass-preserving sum resampling of population counts",
        "role": "population-baseline robustness only; not the sole primary matching covariate",
        "caution": "WorldPop spatial redistribution uses modeled ancillary covariates and may encode later-period information",
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
