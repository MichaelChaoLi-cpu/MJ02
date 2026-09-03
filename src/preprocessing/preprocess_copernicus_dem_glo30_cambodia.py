#!/usr/bin/env python3
"""Construct national 1 km terrain covariates from Copernicus DEM GLO-30."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw/terrain/copernicus_dem_glo30_2021"
GRID = ROOT / "data/processed/cambodia_national_1km_grid_preprocessed.parquet"
OUTPUT = ROOT / "data/processed/cambodia_national_terrain_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/copernicus-dem-glo30"
MOSAIC_VRT = AUDIT_DIR / "copernicus_dem_glo30_cambodia.vrt"
SCRATCH_DIR = Path(
    os.environ.get("MJ02_TERRAIN_SCRATCH_DIR", "/private/tmp/mj02-copernicus-dem-glo30")
)
PROJECTED_DEM = SCRATCH_DIR / "copernicus_dem_glo30_cambodia_epsg32648.tif"
SLOPE_RASTER = SCRATCH_DIR / "copernicus_dem_glo30_cambodia_slope_degrees.tif"

TARGET_CRS = "EPSG:32648"
TARGET_RESOLUTION_M = 30
STEEP_SLOPE_THRESHOLD_DEGREES = 15.0
SLOPE_HISTOGRAM_WIDTH_DEGREES = 0.5
SLOPE_HISTOGRAM_MAX_DEGREES = 90.0


def run(command: list[str]) -> None:
    print("Running:", " ".join(command[:4]), "...", flush=True)
    subprocess.run(command, check=True)


def create_derived_rasters(grid: pd.DataFrame, tiles: list[Path]) -> None:
    if not MOSAIC_VRT.exists():
        run(["gdalbuildvrt", str(MOSAIC_VRT), *[str(path) for path in tiles]])
    bounds = [
        float(grid["Grid West m"].min()),
        float(grid["Grid South m"].min()),
        float(grid["Grid East m"].max()),
        float(grid["Grid North m"].max()),
    ]
    if not PROJECTED_DEM.exists():
        run(
            [
                "gdalwarp",
                "-overwrite",
                "-t_srs",
                TARGET_CRS,
                "-tr",
                str(TARGET_RESOLUTION_M),
                str(TARGET_RESOLUTION_M),
                "-tap",
                "-te",
                *[str(value) for value in bounds],
                "-r",
                "bilinear",
                "-dstnodata",
                "-9999",
                "-multi",
                "-wo",
                "NUM_THREADS=ALL_CPUS",
                "-co",
                "TILED=YES",
                "-co",
                "COMPRESS=DEFLATE",
                "-co",
                "BIGTIFF=YES",
                str(MOSAIC_VRT),
                str(PROJECTED_DEM),
            ]
        )
    if not SLOPE_RASTER.exists():
        run(
            [
                "gdaldem",
                "slope",
                str(PROJECTED_DEM),
                str(SLOPE_RASTER),
                "-compute_edges",
                "-of",
                "GTiff",
                "-co",
                "TILED=YES",
                "-co",
                "COMPRESS=DEFLATE",
                "-co",
                "BIGTIFF=YES",
            ]
        )


def main() -> None:
    tiles = sorted(RAW_DIR.glob("Copernicus_DSM_COG_10_*_DEM.tif"))
    if not GRID.exists() or not tiles:
        raise FileNotFoundError(
            "Missing grid or Copernicus tiles; run acquire_copernicus_dem_glo30_cambodia.py first"
        )
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(GRID)
    if len(grid) != 179_072 or grid["National Grid Cell ID"].duplicated().any():
        raise RuntimeError("Unexpected national grid structure")
    create_derived_rasters(grid, tiles)

    n = len(grid)
    grid_lookup = {
        (int(column), int(row)): index
        for index, (column, row) in enumerate(zip(grid["Grid Column"], grid["Grid Row"], strict=True))
    }
    count = np.zeros(n, dtype="int64")
    elevation_sum = np.zeros(n, dtype="float64")
    elevation_sum_sq = np.zeros(n, dtype="float64")
    slope_sum = np.zeros(n, dtype="float64")
    steep_count = np.zeros(n, dtype="int64")
    bins = int(SLOPE_HISTOGRAM_MAX_DEGREES / SLOPE_HISTOGRAM_WIDTH_DEGREES) + 1
    slope_histogram = np.zeros((n, bins), dtype="uint16")

    with rasterio.open(PROJECTED_DEM) as dem, rasterio.open(SLOPE_RASTER) as slope:
        if dem.crs != slope.crs or dem.transform != slope.transform or dem.shape != slope.shape:
            raise RuntimeError("Projected DEM and slope raster grids do not match")
        for _, window in dem.block_windows(1):
            elevation = dem.read(1, window=window, masked=True).filled(np.nan)
            slope_values = slope.read(1, window=window, masked=True).filled(np.nan)
            valid = np.isfinite(elevation) & np.isfinite(slope_values)
            if not valid.any():
                continue
            rows, columns = np.nonzero(valid)
            raster_columns = columns + int(window.col_off)
            raster_rows = rows + int(window.row_off)
            eastings, northings = rasterio.transform.xy(
                dem.transform, raster_rows, raster_columns, offset="center"
            )
            cell_columns = np.floor(np.asarray(eastings) / 1000).astype("int32")
            cell_rows = np.floor(np.asarray(northings) / 1000).astype("int32")
            local_indices = np.fromiter(
                (grid_lookup.get((int(c), int(r)), -1) for c, r in zip(cell_columns, cell_rows, strict=True)),
                dtype="int64",
                count=len(cell_columns),
            )
            inside = local_indices >= 0
            if not inside.any():
                continue
            indices = local_indices[inside]
            elevations = elevation[rows, columns][inside].astype("float64")
            slopes = slope_values[rows, columns][inside].astype("float64")
            count += np.bincount(indices, minlength=n)
            elevation_sum += np.bincount(indices, weights=elevations, minlength=n)
            elevation_sum_sq += np.bincount(indices, weights=elevations**2, minlength=n)
            slope_sum += np.bincount(indices, weights=slopes, minlength=n)
            steep_count += np.bincount(
                indices, weights=slopes >= STEEP_SLOPE_THRESHOLD_DEGREES, minlength=n
            ).astype("int64")
            slope_bins = np.clip(
                np.floor(slopes / SLOPE_HISTOGRAM_WIDTH_DEGREES).astype("int16"), 0, bins - 1
            )
            np.add.at(slope_histogram, (indices, slope_bins), 1)

    observed = count > 0
    mean_elevation = np.full(n, np.nan, dtype="float64")
    elevation_sd = np.full(n, np.nan, dtype="float64")
    mean_slope = np.full(n, np.nan, dtype="float64")
    steep_share = np.full(n, np.nan, dtype="float64")
    mean_elevation[observed] = elevation_sum[observed] / count[observed]
    variance = np.maximum(
        elevation_sum_sq[observed] / count[observed] - mean_elevation[observed] ** 2, 0
    )
    elevation_sd[observed] = np.sqrt(variance)
    mean_slope[observed] = slope_sum[observed] / count[observed]
    steep_share[observed] = steep_count[observed] / count[observed]

    cumulative = np.cumsum(slope_histogram.astype("uint32"), axis=1)
    threshold = np.ceil(count * 0.90).astype("uint32")
    p90_bin = np.argmax(cumulative >= threshold[:, None], axis=1)
    slope_p90 = (p90_bin + 0.5) * SLOPE_HISTOGRAM_WIDTH_DEGREES
    slope_p90[~observed] = np.nan

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
    output["Mean Elevation m"] = mean_elevation.astype("float32")
    output["Elevation SD m"] = elevation_sd.astype("float32")
    output["Mean Slope Degrees"] = mean_slope.astype("float32")
    output["Slope P90 Degrees"] = slope_p90.astype("float32")
    output["Steep Terrain Share"] = steep_share.astype("float32")
    output["Terrain Valid Pixel Count"] = count.astype("int32")
    output["Terrain Observed"] = observed
    output.to_parquet(OUTPUT, index=False)

    summary = pd.DataFrame(
        {
            "Metric": [
                "Grid cells",
                "Grid cells observed",
                "Grid cells missing",
                "Mean elevation m",
                "Median elevation m",
                "P90 elevation m",
                "Mean slope degrees",
                "Median slope degrees",
                "P90 of grid mean slope degrees",
                "Mean steep terrain share",
            ],
            "Value": [
                n,
                int(observed.sum()),
                int((~observed).sum()),
                float(np.nanmean(mean_elevation)),
                float(np.nanmedian(mean_elevation)),
                float(np.nanquantile(mean_elevation, 0.90)),
                float(np.nanmean(mean_slope)),
                float(np.nanmedian(mean_slope)),
                float(np.nanquantile(mean_slope, 0.90)),
                float(np.nanmean(steep_share)),
            ],
        }
    )
    summary.to_csv(AUDIT_DIR / "terrain_summary.csv", index=False)
    metadata = {
        "source": "Copernicus DEM GLO-30 Public, 2021 release",
        "role": "time-invariant predetermined terrain covariates",
        "source_tiles": len(tiles),
        "projected_crs": TARGET_CRS,
        "projected_resolution_m": TARGET_RESOLUTION_M,
        "slope_method": "gdaldem slope on 30 m EPSG:32648 bilinear-reprojected DSM",
        "slope_p90_method": "upper edge midpoint of 0.5-degree histogram bin containing the 90th percentile",
        "steep_threshold_degrees": STEEP_SLOPE_THRESHOLD_DEGREES,
        "outcome_blind": True,
        "intermediate_rasters_retained": False,
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
    }
    (AUDIT_DIR / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(summary.to_string(index=False))
    PROJECTED_DEM.unlink(missing_ok=True)
    SLOPE_RASTER.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
