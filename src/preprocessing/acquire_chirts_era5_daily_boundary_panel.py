#!/usr/bin/env python3
"""Extract daily CHIRTS-ERA5 Tmax and Tmin for the frozen boundary grid.

The official annual NetCDF files are global and approximately 6.7 GiB each.  The
Climate Hazards Center server supports HTTP byte ranges, so this script uses
GDAL's /vsicurl/ interface and reads only the small raster window containing the
27 CHIRPS cells already fixed for the historical-boundary design.  It never
downloads or rewrites the global source files.

The source is an experimental, bias-corrected and downscaled CHIRTS-ERA5 product.
Extracted annual subsets are retained under data/exp for resumability and source
auditing; the compiled analysis-ready daily panel is written to data/processed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


ROOT = Path(__file__).resolve().parents[2]
CELL_FILE = Path(
    "data/exp/data-preprocessing/dynamic-rainfall-source/"
    "climateserv-chirps/chirps_cells.csv"
)
SOURCE_DIR = Path("data/exp/data-preprocessing/temperature-source/chirts-era5")
DEFAULT_OUTPUT = Path(
    "data/processed/historical_boundary_daily_temperature_preprocessed.parquet"
)
BASE_URL = "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5"
OFFICIAL_PAGE = "https://www.chc.ucsb.edu/data/chirts-era5"
DATA_DOI = "https://doi.org/10.15780/G2F08J"
GRID_SIZE = 0.05
WEST_CENTER = -179.975
SOUTH_CENTER = -59.975

VARIABLES = {
    "tmax": {
        "filename_token": "Tmax",
        "column": "Daily Maximum Temperature C",
    },
    "tmin": {
        "filename_token": "Tmin",
        "column": "Daily Minimum Temperature C",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract annual subsets that already pass their local checks.",
    )
    return parser.parse_args()


def project_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def source_url(variable: str, year: int) -> str:
    token = VARIABLES[variable]["filename_token"]
    return (
        f"{BASE_URL}/{variable}/netcdf/daily/"
        f"CHIRTS-ERA5.daily_{token}.{year}.nc"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cells(root: Path) -> pd.DataFrame:
    path = project_path(root, CELL_FILE)
    cells = pd.read_csv(path, dtype={"CHIRPS Cell ID": "string"})
    required = [
        "CHIRPS Cell ID",
        "CHIRPS Cell Longitude",
        "CHIRPS Cell Latitude",
    ]
    missing = set(required).difference(cells.columns)
    if missing:
        raise ValueError(f"Cell file is missing columns: {sorted(missing)}")
    cells = cells[required].drop_duplicates("CHIRPS Cell ID").copy()
    cells["CHIRTS Column Index"] = np.rint(
        (cells["CHIRPS Cell Longitude"] - WEST_CENTER) / GRID_SIZE
    ).astype(int)
    cells["CHIRTS Row Index"] = np.rint(
        (cells["CHIRPS Cell Latitude"] - SOUTH_CENTER) / GRID_SIZE
    ).astype(int)
    reconstructed_lon = WEST_CENTER + cells["CHIRTS Column Index"] * GRID_SIZE
    reconstructed_lat = SOUTH_CENTER + cells["CHIRTS Row Index"] * GRID_SIZE
    if not np.allclose(reconstructed_lon, cells["CHIRPS Cell Longitude"], atol=1e-8):
        raise ValueError("CHIRPS and CHIRTS longitude grids do not align")
    if not np.allclose(reconstructed_lat, cells["CHIRPS Cell Latitude"], atol=1e-8):
        raise ValueError("CHIRPS and CHIRTS latitude grids do not align")
    return cells.sort_values("CHIRPS Cell ID").reset_index(drop=True)


def extract_variable(
    cells: pd.DataFrame,
    variable: str,
    year: int,
    allow_structural_missing: bool = False,
) -> pd.DataFrame:
    url = source_url(variable, year)
    vsi_url = f"/vsicurl/{url}"
    row_min = int(cells["CHIRTS Row Index"].min())
    row_max = int(cells["CHIRTS Row Index"].max())
    col_min = int(cells["CHIRTS Column Index"].min())
    col_max = int(cells["CHIRTS Column Index"].max())
    window = Window(
        col_off=col_min,
        row_off=row_min,
        width=col_max - col_min + 1,
        height=row_max - row_min + 1,
    )
    expected_dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    with rasterio.Env(
        GDAL_SKIP="netCDF",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".nc",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        GDAL_HTTP_TIMEOUT="120",
    ):
        with rasterio.open(vsi_url) as dataset:
            if (dataset.width, dataset.height) != (7200, 2600):
                raise ValueError(
                    f"Unexpected CHIRTS-ERA5 grid for {variable} {year}: "
                    f"{dataset.width} x {dataset.height}"
                )
            if dataset.count != len(expected_dates):
                raise ValueError(
                    f"Unexpected band count for {variable} {year}: "
                    f"{dataset.count}; expected {len(expected_dates)}"
                )
            values = dataset.read(window=window).astype("float32")
            nodata = dataset.nodata
    if nodata is not None:
        values[values == nodata] = np.nan

    parts: list[pd.DataFrame] = []
    column = VARIABLES[variable]["column"]
    for _, cell in cells.iterrows():
        local_row = int(cell["CHIRTS Row Index"]) - row_min
        local_col = int(cell["CHIRTS Column Index"]) - col_min
        series = values[:, local_row, local_col]
        parts.append(
            pd.DataFrame(
                {
                    "CHIRPS Cell ID": cell["CHIRPS Cell ID"],
                    "CHIRPS Cell Longitude": float(cell["CHIRPS Cell Longitude"]),
                    "CHIRPS Cell Latitude": float(cell["CHIRPS Cell Latitude"]),
                    "Date": expected_dates,
                    column: series,
                }
            )
        )
    output = pd.concat(parts, ignore_index=True)
    if output[column].isna().any() and not allow_structural_missing:
        raise ValueError(
            f"Missing {variable} values in the boundary subset for {year}: "
            f"{int(output[column].isna().sum())}"
        )
    return output


def annual_subset_is_valid(path: Path, year: int, expected_cells: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    expected_days = len(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"))
    required = {
        "CHIRPS Cell ID",
        "Date",
        "Daily Maximum Temperature C",
        "Daily Minimum Temperature C",
    }
    return (
        required.issubset(frame.columns)
        and len(frame) == expected_days * expected_cells
        and frame["CHIRPS Cell ID"].nunique() == expected_cells
        and not frame[list(required - {"CHIRPS Cell ID", "Date"})].isna().any().any()
        and not frame.duplicated(["CHIRPS Cell ID", "Date"]).any()
    )


def extract_year(
    cells: pd.DataFrame,
    year: int,
    allow_structural_missing: bool = False,
) -> pd.DataFrame:
    tmax = extract_variable(cells, "tmax", year, allow_structural_missing)
    tmin = extract_variable(cells, "tmin", year, allow_structural_missing)
    keys = [
        "CHIRPS Cell ID",
        "CHIRPS Cell Longitude",
        "CHIRPS Cell Latitude",
        "Date",
    ]
    annual = tmax.merge(tmin, on=keys, how="inner", validate="one_to_one")
    invalid = annual["Daily Maximum Temperature C"] < annual["Daily Minimum Temperature C"]
    if invalid.any():
        raise ValueError(f"Tmax is below Tmin in {int(invalid.sum())} rows for {year}")
    return annual.sort_values(["CHIRPS Cell ID", "Date"]).reset_index(drop=True)


def extract_or_reuse_year(
    cells: pd.DataFrame, year: int, path: Path, force: bool
) -> tuple[int, Path, int, str]:
    if force or not annual_subset_is_valid(path, year, len(cells)):
        annual = extract_year(cells, year)
        annual.to_parquet(path, index=False)
        return year, path, len(annual), "extracted"
    rows = len(pd.read_parquet(path, columns=["Date"]))
    return year, path, rows, "reused"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    cells = load_cells(root)
    source_dir = project_path(root, SOURCE_DIR)
    extract_dir = source_dir / "annual_boundary_extracts"
    extract_dir.mkdir(parents=True, exist_ok=True)

    years = list(range(args.start_year, args.end_year + 1))
    paths_by_year = {
        year: extract_dir / f"chirts_era5_boundary_daily_{year}.parquet"
        for year in years
    }
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    with ProcessPoolExecutor(max_workers=min(args.workers, len(years))) as executor:
        futures = {
            executor.submit(
                extract_or_reuse_year,
                cells,
                year,
                paths_by_year[year],
                args.force,
            ): year
            for year in years
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            year, path, rows, status = future.result()
            print(
                f"Completed {completed}/{len(years)}: {year} {status}, "
                f"rows={rows:,}, file={path.relative_to(root)}",
                flush=True,
            )
    annual_paths = [paths_by_year[year] for year in years]

    daily = pd.concat(
        [pd.read_parquet(path) for path in annual_paths], ignore_index=True
    ).sort_values(["CHIRPS Cell ID", "Date"]).reset_index(drop=True)
    if daily.duplicated(["CHIRPS Cell ID", "Date"]).any():
        raise ValueError("Compiled temperature panel contains duplicate cell-days")
    if daily["Date"].min() != pd.Timestamp(args.start_year, 1, 1):
        raise ValueError("Compiled temperature panel has an unexpected start date")
    if daily["Date"].max() != pd.Timestamp(args.end_year, 12, 31):
        raise ValueError("Compiled temperature panel has an unexpected end date")

    output = project_path(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output, index=False)

    coverage = (
        daily.assign(Year=daily["Date"].dt.year)
        .groupby("Year", observed=True)
        .agg(
            cells=("CHIRPS Cell ID", "nunique"),
            days=("Date", "nunique"),
            rows=("Date", "size"),
            missing_tmax=("Daily Maximum Temperature C", lambda x: int(x.isna().sum())),
            missing_tmin=("Daily Minimum Temperature C", lambda x: int(x.isna().sum())),
            minimum_tmin_c=("Daily Minimum Temperature C", "min"),
            maximum_tmax_c=("Daily Maximum Temperature C", "max"),
        )
        .reset_index()
    )
    coverage.to_csv(source_dir / "coverage_by_year.csv", index=False)

    manifest_rows = []
    for year, path in zip(range(args.start_year, args.end_year + 1), annual_paths):
        manifest_rows.append(
            {
                "year": year,
                "tmax_url": source_url("tmax", year),
                "tmin_url": source_url("tmin", year),
                "local_extract": str(path.relative_to(root)),
                "local_extract_sha256": sha256(path),
                "rows": len(pd.read_parquet(path, columns=["Date"])),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(source_dir / "source_manifest.csv", index=False)
    metadata = {
        "dataset": "CHIRTS-ERA5 daily Tmax and Tmin",
        "provider": "Climate Hazards Center, University of California Santa Barbara",
        "official_page": OFFICIAL_PAGE,
        "data_repository": BASE_URL,
        "doi": DATA_DOI,
        "license": "CC BY 4.0",
        "status": "experimental bias-corrected and downscaled product",
        "spatial_resolution_degrees": GRID_SIZE,
        "temporal_resolution": "daily",
        "years": [args.start_year, args.end_year],
        "grid_cells": int(cells["CHIRPS Cell ID"].nunique()),
        "extraction_method": "GDAL /vsicurl/ HTTP byte-range window read",
        "global_files_retained_locally": False,
        "accessed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (source_dir / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    output_label = str(output.relative_to(root)) if output.is_relative_to(root) else str(output)
    print(f"Saved: {output_label}")
    print(
        f"Rows={len(daily):,}; cells={daily['CHIRPS Cell ID'].nunique():,}; "
        f"dates={daily['Date'].min().date()} to {daily['Date'].max().date()}"
    )
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
