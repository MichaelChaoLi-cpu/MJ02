#!/usr/bin/env python3
"""Acquire public daily CHIRPS rainfall for boundary-village grid cells.

ClimateSERV limits one request to 20 years.  The script therefore requests
1991--2000, 2001--2020, 2021, and 2022--2024 separately for each distinct 0.05-degree CHIRPS cell used
by the frozen five-kilometre boundary sample.  Raw responses are retained and
the final source table has one row per CHIRPS cell and day.  No treatment-side
outcome contrast is estimated here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[2]
DESIGN_PANEL = Path("data/processed/historical_boundary_annual_spatial_climate_preprocessed.parquet")
SOURCE_DIR = Path("data/exp/data-preprocessing/dynamic-rainfall-source/climateserv-chirps")
DEFAULT_OUTPUT = Path("data/processed/historical_boundary_daily_chirps_preprocessed.parquet")
BASE_URL = "https://climateserv.servirglobal.net/api"
PERIODS = (
    ("1991_2000", "01/01/1991", "12/31/2000"),
    ("2001_2020", "01/01/2001", "12/31/2020"),
    ("2021", "01/01/2021", "12/31/2021"),
    ("2022_2024", "01/01/2022", "12/31/2024"),
)
GRID_SIZE = 0.05
LAT_ORIGIN = -50.0
LON_ORIGIN = -180.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=8.0)
    parser.add_argument("--timeout-minutes", type=float, default=45.0)
    return parser.parse_args()


def project_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def request_json(endpoint: str, params: dict[str, Any], timeout: float = 90.0) -> Any:
    response = requests.get(f"{BASE_URL}/{endpoint}/", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_cells(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "Village Code", "Longitude", "Latitude",
        "Historical-Boundary Common Support 5 km",
    ]
    villages = pd.read_parquet(project_path(root, DESIGN_PANEL), columns=columns)
    villages = (
        villages.loc[villages["Historical-Boundary Common Support 5 km"].eq(1)]
        .drop_duplicates("Village Code")
        .copy()
    )
    villages["CHIRPS Longitude Index"] = ((villages["Longitude"] - LON_ORIGIN) / GRID_SIZE).map(math.floor)
    villages["CHIRPS Latitude Index"] = ((villages["Latitude"] - LAT_ORIGIN) / GRID_SIZE).map(math.floor)
    villages["CHIRPS Cell Longitude"] = LON_ORIGIN + (villages["CHIRPS Longitude Index"] + 0.5) * GRID_SIZE
    villages["CHIRPS Cell Latitude"] = LAT_ORIGIN + (villages["CHIRPS Latitude Index"] + 0.5) * GRID_SIZE
    villages["CHIRPS Cell ID"] = (
        villages["CHIRPS Longitude Index"].astype(str)
        + "_"
        + villages["CHIRPS Latitude Index"].astype(str)
    )
    cells = (
        villages[[
            "CHIRPS Cell ID", "CHIRPS Longitude Index", "CHIRPS Latitude Index",
            "CHIRPS Cell Longitude", "CHIRPS Cell Latitude",
        ]]
        .drop_duplicates("CHIRPS Cell ID")
        .sort_values("CHIRPS Cell ID")
        .reset_index(drop=True)
    )
    return villages, cells


def point_polygon(longitude: float, latitude: float) -> dict[str, Any]:
    # A small polygon wholly inside one 0.05-degree CHIRPS cell samples that
    # cell while avoiding boundary ambiguity in polygon-raster intersection.
    half_width = 0.005
    return {
        "type": "Polygon",
        "coordinates": [[
            [longitude - half_width, latitude - half_width],
            [longitude - half_width, latitude + half_width],
            [longitude + half_width, latitude + half_width],
            [longitude + half_width, latitude - half_width],
            [longitude - half_width, latitude - half_width],
        ]],
    }


def run_job(
    cell: dict[str, Any],
    period_name: str,
    begin: str,
    end: str,
    raw_dir: Path,
    poll_seconds: float,
    timeout_minutes: float,
) -> Path:
    cell_id = str(cell["CHIRPS Cell ID"])
    output = raw_dir / f"chirps_{cell_id}_{period_name}.json"
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("data"):
            return output

    geometry = point_polygon(
        float(cell["CHIRPS Cell Longitude"]),
        float(cell["CHIRPS Cell Latitude"]),
    )
    submit = request_json(
        "submitDataRequest",
        {
            "datatype": 0,
            "begintime": begin,
            "endtime": end,
            "intervaltype": 0,
            "operationtype": 5,
            "dateType_Category": "default",
            "isZip_CurrentDataType": "false",
            "geometry": json.dumps(geometry, separators=(",", ":")),
        },
    )
    if not isinstance(submit, list) or not submit or not isinstance(submit[0], str):
        raise RuntimeError(f"Unexpected submission response for {cell_id}: {submit!r}")
    job_id = submit[0]
    deadline = time.monotonic() + timeout_minutes * 60
    while True:
        progress = request_json("getDataRequestProgress", {"id": job_id})
        value = float(progress[0]) if isinstance(progress, list) and progress else -1.0
        if value >= 100.0:
            break
        if value < 0:
            raise RuntimeError(f"ClimateSERV reported an error for job {job_id}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"ClimateSERV job {job_id} exceeded {timeout_minutes:g} minutes")
        time.sleep(poll_seconds)
    payload = request_json("getDataFromRequest", {"id": job_id}, timeout=180.0)
    if not payload.get("data"):
        raise RuntimeError(f"ClimateSERV returned no data for job {job_id}")
    payload["acquisition_metadata"] = {
        "job_id": job_id,
        "cell_id": cell_id,
        "cell_longitude": float(cell["CHIRPS Cell Longitude"]),
        "cell_latitude": float(cell["CHIRPS Cell Latitude"]),
        "period": period_name,
        "begin": begin,
        "end": end,
        "source": "ClimateSERV UCSB CHIRPS Rainfall datatype 0",
        "api": BASE_URL,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return output


def compile_daily(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload["acquisition_metadata"]
        for item in payload["data"]:
            rows.append({
                "CHIRPS Cell ID": metadata["cell_id"],
                "CHIRPS Cell Longitude": metadata["cell_longitude"],
                "CHIRPS Cell Latitude": metadata["cell_latitude"],
                "Date": pd.Timestamp(year=int(item["year"]), month=int(item["month"]), day=int(item["day"])),
                "Daily Rainfall mm": float(item["value"]["avg"]),
                "ClimateSERV Job ID": metadata["job_id"],
            })
    daily = pd.DataFrame(rows).sort_values(["CHIRPS Cell ID", "Date"]).reset_index(drop=True)
    duplicates = daily.duplicated(["CHIRPS Cell ID", "Date"])
    if duplicates.any():
        raise ValueError(f"Duplicate cell-days returned: {int(duplicates.sum())}")
    return daily


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_dir = project_path(root, SOURCE_DIR)
    raw_dir = source_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    villages, cells = load_cells(root)
    villages.to_csv(source_dir / "village_to_chirps_cell.csv", index=False)
    cells.to_csv(source_dir / "chirps_cells.csv", index=False)

    tasks = [
        (cell, period_name, begin, end)
        for cell in cells.to_dict("records")
        for period_name, begin, end in PERIODS
    ]
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_job, cell, period_name, begin, end, raw_dir,
                args.poll_seconds, args.timeout_minutes,
            ): (cell["CHIRPS Cell ID"], period_name)
            for cell, period_name, begin, end in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            cell_id, period_name = futures[future]
            paths.append(future.result())
            print(f"Completed {completed}/{len(tasks)}: cell {cell_id}, {period_name}", flush=True)

    daily = compile_daily(paths)
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
            missing_rainfall=("Daily Rainfall mm", lambda values: int(values.isna().sum())),
        )
        .reset_index()
    )
    coverage.to_csv(source_dir / "coverage_by_year.csv", index=False)
    metadata = {
        "dataset": "UCSB CHIRPS Rainfall via ClimateSERV",
        "datatype": 0,
        "temporal_resolution": "daily",
        "spatial_resolution_degrees": GRID_SIZE,
        "years": [1991, 2024],
        "boundary_villages": int(villages["Village Code"].nunique()),
        "unique_chirps_cells": int(cells["CHIRPS Cell ID"].nunique()),
        "official_documentation": "https://climateserv.servirglobal.net/develop-api",
    }
    (source_dir / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved: {output.relative_to(root)}")
    print(f"Rows: {len(daily):,}; cells: {daily['CHIRPS Cell ID'].nunique():,}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
