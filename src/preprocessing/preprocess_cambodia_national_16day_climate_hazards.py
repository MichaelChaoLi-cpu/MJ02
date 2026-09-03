#!/usr/bin/env python3
"""Construct nationwide 16-day climate hazards aligned to MODIS calendar slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


THRESHOLDS = Path("data/processed/cambodia_national_cell_month_climate_thresholds_preprocessed.parquet")
RAW_DIR = Path("data/exp/data-preprocessing/national-climate-hazards/16day-raw")
OUTPUT = Path("data/processed/cambodia_national_16day_climate_hazards_preprocessed.parquet")
AUDIT_DIR = Path("data/exp/data-preprocessing/national-climate-hazards")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def connect(root: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '8GB'")
    con.execute(f"SET temp_directory = '{sql_path(root / 'data/exp/data-preprocessing/national-climate-hazards/duckdb-tmp')}'")
    return con


def valid(path: Path, year: int, cells: int = 6252) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path, columns=["Climate Cell ID", "Year", "MODIS Composite Slot"])
    except Exception:
        return False
    return (
        len(frame) == cells * 23
        and frame["Climate Cell ID"].nunique() == cells
        and frame["MODIS Composite Slot"].nunique() == 23
        and frame["Year"].eq(year).all()
        and not frame.duplicated(["Climate Cell ID", "MODIS Composite Slot"]).any()
    )


def build_year(con: duckdb.DuckDBPyConnection, root: Path, year: int, force: bool) -> tuple[Path, str]:
    output = root / RAW_DIR / f"year={year}" / "hazards.parquet"
    if not force and valid(output, year):
        return output, "reused"
    output.parent.mkdir(parents=True, exist_ok=True)
    precip = root / f"data/processed/cambodia_national_daily_precipitation/year={year}/precipitation.parquet"
    temp = root / f"data/processed/cambodia_national_daily_temperature/year={year}/temperature.parquet"
    thresholds = root / THRESHOLDS
    query = f"""
    COPY (
      WITH daily AS (
        SELECT
          p."Climate Cell ID",
          p."Climate Cell Longitude",
          p."Climate Cell Latitude",
          p."Date",
          {year}::INTEGER AS "Year",
          floor((dayofyear(p."Date") - 1) / 16)::INTEGER + 1 AS "MODIS Composite Slot",
          make_date({year}, 1, 1)
            + floor((dayofyear(p."Date") - 1) / 16)::INTEGER * INTERVAL 16 DAY AS "Composite Date",
          p."Daily Precipitation mm",
          t."Daily Maximum Temperature C",
          t."Daily Minimum Temperature C",
          CASE WHEN p."Daily Precipitation mm" IS NULL OR th."Wet-Day Precipitation P95 mm" IS NULL THEN NULL
               WHEN p."Daily Precipitation mm" > th."Wet-Day Precipitation P95 mm" THEN 1 ELSE 0 END AS heavy_rain_day,
          CASE WHEN t."Daily Maximum Temperature C" IS NULL OR th."Daily Maximum Temperature P90 C" IS NULL THEN NULL
               WHEN t."Daily Maximum Temperature C" > th."Daily Maximum Temperature P90 C" THEN 1 ELSE 0 END AS hot_day,
          CASE WHEN t."Daily Minimum Temperature C" IS NULL OR th."Daily Minimum Temperature P90 C" IS NULL THEN NULL
               WHEN t."Daily Minimum Temperature C" > th."Daily Minimum Temperature P90 C" THEN 1 ELSE 0 END AS hot_night
        FROM read_parquet('{sql_path(precip)}') p
        LEFT JOIN read_parquet('{sql_path(temp)}') t USING ("Climate Cell ID", "Date")
        LEFT JOIN read_parquet('{sql_path(thresholds)}') th
          ON p."Climate Cell ID" = th."Climate Cell ID"
         AND month(p."Date") = th."Calendar Month"
      ),
      grouped AS (
        SELECT
          *,
          sum(CASE WHEN hot_day IS NULL OR hot_day = 0 THEN 1 ELSE 0 END)
            OVER (PARTITION BY "Climate Cell ID" ORDER BY "Date") AS hot_group
        FROM daily
      ),
      runs AS (
        SELECT *, sum(hot_day) OVER (PARTITION BY "Climate Cell ID", hot_group) AS hot_run_length
        FROM grouped
      )
      SELECT
        "Climate Cell ID",
        any_value("Climate Cell Longitude") AS "Climate Cell Longitude",
        any_value("Climate Cell Latitude") AS "Climate Cell Latitude",
        {year}::INTEGER AS "Year",
        "MODIS Composite Slot",
        any_value("Composite Date") AS "Composite Date",
        count(*) AS "Calendar Days in Composite",
        sum("Daily Precipitation mm") AS "Composite Precipitation Total mm",
        max("Daily Precipitation mm") AS "Composite Maximum One-Day Precipitation mm",
        sum(heavy_rain_day) AS "Composite Extreme Wet Day Count",
        avg("Daily Maximum Temperature C") AS "Composite Mean Daily Maximum Temperature C",
        max("Daily Maximum Temperature C") AS "Composite Maximum Daily Temperature C",
        sum(hot_day) AS "Composite Hot Day Count",
        sum(hot_night) AS "Composite Hot Night Count",
        CASE WHEN count(hot_day) = 0 THEN NULL
             ELSE sum(CASE WHEN hot_day = 1 AND hot_run_length >= 3 THEN 1 ELSE 0 END)
        END AS "Composite Heatwave Day Count",
        count("Daily Precipitation mm") AS "Composite Valid Precipitation Days",
        count("Daily Maximum Temperature C") AS "Composite Valid Temperature Days"
      FROM runs
      GROUP BY "Climate Cell ID", "MODIS Composite Slot"
      ORDER BY "Climate Cell ID", "MODIS Composite Slot"
    ) TO '{sql_path(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    if not valid(output, year):
        raise ValueError(f"16-day climate partition failed validation for {year}")
    return output, "constructed"


def build_final(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    source = sql_path(root / RAW_DIR / "year=*/hazards.parquet")
    output = root / OUTPUT
    variables = [
        "Composite Precipitation Total mm",
        "Composite Maximum One-Day Precipitation mm",
        "Composite Extreme Wet Day Count",
        "Composite Mean Daily Maximum Temperature C",
        "Composite Maximum Daily Temperature C",
        "Composite Hot Day Count",
        "Composite Hot Night Count",
        "Composite Heatwave Day Count",
    ]
    stats, z = [], []
    for index, variable in enumerate(variables):
        a = f"v{index}"
        stats.extend([
            f'avg("{variable}") AS {a}_mean',
            f'stddev_samp("{variable}") AS {a}_sd',
            f'count("{variable}") AS {a}_count',
        ])
        z.extend([
            f'b.{a}_mean AS "{variable} Mean 1991-2020"',
            f'b.{a}_sd AS "{variable} SD 1991-2020"',
            f'b.{a}_count AS "{variable} Valid Years 1991-2020"',
            f'CASE WHEN b.{a}_sd > 0 AND b.{a}_count >= 20 THEN (x."{variable}" - b.{a}_mean) / b.{a}_sd END AS "{variable} Anomaly Z"',
        ])
    query = f"""
    COPY (
      WITH x AS (SELECT * FROM read_parquet('{source}', hive_partitioning=true)),
      b AS (
        SELECT "Climate Cell ID", "MODIS Composite Slot", {', '.join(stats)}
        FROM x WHERE "Year" BETWEEN 1991 AND 2020 GROUP BY 1, 2
      ),
      standardized AS (
        SELECT x.*, {', '.join(z)}
        FROM x LEFT JOIN b USING ("Climate Cell ID", "MODIS Composite Slot")
      )
      SELECT
        *,
        greatest(-"Composite Precipitation Total mm Anomaly Z", 0) AS "Composite Dry Rainfall Intensity",
        greatest("Composite Precipitation Total mm Anomaly Z", 0) AS "Composite Wet Rainfall Intensity",
        greatest("Composite Hot Day Count Anomaly Z", 0) AS "Composite Heat Intensity",
        greatest(-"Composite Precipitation Total mm Anomaly Z", 0)
          * greatest("Composite Hot Day Count Anomaly Z", 0) AS "Composite Compound Hot-Dry Intensity"
      FROM standardized ORDER BY "Climate Cell ID", "Composite Date"
    ) TO '{sql_path(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not (root / THRESHOLDS).exists():
        raise FileNotFoundError("Run annual national climate-hazard preprocessing first")
    con = connect(root)
    coverage = []
    for year in range(args.start_year, args.end_year + 1):
        path, status = build_year(con, root, year, args.force)
        frame = pd.read_parquet(path, columns=[
            "Climate Cell ID", "MODIS Composite Slot", "Composite Precipitation Total mm",
            "Composite Hot Day Count", "Composite Heatwave Day Count",
        ])
        coverage.append({
            "Year": year, "Rows": len(frame), "Climate Cells": frame["Climate Cell ID"].nunique(),
            "Composite Slots": frame["MODIS Composite Slot"].nunique(),
            "Rainfall Available Share": float(frame["Composite Precipitation Total mm"].notna().mean()),
            "Heat Available Share": float(frame["Composite Hot Day Count"].notna().mean()),
            "Heatwave Available Share": float(frame["Composite Heatwave Day Count"].notna().mean()),
            "Status": status,
        })
        print(f"Completed {year}: {status}; rows={len(frame):,}", flush=True)
    build_final(con, root)
    audit = root / AUDIT_DIR
    pd.DataFrame(coverage).to_csv(audit / "16day_hazard_coverage_by_year.csv", index=False)
    final = pd.read_parquet(root / OUTPUT, columns=["Climate Cell ID", "Composite Date"])
    metadata = {
        "dataset": "Cambodia nationwide 16-day climate hazards",
        "years": [args.start_year, args.end_year],
        "climate_cells": int(final["Climate Cell ID"].nunique()),
        "rows": len(final),
        "slots_per_year": 23,
        "reference_period": "1991-2020",
        "calendar_alignment": "23 intervals restarting January 1 each year; final interval has 13 or 14 days",
        "imputation": "none",
    }
    (audit / "16day_hazard_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
