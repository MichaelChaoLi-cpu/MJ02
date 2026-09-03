#!/usr/bin/env python3
"""Construct nationwide annual and monsoon climate hazards from daily panels.

The script keeps rainfall shortage, heavy rainfall, heat, and hot-night measures
separate. It does not label rainfall extremes as observed floods and performs no
spatial or temporal imputation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


PRECIP_GLOB = "data/processed/cambodia_national_daily_precipitation/year=*/precipitation.parquet"
TEMP_GLOB = "data/processed/cambodia_national_daily_temperature/year=*/temperature.parquet"
THRESHOLDS = Path("data/processed/cambodia_national_cell_month_climate_thresholds_preprocessed.parquet")
RAW_DIR = Path("data/exp/data-preprocessing/national-climate-hazards/annual-raw")
OUTPUT = Path("data/processed/cambodia_national_annual_climate_hazards_preprocessed.parquet")
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
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '8GB'")
    connection.execute(f"SET temp_directory = '{sql_path(root / 'data/exp/data-preprocessing/national-climate-hazards/duckdb-tmp')}'")
    return connection


def build_thresholds(connection: duckdb.DuckDBPyConnection, root: Path, force: bool) -> Path:
    output = root / THRESHOLDS
    if output.exists() and not force:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    precip = sql_path(root / PRECIP_GLOB)
    temp = sql_path(root / TEMP_GLOB)
    query = f"""
    COPY (
      WITH p AS (
        SELECT
          "Climate Cell ID",
          month("Date") AS "Calendar Month",
          quantile_cont("Daily Precipitation mm", 0.95)
            FILTER (WHERE "Daily Precipitation mm" >= 1) AS "Wet-Day Precipitation P95 mm",
          count("Daily Precipitation mm") AS "Baseline Precipitation Days"
        FROM read_parquet('{precip}', hive_partitioning=true)
        WHERE year BETWEEN 1991 AND 2020
        GROUP BY 1, 2
      ),
      t AS (
        SELECT
          "Climate Cell ID",
          month("Date") AS "Calendar Month",
          quantile_cont("Daily Maximum Temperature C", 0.90) AS "Daily Maximum Temperature P90 C",
          quantile_cont("Daily Minimum Temperature C", 0.90) AS "Daily Minimum Temperature P90 C",
          count("Daily Maximum Temperature C") AS "Baseline Temperature Days"
        FROM read_parquet('{temp}', hive_partitioning=true)
        WHERE year BETWEEN 1991 AND 2020
        GROUP BY 1, 2
      )
      SELECT * FROM p FULL OUTER JOIN t USING ("Climate Cell ID", "Calendar Month")
      ORDER BY "Climate Cell ID", "Calendar Month"
    ) TO '{sql_path(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    connection.execute(query)
    return output


def annual_valid(path: Path, year: int, expected_cells: int = 6252) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path, columns=["Climate Cell ID", "Year"])
    except Exception:
        return False
    return (
        len(frame) == expected_cells
        and frame["Climate Cell ID"].nunique() == expected_cells
        and frame["Year"].eq(year).all()
        and not frame["Climate Cell ID"].duplicated().any()
    )


def build_raw_year(
    connection: duckdb.DuckDBPyConnection,
    root: Path,
    thresholds: Path,
    year: int,
    force: bool,
) -> tuple[Path, str]:
    output = root / RAW_DIR / f"year={year}" / "hazards.parquet"
    if not force and annual_valid(output, year):
        return output, "reused"
    output.parent.mkdir(parents=True, exist_ok=True)
    precip = root / f"data/processed/cambodia_national_daily_precipitation/year={year}/precipitation.parquet"
    temp = root / f"data/processed/cambodia_national_daily_temperature/year={year}/temperature.parquet"
    query = f"""
    COPY (
      WITH daily AS (
        SELECT
          p."Climate Cell ID",
          p."Climate Cell Longitude",
          p."Climate Cell Latitude",
          p."Date",
          {year}::INTEGER AS "Year",
          p."Daily Precipitation mm",
          t."Daily Maximum Temperature C",
          t."Daily Minimum Temperature C",
          th."Wet-Day Precipitation P95 mm",
          th."Daily Maximum Temperature P90 C",
          th."Daily Minimum Temperature P90 C"
        FROM read_parquet('{sql_path(precip)}') p
        LEFT JOIN read_parquet('{sql_path(temp)}') t
          USING ("Climate Cell ID", "Date")
        LEFT JOIN read_parquet('{sql_path(thresholds)}') th
          ON p."Climate Cell ID" = th."Climate Cell ID"
         AND month(p."Date") = th."Calendar Month"
      ),
      annual AS (
        SELECT
          "Climate Cell ID",
          any_value("Climate Cell Longitude") AS "Climate Cell Longitude",
          any_value("Climate Cell Latitude") AS "Climate Cell Latitude",
          {year}::INTEGER AS "Year",
          sum("Daily Precipitation mm") AS "Annual Precipitation Total mm",
          count("Daily Precipitation mm") AS "Annual Valid Precipitation Days",
          avg("Daily Maximum Temperature C") AS "Annual Mean Daily Maximum Temperature C",
          avg("Daily Minimum Temperature C") AS "Annual Mean Daily Minimum Temperature C",
          count("Daily Maximum Temperature C") AS "Annual Valid Temperature Days"
        FROM daily GROUP BY 1
      ),
      monsoon_base AS (
        SELECT
          *,
          CASE WHEN "Daily Precipitation mm" IS NULL THEN NULL
               WHEN "Daily Precipitation mm" < 1 THEN 1 ELSE 0 END AS dry_day,
          CASE WHEN "Daily Precipitation mm" IS NULL OR "Wet-Day Precipitation P95 mm" IS NULL THEN NULL
               WHEN "Daily Precipitation mm" > "Wet-Day Precipitation P95 mm" THEN 1 ELSE 0 END AS heavy_rain_day,
          CASE WHEN "Daily Maximum Temperature C" IS NULL OR "Daily Maximum Temperature P90 C" IS NULL THEN NULL
               WHEN "Daily Maximum Temperature C" > "Daily Maximum Temperature P90 C" THEN 1 ELSE 0 END AS hot_day,
          CASE WHEN "Daily Minimum Temperature C" IS NULL OR "Daily Minimum Temperature P90 C" IS NULL THEN NULL
               WHEN "Daily Minimum Temperature C" > "Daily Minimum Temperature P90 C" THEN 1 ELSE 0 END AS hot_night,
          sum("Daily Precipitation mm") OVER (
            PARTITION BY "Climate Cell ID" ORDER BY "Date" ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
          ) AS rolling_five_day_precipitation,
          sum(CASE WHEN "Daily Precipitation mm" IS NULL OR "Daily Precipitation mm" >= 1 THEN 1 ELSE 0 END)
            OVER (PARTITION BY "Climate Cell ID" ORDER BY "Date") AS dry_group,
          sum(CASE WHEN "Daily Maximum Temperature C" IS NULL OR "Daily Maximum Temperature P90 C" IS NULL
                        OR "Daily Maximum Temperature C" <= "Daily Maximum Temperature P90 C" THEN 1 ELSE 0 END)
            OVER (PARTITION BY "Climate Cell ID" ORDER BY "Date") AS hot_group
        FROM daily
        WHERE month("Date") BETWEEN 5 AND 10
      ),
      monsoon_runs AS (
        SELECT
          *,
          sum(dry_day) OVER (PARTITION BY "Climate Cell ID", dry_group) AS dry_run_length,
          sum(hot_day) OVER (PARTITION BY "Climate Cell ID", hot_group) AS hot_run_length
        FROM monsoon_base
      ),
      monsoon AS (
        SELECT
          "Climate Cell ID",
          sum("Daily Precipitation mm") AS "May October Precipitation Total mm",
          max("Daily Precipitation mm") AS "May October Maximum One-Day Precipitation mm",
          max(rolling_five_day_precipitation) AS "May October Maximum Five-Day Precipitation mm",
          sum(heavy_rain_day) AS "May October Extreme Wet Day Count",
          max(dry_run_length) AS "May October Maximum Consecutive Dry Days",
          avg("Daily Maximum Temperature C") AS "May October Mean Daily Maximum Temperature C",
          max("Daily Maximum Temperature C") AS "May October Maximum Daily Temperature C",
          sum(hot_day) AS "May October Hot Day Count",
          sum(hot_night) AS "May October Hot Night Count",
          CASE WHEN count(hot_day) = 0 THEN NULL
               ELSE sum(CASE WHEN hot_day = 1 AND hot_run_length >= 3 THEN 1 ELSE 0 END)
          END AS "May October Heatwave Day Count",
          count("Daily Precipitation mm") AS "May October Valid Precipitation Days",
          count("Daily Maximum Temperature C") AS "May October Valid Temperature Days"
        FROM monsoon_runs GROUP BY 1
      )
      SELECT * FROM annual LEFT JOIN monsoon USING ("Climate Cell ID")
      ORDER BY "Climate Cell ID"
    ) TO '{sql_path(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    connection.execute(query)
    if not annual_valid(output, year):
        raise ValueError(f"Annual climate hazard partition failed validation for {year}")
    return output, "constructed"


def build_standardized_panel(connection: duckdb.DuckDBPyConnection, root: Path) -> None:
    source = sql_path(root / RAW_DIR / "year=*/hazards.parquet")
    output = root / OUTPUT
    variables = [
        "Annual Precipitation Total mm",
        "May October Precipitation Total mm",
        "May October Maximum One-Day Precipitation mm",
        "May October Maximum Five-Day Precipitation mm",
        "May October Extreme Wet Day Count",
        "May October Maximum Consecutive Dry Days",
        "May October Mean Daily Maximum Temperature C",
        "May October Maximum Daily Temperature C",
        "May October Hot Day Count",
        "May October Hot Night Count",
        "May October Heatwave Day Count",
    ]
    stats = []
    z_columns = []
    for index, variable in enumerate(variables):
        alias = f"v{index}"
        stats.extend([
            f'avg("{variable}") AS {alias}_mean',
            f'stddev_samp("{variable}") AS {alias}_sd',
            f'count("{variable}") AS {alias}_count',
        ])
        z_columns.extend([
            f'b.{alias}_mean AS "{variable} Mean 1991-2020"',
            f'b.{alias}_sd AS "{variable} SD 1991-2020"',
            f'b.{alias}_count AS "{variable} Valid Years 1991-2020"',
            f'CASE WHEN b.{alias}_sd > 0 AND b.{alias}_count >= 20 THEN (a."{variable}" - b.{alias}_mean) / b.{alias}_sd END AS "{variable} Anomaly Z"',
        ])
    query = f"""
    COPY (
      WITH a AS (SELECT * FROM read_parquet('{source}', hive_partitioning=true)),
      b AS (
        SELECT "Climate Cell ID", {', '.join(stats)}
        FROM a WHERE "Year" BETWEEN 1991 AND 2020 GROUP BY 1
      ),
      z AS (
        SELECT a.*, {', '.join(z_columns)}
        FROM a LEFT JOIN b USING ("Climate Cell ID")
      )
      SELECT
        *,
        greatest(-"May October Precipitation Total mm Anomaly Z", 0) AS "May October Dry Rainfall Intensity",
        greatest("May October Maximum Five-Day Precipitation mm Anomaly Z", 0) AS "May October Extreme Wet Rainfall Intensity",
        greatest("May October Hot Day Count Anomaly Z", 0) AS "May October Heat Intensity",
        greatest(-"May October Precipitation Total mm Anomaly Z", 0)
          * greatest("May October Hot Day Count Anomaly Z", 0) AS "May October Compound Hot-Dry Intensity"
      FROM z ORDER BY "Climate Cell ID", "Year"
    ) TO '{sql_path(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    connection.execute(query)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    audit = root / AUDIT_DIR
    audit.mkdir(parents=True, exist_ok=True)
    connection = connect(root)
    thresholds = build_thresholds(connection, root, args.force)
    coverage = []
    for year in range(args.start_year, args.end_year + 1):
        path, status = build_raw_year(connection, root, thresholds, year, args.force)
        frame = pd.read_parquet(path, columns=[
            "Climate Cell ID", "May October Precipitation Total mm",
            "May October Hot Day Count", "May October Heatwave Day Count",
        ])
        coverage.append({
            "Year": year,
            "Rows": len(frame),
            "Climate Cells": frame["Climate Cell ID"].nunique(),
            "Rainfall Available Share": float(frame["May October Precipitation Total mm"].notna().mean()),
            "Heat Available Share": float(frame["May October Hot Day Count"].notna().mean()),
            "Heatwave Available Share": float(frame["May October Heatwave Day Count"].notna().mean()),
            "Status": status,
        })
        print(f"Completed {year}: {status}; cells={len(frame):,}", flush=True)
    build_standardized_panel(connection, root)
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(audit / "annual_hazard_coverage_by_year.csv", index=False)
    final = pd.read_parquet(root / OUTPUT, columns=["Climate Cell ID", "Year"])
    metadata = {
        "dataset": "Cambodia nationwide annual climate hazards",
        "years": [args.start_year, args.end_year],
        "climate_cells": int(final["Climate Cell ID"].nunique()),
        "rows": len(final),
        "reference_period": "1991-2020",
        "hot_threshold": "cell-by-calendar-month P90 in 1991-2020",
        "extreme_wet_day_threshold": "cell-by-calendar-month wet-day P95 in 1991-2020",
        "dry_day_threshold": "daily precipitation below 1 mm",
        "monsoon_window": "May-October",
        "imputation": "none",
        "interpretation": "extreme rainfall proxies are not observed flood inundation",
    }
    (audit / "annual_hazard_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
