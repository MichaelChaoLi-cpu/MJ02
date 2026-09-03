#!/usr/bin/env python3
"""Construct climate-cell seasonal EVI/NDVI outcomes aligned to Cambodia's monsoon year."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


VEGETATION = Path("data/processed/cambodia_national_modis_vegetation_16day")
GRID_CLIMATE = Path("data/processed/cambodia_national_1km_to_climate_cell_preprocessed.parquet")
OUTPUT = Path("data/processed/cambodia_national_seasonal_vegetation_preprocessed.parquet")
AUDIT = Path("data/exp/data-preprocessing/climate-welfare/seasonal-vegetation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def q(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    for relative in (VEGETATION, GRID_CLIMATE):
        if not (root / relative).exists():
            raise FileNotFoundError(root / relative)
    audit = root / AUDIT
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "duckdb-tmp").mkdir(parents=True, exist_ok=True)
    output = root / OUTPUT

    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{q(audit / 'duckdb-tmp')}'")
    query = f"""
    COPY (
      WITH cell_date AS MATERIALIZED (
        SELECT g."Climate Cell ID", v."Composite Date", v."Year", v."Day of Year",
          v."MODIS Composite Slot",
          sum(v."Mean EVI" * v."EVI Valid Pixel Count")
            / nullif(sum(v."EVI Valid Pixel Count") FILTER (WHERE v."Mean EVI" IS NOT NULL), 0)
            AS "Climate Cell Mean EVI",
          sum(v."Mean NDVI" * v."NDVI Valid Pixel Count")
            / nullif(sum(v."NDVI Valid Pixel Count") FILTER (WHERE v."Mean NDVI" IS NOT NULL), 0)
            AS "Climate Cell Mean NDVI",
          sum(v."EVI Valid Pixel Count") AS "Climate Cell EVI Valid Pixel Count",
          sum(v."NDVI Valid Pixel Count") AS "Climate Cell NDVI Valid Pixel Count"
        FROM read_parquet('{q(root / VEGETATION)}/**/*.parquet', hive_partitioning = true) v
        INNER JOIN read_parquet('{q(root / GRID_CLIMATE)}') g USING ("National Grid Cell ID")
        WHERE v."Year" BETWEEN 2001 AND 2024
        GROUP BY 1, 2, 3, 4, 5
      ),
      slot_baseline AS (
        SELECT "Climate Cell ID", "MODIS Composite Slot",
          avg("Climate Cell Mean EVI") FILTER (WHERE "Year" BETWEEN 2001 AND 2020)
            AS evi_mean,
          stddev_samp("Climate Cell Mean EVI") FILTER (WHERE "Year" BETWEEN 2001 AND 2020)
            AS evi_sd,
          count("Climate Cell Mean EVI") FILTER (WHERE "Year" BETWEEN 2001 AND 2020)
            AS evi_years,
          avg("Climate Cell Mean NDVI") FILTER (WHERE "Year" BETWEEN 2001 AND 2020)
            AS ndvi_mean,
          stddev_samp("Climate Cell Mean NDVI") FILTER (WHERE "Year" BETWEEN 2001 AND 2020)
            AS ndvi_sd,
          count("Climate Cell Mean NDVI") FILTER (WHERE "Year" BETWEEN 2001 AND 2020)
            AS ndvi_years
        FROM cell_date GROUP BY 1, 2
      ),
      anomaly AS (
        SELECT d.*,
          CASE WHEN b.evi_sd > 0 AND b.evi_years >= 10
            THEN (d."Climate Cell Mean EVI" - b.evi_mean) / b.evi_sd END
            AS "Climate Cell EVI Slot Anomaly Z",
          CASE WHEN b.ndvi_sd > 0 AND b.ndvi_years >= 10
            THEN (d."Climate Cell Mean NDVI" - b.ndvi_mean) / b.ndvi_sd END
            AS "Climate Cell NDVI Slot Anomaly Z"
        FROM cell_date d
        LEFT JOIN slot_baseline b USING ("Climate Cell ID", "MODIS Composite Slot")
      ),
      production_dates AS (
        SELECT *,
          CASE WHEN month("Composite Date") BETWEEN 5 AND 12 THEN "Year"
               WHEN month("Composite Date") BETWEEN 1 AND 2 THEN "Year" - 1 END
            AS "Production Season Year"
        FROM anomaly
        WHERE month("Composite Date") NOT BETWEEN 3 AND 4
      ),
      aggregated AS (
        SELECT "Climate Cell ID", "Production Season Year",
          avg("Climate Cell EVI Slot Anomaly Z") FILTER (
            WHERE month("Composite Date") BETWEEN 5 AND 10)
            AS may_oct_evi_z_raw,
          avg("Climate Cell NDVI Slot Anomaly Z") FILTER (
            WHERE month("Composite Date") BETWEEN 5 AND 10)
            AS may_oct_ndvi_z_raw,
          count("Climate Cell EVI Slot Anomaly Z") FILTER (
            WHERE month("Composite Date") BETWEEN 5 AND 10) AS may_oct_evi_n,
          count("Climate Cell NDVI Slot Anomaly Z") FILTER (
            WHERE month("Composite Date") BETWEEN 5 AND 10) AS may_oct_ndvi_n,
          avg("Climate Cell EVI Slot Anomaly Z") FILTER (
            WHERE month("Composite Date") IN (11, 12, 1, 2))
            AS nov_feb_evi_z_raw,
          avg("Climate Cell NDVI Slot Anomaly Z") FILTER (
            WHERE month("Composite Date") IN (11, 12, 1, 2))
            AS nov_feb_ndvi_z_raw,
          count("Climate Cell EVI Slot Anomaly Z") FILTER (
            WHERE month("Composite Date") IN (11, 12, 1, 2)) AS nov_feb_evi_n,
          count("Climate Cell NDVI Slot Anomaly Z") FILTER (
            WHERE month("Composite Date") IN (11, 12, 1, 2)) AS nov_feb_ndvi_n,
          avg("Climate Cell EVI Slot Anomaly Z") AS may_feb_evi_z_raw,
          avg("Climate Cell NDVI Slot Anomaly Z") AS may_feb_ndvi_z_raw,
          min("Climate Cell EVI Slot Anomaly Z") AS may_feb_evi_min_raw,
          max("Climate Cell EVI Slot Anomaly Z") AS may_feb_evi_peak_raw,
          count("Climate Cell EVI Slot Anomaly Z") AS may_feb_evi_n,
          count("Climate Cell NDVI Slot Anomaly Z") AS may_feb_ndvi_n
        FROM production_dates
        WHERE "Production Season Year" BETWEEN 2001 AND 2024
        GROUP BY 1, 2
      )
      SELECT "Climate Cell ID", "Production Season Year",
        CASE WHEN may_oct_evi_n >= 7 THEN may_oct_evi_z_raw END
          AS "May-October Mean EVI Anomaly Z",
        CASE WHEN may_oct_ndvi_n >= 7 THEN may_oct_ndvi_z_raw END
          AS "May-October Mean NDVI Anomaly Z",
        may_oct_evi_n AS "May-October Valid EVI Composites",
        may_oct_ndvi_n AS "May-October Valid NDVI Composites",
        CASE WHEN nov_feb_evi_n >= 5 THEN nov_feb_evi_z_raw END
          AS "November-February Mean EVI Anomaly Z",
        CASE WHEN nov_feb_ndvi_n >= 5 THEN nov_feb_ndvi_z_raw END
          AS "November-February Mean NDVI Anomaly Z",
        nov_feb_evi_n AS "November-February Valid EVI Composites",
        nov_feb_ndvi_n AS "November-February Valid NDVI Composites",
        CASE WHEN may_feb_evi_n >= 12 AND may_oct_evi_n >= 7 AND nov_feb_evi_n >= 5
          THEN may_feb_evi_z_raw END
          AS "May-February Production-Season Mean EVI Anomaly Z",
        CASE WHEN may_feb_ndvi_n >= 12 AND may_oct_ndvi_n >= 7 AND nov_feb_ndvi_n >= 5
          THEN may_feb_ndvi_z_raw END
          AS "May-February Production-Season Mean NDVI Anomaly Z",
        CASE WHEN may_feb_evi_n >= 12 AND may_oct_evi_n >= 7 AND nov_feb_evi_n >= 5
          THEN may_feb_evi_min_raw END
          AS "May-February Minimum EVI Anomaly Z",
        CASE WHEN may_feb_evi_n >= 12 AND may_oct_evi_n >= 7 AND nov_feb_evi_n >= 5
          THEN may_feb_evi_peak_raw END
          AS "May-February Peak EVI Anomaly Z",
        may_feb_evi_n AS "May-February Valid EVI Composites",
        may_feb_ndvi_n AS "May-February Valid NDVI Composites",
        CASE WHEN may_feb_evi_n >= 12 AND may_oct_evi_n >= 7 AND nov_feb_evi_n >= 5
          THEN 1 ELSE 0 END
          AS "Complete Production-Season EVI Support"
      FROM aggregated
      ORDER BY "Climate Cell ID", "Production Season Year"
    ) TO '{q(output)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con.execute(query)
    con.close()

    frame = pd.read_parquet(output)
    keys = ["Climate Cell ID", "Production Season Year"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate climate-cell production-season keys")
    coverage = (
        frame.groupby("Production Season Year", observed=True)
        .agg(
            Climate_Cells=("Climate Cell ID", "nunique"),
            May_October_EVI=("May-October Mean EVI Anomaly Z", "count"),
            November_February_EVI=("November-February Mean EVI Anomaly Z", "count"),
            Full_Production_Season_EVI=("May-February Production-Season Mean EVI Anomaly Z", "count"),
        )
        .reset_index()
    )
    coverage.to_csv(audit / "coverage_by_production_season.csv", index=False)
    metadata = {
        "unit": "climate cell by May-February production-season year",
        "rows": int(len(frame)),
        "climate_cells": int(frame["Climate Cell ID"].nunique()),
        "production_season_years": [
            int(frame["Production Season Year"].min()),
            int(frame["Production Season Year"].max()),
        ],
        "primary_dynamic_outcome": "May-February Production-Season Mean EVI Anomaly Z",
        "secondary_windows": [
            "May-October Mean EVI Anomaly Z",
            "November-February Mean EVI Anomaly Z",
        ],
        "sensor_robustness": "parallel NDVI measures",
        "baseline": "climate-cell by MODIS composite slot, 2001-2020, at least 10 valid years",
        "minimum_valid_composites": {"May-October": 7, "November-February": 5, "May-February": 12},
        "outcomes_read_for_window_selection": False,
    }
    (audit / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (audit / "README.md").write_text(
        "# Season-aligned national vegetation panel\n\n"
        "The primary dynamic outcome is mean EVI anomaly over the fixed May-to-February "
        "production-season window. May-October and November-February decompose concurrent and "
        "post-monsoon vegetation response. Windows and minimum composite counts were fixed without "
        "examining NPP, EVI, NDVI, or household response coefficients. The frozen threshold requires "
        "roughly 60% of expected 16-day composites in each window.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
