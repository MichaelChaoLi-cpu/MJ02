#!/usr/bin/env python3
"""Construct four absolute annual climate shocks at cell and village buffers.

The four prespecified measures are full-calendar-year counts or maxima aligned
to annual cropland NPP: days with daily maximum temperature at or above 35 C,
heat degree-days above 35 C, Rx5day, and maximum consecutive dry days using a
daily precipitation threshold below 1 mm.  Heat-day count and degree-days are
retained as alternative heat specifications and are not intended to enter the
same primary regression by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


PRECIP = Path("data/processed/cambodia_national_daily_precipitation")
TEMPERATURE = Path("data/processed/cambodia_national_daily_temperature")
CELL_RAW = Path("data/exp/data-preprocessing/climate-npp-two-stage/annual-climate-cell-raw")
CELL_OUTPUT = Path("data/processed/cambodia_national_annual_absolute_climate_shocks_preprocessed.parquet")
MEMBERSHIP = Path("data/processed/cses_village_buffer_grid_crosswalk_preprocessed.parquet")
GRID_CLIMATE = Path("data/processed/cambodia_national_1km_to_climate_cell_preprocessed.parquet")
VILLAGE_OUTPUT = Path("data/processed/cses_village_buffer_annual_absolute_climate_shocks_preprocessed.parquet")
AUDIT = Path("data/exp/data-preprocessing/climate-npp-two-stage/climate-shocks")


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def connection(temp: Path) -> duckdb.DuckDBPyConnection:
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{sql_path(temp)}'")
    return con


def valid_year(path: Path, year: int, expected_cells: int = 6252) -> bool:
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


def build_year(con: duckdb.DuckDBPyConnection, root: Path, year: int, force: bool) -> tuple[Path, str]:
    output = root / CELL_RAW / f"year={year}" / "climate_shocks.parquet"
    if not force and valid_year(output, year):
        return output, "reused"
    output.parent.mkdir(parents=True, exist_ok=True)
    precip = root / PRECIP / f"year={year}" / "precipitation.parquet"
    temperature = root / TEMPERATURE / f"year={year}" / "temperature.parquet"
    if not precip.exists() or not temperature.exists():
        raise FileNotFoundError(f"Missing daily climate inputs for {year}")
    query = f"""
    COPY (
      WITH daily AS (
        SELECT
          p."Climate Cell ID",
          p."Climate Cell Longitude",
          p."Climate Cell Latitude",
          p."Date",
          p."Daily Precipitation mm",
          t."Daily Maximum Temperature C",
          CASE WHEN p."Daily Precipitation mm" IS NULL THEN NULL
               WHEN p."Daily Precipitation mm" < 1 THEN 1 ELSE 0 END AS dry_day,
          CASE WHEN t."Daily Maximum Temperature C" IS NULL THEN NULL
               WHEN t."Daily Maximum Temperature C" >= 35 THEN 1 ELSE 0 END AS heat_day_35,
          greatest(t."Daily Maximum Temperature C" - 35, 0) AS heat_degree_day_35,
          sum(p."Daily Precipitation mm") OVER (
            PARTITION BY p."Climate Cell ID" ORDER BY p."Date"
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
          ) AS rolling_five_day_precipitation,
          sum(CASE WHEN p."Daily Precipitation mm" IS NULL
                        OR p."Daily Precipitation mm" >= 1 THEN 1 ELSE 0 END)
            OVER (PARTITION BY p."Climate Cell ID" ORDER BY p."Date") AS dry_group
        FROM read_parquet('{sql_path(precip)}') p
        LEFT JOIN read_parquet('{sql_path(temperature)}') t
          USING ("Climate Cell ID", "Date")
      ),
      runs AS (
        SELECT *,
          sum(dry_day) OVER (PARTITION BY "Climate Cell ID", dry_group) AS dry_run_length
        FROM daily
      )
      SELECT
        "Climate Cell ID",
        any_value("Climate Cell Longitude") AS "Climate Cell Longitude",
        any_value("Climate Cell Latitude") AS "Climate Cell Latitude",
        {year}::INTEGER AS "Year",
        sum(heat_day_35) AS "Annual Heat Days at or Above 35 C",
        sum(heat_degree_day_35) AS "Annual Heat Degree-Days Above 35 C",
        max(rolling_five_day_precipitation) AS "Annual Maximum Consecutive Five-Day Precipitation Rx5day mm",
        max(dry_run_length) AS "Annual Maximum Consecutive Dry Days Below 1 mm",
        sum("Daily Precipitation mm") AS "Annual Precipitation Total mm",
        count("Daily Maximum Temperature C") AS "Annual Valid Temperature Days",
        count("Daily Precipitation mm") AS "Annual Valid Precipitation Days"
      FROM runs GROUP BY 1 ORDER BY 1
    ) TO '{sql_path(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    if not valid_year(output, year):
        raise ValueError(f"Failed annual cell validation for {year}")
    return output, "constructed"


def combine_cells(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    source = root / CELL_RAW / "year=*" / "climate_shocks.parquet"
    output = root / CELL_OUTPUT
    query = f"""
    COPY (
      SELECT * FROM read_parquet('{sql_path(source)}', hive_partitioning=true)
      ORDER BY "Climate Cell ID", "Year"
    ) TO '{sql_path(output)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con.execute(query)


def build_village_buffers(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    membership = root / MEMBERSHIP
    grid_climate = root / GRID_CLIMATE
    cell = root / CELL_OUTPUT
    output = root / VILLAGE_OUTPUT
    variables = [
        "Annual Heat Days at or Above 35 C",
        "Annual Heat Degree-Days Above 35 C",
        "Annual Maximum Consecutive Five-Day Precipitation Rx5day mm",
        "Annual Maximum Consecutive Dry Days Below 1 mm",
        "Annual Precipitation Total mm",
        "Annual Valid Temperature Days",
        "Annual Valid Precipitation Days",
    ]
    weighted = [
        f'sum(c."{variable}" * w.grid_weight) '
        f'/ nullif(sum(w.grid_weight) FILTER (WHERE c."{variable}" IS NOT NULL), 0) '
        f'AS "Village Buffer Mean {variable}"'
        for variable in variables
    ]
    query = f"""
    COPY (
      WITH weights AS (
        SELECT
          m."Village Code", m."Buffer Radius km", g."Climate Cell ID",
          count(*)::DOUBLE AS grid_weight
        FROM read_parquet('{sql_path(membership)}') m
        INNER JOIN read_parquet('{sql_path(grid_climate)}') g
          USING ("National Grid Cell ID")
        GROUP BY 1, 2, 3
      )
      SELECT
        w."Village Code", w."Buffer Radius km", c."Year",
        count(DISTINCT w."Climate Cell ID") AS "Village Buffer Climate Cell Count",
        sum(w.grid_weight) AS "Village Buffer Included 1 km Grid Cell Count",
        {', '.join(weighted)}
      FROM weights w
      INNER JOIN read_parquet('{sql_path(cell)}') c USING ("Climate Cell ID")
      GROUP BY 1, 2, 3
      ORDER BY 1, 2, 3
    ) TO '{sql_path(output)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con.execute(query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    audit = root / AUDIT
    audit.mkdir(parents=True, exist_ok=True)
    con = connection(audit / "duckdb-tmp")
    coverage: list[dict[str, object]] = []
    for year in range(args.start_year, args.end_year + 1):
        path, status = build_year(con, root, year, args.force)
        frame = pd.read_parquet(path)
        coverage.append(
            {
                "Year": year,
                "Rows": len(frame),
                "Climate Cells": frame["Climate Cell ID"].nunique(),
                "Heat Days Coverage": float(frame["Annual Heat Days at or Above 35 C"].notna().mean()),
                "Heat Degree-Days Coverage": float(frame["Annual Heat Degree-Days Above 35 C"].notna().mean()),
                "Rx5day Coverage": float(frame["Annual Maximum Consecutive Five-Day Precipitation Rx5day mm"].notna().mean()),
                "Consecutive Dry Days Coverage": float(frame["Annual Maximum Consecutive Dry Days Below 1 mm"].notna().mean()),
                "Status": status,
            }
        )
        print(f"Completed {year}: {status}; cells={len(frame):,}", flush=True)
    combine_cells(con, root)
    build_village_buffers(con, root)
    con.close()

    cell = pd.read_parquet(root / CELL_OUTPUT)
    village = pd.read_parquet(root / VILLAGE_OUTPUT)
    cell_keys = ["Climate Cell ID", "Year"]
    village_keys = ["Village Code", "Buffer Radius km", "Year"]
    if cell.duplicated(cell_keys).any() or village.duplicated(village_keys).any():
        raise ValueError("Duplicate climate-shock keys")
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(audit / "annual_climate_shock_coverage_by_year.csv", index=False)
    village_coverage = village.groupby(["Buffer Radius km", "Year"], observed=True).agg(
        **{
            "Villages": ("Village Code", "nunique"),
            "Heat Days Observed": ("Village Buffer Mean Annual Heat Days at or Above 35 C", "count"),
            "Heat Degree-Days Observed": ("Village Buffer Mean Annual Heat Degree-Days Above 35 C", "count"),
            "Rx5day Observed": ("Village Buffer Mean Annual Maximum Consecutive Five-Day Precipitation Rx5day mm", "count"),
            "Consecutive Dry Days Observed": ("Village Buffer Mean Annual Maximum Consecutive Dry Days Below 1 mm", "count"),
        }
    ).reset_index()
    village_coverage.to_csv(audit / "village_buffer_climate_shock_coverage.csv", index=False)
    dictionary = pd.DataFrame(
        [
            ["Annual Heat Days at or Above 35 C", "days/year", "Count of days with daily Tmax >= 35 C", "primary heat exposure"],
            ["Annual Heat Degree-Days Above 35 C", "degree-C days/year", "Sum of max(daily Tmax - 35 C, 0)", "alternative heat intensity; not jointly entered with heat-day count by default"],
            ["Annual Maximum Consecutive Five-Day Precipitation Rx5day mm", "mm", "Maximum rolling sum of precipitation over five consecutive calendar days", "extreme-rainfall exposure; not direct flood inundation"],
            ["Annual Maximum Consecutive Dry Days Below 1 mm", "days", "Longest run of days with precipitation below 1 mm", "dry-spell exposure"],
        ],
        columns=["Variable", "Unit", "Construction", "Model Role"],
    )
    dictionary.to_csv(audit / "climate_shock_variable_dictionary.csv", index=False)
    metadata = {
        "cell_output": str(CELL_OUTPUT),
        "village_output": str(VILLAGE_OUTPUT),
        "years": [args.start_year, args.end_year],
        "climate_cells": int(cell["Climate Cell ID"].nunique()),
        "cell_year_rows": int(len(cell)),
        "villages": int(village["Village Code"].nunique()),
        "buffer_radii_km": sorted(village["Buffer Radius km"].unique().astype(int).tolist()),
        "village_year_buffer_rows": int(len(village)),
        "calendar_window": "January 1 through December 31 of the NPP outcome year",
        "heat_threshold": "daily maximum temperature at or above 35 C",
        "dry_day_threshold": "daily precipitation below 1 mm",
        "primary_buffer_km": 5,
        "sensitivity_buffers_km": [2, 10],
        "imputation": "none",
        "joint_heat_specification": "Heat-day count and heat degree-days are alternatives because they encode highly overlapping exposure",
    }
    (audit / "climate_shock_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(village_coverage.loc[village_coverage["Buffer Radius km"].eq(5)].tail().to_string(index=False))


if __name__ == "__main__":
    main()
