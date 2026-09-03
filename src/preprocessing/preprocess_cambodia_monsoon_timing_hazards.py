#!/usr/bin/env python3
"""Construct outcome-blind monsoon timing and absolute heat candidates.

The release keeps two agronomic onset definitions plus a Southeast-Asia
cumulative-rainfall-anomaly definition. Candidate definitions are constructed
without reading NPP or CSES outcomes. Daily rainfall below 1 mm is retained
only for the legacy relative hot-dry diagnostic; the primary heat measures use
absolute daily maximum-air-temperature thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


PRECIP = Path("data/processed/cambodia_national_daily_precipitation")
TEMP = Path("data/processed/cambodia_national_daily_temperature")
THRESHOLDS = Path(
    "data/processed/cambodia_national_cell_month_climate_thresholds_preprocessed.parquet"
)
PARTITIONS = Path("data/exp/data-preprocessing/climate-welfare/monsoon-cell-year")
OUTPUT = Path("data/processed/cambodia_national_monsoon_timing_preprocessed.parquet")
AUDIT = Path("data/exp/data-preprocessing/climate-welfare/monsoon-timing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    return parser.parse_args()


def q(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def connect(root: Path) -> duckdb.DuckDBPyConnection:
    temp = root / AUDIT / "duckdb-tmp"
    temp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{q(temp)}'")
    return con


def partition_valid(path: Path, year: int, expected_cells: int = 6252) -> bool:
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


def build_year(
    con: duckdb.DuckDBPyConnection,
    root: Path,
    year: int,
    force: bool,
) -> tuple[Path, str]:
    output = root / PARTITIONS / f"year={year}" / "monsoon.parquet"
    if not force and partition_valid(output, year):
        return output, "reused"
    output.parent.mkdir(parents=True, exist_ok=True)
    precip = root / PRECIP / f"year={year}" / "precipitation.parquet"
    temp = root / TEMP / f"year={year}" / "temperature.parquet"
    thresholds = root / THRESHOLDS
    for path in (precip, temp, thresholds):
        if not path.exists():
            raise FileNotFoundError(path)

    # Candidate A: first 3-day wet spell >=20 mm after 15 April, with no
    # 10-consecutive-dry-day spell in the next 20 days.
    # Candidate B: first 3-day wet spell >=30 mm after 15 April, with no
    # 14-day period totaling <5 mm in the next 30 days.
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
          th."Daily Maximum Temperature P90 C",
          CASE WHEN p."Daily Precipitation mm" IS NULL THEN NULL
               WHEN p."Daily Precipitation mm" < 1 THEN 1 ELSE 0 END AS dry_day,
          CASE WHEN t."Daily Maximum Temperature C" IS NULL
                    OR th."Daily Maximum Temperature P90 C" IS NULL THEN NULL
               WHEN t."Daily Maximum Temperature C" > th."Daily Maximum Temperature P90 C"
                 THEN 1 ELSE 0 END AS hot_day,
          CASE WHEN t."Daily Maximum Temperature C" IS NULL THEN NULL
               WHEN t."Daily Maximum Temperature C" >= 33 THEN 1 ELSE 0 END AS heat_day_33,
          CASE WHEN t."Daily Maximum Temperature C" IS NULL THEN NULL
               WHEN t."Daily Maximum Temperature C" >= 35 THEN 1 ELSE 0 END AS heat_day_35,
          CASE WHEN t."Daily Maximum Temperature C" IS NULL THEN NULL
               WHEN t."Daily Maximum Temperature C" >= 37 THEN 1 ELSE 0 END AS heat_day_37,
          CASE WHEN t."Daily Maximum Temperature C" IS NULL THEN NULL
               ELSE greatest(t."Daily Maximum Temperature C" - 35, 0) END
            AS heat_degree_days_35
        FROM read_parquet('{q(precip)}') p
        LEFT JOIN read_parquet('{q(temp)}') t USING ("Climate Cell ID", "Date")
        LEFT JOIN read_parquet('{q(thresholds)}') th
          ON p."Climate Cell ID" = th."Climate Cell ID"
         AND month(p."Date") = th."Calendar Month"
      ),
      annual_mean AS (
        SELECT "Climate Cell ID", avg("Daily Precipitation mm") AS annual_daily_mean
        FROM daily GROUP BY 1
      ),
      windows AS (
        SELECT d.*,
          sum("Daily Precipitation mm") OVER (
            PARTITION BY d."Climate Cell ID" ORDER BY d."Date"
            ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING
          ) AS rain_3d,
          count("Daily Precipitation mm") OVER (
            PARTITION BY d."Climate Cell ID" ORDER BY d."Date"
            ROWS BETWEEN CURRENT ROW AND 2 FOLLOWING
          ) AS rain_3d_valid,
          sum(dry_day) OVER (
            PARTITION BY d."Climate Cell ID" ORDER BY d."Date"
            ROWS BETWEEN CURRENT ROW AND 9 FOLLOWING
          ) AS dry_10d_days,
          count(dry_day) OVER (
            PARTITION BY d."Climate Cell ID" ORDER BY d."Date"
            ROWS BETWEEN CURRENT ROW AND 9 FOLLOWING
          ) AS dry_10d_valid,
          sum("Daily Precipitation mm") OVER (
            PARTITION BY d."Climate Cell ID" ORDER BY d."Date"
            ROWS BETWEEN CURRENT ROW AND 13 FOLLOWING
          ) AS rain_14d,
          count("Daily Precipitation mm") OVER (
            PARTITION BY d."Climate Cell ID" ORDER BY d."Date"
            ROWS BETWEEN CURRENT ROW AND 13 FOLLOWING
          ) AS rain_14d_valid,
          sum("Daily Precipitation mm" - a.annual_daily_mean) OVER (
            PARTITION BY d."Climate Cell ID" ORDER BY d."Date"
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
          ) AS cumulative_rainfall_anomaly
        FROM daily d
        LEFT JOIN annual_mean a USING ("Climate Cell ID")
      ),
      spell_starts AS (
        SELECT *,
          CASE WHEN dry_10d_valid = 10 AND dry_10d_days = 10 THEN 1 ELSE 0 END
            AS dry_10d_start,
          CASE WHEN rain_14d_valid = 14 AND rain_14d < 5 THEN 1 ELSE 0 END
            AS low_rain_14d_start
        FROM windows
      ),
      persistence AS (
        SELECT *,
          max(dry_10d_start) OVER (
            PARTITION BY "Climate Cell ID" ORDER BY "Date"
            ROWS BETWEEN 3 FOLLOWING AND 13 FOLLOWING
          ) AS dry_10d_after_initial_wet_spell,
          max(low_rain_14d_start) OVER (
            PARTITION BY "Climate Cell ID" ORDER BY "Date"
            ROWS BETWEEN 3 FOLLOWING AND 19 FOLLOWING
          ) AS low_rain_14d_after_initial_wet_spell
        FROM spell_starts
      ),
      onset_summary AS (
        SELECT
          "Climate Cell ID",
          min("Date") FILTER (
            WHERE "Date" BETWEEN make_date({year}, 4, 15) AND make_date({year}, 8, 31)
              AND rain_3d_valid = 3 AND rain_3d >= 20
          ) AS first_wet_spell_a,
          min("Date") FILTER (
            WHERE "Date" BETWEEN make_date({year}, 4, 15) AND make_date({year}, 8, 31)
              AND rain_3d_valid = 3 AND rain_3d >= 20
              AND coalesce(dry_10d_after_initial_wet_spell, 0) = 0
          ) AS onset_a,
          min("Date") FILTER (
            WHERE "Date" BETWEEN make_date({year}, 4, 15) AND make_date({year}, 8, 31)
              AND rain_3d_valid = 3 AND rain_3d >= 30
          ) AS first_wet_spell_b,
          min("Date") FILTER (
            WHERE "Date" BETWEEN make_date({year}, 4, 15) AND make_date({year}, 8, 31)
              AND rain_3d_valid = 3 AND rain_3d >= 30
              AND coalesce(low_rain_14d_after_initial_wet_spell, 0) = 0
          ) AS onset_b,
          arg_min("Date", cumulative_rainfall_anomaly) FILTER (
            WHERE "Date" BETWEEN make_date({year}, 4, 1) AND make_date({year}, 8, 31)
          ) AS onset_accumulation,
          arg_max("Date", cumulative_rainfall_anomaly) FILTER (
            WHERE "Date" BETWEEN make_date({year}, 8, 1) AND make_date({year}, 11, 30)
          ) AS cessation_accumulation
        FROM persistence GROUP BY 1
      ),
      post_onset AS (
        SELECT p.*, o.onset_a,
          sum(CASE WHEN p.dry_day = 0 OR p.dry_day IS NULL THEN 1 ELSE 0 END) OVER (
            PARTITION BY p."Climate Cell ID" ORDER BY p."Date"
          ) AS dry_group
        FROM persistence p
        INNER JOIN onset_summary o USING ("Climate Cell ID")
        WHERE o.onset_a IS NOT NULL
          AND p."Date" BETWEEN o.onset_a AND make_date({year}, 10, 31)
      ),
      dry_runs AS (
        SELECT "Climate Cell ID", dry_group,
          min("Date") AS dry_start,
          max("Date") AS dry_end,
          count(*) AS dry_length,
          sum(hot_day) AS hot_days_in_dry_run
        FROM post_onset WHERE dry_day = 1 GROUP BY 1, 2
      ),
      ranked_dry_runs AS (
        SELECT *, row_number() OVER (
          PARTITION BY "Climate Cell ID" ORDER BY dry_length DESC, dry_start
        ) AS dry_rank
        FROM dry_runs
      ),
      post_onset_summary AS (
        SELECT "Climate Cell ID",
          sum(CASE WHEN dry_day = 1 AND hot_day = 1 THEN 1 ELSE 0 END)
            AS post_onset_hot_dry_days,
          sum(CASE WHEN dry_day = 1 THEN 1 ELSE 0 END) AS post_onset_dry_days,
          sum(heat_day_33) AS post_onset_absolute_heat_days_33,
          sum(heat_day_35) AS post_onset_absolute_heat_days_35,
          sum(heat_day_37) AS post_onset_absolute_heat_days_37,
          sum(heat_degree_days_35) AS post_onset_heat_degree_days_35
        FROM post_onset GROUP BY 1
      ),
      post_onset_b AS (
        SELECT p.*, o.onset_b,
          sum(CASE WHEN p.dry_day = 0 OR p.dry_day IS NULL THEN 1 ELSE 0 END) OVER (
            PARTITION BY p."Climate Cell ID" ORDER BY p."Date"
          ) AS dry_group_b
        FROM persistence p
        INNER JOIN onset_summary o USING ("Climate Cell ID")
        WHERE o.onset_b IS NOT NULL
          AND p."Date" BETWEEN o.onset_b AND make_date({year}, 10, 31)
      ),
      dry_runs_b AS (
        SELECT "Climate Cell ID", dry_group_b,
          min("Date") AS dry_start_b,
          max("Date") AS dry_end_b,
          count(*) AS dry_length_b,
          sum(hot_day) AS hot_days_in_dry_run_b
        FROM post_onset_b WHERE dry_day = 1 GROUP BY 1, 2
      ),
      ranked_dry_runs_b AS (
        SELECT *, row_number() OVER (
          PARTITION BY "Climate Cell ID" ORDER BY dry_length_b DESC, dry_start_b
        ) AS dry_rank_b
        FROM dry_runs_b
      ),
      post_onset_summary_b AS (
        SELECT "Climate Cell ID",
          sum(CASE WHEN dry_day = 1 AND hot_day = 1 THEN 1 ELSE 0 END)
            AS post_onset_hot_dry_days_b,
          sum(CASE WHEN dry_day = 1 THEN 1 ELSE 0 END) AS post_onset_dry_days_b,
          sum(heat_day_33) AS post_onset_absolute_heat_days_33_b,
          sum(heat_day_35) AS post_onset_absolute_heat_days_35_b,
          sum(heat_day_37) AS post_onset_absolute_heat_days_37_b,
          sum(heat_degree_days_35) AS post_onset_heat_degree_days_35_b
        FROM post_onset_b GROUP BY 1
      ),
      annual AS (
        SELECT "Climate Cell ID",
          any_value("Climate Cell Longitude") AS "Climate Cell Longitude",
          any_value("Climate Cell Latitude") AS "Climate Cell Latitude"
        FROM daily GROUP BY 1
      )
      SELECT
        a."Climate Cell ID", a."Climate Cell Longitude", a."Climate Cell Latitude",
        {year}::INTEGER AS "Year",
        o.first_wet_spell_a AS "First Wet Spell Date Candidate A",
        o.onset_a AS "Wet-Season Onset Date Candidate A",
        dayofyear(o.onset_a) AS "Wet-Season Onset DOY Candidate A",
        CASE WHEN o.first_wet_spell_a IS NULL THEN NULL
             WHEN o.onset_a IS NULL OR o.onset_a > o.first_wet_spell_a THEN 1 ELSE 0 END
          AS "False Onset Indicator Candidate A",
        date_diff('day', o.first_wet_spell_a, o.onset_a)
          AS "False Onset Gap Days Candidate A",
        o.first_wet_spell_b AS "First Wet Spell Date Candidate B",
        o.onset_b AS "Wet-Season Onset Date Candidate B",
        dayofyear(o.onset_b) AS "Wet-Season Onset DOY Candidate B",
        CASE WHEN o.first_wet_spell_b IS NULL THEN NULL
             WHEN o.onset_b IS NULL OR o.onset_b > o.first_wet_spell_b THEN 1 ELSE 0 END
          AS "False Onset Indicator Candidate B",
        date_diff('day', o.first_wet_spell_b, o.onset_b)
          AS "False Onset Gap Days Candidate B",
        o.onset_accumulation AS "Accumulation Wet-Season Onset Date",
        dayofyear(o.onset_accumulation) AS "Accumulation Wet-Season Onset DOY",
        o.cessation_accumulation AS "Accumulation Wet-Season Cessation Date",
        dayofyear(o.cessation_accumulation) AS "Accumulation Wet-Season Cessation DOY",
        CASE WHEN o.cessation_accumulation > o.onset_accumulation
             THEN date_diff('day', o.onset_accumulation, o.cessation_accumulation) + 1 END
          AS "Accumulation Wet-Season Length Days",
        r.dry_start AS "Longest Intraseasonal Dry Spell Start Date Candidate A",
        dayofyear(r.dry_start) AS "Longest Intraseasonal Dry Spell Start DOY Candidate A",
        r.dry_length AS "Longest Intraseasonal Dry Spell Days Candidate A",
        r.hot_days_in_dry_run AS "Hot Days During Longest Dry Spell Candidate A",
        s.post_onset_dry_days AS "Post-Onset Dry Day Count Candidate A",
        s.post_onset_hot_dry_days AS "Post-Onset Hot-Dry Day Count Candidate A",
        s.post_onset_absolute_heat_days_33 AS "Post-Onset Absolute Heat Day Count 33 C Candidate A",
        s.post_onset_absolute_heat_days_35 AS "Post-Onset Absolute Heat Day Count 35 C Candidate A",
        s.post_onset_absolute_heat_days_37 AS "Post-Onset Absolute Heat Day Count 37 C Candidate A",
        s.post_onset_heat_degree_days_35 AS "Post-Onset Heat Degree-Days Above 35 C Candidate A",
        rb.dry_start_b AS "Longest Intraseasonal Dry Spell Start Date Candidate B",
        dayofyear(rb.dry_start_b) AS "Longest Intraseasonal Dry Spell Start DOY Candidate B",
        rb.dry_length_b AS "Longest Intraseasonal Dry Spell Days Candidate B",
        rb.hot_days_in_dry_run_b AS "Hot Days During Longest Dry Spell Candidate B",
        sb.post_onset_dry_days_b AS "Post-Onset Dry Day Count Candidate B",
        sb.post_onset_hot_dry_days_b AS "Post-Onset Hot-Dry Day Count Candidate B",
        sb.post_onset_absolute_heat_days_33_b AS "Post-Onset Absolute Heat Day Count 33 C Candidate B",
        sb.post_onset_absolute_heat_days_35_b AS "Post-Onset Absolute Heat Day Count 35 C Candidate B",
        sb.post_onset_absolute_heat_days_37_b AS "Post-Onset Absolute Heat Day Count 37 C Candidate B",
        sb.post_onset_heat_degree_days_35_b AS "Post-Onset Heat Degree-Days Above 35 C Candidate B"
      FROM annual a
      LEFT JOIN onset_summary o USING ("Climate Cell ID")
      LEFT JOIN ranked_dry_runs r
        ON a."Climate Cell ID" = r."Climate Cell ID" AND r.dry_rank = 1
      LEFT JOIN post_onset_summary s USING ("Climate Cell ID")
      LEFT JOIN ranked_dry_runs_b rb
        ON a."Climate Cell ID" = rb."Climate Cell ID" AND rb.dry_rank_b = 1
      LEFT JOIN post_onset_summary_b sb USING ("Climate Cell ID")
      ORDER BY a."Climate Cell ID"
    ) TO '{q(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    if not partition_valid(output, year):
        raise RuntimeError(f"Invalid monsoon partition for {year}")
    return output, "constructed"


def build_final(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    source = root / PARTITIONS / "year=*" / "monsoon.parquet"
    baseline_check = con.execute(
        f"""
        SELECT count(DISTINCT "Year"), min("Year"), max("Year")
        FROM read_parquet('{q(source)}', hive_partitioning=true)
        WHERE "Year" BETWEEN 1991 AND 2020
        """
    ).fetchone()
    if baseline_check != (30, 1991, 2020):
        raise RuntimeError(f"Complete 1991-2020 baseline required; observed {baseline_check}")

    variables = [
        "Wet-Season Onset DOY Candidate A",
        "Wet-Season Onset DOY Candidate B",
        "Accumulation Wet-Season Onset DOY",
        "Accumulation Wet-Season Cessation DOY",
        "Accumulation Wet-Season Length Days",
        "Longest Intraseasonal Dry Spell Days Candidate A",
        "Hot Days During Longest Dry Spell Candidate A",
        "Post-Onset Hot-Dry Day Count Candidate A",
        "Longest Intraseasonal Dry Spell Days Candidate B",
        "Hot Days During Longest Dry Spell Candidate B",
        "Post-Onset Hot-Dry Day Count Candidate B",
    ]
    stats: list[str] = []
    additions: list[str] = []
    for index, variable in enumerate(variables):
        alias = f"v{index}"
        stats.extend(
            [
                f'avg("{variable}") AS {alias}_mean',
                f'stddev_samp("{variable}") AS {alias}_sd',
                f'count("{variable}") AS {alias}_count',
            ]
        )
        additions.extend(
            [
                f'b.{alias}_mean AS "{variable} Mean 1991-2020"',
                f'b.{alias}_sd AS "{variable} SD 1991-2020"',
                f'b.{alias}_count AS "{variable} Valid Years 1991-2020"',
                f'CASE WHEN b.{alias}_sd > 0 AND b.{alias}_count >= 20 '
                f'THEN (a."{variable}" - b.{alias}_mean) / b.{alias}_sd END '
                f'AS "{variable} Anomaly Z"',
            ]
        )
    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    query = f"""
    COPY (
      WITH a AS (
        SELECT * FROM read_parquet('{q(source)}', hive_partitioning=true)
      ),
      b AS (
        SELECT "Climate Cell ID", {', '.join(stats)}
        FROM a WHERE "Year" BETWEEN 1991 AND 2020 GROUP BY 1
      )
      SELECT a.*, {', '.join(additions)}
      FROM a LEFT JOIN b USING ("Climate Cell ID")
      ORDER BY a."Climate Cell ID", a."Year"
    ) TO '{q(output)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con.execute(query)


def write_audit(root: Path, years: list[int], statuses: list[str]) -> None:
    audit = root / AUDIT
    audit.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(root / OUTPUT)
    coverage_rows = []
    for year, frame in panel.groupby("Year", sort=True):
        coverage_rows.append(
            {
                "Year": int(year),
                "Climate Cells": int(frame["Climate Cell ID"].nunique()),
                "Candidate A Onset Available Share": float(
                    frame["Wet-Season Onset DOY Candidate A"].notna().mean()
                ),
                "Candidate B Onset Available Share": float(
                    frame["Wet-Season Onset DOY Candidate B"].notna().mean()
                ),
                "Candidate A False Onset Share": float(
                    frame["False Onset Indicator Candidate A"].mean()
                ),
                "Candidate B False Onset Share": float(
                    frame["False Onset Indicator Candidate B"].mean()
                ),
                "Longest Dry Spell Available Share": float(
                    frame["Longest Intraseasonal Dry Spell Days Candidate A"].notna().mean()
                ),
                "Candidate B Longest Dry Spell Available Share": float(
                    frame["Longest Intraseasonal Dry Spell Days Candidate B"].notna().mean()
                ),
            }
        )
    pd.DataFrame(coverage_rows).to_csv(audit / "coverage_by_year.csv", index=False)
    metadata = {
        "years_requested": years,
        "partition_statuses": dict(zip(years, statuses)),
        "reference_period": [1991, 2020],
        "dry_day": "daily precipitation below 1 mm",
        "candidate_a": (
            "first 3-day total >=20 mm from 15 April to 31 August, with no "
            "10-consecutive-dry-day spell in the next 20 days"
        ),
        "candidate_b": (
            "first 3-day total >=30 mm from 15 April to 31 August, with no "
            "14-day period totaling <5 mm in the next 30 days"
        ),
        "accumulation_method": (
            "annual cumulative daily rainfall anomaly; minimum in April-August is onset "
            "and maximum in August-November is cessation"
        ),
        "primary_heat_threshold": "CHIRTS-ERA5 daily maximum air temperature at or above 35 degrees C",
        "absolute_heat_sensitivity_thresholds_c": [33, 37],
        "absolute_heat_reporting_scale": (
            "post-onset day counts in natural units; regression effects per 10 days"
        ),
        "secondary_heat_intensity": "post-onset degree-days above 35 degrees C",
        "relative_heat_threshold_role": (
            "Appendix diagnostic: climate-cell by calendar-month daily maximum temperature P90, 1991-2020"
        ),
        "outcome_blind": True,
        "npp_or_cses_read": False,
        "rows": int(len(panel)),
        "climate_cells": int(panel["Climate Cell ID"].nunique()),
    }
    (audit / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary = pd.DataFrame(coverage_rows)
    readme = f"""# Outcome-blind monsoon timing candidates

- Climate cells: {metadata['climate_cells']:,}
- Cell-years: {metadata['rows']:,}
- Reference period: 1991-2020
- Candidate A mean onset coverage: {summary['Candidate A Onset Available Share'].mean():.3f}
- Candidate B mean onset coverage: {summary['Candidate B Onset Available Share'].mean():.3f}

The file contains two agronomic candidates and one cumulative-anomaly timing
benchmark. No NPP or household outcome was read during construction. The
preferred definition must be frozen from event support and agronomic logic
before outcome estimation.
"""
    (audit / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if args.start_year > args.end_year:
        raise ValueError("start-year must be no later than end-year")
    con = connect(root)
    years = list(range(args.start_year, args.end_year + 1))
    statuses: list[str] = []
    try:
        for year in years:
            path, status = build_year(con, root, year, args.force)
            statuses.append(status)
            print(f"Completed {year}: {status}; {path.relative_to(root)}", flush=True)
        if not args.skip_final:
            build_final(con, root)
    finally:
        con.close()
    if not args.skip_final:
        write_audit(root, years, statuses)
        print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
