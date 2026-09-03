#!/usr/bin/env python3
"""Build the reusable Cambodia 1 km annual satellite-climate panel."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "data/processed/cambodia_national_1km_grid_preprocessed.parquet"
CROSSWALK = ROOT / "data/processed/cambodia_national_1km_to_climate_cell_preprocessed.parquet"
LONGNTL = ROOT / "data/processed/cambodia_national_longntl_annual_preprocessed.parquet"
NPP = ROOT / "data/processed/cambodia_national_modis_npp_annual_preprocessed.parquet"
CLIMATE = ROOT / "data/processed/cambodia_national_annual_climate_hazards_preprocessed.parquet"
OUTPUT = ROOT / "data/processed/cambodia_national_annual_satellite_climate_panel_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/national-annual-satellite-climate-panel"


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> None:
    required = [GRID, CROSSWALK, LONGNTL, NPP, CLIMATE]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    query = f"""
    COPY (
      SELECT
        l."National Grid Cell ID",
        l."Year",
        g."Longitude",
        g."Latitude",
        g."Province Code",
        g."Province Name",
        g."District Code",
        g."District Name",
        g."Commune Code",
        g."Commune Name",
        x."Climate Cell ID",
        l."Annual NPP-VIIRS-like Radiance",
        l."Asinh Annual NPP-VIIRS-like Radiance",
        l."Any Nonzero Annual NPP-VIIRS-like Radiance",
        l."LongNTL Source Stage",
        l."Reconstructed LongNTL Indicator",
        n."Annual Land NPP Mean kg C per m2",
        n."Mean NPP QC Filled Growing-Season Days Percent",
        n."Grid 2001-2020 Mean Annual Land NPP kg C per m2",
        n."Grid 2001-2020 SD Annual Land NPP kg C per m2",
        n."Grid 2001-2020 Valid NPP Year Count",
        n."Annual Land NPP Anomaly kg C per m2",
        n."Annual Land NPP Anomaly Z 2001-2020",
        n."NPP Complete 2001-2020 Baseline",
        c.* EXCLUDE ("Climate Cell ID", "Climate Cell Longitude", "Climate Cell Latitude", "Year")
      FROM read_parquet('{sql_path(LONGNTL)}') AS l
      INNER JOIN read_parquet('{sql_path(GRID)}') AS g
        USING ("National Grid Cell ID")
      INNER JOIN read_parquet('{sql_path(CROSSWALK)}') AS x
        USING ("National Grid Cell ID")
      LEFT JOIN read_parquet('{sql_path(NPP)}') AS n
        USING ("National Grid Cell ID", "Year")
      LEFT JOIN read_parquet('{sql_path(CLIMATE)}') AS c
        ON x."Climate Cell ID" = c."Climate Cell ID"
       AND l."Year" = c."Year"
      WHERE l."Year" BETWEEN 2000 AND 2024
      ORDER BY l."National Grid Cell ID", l."Year"
    ) TO '{sql_path(OUTPUT)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con.execute(query)

    summary = con.execute(
        f"""
        SELECT
          "Year",
          count(*) AS "Grid-Year Rows",
          count(DISTINCT "National Grid Cell ID") AS "Grid Cells",
          count("Annual NPP-VIIRS-like Radiance") AS "Valid LongNTL",
          count("Annual Land NPP Mean kg C per m2") AS "Valid NPP",
          count("Annual Precipitation Total mm") AS "Valid Precipitation",
          count("May October Mean Daily Maximum Temperature C") AS "Valid Temperature"
        FROM read_parquet('{sql_path(OUTPUT)}')
        GROUP BY "Year"
        ORDER BY "Year"
        """
    ).fetchdf()
    summary.to_csv(AUDIT_DIR / "annual_panel_coverage_by_year.csv", index=False)

    checks = con.execute(
        f"""
        SELECT
          count(*) AS rows,
          count(DISTINCT "National Grid Cell ID") AS cells,
          min("Year") AS min_year,
          max("Year") AS max_year,
          count(*) - count(DISTINCT ("National Grid Cell ID", "Year")) AS duplicate_keys,
          count(*) FILTER (WHERE "Climate Cell ID" IS NULL) AS missing_climate_crosswalk,
          count(*) FILTER (WHERE "Annual Precipitation Total mm" IS NULL) AS missing_precipitation,
          count(*) FILTER (WHERE "May October Mean Daily Maximum Temperature C" IS NULL) AS missing_temperature,
          count(*) FILTER (WHERE "Annual Land NPP Mean kg C per m2" IS NULL AND "Year" >= 2001) AS missing_npp_2001_2024
        FROM read_parquet('{sql_path(OUTPUT)}')
        """
    ).fetchone()
    names = [item[0] for item in con.description]
    audit = dict(zip(names, checks))
    audit.update(
        {
            "panel_unit": "Cambodia national 1 km grid cell by calendar year",
            "years": [2000, 2024],
            "join_policy": "LongNTL skeleton; administrative and climate crosswalk inner joins; NPP and annual climate left joins",
            "missing_value_policy": "No imputation; NPP is structurally unavailable in 2000",
            "extreme_rainfall_interpretation": "Rainfall extremes are meteorological hazards, not observed flood inundation",
            "source_files": [str(path.relative_to(ROOT)) for path in required],
            "output_file": str(OUTPUT.relative_to(ROOT)),
        }
    )
    (AUDIT_DIR / "annual_panel_metadata.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    expected_rows = 179_072 * 25
    if audit["rows"] != expected_rows or audit["duplicate_keys"] != 0:
        raise RuntimeError(f"Panel key validation failed: {audit}")
    if audit["missing_climate_crosswalk"] != 0:
        raise RuntimeError(f"Climate crosswalk validation failed: {audit}")

    print(f"Saved {OUTPUT.relative_to(ROOT)}; rows={audit['rows']:,}; cells={audit['cells']:,}")
    print(f"Coverage audit: {AUDIT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
