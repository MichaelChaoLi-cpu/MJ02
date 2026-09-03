#!/usr/bin/env python3
"""Build survey-window MODIS NDVI weighted by dynamic annual cropland shares."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HOUSEHOLDS = ROOT / (
    "data/exp/analysis/climate-welfare/rolling-multiseason-ndvi-food-test/"
    "household_rolling_12month_sample.parquet"
)
MEMBERSHIP = ROOT / "data/processed/cses_village_buffer_grid_crosswalk_preprocessed.parquet"
VEGETATION = ROOT / "data/processed/cambodia_national_modis_vegetation_16day"
CROPLAND = ROOT / "data/processed/cambodia_national_annual_cropland_share_preprocessed.parquet"
OUTPUT = ROOT / "data/processed/cses_village_dynamic_cropland_ndvi_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/dynamic-modis-cropland-ndvi"


def q(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    required = [HOUSEHOLDS, MEMBERSHIP, VEGETATION, CROPLAND]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cropland-ndvi-", dir=AUDIT_DIR) as temporary:
        con = duckdb.connect((Path(temporary) / "cropland_ndvi.duckdb").as_posix())
        con.execute("SET threads = 4")
        con.execute("SET memory_limit = '8GB'")
        con.execute("SET preserve_insertion_order = false")
        con.execute(f"SET temp_directory = '{q(Path(temporary) / 'spill')}'")
        con.execute(f"""
          CREATE TABLE survey_windows AS
          SELECT DISTINCT "Village Code", interview_date, window_start, window_end
          FROM read_parquet('{q(HOUSEHOLDS)}')
        """)
        con.execute(f"""
          CREATE TABLE membership AS
          SELECT m."Village Code", m."National Grid Cell ID"
          FROM read_parquet('{q(MEMBERSHIP)}') m
          SEMI JOIN survey_windows s USING ("Village Code")
          WHERE m."Buffer Radius km" = 5
        """)
        print("Aggregating same-year cropland-share-weighted MOD13Q1 NDVI...", flush=True)
        con.execute(f"""
          CREATE TABLE village_composite AS
          SELECT m."Village Code", v."Composite Date", v."Year",
            sum(v."Mean NDVI" * v."NDVI Valid Pixel Count" * c."Strict Cropland Share")
              / nullif(sum(v."NDVI Valid Pixel Count" * c."Strict Cropland Share") FILTER (
                  WHERE v."Mean NDVI" IS NOT NULL AND c."Strict Cropland Share" > 0), 0)
              AS strict_cropland_weighted_ndvi,
            sum(v."Mean NDVI" * v."NDVI Valid Pixel Count" * c."Inclusive Agricultural Share")
              / nullif(sum(v."NDVI Valid Pixel Count" * c."Inclusive Agricultural Share") FILTER (
                  WHERE v."Mean NDVI" IS NOT NULL AND c."Inclusive Agricultural Share" > 0), 0)
              AS inclusive_agriculture_weighted_ndvi,
            sum(c."Strict Cropland Share") FILTER (WHERE v."Mean NDVI" IS NOT NULL)
              AS strict_equivalent_1km_cells,
            sum(c."Inclusive Agricultural Share") FILTER (WHERE v."Mean NDVI" IS NOT NULL)
              AS inclusive_equivalent_1km_cells,
            count(*) FILTER (
              WHERE v."Mean NDVI" IS NOT NULL AND c."Strict Cropland Share" > 0
            ) AS strict_contributing_grid_cells,
            count(*) FILTER (
              WHERE v."Mean NDVI" IS NOT NULL AND c."Inclusive Agricultural Share" > 0
            ) AS inclusive_contributing_grid_cells
          FROM membership m
          INNER JOIN read_parquet('{q(VEGETATION)}/**/*.parquet', hive_partitioning = true) v
            USING ("National Grid Cell ID")
          INNER JOIN read_parquet('{q(CROPLAND)}') c
            ON c."National Grid Cell ID" = v."National Grid Cell ID"
            AND c."Year" = v."Year"
          WHERE v."Year" BETWEEN 2006 AND 2021
          GROUP BY 1, 2, 3
        """)
        print("Aligning cropland NDVI to twelve complete pre-interview months...", flush=True)
        con.execute(f"""
          COPY (
            SELECT s."Village Code", s.interview_date, s.window_start, s.window_end,
              avg(v.strict_cropland_weighted_ndvi)
                AS "Trailing Twelve-Month Strict-Cropland Weighted Absolute NDVI",
              avg(v.inclusive_agriculture_weighted_ndvi)
                AS "Trailing Twelve-Month Inclusive-Agriculture Weighted Absolute NDVI",
              count(v.strict_cropland_weighted_ndvi)
                AS "Strict-Cropland Valid NDVI Composites",
              count(v.inclusive_agriculture_weighted_ndvi)
                AS "Inclusive-Agriculture Valid NDVI Composites",
              avg(v.strict_equivalent_1km_cells)
                AS "Mean Strict-Cropland Equivalent 1km Cells",
              avg(v.inclusive_equivalent_1km_cells)
                AS "Mean Inclusive-Agriculture Equivalent 1km Cells",
              avg(v.strict_contributing_grid_cells)
                AS "Mean Strict-Cropland Contributing Grid Cells",
              avg(v.inclusive_contributing_grid_cells)
                AS "Mean Inclusive-Agriculture Contributing Grid Cells"
            FROM survey_windows s
            LEFT JOIN village_composite v ON v."Village Code" = s."Village Code"
              AND v."Composite Date" >= s.window_start
              AND v."Composite Date" + INTERVAL 15 DAY <= s.window_end
            GROUP BY 1, 2, 3, 4
            ORDER BY 1, 2
          ) TO '{q(OUTPUT)}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        con.close()

    output = pd.read_parquet(OUTPUT)
    strict = "Trailing Twelve-Month Strict-Cropland Weighted Absolute NDVI"
    inclusive = "Trailing Twelve-Month Inclusive-Agriculture Weighted Absolute NDVI"
    if output.duplicated(["Village Code", "interview_date"]).any():
        raise RuntimeError("Cropland-NDVI output has duplicate village-interview keys")
    coverage = pd.DataFrame(
        [
            {
                "Measure": strict,
                "Village-Interview Windows": len(output),
                "Observed Windows": int(output[strict].notna().sum()),
                "Observed Share": float(output[strict].notna().mean()),
                "Windows With At Least 18 Valid Composites": int(
                    output["Strict-Cropland Valid NDVI Composites"].ge(18).sum()
                ),
                "Median Equivalent 1km Cells": float(
                    output["Mean Strict-Cropland Equivalent 1km Cells"].median()
                ),
            },
            {
                "Measure": inclusive,
                "Village-Interview Windows": len(output),
                "Observed Windows": int(output[inclusive].notna().sum()),
                "Observed Share": float(output[inclusive].notna().mean()),
                "Windows With At Least 18 Valid Composites": int(
                    output["Inclusive-Agriculture Valid NDVI Composites"].ge(18).sum()
                ),
                "Median Equivalent 1km Cells": float(
                    output["Mean Inclusive-Agriculture Equivalent 1km Cells"].median()
                ),
            },
        ]
    )
    coverage.to_csv(AUDIT_DIR / "coverage_summary.csv", index=False)
    metadata = {
        "unit": "CSES-linked village by interview date",
        "buffer_radius_km": 5,
        "vegetation_source": "MODIS MOD13Q1.061 16-day NDVI",
        "land_cover_source": "MODIS MCD12Q1.061 annual LC_Type1",
        "strict_definition": "IGBP class 12 Croplands",
        "inclusive_definition": "IGBP classes 12 and 14",
        "weighting": (
            "Within each 5 km village buffer and composite, weight each 1 km grid-cell mean "
            "NDVI by its valid MOD13Q1 pixel count times its same-year MCD12Q1 cropland share"
        ),
        "temporal_window": "twelve complete months before the interview month",
        "scale": "absolute NDVI index units; not standardized",
        "screening_limit": (
            "Cropland-share weighting of 1 km cell means is an exploratory screen, not an exact "
            "250 m cropland-pixel mask"
        ),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "processed_at_utc": datetime.now(UTC).isoformat(),
    }
    (AUDIT_DIR / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
