#!/usr/bin/env python3
"""Construct interview-preceding climate windows for CSES village buffers.

The annual CSES-satellite release attaches calendar-year climate to a survey
record.  That can include weather occurring after an interview.  This script
instead ends the 12-month window on the final day of the month before interview
and attaches the last fully completed May-October season.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRECIP = ROOT / "data/processed/cambodia_national_daily_precipitation"
TEMP = ROOT / "data/processed/cambodia_national_daily_temperature"
MEMBERSHIP = ROOT / "data/processed/cses_village_buffer_grid_crosswalk_preprocessed.parquet"
GRID_CLIMATE = ROOT / "data/processed/cambodia_national_1km_to_climate_cell_preprocessed.parquet"
ANNUAL_BUFFER = ROOT / "data/processed/cses_village_buffer_annual_satellite_preprocessed.parquet"
CSES_CONFLICT = ROOT / "data/processed/cambodia_thailand_cses_conflict_panel_candidate_preprocessed.parquet"

ROLLING_CELL_OUT = ROOT / "data/processed/cambodia_national_monthly_rolling_climate_preprocessed.parquet"
INTERVIEW_CLIMATE_OUT = ROOT / "data/processed/cses_village_interview_aligned_climate_preprocessed.parquet"
FINAL_OUT = ROOT / "data/processed/cambodia_thailand_cses_conflict_climate_panel_candidate_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/cses-interview-aligned-climate"


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def build_cell_rolling_climate() -> None:
    ROLLING_CELL_OUT.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET threads = 4")
    con.execute(f"""
      COPY (
        WITH precip_month AS (
          SELECT
            "Climate Cell ID",
            date_trunc('month', "Date")::DATE AS "Month",
            sum("Daily Precipitation mm") AS "Monthly Precipitation Total mm",
            count("Daily Precipitation mm") AS "Monthly Valid Precipitation Days"
          FROM read_parquet('{q(PRECIP)}/**/*.parquet', hive_partitioning=true)
          GROUP BY 1, 2
        ),
        temp_month AS (
          SELECT
            "Climate Cell ID",
            date_trunc('month', "Date")::DATE AS "Month",
            sum("Daily Maximum Temperature C") AS "Monthly Maximum Temperature Sum C",
            sum("Daily Minimum Temperature C") AS "Monthly Minimum Temperature Sum C",
            count("Daily Maximum Temperature C") AS "Monthly Valid Temperature Days"
          FROM read_parquet('{q(TEMP)}/**/*.parquet', hive_partitioning=true)
          GROUP BY 1, 2
        ),
        monthly AS (
          SELECT p.*, t."Monthly Maximum Temperature Sum C",
            t."Monthly Minimum Temperature Sum C", t."Monthly Valid Temperature Days"
          FROM precip_month p
          INNER JOIN temp_month t USING ("Climate Cell ID", "Month")
        ),
        rolling AS (
          SELECT *,
            count(*) OVER w AS "Rolling Month Count",
            sum("Monthly Precipitation Total mm") OVER w AS "Preceding 12 Month Precipitation Total mm",
            sum("Monthly Valid Precipitation Days") OVER w AS "Preceding 12 Month Valid Precipitation Days",
            sum("Monthly Maximum Temperature Sum C") OVER w
              / nullif(sum("Monthly Valid Temperature Days") OVER w, 0)
              AS "Preceding 12 Month Mean Daily Maximum Temperature C",
            sum("Monthly Minimum Temperature Sum C") OVER w
              / nullif(sum("Monthly Valid Temperature Days") OVER w, 0)
              AS "Preceding 12 Month Mean Daily Minimum Temperature C"
          FROM monthly
          WINDOW w AS (
            PARTITION BY "Climate Cell ID" ORDER BY "Month"
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
          )
        ),
        complete AS (
          SELECT *, month("Month") AS "Window End Calendar Month"
          FROM rolling
          WHERE "Rolling Month Count" = 12
        ),
        baseline AS (
          SELECT *,
            avg("Preceding 12 Month Precipitation Total mm") FILTER (
              WHERE year("Month") BETWEEN 1992 AND 2020
            ) OVER (PARTITION BY "Climate Cell ID", "Window End Calendar Month")
              AS "Baseline 1992-2020 Precipitation Mean mm",
            stddev_samp("Preceding 12 Month Precipitation Total mm") FILTER (
              WHERE year("Month") BETWEEN 1992 AND 2020
            ) OVER (PARTITION BY "Climate Cell ID", "Window End Calendar Month")
              AS "Baseline 1992-2020 Precipitation SD mm",
            avg("Preceding 12 Month Mean Daily Maximum Temperature C") FILTER (
              WHERE year("Month") BETWEEN 1992 AND 2020
            ) OVER (PARTITION BY "Climate Cell ID", "Window End Calendar Month")
              AS "Baseline 1992-2020 Maximum Temperature Mean C",
            stddev_samp("Preceding 12 Month Mean Daily Maximum Temperature C") FILTER (
              WHERE year("Month") BETWEEN 1992 AND 2020
            ) OVER (PARTITION BY "Climate Cell ID", "Window End Calendar Month")
              AS "Baseline 1992-2020 Maximum Temperature SD C"
          FROM complete
        )
        SELECT
          "Climate Cell ID", "Month" AS "Window End Month",
          "Window End Calendar Month",
          "Preceding 12 Month Precipitation Total mm",
          "Preceding 12 Month Valid Precipitation Days",
          "Preceding 12 Month Mean Daily Maximum Temperature C",
          "Preceding 12 Month Mean Daily Minimum Temperature C",
          ("Preceding 12 Month Precipitation Total mm" - "Baseline 1992-2020 Precipitation Mean mm")
            / nullif("Baseline 1992-2020 Precipitation SD mm", 0)
            AS "Preceding 12 Month Precipitation Anomaly Z",
          ("Preceding 12 Month Mean Daily Maximum Temperature C" - "Baseline 1992-2020 Maximum Temperature Mean C")
            / nullif("Baseline 1992-2020 Maximum Temperature SD C", 0)
            AS "Preceding 12 Month Maximum Temperature Anomaly Z"
        FROM baseline
        ORDER BY "Climate Cell ID", "Window End Month"
      ) TO '{q(ROLLING_CELL_OUT)}'
      (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)
    con.close()


def build_interview_windows() -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET threads = 4")
    con.execute(f"""
      COPY (
        WITH weights AS (
          SELECT m."Village Code", m."Buffer Radius km", g."Climate Cell ID",
            count(*)::DOUBLE AS "Grid Cell Weight"
          FROM read_parquet('{q(MEMBERSHIP)}') m
          INNER JOIN read_parquet('{q(GRID_CLIMATE)}') g USING ("National Grid Cell ID")
          GROUP BY 1, 2, 3
        ),
        survey_windows AS (
          SELECT DISTINCT
            "Village Code", "Buffer Radius km", "Survey Year",
            "Survey Month Minimum"::INTEGER AS "Interview Month",
            (make_date("Survey Year"::INTEGER, "Survey Month Minimum"::INTEGER, 1)
              - INTERVAL 1 MONTH)::DATE AS "Preceding Window End Month",
            CASE WHEN "Survey Month Minimum"::INTEGER >= 11
              THEN "Survey Year"::INTEGER ELSE "Survey Year"::INTEGER - 1 END
              AS "Last Complete May October Season Year"
          FROM read_parquet('{q(CSES_CONFLICT)}')
          WHERE "Public Village Point Matched" = 1
            AND "Single Survey Month Within Village" = 1
            AND "Survey Month Minimum" BETWEEN 1 AND 12
        ),
        twelve_month AS (
          SELECT s."Village Code", s."Buffer Radius km", s."Survey Year",
            s."Interview Month", s."Preceding Window End Month",
            s."Last Complete May October Season Year",
            sum(r."Preceding 12 Month Precipitation Total mm" * w."Grid Cell Weight")
              / nullif(sum(w."Grid Cell Weight") FILTER (
                WHERE r."Preceding 12 Month Precipitation Total mm" IS NOT NULL), 0)
              AS "Preceding 12 Month Precipitation Total mm",
            sum(r."Preceding 12 Month Precipitation Anomaly Z" * w."Grid Cell Weight")
              / nullif(sum(w."Grid Cell Weight") FILTER (
                WHERE r."Preceding 12 Month Precipitation Anomaly Z" IS NOT NULL), 0)
              AS "Preceding 12 Month Precipitation Anomaly Z",
            sum(r."Preceding 12 Month Mean Daily Maximum Temperature C" * w."Grid Cell Weight")
              / nullif(sum(w."Grid Cell Weight") FILTER (
                WHERE r."Preceding 12 Month Mean Daily Maximum Temperature C" IS NOT NULL), 0)
              AS "Preceding 12 Month Mean Daily Maximum Temperature C",
            sum(r."Preceding 12 Month Maximum Temperature Anomaly Z" * w."Grid Cell Weight")
              / nullif(sum(w."Grid Cell Weight") FILTER (
                WHERE r."Preceding 12 Month Maximum Temperature Anomaly Z" IS NOT NULL), 0)
              AS "Preceding 12 Month Maximum Temperature Anomaly Z",
            sum(w."Grid Cell Weight") AS "Interview Climate Buffer Grid Cell Count"
          FROM survey_windows s
          INNER JOIN weights w USING ("Village Code", "Buffer Radius km")
          INNER JOIN read_parquet('{q(ROLLING_CELL_OUT)}') r
            ON r."Climate Cell ID" = w."Climate Cell ID"
            AND r."Window End Month" = s."Preceding Window End Month"
          GROUP BY 1, 2, 3, 4, 5, 6
        )
        SELECT t.*,
          a."Buffer Mean May October Precipitation Total mm" AS "Last Complete May October Precipitation Total mm",
          a."Buffer Mean May October Precipitation Total mm Anomaly Z" AS "Last Complete May October Precipitation Anomaly Z",
          a."Buffer Mean May October Dry Rainfall Intensity" AS "Last Complete May October Dry Rainfall Intensity",
          a."Buffer Mean May October Maximum Consecutive Dry Days Anomaly Z" AS "Last Complete May October Maximum Consecutive Dry Days Anomaly Z",
          a."Buffer Mean May October Mean Daily Maximum Temperature C Anomaly Z" AS "Last Complete May October Mean Maximum Temperature Anomaly Z",
          a."Buffer Mean May October Heat Intensity" AS "Last Complete May October Heat Intensity",
          a."Buffer Mean May October Compound Hot-Dry Intensity" AS "Last Complete May October Compound Hot-Dry Intensity"
        FROM twelve_month t
        LEFT JOIN read_parquet('{q(ANNUAL_BUFFER)}') a
          ON a."Village Code" = t."Village Code"
          AND a."Buffer Radius km" = t."Buffer Radius km"
          AND a."Year" = t."Last Complete May October Season Year"
        ORDER BY t."Survey Year", t."Village Code", t."Buffer Radius km"
      ) TO '{q(INTERVIEW_CLIMATE_OUT)}'
      (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()
    climate = pd.read_parquet(INTERVIEW_CLIMATE_OUT)
    climate["Preceding 12 Month Dry Rainfall Intensity"] = (
        -climate["Preceding 12 Month Precipitation Anomaly Z"]
    ).clip(lower=0)
    climate["Preceding 12 Month Heat Intensity"] = climate[
        "Preceding 12 Month Maximum Temperature Anomaly Z"
    ].clip(lower=0)
    climate["Preceding 12 Month Compound Hot-Dry Intensity"] = (
        climate["Preceding 12 Month Dry Rainfall Intensity"]
        * climate["Preceding 12 Month Heat Intensity"]
    )
    climate.to_parquet(INTERVIEW_CLIMATE_OUT, index=False)
    return climate


def build_final_panel(climate: pd.DataFrame) -> pd.DataFrame:
    cses = pd.read_parquet(CSES_CONFLICT)
    keys = ["Village Code", "Buffer Radius km", "Survey Year"]
    final = cses.merge(climate, on=keys, how="left", validate="one_to_one")
    final["Interview-Aligned Climate Available"] = final[
        "Preceding 12 Month Precipitation Anomaly Z"
    ].notna().astype("int8")
    if final.duplicated(keys).any():
        raise RuntimeError("Final CSES conflict-climate panel has duplicate keys")
    final.to_parquet(FINAL_OUT, index=False)
    return final


def write_audit(climate: pd.DataFrame, final: pd.DataFrame) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    coverage = (
        final.groupby(["Survey Year", "Buffer Radius km"], dropna=False)
        .agg(
            CSES_Rows=("Village Code", "size"),
            Spatially_Matched_Rows=("Public Village Point Matched", "sum"),
            Interview_Climate_Rows=("Interview-Aligned Climate Available", "sum"),
            Affected_District_Rows=("Candidate Affected District", "sum"),
        )
        .reset_index()
    )
    coverage.to_csv(AUDIT_DIR / "interview_aligned_climate_coverage.csv", index=False)
    dictionary = pd.DataFrame({
        "Variable": climate.columns,
        "Data Type": [str(climate[c].dtype) for c in climate.columns],
        "Missing Percent": [float(climate[c].isna().mean() * 100) for c in climate.columns],
    })
    dictionary.to_csv(AUDIT_DIR / "variable_dictionary.csv", index=False)
    metadata = {
        "preceding_window_rule": "12 complete calendar months ending on the final day of the month before interview",
        "season_rule": "most recent fully completed May-October season before interview month",
        "standardization": "climate-cell and window-end-calendar-month baseline, 1992-2020",
        "multi_month_village_policy": "17 village-years spanning multiple interview months remain missing",
        "no_imputation": True,
        "rolling_cell_rows": int(pd.read_parquet(ROLLING_CELL_OUT, columns=["Climate Cell ID"]).shape[0]),
        "interview_climate_rows": int(len(climate)),
        "final_cses_rows": int(len(final)),
        "outputs": {
            "rolling_cell_climate": str(ROLLING_CELL_OUT.relative_to(ROOT)),
            "interview_aligned_climate": str(INTERVIEW_CLIMATE_OUT.relative_to(ROOT)),
            "final_candidate_panel": str(FINAL_OUT.relative_to(ROOT)),
        },
    }
    (AUDIT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    decisions = {
        "preceding_12_month_window": "12 complete calendar months ending before interview month",
        "last_complete_growing_season": "May-October of survey year only for November-December interviews; otherwise previous year",
        "baseline": "1992-2020 by climate cell and window-end calendar month",
        "primary_cses_shock": "Preceding 12 Month Dry Rainfall Intensity",
        "temperature_extension": "Preceding 12 Month Heat Intensity",
        "ambiguous_interview_months": "leave missing; do not choose one month or impute",
        "buffer_radii_km": [2, 5, 10],
        "primary_buffer_km": 5,
    }
    (AUDIT_DIR / "decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme = f"""# CSES interview-aligned climate preprocessing

- Climate-cell rolling windows: {metadata['rolling_cell_rows']:,}
- Mapped village-buffer-survey records with resolved interview climate: {len(climate):,}
- Final CSES rows retained, including unmatched records: {len(final):,}

The preceding 12-month window ends before the interview month. The growing-season
measure uses the most recent fully completed May-October season. Seventeen village-years
with multiple interview months remain missing, and no weather value is imputed.
"""
    (AUDIT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(coverage.to_string(index=False))


def main() -> None:
    for path in [PRECIP, TEMP, MEMBERSHIP, GRID_CLIMATE, ANNUAL_BUFFER, CSES_CONFLICT]:
        if not path.exists():
            raise FileNotFoundError(path)
    build_cell_rolling_climate()
    climate = build_interview_windows()
    final = build_final_panel(climate)
    write_audit(climate, final)


if __name__ == "__main__":
    main()
