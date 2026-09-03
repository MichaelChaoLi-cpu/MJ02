#!/usr/bin/env python3
"""Build the CSES household climate-ecology-welfare analysis frame.

The frame retains all geocoded main-sample households, explicitly marks the
subset linked to national public village points, and attaches only the last
completed May-October climate season. No outcome is imputed or winsorized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


HOUSEHOLDS = Path("data/processed/direction3_household_core_outcomes_preprocessed.parquet")
MEMBERS = Path("data/processed/direction3_education_core_outcomes_preprocessed.parquet")
PUBLIC_LINK = Path(
    "data/processed/cses_to_cambodia_public_village_point_crosswalk_preprocessed.parquet"
)
VILLAGE_PANEL = Path(
    "data/processed/cambodia_public_village_monsoon_npp_panel_preprocessed.parquet"
)
TIMING_2019 = Path("data/processed/cses_2019_interview_timing_preprocessed.parquet")
CPI_ANNUAL = Path("data/processed/cambodia_cpi_annual_preprocessed.parquet")
CPI_MONTHLY = Path("data/processed/cambodia_cpi_monthly_preprocessed.parquet")
OUTPUT = Path("data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet")
AUDIT = Path("data/exp/data-preprocessing/climate-welfare/cses-welfare-frame")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def q(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    for relative in (
        HOUSEHOLDS, MEMBERS, PUBLIC_LINK, VILLAGE_PANEL, TIMING_2019, CPI_ANNUAL, CPI_MONTHLY
    ):
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
      WITH member_composition AS (
        SELECT "Survey Year", "Household ID",
          count(*) FILTER (WHERE "Age Years" IS NOT NULL) AS "Members with Observed Age",
          avg("Female") FILTER (WHERE "Female" IS NOT NULL) AS "Female Household Member Share",
          avg("Age Years") FILTER (WHERE "Age Years" IS NOT NULL) AS "Mean Household Member Age Years",
          avg(CASE WHEN "Age Years" BETWEEN 0 AND 14 THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE "Age Years" IS NOT NULL) AS "Child Age 0-14 Share",
          avg(CASE WHEN "Age Years" BETWEEN 15 AND 64 THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE "Age Years" IS NOT NULL) AS "Working Age 15-64 Share",
          avg(CASE WHEN "Age Years" >= 65 THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE "Age Years" IS NOT NULL) AS "Older Age 65 Plus Share",
          sum(CASE WHEN "Age Years" < 15 OR "Age Years" >= 65 THEN 1 ELSE 0 END)::DOUBLE
            / nullif(sum(CASE WHEN "Age Years" BETWEEN 15 AND 64 THEN 1 ELSE 0 END), 0)
            AS "Household Dependency Ratio"
        FROM read_parquet('{q(root / MEMBERS)}')
        WHERE "Main Linked Sample" = 1
        GROUP BY 1, 2
      ),
      household_base AS (
        SELECT
          h."Survey Year", h."Survey Wave", h."Household ID", h."PSU",
          h."Province Code", h."Province Name", h."District Code", h."District Name",
          h."Commune Code", h."Commune Name", h."Village Code", h."Village Name",
          h."Urban Rural",
          coalesce(t."Interview Calendar Year", h."Survey Year") AS "Interview Calendar Year",
          coalesce(
            t."Interview Month",
            try_cast(try_cast(h."Survey Month" AS DOUBLE) AS INTEGER)
          ) AS "Interview Month",
          CASE WHEN t."Household ID" IS NOT NULL THEN 1 ELSE 0 END
            AS "Interview Timing Recovered from Raw 2019",
          CASE WHEN t."Interview Calendar Year" = 2020 AND t."Interview Month" = 12
            THEN 1 ELSE 0 END AS "Interview Timing Source Warning",
          h."Household Survey Weight", h."Household Size",
          h."Children Age 6 to 17 Count",
          CASE WHEN coalesce(t."Interview Month", try_cast(try_cast(h."Survey Month" AS DOUBLE) AS INTEGER))
                    BETWEEN 11 AND 12
                 THEN coalesce(t."Interview Calendar Year", h."Survey Year")
               WHEN coalesce(t."Interview Month", try_cast(try_cast(h."Survey Month" AS DOUBLE) AS INTEGER))
                    BETWEEN 1 AND 10
                 THEN coalesce(t."Interview Calendar Year", h."Survey Year") - 1 END
            AS "Last Complete May-October Season Year",
          h."Agricultural Household" AS "Agricultural Participation",
          h."Cultivated Crop Area m2" / 10000.0 AS "Cultivated Crop Area ha",
          CASE WHEN h."Cultivated Crop Area m2" > 0
                 AND h."Nominal Crop Production Value Riels" >= 0
                 AND ca."Annual Deflator to 2021" IS NOT NULL
            THEN h."Nominal Crop Production Value Riels" * ca."Annual Deflator to 2021"
              * 10000.0 / h."Cultivated Crop Area m2" END
            AS "Real 2021 Crop Production Value per Cultivated ha Riels",
          h."Nominal Crop Production Value Riels" * ca."Annual Deflator to 2021"
            AS "Real 2021 Crop Production Value Riels",
          h."Crop Diversity Count",
          h."Reported Food Consumption Value Riels"
            / nullif(h."Household Size", 0)
            * coalesce(cm."Deflator to 2021", cf."Annual Deflator to 2021")
            AS "Real 2021 Food Consumption Value per Household Member Riels",
          h."Own Produced Food Consumption Value Riels"
            / nullif(h."Household Size", 0)
            * coalesce(cm."Deflator to 2021", cf."Annual Deflator to 2021")
            AS "Real 2021 Own Produced Food Value per Household Member Riels",
          h."Any Severe Food Insecurity Experience", h."Food Insecurity Severity Sum",
          h."Irrigable Parcel Share", h."Any Irrigable Parcel",
          h."Irrigation Status Observation Count",
          h."Has Nonagriculture 1 Record" AS "Any Nonagriculture Module Record Candidate",
          h."Nonagriculture 1 Row Count" AS "Nonagriculture Module Row Count Candidate",
          CASE WHEN cm."Deflator to 2021" IS NOT NULL THEN 'monthly food CPI'
               WHEN cf."Annual Deflator to 2021" IS NOT NULL THEN 'annual food CPI fallback' END
            AS "Food Monetary Deflation Method",
          CASE WHEN ca."Annual Deflator to 2021" IS NOT NULL THEN 'annual all-items CPI' END
            AS "Agricultural Monetary Deflation Method"
        FROM read_parquet('{q(root / HOUSEHOLDS)}') h
        LEFT JOIN read_parquet('{q(root / TIMING_2019)}') t
          ON h."Survey Year" = 2019 AND h."Household ID" = t."Household ID"
        LEFT JOIN read_parquet('{q(root / CPI_ANNUAL)}') ca
          ON ca."Year" = coalesce(t."Interview Calendar Year", h."Survey Year")
         AND ca."CPI Component Code" = '_T'
        LEFT JOIN read_parquet('{q(root / CPI_MONTHLY)}') cm
          ON cm."Year" = coalesce(t."Interview Calendar Year", h."Survey Year")
         AND cm."Month" = coalesce(
           t."Interview Month", try_cast(try_cast(h."Survey Month" AS DOUBLE) AS INTEGER)
         )
         AND cm."CPI Component Code" = 'CP01'
        LEFT JOIN read_parquet('{q(root / CPI_ANNUAL)}') cf
          ON cf."Year" = coalesce(t."Interview Calendar Year", h."Survey Year")
         AND cf."CPI Component Code" = 'CP01'
        WHERE h."Main Linked Sample" = 1
      ),
      linked AS (
        SELECT h.*, x."Public Village Point Matched", x."National Village Point ID",
          x."Public Village Name", x."Point Longitude", x."Point Latitude",
          c.* EXCLUDE ("Survey Year", "Household ID")
        FROM household_base h
        LEFT JOIN read_parquet('{q(root / PUBLIC_LINK)}') x USING ("Village Code")
        LEFT JOIN member_composition c USING ("Survey Year", "Household ID")
      ),
      season_panel AS (
        SELECT * FROM read_parquet('{q(root / VILLAGE_PANEL)}')
        WHERE "Buffer Radius km" = 5
      ),
      prior_panel AS (
        SELECT "National Village Point ID", "Year",
          "Village Buffer Mean Annual Land NPP Anomaly kg C per m2"
            AS "Prior Calendar-Year Village NPP Anomaly kg C per m2",
          "Village Buffer Mean Annual Land NPP Anomaly Z 2001-2020"
            AS "Prior Calendar-Year Village NPP Anomaly Z 2001-2020",
          "Baseline-Cropland-Weighted Annual Land NPP Anomaly kg C per m2"
            AS "Prior Calendar-Year Baseline-Cropland-Weighted NPP Anomaly kg C per m2",
          "Baseline-Cropland-Weighted Annual Land NPP Anomaly Z 2001-2020"
            AS "Prior Calendar-Year Baseline-Cropland-Weighted NPP Anomaly Z 2001-2020"
        FROM season_panel
      )
      SELECT l.*,
        CASE WHEN l."Public Village Point Matched" = 1
                  AND s."National Village Point ID" IS NOT NULL THEN 1 ELSE 0 END
          AS "Climate Ecology Link Available",
        CASE WHEN l."Last Complete May-October Season Year" < l."Interview Calendar Year"
          THEN 1 ELSE 0 END AS "Season Annual NPP Fully Pre-Interview",
        s."Village Buffer Mean Wet-Season Onset DOY Candidate A"
          AS "Last Complete Season Wet-Season Onset DOY Candidate A",
        s."Village Buffer Mean Wet-Season Onset DOY Candidate A Anomaly Z"
          AS "Last Complete Season Wet-Season Onset Anomaly Z Candidate A",
        s."Village Buffer Mean False Onset Indicator Candidate A"
          AS "Last Complete Season False Onset Share Candidate A",
        s."Village Buffer Mean Wet-Season Onset DOY Candidate B"
          AS "Last Complete Season Wet-Season Onset DOY Candidate B",
        s."Village Buffer Mean Wet-Season Onset DOY Candidate B Anomaly Z"
          AS "Last Complete Season Wet-Season Onset Anomaly Z Candidate B",
        s."Village Buffer Mean False Onset Indicator Candidate B"
          AS "Last Complete Season False Onset Share Candidate B",
        s."Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate A"
          AS "Last Complete Season Longest Dry Spell Days Candidate A",
        s."Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate A Anomaly Z"
          AS "Last Complete Season Longest Dry Spell Anomaly Z Candidate A",
        s."Village Buffer Mean Post-Onset Hot-Dry Day Count Candidate A"
          AS "Last Complete Season Hot-Dry Day Count Candidate A",
        s."Village Buffer Mean Post-Onset Hot-Dry Day Count Candidate A Anomaly Z"
          AS "Last Complete Season Hot-Dry Day Count Anomaly Z Candidate A",
        s."Village Buffer Mean Post-Onset Absolute Heat Day Count 33 C Candidate A"
          AS "Last Complete Season Absolute Heat Day Count 33 C Candidate A",
        s."Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate A"
          AS "Last Complete Season Absolute Heat Day Count 35 C Candidate A",
        s."Village Buffer Mean Post-Onset Absolute Heat Day Count 37 C Candidate A"
          AS "Last Complete Season Absolute Heat Day Count 37 C Candidate A",
        s."Village Buffer Mean Post-Onset Heat Degree-Days Above 35 C Candidate A"
          AS "Last Complete Season Heat Degree-Days Above 35 C Candidate A",
        s."Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B"
          AS "Last Complete Season Longest Dry Spell Days Candidate B",
        s."Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z"
          AS "Last Complete Season Longest Dry Spell Anomaly Z Candidate B",
        s."Village Buffer Mean Post-Onset Hot-Dry Day Count Candidate B"
          AS "Last Complete Season Hot-Dry Day Count Candidate B",
        s."Village Buffer Mean Post-Onset Hot-Dry Day Count Candidate B Anomaly Z"
          AS "Last Complete Season Hot-Dry Day Count Anomaly Z Candidate B",
        s."Village Buffer Mean Post-Onset Absolute Heat Day Count 33 C Candidate B"
          AS "Last Complete Season Absolute Heat Day Count 33 C Candidate B",
        s."Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate B"
          AS "Last Complete Season Absolute Heat Day Count 35 C Candidate B",
        s."Village Buffer Mean Post-Onset Absolute Heat Day Count 37 C Candidate B"
          AS "Last Complete Season Absolute Heat Day Count 37 C Candidate B",
        s."Village Buffer Mean Post-Onset Heat Degree-Days Above 35 C Candidate B"
          AS "Last Complete Season Heat Degree-Days Above 35 C Candidate B",
        s."Village Buffer Mean May October Precipitation Total mm"
          AS "Last Complete Season May-October Precipitation Total mm",
        s."Village Buffer Mean May October Precipitation Total mm Anomaly Z"
          AS "Last Complete Season May-October Precipitation Anomaly Z",
        s."Village Buffer Mean Annual Land NPP Anomaly kg C per m2"
          AS "Last Complete Season Annual NPP Anomaly kg C per m2",
        s."Village Buffer Mean Annual Land NPP Anomaly Z 2001-2020"
          AS "Last Complete Season Annual NPP Anomaly Z 2001-2020",
        s."Baseline-Cropland-Weighted Annual Land NPP Anomaly kg C per m2"
          AS "Last Complete Season Baseline-Cropland-Weighted NPP Anomaly kg C per m2",
        s."Baseline-Cropland-Weighted Annual Land NPP Anomaly Z 2001-2020"
          AS "Last Complete Season Baseline-Cropland-Weighted NPP Anomaly Z 2001-2020",
        s."Village Buffer Mean Distance to Road Excluding Post 2007 AidData Corridors km"
          AS "Historical Road Distance km",
        s."Village Buffer Mean Log Baseline Population 2000"
          AS "Log Baseline Population 2000",
        s."Village Buffer Mean Baseline Cropland Share" AS "Baseline Cropland Share",
        s."Village Buffer Mean Mean Elevation m" AS "Mean Elevation m",
        s."Village Buffer Mean Mean Slope Degrees" AS "Mean Slope Degrees",
        p.* EXCLUDE ("National Village Point ID", "Year")
      FROM linked l
      LEFT JOIN season_panel s
        ON l."National Village Point ID" = s."National Village Point ID"
       AND l."Last Complete May-October Season Year" = s."Year"
      LEFT JOIN prior_panel p
        ON l."National Village Point ID" = p."National Village Point ID"
       AND l."Interview Calendar Year" - 1 = p."Year"
      ORDER BY l."Survey Year", l."Household ID"
    ) TO '{q(output)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con.execute(query)
    con.close()

    frame = pd.read_parquet(output)
    keys = ["Survey Year", "Household ID"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate household-wave keys in welfare frame")
    if len(frame) != 62920:
        raise RuntimeError(f"Expected 62,920 households, found {len(frame):,}")

    coverage = (
        frame.groupby("Survey Year", observed=True)
        .agg(
            Households=("Household ID", "size"),
            Villages=("Village Code", "nunique"),
            Public_Linked_Households=("Public Village Point Matched", "sum"),
            Climate_Ecology_Linked_Households=("Climate Ecology Link Available", "sum"),
            Food_Consumption_Observed=(
                "Real 2021 Food Consumption Value per Household Member Riels", "count"
            ),
            Agricultural_Participation_Observed=("Agricultural Participation", "count"),
            Crop_Intensity_Observed=(
                "Real 2021 Crop Production Value per Cultivated ha Riels", "count"
            ),
            Irrigation_Observed=("Any Irrigable Parcel", "count"),
            Food_Insecurity_Observed=("Any Severe Food Insecurity Experience", "count"),
        )
        .reset_index()
    )
    coverage.to_csv(audit / "coverage_by_survey_year.csv", index=False)
    dictionary = pd.DataFrame(
        {
            "Variable": frame.columns,
            "Data Type": [str(frame[column].dtype) for column in frame.columns],
            "Missing Percent": [float(frame[column].isna().mean() * 100) for column in frame.columns],
        }
    )
    dictionary.to_csv(audit / "variable_dictionary.csv", index=False)
    metadata = {
        "unit": "geocoded CSES household by survey wave",
        "rows": int(len(frame)),
        "survey_years": sorted(frame["Survey Year"].unique().astype(int).tolist()),
        "villages": int(frame["Village Code"].nunique()),
        "public_village_linked_households": int(frame["Public Village Point Matched"].eq(1).sum()),
        "climate_ecology_linked_households": int(frame["Climate Ecology Link Available"].sum()),
        "recovered_2019_2020_interview_timing_households": int(
            frame["Interview Timing Recovered from Raw 2019"].sum()
        ),
        "primary_outcomes": [
            "Agricultural Participation",
            "Real 2021 Crop Production Value per Cultivated ha Riels",
            "Real 2021 Food Consumption Value per Household Member Riels",
        ],
        "buffer_candidates": [
            "Any Irrigable Parcel",
            "Historical Road Distance km",
            "Log Baseline Population 2000",
        ],
        "nonagriculture_module_candidate": (
            "retained for coverage audit only; not confirmatory because the 2019 module structure differs"
        ),
        "missing_data": "no imputation",
        "outliers": "no winsorization",
        "season_rule": (
            "May-October of the actual interview calendar year only for November-December "
            "interviews; otherwise the previous calendar year"
        ),
        "monetary_deflation": (
            "food values use actual interview year-month food CPI; crop values use actual "
            "interview-year annual all-items CPI; both are expressed in 2021 riels"
        ),
        "observed_npp_household_rule": (
            "last-complete-season annual NPP is not primary for same-year November-December interviews; "
            "the prior-calendar-year NPP field is fully pre-interview"
        ),
    }
    (audit / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (audit / "README.md").write_text(
        "# CSES climate-ecology-welfare frame\n\n"
        f"- Geocoded main-sample households: {len(frame):,}\n"
        f"- Public-village-linked households: {metadata['public_village_linked_households']:,}\n"
        f"- Distinct CSES villages: {metadata['villages']:,}\n"
        "- Primary outcomes are agricultural participation, real crop-production value per "
        "cultivated hectare, and real food consumption per household member.\n"
        "- No missing value was imputed and no outcome was winsorized.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
