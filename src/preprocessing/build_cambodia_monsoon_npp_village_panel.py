#!/usr/bin/env python3
"""Build the national village-buffer monsoon and NPP analysis panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


MONSOON = Path("data/processed/cambodia_national_monsoon_timing_preprocessed.parquet")
ANNUAL_HAZARDS = Path(
    "data/processed/cambodia_national_annual_climate_hazards_preprocessed.parquet"
)
GRID_CLIMATE = Path("data/processed/cambodia_national_1km_to_climate_cell_preprocessed.parquet")
VILLAGE_MEMBERSHIP = Path("data/processed/cambodia_public_village_buffer_grid_crosswalk")
BASE_PANEL = Path(
    "data/processed/cambodia_public_village_npp_conflict_panel_candidate_preprocessed.parquet"
)
GRID_NPP = Path("data/processed/cambodia_national_modis_npp_annual_preprocessed.parquet")
GRID_CONTEXT = Path("data/processed/cambodia_national_predetermined_covariates_preprocessed.parquet")
OUTPUT = Path("data/processed/cambodia_public_village_monsoon_npp_panel_preprocessed.parquet")
AUDIT = Path("data/exp/data-preprocessing/climate-welfare/village-monsoon-npp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def q(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = [MONSOON, ANNUAL_HAZARDS, GRID_CLIMATE, BASE_PANEL, GRID_NPP, GRID_CONTEXT]
    paths.extend(VILLAGE_MEMBERSHIP / f"radius_{radius}_km.parquet" for radius in (2, 5, 10))
    for relative in paths:
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

    membership_parts = [
        f"SELECT * FROM read_parquet('{q(root / VILLAGE_MEMBERSHIP / f'radius_{radius}_km.parquet')}')"
        for radius in (2, 5, 10)
    ]
    membership_sql = " UNION ALL ".join(membership_parts)
    monsoon_fields = [
        "Annual Precipitation Total mm Anomaly Z",
        "May October Precipitation Total mm Anomaly Z",
        "Wet-Season Onset DOY Candidate A",
        "Wet-Season Onset DOY Candidate A Anomaly Z",
        "False Onset Indicator Candidate A",
        "Wet-Season Onset DOY Candidate B",
        "Wet-Season Onset DOY Candidate B Anomaly Z",
        "False Onset Indicator Candidate B",
        "Accumulation Wet-Season Onset DOY",
        "Accumulation Wet-Season Cessation DOY",
        "Accumulation Wet-Season Length Days",
        "Accumulation Wet-Season Length Days Anomaly Z",
        "Longest Intraseasonal Dry Spell Start DOY Candidate A",
        "Longest Intraseasonal Dry Spell Days Candidate A",
        "Longest Intraseasonal Dry Spell Days Candidate A Anomaly Z",
        "Hot Days During Longest Dry Spell Candidate A",
        "Post-Onset Hot-Dry Day Count Candidate A",
        "Post-Onset Hot-Dry Day Count Candidate A Anomaly Z",
        "Post-Onset Absolute Heat Day Count 33 C Candidate A",
        "Post-Onset Absolute Heat Day Count 35 C Candidate A",
        "Post-Onset Absolute Heat Day Count 37 C Candidate A",
        "Post-Onset Heat Degree-Days Above 35 C Candidate A",
        "Longest Intraseasonal Dry Spell Start DOY Candidate B",
        "Longest Intraseasonal Dry Spell Days Candidate B",
        "Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z",
        "Hot Days During Longest Dry Spell Candidate B",
        "Post-Onset Hot-Dry Day Count Candidate B",
        "Post-Onset Hot-Dry Day Count Candidate B Anomaly Z",
        "Post-Onset Absolute Heat Day Count 33 C Candidate B",
        "Post-Onset Absolute Heat Day Count 35 C Candidate B",
        "Post-Onset Absolute Heat Day Count 37 C Candidate B",
        "Post-Onset Heat Degree-Days Above 35 C Candidate B",
    ]
    weighted_fields = [
        (
            f'sum(m."{field}" * w.grid_weight) '
            f'/ nullif(sum(w.grid_weight) FILTER (WHERE m."{field}" IS NOT NULL), 0) '
            f'AS "Village Buffer Mean {field}"'
        )
        for field in monsoon_fields
    ]
    query = f"""
    COPY (
      WITH membership AS ({membership_sql}),
      climate_weights AS (
        SELECT x."National Village Point ID", x."Buffer Radius km",
          g."Climate Cell ID", count(*)::DOUBLE AS grid_weight
        FROM membership x
        INNER JOIN read_parquet('{q(root / GRID_CLIMATE)}') g USING ("National Grid Cell ID")
        GROUP BY 1, 2, 3
      ),
      climate AS (
        SELECT m.*,
          a."Annual Precipitation Total mm Anomaly Z",
          a."May October Precipitation Total mm Anomaly Z"
        FROM read_parquet('{q(root / MONSOON)}') m
        LEFT JOIN read_parquet('{q(root / ANNUAL_HAZARDS)}') a
          USING ("Climate Cell ID", "Year")
      ),
      village_monsoon AS (
        SELECT w."National Village Point ID", w."Buffer Radius km", m."Year",
          count(DISTINCT w."Climate Cell ID") AS "Village Buffer Climate Cell Count",
          sum(w.grid_weight) AS "Village Buffer Climate Grid Cell Weight",
          {', '.join(weighted_fields)}
        FROM climate_weights w
        INNER JOIN climate m USING ("Climate Cell ID")
        WHERE m."Year" BETWEEN 2001 AND 2024
        GROUP BY 1, 2, 3
      ),
      cropland_raw AS (
        SELECT x."National Village Point ID", n."Year",
          sum(c."Baseline Cropland Share") AS "Baseline Cropland Weight Sum",
          count(*) FILTER (WHERE c."Baseline Cropland Share" > 0)
            AS "Cropland-Weighted Grid Cell Count",
          sum(n."Annual Land NPP Mean kg C per m2" * c."Baseline Cropland Share")
            / nullif(sum(c."Baseline Cropland Share") FILTER (
              WHERE n."Annual Land NPP Mean kg C per m2" IS NOT NULL), 0)
            AS "Baseline-Cropland-Weighted Annual Land NPP Mean kg C per m2",
          sum(n."Mean NPP QC Filled Growing-Season Days Percent" * c."Baseline Cropland Share")
            / nullif(sum(c."Baseline Cropland Share") FILTER (
              WHERE n."Mean NPP QC Filled Growing-Season Days Percent" IS NOT NULL), 0)
            AS "Baseline-Cropland-Weighted Mean NPP QC Filled Growing-Season Days Percent"
        FROM read_parquet('{q(root / VILLAGE_MEMBERSHIP / 'radius_5_km.parquet')}') x
        INNER JOIN read_parquet('{q(root / GRID_NPP)}') n USING ("National Grid Cell ID")
        INNER JOIN read_parquet('{q(root / GRID_CONTEXT)}') c USING ("National Grid Cell ID")
        WHERE n."Year" BETWEEN 2001 AND 2024
          AND c."Baseline Cropland Share" IS NOT NULL
        GROUP BY 1, 2
      ),
      cropland_baseline AS (
        SELECT "National Village Point ID",
          avg("Baseline-Cropland-Weighted Annual Land NPP Mean kg C per m2")
            FILTER (WHERE "Year" BETWEEN 2001 AND 2020)
            AS cropland_npp_mean,
          stddev_samp("Baseline-Cropland-Weighted Annual Land NPP Mean kg C per m2")
            FILTER (WHERE "Year" BETWEEN 2001 AND 2020)
            AS cropland_npp_sd,
          count("Baseline-Cropland-Weighted Annual Land NPP Mean kg C per m2")
            FILTER (WHERE "Year" BETWEEN 2001 AND 2020)
            AS cropland_npp_count
        FROM cropland_raw GROUP BY 1
      ),
      cropland AS (
        SELECT r.*,
          r."Baseline-Cropland-Weighted Annual Land NPP Mean kg C per m2"
            - b.cropland_npp_mean
            AS "Baseline-Cropland-Weighted Annual Land NPP Anomaly kg C per m2",
          CASE WHEN b.cropland_npp_sd > 0 AND b.cropland_npp_count = 20
            THEN (r."Baseline-Cropland-Weighted Annual Land NPP Mean kg C per m2"
              - b.cropland_npp_mean) / b.cropland_npp_sd END
            AS "Baseline-Cropland-Weighted Annual Land NPP Anomaly Z 2001-2020",
          b.cropland_npp_count AS "Baseline-Cropland-Weighted NPP Valid Years 2001-2020"
        FROM cropland_raw r
        LEFT JOIN cropland_baseline b USING ("National Village Point ID")
      ),
      base AS (
        SELECT
          "National Village Point ID", "Buffer Radius km", "Year",
          "Village Buffer Grid Cell Count",
          "Village Buffer Mean Annual Land NPP Mean kg C per m2",
          "Village Buffer Mean Mean NPP QC Filled Growing-Season Days Percent",
          "Village Buffer Mean Annual Land NPP Anomaly kg C per m2",
          "Village Buffer Mean Annual Land NPP Anomaly Z 2001-2020",
          "Village Buffer Mean NPP Complete 2001-2020 Baseline",
          "Village Buffer Mean Annual Precipitation Total mm",
          "Village Buffer Mean May October Precipitation Total mm",
          "Village Buffer Mean May October Maximum Consecutive Dry Days Anomaly Z",
          "Village Buffer Mean May October Mean Daily Maximum Temperature C Anomaly Z",
          "Village Buffer Mean May October Compound Hot-Dry Intensity",
          "Village Buffer Mean Mean Elevation m", "Village Buffer Mean Mean Slope Degrees",
          "Village Buffer Mean Distance to Road Excluding Post 2007 AidData Corridors km",
          "Village Buffer Mean Baseline Cropland Share",
          "Village Buffer Mean Baseline Forest Share",
          "Village Buffer Mean Baseline Population Density per km2",
          "Village Buffer Mean Log Baseline Population 2000",
          "Source Village Administrative Code", "Public Village Name",
          "Point Longitude", "Point Latitude", "National Grid Cell ID",
          "Province Code", "Province Name", "District Code", "District Name",
          "Commune Code", "Commune Name", "Administrative Assignment Within 2 km"
        FROM read_parquet('{q(root / BASE_PANEL)}')
      )
      SELECT b.*, m.* EXCLUDE ("National Village Point ID", "Buffer Radius km", "Year"),
        CASE WHEN b."Buffer Radius km" = 5 THEN c."Baseline Cropland Weight Sum" END
          AS "Baseline Cropland Weight Sum",
        CASE WHEN b."Buffer Radius km" = 5 THEN c."Cropland-Weighted Grid Cell Count" END
          AS "Cropland-Weighted Grid Cell Count",
        CASE WHEN b."Buffer Radius km" = 5
          THEN c."Baseline-Cropland-Weighted Annual Land NPP Mean kg C per m2" END
          AS "Baseline-Cropland-Weighted Annual Land NPP Mean kg C per m2",
        CASE WHEN b."Buffer Radius km" = 5
          THEN c."Baseline-Cropland-Weighted Mean NPP QC Filled Growing-Season Days Percent" END
          AS "Baseline-Cropland-Weighted Mean NPP QC Filled Growing-Season Days Percent",
        CASE WHEN b."Buffer Radius km" = 5
          THEN c."Baseline-Cropland-Weighted Annual Land NPP Anomaly kg C per m2" END
          AS "Baseline-Cropland-Weighted Annual Land NPP Anomaly kg C per m2",
        CASE WHEN b."Buffer Radius km" = 5
          THEN c."Baseline-Cropland-Weighted Annual Land NPP Anomaly Z 2001-2020" END
          AS "Baseline-Cropland-Weighted Annual Land NPP Anomaly Z 2001-2020",
        CASE WHEN b."Buffer Radius km" = 5
          THEN c."Baseline-Cropland-Weighted NPP Valid Years 2001-2020" END
          AS "Baseline-Cropland-Weighted NPP Valid Years 2001-2020"
      FROM base b
      LEFT JOIN village_monsoon m USING ("National Village Point ID", "Buffer Radius km", "Year")
      LEFT JOIN cropland c USING ("National Village Point ID", "Year")
      ORDER BY b."National Village Point ID", b."Buffer Radius km", b."Year"
    ) TO '{q(output)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con.execute(query)
    con.close()

    panel = pd.read_parquet(output)
    keys = ["National Village Point ID", "Buffer Radius km", "Year"]
    if panel.duplicated(keys).any():
        raise RuntimeError("Duplicate village-buffer-year keys")
    expected = 13042 * 3 * 24
    if len(panel) != expected:
        raise RuntimeError(f"Expected {expected:,} rows, found {len(panel):,}")
    coverage = (
        panel.groupby(["Buffer Radius km", "Year"], observed=True)
        .agg(
            Rows=("National Village Point ID", "size"),
            Villages=("National Village Point ID", "nunique"),
            NPP_Available=("Village Buffer Mean Annual Land NPP Anomaly kg C per m2", "count"),
            Onset_A_Available=("Village Buffer Mean Wet-Season Onset DOY Candidate A", "count"),
            Onset_B_Available=("Village Buffer Mean Wet-Season Onset DOY Candidate B", "count"),
            Cropland_NPP_Available=(
                "Baseline-Cropland-Weighted Annual Land NPP Anomaly kg C per m2", "count"
            ),
        )
        .reset_index()
    )
    coverage.to_csv(audit / "coverage_by_radius_year.csv", index=False)
    metadata = {
        "unit": "national public village point by buffer radius by calendar year",
        "rows": int(len(panel)),
        "villages": int(panel["National Village Point ID"].nunique()),
        "years": [int(panel["Year"].min()), int(panel["Year"].max())],
        "radii_km": sorted(panel["Buffer Radius km"].unique().astype(int).tolist()),
        "primary_radius_km": 5,
        "monsoon_weighting": "number of stable one-kilometre grid cells per climate cell within each village buffer",
        "cropland_npp": (
            "five-kilometre sensitivity weighted by baseline cropland share; this is not an exact "
            "cropland-pixel mask and is labeled accordingly"
        ),
        "conflict_fields_retained": False,
    }
    (audit / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (audit / "README.md").write_text(
        "# National village monsoon-NPP panel\n\n"
        f"- Rows: {len(panel):,}\n"
        f"- Villages: {metadata['villages']:,}\n"
        "- Years: 2001-2024\n"
        "- Primary buffer: 5 km; 2 and 10 km are scale checks.\n"
        "- The cropland-weighted NPP sensitivity uses fixed baseline cropland shares and is not "
        "described as an exact cropland-pixel mask.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
