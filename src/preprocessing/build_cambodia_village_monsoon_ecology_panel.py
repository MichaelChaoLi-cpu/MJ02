#!/usr/bin/env python3
"""Join season-aligned vegetation outcomes to the national village monsoon panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


SEASONAL = Path("data/processed/cambodia_national_seasonal_vegetation_preprocessed.parquet")
GRID_CLIMATE = Path("data/processed/cambodia_national_1km_to_climate_cell_preprocessed.parquet")
MEMBERSHIP = Path("data/processed/cambodia_public_village_buffer_grid_crosswalk")
MONSOON_NPP = Path(
    "data/processed/cambodia_public_village_monsoon_npp_panel_preprocessed.parquet"
)
OUTPUT = Path(
    "data/processed/cambodia_public_village_monsoon_ecology_panel_preprocessed.parquet"
)
AUDIT = Path("data/exp/data-preprocessing/climate-welfare/village-monsoon-ecology")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def q(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    required = [SEASONAL, GRID_CLIMATE, MONSOON_NPP]
    required.extend(MEMBERSHIP / f"radius_{radius}_km.parquet" for radius in (2, 5, 10))
    for relative in required:
        if not (root / relative).exists():
            raise FileNotFoundError(root / relative)
    audit = root / AUDIT
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "duckdb-tmp").mkdir(parents=True, exist_ok=True)
    output = root / OUTPUT

    fields = [
        "May-October Mean EVI Anomaly Z",
        "May-October Mean NDVI Anomaly Z",
        "November-February Mean EVI Anomaly Z",
        "November-February Mean NDVI Anomaly Z",
        "May-February Production-Season Mean EVI Anomaly Z",
        "May-February Production-Season Mean NDVI Anomaly Z",
        "May-February Minimum EVI Anomaly Z",
        "May-February Peak EVI Anomaly Z",
        "May-October Valid EVI Composites",
        "November-February Valid EVI Composites",
        "May-February Valid EVI Composites",
        "Complete Production-Season EVI Support",
    ]
    weighted = [
        (
            f'sum(s."{field}" * w.grid_weight) '
            f'/ nullif(sum(w.grid_weight) FILTER (WHERE s."{field}" IS NOT NULL), 0) '
            f'AS "Village Buffer Mean {field}"'
        )
        for field in fields
    ]
    membership_sql = " UNION ALL ".join(
        f"SELECT * FROM read_parquet('{q(root / MEMBERSHIP / f'radius_{radius}_km.parquet')}')"
        for radius in (2, 5, 10)
    )

    con = duckdb.connect()
    con.execute("SET threads = 4")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{q(audit / 'duckdb-tmp')}'")
    query = f"""
    COPY (
      WITH membership AS ({membership_sql}),
      climate_weights AS (
        SELECT x."National Village Point ID", x."Buffer Radius km", g."Climate Cell ID",
          count(*)::DOUBLE AS grid_weight
        FROM membership x
        INNER JOIN read_parquet('{q(root / GRID_CLIMATE)}') g USING ("National Grid Cell ID")
        GROUP BY 1, 2, 3
      ),
      village_vegetation AS (
        SELECT w."National Village Point ID", w."Buffer Radius km",
          s."Production Season Year" AS "Year",
          count(DISTINCT w."Climate Cell ID") AS "Village Buffer Vegetation Climate Cell Count",
          {', '.join(weighted)}
        FROM climate_weights w
        INNER JOIN read_parquet('{q(root / SEASONAL)}') s USING ("Climate Cell ID")
        GROUP BY 1, 2, 3
      ),
      base AS (
        SELECT
          "National Village Point ID", "Buffer Radius km", "Year",
          "Point Longitude", "Point Latitude", "Province Code", "Province Name",
          "District Code", "District Name", "Commune Code", "Commune Name",
          "Public Village Name", "National Grid Cell ID",
          "Village Buffer Mean May October Precipitation Total mm Anomaly Z",
          "Village Buffer Mean Wet-Season Onset DOY Candidate A Anomaly Z",
          "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate A Anomaly Z",
          "Village Buffer Mean Post-Onset Hot-Dry Day Count Candidate A Anomaly Z",
          "Village Buffer Mean Post-Onset Absolute Heat Day Count 33 C Candidate A",
          "Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate A",
          "Village Buffer Mean Post-Onset Absolute Heat Day Count 37 C Candidate A",
          "Village Buffer Mean Post-Onset Heat Degree-Days Above 35 C Candidate A",
          "Village Buffer Mean Wet-Season Onset DOY Candidate B Anomaly Z",
          "Village Buffer Mean False Onset Indicator Candidate B",
          "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z",
          "Village Buffer Mean Post-Onset Hot-Dry Day Count Candidate B Anomaly Z",
          "Village Buffer Mean Post-Onset Absolute Heat Day Count 33 C Candidate B",
          "Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate B",
          "Village Buffer Mean Post-Onset Absolute Heat Day Count 37 C Candidate B",
          "Village Buffer Mean Post-Onset Heat Degree-Days Above 35 C Candidate B",
          "Village Buffer Mean Annual Land NPP Anomaly kg C per m2",
          "Village Buffer Mean Annual Land NPP Anomaly Z 2001-2020",
          "Baseline-Cropland-Weighted Annual Land NPP Anomaly kg C per m2",
          "Village Buffer Mean Baseline Cropland Share",
          "Village Buffer Mean Log Baseline Population 2000",
          "Village Buffer Mean Distance to Road Excluding Post 2007 AidData Corridors km",
          "Village Buffer Mean Mean Elevation m", "Village Buffer Mean Mean Slope Degrees"
        FROM read_parquet('{q(root / MONSOON_NPP)}')
      )
      SELECT b.*, v.* EXCLUDE ("National Village Point ID", "Buffer Radius km", "Year")
      FROM base b
      LEFT JOIN village_vegetation v USING ("National Village Point ID", "Buffer Radius km", "Year")
      ORDER BY b."National Village Point ID", b."Buffer Radius km", b."Year"
    ) TO '{q(output)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con.execute(query)
    con.close()

    frame = pd.read_parquet(output)
    keys = ["National Village Point ID", "Buffer Radius km", "Year"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate village-buffer-production-season keys")
    expected = 13042 * 3 * 24
    if len(frame) != expected:
        raise RuntimeError(f"Expected {expected:,} rows, found {len(frame):,}")
    coverage = (
        frame.groupby(["Buffer Radius km", "Year"], observed=True)
        .agg(
            Villages=("National Village Point ID", "nunique"),
            NPP=("Village Buffer Mean Annual Land NPP Anomaly kg C per m2", "count"),
            May_October_EVI=("Village Buffer Mean May-October Mean EVI Anomaly Z", "count"),
            Production_Season_EVI=(
                "Village Buffer Mean May-February Production-Season Mean EVI Anomaly Z", "count"
            ),
        )
        .reset_index()
    )
    coverage.to_csv(audit / "coverage_by_radius_year.csv", index=False)
    metadata = {
        "unit": "public village point by buffer radius by production-season year",
        "rows": int(len(frame)),
        "villages": int(frame["National Village Point ID"].nunique()),
        "radii_km": [2, 5, 10],
        "years": [2001, 2024],
        "primary_radius_km": 5,
        "primary_dynamic_outcome": "Village Buffer Mean May-February Production-Season Mean EVI Anomaly Z",
        "annual_npp_role": "annual cumulative ecological confirmation with a non-identical calendar window",
        "conflict_fields_retained": False,
    }
    (audit / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (audit / "README.md").write_text(
        "# Village monsoon-ecology panel\n\n"
        "This compact release combines outcome-blind monsoon measures, annual NPP, and fixed "
        "May-February EVI/NDVI production-season outcomes at 2, 5, and 10 km village buffers. "
        "Conflict variables are excluded.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(coverage.loc[coverage["Buffer Radius km"].eq(5)].to_string(index=False))


if __name__ == "__main__":
    main()
