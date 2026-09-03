#!/usr/bin/env python3
"""Prepare the public AidData Cambodia road-project layer for design audits.

The source contains project corridors, not a complete national road network.  The
processed layer is therefore used to flag known post-2007 construction and to
audit historical-road covariates; it is never treated as the primary road-access
measure.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import geopandas as gpd
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_ARCHIVE = ROOT / "data/raw/roads/aiddata_cambodia_roads/cambodia_roads.geojson.zip"
RAW_GEOJSON = ROOT / (
    "data/raw/roads/aiddata_cambodia_roads/extracted/"
    "cambodia_roads_geojson/cambodia_roads.geojson"
)
RAW_CODEBOOK = ROOT / (
    "data/raw/roads/aiddata_cambodia_roads/extracted/"
    "cambodia_roads_geojson/Cambodia Roads Codebook.xlsx"
)
OUTPUT = ROOT / "data/processed/aiddata_cambodia_road_projects_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/aiddata-cambodia-road-projects"

SOURCE_PAGE = "https://www.aiddata.org/publications/highway-to-the-forest"
SOURCE_ARCHIVE_URL = "https://docs.aiddata.org/ad4/datasets/cambodia_roads.geojson.zip"
SOURCE_ARTICLE_DOI = "10.1016/j.jeem.2023.102898"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def main() -> None:
    missing = [path for path in (RAW_ARCHIVE, RAW_GEOJSON, RAW_CODEBOOK) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing AidData road source files: {missing}")

    roads = gpd.read_file(RAW_GEOJSON)
    if roads.crs is None:
        raise RuntimeError("AidData road GeoJSON has no declared CRS")
    roads = roads.to_crs("EPSG:4326")
    if roads.empty or roads.geometry.isna().any() or roads.geometry.is_empty.any():
        raise RuntimeError("AidData road source has empty or missing geometries")
    if not roads.geometry.is_valid.all():
        raise RuntimeError("AidData road source has invalid geometries")

    rename = {
        "project_id": "AidData Project ID",
        "project_title": "AidData Project Title",
        "desciption": "AidData Project Description",
        "ad_sector_names": "AidData Sector",
        "status": "AidData Project Status",
        "transactions_start_year": "Transaction Start Year",
        "transactions_end_year": "Transaction End Year",
        "project_name": "Road Project Name",
        "coding": "Infrastructure Coding",
        "end.date": "Reported Completion Date",
        "work.type": "Road Work Type",
        "confidence": "AidData Confidence",
        "year": "Project Completion Year",
    }
    roads = roads.rename(columns=rename)
    roads["Transaction Start Year"] = numeric_year(roads["Transaction Start Year"])
    roads["Transaction End Year"] = numeric_year(roads["Transaction End Year"])
    roads["Project Completion Year"] = numeric_year(roads["Project Completion Year"])
    roads["Road Work Type"] = roads["Road Work Type"].astype("string").str.lower().str.strip()

    metric = roads.to_crs("EPSG:32648")
    roads["Road Project Length km"] = (metric.length / 1000).astype("float32")
    roads["Known New Road Project"] = roads["Road Work Type"].eq("new")
    roads["Known Post 2007 Completion"] = roads["Project Completion Year"].gt(2007)
    roads["Known Post 2007 New Road Project"] = (
        roads["Known New Road Project"] & roads["Known Post 2007 Completion"]
    )
    roads.insert(
        0,
        "Road Project Feature ID",
        [f"aiddata_khm_road_{index + 1:03d}" for index in range(len(roads))],
    )
    roads = roads.sort_values(
        ["Project Completion Year", "AidData Project ID", "Road Project Feature ID"],
        na_position="last",
    ).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    roads.to_parquet(OUTPUT, index=False)

    summary = (
        roads.groupby("Road Work Type", dropna=False)
        .agg(
            Project_Features=("Road Project Feature ID", "size"),
            Unique_Projects=("AidData Project ID", "nunique"),
            Total_Length_km=("Road Project Length km", "sum"),
            Earliest_Completion_Year=("Project Completion Year", "min"),
            Latest_Completion_Year=("Project Completion Year", "max"),
        )
        .reset_index()
        .rename(
            columns={
                "Road Work Type": "Road Work Type",
                "Project_Features": "Project Features",
                "Unique_Projects": "Unique Projects",
                "Total_Length_km": "Total Length km",
                "Earliest_Completion_Year": "Earliest Completion Year",
                "Latest_Completion_Year": "Latest Completion Year",
            }
        )
    )
    summary.to_csv(AUDIT_DIR / "road_project_summary_by_work_type.csv", index=False)

    inventory = roads[
        [
            "Road Project Feature ID",
            "AidData Project ID",
            "Road Project Name",
            "Road Work Type",
            "Transaction Start Year",
            "Project Completion Year",
            "Road Project Length km",
            "Known Post 2007 New Road Project",
            "AidData Confidence",
        ]
    ].copy()
    inventory.to_csv(AUDIT_DIR / "road_project_feature_inventory.csv", index=False)

    metadata = {
        "title": "AidData geocoded Chinese government-financed road projects in Cambodia",
        "source_page": SOURCE_PAGE,
        "source_archive_url": SOURCE_ARCHIVE_URL,
        "source_article_doi": SOURCE_ARTICLE_DOI,
        "raw_archive_sha256": sha256(RAW_ARCHIVE),
        "raw_archive_bytes": RAW_ARCHIVE.stat().st_size,
        "source_features": int(len(roads)),
        "unique_projects": int(roads["AidData Project ID"].nunique()),
        "total_project_length_km_epsg32648": float(roads["Road Project Length km"].sum()),
        "completion_year_min": int(roads["Project Completion Year"].min()),
        "completion_year_max": int(roads["Project Completion Year"].max()),
        "output_crs": roads.crs.to_string(),
        "processing_crs_for_length": "EPSG:32648",
        "role": "audit known post-2007 road projects and potential contamination of the historical road covariate",
        "not_for": "complete national road access or a standalone pre-conflict road network",
        "outcome_blind": True,
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
    }
    (AUDIT_DIR / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
