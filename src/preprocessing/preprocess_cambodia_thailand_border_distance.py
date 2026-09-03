#!/usr/bin/env python3
"""Construct Cambodia-grid distance to the Cambodia-Thailand shared border."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "data/processed/cambodia_national_1km_grid_preprocessed.parquet"
CAMBODIA = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
THAILAND_ARCHIVE = ROOT / (
    "data/raw/geography/thailand_rtsd_ocha_2019/"
    "tha_admbnda_adm0_rtsd_20190221.zip"
)
THAILAND = ROOT / (
    "data/raw/geography/thailand_rtsd_ocha_2019/extracted/"
    "tha_admbnda_adm0_rtsd_20190221.shp"
)
OUTPUT = ROOT / "data/processed/cambodia_national_thailand_border_distance_preprocessed.parquet"
BORDER_OUTPUT = ROOT / "data/processed/cambodia_thailand_shared_border_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/cambodia-thailand-shared-border"

PRIMARY_MATCH_TOLERANCE_M = 2000
TOLERANCE_AUDIT_M = (100, 250, 500, 1000, 2000, 5000)
DISTANCE_BANDS_KM = (10, 20, 40, 60)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shared_border(cambodia_boundary: object, thailand_boundary: object, tolerance_m: int) -> object:
    line = cambodia_boundary.intersection(thailand_boundary.buffer(tolerance_m))
    if line.is_empty:
        raise RuntimeError(f"No shared-border line found at {tolerance_m} m tolerance")
    return line


def main() -> None:
    missing = [path for path in (GRID, CAMBODIA, THAILAND_ARCHIVE, THAILAND) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing border inputs: {missing}")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    grid = pd.read_parquet(GRID)
    cambodia = gpd.read_file(CAMBODIA).to_crs("EPSG:32648")
    thailand = gpd.read_file(THAILAND).to_crs("EPSG:32648")
    cambodia_polygon = cambodia.geometry.union_all()
    thailand_polygon = thailand.geometry.union_all()
    if not cambodia_polygon.is_valid or not thailand_polygon.is_valid:
        raise RuntimeError("Cambodia or Thailand national geometry is invalid")

    tolerance_rows = []
    for tolerance in TOLERANCE_AUDIT_M:
        candidate = shared_border(
            cambodia_polygon.boundary,
            thailand_polygon.boundary,
            tolerance,
        )
        tolerance_rows.append(
            {
                "Boundary Match Tolerance m": tolerance,
                "Extracted Shared Border Length km": candidate.length / 1000,
                "Geometry Type": candidate.geom_type,
            }
        )
    pd.DataFrame(tolerance_rows).to_csv(
        AUDIT_DIR / "shared_border_tolerance_sensitivity.csv", index=False
    )

    border = shared_border(
        cambodia_polygon.boundary,
        thailand_polygon.boundary,
        PRIMARY_MATCH_TOLERANCE_M,
    )
    if not 700_000 <= border.length <= 900_000:
        raise RuntimeError(f"Implausible Cambodia-Thailand border length: {border.length / 1000:.1f} km")
    gpd.GeoDataFrame(
        {
            "Border ID": ["KHM_THA_primary_cambodia_side"],
            "Cambodia Boundary Source": ["Cambodia Department of Geography via OCHA/MEF"],
            "Thailand Boundary Source": ["Royal Thai Survey Department via ICRC/OCHA"],
            "Boundary Match Tolerance m": [PRIMARY_MATCH_TOLERANCE_M],
            "Shared Border Length km": [border.length / 1000],
        },
        geometry=[border],
        crs="EPSG:32648",
    ).to_parquet(BORDER_OUTPUT, index=False)

    points = shapely.points(
        grid["Grid Centre Easting m"].to_numpy(dtype=float),
        grid["Grid Centre Northing m"].to_numpy(dtype=float),
    )
    distance_km = shapely.distance(points, border) / 1000
    nearest_lines = shapely.shortest_line(points, border)
    nearest_points = shapely.get_point(nearest_lines, 1)
    nearest_easting = shapely.get_x(nearest_points)
    nearest_northing = shapely.get_y(nearest_points)
    to_wgs84 = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)
    nearest_longitude, nearest_latitude = to_wgs84.transform(nearest_easting, nearest_northing)
    longitude_bin_start = np.floor(np.asarray(nearest_longitude) * 2) / 2

    output = grid[
        [
            "National Grid Cell ID",
            "Province Code",
            "Province Name",
            "District Code",
            "District Name",
            "Commune Code",
            "Commune Name",
        ]
    ].copy()
    output["Distance to Cambodia Thailand Border km"] = distance_km.astype("float32")
    output["Nearest Border Easting m"] = nearest_easting.astype("float32")
    output["Nearest Border Northing m"] = nearest_northing.astype("float32")
    output["Nearest Border Longitude"] = np.asarray(nearest_longitude, dtype="float32")
    output["Nearest Border Latitude"] = np.asarray(nearest_latitude, dtype="float32")
    output["Nearest Border Sector"] = [
        f"Longitude {start:.1f} to {start + 0.5:.1f} E" for start in longitude_bin_start
    ]
    for radius in DISTANCE_BANDS_KM:
        output[f"Within {radius} km of Cambodia Thailand Border"] = distance_km <= radius
    if output["National Grid Cell ID"].duplicated().any() or len(output) != 179_072:
        raise RuntimeError("Border-distance output failed national grid-key validation")
    if not np.isfinite(distance_km).all() or distance_km.min() < 0:
        raise RuntimeError("Border-distance output contains invalid values")
    output.to_parquet(OUTPUT, index=False)

    summary = pd.DataFrame(
        {
            "Metric": [
                "Grid cells",
                "Shared border length km",
                "Minimum distance km",
                "Median distance km",
                "P90 distance km",
                *[f"Cells within {radius} km" for radius in DISTANCE_BANDS_KM],
            ],
            "Value": [
                len(output),
                border.length / 1000,
                float(np.min(distance_km)),
                float(np.median(distance_km)),
                float(np.quantile(distance_km, 0.90)),
                *[int(np.sum(distance_km <= radius)) for radius in DISTANCE_BANDS_KM],
            ],
        }
    )
    summary.to_csv(AUDIT_DIR / "border_distance_summary.csv", index=False)
    output["Nearest Border Sector"].value_counts().rename_axis("Nearest Border Sector").reset_index(
        name="Grid Cells"
    ).sort_values("Nearest Border Sector").to_csv(
        AUDIT_DIR / "nearest_border_sector_counts.csv", index=False
    )
    metadata = {
        "panel_unit": "Cambodia national 1 km grid cell",
        "grid_cells": int(len(output)),
        "distance_crs": "EPSG:32648",
        "distance_rule": "Euclidean distance from grid-cell centre to the Cambodia-source boundary segments lying within 2 km of the Thailand-source boundary",
        "primary_match_tolerance_m": PRIMARY_MATCH_TOLERANCE_M,
        "sector_rule": "0.5-degree longitude bin of the nearest shared-border point",
        "cambodia_source": CAMBODIA.relative_to(ROOT).as_posix(),
        "thailand_source": THAILAND.relative_to(ROOT).as_posix(),
        "thailand_archive_sha256": sha256(THAILAND_ARCHIVE),
        "boundary_caution": "shared-border geometry is a reproducible analytical reference and does not express a position on disputed delimitation",
        "outcome_blind": True,
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "border_output": BORDER_OUTPUT.relative_to(ROOT).as_posix(),
    }
    (AUDIT_DIR / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
