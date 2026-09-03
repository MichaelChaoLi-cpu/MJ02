#!/usr/bin/env python3
"""Construct Cambodia 1 km historical-road access covariates from gROADSv1.

The source is a heterogeneous compilation dated approximately 1980--2010.
Accordingly, the output is a historical-access proxy, not a literal 2007 road
inventory.  A sensitivity measure removes 250 m corridors around AidData lines
identified as post-2007 new-road projects.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


ROOT = Path(__file__).resolve().parents[2]
GROADS_DIR = ROOT / "data/raw/roads/groadsv1_asia/extracted"
GRID = ROOT / "data/processed/cambodia_national_1km_grid_preprocessed.parquet"
CAMBODIA = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
AIDDATA = ROOT / "data/processed/aiddata_cambodia_road_projects_preprocessed.parquet"
OUTPUT = ROOT / "data/processed/cambodia_national_historical_road_access_preprocessed.parquet"
ROADS_OUTPUT = ROOT / "data/processed/groadsv1_cambodia_clipped_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/groadsv1-historical-roads"

METRIC_CRS = "EPSG:32648"
POST_2007_CORRIDOR_M = 250


def read_cambodia_roads(cambodia_wgs84: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    pieces: list[gpd.GeoDataFrame] = []
    bounds = tuple(cambodia_wgs84.total_bounds)
    for source in sorted(GROADS_DIR.rglob("*.shp")):
        candidate = gpd.read_file(source, bbox=bounds)
        if candidate.empty:
            continue
        if candidate.crs is None:
            raise RuntimeError(f"gROADSv1 source lacks a CRS: {source}")
        candidate = candidate[candidate.geometry.geom_type.isin(["LineString", "MultiLineString"])]
        if not candidate.empty:
            candidate["gROADSv1 Source File"] = source.relative_to(ROOT).as_posix()
            pieces.append(candidate)
    if not pieces:
        raise RuntimeError("No Cambodia line features found in extracted gROADSv1 shapefiles")
    roads = gpd.GeoDataFrame(pd.concat(pieces, ignore_index=True), crs=pieces[0].crs)
    return roads.to_crs(METRIC_CRS)


def grid_road_measures(
    grid: pd.DataFrame, roads_geometry: np.ndarray, prefix: str
) -> tuple[np.ndarray, np.ndarray]:
    boxes = shapely.box(
        grid["Grid West m"].to_numpy(),
        grid["Grid South m"].to_numpy(),
        grid["Grid East m"].to_numpy(),
        grid["Grid North m"].to_numpy(),
    )
    road_tree = shapely.STRtree(roads_geometry)
    pairs = shapely.STRtree(boxes).query(roads_geometry, predicate="intersects")
    lengths_km = np.zeros(len(grid), dtype="float64")
    if pairs.shape[1]:
        intersections = shapely.intersection(roads_geometry[pairs[0]], boxes[pairs[1]])
        np.add.at(lengths_km, pairs[1], shapely.length(intersections) / 1000)

    points = shapely.points(
        grid["Grid Centre Easting m"].to_numpy(),
        grid["Grid Centre Northing m"].to_numpy(),
    )
    nearest_indices, nearest_distance = road_tree.query_nearest(
        points, return_distance=True, all_matches=False
    )
    distances_km = np.full(len(grid), np.nan, dtype="float64")
    distances_km[nearest_indices[0]] = nearest_distance / 1000
    if not np.isfinite(distances_km).all():
        raise RuntimeError(f"Missing nearest-road distances for {prefix}")
    return lengths_km, distances_km


def main() -> None:
    shapefiles = sorted(GROADS_DIR.rglob("*.shp")) if GROADS_DIR.exists() else []
    missing = [path for path in (GRID, CAMBODIA, AIDDATA) if not path.exists()]
    if missing or not shapefiles:
        raise FileNotFoundError(
            "Missing historical-road inputs. Run acquire_groadsv1_asia.py first. "
            f"Missing: {missing}; shapefiles found: {len(shapefiles)}"
        )
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    grid = pd.read_parquet(GRID)
    cambodia_wgs84 = gpd.read_file(CAMBODIA).to_crs("EPSG:4326")
    cambodia_metric = cambodia_wgs84.to_crs(METRIC_CRS)
    national_polygon = cambodia_metric.geometry.union_all()

    roads = read_cambodia_roads(cambodia_wgs84)
    roads["geometry"] = shapely.make_valid(roads.geometry.to_numpy())
    roads = roads.explode(index_parts=False, ignore_index=True)
    roads = roads[roads.geometry.geom_type.eq("LineString")].copy()
    roads["geometry"] = shapely.intersection(roads.geometry.to_numpy(), national_polygon)
    roads = roads[~roads.geometry.is_empty].copy().reset_index(drop=True)
    roads["gROADSv1 Feature ID"] = [f"groads_khm_{i + 1:07d}" for i in range(len(roads))]
    roads["Road Length km"] = (roads.length / 1000).astype("float32")
    roads.to_parquet(ROADS_OUTPUT, index=False)

    aiddata = gpd.read_parquet(AIDDATA).to_crs(METRIC_CRS)
    post_new = aiddata.loc[aiddata["Known Post 2007 New Road Project"].fillna(False)]
    if post_new.empty:
        raise RuntimeError("AidData audit contains no post-2007 new-road lines")
    exclusion = shapely.union_all(shapely.buffer(post_new.geometry.to_numpy(), POST_2007_CORRIDOR_M))
    conservative_geometry = shapely.difference(roads.geometry.to_numpy(), exclusion)
    conservative_geometry = conservative_geometry[~shapely.is_empty(conservative_geometry)]

    primary_length, primary_distance = grid_road_measures(
        grid, roads.geometry.to_numpy(), "primary gROADSv1"
    )
    conservative_length, conservative_distance = grid_road_measures(
        grid, conservative_geometry, "AidData-corridor exclusion"
    )

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
    output["Historical Road Density Proxy km per km2"] = primary_length.astype("float32")
    output["Distance to Nearest Historical Road Proxy km"] = primary_distance.astype("float32")
    output["Road Density Excluding Post 2007 AidData Corridors km per km2"] = (
        conservative_length.astype("float32")
    )
    output["Distance to Road Excluding Post 2007 AidData Corridors km"] = (
        conservative_distance.astype("float32")
    )
    if len(output) != 179_072 or output["National Grid Cell ID"].duplicated().any():
        raise RuntimeError("Historical-road output failed national grid-key validation")
    output.to_parquet(OUTPUT, index=False)

    summary = pd.DataFrame(
        {
            "Metric": [
                "gROADSv1 Cambodia features",
                "gROADSv1 Cambodia length km",
                "post-2007 AidData new-road features",
                "grid cells with primary road",
                "grid cells with conservative road",
                "median distance to primary road km",
                "p90 distance to primary road km",
            ],
            "Value": [
                len(roads),
                float(roads["Road Length km"].sum()),
                len(post_new),
                int((primary_length > 0).sum()),
                int((conservative_length > 0).sum()),
                float(np.median(primary_distance)),
                float(np.quantile(primary_distance, 0.90)),
            ],
        }
    )
    summary.to_csv(AUDIT_DIR / "historical_road_summary.csv", index=False)
    metadata = {
        "source": "gROADSv1 Asia shapefile; approximately 1980-2010 source vintages",
        "interpretation": "historical road-access proxy, not a literal 2007 inventory",
        "primary_rule": "all gROADSv1 line length intersecting each stable 1 km cell",
        "sensitivity_rule": (
            f"remove gROADSv1 portions within {POST_2007_CORRIDOR_M} m of AidData "
            "new-road projects completed after 2007"
        ),
        "distance_rule": "Euclidean distance from grid-cell centre in EPSG:32648",
        "outcome_blind": True,
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "clipped_roads_output": ROADS_OUTPUT.relative_to(ROOT).as_posix(),
    }
    (AUDIT_DIR / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
