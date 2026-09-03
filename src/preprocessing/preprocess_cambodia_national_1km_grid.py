#!/usr/bin/env python3
"""Build a stable nationwide 1 km analysis grid for reusable satellite panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer


STEP_M = 1_000
CRS = "EPSG:32648"
BOUNDARY = Path("data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson")
OUTPUT = Path("data/processed/cambodia_national_1km_grid_preprocessed.parquet")
AUDIT_DIR = Path("data/exp/data-preprocessing/national-satellite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    communes = gpd.read_file(root / BOUNDARY).to_crs(CRS)
    national = communes.geometry.union_all()
    min_x, min_y, max_x, max_y = national.bounds
    first_col = int(np.floor(min_x / STEP_M))
    last_col = int(np.ceil(max_x / STEP_M))
    first_row = int(np.floor(min_y / STEP_M))
    last_row = int(np.ceil(max_y / STEP_M))
    columns, rows = np.meshgrid(
        np.arange(first_col, last_col, dtype=int),
        np.arange(first_row, last_row, dtype=int),
    )
    columns = columns.ravel()
    rows = rows.ravel()
    eastings = (columns + 0.5) * STEP_M
    northings = (rows + 0.5) * STEP_M
    candidates = gpd.GeoDataFrame(
        {"Grid Column": columns, "Grid Row": rows},
        geometry=gpd.points_from_xy(eastings, northings, crs=CRS),
    )
    inside = candidates.loc[candidates.geometry.within(national)].copy()
    administration = communes[
        [
            "ADM1_PCODE",
            "ADM1_EN",
            "ADM2_PCODE",
            "ADM2_EN",
            "ADM3_PCODE",
            "ADM3_EN",
            "geometry",
        ]
    ].copy()
    linked = gpd.sjoin(inside, administration, how="left", predicate="within")
    linked = linked.drop(columns=["index_right"])
    if linked.duplicated(["Grid Row", "Grid Column"]).any():
        raise RuntimeError("A national grid centre linked to multiple communes")
    if linked["ADM3_PCODE"].isna().any():
        raise RuntimeError("At least one in-country grid centre lacks an administrative link")

    transformer = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(
        linked.geometry.x.to_numpy(), linked.geometry.y.to_numpy()
    )
    output = pd.DataFrame(
        {
            "National Grid Cell ID": [
                f"khm1km_e{column:04d}_n{row:04d}"
                for column, row in zip(linked["Grid Column"], linked["Grid Row"], strict=True)
            ],
            "Grid Row": linked["Grid Row"].astype("int32"),
            "Grid Column": linked["Grid Column"].astype("int32"),
            "Grid Centre Easting m": linked.geometry.x.astype(float),
            "Grid Centre Northing m": linked.geometry.y.astype(float),
            "Grid West m": linked["Grid Column"].astype(float) * STEP_M,
            "Grid South m": linked["Grid Row"].astype(float) * STEP_M,
            "Grid East m": (linked["Grid Column"].astype(float) + 1) * STEP_M,
            "Grid North m": (linked["Grid Row"].astype(float) + 1) * STEP_M,
            "Longitude": longitude,
            "Latitude": latitude,
            "Province Code": linked["ADM1_PCODE"].astype("string"),
            "Province Name": linked["ADM1_EN"].astype("string"),
            "District Code": linked["ADM2_PCODE"].astype("string"),
            "District Name": linked["ADM2_EN"].astype("string"),
            "Commune Code": linked["ADM3_PCODE"].astype("string"),
            "Commune Name": linked["ADM3_EN"].astype("string"),
        }
    ).sort_values(["Grid Row", "Grid Column"]).reset_index(drop=True)
    if output["National Grid Cell ID"].duplicated().any():
        raise RuntimeError("National grid identifiers are not unique")
    output_path = root / OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False)

    audit = root / AUDIT_DIR
    audit.mkdir(parents=True, exist_ok=True)
    metadata = {
        "grid_crs": CRS,
        "grid_resolution_metres": STEP_M,
        "inclusion_rule": "1 km cell centre falls inside the Cambodia commune union",
        "administrative_link_rule": "cell-centre point-in-polygon using the frozen commune layer",
        "cells": len(output),
        "provinces": int(output["Province Code"].nunique()),
        "districts": int(output["District Code"].nunique()),
        "communes": int(output["Commune Code"].nunique()),
        "longitude_range": [float(output["Longitude"].min()), float(output["Longitude"].max())],
        "latitude_range": [float(output["Latitude"].min()), float(output["Latitude"].max())],
        "raw_native_raster_policy": "Retain nationwide native-resolution source clips separately; the 1 km grid is an analysis index, not a destructive resample.",
    }
    (audit / "national_1km_grid_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    coverage = (
        output.groupby(["Province Code", "Province Name"], observed=True)
        .agg(Grid_Cells=("National Grid Cell ID", "size"), Districts=("District Code", "nunique"), Communes=("Commune Code", "nunique"))
        .reset_index()
    )
    coverage.to_csv(audit / "national_1km_grid_coverage_by_province.csv", index=False)
    print(f"Saved: {OUTPUT}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
