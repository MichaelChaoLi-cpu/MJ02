#!/usr/bin/env python3
"""Acquire pre-conflict nationwide MCD12Q1 land-cover clips for Cambodia."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.windows import from_bounds


ROOT = Path(__file__).resolve().parents[2]
BOUNDARY = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
OUTPUT_DIR = ROOT / "data/raw/land_cover/modis_mcd12q1_v061"
ZENODO_RECORD = "8367523"
ZENODO_DOI = "10.5281/zenodo.8367523"
OFFICIAL_PRODUCT_DOI = "10.5067/MODIS/MCD12Q1.061"
VERSION = "v20230818"
YEARS = tuple(range(2001, 2008))
FILENAME = (
    "lc_mcd12q1v061.t1_c_500m_s_{year}0101_{year}1231_"
    "go_epsg.4326_v20230818.tif"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_url(year: int) -> str:
    name = FILENAME.format(year=year)
    return f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{name}/content"


def national_bbox(boundary_path: Path) -> tuple[float, float, float, float]:
    boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    west, south, east, north = boundary.total_bounds
    margin = 0.05
    return west - margin, south - margin, east + margin, north + margin


def acquire_clip(
    url: str,
    destination: Path,
    bbox: tuple[float, float, float, float],
) -> dict[str, object]:
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(url) as source:
            window = from_bounds(*bbox, transform=source.transform)
            window = window.round_offsets().round_lengths()
            array = source.read(1, window=window)
            transform = source.window_transform(window)
            profile = source.profile.copy()
            profile.update(
                driver="GTiff",
                height=array.shape[0],
                width=array.shape[1],
                transform=transform,
                count=1,
                compress="deflate",
                predictor=2,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(destination, "w", **profile) as target:
                target.write(array, 1)
                target.update_tags(
                    source_url=url,
                    source_record=ZENODO_RECORD,
                    source_doi=ZENODO_DOI,
                    official_product_doi=OFFICIAL_PRODUCT_DOI,
                    band="LC_Type1",
                    classification="IGBP",
                    purpose="predetermined 2001-2007 national baseline",
                )
            return {
                "source_crs": source.crs.to_string(),
                "source_resolution_x": source.res[0],
                "source_resolution_y": source.res[1],
                "clip_width": array.shape[1],
                "clip_height": array.shape[0],
                "clip_min": int(array.min()),
                "clip_max": int(array.max()),
            }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    boundary = root / BOUNDARY.relative_to(ROOT)
    output_dir = root / OUTPUT_DIR.relative_to(ROOT)
    if not boundary.exists():
        raise FileNotFoundError(boundary)
    bbox = national_bbox(boundary)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for year in YEARS:
        destination = output_dir / FILENAME.format(year=year)
        url = source_url(year)
        if destination.exists() and not args.refresh:
            with rasterio.open(destination) as dataset:
                array = dataset.read(1)
                details = {
                    "source_crs": dataset.crs.to_string(),
                    "source_resolution_x": dataset.res[0],
                    "source_resolution_y": dataset.res[1],
                    "clip_width": dataset.width,
                    "clip_height": dataset.height,
                    "clip_min": int(array.min()),
                    "clip_max": int(array.max()),
                }
        else:
            details = acquire_clip(url, destination, bbox)
        rows.append(
            {
                "year": year,
                "source_record": ZENODO_RECORD,
                "source_doi": ZENODO_DOI,
                "official_product_doi": OFFICIAL_PRODUCT_DOI,
                "version": VERSION,
                "source_url": url,
                "band": "LC_Type1",
                "classification": "IGBP",
                "queried_bbox_wgs84": ",".join(f"{value:.8f}" for value in bbox),
                "local_path": destination.relative_to(root).as_posix(),
                "local_sha256": sha256(destination),
                "local_bytes": destination.stat().st_size,
                "acquired_at_utc": datetime.now(UTC).isoformat(),
                **details,
            }
        )
        print(f"Prepared MCD12Q1 {year}: {destination.relative_to(root)}")

    pd.DataFrame(rows).to_csv(output_dir / "source_manifest.csv", index=False)
    metadata = {
        "title": "MODIS MCD12Q1 Land Cover and Land Use Time Series Global Mosaics",
        "zenodo_record": ZENODO_RECORD,
        "zenodo_doi": ZENODO_DOI,
        "official_product_doi": OFFICIAL_PRODUCT_DOI,
        "version": VERSION,
        "years": list(YEARS),
        "band": "LC_Type1",
        "classification": "IGBP",
        "national_bbox_wgs84": list(bbox),
        "purpose": "outcome-blind pre-conflict land-cover shares for the Cambodia national 1 km grid",
        "timing_rule": "freeze 2001-2007; do not use post-conflict land cover as a control",
    }
    (output_dir / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
