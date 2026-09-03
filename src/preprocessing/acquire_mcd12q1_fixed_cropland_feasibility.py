#!/usr/bin/env python3
"""Acquire early-period MCD12Q1 LC_Type1 clips for a blinded feasibility gate.

Only the Cambodia boundary-study bounding box is read from the public Zenodo
Cloud-Optimized GeoTIFFs.  The global rasters are never downloaded.  Outputs
remain exploratory under data/exp/feasibility-check until the gate is passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.windows import from_bounds


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path("data/exp/feasibility-check/fixed-cropland/source")
BBOX = (103.85, 11.06, 104.81, 11.94)
ZENODO_RECORD = "8367523"
ZENODO_DOI = "10.5281/zenodo.8367523"
OFFICIAL_PRODUCT_DOI = "10.5067/MODIS/MCD12Q1.061"
VERSION = "v20230818"
YEARS = (2001, 2002, 2003)
FILENAME = (
    "lc_mcd12q1v061.t1_c_500m_s_{year}0101_{year}1231_"
    "go_epsg.4326_v20230818.tif"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
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


def acquire_clip(url: str, destination: Path) -> dict[str, object]:
    with rasterio.open(url) as source:
        window = from_bounds(*BBOX, transform=source.transform)
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
            blockxsize=128,
            blockysize=128,
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
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for year in YEARS:
        name = FILENAME.format(year=year)
        destination = output_dir / name
        url = source_url(year)
        if destination.exists() and not args.refresh:
            with rasterio.open(destination) as dataset:
                details = {
                    "source_crs": dataset.crs.to_string(),
                    "source_resolution_x": dataset.res[0],
                    "source_resolution_y": dataset.res[1],
                    "clip_width": dataset.width,
                    "clip_height": dataset.height,
                    "clip_min": int(dataset.read(1).min()),
                    "clip_max": int(dataset.read(1).max()),
                }
        else:
            details = acquire_clip(url, destination)
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
                "queried_bbox_wgs84": ",".join(map(str, BBOX)),
                "local_path": destination.relative_to(root).as_posix(),
                "local_sha256": sha256(destination),
                "acquired_at_utc": datetime.now(UTC).isoformat(),
                **details,
            }
        )
        print(f"Prepared {year}: {destination.relative_to(root)}")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output_dir / "source_manifest.csv", index=False)
    metadata = {
        "title": "MODIS MCD12Q1 Land Cover and Land Use Time Series Global Mosaics 2001-2022 (500 m)",
        "creator": "Rolf Simoes (OpenGeoHub)",
        "zenodo_record": ZENODO_RECORD,
        "zenodo_doi": ZENODO_DOI,
        "official_product_doi": OFFICIAL_PRODUCT_DOI,
        "version": VERSION,
        "band": "LC_Type1",
        "classification": "IGBP",
        "strict_cropland_class": 12,
        "inclusive_agricultural_classes": [12, 14],
        "mask_rule": "classified in the named class set in at least two of 2001-2003",
        "purpose": "outcome-blind support and power feasibility gate only",
    }
    (output_dir / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved manifest: {(output_dir / 'source_manifest.csv').relative_to(root)}")


if __name__ == "__main__":
    main()
