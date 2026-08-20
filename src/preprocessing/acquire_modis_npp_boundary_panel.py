#!/usr/bin/env python3
"""Acquire checksum-pinned MODIS annual NPP clips for the historical boundary study.

The script deliberately writes remote-source clips to ``data/exp`` rather than
``data/raw``.  The upstream STAC item identifiers and unsigned asset URLs are
preserved in a manifest, while temporary SAS credentials are never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

# Avoid an incompatible user-level Anaconda PROJ/GDAL database leaking into uv.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import pandas as pd
import rasterio
import requests
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds


STAC_ROOT = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
COLLECTION = "modis-17A3HGF-061"
COLLECTION_DOI = "10.5067/MODIS/MOD17A3HGF.061"
ASSETS = ("Npp_500m", "Npp_QC_500m")
DEFAULT_BBOX = (103.85, 11.06, 104.81, 11.94)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/exp/data-preprocessing/annual-spatial-source/modis-npp"),
    )
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, object] | None = None,
    retries: int,
) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=90)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_year(
    session: requests.Session,
    year: int,
    bbox: tuple[float, float, float, float],
    retries: int,
) -> list[dict[str, object]]:
    result = request_json(
        session,
        f"{STAC_ROOT}/search",
        params={
            "collections": COLLECTION,
            "bbox": ",".join(str(value) for value in bbox),
            "datetime": f"{year}-01-01/{year}-12-31",
            "limit": 100,
        },
        retries=retries,
    )
    features = result.get("features", [])
    terra = [item for item in features if str(item.get("id", "")).startswith("MOD17A3HGF")]
    by_tile: dict[tuple[int, int], dict[str, object]] = {}
    for item in terra:
        properties = item["properties"]
        tile = (
            int(properties["modis:horizontal-tile"]),
            int(properties["modis:vertical-tile"]),
        )
        if tile not in by_tile or str(item["id"]) > str(by_tile[tile]["id"]):
            by_tile[tile] = item
    if not by_tile:
        raise RuntimeError(f"No Terra MOD17A3HGF items found for {year}")
    return [by_tile[tile] for tile in sorted(by_tile)]


def signed_href(
    session: requests.Session, href: str, retries: int
) -> tuple[str, str | None]:
    signed = request_json(
        session,
        SIGN_URL,
        params={"href": href},
        retries=retries,
    )
    return str(signed["href"]), signed.get("msft:expiry")


def clip_asset(
    session: requests.Session,
    item: dict[str, object],
    asset_name: str,
    bbox: tuple[float, float, float, float],
    destination: Path,
    retries: int,
) -> dict[str, object] | None:
    asset = item["assets"][asset_name]
    upstream_href = str(asset["href"])
    if destination.exists():
        try:
            with rasterio.open(destination) as existing:
                if existing.width <= 0 or existing.height <= 0 or existing.count != 1:
                    raise RasterioIOError("Existing clip is empty or malformed")
                return {
                    "year": int(
                        (
                            item["properties"].get("start_datetime")
                            or item["properties"].get("datetime")
                        )[:4]
                    ),
                    "collection": COLLECTION,
                    "collection_doi": COLLECTION_DOI,
                    "item_id": item["id"],
                    "horizontal_tile": item["properties"]["modis:horizontal-tile"],
                    "vertical_tile": item["properties"]["modis:vertical-tile"],
                    "platform": item["properties"].get("platform"),
                    "asset": asset_name,
                    "upstream_href": upstream_href,
                    "local_path": destination.as_posix(),
                    "local_sha256": sha256sum(destination),
                    "source_crs_wkt": existing.crs.to_wkt(),
                    "source_shape": "not re-read; resumed from verified local clip",
                    "source_dtype": existing.dtypes[0],
                    "source_nodata": existing.nodata,
                    "clip_window_col_off": pd.NA,
                    "clip_window_row_off": pd.NA,
                    "clip_width": existing.width,
                    "clip_height": existing.height,
                    "temporary_token_expiry_utc": None,
                    "resumed_existing_clip": True,
                }
        except RasterioIOError:
            destination.unlink()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            href, token_expiry = signed_href(session, upstream_href, retries)
            with rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MULTIRANGE="YES",
                GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
            ):
                with rasterio.open(href) as source:
                    native_bounds = transform_bounds(
                        "EPSG:4326", source.crs, *bbox, densify_pts=21
                    )
                    window = from_bounds(*native_bounds, transform=source.transform)
                    window = window.round_offsets().round_lengths()
                    window = window.intersection(Window(0, 0, source.width, source.height))
                    if window.width <= 0 or window.height <= 0:
                        return None
                    data = source.read(1, window=window)
                    profile = source.profile.copy()
                    profile.update(
                        driver="GTiff",
                        height=data.shape[0],
                        width=data.shape[1],
                        transform=source.window_transform(window),
                        count=1,
                        compress="deflate",
                        predictor=2,
                        tiled=True,
                        blockxsize=min(256, max(16, (data.shape[1] // 16) * 16)),
                        blockysize=min(256, max(16, (data.shape[0] // 16) * 16)),
                    )
                    source_crs = source.crs.to_wkt()
                    source_nodata = source.nodata
                    source_dtype = source.dtypes[0]
                    source_shape = f"{source.height}x{source.width}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(destination, "w", **profile) as output:
                output.write(data, 1)
            return {
                "year": int((item["properties"].get("start_datetime") or item["properties"].get("datetime"))[:4]),
                "collection": COLLECTION,
                "collection_doi": COLLECTION_DOI,
                "item_id": item["id"],
                "horizontal_tile": item["properties"]["modis:horizontal-tile"],
                "vertical_tile": item["properties"]["modis:vertical-tile"],
                "platform": item["properties"].get("platform"),
                "asset": asset_name,
                "upstream_href": upstream_href,
                "local_path": destination.as_posix(),
                "local_sha256": sha256sum(destination),
                "source_crs_wkt": source_crs,
                "source_shape": source_shape,
                "source_dtype": source_dtype,
                "source_nodata": source_nodata,
                "clip_window_col_off": int(window.col_off),
                "clip_window_row_off": int(window.row_off),
                "clip_width": int(window.width),
                "clip_height": int(window.height),
                "temporary_token_expiry_utc": token_expiry,
                "resumed_existing_clip": False,
            }
        except (RasterioIOError, requests.RequestException, RuntimeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not clip {item['id']} {asset_name}") from last_error


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    clips_dir = output_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "MJ02-research-preprocessing/1.0"})

    collection_metadata = request_json(
        session,
        f"{STAC_ROOT}/collections/{COLLECTION}",
        retries=args.retries,
    )
    (output_dir / "collection_metadata.json").write_text(
        json.dumps(collection_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    all_items: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for year in range(args.start_year, args.end_year + 1):
        items = query_year(session, year, DEFAULT_BBOX, args.retries)
        all_items.extend(items)
        for item in items:
            tile = (
                f"h{int(item['properties']['modis:horizontal-tile']):02d}"
                f"v{int(item['properties']['modis:vertical-tile']):02d}"
            )
            for asset_name in ASSETS:
                destination = clips_dir / f"{year}_{tile}_{asset_name}.tif"
                row = clip_asset(
                    session,
                    item,
                    asset_name,
                    DEFAULT_BBOX,
                    destination,
                    args.retries,
                )
                if row is not None:
                    row["local_path"] = destination.relative_to(root).as_posix()
                    row["acquired_at_utc"] = datetime.now(UTC).isoformat()
                    manifest_rows.append(row)
        print(f"acquired {year}: {len(items)} Terra tiles", flush=True)

    (output_dir / "stac_items.json").write_text(
        json.dumps({"type": "FeatureCollection", "features": all_items}, indent=2),
        encoding="utf-8",
    )
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["year", "horizontal_tile", "vertical_tile", "asset"]
    )
    manifest.to_csv(output_dir / "source_manifest.csv", index=False)
    summary = {
        "collection": COLLECTION,
        "collection_doi": COLLECTION_DOI,
        "upstream_provider": "NASA LP DAAC at USGS EROS Center",
        "cloud_host": "Microsoft Planetary Computer",
        "years": [args.start_year, args.end_year],
        "bbox_epsg4326": list(DEFAULT_BBOX),
        "assets": list(ASSETS),
        "n_files": len(manifest),
        "temporary_credentials_persisted": False,
        "manifest": "source_manifest.csv",
    }
    (output_dir / "README.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(manifest)} checksum-pinned clips to {output_dir}")


if __name__ == "__main__":
    main()
