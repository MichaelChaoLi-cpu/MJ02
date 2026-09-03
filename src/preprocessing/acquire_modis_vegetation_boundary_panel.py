#!/usr/bin/env python3
"""Acquire checksum-pinned MODIS 16-day vegetation clips for the boundary study.

The script uses the same public Planetary Computer STAC and signed-COG workflow as
the existing annual NPP acquisition. Temporary SAS credentials are never persisted.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from acquire_modis_npp_boundary_panel import clip_asset, request_json


STAC_ROOT = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "modis-13Q1-061"
COLLECTION_DOI = "10.5067/MODIS/MOD13Q1.061"
ASSETS = (
    "250m_16_days_EVI",
    "250m_16_days_NDVI",
    "250m_16_days_pixel_reliability",
)
DEFAULT_BBOX = (103.85, 11.06, 104.81, 11.94)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/exp/data-preprocessing/dynamic-vegetation-source/modis-13Q1"),
    )
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=DEFAULT_BBOX,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
    )
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--tile-in-filename",
        action="store_true",
        help="Include the MODIS tile in clip names; required for multi-tile extents.",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Replace existing generated clips for the requested year range.",
    )
    return parser.parse_args()


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
            "limit": 1000,
        },
        retries=retries,
    )
    features = [
        item for item in result.get("features", [])
        if str(item.get("id", "")).startswith("MOD13Q1")
        and item_date(item).startswith(f"{year}-")
    ]
    if not features:
        raise RuntimeError(f"No MOD13Q1 items found for {year}")
    # Planetary Computer may retain superseded production versions for the
    # same tile and composite date.  Freeze the lexicographically latest item
    # ID, whose suffix is the MODIS production timestamp.
    by_date_tile: dict[tuple[str, int, int], dict[str, object]] = {}
    for item in features:
        properties = item["properties"]
        key = (
            item_date(item),
            int(properties["modis:horizontal-tile"]),
            int(properties["modis:vertical-tile"]),
        )
        if key not in by_date_tile or str(item["id"]) > str(by_date_tile[key]["id"]):
            by_date_tile[key] = item
    selected = [by_date_tile[key] for key in sorted(by_date_tile)]
    return selected


def item_date(item: dict[str, object]) -> str:
    properties = item["properties"]
    stamp = properties.get("datetime") or properties.get("start_datetime")
    return str(stamp)[:10]


def acquire_one(
    item: dict[str, object],
    asset: str,
    destination: Path,
    bbox: tuple[float, float, float, float],
    retries: int,
    refresh_existing: bool,
) -> dict[str, object] | None:
    if refresh_existing and destination.exists():
        destination.unlink()
    session = requests.Session()
    session.headers.update({"User-Agent": "MJ02-dynamic-vegetation-preprocessing/1.0"})
    return clip_asset(session, item, asset, bbox, destination, retries)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    clips_dir = output_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox = tuple(args.bbox)
    session = requests.Session()
    session.headers.update({"User-Agent": "MJ02-dynamic-vegetation-preprocessing/1.0"})

    metadata = request_json(
        session, f"{STAC_ROOT}/collections/{COLLECTION}", retries=args.retries
    )
    (output_dir / "collection_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest_rows: list[dict[str, object]] = []
    item_rows: list[dict[str, object]] = []
    for year in range(args.start_year, args.end_year + 1):
        items = query_year(session, year, bbox, args.retries)
        tile_count = len(
            {
                (
                    item["properties"].get("modis:horizontal-tile"),
                    item["properties"].get("modis:vertical-tile"),
                )
                for item in items
            }
        )
        if tile_count > 1 and not args.tile_in_filename:
            raise ValueError("Multi-tile extent requires --tile-in-filename")
        jobs = []
        for item in items:
            date = item_date(item)
            item_rows.append({
                "collection": COLLECTION,
                "collection_doi": COLLECTION_DOI,
                "item_id": item["id"],
                "composite_date": date,
                "horizontal_tile": item["properties"].get("modis:horizontal-tile"),
                "vertical_tile": item["properties"].get("modis:vertical-tile"),
                "queried_bbox_wgs84": ",".join(map(str, bbox)),
            })
            tile = (
                f"h{int(item['properties']['modis:horizontal-tile']):02d}"
                f"v{int(item['properties']['modis:vertical-tile']):02d}"
            )
            for asset in ASSETS:
                filename = (
                    f"{date}_{tile}_{asset}.tif"
                    if args.tile_in_filename
                    else f"{date}_{asset}.tif"
                )
                jobs.append((item, asset, clips_dir / filename))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    acquire_one, item, asset, destination, bbox, args.retries,
                    args.refresh_existing,
                ):
                (item, asset, destination)
                for item, asset, destination in jobs
            }
            for future in as_completed(future_map):
                item, asset, destination = future_map[future]
                date = item_date(item)
                record = future.result()
                if record is None:
                    continue
                record.update({
                    "collection": COLLECTION,
                    "collection_doi": COLLECTION_DOI,
                    "composite_date": date,
                    "queried_bbox_wgs84": ",".join(map(str, bbox)),
                    "acquired_at_utc": datetime.now(UTC).isoformat(),
                })
                record["local_path"] = destination.relative_to(root).as_posix()
                manifest_rows.append(record)
        print(f"Completed {year}: {len(items)} composites; {len(items) * len(ASSETS)} asset clips")

    items_table = pd.DataFrame(item_rows).drop_duplicates("item_id").sort_values("composite_date")
    manifest = pd.DataFrame(manifest_rows).sort_values(["composite_date", "asset"])
    items_table.to_csv(output_dir / "stac_items.csv", index=False)
    manifest.to_csv(output_dir / "source_manifest.csv", index=False)
    if manifest.empty:
        raise RuntimeError("No MOD13Q1 clips were written")
    expected = len(items_table) * len(ASSETS)
    if len(manifest) != expected:
        raise RuntimeError(f"Incomplete acquisition: expected {expected} clips, found {len(manifest)}")
    print(f"Saved: {(output_dir / 'source_manifest.csv').relative_to(root)}")
    print(f"Items: {len(items_table)}; clips: {len(manifest)}")


if __name__ == "__main__":
    main()
