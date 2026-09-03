#!/usr/bin/env python3
"""Download Cambodia-covering Copernicus DEM GLO-30 COG tiles from AWS.

The public S3 bucket permits anonymous HTTPS access.  Only 1 x 1 degree tiles
intersecting Cambodia's documented national geometry are downloaded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from math import ceil, floor
from pathlib import Path

os.environ.pop("PROJ_LIB", None)

import geopandas as gpd
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[2]
BOUNDARY = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
RAW_DIR = ROOT / "data/raw/terrain/copernicus_dem_glo30_2021"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/copernicus-dem-glo30"

SOURCE_ROOT = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"
MAX_WORKERS = 6


def tile_name(latitude: int, longitude: int) -> str:
    latitude_code = f"N{latitude:02d}" if latitude >= 0 else f"S{abs(latitude):02d}"
    longitude_code = f"E{longitude:03d}" if longitude >= 0 else f"W{abs(longitude):03d}"
    return f"Copernicus_DSM_COG_10_{latitude_code}_00_{longitude_code}_00_DEM"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_one(name: str) -> dict[str, object]:
    target = RAW_DIR / f"{name}.tif"
    url = f"{SOURCE_ROOT}/{name}/{name}.tif"
    etag = None
    if not target.exists():
        temporary = target.with_suffix(".tif.part")
        request = urllib.request.Request(url, headers={"User-Agent": "MJ02-research/1.0"})
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
            etag = response.headers.get("ETag")
            shutil.copyfileobj(response, out, length=1024 * 1024)
        temporary.replace(target)
    if target.stat().st_size == 0:
        raise RuntimeError(f"Downloaded empty tile: {target}")
    return {
        "Tile": name,
        "Source URL": url,
        "Local File": target.relative_to(ROOT).as_posix(),
        "Bytes": target.stat().st_size,
        "SHA256": sha256(target),
        "ETag": etag,
    }


def main() -> None:
    if not BOUNDARY.exists():
        raise FileNotFoundError(BOUNDARY)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    country = gpd.read_file(BOUNDARY).to_crs("EPSG:4326").geometry.union_all()
    min_x, min_y, max_x, max_y = country.bounds
    names = []
    for latitude in range(floor(min_y), ceil(max_y)):
        for longitude in range(floor(min_x), ceil(max_x)):
            if country.intersects(box(longitude, latitude, longitude + 1, latitude + 1)):
                names.append(tile_name(latitude, longitude))
    if not names:
        raise RuntimeError("No Copernicus DEM tiles intersect Cambodia")

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, name): name for name in names}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(
                f"[{completed:02d}/{len(names):02d}] {record['Tile']} "
                f"{int(record['Bytes']) / 1024**2:.1f} MiB",
                flush=True,
            )

    records.sort(key=lambda row: str(row["Tile"]))
    import pandas as pd

    pd.DataFrame(records).to_csv(AUDIT_DIR / "source_tile_manifest.csv", index=False)
    metadata = {
        "dataset": "Copernicus DEM GLO-30 Public, 2021 release",
        "source": "Amazon Web Services Registry of Open Data",
        "source_root": SOURCE_ROOT,
        "access": "anonymous HTTPS; no AWS account",
        "selection_rule": "all 1 x 1 degree tiles intersecting Cambodia government-origin national geometry",
        "tile_count": len(records),
        "total_bytes": int(sum(int(record["Bytes"]) for record in records)),
        "boundary": BOUNDARY.relative_to(ROOT).as_posix(),
        "acquired_at_utc": datetime.now(UTC).isoformat(),
    }
    (AUDIT_DIR / "acquisition_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
