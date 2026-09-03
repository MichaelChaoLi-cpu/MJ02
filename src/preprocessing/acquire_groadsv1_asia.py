#!/usr/bin/env python3
"""Download and extract the public gROADSv1 Asia shapefile archive.

NASA EarthData serves this public archive behind bearer-token authentication.
Set ``NASA_EARTHDATA_TOKEN`` to a token or to a text file containing the token.
The token is never written to logs or project files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/raw/roads/groadsv1_asia"
ARCHIVE = RAW_DIR / "groads-v1-asia-shp.zip"
EXTRACTED = RAW_DIR / "extracted"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/groadsv1-historical-roads"

SOURCE_URL = (
    "https://data.earthdata.nasa.gov/nasa-earth/human-dimensions/"
    "sedac-root/downloads/data/groads/groads-global-roads-open-access-v1/"
    "groads-v1-asia-shp.zip"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_token() -> str:
    value = os.environ.get("NASA_EARTHDATA_TOKEN", "").strip()
    if not value:
        raise RuntimeError(
            "NASA_EARTHDATA_TOKEN is not set. Create a NASA EarthData token, "
            "then export it for this one acquisition command."
        )
    # EarthData user tokens are JWTs beginning with ``eyJ``.  Test a value as
    # a file path only when it is not already recognisable as a token; this
    # also prevents an overlong JWT from reaching an OS path-length check.
    if not value.startswith("eyJ") and len(value) < 1024:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            value = candidate.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("NASA_EARTHDATA_TOKEN resolved to an empty value")
    return value


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    token = read_token()

    if not ARCHIVE.exists():
        temporary = ARCHIVE.with_suffix(".zip.part")
        request = urllib.request.Request(
            SOURCE_URL,
            headers={"Authorization": f"Bearer {token}", "User-Agent": "MJ02-research/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
        except urllib.error.HTTPError as error:
            temporary.unlink(missing_ok=True)
            if error.code in (401, 403):
                raise RuntimeError(
                    "EarthData rejected the token; refresh NASA_EARTHDATA_TOKEN and rerun"
                ) from error
            raise
        temporary.replace(ARCHIVE)

    if not zipfile.is_zipfile(ARCHIVE):
        raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {ARCHIVE}")
    with zipfile.ZipFile(ARCHIVE) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")
        archive.extractall(EXTRACTED)

    shapefiles = sorted(path.relative_to(ROOT).as_posix() for path in EXTRACTED.rglob("*.shp"))
    if not shapefiles:
        raise RuntimeError("No shapefile was found after extracting gROADSv1 Asia")
    metadata = {
        "dataset": "Global Roads Open Access Data Set, Version 1 (gROADSv1)",
        "region": "Asia",
        "format": "Shapefile",
        "source_url": SOURCE_URL,
        "authentication": "NASA EarthData bearer token; token not persisted",
        "archive": ARCHIVE.relative_to(ROOT).as_posix(),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "extracted_shapefiles": shapefiles,
        "acquired_at_utc": datetime.now(UTC).isoformat(),
    }
    (AUDIT_DIR / "acquisition_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
