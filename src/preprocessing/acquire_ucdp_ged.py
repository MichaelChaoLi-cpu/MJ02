#!/usr/bin/env python3
"""Acquire the versioned UCDP GED archive with resumable range requests."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[2]
URL = "https://ucdp.uu.se/downloads/ged/ged261-csv.zip"
OUTPUT = ROOT / "data/raw/UCDP/ged261-csv.zip"
PART_DIR = ROOT / "data/exp/data-acquisition/ucdp-ged-26.1/parts"
METADATA = ROOT / "data/exp/data-acquisition/ucdp-ged-26.1/acquisition_metadata.json"
PARTS = 8
MAX_WORKERS = 3
MAX_ATTEMPTS = 4
CHUNK_BYTES = 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_size() -> int:
    response = requests.head(URL, allow_redirects=True, timeout=60)
    response.raise_for_status()
    size = int(response.headers["Content-Length"])
    if size <= 0:
        raise RuntimeError("UCDP archive reported an invalid content length")
    return size


def download_part(index: int, start: int, end: int) -> tuple[int, str]:
    path = PART_DIR / f"part-{index:02d}-{start}-{end}.bin"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return index, "cached"
    headers = {"Range": f"bytes={start}-{end}", "User-Agent": "MiliFrame-research-data-acquisition/1.0"}
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with requests.get(URL, headers=headers, stream=True, timeout=(60, 180)) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"Range request {index} returned HTTP {response.status_code}, expected 206")
                with path.open("wb") as handle:
                    for block in response.iter_content(chunk_size=CHUNK_BYTES):
                        if block:
                            handle.write(block)
            break
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(5 * attempt)
    else:
        raise RuntimeError(f"Part {index} failed") from last_error
    if path.stat().st_size != expected:
        raise RuntimeError(f"Part {index} has {path.stat().st_size} bytes, expected {expected}")
    return index, "downloaded"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PART_DIR.mkdir(parents=True, exist_ok=True)
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    size = remote_size()
    step = (size + PARTS - 1) // PARTS
    ranges = [(i, i * step, min(size - 1, (i + 1) * step - 1)) for i in range(PARTS)]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(download_part, *item) for item in ranges]
        for future in as_completed(futures):
            index, status = future.result()
            print(f"UCDP part {index + 1}/{PARTS}: {status}", flush=True)

    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".assembling")
    with temporary.open("wb") as destination:
        for index, start, end in ranges:
            path = PART_DIR / f"part-{index:02d}-{start}-{end}.bin"
            with path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=CHUNK_BYTES)
    if temporary.stat().st_size != size:
        raise RuntimeError("Assembled UCDP archive has the wrong size")
    temporary.replace(OUTPUT)
    with zipfile.ZipFile(OUTPUT) as archive:
        bad_member = archive.testzip()
        members = archive.namelist()
    if bad_member is not None:
        raise RuntimeError(f"UCDP ZIP integrity test failed at {bad_member}")

    metadata = {
        "source": "UCDP Georeferenced Event Dataset Global version 26.1",
        "url": URL,
        "bytes": size,
        "sha256": sha256(OUTPUT),
        "zip_members": members,
        "range_parts": PARTS,
        "maximum_parallel_connections": MAX_WORKERS,
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
