"""Acquire public Global Flood Database event maps intersecting Cambodia.

The source website exposes event lists by country and source-preserving ZIP
archives containing a multi-band GeoTIFF, event properties, licence, and data
documentation. This script downloads only events overlapping the 2007--2018
portion of the georeferenced survey period and validates every archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import ssl
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path


SOURCE_PAGE = "https://global-flood-database.cloudtostreet.info/"
CATALOG_URL = f"{SOURCE_PAGE}collection/KHM"
ARCHIVE_BASE = "https://storage.googleapis.com/gfd_v1_4"
DATASET_DOCUMENTATION = "https://storage.googleapis.com/gfd_metadata/README_GFD.pdf"
DATASET_CATALOG = (
    "https://developers.google.com/earth-engine/datasets/catalog/"
    "GLOBAL_FLOOD_DB_MODIS_EVENTS_V1"
)
EVENT_PATTERN = re.compile(
    r"(?P<name>DFO_(?P<event_id>\d+)_From_(?P<start>\d{8})_to_(?P<end>\d{8}))$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default="2018-12-31")
    return parser.parse_args()


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = ssl_context()


def request_bytes(url: str, attempts: int = 5) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "MJ02-flood-data-acquisition/1.0"}
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=180, context=SSL_CONTEXT
            ) as response:
                return response.read()
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(2**attempt, 20))
    raise RuntimeError("Unreachable retry state")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_event(asset: str) -> dict[str, object]:
    name = asset.rsplit("/", 1)[-1]
    match = EVENT_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(f"Unrecognised event asset: {asset}")
    return {
        "asset": asset,
        "event_name": name,
        "event_id": int(match.group("event_id")),
        "start": date.fromisoformat(
            f"{match.group('start')[:4]}-{match.group('start')[4:6]}-{match.group('start')[6:]}"
        ),
        "end": date.fromisoformat(
            f"{match.group('end')[:4]}-{match.group('end')[4:6]}-{match.group('end')[6:]}"
        ),
    }


def validate_archive(path: Path, event_name: str) -> dict[str, object]:
    event_id = event_name.split("_")[1]
    properties_name = f"DFO_{event_id}_properties.json"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        expected = {
            f"{event_name}.tif",
            properties_name,
            "README_GFD.pdf",
            "LICENSE.txt",
        }
        missing = sorted(expected - set(names))
        if missing:
            raise ValueError(f"{path.name} is missing {missing}")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt archive member in {path.name}: {bad}")
        properties = json.loads(
            archive.read(properties_name).decode("utf-8")
        )
    if int(properties["id"]) != int(event_id):
        raise ValueError(f"Event ID mismatch in {path.name}")
    return properties


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output_dir = root / "data/raw/flood/global_flood_database"
    output_dir.mkdir(parents=True, exist_ok=True)

    assets = json.loads(request_bytes(CATALOG_URL))
    events = [parse_event(asset) for asset in assets]
    events = [
        event
        for event in events
        if event["end"] >= start and event["start"] <= end
    ]
    events.sort(key=lambda event: (event["start"], event["event_id"]))
    print(f"Cambodia events selected: {len(events):,}", flush=True)

    records: list[dict[str, object]] = []
    for index, event in enumerate(events, start=1):
        event_name = str(event["event_name"])
        target = output_dir / f"{event_name}.zip"
        source_url = f"{ARCHIVE_BASE}/{event_name}.zip"
        if not target.exists():
            print(
                f"[{index:02d}/{len(events):02d}] downloading {target.name}", flush=True
            )
            atomic_write(target, request_bytes(source_url))
        else:
            print(f"[{index:02d}/{len(events):02d}] reusing {target.name}", flush=True)
        properties = validate_archive(target, event_name)
        records.append(
            {
                "Event ID": event["event_id"],
                "Event Name": event_name,
                "Event Start Date": event["start"].isoformat(),
                "Event End Date": event["end"].isoformat(),
                "Primary Reported Country": properties.get("dfo_country"),
                "Intersecting Countries": properties.get("countries"),
                "Main Cause": properties.get("dfo_main_cause"),
                "Severity": properties.get("dfo_severity"),
                "Validation Type": properties.get("dfo_validation_type"),
                "Source Page": SOURCE_PAGE,
                "Dataset Catalog": DATASET_CATALOG,
                "Documentation": DATASET_DOCUMENTATION,
                "Source URL": source_url,
                "Local Path": str(target.relative_to(root)),
                "Bytes": target.stat().st_size,
                "SHA256": sha256(target),
            }
        )

    manifest = output_dir / "gfd_cambodia_event_manifest.csv"
    temporary = manifest.with_suffix(".csv.part")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(manifest)
    print(
        f"wrote {manifest}: events={len(records):,}, "
        f"bytes={sum(int(record['Bytes']) for record in records):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
