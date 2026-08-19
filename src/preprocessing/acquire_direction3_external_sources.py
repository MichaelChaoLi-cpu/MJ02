"""Acquire public external sources for the historical-conflict resilience study.

The script downloads source-preserving snapshots only. It does not construct
analytical variables or modify the CSES archives. Yale CGEO layers are obtained
from the public Yale ArcGIS service and paginated by object ID so that the
FeatureServer record limit cannot silently truncate the bombing layer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable


CGEO_SERVICE = (
    "https://services1.arcgis.com/7uJv7I3kgh2y7Pe0/arcgis/rest/services/"
    "CGP_Vector_Data/FeatureServer"
)
CGEO_PAGE = "https://macmillan.yale.edu/gsp/geographic-database-cgeo"
CGEO_TERMS = (
    "https://macmillan.yale.edu/gsp/terms-use-cambodian-genocide-databases"
)
WFP_DATASET = "https://data.humdata.org/dataset/wfp-food-prices-for-cambodia"
WFP_PRICES = (
    "https://data.humdata.org/dataset/086162d6-7ad7-447d-83cc-1535201aa584/"
    "resource/3ff6e50c-0e28-422f-a181-d7a17e57174a/download/"
    "wfp_food_prices_khm.csv"
)
WFP_MARKETS = (
    "https://data.humdata.org/dataset/086162d6-7ad7-447d-83cc-1535201aa584/"
    "resource/a30c558a-68cc-463f-825e-eb1086947811/download/"
    "wfp_markets_khm.csv"
)
CHIRPS_PAGE = "https://www.chc.ucsb.edu/data/chirps"
CHIRPS_SUBSET = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
    "chirps20GlobalMonthlyP05.nc?"
    "precip[(1981-01-01T00:00:00Z):1:(2021-12-01T00:00:00Z)]"
    "[(10.275):1:(14.775)][(102.275):1:(107.725)]"
)

CGEO_LAYERS = {
    0: "historical_villages",
    1: "khmer_rouge_prisons",
    3: "khmer_rouge_burials",
    4: "us_bombing_sites",
}


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = ssl_context()


def request_bytes(
    url: str,
    *,
    form: dict[str, Any] | None = None,
    attempts: int = 5,
) -> bytes:
    data = None
    if form is not None:
        data = urllib.parse.urlencode(form).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": "MJ02-research-data-acquisition/1.0"},
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                request, timeout=120, context=SSL_CONTEXT
            ) as response:
                return response.read()
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(2**attempt, 20))
    raise RuntimeError("Unreachable retry state")


def request_json(url: str, *, form: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = json.loads(request_bytes(url, form=form))
    if "error" in payload:
        raise RuntimeError(f"Remote API error for {url}: {payload['error']}")
    return payload


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(path)


def download_file(url: str, path: Path) -> None:
    print(f"downloading {path}", flush=True)
    atomic_write(path, request_bytes(url))


def chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def acquire_cgeo_layer(layer_id: int, slug: str, output_dir: Path) -> Path:
    layer_url = f"{CGEO_SERVICE}/{layer_id}"
    metadata = request_json(f"{layer_url}?f=pjson")
    metadata_path = output_dir / "metadata" / f"{slug}.json"
    atomic_write(metadata_path, json.dumps(metadata, indent=2).encode("utf-8"))

    oid_field = metadata["objectIdField"]
    id_result = request_json(
        f"{layer_url}/query",
        form={"where": "1=1", "returnIdsOnly": "true", "f": "json"},
    )
    object_ids = sorted(int(value) for value in id_result["objectIds"])
    max_records = int(metadata.get("maxRecordCount", 1000))
    page_size = min(max_records, 1000)
    features: list[dict[str, Any]] = []

    pages = list(chunks(object_ids, page_size))
    print(
        f"acquiring Yale CGEO layer {layer_id} ({slug}): "
        f"{len(object_ids):,} records in {len(pages):,} pages",
        flush=True,
    )
    for index, page_ids in enumerate(pages, start=1):
        page = request_json(
            f"{layer_url}/query",
            form={
                "objectIds": ",".join(str(value) for value in page_ids),
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "geometryPrecision": "7",
                "f": "geojson",
            },
        )
        features.extend(page.get("features", []))
        if index == len(pages) or index % 10 == 0:
            print(
                f"  layer {layer_id}: page {index:,}/{len(pages):,}, "
                f"features {len(features):,}",
                flush=True,
            )

    returned_ids = [int(feature["properties"][oid_field]) for feature in features]
    if len(features) != len(object_ids):
        raise ValueError(
            f"Layer {layer_id} expected {len(object_ids)} records, got {len(features)}"
        )
    if len(set(returned_ids)) != len(object_ids) or set(returned_ids) != set(object_ids):
        raise ValueError(f"Layer {layer_id} has duplicate or missing object IDs")

    features.sort(key=lambda feature: int(feature["properties"][oid_field]))
    collection = {
        "type": "FeatureCollection",
        "name": metadata["name"],
        "source": layer_url,
        "features": features,
    }
    target = output_dir / f"yale_cgeo_{slug}.geojson"
    atomic_write(
        target,
        json.dumps(collection, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        ),
    )
    return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return sum(1 for _ in csv.reader(source)) - 1


def geojson_features(path: Path) -> int:
    return len(json.loads(path.read_text(encoding="utf-8"))["features"])


def write_manifest(root: Path, records: list[dict[str, Any]]) -> None:
    target = root / "data" / "raw" / "direction3_external_source_manifest.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "source_page",
        "source_url",
        "terms_url",
        "local_path",
        "rows_or_features",
        "bytes",
        "sha256",
    ]
    temporary = target.with_suffix(".csv.part")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--skip-chirps",
        action="store_true",
        help="Do not download the 1981-2021 CHIRPS baseline subset.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    conflict_dir = root / "data" / "raw" / "conflict"
    price_dir = root / "data" / "raw" / "food_prices"
    climate_dir = root / "data" / "raw" / "climate"
    records: list[dict[str, Any]] = []

    acquired: list[tuple[str, str, str, str, Path, int]] = []
    for layer_id, slug in CGEO_LAYERS.items():
        path = acquire_cgeo_layer(layer_id, slug, conflict_dir)
        acquired.append(
            (
                f"Yale CGEO {slug}",
                CGEO_PAGE,
                f"{CGEO_SERVICE}/{layer_id}",
                CGEO_TERMS,
                path,
                geojson_features(path),
            )
        )

    price_path = price_dir / "wfp_food_prices_khm.csv"
    market_path = price_dir / "wfp_markets_khm.csv"
    download_file(WFP_PRICES, price_path)
    download_file(WFP_MARKETS, market_path)
    acquired.extend(
        [
            (
                "WFP Cambodia food prices",
                WFP_DATASET,
                WFP_PRICES,
                WFP_DATASET,
                price_path,
                csv_rows(price_path),
            ),
            (
                "WFP Cambodia markets",
                WFP_DATASET,
                WFP_MARKETS,
                WFP_DATASET,
                market_path,
                csv_rows(market_path),
            ),
        ]
    )

    if not args.skip_chirps:
        climate_path = climate_dir / "chirps_v2_monthly_cambodia_1981_2021.nc"
        download_file(CHIRPS_SUBSET, climate_path)
        acquired.append(
            (
                "CHIRPS v2 Cambodia monthly precipitation 1981-2021",
                CHIRPS_PAGE,
                CHIRPS_SUBSET,
                CHIRPS_PAGE,
                climate_path,
                492,
            )
        )

    for dataset, page, url, terms, path, count in acquired:
        records.append(
            {
                "dataset": dataset,
                "source_page": page,
                "source_url": url,
                "terms_url": terms,
                "local_path": str(path.relative_to(root)),
                "rows_or_features": count,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_manifest(root, records)
    print(f"manifest_records={len(records):,}", flush=True)


if __name__ == "__main__":
    main()
