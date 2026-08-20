#!/usr/bin/env python3
"""Acquire and checksum the minimal public historical-boundary replication bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
import urllib.request
from pathlib import Path


DATAVERSE_DOI = "doi:10.7910/DVN/RK5GOH"
BASE_URL = "https://dataverse.harvard.edu/api/access/datafile"
FILES = (
    (6990145, "Democratic_Kampuchea_Zones.zip", "05bf659b2ab45f85275fd6f57f72597b", 20784),
    (6990178, "main_0310.rds", "cd0e81047d839f1792bd5433885f1b85", 234113),
    (6990115, "gadm36_KHM_1_sf.rds", "0d856eb51d171edd537f7d61c2451ff5", 1031027),
    (6990179, "road_df2.rds", "4d333dd1d54e2c92d71c18682e2ac555", 38034),
    (6990132, "README.Rmd", "8478a7dce9ce1df13c1e8ab3230daa4d", 5150),
    (6990129, "grd-helper-functions.R", "7578ef2db7576e2f3e091456e749befe", 185201),
    (6990170, "packages.R", "5ebc7aeb600b947e89fde9e57760e5ca", 358),
    (6990189, "fig2-map-in.R", "4a5446837a0b0a18008ab24b4664232c", 2844),
    (6990187, "figA2-density-in.R", "08d41e4d68bf6cb58f10c16351828ded", 991),
    (6990119, "figB3-donut-in.R", "67fd2a8dbcc21dbb4e51c659d7e620bf", 3385),
    (6990141, "figB4-NR3-in.R", "fdf752f75525c4237e927d0daafbdc7e", 483),
    (6990114, "tab2-baseline-in.R", "8dce28adc75bd6737968a4ce450b229e", 2700),
    (6990131, "tabB2-latlong-in.R", "2a84f827577e622f22e7d66822f98104", 2633),
    (6990156, "tabB3-power-in.R", "ac687a9651ac0077f4478ca6e8400510", 2015),
    (6990099, "tabB5-placebo-in.R", "4038e4de8bcc84c94bf1e6fbae3593c7", 1462),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - archive checksum verification
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "MJ02-replication-audit/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".download")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            temporary.replace(destination)
            return
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for file_id, filename, expected_md5, expected_size in FILES:
        destination = output / filename
        url = f"{BASE_URL}/{file_id}"
        if not destination.exists() or md5sum(destination) != expected_md5:
            download(url, destination)
        observed_md5 = md5sum(destination)
        observed_size = destination.stat().st_size
        if observed_md5 != expected_md5 or observed_size != expected_size:
            raise RuntimeError(
                f"Checksum or size mismatch for {filename}: {observed_md5}, {observed_size}"
            )
        manifest_rows.append(
            {
                "Dataset DOI": DATAVERSE_DOI,
                "Dataset Version": "1.0",
                "Release Date": "2023-05-02",
                "License": "CC0 1.0",
                "Datafile ID": file_id,
                "Filename": filename,
                "Source URL": url,
                "Expected MD5": expected_md5,
                "Observed MD5": observed_md5,
                "Expected Bytes": expected_size,
                "Observed Bytes": observed_size,
                "Verification": "pass",
            }
        )

    manifest_path = output / "source_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"verified files: {len(manifest_rows)}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
