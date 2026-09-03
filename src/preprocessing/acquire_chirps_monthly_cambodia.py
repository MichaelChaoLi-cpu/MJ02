#!/usr/bin/env python3
"""Acquire the fixed Cambodia CHIRPS v2 monthly subset through 2024.

The source is NOAA ERDDAP's mirror of the UCSB Climate Hazards Center CHIRPS
Version 2 monthly product.  The query preserves the exact spatial bounds and
0.05-degree grid used by the frozen 1981--2021 source while extending only the
time endpoint.  The earlier source file is retained for an overlap audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import urllib.request
from pathlib import Path

import pandas as pd
import xarray as xr


SOURCE_PAGE = "https://www.chc.ucsb.edu/data/chirps"
SOURCE_URL = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/"
    "chirps20GlobalMonthlyP05.nc?"
    "precip[(1981-01-01T00:00:00Z):1:(2024-12-01T00:00:00Z)]"
    "[(10.275):1:(14.775)][(102.275):1:(107.725)]"
)
EXPECTED_MONTHS = 44 * 12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/climate/chirps_v2_monthly_cambodia_1981_2024.nc"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/exp/data-preprocessing/chirps-1981-2024"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    audit = args.audit_output if args.audit_output.is_absolute() else root / args.audit_output
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)

    partial = output.with_suffix(output.suffix + ".partial")
    request = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": "MJ02-research-data-acquisition/1.0"}
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=300, context=context) as response:
        with partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)

    with xr.open_dataset(partial) as dataset:
        times = pd.DatetimeIndex(dataset["time"].values)
        latitudes = dataset["latitude"].values
        longitudes = dataset["longitude"].values
        shape = tuple(int(value) for value in dataset["precip"].shape)
    if (
        len(times) != EXPECTED_MONTHS
        or times.min() != pd.Timestamp("1981-01-01")
        or times.max() != pd.Timestamp("2024-12-01")
        or len(latitudes) != 91
        or len(longitudes) != 110
        or shape != (EXPECTED_MONTHS, 91, 110)
    ):
        raise RuntimeError(
            f"Unexpected CHIRPS subset: dates={times.min()}..{times.max()}, shape={shape}"
        )
    partial.replace(output)

    metadata = {
        "dataset": "CHIRPS Version 2 monthly precipitation",
        "institution": "UCSB Climate Hazards Center",
        "access_service": "NOAA ERDDAP chirps20GlobalMonthlyP05",
        "source_page": SOURCE_PAGE,
        "source_url": SOURCE_URL,
        "time_coverage": ["1981-01-01", "2024-12-01"],
        "spatial_bounds_cell_centres": {
            "latitude": [10.275, 14.775],
            "longitude": [102.275, 107.725],
        },
        "shape": list(shape),
        "local_path": str(output.relative_to(root)),
        "bytes": output.stat().st_size,
        "sha256": sha256(output),
    }
    (audit / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output} ({output.stat().st_size:,} bytes)")
    print(f"SHA256 {metadata['sha256']}")


if __name__ == "__main__":
    main()
