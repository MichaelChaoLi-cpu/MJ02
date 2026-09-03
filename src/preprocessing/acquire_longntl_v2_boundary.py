#!/usr/bin/env python3
"""Acquire checksum-bound LongNTL Version 2 crops for the boundary study.

The official annual archives each contain an approximately 10 GB global GeoTIFF.
This script downloads one ZIP at a time, verifies the Dataverse MD5 checksum, and
uses GDAL's virtual ZIP reader to extract only the fixed Kampong Speu boundary
extent.  The global archives are temporary and are not retained in the project.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


DOI = "doi:10.7910/DVN/YGIVCD"
DATASET_VERSION = "10.1"
DATASET_TITLE = "The global NPP-VIIRS-like nighttime light data (Version 2) for 1992-2025"
ARTICLE_DOI = "10.34133/remotesensing.0874"

# Fixed to the existing observed-VIIRS research crop, with GDAL snapping to the
# native 15-arc-second grid. Order: west, north, east, south.
BOUNDS = (103.8363, 11.9521, 104.8244, 11.0493)


@dataclass(frozen=True)
class Source:
    year: int
    file_id: int
    filesize: int
    md5: str

    @property
    def archive_name(self) -> str:
        return f"{self.year}_Version2.zip"

    @property
    def raster_name(self) -> str:
        return f"nppviirs_like_V2_{self.year}.tif"


SOURCES = (
    Source(2000, 13295261, 89367977, "573fb3dc1b22744c83292fd4eca4a84d"),
    Source(2001, 13295264, 142483368, "76d78d7102d7a5fb27ab1af6beb369a0"),
    Source(2002, 13295249, 144636206, "d9db2a4b3565564abd4a72df6377ff9d"),
    Source(2003, 13295255, 145833845, "4ffc276f22ef9842ebf3422bf54c5d08"),
    Source(2004, 13295259, 146542335, "be141cbc72966b42c5cb020cc6296fb1"),
    Source(2005, 13295251, 141574642, "d8bb27968b06740048800f8aa1bd4c48"),
    Source(2006, 13295257, 137797604, "d732f861ec6bd2c423d96d96f1adf50b"),
    Source(2007, 13295263, 143425628, "ee09de7c6b6f8ae0ee9b3c98eb4411e8"),
    Source(2008, 13295250, 139879924, "68dade2aed56cae26dcd78b11c8b06e1"),
    Source(2009, 13295254, 145688972, "cfe1a53d51e0450f067f9049a2dc7c4f"),
    Source(2010, 13295266, 157229152, "c68bcd57d2fff2e5afca2c351f037d9f"),
    Source(2011, 13295270, 185568675, "0186ea9eaf2ccd2948a1b7a4a0655b26"),
    Source(2012, 13295268, 111519930, "99ac0162dba457329fc642182e14f4b6"),
    Source(2013, 13295267, 85094407, "a902340b640b0b0d74898284cec22f63"),
    Source(2014, 13295269, 88757668, "d69d2500e13213576faadd7f35ccec7e"),
    Source(2015, 13295271, 86055161, "89cec9f708704b47e823cebf2c969d19"),
    Source(2016, 13295278, 93644586, "8960ba376d7ac08ba265220083bd1c3c"),
    Source(2017, 13295273, 110008555, "e9f8414e529aa312d6c0f3578ff43796"),
    Source(2018, 13295272, 74232427, "abb0af39ac9e7cc832c8270ce7b623c6"),
    Source(2019, 13295276, 81685527, "6aff232e07042ed438edcefea3e20d73"),
    Source(2020, 13295279, 82415922, "df9db7e3f1b8a5909744d3c0d2b8cd06"),
    Source(2021, 13295274, 85697228, "b96059bc842b07e2ad17d39d19fdb4dc"),
    Source(2022, 14085057, 92838438, "3c32b719e50bccb43d4d3bf20fd6dfdd"),
    Source(2023, 14085058, 95182757, "c106532b36c0d598926bcfca097e096c"),
    Source(2024, 14085059, 98513097, "42cd3ec905b58b097e24b5c44292e269"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/independent_validation/longntl_v2"),
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument(
        "--bounds-west-north-east-south",
        type=float,
        nargs=4,
        default=BOUNDS,
        metavar=("WEST", "NORTH", "EAST", "SOUTH"),
    )
    parser.add_argument(
        "--crop-label",
        default="kampong_speu_boundary",
        help="ASCII label included in output GeoTIFF names.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required source checksum
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(source: Source, destination: Path) -> None:
    url = f"https://dataverse.harvard.edu/api/access/datafile/{source.file_id}"
    request = urllib.request.Request(url, headers={"User-Agent": "MJ02-reproducible-acquisition/1.0"})
    downloaded = 0
    next_report = 32 * 1024 * 1024
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as stream:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                print(f"  downloaded {downloaded / 1024 / 1024:.0f} MiB", flush=True)
                next_report += 32 * 1024 * 1024
    if downloaded != source.filesize:
        raise RuntimeError(
            f"Unexpected size for {source.archive_name}: {downloaded} != {source.filesize}"
        )


def crop(
    source: Source,
    archive: Path,
    destination: Path,
    bounds: tuple[float, float, float, float],
) -> None:
    west, north, east, south = bounds
    virtual_source = f"/vsizip/{archive.resolve()}/{source.raster_name}"
    environment = os.environ.copy()
    environment.pop("PROJ_LIB", None)
    environment["PROJ_DATA"] = "/opt/homebrew/share/proj"
    environment["GDAL_DATA"] = "/opt/homebrew/share/gdal"
    command = [
        "/opt/homebrew/bin/gdal_translate",
        "-q",
        "-projwin",
        str(west),
        str(north),
        str(east),
        str(south),
        "-co",
        "COMPRESS=DEFLATE",
        "-co",
        "TILED=YES",
        virtual_source,
        str(destination),
    ]
    subprocess.run(command, check=True, env=environment)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    selected = [s for s in SOURCES if args.start_year <= s.year <= args.end_year]
    if not selected:
        raise ValueError("No source years selected")
    bounds = tuple(args.bounds_west_north_east_south)
    crop_label = str(args.crop_label).strip()
    if not crop_label or not crop_label.replace("_", "").isalnum():
        raise ValueError("--crop-label must contain only letters, numbers, and underscores")

    records = []
    with tempfile.TemporaryDirectory(prefix="mj02-longntl-v2-") as temp_directory:
        temporary = Path(temp_directory)
        for index, source in enumerate(selected, start=1):
            destination = output / f"longntl_v2_{source.year}_{crop_label}.tif"
            print(f"[{index}/{len(selected)}] {source.year}", flush=True)
            if destination.exists() and not args.force:
                print("  existing crop retained", flush=True)
            else:
                archive = temporary / source.archive_name
                download(source, archive)
                observed_md5 = md5sum(archive)
                if observed_md5 != source.md5:
                    raise RuntimeError(
                        f"Checksum mismatch for {source.archive_name}: {observed_md5}"
                    )
                crop(source, archive, destination, bounds)
                archive.unlink()
            records.append(
                {
                    **asdict(source),
                    "archive_name": source.archive_name,
                    "download_url": (
                        f"https://dataverse.harvard.edu/api/access/datafile/{source.file_id}"
                    ),
                    "crop_path": str(destination.relative_to(root)),
                    "source_stage": "reconstructed" if source.year <= 2012 else "observed composite",
                }
            )

    with (output / "source_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "dataset_title": DATASET_TITLE,
        "dataset_doi": DOI,
        "dataverse_version": DATASET_VERSION,
        "article_doi": ARTICLE_DOI,
        "years": [selected[0].year, selected[-1].year],
        "native_crs": "WGS84",
        "native_resolution": "15 arc seconds (approximately 500 m)",
        "unit": "nW cm-2 sr-1",
        "fixed_crop_bounds_west_north_east_south": bounds,
        "crop_label": crop_label,
        "transition_rule": "1992-2012 reconstructed; post-2012 annual observed composites",
        "missing_value_rule": "No imputation introduced during acquisition",
        "archive_retention": "Global archives verified and cropped one at a time, then deleted",
    }
    (output / "source_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} annual crops to {output}")


if __name__ == "__main__":
    main()
