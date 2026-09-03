#!/usr/bin/env python3
"""Extract annual MOD17 NPP from dynamic MCD12Q1 cropland pixels in 5 km village buffers."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform as transform_coordinates
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
CSES = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
NPP_DIR = ROOT / "data/exp/data-preprocessing/national-satellite-source/modis-npp/clips"
LAND_COVER_DIR = ROOT / "data/raw/land_cover/modis_mcd12q1_v061_dynamic"
DEFAULT_OUTPUT = ROOT / "data/processed/cses_public_village_pixel_cropland_npp_annual_preprocessed.parquet"
DEFAULT_AUDIT_DIR = ROOT / "data/exp/data-preprocessing/pixel-cropland-npp"

YEARS = tuple(range(2001, 2022))
NPP_SCALE = 0.0001
NPP_MIN_RAW = -30_000
NPP_MAX_RAW = 32_700
STRICT_CLASSES = {12}
INCLUSIVE_CLASSES = {12, 14}
TILE_PATTERN = re.compile(r"_(h\d{2}v\d{2})_Npp_500m\.tif$")


def pixel_centres(dataset: rasterio.io.DatasetReader) -> np.ndarray:
    transform = dataset.transform
    columns = np.arange(dataset.width, dtype=float) + 0.5
    rows = np.arange(dataset.height, dtype=float) + 0.5
    x = transform.c + columns * transform.a
    y = transform.f + rows * transform.e
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])


def tile_from_path(path: Path) -> str:
    match = TILE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse MODIS tile from {path}")
    return match.group(1)


def npp_paths(year: int) -> dict[str, Path]:
    paths = sorted(NPP_DIR.glob(f"{year}_h??v??_Npp_500m.tif"))
    result = {tile_from_path(path): path for path in paths}
    if set(result) != {"h27v07", "h28v07"}:
        raise FileNotFoundError(f"Incomplete national NPP tiles for {year}: {sorted(result)}")
    return result


def land_cover_path(year: int) -> Path:
    matches = list(
        LAND_COVER_DIR.glob(
            f"lc_mcd12q1v061.t1_c_500m_s_{year}0101_{year}1231_go_epsg.4326_v20230818.tif"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one dynamic MCD12Q1 raster for {year}")
    return matches[0]


def village_frame() -> pd.DataFrame:
    columns = [
        "National Village Point ID",
        "Public Village Name",
        "Point Longitude",
        "Point Latitude",
    ]
    frame = pd.read_parquet(CSES, columns=columns).dropna(
        subset=["National Village Point ID", "Point Longitude", "Point Latitude"]
    )
    frame["National Village Point ID"] = frame["National Village Point ID"].astype(str)
    conflicts = (
        frame.groupby("National Village Point ID", observed=True)[
            ["Point Longitude", "Point Latitude"]
        ]
        .nunique()
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        raise RuntimeError("A national village point has conflicting coordinates")
    return (
        frame.sort_values("National Village Point ID")
        .drop_duplicates("National Village Point ID")
        .reset_index(drop=True)
    )


def build_mappings(
    villages: pd.DataFrame,
    buffer_metres: float,
) -> tuple[dict[str, list[np.ndarray]], dict[str, tuple[object, object, int, int]]]:
    mappings: dict[str, list[np.ndarray]] = {}
    profiles: dict[str, tuple[object, object, int, int]] = {}
    for tile, path in npp_paths(YEARS[0]).items():
        with rasterio.open(path) as dataset:
            centres = pixel_centres(dataset)
            tree = cKDTree(centres)
            village_x, village_y = transform_coordinates(
                "EPSG:4326",
                dataset.crs,
                villages["Point Longitude"].to_numpy(float).tolist(),
                villages["Point Latitude"].to_numpy(float).tolist(),
            )
            mappings[tile] = [
                np.asarray(indices, dtype=np.int32)
                for indices in tree.query_ball_point(
                    np.column_stack([village_x, village_y]), r=buffer_metres
                )
            ]
            profiles[tile] = (
                dataset.crs,
                dataset.transform,
                dataset.height,
                dataset.width,
            )
        print(f"Mapped {len(villages):,} villages to {tile} pixels", flush=True)
    return mappings, profiles


def projected_land_cover(
    source: rasterio.io.DatasetReader,
    profile: tuple[object, object, int, int],
) -> np.ndarray:
    crs, transform, height, width = profile
    destination = np.full((height, width), 255, dtype=np.uint8)
    reproject(
        source=rasterio.band(source, 1),
        destination=destination,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=None,
        dst_transform=transform,
        dst_crs=crs,
        dst_nodata=255,
        resampling=Resampling.nearest,
    )
    return destination.ravel()


def blank_accumulator(size: int) -> dict[str, np.ndarray]:
    return {
        "strict_candidate": np.zeros(size, dtype=np.int32),
        "strict_valid": np.zeros(size, dtype=np.int32),
        "strict_npp_sum": np.zeros(size, dtype=float),
        "strict_qc_sum": np.zeros(size, dtype=float),
        "strict_special_qc": np.zeros(size, dtype=np.int32),
        "inclusive_candidate": np.zeros(size, dtype=np.int32),
        "inclusive_valid": np.zeros(size, dtype=np.int32),
        "inclusive_npp_sum": np.zeros(size, dtype=float),
        "inclusive_qc_sum": np.zeros(size, dtype=float),
        "inclusive_special_qc": np.zeros(size, dtype=np.int32),
    }


def add_tile(
    accumulator: dict[str, np.ndarray],
    mapping: list[np.ndarray],
    classes: np.ndarray,
    npp_raw: np.ndarray,
    qc_raw: np.ndarray,
) -> None:
    npp_valid = (npp_raw >= NPP_MIN_RAW) & (npp_raw <= NPP_MAX_RAW)
    qc_special = qc_raw > 100
    qc_analysis = np.where(qc_special, 0, qc_raw).astype(float)
    for index, pixels in enumerate(mapping):
        if len(pixels) == 0:
            continue
        pixel_classes = classes[pixels]
        for prefix, codes in (
            ("strict", STRICT_CLASSES),
            ("inclusive", INCLUSIVE_CLASSES),
        ):
            candidate = np.isin(pixel_classes, list(codes))
            if not candidate.any():
                continue
            selected = pixels[candidate]
            valid = selected[npp_valid[selected]]
            accumulator[f"{prefix}_candidate"][index] += len(selected)
            accumulator[f"{prefix}_valid"][index] += len(valid)
            accumulator[f"{prefix}_special_qc"][index] += int(
                qc_special[selected].sum()
            )
            if len(valid):
                accumulator[f"{prefix}_npp_sum"][index] += float(
                    npp_raw[valid].sum(dtype=np.float64) * NPP_SCALE
                )
                accumulator[f"{prefix}_qc_sum"][index] += float(
                    qc_analysis[valid].sum(dtype=np.float64)
                )


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(numerator), np.nan, dtype=float),
        where=denominator > 0,
    )


def main(buffer_radius_km: float, output_path: Path, audit_dir: Path) -> None:
    if buffer_radius_km <= 0:
        raise ValueError("Buffer radius must be positive")
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    if not audit_dir.is_absolute():
        audit_dir = ROOT / audit_dir
    buffer_metres = buffer_radius_km * 1_000.0
    villages = village_frame()
    mappings, profiles = build_mappings(villages, buffer_metres)
    pieces: list[pd.DataFrame] = []
    for year in YEARS:
        accumulator = blank_accumulator(len(villages))
        paths = npp_paths(year)
        with rasterio.open(land_cover_path(year)) as land_cover:
            for tile, path in paths.items():
                expected = profiles[tile]
                with rasterio.open(path) as npp_dataset:
                    observed = (
                        npp_dataset.crs,
                        npp_dataset.transform,
                        npp_dataset.height,
                        npp_dataset.width,
                    )
                    if observed != expected:
                        raise RuntimeError(f"NPP tile profile changed: {path}")
                    npp_raw = npp_dataset.read(1).ravel()
                qc_path = Path(str(path).replace("_Npp_500m.tif", "_Npp_QC_500m.tif"))
                with rasterio.open(qc_path) as qc_dataset:
                    qc_raw = qc_dataset.read(1).ravel()
                classes = projected_land_cover(land_cover, expected)
                add_tile(
                    accumulator,
                    mappings[tile],
                    classes,
                    npp_raw,
                    qc_raw,
                )

        part = villages.copy()
        part["Year"] = np.int16(year)
        for prefix, label in (
            ("strict", "Strict-Cropland"),
            ("inclusive", "Inclusive-Agriculture"),
        ):
            candidate = accumulator[f"{prefix}_candidate"]
            valid = accumulator[f"{prefix}_valid"]
            part[f"{label} Candidate 500m Pixel Count"] = candidate
            part[f"{label} Valid NPP 500m Pixel Count"] = valid
            part[f"{label} Valid NPP Pixel Share"] = safe_divide(valid, candidate)
            part[f"Annual {label} Mean NPP kg C per m2"] = safe_divide(
                accumulator[f"{prefix}_npp_sum"], valid
            )
            part[f"Mean {label} Recoded NPP QC Filled Days Percent"] = safe_divide(
                accumulator[f"{prefix}_qc_sum"], valid
            )
            part[f"{label} Original NPP QC Above 100 Pixel Count"] = accumulator[
                f"{prefix}_special_qc"
            ]
        pieces.append(part)
        print(f"Extracted pixel-level cropland NPP for {year}", flush=True)

    output = pd.concat(pieces, ignore_index=True)
    keys = ["National Village Point ID", "Year"]
    if output.duplicated(keys).any():
        raise RuntimeError("Pixel-level cropland NPP output has duplicate village-year keys")
    if len(output) != len(villages) * len(YEARS):
        raise RuntimeError("Pixel-level cropland NPP output has an unexpected row count")
    if output.filter(like="Recoded NPP QC").max().max() > 100:
        raise RuntimeError("Analysis-facing recoded QC remains above 100")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False, compression="zstd")
    coverage_rows: list[dict[str, object]] = []
    for year, group in output.groupby("Year", observed=True):
        for label in ("Strict-Cropland", "Inclusive-Agriculture"):
            candidate = group[f"{label} Candidate 500m Pixel Count"]
            valid = group[f"{label} Valid NPP 500m Pixel Count"]
            value = group[f"Annual {label} Mean NPP kg C per m2"]
            coverage_rows.append(
                {
                    "Year": int(year),
                    "Definition": label,
                    "Villages": len(group),
                    "Villages With NPP": int(value.notna().sum()),
                    "Observed Share": float(value.notna().mean()),
                    "Median Candidate Pixels": float(candidate.median()),
                    "P10 Candidate Pixels": float(candidate.quantile(0.10)),
                    "Median Valid Pixels": float(valid.median()),
                    "Median Valid Pixel Share": float(
                        group[f"{label} Valid NPP Pixel Share"].median()
                    ),
                }
            )
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(audit_dir / "coverage_by_year_and_definition.csv", index=False)
    metadata = {
        "unit": "CSES-linked national public village point by year",
        "years": [YEARS[0], YEARS[-1]],
        "buffer_radius_km": buffer_radius_km,
        "npp_source": "MOD17A3HGF.061 annual NPP at native 500 m resolution",
        "land_cover_source": "MCD12Q1.061 annual LC_Type1 at 500 m",
        "strict_definition": "IGBP class 12 Croplands",
        "inclusive_definition": "IGBP classes 12 and 14",
        "spatial_rule": (
            f"equal-area mean of native 500 m pixel centres within {buffer_radius_km:g} km"
        ),
        "npp_valid_raw_range": [NPP_MIN_RAW, NPP_MAX_RAW],
        "npp_scale": NPP_SCALE,
        "qc_recode": "raw QC values above 100 are recoded to zero in the analysis-facing QC field",
        "qc_traceability": (
            "original above-100 pixel counts are retained; NPP fill values remain excluded by "
            "the independent valid-NPP range rule"
        ),
        "raw_data_modified": False,
        "output": output_path.relative_to(ROOT).as_posix(),
        "processed_at_utc": datetime.now(UTC).isoformat(),
    }
    (audit_dir / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer-radius-km", type=float, default=5.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    arguments = parser.parse_args()
    main(arguments.buffer_radius_km, arguments.output, arguments.audit_dir)
