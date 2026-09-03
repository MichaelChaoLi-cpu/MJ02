#!/usr/bin/env python3
"""Build a village-by-16-day MODIS EVI/NDVI panel for the historical boundary.

The 250 m source pixels are quality screened and aggregated within fixed 1 km
village buffers. Historical-side coefficients are never estimated in this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coordinates
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("data/exp/data-preprocessing/dynamic-vegetation-source/modis-13Q1/source_manifest.csv")
DESIGN_PANEL = Path("data/processed/historical_boundary_annual_spatial_climate_preprocessed.parquet")
DEFAULT_OUTPUT = Path("data/processed/historical_boundary_16day_vegetation_preprocessed.parquet")
EXP_DIR = Path("data/exp/data-preprocessing/dynamic-vegetation-panel")

EVI_ASSET = "250m_16_days_EVI"
NDVI_ASSET = "250m_16_days_NDVI"
RELIABILITY_ASSET = "250m_16_days_pixel_reliability"
BUFFER_METRES = 1000.0
MIN_VALID_PIXELS = 4
MIN_VALID_SHARE = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def project_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_villages(root: Path) -> pd.DataFrame:
    columns = [
        "Village Code", "Village Name", "Longitude", "Latitude",
        "Historical Repression Side", "Higher-Repression Southwest Zone",
        "Signed Distance to Historical Repression Boundary km",
        "Absolute Distance to Historical Repression Boundary km",
        "Historical Boundary Segment", "Historical-Boundary Common Support 5 km",
        "Linked Climate Commune Code", "Linked Climate Commune Name",
    ]
    panel = pd.read_parquet(project_path(root, DESIGN_PANEL), columns=columns)
    villages = panel.loc[panel["Historical-Boundary Common Support 5 km"].eq(1)].drop_duplicates("Village Code").copy()
    if villages["Village Code"].duplicated().any():
        raise ValueError("Village design is not unique")
    return villages.sort_values("Village Code").reset_index(drop=True)


def pixel_centres(dataset: rasterio.io.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices((dataset.height, dataset.width))
    x, y = rasterio.transform.xy(dataset.transform, rows.ravel(), cols.ravel(), offset="center")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def village_pixel_map(
    villages: pd.DataFrame,
    reference: rasterio.io.DatasetReader,
) -> list[np.ndarray]:
    x, y = pixel_centres(reference)
    tree = cKDTree(np.column_stack([x, y]))
    village_x, village_y = transform_coordinates(
        "EPSG:4326",
        reference.crs,
        villages["Longitude"].to_numpy(float).tolist(),
        villages["Latitude"].to_numpy(float).tolist(),
    )
    mapping = [
        np.asarray(indices, dtype=int)
        for indices in tree.query_ball_point(np.column_stack([village_x, village_y]), r=BUFFER_METRES)
    ]
    if min(map(len, mapping)) < MIN_VALID_PIXELS:
        raise ValueError("At least one village has fewer than four candidate MODIS pixels")
    return mapping


def read_flat(path: Path, reference_profile: tuple[object, ...]) -> np.ndarray:
    with rasterio.open(path) as dataset:
        profile = (dataset.crs.to_wkt(), dataset.transform, dataset.width, dataset.height)
        if profile != reference_profile:
            raise ValueError(f"Raster grid differs from reference: {path}")
        return dataset.read(1).ravel()


def aggregate_date(
    date: pd.Timestamp,
    paths: dict[str, Path],
    villages: pd.DataFrame,
    mapping: list[np.ndarray],
    reference_profile: tuple[object, ...],
) -> list[dict[str, object]]:
    evi_raw = read_flat(paths[EVI_ASSET], reference_profile)
    ndvi_raw = read_flat(paths[NDVI_ASSET], reference_profile)
    reliability = read_flat(paths[RELIABILITY_ASSET], reference_profile)
    rows: list[dict[str, object]] = []
    for village_row, indices in zip(villages.to_dict("records"), mapping):
        rel = reliability[indices]
        quality = np.isin(rel, [0, 1])
        evi_values = evi_raw[indices].astype(float)
        ndvi_values = ndvi_raw[indices].astype(float)
        evi_valid = quality & (evi_values >= -2000) & (evi_values <= 10000)
        ndvi_valid = quality & (ndvi_values >= -2000) & (ndvi_values <= 10000)
        candidate = len(indices)
        required = MIN_VALID_PIXELS
        strict_required = max(MIN_VALID_PIXELS, int(np.ceil(MIN_VALID_SHARE * candidate)))
        evi = evi_values[evi_valid] * 0.0001
        ndvi = ndvi_values[ndvi_valid] * 0.0001
        base = dict(village_row)
        base.update({
            "Composite Date": date,
            "Year": date.year,
            "Day of Year": int(date.dayofyear),
            "MODIS Composite Slot": int((date.dayofyear - 1) // 16 + 1),
            "Candidate Pixel Count": candidate,
            "Required Valid Pixel Count": required,
            "Strict 50 Percent Valid Pixel Count": strict_required,
            "Good or Marginal Reliability Pixel Count": int(quality.sum()),
            "Marginal Reliability Pixel Share": float(np.mean(rel == 1)),
            "EVI Valid Pixel Count": int(evi_valid.sum()),
            "NDVI Valid Pixel Count": int(ndvi_valid.sum()),
            "EVI Valid Pixel Share": float(evi_valid.sum() / candidate),
            "NDVI Valid Pixel Share": float(ndvi_valid.sum() / candidate),
            "EVI Meets 50 Percent Valid Share": int(evi_valid.sum() >= strict_required),
            "NDVI Meets 50 Percent Valid Share": int(ndvi_valid.sum() >= strict_required),
            "Mean EVI": float(np.mean(evi)) if len(evi) >= required else np.nan,
            "Median EVI": float(np.median(evi)) if len(evi) >= required else np.nan,
            "Mean NDVI": float(np.mean(ndvi)) if len(ndvi) >= required else np.nan,
            "Median NDVI": float(np.median(ndvi)) if len(ndvi) >= required else np.nan,
        })
        rows.append(base)
    return rows


def add_anomalies(panel: pd.DataFrame) -> pd.DataFrame:
    reference = panel["Year"].between(2001, 2020)
    for outcome, prefix in (("Mean EVI", "EVI"), ("Mean NDVI", "NDVI")):
        baseline = (
            panel.loc[reference]
            .groupby(["Village Code", "MODIS Composite Slot"], observed=True)[outcome]
            .agg(["mean", "std", "count"])
            .rename(columns={"mean": f"{prefix} Slot Mean 2001-2020", "std": f"{prefix} Slot SD 2001-2020", "count": f"{prefix} Slot Valid Years 2001-2020"})
            .reset_index()
        )
        panel = panel.merge(baseline, on=["Village Code", "MODIS Composite Slot"], how="left", validate="many_to_one")
        panel[f"High-Frequency {prefix} Anomaly"] = panel[outcome] - panel[f"{prefix} Slot Mean 2001-2020"]
        panel[f"High-Frequency {prefix} Anomaly Z"] = panel[f"High-Frequency {prefix} Anomaly"] / panel[f"{prefix} Slot SD 2001-2020"]
        insufficient_reference = (
            panel[f"{prefix} Slot Valid Years 2001-2020"].lt(10)
            | panel[f"{prefix} Slot SD 2001-2020"].le(0)
        )
        panel.loc[insufficient_reference, [
            f"High-Frequency {prefix} Anomaly",
            f"High-Frequency {prefix} Anomaly Z",
        ]] = np.nan
    return panel


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_path = project_path(root, args.source_manifest)
    output_path = project_path(root, args.output)
    exp_dir = project_path(root, EXP_DIR)
    manifest = pd.read_csv(source_path)
    manifest["composite_date"] = pd.to_datetime(manifest["composite_date"])
    required_assets = {EVI_ASSET, NDVI_ASSET, RELIABILITY_ASSET}
    if set(manifest["asset"].unique()) != required_assets:
        raise ValueError("Source manifest does not contain exactly the frozen vegetation assets")
    manifest["absolute_path"] = manifest["local_path"].map(lambda value: root / str(value))
    if not manifest["absolute_path"].map(Path.exists).all():
        raise FileNotFoundError("At least one manifest clip is missing")

    grouped = list(manifest.groupby("composite_date", observed=True, sort=True))
    first_path = Path(grouped[0][1].iloc[0]["absolute_path"])
    with rasterio.open(first_path) as reference:
        reference_profile = (reference.crs.to_wkt(), reference.transform, reference.width, reference.height)
        villages = load_villages(root)
        mapping = village_pixel_map(villages, reference)

    rows: list[dict[str, object]] = []
    for index, (date, date_rows) in enumerate(grouped, start=1):
        paths = {row.asset: Path(row.absolute_path) for row in date_rows.itertuples(index=False)}
        if set(paths) != required_assets:
            raise ValueError(f"Incomplete asset family for {date:%Y-%m-%d}")
        rows.extend(aggregate_date(date, paths, villages, mapping, reference_profile))
        if index % 50 == 0 or index == len(grouped):
            print(f"Aggregated {index}/{len(grouped)} composites")

    panel = add_anomalies(pd.DataFrame(rows))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    exp_dir.mkdir(parents=True, exist_ok=True)
    coverage = (
        panel.groupby("Year", observed=True)
        .agg(
            composites=("Composite Date", "nunique"),
            villages=("Village Code", "nunique"),
            rows=("Village Code", "size"),
            evi_available=("Mean EVI", lambda value: int(value.notna().sum())),
            ndvi_available=("Mean NDVI", lambda value: int(value.notna().sum())),
            evi_strict_50_percent=("EVI Meets 50 Percent Valid Share", "sum"),
            ndvi_strict_50_percent=("NDVI Meets 50 Percent Valid Share", "sum"),
            median_candidate_pixels=("Candidate Pixel Count", "median"),
        )
        .reset_index()
    )
    coverage["evi_coverage_share"] = coverage["evi_available"] / coverage["rows"]
    coverage["ndvi_coverage_share"] = coverage["ndvi_available"] / coverage["rows"]
    coverage["evi_strict_50_percent_share"] = coverage["evi_strict_50_percent"] / coverage["rows"]
    coverage["ndvi_strict_50_percent_share"] = coverage["ndvi_strict_50_percent"] / coverage["rows"]
    coverage.to_csv(exp_dir / "coverage_by_year.csv", index=False)
    print(f"Saved: {output_path.relative_to(root)}")
    print(f"Rows: {len(panel):,}; villages: {panel['Village Code'].nunique():,}; composites: {panel['Composite Date'].nunique():,}")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
