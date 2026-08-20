#!/usr/bin/env python3
"""Build a village-year land-productivity panel around the historical boundary.

This script does not estimate a treatment effect.  It constructs a quality-audited
annual outcome panel from checksum-pinned MODIS NPP clips and the pre-existing public
historical-village frame.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

# Avoid an incompatible user-level Anaconda PROJ/GDAL database leaking into uv.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy


NPP_SCALE = 0.0001
NPP_VALID_MIN = -30000
NPP_VALID_MAX = 32700
QC_VALID_MIN = 0
QC_VALID_MAX = 100
BUFFER_RADIUS_M = 1000.0
BASELINE_START = 2001
BASELINE_END = 2020
BANDWIDTHS_KM = (2, 5, 10, 15, 20, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/exp/data-preprocessing/annual-spatial-source/modis-npp"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/historical_boundary_annual_land_productivity_preprocessed.parquet"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/exp/data-preprocessing/annual-spatial-panel"),
    )
    return parser.parse_args()


def normalized_code(values: pd.Series, width: int) -> pd.Series:
    return (
        values.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(width)
    )


def load_zones(source: Path) -> tuple[object, object, object]:
    archive_path = source / "Democratic_Kampuchea_Zones.zip"
    with tempfile.TemporaryDirectory(prefix="mj02-zone-npp-") as temp_dir:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_dir)
        shapefiles = list(Path(temp_dir).rglob("*.shp"))
        if len(shapefiles) != 1:
            raise RuntimeError(f"Expected one zone shapefile, found {len(shapefiles)}")
        zones = gpd.read_file(shapefiles[0]).to_crs(32648)
    southwest = zones.loc[zones["ZONE_NAME"].eq("Southwest"), "geometry"].union_all()
    west = zones.loc[zones["ZONE_NAME"].eq("West"), "geometry"].union_all()
    boundary = southwest.boundary.intersection(west.boundary)
    return southwest, west, boundary


def construct_village_frame(root: Path) -> gpd.GeoDataFrame:
    boundary_source = root / "data/exp/data-preprocessing/historical-boundary-source"
    southwest, west, boundary = load_zones(boundary_source)
    segment_table = pd.read_csv(boundary_source / "boundary_segment_points.csv")
    segment_points = gpd.GeoSeries.from_xy(
        segment_table["Easting EPSG 32648"],
        segment_table["Northing EPSG 32648"],
        crs=32648,
    )

    villages = gpd.read_file(
        root / "data/raw/conflict/yale_cgeo_historical_villages.geojson"
    )
    villages["Village Code"] = normalized_code(villages["CODEPHUM"], 8)
    villages = villages.loc[villages["Village Code"].str.startswith("05")].copy()
    villages["Longitude"] = villages.geometry.x
    villages["Latitude"] = villages.geometry.y
    villages = villages.to_crs(32648)

    def side(point: object) -> str:
        if point.within(southwest):
            return "Southwest"
        if point.within(west):
            return "West"
        return "Outside"

    villages["Historical Repression Side"] = villages.geometry.map(side)
    villages["Higher-Repression Southwest Zone"] = (
        villages["Historical Repression Side"].eq("Southwest").astype("Int8")
    )
    villages["Absolute Distance to Historical Repression Boundary km"] = (
        villages.geometry.distance(boundary) / 1000
    )
    villages["Signed Distance to Historical Repression Boundary km"] = villages[
        "Absolute Distance to Historical Repression Boundary km"
    ].where(
        villages["Historical Repression Side"].eq("Southwest"),
        -villages["Absolute Distance to Historical Repression Boundary km"],
    )
    villages["Historical Boundary Segment"] = villages.geometry.map(
        lambda point: int(segment_points.distance(point).to_numpy().argmin()) + 1
    ).astype("Int8")
    for bandwidth in BANDWIDTHS_KM:
        villages[f"Historical-Boundary Common Support {bandwidth} km"] = (
            villages["Absolute Distance to Historical Repression Boundary km"].le(bandwidth)
        ).astype("Int8")
    return villages[
        [
            "Village Code",
            "PHUM",
            "Longitude",
            "Latitude",
            "Historical Repression Side",
            "Higher-Repression Southwest Zone",
            "Signed Distance to Historical Repression Boundary km",
            "Absolute Distance to Historical Repression Boundary km",
            "Historical Boundary Segment",
            *[f"Historical-Boundary Common Support {value} km" for value in BANDWIDTHS_KM],
            "geometry",
        ]
    ].rename(columns={"PHUM": "Village Name"})


def pixel_centres(
    transform: object, rows: np.ndarray, cols: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = xy(transform, rows, cols, offset="center")
    return np.asarray(x_values), np.asarray(y_values)


def extract_buffer_values(
    values: np.ndarray,
    quality: np.ndarray,
    transform: object,
    point: object,
) -> dict[str, float | int]:
    inverse = ~transform
    col_float, row_float = inverse * (point.x, point.y)
    centre_col = int(np.floor(col_float))
    centre_row = int(np.floor(row_float))
    pixel_size = max(abs(transform.a), abs(transform.e))
    search = int(np.ceil(BUFFER_RADIUS_M / pixel_size)) + 1
    row_min = max(0, centre_row - search)
    row_max = min(values.shape[0] - 1, centre_row + search)
    col_min = max(0, centre_col - search)
    col_max = min(values.shape[1] - 1, centre_col + search)
    rows, cols = np.meshgrid(
        np.arange(row_min, row_max + 1),
        np.arange(col_min, col_max + 1),
        indexing="ij",
    )
    x_values, y_values = pixel_centres(transform, rows.ravel(), cols.ravel())
    inside = np.hypot(x_values - point.x, y_values - point.y) <= BUFFER_RADIUS_M
    selected_values = values[rows.ravel()[inside], cols.ravel()[inside]]
    selected_quality = quality[rows.ravel()[inside], cols.ravel()[inside]]
    valid_npp = (selected_values >= NPP_VALID_MIN) & (selected_values <= NPP_VALID_MAX)
    valid_qc = (selected_quality >= QC_VALID_MIN) & (selected_quality <= QC_VALID_MAX)
    paired_qc = valid_npp & valid_qc
    npp = selected_values[valid_npp].astype(float) * NPP_SCALE
    qc = selected_quality[paired_qc].astype(float)
    return {
        "NPP Candidate Pixel Count": int(inside.sum()),
        "NPP Valid Pixel Count": int(valid_npp.sum()),
        "Annual Land NPP Mean kg C per m2": float(np.mean(npp)) if len(npp) else np.nan,
        "Annual Land NPP Median kg C per m2": float(np.median(npp)) if len(npp) else np.nan,
        "Annual Land NPP Pixel SD kg C per m2": (
            float(np.std(npp, ddof=1)) if len(npp) > 1 else np.nan
        ),
        "Mean NPP QC Filled Growing-Season Days Percent": (
            float(np.mean(qc)) if len(qc) else np.nan
        ),
        "Maximum NPP QC Filled Growing-Season Days Percent": (
            float(np.max(qc)) if len(qc) else np.nan
        ),
        "NPP Pixels with No More Than 50 Percent Filled Days Share": (
            float(np.mean(qc <= 50)) if len(qc) else np.nan
        ),
    }


def build_panel(villages: gpd.GeoDataFrame, source: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    clip_files = sorted((source / "clips").glob("*_h??v??_Npp_500m.tif"))
    years = sorted({int(path.name[:4]) for path in clip_files})
    for year in years:
        npp_path = source / "clips" / f"{year}_h28v07_Npp_500m.tif"
        qc_path = source / "clips" / f"{year}_h28v07_Npp_QC_500m.tif"
        if not npp_path.exists() or not qc_path.exists():
            raise FileNotFoundError(f"Incomplete NPP/QC pair for {year}")
        with rasterio.open(npp_path) as npp_source, rasterio.open(qc_path) as qc_source:
            if (
                npp_source.shape != qc_source.shape
                or npp_source.transform != qc_source.transform
                or npp_source.crs != qc_source.crs
            ):
                raise RuntimeError(f"NPP/QC grids do not align for {year}")
            values = npp_source.read(1)
            quality = qc_source.read(1)
            projected = villages.to_crs(npp_source.crs)
            for village, point in zip(
                villages.drop(columns="geometry").to_dict("records"),
                projected.geometry,
                strict=True,
            ):
                extracted = extract_buffer_values(
                    values, quality, npp_source.transform, point
                )
                rows.append({**village, "Year": year, **extracted})
    panel = pd.DataFrame(rows)
    baseline = panel["Year"].between(BASELINE_START, BASELINE_END)
    baseline_stats = (
        panel.loc[baseline]
        .groupby("Village Code")["Annual Land NPP Mean kg C per m2"]
        .agg(
            **{
                "Village 2001-2020 Mean Annual Land NPP kg C per m2": "mean",
                "Village 2001-2020 SD Annual Land NPP kg C per m2": "std",
                "Village 2001-2020 Valid NPP Year Count": "count",
            }
        )
        .reset_index()
    )
    panel = panel.merge(baseline_stats, on="Village Code", how="left", validate="many_to_one")
    panel["Annual Land NPP Anomaly kg C per m2"] = (
        panel["Annual Land NPP Mean kg C per m2"]
        - panel["Village 2001-2020 Mean Annual Land NPP kg C per m2"]
    )
    panel["Annual Land NPP Anomaly Z 2001-2020"] = panel[
        "Annual Land NPP Anomaly kg C per m2"
    ] / panel["Village 2001-2020 SD Annual Land NPP kg C per m2"]
    panel["NPP Complete 2001-2020 Baseline"] = (
        panel["Village 2001-2020 Valid NPP Year Count"].eq(20).astype("Int8")
    )
    return panel.sort_values(["Village Code", "Year"]).reset_index(drop=True)


def write_audit(panel: pd.DataFrame, output: Path, source: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    coverage = (
        panel.groupby(["Year", "Historical Repression Side"], observed=True)
        .agg(
            Villages=("Village Code", "size"),
            Villages_with_NPP=("Annual Land NPP Mean kg C per m2", "count"),
            Mean_valid_pixels=("NPP Valid Pixel Count", "mean"),
            Mean_filled_days_percent=(
                "Mean NPP QC Filled Growing-Season Days Percent",
                "mean",
            ),
            P95_filled_days_percent=(
                "Mean NPP QC Filled Growing-Season Days Percent",
                lambda values: values.quantile(0.95),
            ),
        )
        .reset_index()
    )
    coverage["NPP_coverage_share"] = (
        coverage["Villages_with_NPP"] / coverage["Villages"]
    )
    coverage.to_csv(output / "modis_npp_coverage_by_year_and_side.csv", index=False)

    support_rows = []
    village_year = panel.drop_duplicates(["Village Code", "Year"])
    for bandwidth in BANDWIDTHS_KM:
        subset = village_year.loc[
            village_year[f"Historical-Boundary Common Support {bandwidth} km"].eq(1)
        ]
        for side, group in subset.groupby("Historical Repression Side"):
            support_rows.append(
                {
                    "bandwidth_km": bandwidth,
                    "side": side,
                    "unique_villages": group["Village Code"].nunique(),
                    "village_years": len(group),
                    "npp_coverage_share": group[
                        "Annual Land NPP Mean kg C per m2"
                    ].notna().mean(),
                    "mean_filled_days_percent": group[
                        "Mean NPP QC Filled Growing-Season Days Percent"
                    ].mean(),
                }
            )
    pd.DataFrame(support_rows).to_csv(
        output / "modis_npp_boundary_support_and_quality.csv", index=False
    )

    manifest = pd.read_csv(source / "source_manifest.csv")
    checks = []
    for row in manifest.itertuples(index=False):
        path = Path(row.local_path)
        if not path.is_absolute():
            path = source.parents[4] / path
        checks.append(path.exists())
    summary = {
        "panel_rows": len(panel),
        "unique_villages": panel["Village Code"].nunique(),
        "years": [int(panel["Year"].min()), int(panel["Year"].max())],
        "years_count": int(panel["Year"].nunique()),
        "source_manifest_rows": len(manifest),
        "source_files_present": bool(all(checks)),
        "npp_coverage_share": float(panel["Annual Land NPP Mean kg C per m2"].notna().mean()),
        "complete_baseline_village_share": float(
            panel.drop_duplicates("Village Code")["NPP Complete 2001-2020 Baseline"].mean()
        ),
        "aggregation": "unweighted mean of valid 500 m pixel centres within 1,000 m of each public historical village point",
        "npp_scale": NPP_SCALE,
        "quality_interpretation": "Npp_QC is the percentage of growing-season days whose FPAR/LAI input was filled; lower is better. It is retained, not used to tune an effect estimate.",
        "baseline": "village-specific 2001-2020 mean and standard deviation",
        "cropland_claim_allowed": False,
        "reason": "No frozen cropland mask is applied; this is land vegetation NPP, not crop yield or cropland-only productivity.",
        "effect_estimation_performed": False,
    }
    (output / "README_modis_npp_panel.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source = args.source if args.source.is_absolute() else root / args.source
    output = args.output if args.output.is_absolute() else root / args.output
    audit_output = (
        args.audit_output if args.audit_output.is_absolute() else root / args.audit_output
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    villages = construct_village_frame(root)
    panel = build_panel(villages, source)
    panel.to_parquet(output, index=False)
    write_audit(panel, audit_output, source)
    print(f"wrote {len(panel):,} village-years to {output}")
    print(
        f"villages={panel['Village Code'].nunique():,}; years={panel['Year'].nunique()}; "
        f"NPP coverage={panel['Annual Land NPP Mean kg C per m2'].notna().mean():.3%}"
    )


if __name__ == "__main__":
    main()
