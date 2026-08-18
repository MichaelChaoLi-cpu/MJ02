#!/usr/bin/env python3
"""Prepare exploratory mine-exposure and climate candidate tables for Direction 3.

Outputs are written under data/exp/data-preprocessing. They are deliberately
named ``candidate`` because final variable definitions require human approval.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr


def safe_token(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return token or "missing"


def prepare_mine(root: Path, output_dir: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    mine_path = root / "data/raw/mine/cambodia_mine_erw_baseline_2009_2014.zip"
    mine = gpd.read_file(f"zip://{mine_path}")
    mine["village_code_full"] = (
        mine["VilCode"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    )
    mine["commune_code_full"] = mine["village_code_full"].str[:6]
    mine["survey_date_raw"] = pd.to_datetime(mine["SurveyDate"], errors="coerce")
    mine["survey_date_suspect"] = ~mine["survey_date_raw"].dt.year.between(2009, 2014)
    mine["survey_date_candidate_corrected"] = mine["survey_date_raw"]
    suspect = mine["survey_date_suspect"] & mine["survey_date_raw"].notna()
    mine.loc[suspect, "survey_date_candidate_corrected"] = (
        mine.loc[suspect, "survey_date_raw"] + pd.DateOffset(years=100)
    )
    mine_wgs84 = mine.to_crs(4326)
    mine_wgs84["longitude"] = mine_wgs84.geometry.x
    mine_wgs84["latitude"] = mine_wgs84.geometry.y

    records = mine_wgs84.rename(columns={
        "BLS_Code": "baseline_record_id",
        "District": "district_name",
        "Province": "province_name",
        "Commune": "commune_name",
        "Village": "village_name",
        "Fear_Level": "fear_level",
        "Land_Class": "land_class",
        "Proximity": "proximity",
        "Operators": "operator",
    })[
        [
            "baseline_record_id", "village_code_full", "commune_code_full",
            "province_name", "district_name", "commune_name", "village_name",
            "fear_level", "land_class", "proximity", "operator",
            "survey_date_raw", "survey_date_suspect", "survey_date_candidate_corrected",
            "longitude", "latitude", "geometry",
        ]
    ]
    records.to_parquet(output_dir / "mine_baseline_records_candidate.parquet", index=False)

    base = records.groupby("village_code_full", dropna=False).agg(
        commune_code_full=("commune_code_full", "first"),
        province_name=("province_name", "first"),
        district_name=("district_name", "first"),
        commune_name=("commune_name", "first"),
        village_name=("village_name", "first"),
        contamination_record_count=("baseline_record_id", "size"),
        survey_date_first=("survey_date_candidate_corrected", "min"),
        survey_date_last=("survey_date_candidate_corrected", "max"),
        suspect_date_count=("survey_date_suspect", "sum"),
        contamination_longitude_mean=("longitude", "mean"),
        contamination_latitude_mean=("latitude", "mean"),
        operator_count=("operator", "nunique"),
    )
    base["log_contamination_record_count"] = np.log1p(base["contamination_record_count"])

    for source_column, prefix in [
        ("fear_level", "fear"),
        ("proximity", "proximity"),
        ("land_class", "land_class"),
    ]:
        values = records[source_column].fillna("missing").replace("(blank)", "missing")
        counts = pd.crosstab(records["village_code_full"], values)
        counts.columns = [f"{prefix}_{safe_token(column)}_count" for column in counts.columns]
        base = base.join(counts, how="left")

    operators = records.groupby("village_code_full")["operator"].agg(
        lambda values: ";".join(sorted(set(str(value) for value in values.dropna())))
    )
    base["operators"] = operators
    exposure = base.reset_index()
    exposure.to_parquet(output_dir / "mine_exposure_village_candidate.parquet", index=False)
    return records, exposure


def grid_cell_indices(geometry, latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[np.ndarray, str]:
    minx, miny, maxx, maxy = geometry.bounds
    lat_idx = np.flatnonzero((latitudes >= miny) & (latitudes <= maxy))
    lon_idx = np.flatnonzero((longitudes >= minx) & (longitudes <= maxx))
    if len(lat_idx) and len(lon_idx):
        lat_grid, lon_grid = np.meshgrid(lat_idx, lon_idx, indexing="ij")
        ys = latitudes[lat_grid.ravel()]
        xs = longitudes[lon_grid.ravel()]
        inside = shapely.contains_xy(geometry, xs, ys) | shapely.intersects_xy(geometry, xs, ys)
        if inside.any():
            flat = lat_grid.ravel()[inside] * len(longitudes) + lon_grid.ravel()[inside]
            return flat.astype(int), "grid_centers_within_polygon"

    point = geometry.representative_point()
    lat_nearest = int(np.abs(latitudes - point.y).argmin())
    lon_nearest = int(np.abs(longitudes - point.x).argmin())
    return np.array([lat_nearest * len(longitudes) + lon_nearest]), "nearest_to_representative_point"


def prepare_climate(root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundary_path = root / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
    communes = gpd.read_file(boundary_path).to_crs(4326)
    communes["commune_code_full"] = (
        pd.to_numeric(communes["com_code"], errors="coerce").astype("Int64").astype("string").str.zfill(6)
    )

    climate_path = root / "data/raw/climate/chirps_v2_monthly_cambodia_2004_2021.nc"
    with xr.open_dataset(climate_path) as dataset:
        times = pd.DatetimeIndex(dataset["time"].values)
        latitudes = dataset["latitude"].values.astype(float)
        longitudes = dataset["longitude"].values.astype(float)
        precip = dataset["precip"].values.astype(float)
    precip_flat = precip.reshape(len(times), -1)

    frames: list[pd.DataFrame] = []
    for row in communes.itertuples(index=False):
        indices, method = grid_cell_indices(row.geometry, latitudes, longitudes)
        values = np.nanmean(precip_flat[:, indices], axis=1)
        frames.append(pd.DataFrame({
            "commune_code_full": row.commune_code_full,
            "commune_name_2014": row.com_name,
            "year": times.year,
            "month": times.month,
            "rainfall_mm": values,
            "grid_cell_count": len(indices),
            "extraction_method": method,
        }))
    monthly = pd.concat(frames, ignore_index=True)

    monthly["calendar_month_mean_mm"] = monthly.groupby(
        ["commune_code_full", "month"]
    )["rainfall_mm"].transform("mean")
    monthly["calendar_month_sd_mm"] = monthly.groupby(
        ["commune_code_full", "month"]
    )["rainfall_mm"].transform("std")
    monthly["monthly_rainfall_anomaly_z"] = (
        (monthly["rainfall_mm"] - monthly["calendar_month_mean_mm"])
        / monthly["calendar_month_sd_mm"].replace(0, np.nan)
    )
    monthly.to_parquet(output_dir / "climate_commune_month_candidate.parquet", index=False)

    annual = monthly.groupby(["commune_code_full", "commune_name_2014", "year"]).agg(
        annual_rainfall_mm=("rainfall_mm", "sum"),
        observed_months=("rainfall_mm", "count"),
        grid_cell_count=("grid_cell_count", "first"),
        extraction_method=("extraction_method", "first"),
    ).reset_index()
    wet = monthly[monthly["month"].between(5, 10)].groupby(
        ["commune_code_full", "year"]
    )["rainfall_mm"].sum().rename("may_oct_rainfall_mm")
    annual = annual.join(wet, on=["commune_code_full", "year"])

    for variable in ["annual_rainfall_mm", "may_oct_rainfall_mm"]:
        group = annual.groupby("commune_code_full")[variable]
        mean = group.transform("mean")
        sd = group.transform("std").replace(0, np.nan)
        stem = variable.removesuffix("_mm")
        annual[f"{stem}_anomaly_z"] = (annual[variable] - mean) / sd
        lower = group.transform(lambda values: values.quantile(0.10))
        upper = group.transform(lambda values: values.quantile(0.90))
        annual[f"{stem}_bottom_decile"] = annual[variable] <= lower
        annual[f"{stem}_top_decile"] = annual[variable] >= upper

    annual.to_parquet(output_dir / "climate_commune_year_candidate.parquet", index=False)
    return monthly, annual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / "data/exp/data-preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)

    mine_records, exposure = prepare_mine(root, output_dir)
    monthly, annual = prepare_climate(root, output_dir)

    print(f"mine_records={len(mine_records)}")
    print(f"mine_villages={len(exposure)}")
    print(f"suspect_mine_dates={int(mine_records['survey_date_suspect'].sum())}")
    print(f"climate_commune_month_rows={len(monthly)}")
    print(f"climate_commune_year_rows={len(annual)}")
    print(f"climate_communes={monthly['commune_code_full'].nunique()}")
    print(f"nearest_point_communes={(monthly.groupby('commune_code_full')['extraction_method'].first() == 'nearest_to_representative_point').sum()}")


if __name__ == "__main__":
    main()
