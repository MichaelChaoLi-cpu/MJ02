#!/usr/bin/env python3
"""Repair historical CSES administrative-code linkage to CHIRPS rainfall.

The original strict commune-code linkage is preserved. This script creates a
separate enhanced layer that uses, in order: exact commune code, unique
hierarchical name matches, high-confidence fuzzy commune-name matches, and an
explicit district/province rainfall fallback. Every fallback records its method,
spatial resolution, match score, and margin; no original missing value is
overwritten.
"""

from __future__ import annotations

import argparse
import difflib
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr


WAVE_YEAR = {
    "2007": 2007,
    "2009": 2009,
    "2011-12": 2011,
    "2013": 2013,
    "2014": 2014,
    "2016": 2016,
    "2017": 2017,
    "2019": 2019,
    "2021": 2021,
}


def normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(
        r"\b(province|capital|municipality|district|khan|krong|srok|commune|sangkat|khum)\b",
        " ",
        text,
    )
    return re.sub(r"[^a-z0-9]+", "", text)


def similarity(left: str, right: str) -> float:
    return difflib.SequenceMatcher(None, left, right).ratio()


def unique_lookup(frame: pd.DataFrame, keys: list[str], value: str) -> dict[tuple[str, ...], str]:
    grouped = frame.groupby(keys, dropna=False)[value].agg(lambda values: sorted(set(values)))
    return {tuple(index if isinstance(index, tuple) else (index,)): values[0] for index, values in grouped.items() if len(values) == 1}


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
            return flat.astype(int), "grid centers within polygon"
    point = geometry.representative_point()
    lat_nearest = int(np.abs(latitudes - point.y).argmin())
    lon_nearest = int(np.abs(longitudes - point.x).argmin())
    return np.array([lat_nearest * len(longitudes) + lon_nearest]), "nearest to representative point"


def extract_monthly(
    geographies: gpd.GeoDataFrame,
    code_column: str,
    name_column: str,
    output_code: str,
    output_name: str,
    times: pd.DatetimeIndex,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    precip_flat: np.ndarray,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in geographies.itertuples(index=False):
        code = str(getattr(row, code_column)).replace("KH", "")
        indices, method = grid_cell_indices(row.geometry, latitudes, longitudes)
        values = np.nanmean(precip_flat[:, indices], axis=1)
        frames.append(
            pd.DataFrame(
                {
                    output_code: code,
                    output_name: getattr(row, name_column),
                    "Year": times.year,
                    "Month": times.month,
                    "Rainfall mm": values,
                    "Climate Grid Cell Count": len(indices),
                    "Climate Extraction Method": method,
                }
            )
        )
    monthly = pd.concat(frames, ignore_index=True)
    monthly["Calendar Month Mean Rainfall mm"] = monthly.groupby([output_code, "Month"])[
        "Rainfall mm"
    ].transform("mean")
    monthly["Calendar Month Rainfall SD mm"] = monthly.groupby([output_code, "Month"])[
        "Rainfall mm"
    ].transform("std")
    monthly["Monthly Rainfall Anomaly Z"] = (
        (monthly["Rainfall mm"] - monthly["Calendar Month Mean Rainfall mm"])
        / monthly["Calendar Month Rainfall SD mm"].replace(0, np.nan)
    )
    return monthly


def annual_from_monthly(monthly: pd.DataFrame, code_column: str, name_column: str) -> pd.DataFrame:
    annual = monthly.groupby([code_column, name_column, "Year"]).agg(
        **{
            "Annual Rainfall mm": ("Rainfall mm", "sum"),
            "Observed Climate Months": ("Rainfall mm", "count"),
            "Climate Grid Cell Count": ("Climate Grid Cell Count", "first"),
            "Climate Extraction Method": ("Climate Extraction Method", "first"),
        }
    ).reset_index()
    wet = monthly[monthly["Month"].between(5, 10)].groupby([code_column, "Year"])[
        "Rainfall mm"
    ].sum().rename("May October Rainfall mm")
    annual = annual.join(wet, on=[code_column, "Year"])
    for variable, stem in [
        ("Annual Rainfall mm", "Annual Rainfall"),
        ("May October Rainfall mm", "May October Rainfall"),
    ]:
        grouped = annual.groupby(code_column)[variable]
        mean = grouped.transform("mean")
        sd = grouped.transform("std").replace(0, np.nan)
        annual[f"{stem} Anomaly Z"] = (annual[variable] - mean) / sd
        lower = grouped.transform(lambda values: values.quantile(0.10))
        upper = grouped.transform(lambda values: values.quantile(0.90))
        annual[f"{stem} Bottom Decile"] = annual[variable] <= lower
        annual[f"{stem} Top Decile"] = annual[variable] >= upper
    return annual


def build_geography_crosswalk(cses: pd.DataFrame, boundary: gpd.GeoDataFrame) -> pd.DataFrame:
    boundary = boundary.copy()
    boundary["climate_commune_code"] = boundary["ADM3_PCODE"].str.replace("KH", "", regex=False)
    boundary["climate_district_code"] = boundary["ADM2_PCODE"].str.replace("KH", "", regex=False)
    boundary["climate_province_code"] = boundary["ADM1_PCODE"].str.replace("KH", "", regex=False)
    for target, source in [
        ("pn", "ADM1_EN"),
        ("dn", "ADM2_EN"),
        ("cn", "ADM3_EN"),
    ]:
        boundary[target] = boundary[source].map(normalize_name)

    districts = boundary[
        ["climate_district_code", "climate_province_code", "ADM1_EN", "ADM2_EN", "pn", "dn"]
    ].drop_duplicates("climate_district_code")
    provinces = boundary[
        ["climate_province_code", "ADM1_EN", "pn"]
    ].drop_duplicates("climate_province_code")

    exact_commune_lookups = [
        ("exact commune province district name", ["pn", "dn", "cn"]),
        ("exact commune province name", ["pn", "cn"]),
        ("exact commune district name", ["dn", "cn"]),
        ("unique commune name", ["cn"]),
    ]
    commune_maps = [
        (method, keys, unique_lookup(boundary, keys, "climate_commune_code"))
        for method, keys in exact_commune_lookups
    ]
    district_maps = [
        (
            "exact district province name",
            ["pn", "dn"],
            unique_lookup(districts, ["pn", "dn"], "climate_district_code"),
        ),
        (
            "unique district name",
            ["dn"],
            unique_lookup(districts, ["dn"], "climate_district_code"),
        ),
    ]
    province_map = unique_lookup(provinces, ["pn"], "climate_province_code")
    direct_codes = set(boundary["climate_commune_code"])

    output = cses.copy()
    output["Survey Year"] = output["Survey Wave"].map(WAVE_YEAR).astype("Int64")
    for target, source in [
        ("pn", "Province Name"),
        ("dn", "District Name"),
        ("cn", "Commune Name"),
    ]:
        output[target] = output[source].map(normalize_name)

    match_rows: list[dict[str, object]] = []
    for _, values in output.iterrows():
        original_code = str(values["Commune Code"])
        result: dict[str, object] = {
            "Climate Geography Resolution": pd.NA,
            "Climate Geography Code": pd.NA,
            "Climate Link Method": pd.NA,
            "Climate Match Score": np.nan,
            "Climate Match Margin": np.nan,
            "Matched Climate Province Name": pd.NA,
            "Matched Climate District Name": pd.NA,
            "Matched Climate Commune Name": pd.NA,
        }
        if original_code in direct_codes:
            result.update(
                {
                    "Climate Geography Resolution": "commune",
                    "Climate Geography Code": original_code,
                    "Climate Link Method": "exact commune code",
                    "Climate Match Score": 1.0,
                    "Climate Match Margin": 1.0,
                }
            )
        else:
            for method, keys, lookup in commune_maps:
                key = tuple(values[name] for name in keys)
                if key in lookup:
                    result.update(
                        {
                            "Climate Geography Resolution": "commune",
                            "Climate Geography Code": lookup[key],
                            "Climate Link Method": method,
                            "Climate Match Score": 1.0,
                            "Climate Match Margin": 1.0,
                        }
                    )
                    break

        if pd.isna(result["Climate Geography Code"]):
            scored: list[tuple[float, str]] = []
            for candidate in boundary.itertuples(index=False):
                score = (
                    0.65 * similarity(values["cn"], candidate.cn)
                    + 0.30 * similarity(values["dn"], candidate.dn)
                    + 0.05 * similarity(values["pn"], candidate.pn)
                )
                scored.append((score, candidate.climate_commune_code))
            scored.sort(reverse=True)
            best, second = scored[0], scored[1]
            margin = best[0] - second[0]
            if best[0] >= 0.85 and margin >= 0.08:
                result.update(
                    {
                        "Climate Geography Resolution": "commune",
                        "Climate Geography Code": best[1],
                        "Climate Link Method": "high confidence fuzzy commune name",
                        "Climate Match Score": best[0],
                        "Climate Match Margin": margin,
                    }
                )

        if pd.isna(result["Climate Geography Code"]):
            for method, keys, lookup in district_maps:
                key = tuple(values[name] for name in keys)
                if key in lookup:
                    result.update(
                        {
                            "Climate Geography Resolution": "district",
                            "Climate Geography Code": lookup[key],
                            "Climate Link Method": method,
                            "Climate Match Score": 1.0,
                            "Climate Match Margin": 1.0,
                        }
                    )
                    break

        if pd.isna(result["Climate Geography Code"]):
            scored = []
            for candidate in districts.itertuples(index=False):
                score = 0.80 * similarity(values["dn"], candidate.dn) + 0.20 * similarity(
                    values["pn"], candidate.pn
                )
                scored.append((score, candidate.climate_district_code))
            scored.sort(reverse=True)
            best, second = scored[0], scored[1]
            margin = best[0] - second[0]
            if best[0] >= 0.80 and margin >= 0.05:
                result.update(
                    {
                        "Climate Geography Resolution": "district",
                        "Climate Geography Code": best[1],
                        "Climate Link Method": "high confidence fuzzy district name",
                        "Climate Match Score": best[0],
                        "Climate Match Margin": margin,
                    }
                )

        if pd.isna(result["Climate Geography Code"]):
            province_key = (values["pn"],)
            if province_key in province_map:
                result.update(
                    {
                        "Climate Geography Resolution": "province",
                        "Climate Geography Code": province_map[province_key],
                        "Climate Link Method": "exact province name fallback",
                        "Climate Match Score": 1.0,
                        "Climate Match Margin": 1.0,
                    }
                )
            else:
                scored = sorted(
                    [
                        (similarity(values["pn"], candidate.pn), candidate.climate_province_code)
                        for candidate in provinces.itertuples(index=False)
                    ],
                    reverse=True,
                )
                best, second = scored[0], scored[1]
                result.update(
                    {
                        "Climate Geography Resolution": "province",
                        "Climate Geography Code": best[1],
                        "Climate Link Method": "fuzzy province name fallback",
                        "Climate Match Score": best[0],
                        "Climate Match Margin": best[0] - second[0],
                    }
                )

        resolution = result["Climate Geography Resolution"]
        code = result["Climate Geography Code"]
        if resolution == "commune":
            matched = boundary.loc[boundary["climate_commune_code"].eq(code)].iloc[0]
            result.update(
                {
                    "Matched Climate Province Name": matched["ADM1_EN"],
                    "Matched Climate District Name": matched["ADM2_EN"],
                    "Matched Climate Commune Name": matched["ADM3_EN"],
                }
            )
        elif resolution == "district":
            matched = districts.loc[districts["climate_district_code"].eq(code)].iloc[0]
            result.update(
                {
                    "Matched Climate Province Name": matched["ADM1_EN"],
                    "Matched Climate District Name": matched["ADM2_EN"],
                }
            )
        else:
            matched = provinces.loc[provinces["climate_province_code"].eq(code)].iloc[0]
            result["Matched Climate Province Name"] = matched["ADM1_EN"]
        match_rows.append(result)

    matches = pd.DataFrame(match_rows)
    output = pd.concat([output.drop(columns=["pn", "dn", "cn"]), matches], axis=1)
    output["Exact Commune Climate Link Matched"] = output["Climate Link Method"].eq("exact commune code")
    return output


def attach_enhanced_climate(
    crosswalk: pd.DataFrame,
    commune: pd.DataFrame,
    district: pd.DataFrame,
    province: pd.DataFrame,
) -> pd.DataFrame:
    climate_columns = [
        "Annual Rainfall mm",
        "Observed Climate Months",
        "Climate Grid Cell Count",
        "Climate Extraction Method",
        "May October Rainfall mm",
        "Annual Rainfall Anomaly Z",
        "Annual Rainfall Bottom Decile",
        "Annual Rainfall Top Decile",
        "May October Rainfall Anomaly Z",
        "May October Rainfall Bottom Decile",
        "May October Rainfall Top Decile",
    ]
    pieces: list[pd.DataFrame] = []
    for resolution, climate, code_column in [
        ("commune", commune, "Commune Code"),
        ("district", district, "District Code"),
        ("province", province, "Province Code"),
    ]:
        subset = crosswalk[crosswalk["Climate Geography Resolution"].eq(resolution)].copy()
        climate_join = climate[[code_column, "Year"] + climate_columns].rename(
            columns={code_column: "Climate Join Code", "Year": "Climate Join Year"}
        )
        joined = subset.merge(
            climate_join,
            left_on=["Climate Geography Code", "Survey Year"],
            right_on=["Climate Join Code", "Climate Join Year"],
            how="left",
            validate="many_to_one",
        ).drop(columns=["Climate Join Code", "Climate Join Year"])
        pieces.append(joined)
    output = pd.concat(pieces, ignore_index=True, sort=False)
    output["Enhanced Climate Link Matched"] = output["Annual Rainfall mm"].notna()
    return output.sort_values(["Survey Year", "PSU"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    exp = root / "data" / "exp" / "data-preprocessing"

    boundary = gpd.read_file(raw / "geography" / "cambodia_commune_boundaries_2018_2024.geojson")
    boundary = boundary.to_crs(4326)
    cses = pd.read_parquet(processed / "cses_village_crosswalk_preprocessed.parquet")
    crosswalk = build_geography_crosswalk(cses, boundary)
    crosswalk_path = processed / "direction3_historical_geography_crosswalk_preprocessed.parquet"
    crosswalk.to_parquet(crosswalk_path, index=False)

    climate_path = raw / "climate" / "chirps_v2_monthly_cambodia_2004_2021.nc"
    with xr.open_dataset(climate_path) as dataset:
        times = pd.DatetimeIndex(dataset["time"].values)
        latitudes = dataset["latitude"].values.astype(float)
        longitudes = dataset["longitude"].values.astype(float)
        precip_flat = dataset["precip"].values.astype(float).reshape(len(times), -1)

    districts = boundary.dissolve(by="ADM2_PCODE", as_index=False).sort_values("ADM2_PCODE")
    provinces = boundary.dissolve(by="ADM1_PCODE", as_index=False).sort_values("ADM1_PCODE")
    district_month = extract_monthly(
        districts,
        "ADM2_PCODE",
        "ADM2_EN",
        "District Code",
        "District Name",
        times,
        latitudes,
        longitudes,
        precip_flat,
    )
    province_month = extract_monthly(
        provinces,
        "ADM1_PCODE",
        "ADM1_EN",
        "Province Code",
        "Province Name",
        times,
        latitudes,
        longitudes,
        precip_flat,
    )
    district_year = annual_from_monthly(district_month, "District Code", "District Name")
    province_year = annual_from_monthly(province_month, "Province Code", "Province Name")

    district_month.to_parquet(processed / "climate_district_month_preprocessed.parquet", index=False)
    district_year.to_parquet(processed / "climate_district_year_preprocessed.parquet", index=False)
    province_month.to_parquet(processed / "climate_province_month_preprocessed.parquet", index=False)
    province_year.to_parquet(processed / "climate_province_year_preprocessed.parquet", index=False)

    commune_year = pd.read_parquet(processed / "climate_commune_year_preprocessed.parquet")
    enhanced = attach_enhanced_climate(crosswalk, commune_year, district_year, province_year)
    enhanced_path = processed / "direction3_psu_year_climate_enhanced_preprocessed.parquet"
    enhanced.to_parquet(enhanced_path, index=False)

    validation = enhanced.groupby(
        ["Climate Geography Resolution", "Climate Link Method"], dropna=False
    ).agg(
        **{
            "PSU Wave Rows": ("PSU", "size"),
            "Enhanced Climate Matched Rows": ("Enhanced Climate Link Matched", "sum"),
            "Minimum Match Score": ("Climate Match Score", "min"),
            "Minimum Match Margin": ("Climate Match Margin", "min"),
        }
    ).reset_index()
    validation.to_csv(exp / "direction3_climate_linkage_repair_validation.csv", index=False)

    print(f"historical_crosswalk_rows={len(crosswalk):,}")
    print(f"district_month_rows={len(district_month):,}")
    print(f"province_month_rows={len(province_month):,}")
    print(f"enhanced_climate_rows={len(enhanced):,}")
    print(f"enhanced_climate_matches={int(enhanced['Enhanced Climate Link Matched'].sum()):,}")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
