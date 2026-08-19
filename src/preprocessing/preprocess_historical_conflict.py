"""Construct place-based historical conflict exposures from Yale CGEO.

The primary bombing measure is the log of unique bombing locations per 100
square kilometres. Locations are collapsed on a 10-metre projected grid to
avoid treating repeated ordnance records at the same target as separate places.
Reported bombing load is intentionally excluded because Yale warns that the
field is often inaccurate. Prison and burial components remain separate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


EXPOSURE_COLUMNS = [
    "Geography Area km2",
    "Bombing Record Count",
    "Bombing Unique Location Count",
    "Bombing Unique Date Count",
    "Bombing Reported Aircraft Count",
    "Bombing Unique Locations per 100 km2",
    "Log Bombing Unique Locations per 100 km2",
    "Any US Bombing Record",
    "First Bombing Date",
    "Last Bombing Date",
    "Khmer Rouge Prison Count",
    "Any Khmer Rouge Prison",
    "Distance to Nearest Khmer Rouge Prison km",
    "Khmer Rouge Burial Site Count",
    "Any Khmer Rouge Burial Site",
    "Distance to Nearest Khmer Rouge Burial Site km",
    "Khmer Rouge Reported Grave Count",
    "Khmer Rouge Reported Body Count",
    "Any Khmer Rouge Site",
]


def geography_layers(
    boundaries: gpd.GeoDataFrame,
    district_names: dict[str, str],
    province_names: dict[str, str],
) -> dict[str, gpd.GeoDataFrame]:
    communes = boundaries.copy()
    communes["Geography Code"] = communes["com_code"].astype(int).astype(str).str.zfill(6)
    communes["Geography Name"] = communes["com_name"]

    districts = boundaries.dissolve(by="dis_code", as_index=False, aggfunc="first")
    districts["Geography Code"] = districts["dis_code"].astype(int).astype(str).str.zfill(4)
    districts["Geography Name"] = districts["Geography Code"].map(district_names)

    provinces = boundaries.dissolve(by="pro_code", as_index=False, aggfunc="first")
    provinces["Geography Code"] = provinces["pro_code"].astype(int).astype(str).str.zfill(2)
    provinces["Geography Name"] = provinces["Geography Code"].map(province_names)

    return {
        "commune": communes[["Geography Code", "Geography Name", "geometry"]].copy(),
        "district": districts[["Geography Code", "Geography Name", "geometry"]].copy(),
        "province": provinces[["Geography Code", "Geography Name", "geometry"]].copy(),
    }


def load_points(path: Path, crs: object) -> gpd.GeoDataFrame:
    points = gpd.read_file(path).to_crs(crs)
    if points.geometry.isna().any() or points.geometry.is_empty.any():
        raise ValueError(f"Missing or empty geometry in {path}")
    return points


def assign_points(
    points: gpd.GeoDataFrame, geographies: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    assigned = gpd.sjoin(
        points,
        geographies[["Geography Code", "geometry"]],
        how="left",
        predicate="within",
    )
    return assigned.drop(columns=["index_right"])


def nearest_distance_km(
    geographies: gpd.GeoDataFrame, points: gpd.GeoDataFrame
) -> pd.Series:
    point_geometries = points.geometry
    values = []
    for geometry in geographies.geometry:
        representative = geometry.representative_point()
        values.append(float(point_geometries.distance(representative).min()) / 1000)
    return pd.Series(values, index=geographies.index, dtype=float)


def aggregate_level(
    geographies: gpd.GeoDataFrame,
    bombing: gpd.GeoDataFrame,
    prisons: gpd.GeoDataFrame,
    burials: gpd.GeoDataFrame,
    resolution: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    base = geographies.copy()
    base["Geography Resolution"] = resolution
    base["Geography Area km2"] = base.geometry.area / 1_000_000

    bombing_assigned = assign_points(bombing, base)
    prisons_assigned = assign_points(prisons, base)
    burials_assigned = assign_points(burials, base)

    bombing_assigned["Bombing Date"] = pd.to_datetime(
        bombing_assigned["DATE"], unit="ms", errors="coerce"
    )
    bombing_assigned["Bombing Location Key"] = (
        (bombing_assigned.geometry.x / 10).round().astype("Int64").astype(str)
        + ":"
        + (bombing_assigned.geometry.y / 10).round().astype("Int64").astype(str)
    )
    bombing_agg = (
        bombing_assigned.dropna(subset=["Geography Code"])
        .groupby("Geography Code", as_index=False)
        .agg(
            **{
                "Bombing Record Count": ("OBJECTID", "size"),
                "Bombing Unique Location Count": ("Bombing Location Key", "nunique"),
                "Bombing Unique Date Count": ("Bombing Date", "nunique"),
                "Bombing Reported Aircraft Count": ("NUM_ACRFT", "sum"),
                "First Bombing Date": ("Bombing Date", "min"),
                "Last Bombing Date": ("Bombing Date", "max"),
            }
        )
    )
    prison_agg = (
        prisons_assigned.dropna(subset=["Geography Code"])
        .groupby("Geography Code", as_index=False)
        .agg(**{"Khmer Rouge Prison Count": ("OBJECTID", "size")})
    )
    burial_agg = (
        burials_assigned.dropna(subset=["Geography Code"])
        .groupby("Geography Code", as_index=False)
        .agg(
            **{
                "Khmer Rouge Burial Site Count": ("OBJECTID", "size"),
                "Khmer Rouge Reported Grave Count": ("MASSGRAVES", "sum"),
                "Khmer Rouge Reported Body Count": ("BODIES", "sum"),
            }
        )
    )

    output = base.drop(columns="geometry")
    for frame in [bombing_agg, prison_agg, burial_agg]:
        output = output.merge(frame, on="Geography Code", how="left", validate="one_to_one")

    count_columns = [
        "Bombing Record Count",
        "Bombing Unique Location Count",
        "Bombing Unique Date Count",
        "Bombing Reported Aircraft Count",
        "Khmer Rouge Prison Count",
        "Khmer Rouge Burial Site Count",
        "Khmer Rouge Reported Grave Count",
        "Khmer Rouge Reported Body Count",
    ]
    output[count_columns] = output[count_columns].fillna(0)
    output["Bombing Unique Locations per 100 km2"] = (
        100 * output["Bombing Unique Location Count"] / output["Geography Area km2"]
    )
    output["Log Bombing Unique Locations per 100 km2"] = np.log1p(
        output["Bombing Unique Locations per 100 km2"]
    )
    output["Any US Bombing Record"] = output["Bombing Record Count"].gt(0)
    output["Any Khmer Rouge Prison"] = output["Khmer Rouge Prison Count"].gt(0)
    output["Any Khmer Rouge Burial Site"] = output["Khmer Rouge Burial Site Count"].gt(0)
    output["Any Khmer Rouge Site"] = output[
        ["Khmer Rouge Prison Count", "Khmer Rouge Burial Site Count"]
    ].sum(axis=1).gt(0)
    output["Distance to Nearest Khmer Rouge Prison km"] = nearest_distance_km(base, prisons)
    output["Distance to Nearest Khmer Rouge Burial Site km"] = nearest_distance_km(base, burials)

    output = output[
        ["Geography Resolution", "Geography Code", "Geography Name"] + EXPOSURE_COLUMNS
    ].sort_values("Geography Code").reset_index(drop=True)
    validation = [
        {
            "Geography Resolution": resolution,
            "Source": "US Bombing",
            "Source Records": len(bombing),
            "Spatially Assigned Records": int(bombing_assigned["Geography Code"].notna().sum()),
            "Exposed Geographies": int(output["Any US Bombing Record"].sum()),
        },
        {
            "Geography Resolution": resolution,
            "Source": "Khmer Rouge Prisons",
            "Source Records": len(prisons),
            "Spatially Assigned Records": int(prisons_assigned["Geography Code"].notna().sum()),
            "Exposed Geographies": int(output["Any Khmer Rouge Prison"].sum()),
        },
        {
            "Geography Resolution": resolution,
            "Source": "Khmer Rouge Burials",
            "Source Records": len(burials),
            "Spatially Assigned Records": int(burials_assigned["Geography Code"].notna().sum()),
            "Exposed Geographies": int(output["Any Khmer Rouge Burial Site"].sum()),
        },
    ]
    return output, validation


def attach_to_psu(
    crosswalk: pd.DataFrame, exposures: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    pieces = []
    for resolution, exposure in exposures.items():
        subset = crosswalk[crosswalk["Climate Geography Resolution"].eq(resolution)].copy()
        joined = subset.merge(
            exposure,
            left_on="Climate Geography Code",
            right_on="Geography Code",
            how="left",
            validate="many_to_one",
        )
        pieces.append(joined)
    output = pd.concat(pieces, ignore_index=True, sort=False)
    output["Historical Conflict Link Matched"] = output[
        "Log Bombing Unique Locations per 100 km2"
    ].notna()
    keep = [
        "Survey Year",
        "Survey Wave",
        "PSU",
        "Climate Geography Resolution",
        "Climate Geography Code",
        "Climate Link Method",
        "Climate Match Score",
        "Climate Match Margin",
        "Matched Climate Province Name",
        "Matched Climate District Name",
        "Matched Climate Commune Name",
        "Historical Conflict Link Matched",
    ] + EXPOSURE_COLUMNS
    return output[keep].sort_values(["Survey Year", "PSU"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    exp = root / "data" / "exp" / "data-preprocessing"
    processed.mkdir(parents=True, exist_ok=True)
    exp.mkdir(parents=True, exist_ok=True)

    boundaries = gpd.read_file(raw / "geography" / "odc_cambodia_communes_2014.gpkg")
    if boundaries.crs is None or not boundaries.crs.is_projected:
        boundaries = boundaries.to_crs(32648)
    names = gpd.read_file(raw / "geography" / "cambodia_commune_boundaries_2018_2024.geojson")
    district_names = (
        names.assign(code=names["ADM2_PCODE"].str.replace("KH", "", regex=False))
        .drop_duplicates("code")
        .set_index("code")["ADM2_EN"]
        .to_dict()
    )
    province_names = (
        names.assign(code=names["ADM1_PCODE"].str.replace("KH", "", regex=False))
        .drop_duplicates("code")
        .set_index("code")["ADM1_EN"]
        .to_dict()
    )
    geographies = geography_layers(boundaries, district_names, province_names)

    bombing = load_points(raw / "conflict" / "yale_cgeo_us_bombing_sites.geojson", boundaries.crs)
    prisons = load_points(
        raw / "conflict" / "yale_cgeo_khmer_rouge_prisons.geojson", boundaries.crs
    )
    burials = load_points(
        raw / "conflict" / "yale_cgeo_khmer_rouge_burials.geojson", boundaries.crs
    )

    exposures: dict[str, pd.DataFrame] = {}
    validations: list[dict[str, object]] = []
    for resolution, geography in geographies.items():
        exposure, validation = aggregate_level(
            geography, bombing, prisons, burials, resolution
        )
        exposures[resolution] = exposure
        validations.extend(validation)
        path = processed / f"historical_conflict_{resolution}_preprocessed.parquet"
        exposure.to_parquet(path, index=False)
        print(f"{path.name}: rows={len(exposure):,}, columns={len(exposure.columns):,}")

    crosswalk = pd.read_parquet(
        processed / "direction3_historical_geography_crosswalk_preprocessed.parquet"
    )
    psu = attach_to_psu(crosswalk, exposures)
    psu_path = processed / "direction3_psu_conflict_exposure_preprocessed.parquet"
    psu.to_parquet(psu_path, index=False)
    if not psu["Historical Conflict Link Matched"].all():
        raise ValueError("One or more PSU-wave rows lack historical conflict exposure")

    validation = pd.DataFrame(validations)
    validation.to_csv(exp / "direction3_historical_conflict_validation.csv", index=False)
    distribution = psu[
        ["Survey Year", "PSU"]
        + [
            "Log Bombing Unique Locations per 100 km2",
            "Khmer Rouge Prison Count",
            "Khmer Rouge Burial Site Count",
        ]
    ].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T
    distribution.to_csv(exp / "direction3_historical_conflict_distribution.csv")
    print(f"{psu_path.name}: rows={len(psu):,}, columns={len(psu.columns):,}")
    print(f"linked_psu_rows={int(psu['Historical Conflict Link Matched'].sum()):,}")


if __name__ == "__main__":
    main()
