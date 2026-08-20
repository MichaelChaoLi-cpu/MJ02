#!/usr/bin/env python3
"""Construct blinded historical-boundary design variables for CSES PSU-wave rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import unicodedata
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd


SIDE = "Historical Repression Side"
TREATMENT = "Higher-Repression Southwest Zone"
SIGNED_DISTANCE = "Signed Distance to Historical Repression Boundary km"
ABS_DISTANCE = "Absolute Distance to Historical Repression Boundary km"
SEGMENT = "Historical Boundary Segment"
BANDWIDTHS_KM = (2, 5, 10, 15, 20, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/historical_boundary_design_preprocessed.parquet"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/exp/data-preprocessing/historical-boundary"),
    )
    return parser.parse_args()


def normalized_code(values: pd.Series, width: int) -> pd.Series:
    return (
        values.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(width)
    )


def normalized_name(value: object) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    )
    return re.sub(r"[^a-z0-9]", "", ascii_text.lower())


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - archive verification
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source(source: Path) -> pd.DataFrame:
    manifest = pd.read_csv(source / "source_manifest.csv", dtype={"Datafile ID": str})
    for _, row in manifest.iterrows():
        path = source / row["Filename"]
        if not path.exists():
            raise FileNotFoundError(path)
        if md5sum(path) != row["Expected MD5"]:
            raise RuntimeError(f"Checksum mismatch: {path}")
    return manifest


def load_zones(source: Path) -> tuple[object, object]:
    archive_path = source / "Democratic_Kampuchea_Zones.zip"
    with tempfile.TemporaryDirectory(prefix="mj02-zone-preprocess-") as temp_dir:
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_dir)
        shapefiles = list(Path(temp_dir).rglob("*.shp"))
        if len(shapefiles) != 1:
            raise RuntimeError(f"Expected one zone shapefile, found {len(shapefiles)}")
        zones = gpd.read_file(shapefiles[0]).to_crs(32648)
    southwest = zones.loc[zones["ZONE_NAME"].eq("Southwest"), "geometry"].union_all()
    west = zones.loc[zones["ZONE_NAME"].eq("West"), "geometry"].union_all()
    return southwest, west


def link_public_villages(root: Path) -> pd.DataFrame:
    crosswalk = pd.read_parquet(
        root / "data/processed/direction3_historical_geography_crosswalk_preprocessed.parquet"
    )
    crosswalk = crosswalk.loc[crosswalk["Province Name"].eq("Kampong Speu")].copy()
    crosswalk["Survey Wave"] = crosswalk["Survey Wave"].astype(str)
    crosswalk["PSU"] = crosswalk["PSU"].astype(str)
    crosswalk["Village Code Normalized"] = normalized_code(crosswalk["Village Code"], 8)
    crosswalk["Commune Code Normalized"] = normalized_code(crosswalk["Commune Code"], 6)
    crosswalk["Village Name Normalized"] = crosswalk["Village Name"].map(normalized_name)

    villages = gpd.read_file(
        root / "data/raw/conflict/yale_cgeo_historical_villages.geojson"
    ).to_crs(32648)
    villages["Matched Public Village Code"] = normalized_code(villages["CODEPHUM"], 8)
    villages["Commune Code Normalized"] = villages["Matched Public Village Code"].str[:6]
    villages["Public Village Name Normalized"] = villages["PHUM"].map(normalized_name)
    villages = villages[
        [
            "Matched Public Village Code",
            "Commune Code Normalized",
            "Public Village Name Normalized",
            "geometry",
        ]
    ].drop_duplicates("Matched Public Village Code")

    exact = villages.rename(columns={"Matched Public Village Code": "Village Code Normalized"})[
        ["Village Code Normalized", "geometry"]
    ]
    linked = crosswalk.merge(exact, on="Village Code Normalized", how="left", validate="many_to_one")
    linked["Matched Public Village Code"] = linked["Village Code Normalized"].where(
        linked["geometry"].notna()
    )
    linked["Public Village Point Link Method"] = linked["geometry"].notna().map(
        {True: "exact village code", False: "unresolved"}
    )

    name_candidates = (
        villages.groupby(["Commune Code Normalized", "Public Village Name Normalized"])
        .filter(lambda group: len(group) == 1)
        .rename(columns={"Public Village Name Normalized": "Village Name Normalized"})
    )
    unresolved = linked["geometry"].isna()
    fallback = linked.loc[
        unresolved,
        ["Commune Code Normalized", "Village Name Normalized"],
    ].merge(
        name_candidates,
        on=["Commune Code Normalized", "Village Name Normalized"],
        how="left",
        validate="many_to_one",
    )
    fallback.index = linked.index[unresolved]
    resolved_fallback = fallback["geometry"].notna()
    linked.loc[fallback.index[resolved_fallback], "geometry"] = fallback.loc[
        resolved_fallback, "geometry"
    ]
    linked.loc[fallback.index[resolved_fallback], "Matched Public Village Code"] = fallback.loc[
        resolved_fallback, "Matched Public Village Code"
    ]
    linked.loc[
        fallback.index[resolved_fallback], "Public Village Point Link Method"
    ] = "unique exact normalized name within commune"
    linked["Public Village Point Matched"] = linked["geometry"].notna()
    return linked


def construct_design(linked: pd.DataFrame, source: Path) -> pd.DataFrame:
    southwest, west = load_zones(source)
    segment_table = pd.read_csv(source / "boundary_segment_points.csv")
    segment_points = gpd.GeoSeries.from_xy(
        segment_table["Easting EPSG 32648"],
        segment_table["Northing EPSG 32648"],
        crs=32648,
    )
    boundary = southwest.boundary.intersection(west.boundary)

    def assign_side(point: object) -> str:
        if point is None or not hasattr(point, "within"):
            return "unresolved"
        if point.within(southwest):
            return "Southwest"
        if point.within(west):
            return "West"
        return "outside zones"

    linked[SIDE] = linked["geometry"].map(assign_side)
    linked[TREATMENT] = pd.Series(pd.NA, index=linked.index, dtype="Int8")
    linked.loc[linked[SIDE].eq("Southwest"), TREATMENT] = 1
    linked.loc[linked[SIDE].eq("West"), TREATMENT] = 0
    unsigned_distance = linked["geometry"].map(
        lambda point: point.distance(boundary) / 1000
        if point is not None and hasattr(point, "distance")
        else pd.NA
    )
    linked[SIGNED_DISTANCE] = pd.to_numeric(unsigned_distance, errors="coerce")
    linked.loc[linked[SIDE].eq("West"), SIGNED_DISTANCE] *= -1
    linked.loc[~linked[SIDE].isin(["Southwest", "West"]), SIGNED_DISTANCE] = pd.NA
    linked[ABS_DISTANCE] = linked[SIGNED_DISTANCE].abs()

    linked[SEGMENT] = pd.Series(pd.NA, index=linked.index, dtype="Int8")
    matched = linked["Public Village Point Matched"] & linked[SIDE].isin(["Southwest", "West"])
    matched_points = gpd.GeoSeries(linked.loc[matched, "geometry"], crs=32648)
    nearest_segment = matched_points.map(
        lambda point: int(segment_points.distance(point).to_numpy().argmin()) + 1
    )
    linked.loc[matched, SEGMENT] = nearest_segment.astype("Int8")
    linked["Historical-Boundary Link Eligible"] = matched.astype("Int8")
    for bandwidth in BANDWIDTHS_KM:
        name = f"Historical-Boundary Common Support {bandwidth} km"
        linked[name] = (
            matched & linked[ABS_DISTANCE].le(bandwidth)
        ).astype("Int8")
    return linked


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source = args.source.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    audit_output = args.audit_output if args.audit_output.is_absolute() else root / args.audit_output
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.mkdir(parents=True, exist_ok=True)

    manifest = verify_source(source)
    linked = construct_design(link_public_villages(root), source)
    output_columns = [
        "Survey Year",
        "Survey Wave",
        "PSU",
        "Province Code Component",
        "District Code Component",
        "Commune Code Normalized",
        "Village Code Normalized",
        "Matched Public Village Code",
        "Province Name",
        "District Name",
        "Commune Name",
        "Village Name",
        "Public Village Point Link Method",
        "Public Village Point Matched",
        "Historical-Boundary Link Eligible",
        SIDE,
        TREATMENT,
        SIGNED_DISTANCE,
        ABS_DISTANCE,
        SEGMENT,
        *[f"Historical-Boundary Common Support {bandwidth} km" for bandwidth in BANDWIDTHS_KM],
    ]
    processed = linked[output_columns].copy()
    processed.to_parquet(output, index=False)

    unresolved = processed.loc[~processed["Public Village Point Matched"]].copy()
    unresolved.to_csv(audit_output / "unresolved_boundary_village_links.csv", index=False)
    linkage_audit = (
        processed.groupby("Public Village Point Link Method", dropna=False)
        .agg(
            **{
                "PSU-Wave Rows": ("PSU", "size"),
                "Unique Village Codes": ("Village Code Normalized", "nunique"),
                "Survey Waves": ("Survey Wave", "nunique"),
            }
        )
        .reset_index()
    )
    linkage_audit.to_csv(audit_output / "historical_boundary_linkage_audit.csv", index=False)

    variable_rows = []
    for column in output_columns:
        variable_rows.append(
            {
                "readable_name": column,
                "dtype": str(processed[column].dtype),
                "null_percentage": float(processed[column].isna().mean() * 100),
                "is_final_variable": (
                    "no"
                    if column.startswith("Historical-Boundary Common Support")
                    else "no" if column in {TREATMENT, SIGNED_DISTANCE, SEGMENT} else "reference"
                ),
                "preprocessing": (
                    "deterministic projected spatial construction; no imputation; no outlier treatment"
                ),
            }
        )
    pd.DataFrame(variable_rows).to_csv(audit_output / "variable_list.csv", index=False)

    decisions = {
        "dataset": "Historical Southwest-West boundary design",
        "source_manifest": str(source / "source_manifest.csv"),
        "output": str(output),
        "rows": len(processed),
        "variables": variable_rows,
        "rules": {
            "primary_link": "exact public village code",
            "fallback_link": "unique exact normalized village name within the same commune",
            "unresolved": "preserve as missing; no fuzzy match",
            "treatment": "one for Southwest and zero for West",
            "distance": "projected EPSG 32648 distance to the shared Southwest-West boundary, positive in Southwest",
            "segments": "nearest of five independently reconstructed public boundary vertices",
            "common_support": "candidate fixed bandwidth indicators; final bandwidth not selected",
            "outcomes_read": False,
        },
        "source_files_verified": int(len(manifest)),
    }
    (audit_output / "decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    readme = f"""# Historical Boundary Data Preprocessing

- Source files verified: {len(manifest)}
- Output rows: {len(processed)}
- Output columns: {len(processed.columns)}
- Exact-code links: {(processed['Public Village Point Link Method'] == 'exact village code').sum()}
- Deterministic name fallback links: {(processed['Public Village Point Link Method'] == 'unique exact normalized name within commune').sum()}
- Unresolved links: {(~processed['Public Village Point Matched']).sum()}
- Treatment sides represented: {', '.join(sorted(processed.loc[processed['Historical-Boundary Link Eligible'].eq(1), SIDE].unique()))}
- Boundary segments represented: {', '.join(map(str, sorted(processed[SEGMENT].dropna().astype(int).unique())))}

No outcomes were read. No missing value was imputed, no fuzzy-name match was accepted, and no
bandwidth was selected using an outcome. Candidate common-support indicators remain non-final until
the design-power and continuity gates are completed.
"""
    (audit_output / "README.md").write_text(readme, encoding="utf-8")

    print(f"output: {output}")
    print(f"shape: {processed.shape}")
    print(linkage_audit.to_string(index=False))


if __name__ == "__main__":
    main()
