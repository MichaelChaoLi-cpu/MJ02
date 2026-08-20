#!/usr/bin/env python3
"""Link the blinded VIIRS panel to the frozen boundary and rainfall design.

All spatial construction is deterministic and does not summarize or model radiance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from preprocess_historical_boundary_design import load_zones, verify_source


YEARS = tuple(range(2013, 2022))
BANDWIDTHS_KM = (2, 5, 10, 15, 20, 30)
SIDE = "Historical Repression Side"
TREATMENT = "Higher-Repression Southwest Zone"
DISTANCE = "Signed Distance to Historical Repression Boundary km"
ABS_DISTANCE = "Absolute Distance to Historical Repression Boundary km"
SEGMENT = "Historical Boundary Segment"
COMMUNE = "Linked Climate Commune Code"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--viirs",
        type=Path,
        default=Path("data/processed/viirs_vnl_v21_pixel_year_preprocessed.parquet"),
    )
    parser.add_argument(
        "--climate",
        type=Path,
        default=Path("data/processed/chirps_long_baseline_commune_year_preprocessed.parquet"),
    )
    parser.add_argument(
        "--boundary-source",
        type=Path,
        default=Path("data/exp/data-preprocessing/historical-boundary-source"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/viirs_historical_boundary_climate_preprocessed.parquet"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/exp/data-preprocessing/viirs-boundary-climate"),
    )
    return parser.parse_args()


def resolved(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_spatial_crosswalk(root: Path, cells: pd.DataFrame, source: Path) -> pd.DataFrame:
    verify_source(source)
    southwest, west = load_zones(source)
    boundary = southwest.boundary.intersection(west.boundary)
    segment_table = pd.read_csv(source / "boundary_segment_points.csv")
    segment_points = gpd.GeoSeries.from_xy(
        segment_table["Easting EPSG 32648"],
        segment_table["Northing EPSG 32648"],
        crs=32648,
    )

    points = gpd.GeoDataFrame(
        cells.copy(),
        geometry=gpd.points_from_xy(cells["Longitude"], cells["Latitude"], crs=4326),
        crs=4326,
    ).to_crs(32648)
    points[SIDE] = "outside zones"
    points.loc[points.geometry.within(southwest), SIDE] = "Southwest"
    points.loc[points.geometry.within(west), SIDE] = "West"
    points[TREATMENT] = pd.Series(pd.NA, index=points.index, dtype="Int8")
    points.loc[points[SIDE].eq("Southwest"), TREATMENT] = 1
    points.loc[points[SIDE].eq("West"), TREATMENT] = 0

    eligible = points[SIDE].isin(["Southwest", "West"])
    unsigned = points.geometry.distance(boundary) / 1000
    points[DISTANCE] = unsigned.where(eligible)
    points.loc[points[SIDE].eq("West"), DISTANCE] *= -1
    points[ABS_DISTANCE] = points[DISTANCE].abs()
    points[SEGMENT] = pd.Series(pd.NA, index=points.index, dtype="Int8")
    points.loc[eligible, SEGMENT] = points.loc[eligible, "geometry"].map(
        lambda point: int(segment_points.distance(point).to_numpy().argmin()) + 1
    ).astype("Int8")
    for bandwidth in BANDWIDTHS_KM:
        points[f"Historical-Boundary Common Support {bandwidth} km"] = (
            eligible & points[ABS_DISTANCE].le(bandwidth)
        ).astype("Int8")

    commune_boundaries = gpd.read_file(
        root / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
    )[["ADM3_PCODE", "ADM3_EN", "geometry"]].to_crs(32648)
    commune_boundaries[COMMUNE] = (
        commune_boundaries["ADM3_PCODE"].astype("string").str.replace("KH", "", regex=False)
    )
    commune_boundaries = commune_boundaries.drop_duplicates(COMMUNE)
    joined = gpd.sjoin(
        points,
        commune_boundaries[[COMMUNE, "ADM3_EN", "geometry"]],
        how="left",
        predicate="within",
    )
    if joined.duplicated("Grid Cell ID").any():
        raise RuntimeError("A VIIRS grid-cell centre matched more than one climate commune")
    joined = joined.rename(columns={"ADM3_EN": "Linked Climate Commune Name"})
    joined["Annual Climate Link Method"] = joined[COMMUNE].notna().map(
        {True: "grid-cell centre within modern climate commune", False: "unresolved"}
    )
    joined["Published Kampong Speu Replication Frame"] = (
        joined[COMMUNE].astype("string").str.startswith("05", na=False).astype("Int8")
    )
    # The public replication frame is confined to Kampong Speu. Restricting the
    # common-support flags prevents other portions of the same historical zone border
    # (including the Phnom Penh and Kandal vicinity) from changing the local estimand.
    for bandwidth in BANDWIDTHS_KM:
        joined[f"Historical-Boundary Common Support {bandwidth} km"] = (
            joined[SIDE].isin(["Southwest", "West"])
            & joined[ABS_DISTANCE].le(bandwidth)
            & joined["Published Kampong Speu Replication Frame"].eq(1)
        ).astype("Int8")
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


def make_audits(crosswalk: pd.DataFrame, panel: pd.DataFrame, output: Path) -> None:
    support_records = []
    for bandwidth in BANDWIDTHS_KM:
        supported = crosswalk.loc[
            crosswalk[f"Historical-Boundary Common Support {bandwidth} km"].eq(1)
        ]
        for side, group in supported.groupby(SIDE, observed=True):
            support_records.append(
                {
                    "Bandwidth km": bandwidth,
                    "Historical Repression Side": side,
                    "Unique Grid Cells": group["Grid Cell ID"].nunique(),
                    "Boundary Segments": group[SEGMENT].nunique(),
                    "Climate Communes": group[COMMUNE].nunique(),
                }
            )
    pd.DataFrame(support_records).to_csv(
        output / "viirs_boundary_linkage_coverage.csv", index=False
    )

    primary = crosswalk.loc[
        crosswalk["Historical-Boundary Common Support 5 km"].eq(1)
    ]
    commune_side = (
        primary.groupby([COMMUNE, "Linked Climate Commune Name", SIDE], dropna=False)
        .agg(**{"Unique Grid Cells": ("Grid Cell ID", "nunique")})
        .reset_index()
    )
    side_counts = primary.groupby(COMMUNE, observed=True)[TREATMENT].nunique()
    cross_side = set(side_counts.loc[side_counts.eq(2)].index.astype(str))
    commune_side["Cross-Side Commune at 5 km"] = commune_side[COMMUNE].astype(str).isin(cross_side)
    commune_side.to_csv(output / "viirs_cross_side_communes.csv", index=False)

    linkage = pd.DataFrame(
        [
            {
                "Grid Cells": crosswalk["Grid Cell ID"].nunique(),
                "Inside Historical Zones": int(crosswalk[SIDE].isin(["Southwest", "West"]).sum()),
                "Linked Climate Commune": int(crosswalk[COMMUNE].notna().sum()),
                "Five km Grid Cells": int(primary["Grid Cell ID"].nunique()),
                "Five km Cross-Side Communes": len(cross_side),
                "Panel Rows": len(panel),
                "Rainfall Linked Rows": int(panel["Annual Climate Shock Available"].sum()),
            }
        ]
    )
    linkage.to_csv(output / "viirs_climate_linkage_validation.csv", index=False)

    metadata = {
        "spatial_unit": "VIIRS grid-cell centre",
        "treatment_assignment": "point within checksum-verified Southwest or West zone",
        "distance": "EPSG 32648 distance to shared Southwest-West boundary",
        "primary_bandwidth_km": 5,
        "alternative_bandwidths_km": [2, 10, 15, 20, 30],
        "replication_frame": "grid-cell centre in a modern Kampong Speu commune (code prefix 05)",
        "climate_link": "grid-cell centre within modern climate commune",
        "period": f"{YEARS[0]}-{YEARS[-1]}",
        "outcome_coefficients_inspected": False,
    }
    (output / "decisions.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    viirs_path = resolved(root, args.viirs)
    climate_path = resolved(root, args.climate)
    source = resolved(root, args.boundary_source)
    output_path = resolved(root, args.output)
    audit_output = resolved(root, args.audit_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_output.mkdir(parents=True, exist_ok=True)

    viirs = pd.read_parquet(viirs_path)
    cells = viirs[["Grid Cell ID", "Grid Row", "Grid Column", "Longitude", "Latitude"]].drop_duplicates(
        "Grid Cell ID"
    )
    if len(cells) * len(YEARS) != len(viirs):
        raise RuntimeError("VIIRS input is not a balanced cell-year panel")
    crosswalk = build_spatial_crosswalk(root, cells, source)
    climate = pd.read_parquet(climate_path)
    climate["Climate Geography Code"] = climate["Climate Geography Code"].astype("string").str.zfill(6)
    climate = climate.loc[climate["Year"].isin(YEARS)].copy()
    if climate.duplicated(["Climate Geography Code", "Year"]).any():
        raise RuntimeError("Climate commune-year keys are not unique")

    panel = viirs.merge(
        crosswalk.drop(columns=["Grid Row", "Grid Column", "Longitude", "Latitude"]),
        on="Grid Cell ID",
        how="left",
        validate="many_to_one",
    )
    panel = panel.merge(
        climate,
        left_on=[COMMUNE, "Year"],
        right_on=["Climate Geography Code", "Year"],
        how="left",
        validate="many_to_one",
    ).drop(columns="Climate Geography Code")
    panel["Annual Climate Shock Available"] = (
        panel["Annual Rainfall Anomaly Z (1991-2020)"].notna().astype("Int8")
    )
    if len(panel) != len(viirs) or panel.duplicated(["Grid Cell ID", "Year"]).any():
        raise RuntimeError("Spatial-climate linkage changed VIIRS panel keys")
    primary = panel["Historical-Boundary Common Support 5 km"].eq(1)
    if not panel.loc[primary, "Annual Climate Shock Available"].eq(1).all():
        raise RuntimeError("Rainfall linkage is incomplete inside primary 5 km support")

    panel.to_parquet(output_path, index=False)
    crosswalk.to_csv(audit_output / "viirs_spatial_crosswalk.csv", index=False)
    make_audits(crosswalk, panel, audit_output)
    print(f"Wrote {len(panel):,} rows to {output_path}")
    print(f"Five-kilometre support: {int(primary.sum()):,} cell-years")
    print(f"Wrote linkage audit to {audit_output}")


if __name__ == "__main__":
    main()
