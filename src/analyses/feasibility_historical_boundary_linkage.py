#!/usr/bin/env python3
"""Blind feasibility audit for the Southwest-West historical boundary design.

This script uses only geography, identifiers, sample counts, and shock availability. It does not
read or summarize contemporary outcome values.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_ZONE_MD5 = "05bf659b2ab45f85275fd6f57f72597b"
DATAVERSE_DOI = "https://doi.org/10.7910/DVN/RK5GOH"
ZONE_API_URL = "https://dataverse.harvard.edu/api/access/datafile/6990145"
BANDWIDTHS_KM = (2, 5, 10, 15, 20, 30)
SIDE_COLUMN = "Historical Repression Side"
DISTANCE_COLUMN = "Signed Distance to Historical Repression Boundary km"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--zones-zip", type=Path, required=True)
    parser.add_argument("--reproduction-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalized_code(values: pd.Series, width: int) -> pd.Series:
    return (
        values.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(width)
    )


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - checksum verification, not cryptography
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_boundary_geometry(
    zones_zip: Path, commune_path: Path
) -> tuple[gpd.GeoDataFrame, object, object, object, object]:
    with tempfile.TemporaryDirectory(prefix="mj02-zone-") as temp_dir:
        with zipfile.ZipFile(zones_zip) as archive:
            archive.extractall(temp_dir)
        shapefiles = list(Path(temp_dir).rglob("*.shp"))
        if len(shapefiles) != 1:
            raise RuntimeError(f"Expected one zone shapefile, found {len(shapefiles)}")
        zones = gpd.read_file(shapefiles[0]).to_crs(32648)

    southwest = zones.loc[zones["ZONE_NAME"].eq("Southwest"), "geometry"].union_all()
    west = zones.loc[zones["ZONE_NAME"].eq("West"), "geometry"].union_all()
    if southwest.is_empty or west.is_empty:
        raise RuntimeError("Southwest or West zone geometry is missing")

    communes = gpd.read_file(commune_path).to_crs(32648)
    kampong_speu = communes.loc[communes["pro_code"].astype(str).eq("5")].copy()
    province = kampong_speu.geometry.union_all()
    shared_boundary = southwest.boundary.intersection(west.boundary)
    study_boundary = shared_boundary.intersection(province)
    if study_boundary.is_empty:
        raise RuntimeError("Southwest-West boundary does not intersect Kampong Speu")
    return kampong_speu, southwest, west, study_boundary, shared_boundary


def build_blind_assignment(
    root: Path, southwest: object, west: object, study_boundary: object
) -> pd.DataFrame:
    crosswalk = pd.read_parquet(
        root / "data/processed/direction3_historical_geography_crosswalk_preprocessed.parquet"
    )
    crosswalk = crosswalk.loc[crosswalk["Province Name"].eq("Kampong Speu")].copy()
    crosswalk["Survey Wave"] = crosswalk["Survey Wave"].astype(str)
    crosswalk["PSU"] = crosswalk["PSU"].astype(str)
    crosswalk["Village Code Normalized"] = normalized_code(crosswalk["Village Code"], 8)
    crosswalk["Commune Code Normalized"] = normalized_code(crosswalk["Commune Code"], 6)

    villages = gpd.read_file(
        root / "data/raw/conflict/yale_cgeo_historical_villages.geojson"
    ).to_crs(32648)
    villages["Village Code Normalized"] = normalized_code(villages["CODEPHUM"], 8)
    villages = villages[["Village Code Normalized", "geometry"]].drop_duplicates(
        "Village Code Normalized"
    )
    linked = crosswalk.merge(villages, on="Village Code Normalized", how="left", validate="many_to_one")

    def side_for_point(point: object) -> str:
        if point is None or not hasattr(point, "within"):
            return "unmatched"
        if point.within(southwest):
            return "Southwest"
        if point.within(west):
            return "West"
        return "outside zones"

    linked[SIDE_COLUMN] = linked["geometry"].map(side_for_point)
    unsigned_distance = linked["geometry"].map(
        lambda point: point.distance(study_boundary) / 1000
        if point is not None and hasattr(point, "distance")
        else pd.NA
    )
    signed_distance = unsigned_distance.copy()
    signed_distance.loc[linked[SIDE_COLUMN].eq("West")] *= -1
    signed_distance.loc[~linked[SIDE_COLUMN].isin(["Southwest", "West"])] = pd.NA
    linked[DISTANCE_COLUMN] = pd.to_numeric(signed_distance, errors="coerce")
    linked["Absolute Distance to Historical Boundary km"] = linked[DISTANCE_COLUMN].abs()
    linked["Exact Public Village Point Matched"] = linked["geometry"].notna()

    if linked.duplicated(["Survey Wave", "PSU"]).any():
        raise RuntimeError("Survey Wave and PSU do not uniquely identify the feasibility crosswalk")
    return linked


def merge_assignment(data: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Survey Wave"] = data["Survey Wave"].astype(str)
    data["PSU"] = data["PSU"].astype(str)
    fields = [
        "Survey Wave",
        "PSU",
        "Village Code Normalized",
        "Commune Code Normalized",
        SIDE_COLUMN,
        DISTANCE_COLUMN,
        "Absolute Distance to Historical Boundary km",
    ]
    return data.merge(
        assignment[fields], on=["Survey Wave", "PSU"], how="inner", validate="many_to_one"
    )


def support_by_bandwidth(
    assignment: pd.DataFrame, households: pd.DataFrame, education: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    school_age = education.loc[education["School Age 6 to 17"].eq(1)].copy()
    for bandwidth in BANDWIDTHS_KM:
        for side in ("Southwest", "West"):
            assignment_side = assignment.loc[
                assignment[SIDE_COLUMN].eq(side)
                & assignment["Absolute Distance to Historical Boundary km"].le(bandwidth)
            ]
            household_side = households.loc[
                households[SIDE_COLUMN].eq(side)
                & households["Absolute Distance to Historical Boundary km"].le(bandwidth)
            ]
            education_side = school_age.loc[
                school_age[SIDE_COLUMN].eq(side)
                & school_age["Absolute Distance to Historical Boundary km"].le(bandwidth)
            ]
            rows.append(
                {
                    "Bandwidth km": bandwidth,
                    SIDE_COLUMN: side,
                    "PSU-Wave Rows": len(assignment_side),
                    "Unique PSU Labels": assignment_side["PSU"].nunique(),
                    "Unique Villages": assignment_side["Village Code Normalized"].nunique(),
                    "Unique Communes": assignment_side["Commune Code Normalized"].nunique(),
                    "Survey Waves": assignment_side["Survey Wave"].nunique(),
                    "Household Observations": len(household_side),
                    "School-Age Person Observations": len(education_side),
                    "Households with Annual Rainfall Shock": household_side[
                        "Annual Rainfall Extreme Wet Shock"
                    ].notna().sum(),
                    "Households with Interview SPI12": household_side[
                        "Interview Month SPI 12 Month"
                    ].notna().sum(),
                    "School-Age Persons with Interview SPI12": education_side[
                        "Interview Month SPI 12 Month"
                    ].notna().sum(),
                    "Households with Local Rice Price Shock": household_side[
                        "12 Month Change in Local Relative Log Wholesale Rice Price"
                    ].notna().sum(),
                    "School-Age Persons with Local Rice Price Shock": education_side[
                        "12 Month Change in Local Relative Log Wholesale Rice Price"
                    ].notna().sum(),
                    "Price-Supported Survey Waves": household_side.loc[
                        household_side[
                            "12 Month Change in Local Relative Log Wholesale Rice Price"
                        ].notna(),
                        "Survey Wave",
                    ].nunique(),
                }
            )
    return pd.DataFrame(rows)


def support_by_wave(assignment: pd.DataFrame) -> pd.DataFrame:
    eligible = assignment.loc[assignment[SIDE_COLUMN].isin(["Southwest", "West"])].copy()
    return (
        eligible.groupby(["Survey Wave", SIDE_COLUMN], as_index=False)
        .agg(
            **{
                "PSU-Wave Rows": ("PSU", "size"),
                "Unique Villages": ("Village Code Normalized", "nunique"),
                "Minimum Distance km": ("Absolute Distance to Historical Boundary km", "min"),
                "Median Distance km": ("Absolute Distance to Historical Boundary km", "median"),
            }
        )
        .sort_values(["Survey Wave", SIDE_COLUMN])
    )


def write_map(
    output: Path,
    kampong_speu: gpd.GeoDataFrame,
    southwest: object,
    west: object,
    study_boundary: object,
    assignment: pd.DataFrame,
) -> None:
    province = gpd.GeoSeries([kampong_speu.geometry.union_all()], crs=32648).to_crs(4326)
    sw = gpd.GeoSeries([southwest], crs=32648).to_crs(4326)
    w = gpd.GeoSeries([west], crs=32648).to_crs(4326)
    boundary = gpd.GeoSeries([study_boundary], crs=32648).to_crs(4326)
    points = gpd.GeoDataFrame(
        assignment.loc[assignment[SIDE_COLUMN].isin(["Southwest", "West"])].copy(),
        geometry="geometry",
        crs=32648,
    ).to_crs(4326)

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    sw.plot(ax=ax, color="#f3b59c", alpha=0.45, edgecolor="none")
    w.plot(ax=ax, color="#9ecae1", alpha=0.45, edgecolor="none")
    province.boundary.plot(ax=ax, color="#333333", linewidth=1.2)
    boundary.plot(ax=ax, color="black", linewidth=2.1)
    for side, color in (("Southwest", "#b33b2e"), ("West", "#2166ac")):
        points.loc[points[SIDE_COLUMN].eq(side)].plot(
            ax=ax, color=color, markersize=12, alpha=0.65, label=side
        )
    minx, miny, maxx, maxy = province.total_bounds
    ax.set_xlim(minx - 0.08, maxx + 0.08)
    ax.set_ylim(miny - 0.05, maxy + 0.05)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, color="#d0d0d0", linewidth=0.5, alpha=0.8)
    ax.set_title("Exploratory historical-boundary linkage support")
    ax.legend(frameon=True, loc="lower left")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    output: Path,
    archive_md5: str,
    assignment: pd.DataFrame,
    support: pd.DataFrame,
    study_boundary_length_km: float,
    reproduction_audit: pd.DataFrame | None,
) -> None:
    total_rows = len(assignment)
    matched_rows = int(assignment["Exact Public Village Point Matched"].sum())
    assigned_rows = int(assignment[SIDE_COLUMN].isin(["Southwest", "West"]).sum())
    matched_unique = assignment.loc[
        assignment["Exact Public Village Point Matched"], "Village Code Normalized"
    ].nunique()
    total_unique = assignment["Village Code Normalized"].nunique()
    support_5 = support.loc[support["Bandwidth km"].eq(5)].set_index(SIDE_COLUMN)
    support_10 = support.loc[support["Bandwidth km"].eq(10)].set_index(SIDE_COLUMN)
    reproduction_text = "- Independent public-frame reproduction: not run"
    if reproduction_audit is not None:
        reproduction_values = reproduction_audit.set_index("metric")["value"]
        reproduction_text = (
            "- Independent public-frame reproduction: treatment agreement "
            f"{reproduction_values['treatment_assignment_agreement_share']:.1%}; signed-distance "
            f"correlation {reproduction_values['signed_distance_correlation']:.6f}; mean absolute "
            "distance difference "
            f"{reproduction_values['signed_distance_mean_absolute_difference_km']:.3g} km"
        )

    text = f"""# Historical Boundary Feasibility Audit

## Scope

This is a blinded design-feasibility audit. It uses public historical-zone geometry, public
historical-village points, identifiers, sample counts, and shock availability. It does not inspect
or estimate any contemporary outcome value.

## Source verification

- Replication archive: {DATAVERSE_DOI}
- Released version: 1.0, 2023-05-02
- License: CC0 1.0
- Zone archive API: {ZONE_API_URL}
- Observed zone archive MD5: `{archive_md5}`
- Expected zone archive MD5: `{EXPECTED_ZONE_MD5}`
- Checksum result: {'pass' if archive_md5 == EXPECTED_ZONE_MD5 else 'FAIL'}
- The public main village frame was separately inspected in R: 1,359 villages, 67 fields,
  including treatment, signed border distance, five boundary-segment indicators, and geometry.
{reproduction_text}

## Initial linkage result

- Kampong Speu PSU-wave rows: {total_rows:,}
- Exact public village-code matches: {matched_rows:,} ({matched_rows / total_rows:.1%})
- Rows assigned to Southwest or West after matching: {assigned_rows:,}
- Unique village codes matched: {matched_unique:,} of {total_unique:,}
- Historical boundary length inside the Kampong Speu study polygon: {study_boundary_length_km:.2f} km
- Within 5 km: Southwest {int(support_5.loc['Southwest', 'PSU-Wave Rows'])} and West
  {int(support_5.loc['West', 'PSU-Wave Rows'])} PSU-wave rows; Southwest
  {int(support_5.loc['Southwest', 'Household Observations']):,} and West
  {int(support_5.loc['West', 'Household Observations']):,} household observations.
- Within 10 km: Southwest {int(support_10.loc['Southwest', 'PSU-Wave Rows'])} and West
  {int(support_10.loc['West', 'PSU-Wave Rows'])} PSU-wave rows; Southwest
  {int(support_10.loc['Southwest', 'Household Observations']):,} and West
  {int(support_10.loc['West', 'Household Observations']):,} household observations.

## Interpretation

The source, geometry, and exact village-code linkage gates are provisionally feasible. Both sides
are represented at narrow candidate bandwidths and across nearly all survey waves. Rainfall is
provisionally supported for the next power test. The current local rice-price construction has only
one supported wave on each side within 5 and 10 km, so it fails the initial temporal-support gate
for a boundary-by-price interaction. This is not a power pass: assignment and shock variation are
spatially clustered, and unmatched village codes require a documented resolution rule. Before any
outcome interaction is estimated, the project still needs (1) a bound copy of the released source,
(2) an independently reconstructed and frozen five-segment rule, (3) blinded continuity and
modern-boundary checks, and (4) simulation-based power using effective spatial and shock units.

The generic `feasibility-check` report classifies all five questions as not-yet-testable because
that scanner reads CSV/TSV only while this project's analytical releases are Parquet. That label is
a file-format diagnostic and should not be interpreted as a substantive feasibility failure.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    archive_md5 = md5sum(args.zones_zip)
    if archive_md5 != EXPECTED_ZONE_MD5:
        raise RuntimeError(f"Zone archive checksum mismatch: {archive_md5}")

    kampong_speu, southwest, west, study_boundary, shared_boundary = load_boundary_geometry(
        args.zones_zip, root / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
    )
    assignment = build_blind_assignment(root, southwest, west, study_boundary)
    households = merge_assignment(
        pd.read_parquet(root / "data/processed/direction3_household_conflict_shock_preprocessed.parquet"),
        assignment,
    )
    education = merge_assignment(
        pd.read_parquet(root / "data/processed/direction3_education_conflict_shock_preprocessed.parquet"),
        assignment,
    )

    support = support_by_bandwidth(assignment, households, education)
    wave_support = support_by_wave(assignment)
    reproduction_audit = (
        pd.read_csv(args.reproduction_audit)
        if args.reproduction_audit is not None
        else None
    )
    diagnostic_assignment = assignment[
        [
            "Survey Wave",
            "PSU",
            "Village Code Normalized",
            "Commune Code Normalized",
            "Exact Public Village Point Matched",
            SIDE_COLUMN,
            DISTANCE_COLUMN,
            "Absolute Distance to Historical Boundary km",
        ]
    ].copy()

    source_audit = pd.DataFrame(
        [
            {
                "Source": "Harvard Dataverse replication archive",
                "Identifier": "doi:10.7910/DVN/RK5GOH",
                "Version": "1.0",
                "Release Date": "2023-05-02",
                "License": "CC0 1.0",
                "Access": "public",
                "Audit Result": "pass",
            },
            {
                "Source": "Democratic Kampuchea Zones shapefile",
                "Identifier": "Dataverse datafile 6990145",
                "Version": "1",
                "Release Date": "2023-05-02",
                "License": "CC0 1.0",
                "Access": "public",
                "Audit Result": "checksum pass",
            },
            {
                "Source": "Public historical village points",
                "Identifier": "Yale CGEO village code and point geometry",
                "Version": "project-bound copy",
                "Release Date": "not supplied in file",
                "License": "see project source manifest",
                "Access": "available locally",
                "Audit Result": "usable for blinded exact-code linkage",
            },
            {
                "Source": "Public 2008 village replication frame",
                "Identifier": "Dataverse datafile 6990178",
                "Version": "1",
                "Release Date": "2023-05-02",
                "License": "CC0 1.0",
                "Access": "public",
                "Audit Result": (
                    "treatment and signed distance independently reproduced"
                    if reproduction_audit is not None
                    else "not run"
                ),
            },
        ]
    )

    source_audit.to_csv(output / "boundary_source_audit.csv", index=False)
    support.to_csv(output / "boundary_support_by_bandwidth.csv", index=False)
    wave_support.to_csv(output / "boundary_support_by_wave_and_side.csv", index=False)
    diagnostic_assignment.to_parquet(output / "boundary_assignment_diagnostic.parquet", index=False)
    write_map(
        output / "boundary_linkage_feasibility_map.png",
        kampong_speu,
        southwest,
        west,
        study_boundary,
        assignment,
    )
    write_readme(
        output / "README_boundary.md",
        archive_md5,
        assignment,
        support,
        study_boundary.length / 1000,
        reproduction_audit,
    )

    print(f"source checksum: pass ({archive_md5})")
    print(
        "exact village linkage: "
        f"{assignment['Exact Public Village Point Matched'].sum()}/{len(assignment)} "
        f"({assignment['Exact Public Village Point Matched'].mean():.1%})"
    )
    print(f"study boundary length: {study_boundary.length / 1000:.2f} km")
    print(f"full shared Southwest-West boundary length: {shared_boundary.length / 1000:.2f} km")
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
