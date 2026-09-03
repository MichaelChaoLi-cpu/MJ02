#!/usr/bin/env python3
"""Link CSES villages to public points and summarize national grids in buffers.

The script produces an explicit village-to-grid-cell crosswalk for 2, 5, and
10 km circular buffers.  Annual satellite-climate variables and predetermined
geographic variables are averaged over the same included Cambodia land cells.
Unmatched CSES villages remain in the final village-year release with missing
satellite fields and an explicit linkage status.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree


CSES = Path("data/processed/cses_mda_village_year_preprocessed.parquet")
HOUSEHOLDS = Path("data/processed/direction3_household_core_outcomes_preprocessed.parquet")
PUBLIC_POINTS = Path("data/raw/conflict/yale_cgeo_historical_villages.geojson")
GRID = Path("data/processed/cambodia_national_1km_grid_preprocessed.parquet")
ANNUAL = Path("data/processed/cambodia_national_annual_satellite_climate_panel_preprocessed.parquet")
STATIC = Path("data/processed/cambodia_national_predetermined_covariates_preprocessed.parquet")
CCNL = Path("data/processed/cambodia_national_ccnl_dmsp_annual_preprocessed.parquet")
FLOOD_2011 = Path("data/processed/cambodia_national_2011_gfd_flood_exposure_preprocessed.parquet")

POINTS_OUTPUT = Path("data/processed/cses_village_points_preprocessed.parquet")
MEMBERSHIP_OUTPUT = Path("data/processed/cses_village_buffer_grid_crosswalk_preprocessed.parquet")
ANNUAL_OUTPUT = Path("data/processed/cses_village_buffer_annual_satellite_preprocessed.parquet")
STATIC_OUTPUT = Path("data/processed/cses_village_buffer_static_geography_preprocessed.parquet")
MERGED_OUTPUT = Path("data/processed/cses_village_year_satellite_preprocessed.parquet")
AUDIT_DIR = Path("data/exp/data-preprocessing/cses-village-buffer-satellite")

DEFAULT_RADII_KM = (2, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--radii-km", type=int, nargs="+", default=DEFAULT_RADII_KM)
    return parser.parse_args()


def normalize_code(values: pd.Series, width: int) -> pd.Series:
    result = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return result.where(result.notna() & result.ne("")).str.zfill(width)


def derive_public_commune_code(values: pd.Series) -> pd.Series:
    """Recover six-digit commune codes from the historical CGEO field.

    The source stores a seven-character administrative code. Provinces 1-9
    lost their leading zero, whereas provinces 10 and above already start with
    two province digits. Blindly applying ``zfill(8)`` therefore shifted every
    two-digit province into the wrong province and prevented all such CSES
    villages from matching.
    """
    raw = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    raw = raw.where(raw.notna() & raw.ne(""))
    first_two = pd.to_numeric(raw.str[:2], errors="coerce")
    two_digit_province = first_two.between(10, 25)
    return raw.str.zfill(8).str[:6].where(~two_digit_province, raw.str[:6])


def normalize_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    ascii_text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", ascii_text.lower())


def resolve_village_points(cses: pd.DataFrame, public_path: Path) -> gpd.GeoDataFrame:
    candidates = cses.loc[cses["Village Code"].notna(), [
        "Village Code", "Commune Code", "Province Name", "District Name",
        "Commune Name", "Village Name",
    ]].drop_duplicates().copy()
    candidates["Village Name Normalized"] = candidates["Village Name"].map(normalize_name)

    public = gpd.read_file(public_path).to_crs(4326)
    public["Public Village Code"] = normalize_code(public["CODEPHUM"], 8)
    public["Public Commune Code"] = derive_public_commune_code(public["CODEPHUM"])
    public["Public Village Name"] = public["PHUM"].astype("string")
    public["Village Name Normalized"] = public["PHUM"].map(normalize_name)
    public["Point Longitude"] = public.geometry.x
    public["Point Latitude"] = public.geometry.y
    public = public[[
        "Public Village Code", "Public Commune Code", "Public Village Name",
        "Village Name Normalized", "Point Longitude", "Point Latitude", "geometry",
    ]]

    code_count = public.groupby("Public Village Code", dropna=False).size()
    unique_code = public.loc[
        public["Public Village Code"].map(code_count).eq(1)
    ].set_index("Public Village Code")
    pair_count = public.groupby(
        ["Public Village Code", "Village Name Normalized"], dropna=False
    ).size()
    unique_pair_keys = pair_count[pair_count.eq(1)].index
    unique_pair = public.set_index(
        ["Public Village Code", "Village Name Normalized"]
    ).loc[unique_pair_keys]
    commune_name_count = public.groupby(
        ["Public Commune Code", "Village Name Normalized"], dropna=False
    ).size()
    unique_commune_name_keys = commune_name_count[commune_name_count.eq(1)].index
    unique_commune_name = public.set_index(
        ["Public Commune Code", "Village Name Normalized"]
    ).loc[unique_commune_name_keys]

    resolved_rows: list[dict[str, object]] = []
    for village_code, group in candidates.groupby("Village Code", sort=True):
        matches: list[tuple[str, pd.Series]] = []
        if village_code in unique_code.index:
            matches.append(("unique exact public village code", unique_code.loc[village_code]))
        else:
            for _, row in group.iterrows():
                pair = (village_code, row["Village Name Normalized"])
                if pair in unique_pair.index:
                    matches.append(("unique exact public code and normalized name", unique_pair.loc[pair]))
                commune_pair = (row["Commune Code"], row["Village Name Normalized"])
                if commune_pair in unique_commune_name.index:
                    matches.append(("unique exact normalized name within commune", unique_commune_name.loc[commune_pair]))
        coordinates = {
            (round(float(row["Point Longitude"]), 7), round(float(row["Point Latitude"]), 7))
            for _, row in matches
        }
        base = group.iloc[0]
        record: dict[str, object] = {
            "Village Code": village_code,
            "Commune Code": base["Commune Code"],
            "Province Name": base["Province Name"],
            "District Name": base["District Name"],
            "Commune Name": base["Commune Name"],
            "CSES Village Names Observed": " | ".join(
                sorted(set(group["Village Name"].dropna().astype(str)))
            ),
            "Public Village Point Matched": int(len(coordinates) == 1),
            "Public Village Point Link Method": "unresolved",
            "Public Village Name": pd.NA,
            "Point Longitude": np.nan,
            "Point Latitude": np.nan,
            "geometry": None,
        }
        if len(coordinates) == 1:
            longitude, latitude = next(iter(coordinates))
            strongest = min(matches, key=lambda item: (
                0 if item[0] == "unique exact public village code" else
                1 if item[0] == "unique exact public code and normalized name" else 2
            ))
            record.update({
                "Public Village Point Link Method": strongest[0],
                "Public Village Name": strongest[1]["Public Village Name"],
                "Point Longitude": longitude,
                "Point Latitude": latitude,
                "geometry": gpd.points_from_xy([longitude], [latitude], crs=4326)[0],
            })
        elif len(coordinates) > 1:
            record["Public Village Point Link Method"] = "ambiguous multiple candidate points"
        resolved_rows.append(record)
    return gpd.GeoDataFrame(resolved_rows, geometry="geometry", crs=4326)


def build_linkage_candidates(cses: pd.DataFrame, households: pd.DataFrame) -> pd.DataFrame:
    """Combine village labels from the village release and household main sample.

    The older village release has complete codes but missing administrative names for
    many pre-2019 villages.  The household release retains those names.  Combining the
    two sources allows deterministic within-commune name matching without fuzzy links.
    """
    columns = [
        "Village Code", "Commune Code", "Province Name", "District Name",
        "Commune Name", "Village Name",
    ]
    household_main = households.loc[households["Main Linked Sample"].eq(1), columns]
    return pd.concat([cses[columns], household_main], ignore_index=True).drop_duplicates()


def construct_membership(
    points: gpd.GeoDataFrame, grid: pd.DataFrame, radii_km: list[int]
) -> pd.DataFrame:
    matched = points.loc[points["Public Village Point Matched"].eq(1)].to_crs(32648).copy()
    tree = cKDTree(grid[["Grid Centre Easting m", "Grid Centre Northing m"]].to_numpy(float))
    records: list[pd.DataFrame] = []
    for radius_km in radii_km:
        neighbors = tree.query_ball_point(
            np.column_stack([matched.geometry.x, matched.geometry.y]), radius_km * 1000
        )
        for village_code, point, indices in zip(
            matched["Village Code"], matched.geometry, neighbors, strict=True
        ):
            if not indices:
                continue
            selected = grid.iloc[np.asarray(indices, dtype=int)]
            distance = np.hypot(
                selected["Grid Centre Easting m"].to_numpy(float) - point.x,
                selected["Grid Centre Northing m"].to_numpy(float) - point.y,
            ) / 1000
            records.append(pd.DataFrame({
                "Village Code": village_code,
                "Buffer Radius km": np.int16(radius_km),
                "National Grid Cell ID": selected["National Grid Cell ID"].astype("string").to_numpy(),
                "Grid Cell Centre Distance to Village km": distance.astype("float32"),
            }))
    membership = pd.concat(records, ignore_index=True)
    if membership.duplicated(["Village Code", "Buffer Radius km", "National Grid Cell ID"]).any():
        raise RuntimeError("Duplicate village-buffer-grid membership key")
    return membership


def sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def numeric_columns(path: Path, exclusions: set[str]) -> list[str]:
    schema = pq.ParquetFile(path).schema_arrow
    numeric = []
    for field in schema:
        if field.name in exclusions:
            continue
        if str(field.type).startswith(("int", "uint", "float", "double", "decimal")):
            numeric.append(field.name)
    return numeric


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def aggregate_annual(root: Path, annual_columns: list[str]) -> None:
    membership = root / MEMBERSHIP_OUTPUT
    annual = root / ANNUAL
    output = root / ANNUAL_OUTPUT
    averages = ",\n          ".join(
        f"avg(a.{quote(column)}) AS {quote('Buffer Mean ' + column)}"
        for column in annual_columns
    )
    core = {
        "Annual Land NPP Mean kg C per m2": "NPP",
        "Annual NPP-VIIRS-like Radiance": "LongNTL",
        "Annual Mean Daily Maximum Temperature C": "Annual Maximum Temperature",
        "Annual Precipitation Total mm": "Annual Precipitation",
    }
    diagnostics = ",\n          ".join(
        f"count(a.{quote(column)}) AS {quote('Valid ' + label + ' Grid Cell Count')}, "
        f"stddev_samp(a.{quote(column)}) AS {quote('Buffer SD ' + column)}"
        for column, label in core.items()
    )
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"""
      COPY (
        SELECT
          m."Village Code", m."Buffer Radius km", a."Year",
          count(*) AS "Buffer Grid Cell Count",
          max(m."Grid Cell Centre Distance to Village km") AS "Maximum Included Grid Cell Centre Distance km",
          {diagnostics},
          {averages}
        FROM read_parquet('{sql_path(membership)}') AS m
        INNER JOIN read_parquet('{sql_path(annual)}') AS a USING ("National Grid Cell ID")
        GROUP BY m."Village Code", m."Buffer Radius km", a."Year"
        ORDER BY m."Village Code", m."Buffer Radius km", a."Year"
      ) TO '{sql_path(output)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)
    con.close()


def aggregate_static(root: Path, static_columns: list[str]) -> None:
    membership = root / MEMBERSHIP_OUTPUT
    static = root / STATIC
    output = root / STATIC_OUTPUT
    averages = ",\n          ".join(
        f"avg(s.{quote(column)}) AS {quote('Buffer Mean ' + column)}"
        for column in static_columns
    )
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"""
      COPY (
        SELECT m."Village Code", m."Buffer Radius km", {averages}
        FROM read_parquet('{sql_path(membership)}') AS m
        INNER JOIN read_parquet('{sql_path(static)}') AS s USING ("National Grid Cell ID")
        GROUP BY m."Village Code", m."Buffer Radius km"
        ORDER BY m."Village Code", m."Buffer Radius km"
      ) TO '{sql_path(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    con.close()


def append_buffer_source(
    root: Path,
    source_path: Path,
    target_path: Path,
    numeric: list[str],
    annual: bool,
) -> None:
    """Append another cell-level source summarized over the frozen memberships."""
    membership = root / MEMBERSHIP_OUTPUT
    source = root / source_path
    target = root / target_path
    temporary = target.with_suffix(".append.parquet")
    averages = ",\n            ".join(
        f"avg(s.{quote(column)}) AS {quote('Buffer Mean ' + column)}" for column in numeric
    )
    year_select = ', s."Year"' if annual else ""
    year_group = ', s."Year"' if annual else ""
    join_keys = '"Village Code", "Buffer Radius km", "Year"' if annual else '"Village Code", "Buffer Radius km"'
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"""
      COPY (
        WITH extra AS (
          SELECT m."Village Code", m."Buffer Radius km"{year_select},
            {averages}
          FROM read_parquet('{sql_path(membership)}') AS m
          INNER JOIN read_parquet('{sql_path(source)}') AS s USING ("National Grid Cell ID")
          GROUP BY m."Village Code", m."Buffer Radius km"{year_group}
        )
        SELECT t.*, e.* EXCLUDE ({join_keys})
        FROM read_parquet('{sql_path(target)}') AS t
        LEFT JOIN extra AS e USING ({join_keys})
        ORDER BY t."Village Code", t."Buffer Radius km"{', t."Year"' if annual else ''}
      ) TO '{sql_path(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)
    con.close()
    temporary.replace(target)


def merge_village_wave(root: Path, points: gpd.GeoDataFrame, radii_km: list[int]) -> pd.DataFrame:
    cses = pd.read_parquet(root / CSES)
    radii = pd.DataFrame({"Buffer Radius km": np.asarray(radii_km, dtype="int16")})
    cses["_cross"] = 1
    radii["_cross"] = 1
    release = cses.merge(radii, on="_cross", how="outer").drop(columns="_cross")
    point_fields = points.drop(columns="geometry").copy()
    release = release.merge(
        point_fields, on="Village Code", how="left", validate="many_to_one",
        suffixes=("", " Public Point")
    )
    release["Public Village Point Matched"] = (
        release["Public Village Point Matched"].fillna(0).astype("Int8")
    )
    release["Public Village Point Link Method"] = release[
        "Public Village Point Link Method"
    ].fillna("no spatial village code")
    annual = pd.read_parquet(root / ANNUAL_OUTPUT)
    static = pd.read_parquet(root / STATIC_OUTPUT)
    release = release.merge(
        annual,
        left_on=["Village Code", "Buffer Radius km", "Survey Year"],
        right_on=["Village Code", "Buffer Radius km", "Year"],
        how="left", validate="many_to_one",
    ).drop(columns="Year")
    release = release.merge(
        static, on=["Village Code", "Buffer Radius km"], how="left", validate="many_to_one"
    )
    if release.duplicated(["Survey Wave", "Village Code", "Buffer Radius km"]).any():
        raise RuntimeError("Final CSES village-year-buffer key is not unique")
    return release.sort_values(["Survey Year", "Village Code", "Buffer Radius km"]).reset_index(drop=True)


def write_audits(
    root: Path,
    cses: pd.DataFrame,
    points: gpd.GeoDataFrame,
    membership: pd.DataFrame,
    release: pd.DataFrame,
    radii_km: list[int],
) -> None:
    audit = root / AUDIT_DIR
    audit.mkdir(parents=True, exist_ok=True)
    linkage = points.groupby("Public Village Point Link Method", dropna=False).agg(
        **{"Unique CSES Village Codes": ("Village Code", "nunique")}
    ).reset_index()
    linkage.to_csv(audit / "village_point_linkage_methods.csv", index=False)
    unmatched = points.loc[points["Public Village Point Matched"].eq(0)].drop(columns="geometry")
    unmatched.to_csv(audit / "unmatched_cses_village_points.csv", index=False)
    buffer_coverage = membership.groupby("Buffer Radius km", observed=True).agg(
        **{
            "Matched Villages With Grid Cells": ("Village Code", "nunique"),
            "Village-Grid Membership Rows": ("National Grid Cell ID", "size"),
            "Minimum Grid Cells per Village": ("Village Code", lambda x: int(x.value_counts().min())),
            "Median Grid Cells per Village": ("Village Code", lambda x: float(x.value_counts().median())),
            "Maximum Grid Cells per Village": ("Village Code", lambda x: int(x.value_counts().max())),
        }
    ).reset_index()
    buffer_coverage.to_csv(audit / "buffer_grid_coverage.csv", index=False)
    survey_coverage = release.groupby(
        ["Survey Year", "Buffer Radius km"], observed=True
    ).agg(**{
        "Village-Year Rows": ("Village Code", "size"),
        "Rows With Public Village Point": ("Public Village Point Matched", "sum"),
        "Rows With NPP": ("Buffer Mean Annual Land NPP Mean kg C per m2", "count"),
        "Rows With LongNTL": ("Buffer Mean Annual NPP-VIIRS-like Radiance", "count"),
        "Rows With Temperature": ("Buffer Mean Annual Mean Daily Maximum Temperature C", "count"),
    }).reset_index()
    survey_coverage.to_csv(audit / "survey_wave_satellite_coverage.csv", index=False)
    variable_dictionary = pd.DataFrame({
        "Variable": release.columns,
        "Data Type": [str(release[column].dtype) for column in release.columns],
        "Missing Percent": [float(release[column].isna().mean() * 100) for column in release.columns],
        "Source Family": [
            "buffer annual satellite-climate" if column.startswith(("Buffer Mean Annual", "Buffer Mean May", "Buffer SD Annual", "Valid "))
            else "buffer predetermined geography" if column.startswith("Buffer Mean")
            else "village point linkage" if "Point" in column or column.startswith("Public Village")
            else "CSES village-year"
            for column in release.columns
        ],
    })
    variable_dictionary.to_csv(audit / "variable_dictionary.csv", index=False)
    metadata = {
        "outputs": {
            "village_points": str(POINTS_OUTPUT),
            "village_grid_crosswalk": str(MEMBERSHIP_OUTPUT),
            "annual_village_buffer_satellite": str(ANNUAL_OUTPUT),
            "static_village_buffer_geography": str(STATIC_OUTPUT),
            "cses_village_year_satellite": str(MERGED_OUTPUT),
        },
        "buffer_radii_km": radii_km,
        "buffer_rule": "Circular EPSG:32648 point buffers represented by Cambodia 1 km land-grid cell centres whose distance to the village point does not exceed the radius",
        "annual_years": [2000, 2024],
        "npp_years": [2001, 2024],
        "public_point_source": str(PUBLIC_POINTS),
        "additional_satellite_sources": [str(CCNL), str(FLOOD_2011)],
        "cses_village_year_rows": int(len(cses)),
        "cses_unique_spatial_village_codes": int(cses["Village Code"].nunique()),
        "matched_unique_village_codes": int(points["Public Village Point Matched"].sum()),
        "final_village_year_buffer_rows": int(len(release)),
        "missing_rules": [
            "No fuzzy village-name matching",
            "Historical CGEO administrative codes are parsed with separate rules for one- and two-digit provinces",
            "Ambiguous or unmatched village points remain missing",
            "No missing satellite or CSES value imputation",
            "Only Cambodia land-grid cells are included in buffer means",
        ],
        "interpretation": "CSES is repeated cross-sectional; the annual satellite output is a village-buffer panel, while the merged CSES output is observed village-year by buffer radius",
    }
    (audit / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    decisions = {
        "datasets": {
            str(CSES): {
                "grain": "one observed CSES village code by survey wave after combining multiple PSUs",
                "operations": [
                    "database-side aggregation from final_VL/HH/HL/ED/EC/HO_CSES",
                    "survey-weighted village estimates",
                    "separate retention of village-questionnaire totals",
                    "annual all-items CPI conversion to 2021 riel",
                    "no imputation and no winsorization",
                ],
            },
            str(ANNUAL): {
                "grain": "one mapped CSES village by buffer radius by calendar year",
                "operations": [
                    "mean across explicit Cambodia 1 km land-grid-cell centres within circular buffers",
                    "2, 5, and 10 km radii retained without outcome-based selection",
                    "core-variable valid-cell counts and within-buffer standard deviations retained",
                ],
            },
            str(STATIC): {
                "grain": "one mapped CSES village by buffer radius",
                "operations": [
                    "mean predetermined geography over the identical village-grid membership",
                ],
            },
        },
        "final_output": str(MERGED_OUTPUT),
        "unmatched_policy": "retain CSES rows with missing satellite fields and explicit linkage status",
        "is_final_analysis_variable_selection": "not yet confirmed; release is deliberately broad for design review",
    }
    (audit / "decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme = f"""# CSES village-buffer satellite preprocessing

- CSES village-year rows: {len(cses):,}
- CSES village codes with a spatial identifier: {cses['Village Code'].nunique():,}
- Unique village codes matched to one public point: {int(points['Public Village Point Matched'].sum()):,}
- Buffer radii: {', '.join(map(str, radii_km))} km
- Final village-year-buffer rows: {len(release):,}

The output retains unmatched CSES rows.  Village-questionnaire population totals and
survey-weighted demographic estimates remain separate.  Annual NPP, LongNTL, precipitation,
temperature, and extreme-climate fields are buffer means over the same explicit 1 km cell
membership.  Elevation, slope, historical-road access, baseline land cover, and baseline
population are summarized over those cells as predetermined geography.
"""
    (audit / "README.md").write_text(readme, encoding="utf-8")
    print(linkage.to_string(index=False))
    print(buffer_coverage.to_string(index=False))
    print(survey_coverage.to_string(index=False))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    radii_km = sorted(set(args.radii_km))
    if not radii_km or any(radius <= 0 for radius in radii_km):
        raise ValueError("Buffer radii must be positive")
    required = [CSES, HOUSEHOLDS, PUBLIC_POINTS, GRID, ANNUAL, STATIC, CCNL, FLOOD_2011]
    missing = [str(path) for path in required if not (root / path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs: {missing}")

    cses = pd.read_parquet(root / CSES)
    households = pd.read_parquet(
        root / HOUSEHOLDS,
        columns=[
            "Village Code", "Commune Code", "Province Name", "District Name",
            "Commune Name", "Village Name", "Main Linked Sample",
        ],
    )
    linkage_candidates = build_linkage_candidates(cses, households)
    points = resolve_village_points(linkage_candidates, root / PUBLIC_POINTS)
    (root / POINTS_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    points.to_parquet(root / POINTS_OUTPUT, index=False)

    grid = pd.read_parquet(root / GRID, columns=[
        "National Grid Cell ID", "Grid Centre Easting m", "Grid Centre Northing m",
    ])
    membership = construct_membership(points, grid, radii_km)
    membership.to_parquet(root / MEMBERSHIP_OUTPUT, index=False)

    annual_columns = numeric_columns(
        root / ANNUAL, {"Year", "Longitude", "Latitude"}
    )
    static_columns = numeric_columns(
        root / STATIC,
        {
            "Grid Row", "Grid Column", "Grid Centre Easting m", "Grid Centre Northing m",
            "Grid West m", "Grid South m", "Grid East m", "Grid North m", "Longitude", "Latitude",
            "Nearest Border Longitude", "Nearest Border Latitude",
        },
    )
    aggregate_annual(root, annual_columns)
    aggregate_static(root, static_columns)
    append_buffer_source(
        root, CCNL, ANNUAL_OUTPUT,
        numeric_columns(root / CCNL, {"Year"}), annual=True,
    )
    append_buffer_source(
        root, FLOOD_2011, STATIC_OUTPUT,
        numeric_columns(root / FLOOD_2011, set()), annual=False,
    )
    release = merge_village_wave(root, points, radii_km)
    release.to_parquet(root / MERGED_OUTPUT, index=False)
    write_audits(root, cses, points, membership, release, radii_km)
    print(f"Saved {MERGED_OUTPUT}; rows={len(release):,}; columns={len(release.columns):,}")


if __name__ == "__main__":
    main()
