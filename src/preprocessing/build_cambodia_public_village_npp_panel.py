#!/usr/bin/env python3
"""Build a nationwide public-village NPP and climate panel for Cambodia.

Unlike the CSES-linked release, this panel is not restricted to villages sampled
by a household survey.  All public CGEO village points are assigned to the
nearest Cambodia land-grid cell for administrative context, then summarized at
2, 5, and 10 km.  CSES is therefore a nested socioeconomic validation sample,
not the sampling frame of the ecological analysis.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
RAW_POINTS = ROOT / "data/raw/conflict/yale_cgeo_historical_villages.geojson"
GRID = ROOT / "data/processed/cambodia_national_1km_grid_preprocessed.parquet"
ANNUAL = ROOT / "data/processed/cambodia_national_annual_satellite_climate_panel_preprocessed.parquet"
STATIC = ROOT / "data/processed/cambodia_national_predetermined_covariates_preprocessed.parquet"
EVENTS = ROOT / "data/processed/ucdp_cambodia_thailand_state_conflict_candidates_preprocessed.parquet"
CSES_POINTS = ROOT / "data/processed/cses_village_points_preprocessed.parquet"

POINTS_OUT = ROOT / "data/processed/cambodia_public_village_points_preprocessed.parquet"
MEMBERSHIP_DIR = ROOT / "data/processed/cambodia_public_village_buffer_grid_crosswalk"
PANEL_OUT = ROOT / "data/processed/cambodia_public_village_npp_conflict_panel_candidate_preprocessed.parquet"
CSES_CROSSWALK_OUT = ROOT / "data/processed/cses_to_cambodia_public_village_point_crosswalk_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/cambodia-public-village-npp"

RADII_KM = (2, 5, 10)
NPP_START_YEAR = 2001
NPP_END_YEAR = 2024

SECTOR_RULES = {
    "Preah Vihear": {
        "province_aliases": {"Preah Vihear"},
        "district_aliases": {"Choam Khsant", "Choam Ksant"},
        "first_date": pd.Timestamp("2008-10-15"),
    },
    "Ta Moan-Ta Krabey": {
        "province_aliases": {"Otdar Meanchey", "Oddar Meanchey"},
        "district_aliases": {"Banteay Ampil"},
        "first_date": pd.Timestamp("2011-04-22"),
    },
}

ANNUAL_VARIABLES = [
    "Annual Land NPP Mean kg C per m2",
    "Mean NPP QC Filled Growing-Season Days Percent",
    "Annual Land NPP Anomaly kg C per m2",
    "Annual Land NPP Anomaly Z 2001-2020",
    "NPP Complete 2001-2020 Baseline",
    "Annual Precipitation Total mm",
    "Annual Mean Daily Maximum Temperature C",
    "May October Precipitation Total mm",
    "May October Maximum Consecutive Dry Days Anomaly Z",
    "May October Mean Daily Maximum Temperature C Anomaly Z",
    "May October Dry Rainfall Intensity",
    "May October Heat Intensity",
    "May October Compound Hot-Dry Intensity",
]

STATIC_VARIABLES = [
    "Distance to Cambodia Thailand Border km",
    "Mean Elevation m",
    "Mean Slope Degrees",
    "Distance to Road Excluding Post 2007 AidData Corridors km",
    "Baseline Cropland Share",
    "Baseline Forest Share",
    "Baseline Population Density per km2",
    "Log Baseline Population 2000",
]


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def normalize_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())


def build_points(grid: pd.DataFrame, events: pd.DataFrame) -> gpd.GeoDataFrame:
    raw = gpd.read_file(RAW_POINTS).to_crs(4326).reset_index(drop=True)
    projected = raw.to_crs(32648)
    grid_tree = cKDTree(grid[["Grid Centre Easting m", "Grid Centre Northing m"]].to_numpy(float))
    grid_distance, grid_index = grid_tree.query(np.column_stack([projected.geometry.x, projected.geometry.y]))
    admin = grid.iloc[grid_index].reset_index(drop=True)

    points = gpd.GeoDataFrame({
        "National Village Point ID": [f"CGEO-{int(v):05d}" for v in raw["OBJECTID"]],
        "Source Village Administrative Code": raw["CODEPHUM"].astype("string"),
        "Public Village Name": raw["PHUM"].astype("string"),
        "Point Longitude": raw.geometry.x.astype(float),
        "Point Latitude": raw.geometry.y.astype(float),
        "Nearest National Grid Cell Distance m": grid_distance.astype("float32"),
        "National Grid Cell ID": admin["National Grid Cell ID"].astype("string"),
        "Province Code": admin["Province Code"].astype("string"),
        "Province Name": admin["Province Name"].astype("string"),
        "District Code": admin["District Code"].astype("string"),
        "District Name": admin["District Name"].astype("string"),
        "Commune Code": admin["Commune Code"].astype("string"),
        "Commune Name": admin["Commune Name"].astype("string"),
    }, geometry=raw.geometry, crs=4326)
    points["Administrative Assignment Within 2 km"] = (grid_distance <= 2000).astype("int8")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    event_e, event_n = transformer.transform(events["longitude"].to_numpy(), events["latitude"].to_numpy())
    point_e, point_n = transformer.transform(points["Point Longitude"].to_numpy(), points["Point Latitude"].to_numpy())
    distance = np.sqrt(
        (point_e[:, None] - event_e[None, :]) ** 2
        + (point_n[:, None] - event_n[None, :]) ** 2
    ) / 1000.0
    nearest = distance.argmin(axis=1)
    points["Candidate Nearest UCDP Event Distance km"] = distance.min(axis=1).astype("float32")
    points["Candidate Nearest UCDP Event ID"] = events.iloc[nearest]["id"].to_numpy(dtype="int64")

    points["Candidate Affected District"] = np.int8(0)
    points["Candidate Affected Province"] = np.int8(0)
    points["Candidate Conflict Sector"] = pd.Series(pd.NA, index=points.index, dtype="string")
    points["Candidate First Conflict Date"] = pd.NaT
    points["Candidate Treatment Evidence Grade"] = "Outside documented affected administrative areas"
    for sector, rule in SECTOR_RULES.items():
        province = points["Province Name"].isin(rule["province_aliases"])
        district = province & points["District Name"].isin(rule["district_aliases"])
        points.loc[province, "Candidate Affected Province"] = 1
        points.loc[district, "Candidate Affected District"] = 1
        points.loc[district, "Candidate Conflict Sector"] = sector
        points.loc[district, "Candidate First Conflict Date"] = rule["first_date"]
        points.loc[province, "Candidate Treatment Evidence Grade"] = (
            "D: affected province documented; too coarse for primary treatment"
        )
        points.loc[district, "Candidate Treatment Evidence Grade"] = (
            "C: event-containing affected district; origin villages not enumerated"
        )
    points["Candidate First Conflict Year"] = points["Candidate First Conflict Date"].dt.year.astype("Int16")
    points["Candidate District Treatment Is Frozen"] = np.int8(0)
    return points


def write_membership(points: gpd.GeoDataFrame, grid: pd.DataFrame) -> dict[int, int]:
    MEMBERSHIP_DIR.mkdir(parents=True, exist_ok=True)
    projected = points.to_crs(32648)
    point_xy = np.column_stack([projected.geometry.x, projected.geometry.y])
    grid_xy = grid[["Grid Centre Easting m", "Grid Centre Northing m"]].to_numpy(float)
    tree = cKDTree(grid_xy)
    counts: dict[int, int] = {}
    for radius in RADII_KM:
        neighbours = tree.query_ball_point(point_xy, radius * 1000)
        lengths = np.fromiter((len(x) for x in neighbours), dtype=np.int32, count=len(neighbours))
        if (lengths == 0).any():
            raise RuntimeError(f"Some public village points have no land-grid cell within {radius} km")
        point_index = np.repeat(np.arange(len(points), dtype=np.int32), lengths)
        grid_index = np.concatenate(neighbours).astype(np.int32)
        membership = pd.DataFrame({
            "National Village Point ID": points["National Village Point ID"].to_numpy()[point_index],
            "Buffer Radius km": np.int16(radius),
            "National Grid Cell ID": grid["National Grid Cell ID"].astype("string").to_numpy()[grid_index],
        })
        path = MEMBERSHIP_DIR / f"radius_{radius}_km.parquet"
        membership.to_parquet(path, index=False)
        counts[radius] = int(len(membership))
    return counts


def build_cses_crosswalk(points: pd.DataFrame) -> pd.DataFrame:
    cses = pd.read_parquet(CSES_POINTS)
    public = points[[
        "National Village Point ID", "Public Village Name", "Point Longitude", "Point Latitude"
    ]].copy()
    for frame in [cses, public]:
        frame["Coordinate Longitude 7dp"] = frame["Point Longitude"].round(7)
        frame["Coordinate Latitude 7dp"] = frame["Point Latitude"].round(7)
        frame["Public Village Name Normalized"] = frame["Public Village Name"].map(normalize_name)
    keys = [
        "Coordinate Longitude 7dp", "Coordinate Latitude 7dp", "Public Village Name Normalized"
    ]
    public_key = public[[*keys, "National Village Point ID"]].sort_values(
        "National Village Point ID"
    )
    public_key["Identical Public Point Record Count"] = public_key.groupby(
        keys, dropna=False
    )["National Village Point ID"].transform("size").astype("int16")
    public_key = public_key.drop_duplicates(keys, keep="first")
    output = cses.merge(public_key, on=keys, how="left", validate="many_to_one")
    output["National Public Village Point Matched"] = output[
        "National Village Point ID"
    ].notna().astype("int8")
    output = output[[
        "Village Code", "Public Village Point Matched", "National Public Village Point Matched",
        "National Village Point ID", "Identical Public Point Record Count", "Public Village Name",
        "Point Longitude", "Point Latitude",
    ]]
    if output["Village Code"].duplicated().any():
        raise RuntimeError("CSES-to-public-village crosswalk is not unique by Village Code")
    output.to_parquet(CSES_CROSSWALK_OUT, index=False)
    return output


def aggregate_panel() -> None:
    annual_means = ",\n            ".join(
        f"avg(a.{quote(v)}) AS {quote('Village Buffer Mean ' + v)}" for v in ANNUAL_VARIABLES
    )
    static_means = ",\n            ".join(
        f"avg(s.{quote(v)}) AS {quote('Village Buffer Mean ' + v)}" for v in STATIC_VARIABLES
    )
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET threads = 4")
    con.execute(f"""
      COPY (
        WITH annual_buffer AS (
          SELECT m."National Village Point ID", m."Buffer Radius km", a."Year",
            count(*) AS "Village Buffer Grid Cell Count",
            {annual_means}
          FROM read_parquet('{q(MEMBERSHIP_DIR)}/*.parquet') m
          INNER JOIN read_parquet('{q(ANNUAL)}') a USING ("National Grid Cell ID")
          WHERE a."Year" BETWEEN {NPP_START_YEAR} AND {NPP_END_YEAR}
          GROUP BY 1, 2, 3
        ),
        static_buffer AS (
          SELECT m."National Village Point ID", m."Buffer Radius km",
            {static_means}
          FROM read_parquet('{q(MEMBERSHIP_DIR)}/*.parquet') m
          INNER JOIN read_parquet('{q(STATIC)}') s USING ("National Grid Cell ID")
          GROUP BY 1, 2
        )
        SELECT a.*, s.* EXCLUDE ("National Village Point ID", "Buffer Radius km"),
          p.* EXCLUDE ("National Village Point ID", geometry),
          CASE
            WHEN p."Candidate Affected District" = 1 AND a."Year" > p."Candidate First Conflict Year" THEN 1
            WHEN p."Candidate Affected District" = 1 AND a."Year" = p."Candidate First Conflict Year" THEN NULL
            ELSE 0
          END::TINYINT AS "Candidate Post Conflict Period",
          CASE WHEN p."Candidate Affected District" = 1
            THEN a."Year" - p."Candidate First Conflict Year" ELSE NULL END::SMALLINT
            AS "Candidate Event Time Year"
        FROM annual_buffer a
        LEFT JOIN static_buffer s USING ("National Village Point ID", "Buffer Radius km")
        INNER JOIN read_parquet('{q(POINTS_OUT)}') p USING ("National Village Point ID")
        ORDER BY a."National Village Point ID", a."Buffer Radius km", a."Year"
      ) TO '{q(PANEL_OUT)}'
      (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)
    con.close()


def write_audit(
    points: pd.DataFrame,
    membership_counts: dict[int, int],
    panel: pd.DataFrame,
    cses_crosswalk: pd.DataFrame,
) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    treatment = (
        points.groupby(["Candidate Conflict Sector", "Candidate Treatment Evidence Grade"], dropna=False)
        .agg(
            Villages=("National Village Point ID", "nunique"),
            Provinces=("Province Name", "nunique"),
            Districts=("District Name", "nunique"),
        )
        .reset_index()
    )
    treatment.to_csv(AUDIT_DIR / "candidate_treatment_public_village_counts.csv", index=False)
    support = (
        panel.loc[panel["Buffer Radius km"].eq(5)]
        .groupby(["Candidate Conflict Sector", "Candidate Affected District"], dropna=False)
        .agg(
            Villages=("National Village Point ID", "nunique"),
            First_Year=("Year", "min"),
            Last_Year=("Year", "max"),
            Village_Years=("Year", "size"),
            NPP_Nonmissing=("Village Buffer Mean Annual Land NPP Anomaly Z 2001-2020", "count"),
        )
        .reset_index()
    )
    support.to_csv(AUDIT_DIR / "npp_support_by_candidate_sector.csv", index=False)
    metadata = {
        "public_village_points": int(len(points)),
        "administrative_assignment_within_2km": int(points["Administrative Assignment Within 2 km"].sum()),
        "candidate_affected_district_villages": int(points["Candidate Affected District"].sum()),
        "membership_rows_by_radius": membership_counts,
        "panel_rows": int(len(panel)),
        "cses_village_codes_linked_to_national_public_points": int(
            cses_crosswalk["National Public Village Point Matched"].sum()
        ),
        "panel_years": [int(panel["Year"].min()), int(panel["Year"].max())],
        "treatment_status": "candidate only; not frozen",
        "ecological_sampling_rule": "all public CGEO village points, independent of CSES sampling",
        "outputs": {
            "points": str(POINTS_OUT.relative_to(ROOT)),
            "membership": str(MEMBERSHIP_DIR.relative_to(ROOT)),
            "panel": str(PANEL_OUT.relative_to(ROOT)),
            "cses_crosswalk": str(CSES_CROSSWALK_OUT.relative_to(ROOT)),
        },
    }
    (AUDIT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    decisions = {
        "primary_buffer_km": 5,
        "sensitivity_buffers_km": [2, 10],
        "outcome": "Village Buffer Mean Annual Land NPP Anomaly Z 2001-2020",
        "primary_climate_shock": "Village Buffer Mean May October Dry Rainfall Intensity",
        "treatment": "candidate affected district for support audit only; origin-village treatment remains required",
        "cses_role": "nested socioeconomic validation sample, not ecological sampling frame",
        "missing_data": "no imputation",
    }
    (AUDIT_DIR / "decisions.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme = f"""# Cambodia public-village NPP preprocessing

- Public village points: {len(points):,}
- Candidate affected-district village points: {int(points['Candidate Affected District'].sum()):,}
- CSES village codes nested in the national point frame: {int(cses_crosswalk['National Public Village Point Matched'].sum()):,}
- Village-buffer-year rows, 2001-2024: {len(panel):,}
- Primary radius: 5 km; mandatory sensitivities: 2 and 10 km

This is the national ecological sampling frame. It is independent of CSES survey
selection. Candidate district treatment remains provisional because the public
humanitarian sources do not enumerate all affected origin villages.
"""
    (AUDIT_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(treatment.to_string(index=False))
    print(support.loc[support["Candidate Affected District"].eq(1)].to_string(index=False))


def main() -> None:
    for path in [RAW_POINTS, GRID, ANNUAL, STATIC, EVENTS, CSES_POINTS]:
        if not path.exists():
            raise FileNotFoundError(path)
    grid = pd.read_parquet(GRID, columns=[
        "National Grid Cell ID", "Grid Centre Easting m", "Grid Centre Northing m",
        "Province Code", "Province Name", "District Code", "District Name",
        "Commune Code", "Commune Name",
    ])
    events = pd.read_parquet(EVENTS).loc[lambda x: x["year"].between(2008, 2011)].copy()
    points = build_points(grid, events)
    POINTS_OUT.parent.mkdir(parents=True, exist_ok=True)
    points.to_parquet(POINTS_OUT, index=False)
    cses_crosswalk = build_cses_crosswalk(points.drop(columns="geometry"))
    membership_counts = write_membership(points, grid)
    aggregate_panel()
    panel = pd.read_parquet(PANEL_OUT)
    key = ["National Village Point ID", "Buffer Radius km", "Year"]
    if panel.duplicated(key).any():
        raise RuntimeError("Nationwide public-village panel contains duplicate keys")
    expected = len(points) * len(RADII_KM) * (NPP_END_YEAR - NPP_START_YEAR + 1)
    if len(panel) != expected:
        raise RuntimeError(f"Expected {expected:,} rows, found {len(panel):,}")
    write_audit(points.drop(columns="geometry"), membership_counts, panel, cses_crosswalk)


if __name__ == "__main__":
    main()
