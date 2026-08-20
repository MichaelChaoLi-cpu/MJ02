#!/usr/bin/env python3
"""Link annual MODIS land productivity to preprocessed CHIRPS shocks by commune-year."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


CLIMATE_COLUMNS = [
    "Climate Geography Code",
    "Year",
    "Annual Rainfall mm",
    "Observed Climate Months",
    "Climate Grid Cell Count",
    "Climate Extraction Method",
    "May October Rainfall mm",
    "Annual Rainfall Anomaly Z (1991-2020)",
    "Annual Rainfall Dry Shock",
    "Annual Rainfall Extreme Wet Shock",
    "May October Rainfall Anomaly Z (1991-2020)",
    "May October Rainfall Dry Shock",
    "May October Rainfall Extreme Wet Shock",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--npp",
        type=Path,
        default=Path(
            "data/processed/historical_boundary_annual_land_productivity_preprocessed.parquet"
        ),
    )
    parser.add_argument(
        "--climate",
        type=Path,
        default=Path(
            "data/processed/chirps_long_baseline_commune_year_preprocessed.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/historical_boundary_annual_spatial_climate_preprocessed.parquet"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("data/exp/data-preprocessing/annual-spatial-panel"),
    )
    return parser.parse_args()


def resolved(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_climate_geography_crosswalk(
    root: Path, npp: pd.DataFrame, valid_climate_codes: set[str]
) -> pd.DataFrame:
    villages = npp[
        ["Village Code", "Commune Code", "Village Name", "Longitude", "Latitude"]
    ].drop_duplicates("Village Code")
    villages["Linked Climate Commune Code"] = villages["Commune Code"].where(
        villages["Commune Code"].isin(valid_climate_codes)
    )
    villages["Linked Climate Commune Name"] = pd.NA
    villages["Annual Climate Link Method"] = villages[
        "Linked Climate Commune Code"
    ].notna().map({True: "exact historical commune code", False: "unresolved"})

    unresolved = villages["Linked Climate Commune Code"].isna()
    if unresolved.any():
        boundaries = gpd.read_file(
            root / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
        )[["ADM3_PCODE", "ADM3_EN", "geometry"]].to_crs(4326)
        boundaries["Linked Climate Commune Code"] = (
            boundaries["ADM3_PCODE"].astype("string").str.replace("KH", "", regex=False)
        )
        boundaries = boundaries.loc[
            boundaries["Linked Climate Commune Code"].isin(valid_climate_codes)
        ].copy()
        points = gpd.GeoDataFrame(
            villages.loc[unresolved, ["Village Code"]],
            geometry=gpd.points_from_xy(
                villages.loc[unresolved, "Longitude"],
                villages.loc[unresolved, "Latitude"],
                crs=4326,
            ),
            crs=4326,
        )
        spatial = gpd.sjoin(
            points,
            boundaries[
                ["Linked Climate Commune Code", "ADM3_EN", "geometry"]
            ],
            how="left",
            predicate="within",
        )
        unique = (
            spatial.dropna(subset=["Linked Climate Commune Code"])
            .groupby("Village Code")
            .filter(lambda group: group["Linked Climate Commune Code"].nunique() == 1)
            .drop_duplicates("Village Code")
            .set_index("Village Code")
        )
        target = villages["Village Code"].isin(unique.index)
        villages.loc[target, "Linked Climate Commune Code"] = villages.loc[
            target, "Village Code"
        ].map(unique["Linked Climate Commune Code"])
        villages.loc[target, "Linked Climate Commune Name"] = villages.loc[
            target, "Village Code"
        ].map(unique["ADM3_EN"])
        villages.loc[target, "Annual Climate Link Method"] = (
            "village point within modern climate commune"
        )
    return villages[
        [
            "Village Code",
            "Commune Code",
            "Linked Climate Commune Code",
            "Linked Climate Commune Name",
            "Annual Climate Link Method",
        ]
    ]


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    npp_path = resolved(root, args.npp)
    climate_path = resolved(root, args.climate)
    output = resolved(root, args.output)
    audit_output = resolved(root, args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.mkdir(parents=True, exist_ok=True)

    npp = pd.read_parquet(npp_path)
    npp["Commune Code"] = npp["Village Code"].astype("string").str[:6]
    river = pd.read_csv(
        root
        / "data/exp/feasibility-check/historical-boundary-identification/"
        "predetermined_and_alignment_fields.csv",
        usecols=["vill_code", "river_d"],
    )
    river["Village Code"] = (
        river["vill_code"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)
    )
    river["Log One Plus Distance to River m"] = np.log1p(
        pd.to_numeric(river["river_d"], errors="coerce").clip(lower=0)
    )
    river = river[["Village Code", "Log One Plus Distance to River m"]].drop_duplicates(
        "Village Code"
    )
    npp = npp.merge(river, on="Village Code", how="left", validate="many_to_one")
    missing_river = npp["Log One Plus Distance to River m"].isna()
    if npp.loc[
        missing_river & npp["Absolute Distance to Historical Repression Boundary km"].le(30)
    ].shape[0] > 0:
        raise RuntimeError("Historical river-distance control is missing within 30 km support")
    climate = pd.read_parquet(climate_path, columns=CLIMATE_COLUMNS).copy()
    climate["Climate Geography Code"] = (
        climate["Climate Geography Code"].astype("string").str.zfill(6)
    )
    if climate.duplicated(["Climate Geography Code", "Year"]).any():
        raise RuntimeError("CHIRPS commune-year keys are not unique")

    crosswalk = build_climate_geography_crosswalk(
        root, npp, set(climate["Climate Geography Code"].dropna())
    )
    npp = npp.merge(
        crosswalk.drop(columns="Commune Code"),
        on="Village Code",
        how="left",
        validate="many_to_one",
    )

    panel = npp.merge(
        climate,
        left_on=["Linked Climate Commune Code", "Year"],
        right_on=["Climate Geography Code", "Year"],
        how="left",
        validate="many_to_one",
    )
    panel["Annual Climate Shock Available"] = (
        panel["Annual Rainfall Anomaly Z (1991-2020)"].notna().astype("Int8")
    )
    panel = panel.drop(columns="Climate Geography Code")
    if len(panel) != len(npp):
        raise RuntimeError("The annual spatial merge changed the number of village-years")
    if panel.duplicated(["Village Code", "Year"]).any():
        raise RuntimeError("Duplicate village-years after CHIRPS linkage")

    expected = panel["Year"].between(2001, 2021)
    expected_coverage = panel.loc[expected, "Annual Climate Shock Available"].mean()
    if expected_coverage != 1:
        unresolved = panel.loc[
            expected & panel["Annual Climate Shock Available"].eq(0),
            ["Village Code", "Commune Code", "Year"],
        ]
        unresolved.to_csv(audit_output / "unresolved_npp_chirps_links.csv", index=False)
        raise RuntimeError(
            f"CHIRPS linkage is incomplete in 2001-2021: {expected_coverage:.3%}"
        )
    after_source_end = panel["Year"].gt(2021)
    if panel.loc[after_source_end, "Annual Climate Shock Available"].any():
        raise RuntimeError("Unexpected CHIRPS values after the documented 2021 source end")

    panel.to_parquet(output, index=False)
    crosswalk.sort_values("Village Code").to_csv(
        audit_output / "annual_spatial_climate_geography_crosswalk.csv", index=False
    )
    coverage = (
        panel.groupby("Year", observed=True)
        .agg(
            Villages=("Village Code", "size"),
            NPP_available=("Annual Land NPP Mean kg C per m2", "count"),
            Climate_available=("Annual Climate Shock Available", "sum"),
        )
        .reset_index()
    )
    coverage["NPP_coverage_share"] = coverage["NPP_available"] / coverage["Villages"]
    coverage["Climate_coverage_share"] = coverage["Climate_available"] / coverage["Villages"]
    coverage.to_csv(audit_output / "annual_spatial_release_coverage.csv", index=False)

    summary = {
        "rows": len(panel),
        "unique_villages": panel["Village Code"].nunique(),
        "outcome_years": [int(panel["Year"].min()), int(panel["Year"].max())],
        "shock_years": [2001, 2021],
        "complete_outcome_shock_village_years": int(
            (
                panel["Annual Land NPP Mean kg C per m2"].notna()
                & panel["Annual Climate Shock Available"].eq(1)
            ).sum()
        ),
        "linkage": "first six digits of the eight-digit public historical village code to audited CHIRPS commune-year geography",
        "linkage_fallback": "deterministic village-point-in-modern-commune assignment for otherwise unresolved legacy commune codes; ambiguous or unmatched points remain missing",
        "link_methods": crosswalk["Annual Climate Link Method"].value_counts().to_dict(),
        "missing_rule": "CHIRPS years after 2021 remain missing and are never coded as zero",
        "effect_estimation_performed": False,
        "historical_river_distance_coverage": float(
            panel["Log One Plus Distance to River m"].notna().mean()
        ),
        "historical_river_distance_missing_rule": "Eighteen villages outside 53 km are absent from the public replication regression frame and remain missing; coverage is complete inside every 2-30 km design window.",
    }
    (audit_output / "README_annual_spatial_release.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(panel):,} village-years to {output}")
    print(
        f"complete NPP-shock rows={summary['complete_outcome_shock_village_years']:,}; "
        f"2001-2021 climate linkage={expected_coverage:.3%}"
    )


if __name__ == "__main__":
    main()
