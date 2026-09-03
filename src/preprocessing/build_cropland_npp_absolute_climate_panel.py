#!/usr/bin/env python3
"""Join the four annual absolute climate shocks to 5 km cropland NPP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CROSSWALK = Path("data/processed/cses_to_cambodia_public_village_point_crosswalk_preprocessed.parquet")
CLIMATE = Path("data/processed/cses_village_buffer_annual_absolute_climate_shocks_preprocessed.parquet")
NPP = Path("data/processed/cses_public_village_pixel_cropland_npp_annual_preprocessed.parquet")
OUTPUT = Path("data/processed/cses_public_village_cropland_npp_absolute_climate_panel_preprocessed.parquet")
AUDIT = Path("data/exp/data-preprocessing/climate-npp-two-stage/stage1-panel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = root / AUDIT
    audit.mkdir(parents=True, exist_ok=True)

    crosswalk = pd.read_parquet(root / CROSSWALK)
    crosswalk = crosswalk.loc[crosswalk["National Public Village Point Matched"].eq(1)].copy()
    climate = pd.read_parquet(root / CLIMATE)
    climate = climate.loc[climate["Buffer Radius km"].eq(5)].copy()
    climate = climate.merge(
        crosswalk[["Village Code", "National Village Point ID"]],
        on="Village Code",
        how="inner",
        validate="many_to_one",
    )
    climate_variables = [
        column for column in climate.columns
        if column.startswith("Village Buffer Mean Annual")
    ]
    # Three public points are linked to two current CSES codes.  They represent
    # one physical point and therefore must not receive double weight in stage 1.
    point_climate = climate.groupby(
        ["National Village Point ID", "Year"], observed=True
    ).agg(
        **{
            "Current CSES Village Codes Linked": ("Village Code", "nunique"),
            "Representative CSES Village Code": ("Village Code", "min"),
            "Village Buffer Climate Cell Count": ("Village Buffer Climate Cell Count", "mean"),
            "Village Buffer Included 1 km Grid Cell Count": (
                "Village Buffer Included 1 km Grid Cell Count", "mean"
            ),
            **{column: (column, "mean") for column in climate_variables},
        }
    ).reset_index()
    npp = pd.read_parquet(root / NPP)
    panel = npp.merge(
        point_climate,
        on=["National Village Point ID", "Year"],
        how="left",
        validate="one_to_one",
    )
    panel["Stage 1 Strict Cropland Complete Case"] = panel[
        [
            "Annual Strict-Cropland Mean NPP kg C per m2",
            "Village Buffer Mean Annual Heat Days at or Above 35 C",
            "Village Buffer Mean Annual Heat Degree-Days Above 35 C",
            "Village Buffer Mean Annual Maximum Consecutive Five-Day Precipitation Rx5day mm",
            "Village Buffer Mean Annual Maximum Consecutive Dry Days Below 1 mm",
        ]
    ].notna().all(axis=1)
    panel["Primary Heat Specification"] = "Annual Heat Days at or Above 35 C"
    panel["Alternative Heat Specification"] = "Annual Heat Degree-Days Above 35 C"
    if panel.duplicated(["National Village Point ID", "Year"]).any():
        raise ValueError("Duplicate stage-1 village-year keys")
    panel.to_parquet(root / OUTPUT, index=False)

    coverage = panel.groupby("Year", observed=True).agg(
        **{
            "Public Village Points": ("National Village Point ID", "nunique"),
            "Strict Cropland NPP Observed": ("Annual Strict-Cropland Mean NPP kg C per m2", "count"),
            "Climate Observed": ("Village Buffer Mean Annual Heat Days at or Above 35 C", "count"),
            "Complete Cases": ("Stage 1 Strict Cropland Complete Case", "sum"),
        }
    ).reset_index()
    coverage.to_csv(audit / "stage1_panel_coverage_by_year.csv", index=False)
    duplicates = crosswalk.loc[
        crosswalk.duplicated("National Village Point ID", keep=False)
    ].sort_values(["National Village Point ID", "Village Code"])
    duplicates.to_csv(audit / "duplicate_current_codes_per_public_point.csv", index=False)
    metadata = {
        "output": str(OUTPUT),
        "grain": "one national public village point by year",
        "years": [int(panel["Year"].min()), int(panel["Year"].max())],
        "buffer_radius_km": 5,
        "rows": int(len(panel)),
        "public_village_points": int(panel["National Village Point ID"].nunique()),
        "complete_cases": int(panel["Stage 1 Strict Cropland Complete Case"].sum()),
        "primary_outcome": "Annual Strict-Cropland Mean NPP kg C per m2",
        "primary_heat": "Village Buffer Mean Annual Heat Days at or Above 35 C",
        "alternative_heat": "Village Buffer Mean Annual Heat Degree-Days Above 35 C",
        "other_exposures": [
            "Village Buffer Mean Annual Maximum Consecutive Five-Day Precipitation Rx5day mm",
            "Village Buffer Mean Annual Maximum Consecutive Dry Days Below 1 mm",
        ],
        "duplicate_point_rule": "current CSES codes mapping to the same physical public point are collapsed before estimation",
    }
    (audit / "stage1_panel_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
