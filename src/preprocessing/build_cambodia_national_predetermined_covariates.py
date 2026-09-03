#!/usr/bin/env python3
"""Merge the frozen nationwide predetermined covariates onto the stable 1 km grid."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "data/processed/cambodia_national_1km_grid_preprocessed.parquet"
OUTPUT = ROOT / "data/processed/cambodia_national_predetermined_covariates_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/national-predetermined-covariates"

SOURCES = {
    "terrain": ROOT / "data/processed/cambodia_national_terrain_preprocessed.parquet",
    "border": ROOT / "data/processed/cambodia_national_thailand_border_distance_preprocessed.parquet",
    "roads": ROOT / "data/processed/cambodia_national_historical_road_access_preprocessed.parquet",
    "land_cover": ROOT / "data/processed/cambodia_national_baseline_land_cover_preprocessed.parquet",
    "population": ROOT / "data/processed/cambodia_national_baseline_population_2000_preprocessed.parquet",
    "population_robustness": ROOT / "data/processed/cambodia_national_worldpop_2000_robustness_preprocessed.parquet",
}

KEEP = {
    "terrain": [
        "Mean Elevation m",
        "Elevation SD m",
        "Mean Slope Degrees",
        "Slope P90 Degrees",
        "Steep Terrain Share",
        "Terrain Valid Pixel Count",
        "Terrain Observed",
    ],
    "border": [
        "Distance to Cambodia Thailand Border km",
        "Nearest Border Longitude",
        "Nearest Border Latitude",
        "Nearest Border Sector",
        "Within 10 km of Cambodia Thailand Border",
        "Within 20 km of Cambodia Thailand Border",
        "Within 40 km of Cambodia Thailand Border",
        "Within 60 km of Cambodia Thailand Border",
    ],
    "roads": [
        "Historical Road Density Proxy km per km2",
        "Distance to Nearest Historical Road Proxy km",
        "Road Density Excluding Post 2007 AidData Corridors km per km2",
        "Distance to Road Excluding Post 2007 AidData Corridors km",
    ],
    "land_cover": [
        "Baseline Cropland Share",
        "Baseline Forest Share",
        "Baseline Grass Shrub Share",
        "Baseline Built Share",
        "Baseline Water Wetland Share",
        "Baseline Other Share",
        "Baseline Land Cover Stability Share",
        "Baseline Land Cover Observed",
        "Baseline Classified Share Sum",
        "Baseline Dominant Land Cover Group",
    ],
    "population": [
        "Baseline Population 2000",
        "Baseline Population Density per km2",
        "Log Baseline Population 2000",
        "Baseline Population Observed",
    ],
    "population_robustness": [
        "WorldPop Baseline Population 2000",
        "Log WorldPop Baseline Population 2000",
        "WorldPop Baseline Population Observed",
    ],
}


def main() -> None:
    missing = [path for path in [GRID, *SOURCES.values()] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing predetermined-covariate inputs: {missing}")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    output = pd.read_parquet(GRID)
    if len(output) != 179_072 or output["National Grid Cell ID"].duplicated().any():
        raise RuntimeError("Stable national grid failed key validation")
    source_rows = []
    for family, path in SOURCES.items():
        source = pd.read_parquet(path, columns=["National Grid Cell ID", *KEEP[family]])
        if len(source) != 179_072 or source["National Grid Cell ID"].duplicated().any():
            raise RuntimeError(f"{family} failed one-row-per-grid-cell validation")
        output = output.merge(
            source, on="National Grid Cell ID", how="left", validate="one_to_one"
        )
        source_rows.append(
            {
                "Covariate Family": family,
                "Source": path.relative_to(ROOT).as_posix(),
                "Rows": len(source),
                "Variables Added": len(KEEP[family]),
            }
        )

    primary_variables = [
        "Mean Elevation m",
        "Elevation SD m",
        "Mean Slope Degrees",
        "Slope P90 Degrees",
        "Steep Terrain Share",
        "Distance to Cambodia Thailand Border km",
        "Historical Road Density Proxy km per km2",
        "Distance to Nearest Historical Road Proxy km",
        "Baseline Cropland Share",
        "Baseline Forest Share",
        "Baseline Grass Shrub Share",
        "Baseline Built Share",
        "Baseline Water Wetland Share",
        "Baseline Land Cover Stability Share",
        "Baseline Population 2000",
        "Log Baseline Population 2000",
        "Baseline Population Density per km2",
    ]
    missing_primary = output[primary_variables].isna().sum()
    if int(missing_primary.sum()) != 0:
        raise RuntimeError(f"Primary covariates contain missing values: {missing_primary.to_dict()}")
    if not np.allclose(
        output[
            [
                "Baseline Cropland Share",
                "Baseline Forest Share",
                "Baseline Grass Shrub Share",
                "Baseline Built Share",
                "Baseline Water Wetland Share",
                "Baseline Other Share",
            ]
        ].sum(axis=1),
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError("Baseline land-cover shares do not sum to one")
    output.to_parquet(OUTPUT, index=False)

    pd.DataFrame(source_rows).to_csv(AUDIT_DIR / "covariate_source_merge_audit.csv", index=False)
    missingness = (
        output.isna().sum().rename("Missing Rows").rename_axis("Variable").reset_index()
    )
    missingness["Missing Percent"] = missingness["Missing Rows"] / len(output) * 100
    missingness.to_csv(AUDIT_DIR / "covariate_stack_missingness.csv", index=False)
    metadata = {
        "unit": "Cambodia stable national 1 km grid cell",
        "rows": len(output),
        "columns": len(output.columns),
        "primary_covariates": primary_variables,
        "primary_missing_cells": int(output[primary_variables].isna().any(axis=1).sum()),
        "outcome_blind": True,
        "processed_at_utc": datetime.now(UTC).isoformat(),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
    }
    (AUDIT_DIR / "covariate_stack_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
