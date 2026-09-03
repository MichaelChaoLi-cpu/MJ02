#!/usr/bin/env python3
"""Materialise the human-approved six-region SKATER partition.

The script reads only the pre-outcome regionalisation candidates and support
diagnostics. It does not read NPP, consumption, regression coefficients, or
p-values. Region numbers are relabelled deterministically from north to south
and then west to east so that tables and figures have stable identifiers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data/exp/analysis/climate-npp/outcome-blind-regionalisation-gate"
FEATURES = INPUT_DIR / "outcome_blind_candidate_features_and_support.parquet"
LABELS = INPUT_DIR / "outcome_blind_candidate_region_labels.csv"
DIAGNOSTICS = INPUT_DIR / "outcome_blind_region_selection_diagnostics.csv"
OUTPUT = ROOT / "data/processed/outcome_blind_spatial_regions_preprocessed.parquet"
AUDIT = ROOT / "data/exp/data-preprocessing/outcome-blind-regionalisation/frozen_region_support_audit.csv"

ID = "National Village Point ID"
LONGITUDE = "Point Longitude"
LATITUDE = "Point Latitude"
SELECTED_K = 6
MINIMUM_VILLAGES = 200
MINIMUM_MEAN_STAGE1_YEARS = 15.0
MINIMUM_STAGE2_HOUSEHOLDS = 3000


def deterministic_relabel(frame: pd.DataFrame, raw_column: str) -> tuple[pd.Series, pd.DataFrame]:
    centroids = (
        frame.groupby(raw_column, as_index=False)
        .agg(
            **{
                "Region Centroid Longitude": (LONGITUDE, "mean"),
                "Region Centroid Latitude": (LATITUDE, "mean"),
            }
        )
        .sort_values(
            ["Region Centroid Latitude", "Region Centroid Longitude"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    centroids["Relabelled Region ID"] = np.arange(1, len(centroids) + 1)
    mapping = centroids.set_index(raw_column)["Relabelled Region ID"].to_dict()
    return frame[raw_column].map(mapping).astype("int8"), centroids


def main() -> None:
    features = pd.read_parquet(FEATURES)
    labels = pd.read_csv(LABELS)
    diagnostics = pd.read_csv(DIAGNOSTICS)
    selected_labels = labels.loc[labels["Regions"].eq(SELECTED_K)].copy()
    if "SKATER" not in set(selected_labels["Algorithm"]):
        raise ValueError("The selected six-region labels must contain SKATER.")

    skater_labels = (
        selected_labels.loc[selected_labels["Algorithm"].eq("SKATER"), [ID, "Region ID"]]
        .rename(columns={"Region ID": "SKATER Raw Region ID"})
    )
    frame = features.merge(skater_labels, on=ID, how="left", validate="one_to_one")
    if frame["SKATER Raw Region ID"].isna().any():
        raise ValueError("Some national village points lack the selected SKATER assignment.")

    relabelling_rows: list[pd.DataFrame] = []
    for algorithm in ["SKATER"]:
        raw_column = f"{algorithm} Raw Region ID"
        final_column = f"{algorithm} Region ID"
        frame[final_column], centroids = deterministic_relabel(frame, raw_column)
        centroids.insert(0, "Algorithm", algorithm)
        relabelling_rows.append(centroids)

    audit_rows: list[pd.DataFrame] = []
    for algorithm in ["SKATER"]:
        region_column = f"{algorithm} Region ID"
        audit = (
            frame.assign(Villages=1)
            .groupby(region_column, as_index=False)
            .agg(
                Villages=("Villages", "sum"),
                **{
                    "Stage 1 Village-Years": ("Stage 1 Complete Village-Years", "sum"),
                    "Stage 2 Households": ("Stage 2 Complete Households", "sum"),
                    "Centroid Longitude": (LONGITUDE, "mean"),
                    "Centroid Latitude": (LATITUDE, "mean"),
                },
            )
            .rename(columns={region_column: "Region ID"})
        )
        audit["Mean Stage 1 Years per Village"] = audit["Stage 1 Village-Years"] / audit["Villages"]
        audit["Village Gate Pass"] = audit["Villages"].ge(MINIMUM_VILLAGES)
        audit["Stage 1 Gate Pass"] = audit["Mean Stage 1 Years per Village"].ge(MINIMUM_MEAN_STAGE1_YEARS)
        audit["Stage 2 Gate Pass"] = audit["Stage 2 Households"].ge(MINIMUM_STAGE2_HOUSEHOLDS)
        audit["All Support Gates Pass"] = audit[
            ["Village Gate Pass", "Stage 1 Gate Pass", "Stage 2 Gate Pass"]
        ].all(axis=1)
        audit.insert(0, "Algorithm", algorithm)
        audit_rows.append(audit)

    support_audit = pd.concat(audit_rows, ignore_index=True)
    if not support_audit["All Support Gates Pass"].all():
        failed = support_audit.loc[~support_audit["All Support Gates Pass"]]
        raise ValueError(f"A frozen region fails the approved support gate:\n{failed}")

    selected_diagnostics = diagnostics.loc[
        diagnostics["Regions"].eq(SELECTED_K) & diagnostics["Algorithm"].eq("SKATER")
    ]
    if len(selected_diagnostics) != 1 or not selected_diagnostics["Contiguous"].all():
        raise ValueError("The selected SKATER partition did not pass the pre-outcome contiguity audit.")

    frame["Spatial Regionalisation Region Count"] = SELECTED_K
    frame["Spatial Regionalisation Outcome Blind"] = True
    frame["Spatial Regionalisation Frozen"] = True
    frame["Minimum Required Villages per Region"] = MINIMUM_VILLAGES
    frame["Minimum Required Mean Stage 1 Years per Village"] = MINIMUM_MEAN_STAGE1_YEARS
    frame["Minimum Required Stage 2 Households per Region"] = MINIMUM_STAGE2_HOUSEHOLDS
    frame = frame.drop(columns=["SKATER Raw Region ID"])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(OUTPUT, index=False)
    support_audit.to_csv(AUDIT, index=False)
    pd.concat(relabelling_rows, ignore_index=True).to_csv(
        AUDIT.with_name("frozen_region_relabelling_audit.csv"), index=False
    )

    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved: {AUDIT.relative_to(ROOT)}")
    print(f"villages={len(frame):,}; algorithm=SKATER; selected_regions={SELECTED_K}; outcome_blind=True")
    print(support_audit.to_string(index=False))


if __name__ == "__main__":
    main()
