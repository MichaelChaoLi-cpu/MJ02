#!/usr/bin/env python3
"""Build outcome-blind candidate exposure features for Cambodia-Thailand conflict events."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "data/processed/cambodia_national_1km_grid_preprocessed.parquet"
EVENTS = ROOT / "data/processed/ucdp_cambodia_thailand_state_conflict_candidates_preprocessed.parquet"
OUTPUT = ROOT / "data/processed/cambodia_national_ucdp_border_conflict_exposure_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/national-ucdp-border-conflict-exposure"
RADII_KM = (10, 20, 40, 60)


def phase_mask(events: pd.DataFrame, phase: str) -> np.ndarray:
    if phase == "All 2008-2011":
        return events["year"].between(2008, 2011).to_numpy()
    if phase == "Precursor 2008-2009":
        return events["year"].between(2008, 2009).to_numpy()
    if phase == "UCDP Conflict Year 2011":
        return events["year"].eq(2011).to_numpy()
    if phase == "Higher Spatial Precision 2008-2011":
        return (events["year"].between(2008, 2011) & events["where_prec"].le(2)).to_numpy()
    raise ValueError(phase)


def safe_prefix(label: str) -> str:
    return label.replace("UCDP ", "").replace("Higher Spatial Precision ", "Higher Precision ")


def main() -> None:
    if not GRID.exists() or not EVENTS.exists():
        raise FileNotFoundError("National grid or UCDP candidate events are missing")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    grid_columns = [
        "National Grid Cell ID", "Grid Centre Easting m", "Grid Centre Northing m",
        "Longitude", "Latitude", "Province Code", "Province Name",
        "District Code", "District Name", "Commune Code", "Commune Name",
    ]
    grid = pd.read_parquet(GRID, columns=grid_columns)
    events = pd.read_parquet(EVENTS).sort_values(["date_start", "id"]).reset_index(drop=True)
    events = events.loc[events["year"].between(2008, 2011)].copy()
    if events.empty:
        raise RuntimeError("No Cambodia-Thailand candidate events in 2008-2011")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    event_easting, event_northing = transformer.transform(
        events["longitude"].to_numpy(), events["latitude"].to_numpy()
    )
    grid_xy = grid[["Grid Centre Easting m", "Grid Centre Northing m"]].to_numpy(dtype=float)
    event_xy = np.column_stack([event_easting, event_northing])
    distance_km = np.sqrt(
        (grid_xy[:, None, 0] - event_xy[None, :, 0]) ** 2
        + (grid_xy[:, None, 1] - event_xy[None, :, 1]) ** 2
    ) / 1000.0
    best_deaths = pd.to_numeric(events["best"], errors="coerce").fillna(0).to_numpy(dtype=float)

    output = grid.copy()
    phases = [
        "All 2008-2011",
        "Precursor 2008-2009",
        "UCDP Conflict Year 2011",
        "Higher Spatial Precision 2008-2011",
    ]
    audit_rows: list[dict[str, int | float | str]] = []
    for phase in phases:
        mask = phase_mask(events, phase)
        if not mask.any():
            continue
        phase_distance = distance_km[:, mask]
        prefix = safe_prefix(phase)
        output[f"Candidate {prefix} Nearest Event Distance km"] = phase_distance.min(axis=1)
        for radius in RADII_KM:
            inside = phase_distance <= radius
            count = inside.sum(axis=1).astype("int16")
            deaths = (inside * best_deaths[mask][None, :]).sum(axis=1).astype("float32")
            output[f"Candidate {prefix} Event Count Within {radius} km"] = count
            output[f"Candidate {prefix} Best Deaths Within {radius} km"] = deaths
            output[f"Candidate {prefix} Exposure Within {radius} km"] = (count > 0).astype("int8")
            audit_rows.append(
                {
                    "Phase": phase,
                    "Radius km": radius,
                    "Exposed Grid Cells": int((count > 0).sum()),
                    "Exposed Provinces": int(output.loc[count > 0, "Province Code"].nunique()),
                    "Exposed Communes": int(output.loc[count > 0, "Commune Code"].nunique()),
                }
            )

    primary_distance = output["Candidate Conflict Year 2011 Nearest Event Distance km"]
    output["Candidate 2011 Distance Ring"] = pd.cut(
        primary_distance,
        bins=[-np.inf, 10, 20, 40, 60, np.inf],
        labels=["0-10 km", "10-20 km", "20-40 km", "40-60 km", "Over 60 km"],
        right=True,
    ).astype(str)
    nearest_index = distance_km.argmin(axis=1)
    output["Candidate Nearest UCDP Event ID"] = events.iloc[nearest_index]["id"].to_numpy(dtype="int64")
    output["Candidate Nearest UCDP Event Date"] = events.iloc[nearest_index]["date_start"].to_numpy()
    output["Candidate Nearest UCDP Event Place"] = events.iloc[nearest_index]["where_coordinates"].to_numpy()
    output["Candidate Nearest UCDP Event Spatial Precision"] = events.iloc[nearest_index]["where_prec"].to_numpy(dtype="int16")
    output.to_parquet(OUTPUT, index=False)

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(AUDIT_DIR / "candidate_exposure_counts_by_phase_and_radius.csv", index=False)
    event_sites = (
        events.groupby(["where_coordinates", "latitude", "longitude"], dropna=False)
        .agg(
            First_Date=("date_start", "min"),
            Last_Date=("date_end", "max"),
            Events=("id", "size"),
            Best_Deaths=("best", "sum"),
            Minimum_Spatial_Precision_Code=("where_prec", "min"),
            Maximum_Spatial_Precision_Code=("where_prec", "max"),
        )
        .reset_index()
    )
    event_sites.to_csv(AUDIT_DIR / "candidate_event_site_inventory.csv", index=False)
    metadata = {
        "panel_unit": "Cambodia national 1 km grid cell",
        "grid_cells": int(len(output)),
        "candidate_events": int(len(events)),
        "candidate_event_coordinate_locations": int(events[["latitude", "longitude"]].drop_duplicates().shape[0]),
        "distance_crs": "EPSG:32648",
        "distance_rule": "Euclidean distance from 1 km grid-cell centre to UCDP event coordinate",
        "radii_km": list(RADII_KM),
        "outcome_blind": True,
        "fatality_rule": "Sum UCDP best estimates across candidate event records; preserve low/high in source extract",
        "status": "candidate exposure features; not a frozen causal treatment definition",
        "source_events": str(EVENTS.relative_to(ROOT)),
        "source_grid": str(GRID.relative_to(ROOT)),
        "output": str(OUTPUT.relative_to(ROOT)),
    }
    (AUDIT_DIR / "candidate_exposure_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if output["National Grid Cell ID"].duplicated().any() or len(output) != 179_072:
        raise RuntimeError("Candidate exposure output failed grid-key validation")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
