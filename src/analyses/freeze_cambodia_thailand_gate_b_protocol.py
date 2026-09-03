#!/usr/bin/env python3
"""Freeze the outcome-blind Gate B sample, weights, and placebo-site universe."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mj02-matplotlib")

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from diagnose_cambodia_thailand_common_support import (
    CLIMATE,
    COVARIATES,
    EXPOSURE,
    FIXED_PREDICTORS,
    PANEL,
    PRE_CLIMATE_PREDICTORS,
    construct_pre_climate,
)


ROOT = Path(__file__).resolve().parents[2]
EVENTS = ROOT / "data/processed/ucdp_cambodia_thailand_state_conflict_candidates_preprocessed.parquet"
BORDER = ROOT / "data/processed/cambodia_thailand_shared_border_preprocessed.parquet"
OUT_DIR = ROOT / "data/exp/experiment-design/cambodia-thailand-gate-b"

RING_LABELS = ("0-10 km", "10-20 km", "20-40 km", "40-60 km", "Over 60 km")
RING_BINS = (-np.inf, 10, 20, 40, 60, np.inf)
STRICT_CONTROL_KM = 60
FRONTIER_WIDTH_KM = 60
SPATIAL_BLOCK_CELLS = 10
PLACEBO_ANCHOR_STEP_KM = 10
PLACEBO_ACTUAL_EXCLUSION_KM = 120
PLACEBO_MIN_CELLS = {
    "0-10 km": 150,
    "10-20 km": 400,
    "20-40 km": 1500,
    "40-60 km": 2000,
    "Over 60 km": 5000,
}
PLACEBO_MIN_BLOCKS = {
    "0-10 km": 8,
    "10-20 km": 10,
    "20-40 km": 20,
    "40-60 km": 25,
    "Over 60 km": 50,
}

# Absolute longitude and latitude are deliberately omitted: border-sector-by-year
# fixed effects absorb broad location, while the weights balance substantive
# predetermined attributes within the declared frontier support.
WEIGHT_PREDICTORS = [
    variable for variable in FIXED_PREDICTORS if variable not in {"Longitude", "Latitude"}
] + PRE_CLIMATE_PREDICTORS


def effective_sample_size(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.sum(weights**2))


def weighted_mean_variance(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, variance


def load_features() -> pd.DataFrame:
    fixed = pd.read_parquet(COVARIATES)
    exposure = pd.read_parquet(
        EXPOSURE,
        columns=[
            "National Grid Cell ID",
            "Candidate Conflict Year 2011 Nearest Event Distance km",
            "Candidate All 2008-2011 Nearest Event Distance km",
        ],
    )
    links = (
        pd.read_parquet(PANEL, columns=["National Grid Cell ID", "Year", "Climate Cell ID"])
        .loc[lambda frame: frame["Year"].eq(2000), ["National Grid Cell ID", "Climate Cell ID"]]
    )
    climate = construct_pre_climate()
    features = (
        fixed.merge(exposure, on="National Grid Cell ID", validate="one_to_one")
        .merge(links, on="National Grid Cell ID", validate="one_to_one")
        .merge(climate, on="Climate Cell ID", validate="many_to_one")
    )
    complete = ~features[WEIGHT_PREDICTORS].isna().any(axis=1)
    frontier = features["Within 60 km of Cambodia Thailand Border"].astype(bool)
    treated_support = features["Candidate Conflict Year 2011 Nearest Event Distance km"].le(60)
    strict_control = features["Candidate All 2008-2011 Nearest Event Distance km"].gt(60)
    sample = features.loc[complete & frontier & (treated_support | strict_control)].copy()
    sample["Conflict Distance Ring"] = pd.cut(
        sample["Candidate Conflict Year 2011 Nearest Event Distance km"],
        bins=RING_BINS,
        labels=RING_LABELS,
        ordered=True,
    ).astype(str)
    sample["Border Analysis Sector"] = (
        np.floor(sample["Nearest Border Longitude"]).astype(int).astype(str)
        + "-"
        + (np.floor(sample["Nearest Border Longitude"]).astype(int) + 1).astype(str)
        + " E"
    )
    sample["Spatial Block ID"] = (
        (sample["Grid Column"] // SPATIAL_BLOCK_CELLS).astype(int).astype(str)
        + "_"
        + (sample["Grid Row"] // SPATIAL_BLOCK_CELLS).astype(int).astype(str)
    )
    return sample.reset_index(drop=True)


def generalized_overlap_weights(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    x = scaler.fit_transform(sample[WEIGHT_PREDICTORS])
    labels = pd.Categorical(sample["Conflict Distance Ring"], categories=RING_LABELS, ordered=True)
    if (labels.codes < 0).any():
        raise RuntimeError("Unclassified Gate B distance ring")
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    model.fit(x, labels.codes)
    probabilities = np.clip(model.predict_proba(x), 1e-8, 1)
    observed_probability = probabilities[np.arange(len(sample)), labels.codes]
    tilting = 1 / np.sum(1 / probabilities, axis=1)
    raw_weight = tilting / observed_probability
    weights = raw_weight.copy()
    for code in range(len(RING_LABELS)):
        group = labels.codes == code
        weights[group] *= (len(sample) / len(RING_LABELS)) / weights[group].sum()
    output = sample.copy()
    output["Gate B Generalized Overlap Weight"] = weights
    output["Observed Ring Probability"] = observed_probability
    for index, label in enumerate(RING_LABELS):
        output[f"Probability {label}"] = probabilities[:, index]

    balance_rows: list[dict[str, object]] = []
    reference = output["Conflict Distance Ring"].eq("Over 60 km").to_numpy()
    for label in RING_LABELS[:-1]:
        treated = output["Conflict Distance Ring"].eq(label).to_numpy()
        for variable in WEIGHT_PREDICTORS:
            treated_mean, treated_variance = weighted_mean_variance(
                output.loc[treated, variable].to_numpy(float), weights[treated]
            )
            control_mean, control_variance = weighted_mean_variance(
                output.loc[reference, variable].to_numpy(float), weights[reference]
            )
            denominator = np.sqrt((treated_variance + control_variance) / 2)
            smd = (treated_mean - control_mean) / denominator if denominator > 0 else 0.0
            balance_rows.append(
                {
                    "Distance Ring": label,
                    "Reference Ring": "Over 60 km",
                    "Variable": variable,
                    "Weighted Treated Mean": treated_mean,
                    "Weighted Reference Mean": control_mean,
                    "Standardized Mean Difference": smd,
                    "Absolute Standardized Mean Difference": abs(smd),
                }
            )
    return output, pd.DataFrame(balance_rows)


def support_summary(weighted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in RING_LABELS:
        group = weighted[weighted["Conflict Distance Ring"].eq(label)]
        weights = group["Gate B Generalized Overlap Weight"].to_numpy(float)
        block_weights = group.groupby("Spatial Block ID")["Gate B Generalized Overlap Weight"].sum()
        commune_weights = group.groupby("Commune Code")["Gate B Generalized Overlap Weight"].sum()
        rows.append(
            {
                "Distance Ring": label,
                "Grid Cells": len(group),
                "Communes": group["Commune Code"].nunique(),
                "Provinces": group["Province Code"].nunique(),
                "Border Analysis Sectors": group["Border Analysis Sector"].nunique(),
                "Spatial Blocks": group["Spatial Block ID"].nunique(),
                "Cell ESS": effective_sample_size(weights),
                "Commune ESS": effective_sample_size(commune_weights.to_numpy(float)),
                "10 km Block ESS": effective_sample_size(block_weights.to_numpy(float)),
                "Minimum Observed Ring Probability": group["Observed Ring Probability"].min(),
                "P01 Observed Ring Probability": group["Observed Ring Probability"].quantile(0.01),
            }
        )
    return pd.DataFrame(rows)


def freeze_event_sectors_and_placebos(
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_parquet(EVENTS)
    events = events.loc[events["year"].eq(2011), ["where_coordinates", "latitude", "longitude"]]
    events = events.drop_duplicates(["latitude", "longitude"]).reset_index(drop=True)
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    east, north = to_utm.transform(events["longitude"].to_numpy(), events["latitude"].to_numpy())
    coordinates = np.column_stack([east, north])
    events["Conflict Sector Cluster"] = DBSCAN(eps=30_000, min_samples=1).fit_predict(coordinates)
    border = gpd.read_parquet(BORDER).geometry.iloc[0]
    if border.geom_type != "LineString":
        raise RuntimeError(f"Expected one ordered border LineString, got {border.geom_type}")
    sector_rows = []
    for cluster, part in events.groupby("Conflict Sector Cluster", sort=True):
        indices = part.index.to_numpy()
        centre = shapely.Point(float(np.mean(east[indices])), float(np.mean(north[indices])))
        border_position = float(border.project(centre))
        anchor = border.interpolate(border_position)
        sector_rows.append(
            {
                "Conflict Sector ID": f"2011-sector-{cluster + 1}",
                "Unique UCDP Coordinates": len(part),
                "UCDP Places": "; ".join(sorted(part["where_coordinates"].unique())),
                "Sector Centre Easting m": centre.x,
                "Sector Centre Northing m": centre.y,
                "Projected Border Position km": border_position / 1000,
                "Projected Border Easting m": anchor.x,
                "Projected Border Northing m": anchor.y,
            }
        )
    sectors = pd.DataFrame(sector_rows).sort_values("Projected Border Position km").reset_index(drop=True)
    if len(sectors) != 2:
        raise RuntimeError(f"Expected two 2011 conflict sectors after 30 km clustering, got {len(sectors)}")

    positions = sectors["Projected Border Position km"].to_numpy(float)
    separation = float(positions[1] - positions[0])
    to_wgs = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)
    grid_east = sample["Grid Centre Easting m"].to_numpy(float)
    grid_north = sample["Grid Centre Northing m"].to_numpy(float)
    actual_clean = sample["Candidate All 2008-2011 Nearest Event Distance km"].gt(60).to_numpy()
    rows = []
    placement_id = 0
    for first_position in np.arange(20, border.length / 1000 - separation - 20 + 1e-9, PLACEBO_ANCHOR_STEP_KM):
        candidate_positions = np.array([first_position, first_position + separation])
        minimum_actual_distance = float(
            np.min(np.abs(candidate_positions[:, None] - positions[None, :]))
        )
        if minimum_actual_distance < PLACEBO_ACTUAL_EXCLUSION_KM:
            continue
        anchors = [border.interpolate(position * 1000) for position in candidate_positions]
        distance = np.minimum.reduce(
            [
                np.hypot(grid_east - anchor.x, grid_north - anchor.y) / 1000
                for anchor in anchors
            ]
        )
        labels = pd.cut(distance, bins=RING_BINS, labels=RING_LABELS, ordered=True).astype(str)
        eligible_sample = actual_clean
        counts = pd.Series(labels[eligible_sample]).value_counts()
        block_counts = (
            pd.DataFrame(
                {
                    "Ring": labels[eligible_sample],
                    "Block": sample.loc[eligible_sample, "Spatial Block ID"].to_numpy(),
                }
            )
            .groupby("Ring")["Block"]
            .nunique()
        )
        support_pass = all(
            counts.get(label, 0) >= PLACEBO_MIN_CELLS[label]
            and block_counts.get(label, 0) >= PLACEBO_MIN_BLOCKS[label]
            for label in RING_LABELS
        )
        placement_id += 1
        lon1, lat1 = to_wgs.transform(anchors[0].x, anchors[0].y)
        lon2, lat2 = to_wgs.transform(anchors[1].x, anchors[1].y)
        row: dict[str, object] = {
            "Placebo Placement ID": f"P{placement_id:03d}",
            "First Border Position km": candidate_positions[0],
            "Second Border Position km": candidate_positions[1],
            "Sector Separation km": separation,
            "Minimum Along-Border Distance to Actual Sector km": minimum_actual_distance,
            "First Longitude": lon1,
            "First Latitude": lat1,
            "Second Longitude": lon2,
            "Second Latitude": lat2,
            "Eligible": support_pass,
        }
        for label in RING_LABELS:
            stem = label.replace(" ", "_").replace("-", "_")
            row[f"Cells_{stem}"] = int(counts.get(label, 0))
            row[f"Blocks_{stem}"] = int(block_counts.get(label, 0))
        rows.append(row)
    return sectors, pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = load_features()
    weighted, balance = generalized_overlap_weights(sample)
    support = support_summary(weighted)
    sectors, placebos = freeze_event_sectors_and_placebos(weighted)
    weighted.to_parquet(OUT_DIR / "gate_b_analysis_weights.parquet", index=False)
    balance.to_csv(OUT_DIR / "gate_b_ring_balance.csv", index=False)
    support.to_csv(OUT_DIR / "gate_b_ring_support.csv", index=False)
    sectors.to_csv(OUT_DIR / "gate_b_actual_conflict_sectors.csv", index=False)
    placebos.to_csv(OUT_DIR / "gate_b_placebo_site_universe.csv", index=False)
    metadata = {
        "status": "frozen before post-2007 outcome estimation",
        "sample": "Cambodian cells within 60 km of the Cambodia-Thailand border and either within 60 km of a 2011 event or more than 60 km from every 2008-2011 event",
        "rings": list(RING_LABELS),
        "weighting": "multinomial generalized overlap weights, normalized to equal total weight per ring",
        "weight_predictors": WEIGHT_PREDICTORS,
        "outcome_history_in_primary_weights": False,
        "border_analysis_sector": "one-degree longitude bin of nearest border point",
        "spatial_block": "10 by 10 national one-kilometre grid cells",
        "actual_sector_rule": "DBSCAN clustering of unique 2011 UCDP coordinates at 30 km, followed by projection of the unweighted coordinate centroid to the shared border",
        "placebo_rule": "translate the two-sector border template in 10 km increments, preserve actual along-border separation, exclude placements within 120 km along-border of either actual sector, and apply frozen cell/block support thresholds",
        "placebo_minimum_cells": PLACEBO_MIN_CELLS,
        "placebo_minimum_blocks": PLACEBO_MIN_BLOCKS,
        "post_2007_outcomes_loaded": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (OUT_DIR / "gate_b_protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(support.to_string(index=False))
    print("\nMaximum absolute SMD by ring")
    print(balance.groupby("Distance Ring")["Absolute Standardized Mean Difference"].max())
    print(f"\nPlacebo placements: {len(placebos)}; eligible: {int(placebos['Eligible'].sum())}")
    print("\nActual conflict sectors")
    print(sectors.to_string(index=False))


if __name__ == "__main__":
    main()
