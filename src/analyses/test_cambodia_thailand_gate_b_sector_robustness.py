#!/usr/bin/env python3
"""Estimate Gate B LongNTL dose paths by 2011 conflict sector and 0-5 km exclusion."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import DBSCAN

from diagnose_cambodia_thailand_common_support import PANEL
from estimate_cambodia_thailand_gate_b_longntl import (
    OUTCOME,
    PERIOD_ORDER,
    REFERENCE_PERIOD,
    WEATHER_CONTROLS,
    fit_model,
    period_from_year,
)
from freeze_cambodia_thailand_gate_b_protocol import EVENTS, OUT_DIR, load_features
from test_cambodia_thailand_gate_b_placebo_sites import make_weights


def event_sector_coordinates() -> list[tuple[str, np.ndarray]]:
    events = pd.read_parquet(EVENTS)
    events = events.loc[events["year"].eq(2011), ["where_coordinates", "latitude", "longitude"]]
    events = events.drop_duplicates(["latitude", "longitude"]).reset_index(drop=True)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    east, north = transformer.transform(events["longitude"].to_numpy(), events["latitude"].to_numpy())
    coordinates = np.column_stack([east, north])
    labels = DBSCAN(eps=30_000, min_samples=1).fit_predict(coordinates)
    output = []
    for label in sorted(np.unique(labels)):
        part = events.loc[labels == label]
        name = " / ".join(sorted(part["where_coordinates"].unique()))
        output.append((name, coordinates[labels == label]))
    return output


def distance_to_sites(base: pd.DataFrame, sites: np.ndarray) -> np.ndarray:
    east = base["Grid Centre Easting m"].to_numpy(float)
    north = base["Grid Centre Northing m"].to_numpy(float)
    return np.minimum.reduce(
        [np.hypot(east - site[0], north - site[1]) / 1000 for site in sites]
    )


def estimate_scenario(
    base: pd.DataFrame,
    panel: pd.DataFrame,
    distance: np.ndarray,
    scenario: str,
    exclude_inner_5km: bool = False,
) -> dict[str, object]:
    strict_control = base["Candidate All 2008-2011 Nearest Event Distance km"].gt(60).to_numpy()
    treated = distance <= 60
    keep = treated | strict_control
    if exclude_inner_5km:
        keep &= distance > 5
    sample = base.loc[keep].copy()
    sample_distance = distance[keep]
    sample_treated = sample_distance <= 60
    weights, max_smd, block_ess = make_weights(sample, sample_treated)
    sample["Gate B Binary Support Overlap Weight"] = weights
    sample["Continuous Conflict Dose"] = np.maximum(0, 1 - sample_distance / 60)
    frame = panel.merge(
        sample[
            [
                "National Grid Cell ID",
                "Border Analysis Sector",
                "Spatial Block ID",
                "Gate B Binary Support Overlap Weight",
                "Continuous Conflict Dose",
            ]
        ],
        on="National Grid Cell ID",
        validate="many_to_one",
    )
    frame["Sector Year"] = frame["Border Analysis Sector"] + "__" + frame["Year"].astype(str)
    columns = []
    mapping = {}
    for period in PERIOD_ORDER:
        if period == REFERENCE_PERIOD:
            continue
        column = f"ContinuousDosePeriod{PERIOD_ORDER.index(period)}"
        frame[column] = (
            frame["Continuous Conflict Dose"]
            * frame["Analysis Period"].astype(str).eq(period).astype(float)
        )
        columns.append(column)
        mapping[period] = column
    result = fit_model(frame, columns)
    pre_sd = float(frame.loc[frame["Year"].between(2000, 2007), OUTCOME].std(ddof=1))
    row: dict[str, object] = {
        "Scenario": scenario,
        "Grid Cells": sample["National Grid Cell ID"].nunique(),
        "Treated Grid Cells": int(sample_treated.sum()),
        "Treated Spatial Blocks": sample.loc[sample_treated, "Spatial Block ID"].nunique(),
        "Treated 10 km Block ESS": block_ess,
        "Maximum Absolute Weighted SMD": max_smd,
    }
    for period, column in mapping.items():
        row[f"{period} Estimate"] = float(result.params[column]) / pre_sd
        row[f"{period} Standard Error"] = float(result.std_errors[column]) / pre_sd
    return row


def main() -> None:
    base = load_features()
    panel = pd.read_parquet(
        PANEL, columns=["National Grid Cell ID", "Year", OUTCOME, *WEATHER_CONTROLS]
    ).loc[lambda frame: frame["Year"].between(2000, 2024)]
    panel = panel.dropna(subset=[OUTCOME, *WEATHER_CONTROLS]).copy()
    panel["Analysis Period"] = period_from_year(panel["Year"])
    sectors = event_sector_coordinates()
    all_sites = np.vstack([sites for _, sites in sectors])
    all_distance = distance_to_sites(base, all_sites)
    rows = [
        estimate_scenario(base, panel, all_distance, "All 2011 event sectors"),
        estimate_scenario(
            base,
            panel,
            all_distance,
            "All sectors excluding 0-5 km",
            exclude_inner_5km=True,
        ),
    ]
    for name, sites in sectors:
        rows.append(
            estimate_scenario(
                base,
                panel,
                distance_to_sites(base, sites),
                f"Only {name}",
            )
        )
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "gate_b_longntl_sector_and_inner_zone_robustness.csv", index=False)
    metadata = {
        "sector_definition": "30 km DBSCAN clusters of unique 2011 UCDP coordinates",
        "inner_zone_exclusion": "remove cells within 5 km before re-estimating outcome-independent overlap weights",
        "model": "continuous 0-at-60-km distance dose with cell and one-degree border-sector-by-year fixed effects and frozen weather controls",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (OUT_DIR / "gate_b_longntl_sector_robustness_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    columns = [
        "Scenario",
        "Treated Grid Cells",
        "Treated 10 km Block ESS",
        "Maximum Absolute Weighted SMD",
        "Precursor escalation (2008-2010) Estimate",
        "Conflict year (2011) Estimate",
        "Conflict year (2011) Standard Error",
        "Early recovery (2012-2014) Estimate",
    ]
    print(results[columns].to_string(index=False))


if __name__ == "__main__":
    main()
