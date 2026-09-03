#!/usr/bin/env python3
"""Run the frozen two-sector spatial placebo test with the CCNL outcome."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pyproj import Transformer

from estimate_cambodia_thailand_gate_b_longntl import WEATHER_CONTROLS
from freeze_cambodia_thailand_gate_b_protocol import OUT_DIR, load_features
from test_cambodia_thailand_gate_b_placebo_sites import (
    BALANCE_THRESHOLD,
    MIN_TREATED_BLOCK_ESS,
    make_weights,
)
from validate_gate_b_with_ccnl_dmsp import COEFFICIENTS, REFERENCE_YEAR, fit_model, load_panel


PLACEBOS = OUT_DIR / "gate_b_placebo_site_universe.csv"
OUTCOME = "Asinh CCNL DMSP Corrected DN"
OUTPUT = OUT_DIR / "gate_b_ccnl_placebo_site_estimates.csv"
INFERENCE = OUT_DIR / "gate_b_ccnl_placebo_site_inference.csv"
METADATA = OUT_DIR / "gate_b_ccnl_placebo_metadata.json"


def estimate_one(
    base: pd.DataFrame,
    panel: pd.DataFrame,
    first_easting: float,
    first_northing: float,
    second_easting: float,
    second_northing: float,
) -> dict[str, float]:
    distance = np.minimum(
        np.hypot(
            base["Grid Centre Easting m"] - first_easting,
            base["Grid Centre Northing m"] - first_northing,
        ),
        np.hypot(
            base["Grid Centre Easting m"] - second_easting,
            base["Grid Centre Northing m"] - second_northing,
        ),
    ) / 1000
    treated = distance <= 60
    weights, maximum_smd, treated_block_ess = make_weights(base, treated)
    sample = base.copy()
    sample["Gate B Binary Support Overlap Weight"] = weights
    sample["Continuous Conflict Dose"] = np.maximum(0, 1 - distance / 60)
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
    frame = frame.dropna(subset=[OUTCOME, *WEATHER_CONTROLS]).copy()
    result, parameters = fit_model(frame, OUTCOME)
    reference_sd = float(frame.loc[frame["Year"].eq(REFERENCE_YEAR), OUTCOME].std(ddof=1))
    estimates = {
        year: float(result.params[parameter]) / reference_sd
        for year, parameter in zip((2011, 2012, 2013), parameters, strict=True)
    }
    return {
        "Maximum Absolute Weighted SMD": maximum_smd,
        "Treated 10 km Block ESS": treated_block_ess,
        "2011 Standardized Estimate": estimates[2011],
        "2012 Standardized Estimate": estimates[2012],
        "2013 Standardized Estimate": estimates[2013],
    }


def main() -> None:
    base = load_features()
    base = base.loc[base["Candidate All 2008-2011 Nearest Event Distance km"].gt(60)].copy()
    panel = load_panel()
    placebos = pd.read_csv(PLACEBOS).loc[lambda frame: frame["Eligible"]].copy()
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    rows = []
    for _, placebo in placebos.iterrows():
        east1, north1 = to_utm.transform(placebo["First Longitude"], placebo["First Latitude"])
        east2, north2 = to_utm.transform(placebo["Second Longitude"], placebo["Second Latitude"])
        print(f"Estimating CCNL {placebo['Placebo Placement ID']}", flush=True)
        rows.append(
            {
                **placebo.to_dict(),
                **estimate_one(base, panel, east1, north1, east2, north2),
            }
        )
    results = pd.DataFrame(rows)
    results["Passes Estimation Support"] = (
        results["Maximum Absolute Weighted SMD"].le(BALANCE_THRESHOLD)
        & results["Treated 10 km Block ESS"].ge(MIN_TREATED_BLOCK_ESS)
    )
    actual = pd.read_csv(COEFFICIENTS)
    actual_2011 = float(
        actual.loc[
            actual["Scenario"].eq("All 2011 event sectors")
            & actual["Product"].eq("CCNL")
            & actual["Outcome"].eq(OUTCOME)
            & actual["Year"].eq(2011),
            "Standardized Estimate",
        ].iloc[0]
    )
    valid = results.loc[results["Passes Estimation Support"]]
    no_greater = int((valid["2011 Standardized Estimate"] <= actual_2011).sum())
    p_value = (1 + no_greater) / (1 + len(valid))
    inference = pd.DataFrame(
        [
            {
                "Outcome": OUTCOME,
                "Year": 2011,
                "Actual Standardized Estimate": actual_2011,
                "Eligible Placebo Placements": len(placebos),
                "Support-Passing Placebo Placements": len(valid),
                "Placebo Estimates No Greater Than Actual": no_greater,
                "One-Sided Spatial Placebo p-value": p_value,
            }
        ]
    )
    results.to_csv(OUTPUT, index=False)
    inference.to_csv(INFERENCE, index=False)
    metadata = {
        "outcome": OUTCOME,
        "reference_year": REFERENCE_YEAR,
        "placebo_universe": "the frozen Gate B two-sector translation universe",
        "sample": "frontier cells more than 60 km from all actual 2008-2011 events",
        "weights": "placement-specific outcome-independent binary overlap weights",
        "p_value": "(1 + support-passing placebo estimates no greater than actual)/(1 + support-passing placebos)",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(inference.to_string(index=False))
    print("\nPlacebo estimates")
    print(
        results[
            [
                "Placebo Placement ID",
                "2011 Standardized Estimate",
                "Maximum Absolute Weighted SMD",
                "Treated 10 km Block ESS",
                "Passes Estimation Support",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
