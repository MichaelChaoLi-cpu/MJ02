#!/usr/bin/env python3
"""Test Gate B LongNTL dose estimates with climate-cell-by-year fixed effects."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from pyproj import Transformer

from diagnose_cambodia_thailand_common_support import PANEL
from estimate_cambodia_thailand_gate_b_longntl import (
    OUTCOME,
    PERIOD_ORDER,
    REFERENCE_PERIOD,
    period_from_year,
)
from freeze_cambodia_thailand_gate_b_protocol import OUT_DIR, load_features
from test_cambodia_thailand_gate_b_placebo_sites import make_weights


WEIGHTS = OUT_DIR / "gate_b_binary_support_weights.parquet"
PLACEBOS = OUT_DIR / "gate_b_placebo_site_universe.csv"


def fit_local_model(frame: pd.DataFrame, columns: list[str]):
    panel = frame.set_index(["National Grid Cell ID", "Year"]).sort_index()
    exog = panel[columns].copy()
    exog.insert(0, "Constant", 1.0)
    other_effects = pd.DataFrame(
        {"Climate Cell Year": pd.Categorical(panel["Climate Cell Year"]).codes},
        index=panel.index,
    )
    clusters = pd.DataFrame(
        {"Spatial Block": pd.Categorical(panel["Spatial Block ID"]).codes},
        index=panel.index,
    )
    return PanelOLS(
        panel[OUTCOME],
        exog,
        weights=panel["Gate B Binary Support Overlap Weight"],
        entity_effects=True,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=False,
    ).fit(cov_type="clustered", clusters=clusters)


def estimate_periods(frame: pd.DataFrame) -> dict[str, float]:
    frame = frame.copy()
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
    result = fit_local_model(frame, columns)
    pre_sd = float(frame.loc[frame["Year"].between(2000, 2007), OUTCOME].std(ddof=1))
    output = {}
    for period, column in mapping.items():
        estimate = float(result.params[column]) / pre_sd
        standard_error = float(result.std_errors[column]) / pre_sd
        output[f"{period} Estimate"] = estimate
        output[f"{period} Standard Error"] = standard_error
    return output


def main() -> None:
    panel = pd.read_parquet(
        PANEL, columns=["National Grid Cell ID", "Year", "Climate Cell ID", OUTCOME]
    ).loc[lambda frame: frame["Year"].between(2000, 2024)]
    panel = panel.dropna(subset=[OUTCOME]).copy()
    panel["Analysis Period"] = period_from_year(panel["Year"])
    panel["Climate Cell Year"] = panel["Climate Cell ID"] + "__" + panel["Year"].astype(str)

    weights = pd.read_parquet(WEIGHTS)
    actual = panel.merge(weights, on="National Grid Cell ID", validate="many_to_one")
    actual["Continuous Conflict Dose"] = np.where(
        actual["Conflict Distance Ring"].eq("Over 60 km"),
        0.0,
        np.maximum(
            0,
            1 - actual["Candidate Conflict Year 2011 Nearest Event Distance km"] / 60,
        ),
    )
    actual_estimates = estimate_periods(actual)

    base = load_features()
    base = base.loc[base["Candidate All 2008-2011 Nearest Event Distance km"].gt(60)].copy()
    placebos = pd.read_csv(PLACEBOS).loc[lambda frame: frame["Eligible"]].copy()
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    rows = []
    for _, placebo in placebos.iterrows():
        east1, north1 = to_utm.transform(placebo["First Longitude"], placebo["First Latitude"])
        east2, north2 = to_utm.transform(placebo["Second Longitude"], placebo["Second Latitude"])
        distance = np.minimum(
            np.hypot(base["Grid Centre Easting m"] - east1, base["Grid Centre Northing m"] - north1),
            np.hypot(base["Grid Centre Easting m"] - east2, base["Grid Centre Northing m"] - north2),
        ) / 1000
        treated = distance <= 60
        candidate = base.copy()
        candidate["Gate B Binary Support Overlap Weight"], max_smd, block_ess = make_weights(
            candidate, treated
        )
        candidate["Continuous Conflict Dose"] = np.maximum(0, 1 - distance / 60)
        frame = panel.merge(
            candidate[
                [
                    "National Grid Cell ID",
                    "Spatial Block ID",
                    "Gate B Binary Support Overlap Weight",
                    "Continuous Conflict Dose",
                ]
            ],
            on="National Grid Cell ID",
            validate="many_to_one",
        )
        print(f"Climate-cell FE: {placebo['Placebo Placement ID']}", flush=True)
        estimates = estimate_periods(frame)
        rows.append(
            {
                "Placebo Placement ID": placebo["Placebo Placement ID"],
                "Maximum Absolute Weighted SMD": max_smd,
                "Treated 10 km Block ESS": block_ess,
                **estimates,
            }
        )
    placebo_results = pd.DataFrame(rows)
    conflict_column = "Conflict year (2011) Estimate"
    recovery_column = "Early recovery (2012-2014) Estimate"
    actual_conflict = actual_estimates[conflict_column]
    actual_recovery = actual_estimates[recovery_column]
    summary = pd.DataFrame(
        [
            {
                "Period": "Conflict year (2011)",
                "Actual Standardized Estimate": actual_conflict,
                "Placebo Placements": len(placebo_results),
                "One-Sided Spatial Placebo p-value": (1 + (placebo_results[conflict_column] <= actual_conflict).sum()) / (1 + len(placebo_results)),
            },
            {
                "Period": "Early recovery (2012-2014)",
                "Actual Standardized Estimate": actual_recovery,
                "Placebo Placements": len(placebo_results),
                "One-Sided Spatial Placebo p-value": (1 + (placebo_results[recovery_column] <= actual_recovery).sum()) / (1 + len(placebo_results)),
            },
        ]
    )
    pd.DataFrame([{"Location": "Actual 2011 conflict sectors", **actual_estimates}]).to_csv(
        OUT_DIR / "gate_b_longntl_climate_cell_time_fe_actual.csv", index=False
    )
    placebo_results.to_csv(OUT_DIR / "gate_b_longntl_climate_cell_time_fe_placebos.csv", index=False)
    summary.to_csv(OUT_DIR / "gate_b_longntl_climate_cell_time_fe_inference.csv", index=False)
    metadata = {
        "purpose": "absorb every climate-cell-specific annual shock, including spatially varying 2011 weather",
        "fixed_effects": "grid cell and climate-cell by calendar year",
        "weather_controls": "absorbed by climate-cell-by-year effects",
        "weights": "outcome-independent binary frontier-support overlap weights, re-estimated for each placebo placement",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (OUT_DIR / "gate_b_longntl_climate_cell_time_fe_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nActual estimates")
    print(pd.Series(actual_estimates).to_string())
    print("\nSpatial inference")
    print(summary.to_string(index=False))
    print("\nPlacebo conflict-year estimates")
    print(placebo_results[["Placebo Placement ID", conflict_column]].to_string(index=False))


if __name__ == "__main__":
    main()
