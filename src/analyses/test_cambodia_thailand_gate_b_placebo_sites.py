#!/usr/bin/env python3
"""Run frozen spatial placebo-site tests for the Gate B LongNTL dose effect."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from diagnose_cambodia_thailand_common_support import FIXED_PREDICTORS, PANEL, PRE_CLIMATE_PREDICTORS
from estimate_cambodia_thailand_gate_b_longntl import (
    OUTCOME,
    PERIOD_ORDER,
    REFERENCE_PERIOD,
    WEATHER_CONTROLS,
    fit_model,
    period_from_year,
)
from freeze_cambodia_thailand_gate_b_protocol import OUT_DIR, load_features, weighted_mean_variance


PLACEBOS = OUT_DIR / "gate_b_placebo_site_universe.csv"
ACTUAL = OUT_DIR / "gate_b_longntl_continuous_dose_coefficients.csv"
PREDICTORS = FIXED_PREDICTORS + PRE_CLIMATE_PREDICTORS
BALANCE_THRESHOLD = 0.10
MIN_TREATED_BLOCK_ESS = 10


def effective_sample_size(values: np.ndarray) -> float:
    return float(values.sum() ** 2 / np.sum(values**2))


def make_weights(sample: pd.DataFrame, treated: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = StandardScaler().fit_transform(sample[PREDICTORS])
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(x, treated.astype(int))
    propensity = np.clip(model.predict_proba(x)[:, 1], 1e-8, 1 - 1e-8)
    weights = np.where(treated, 1 - propensity, propensity)
    weights[treated] *= (len(sample) / 2) / weights[treated].sum()
    weights[~treated] *= (len(sample) / 2) / weights[~treated].sum()
    absolute_smd = []
    for variable in PREDICTORS:
        mean_t, var_t = weighted_mean_variance(
            sample.loc[treated, variable].to_numpy(float), weights[treated]
        )
        mean_c, var_c = weighted_mean_variance(
            sample.loc[~treated, variable].to_numpy(float), weights[~treated]
        )
        denominator = np.sqrt((var_t + var_c) / 2)
        absolute_smd.append(abs((mean_t - mean_c) / denominator) if denominator > 0 else 0)
    block_weights = (
        sample.loc[treated, ["Spatial Block ID"]]
        .assign(Weight=weights[treated])
        .groupby("Spatial Block ID")["Weight"]
        .sum()
        .to_numpy()
    )
    return weights, float(max(absolute_smd)), effective_sample_size(block_weights)


def estimate_one(
    base: pd.DataFrame,
    panel: pd.DataFrame,
    first_easting: float,
    first_northing: float,
    second_easting: float,
    second_northing: float,
) -> dict[str, float]:
    distance = np.minimum(
        np.hypot(base["Grid Centre Easting m"] - first_easting, base["Grid Centre Northing m"] - first_northing),
        np.hypot(base["Grid Centre Easting m"] - second_easting, base["Grid Centre Northing m"] - second_northing),
    ) / 1000
    sample = base.copy()
    sample["Placebo Distance km"] = distance
    treated = distance <= 60
    weights, max_smd, treated_block_ess = make_weights(sample, treated)
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
    return {
        "Maximum Absolute Weighted SMD": max_smd,
        "Treated 10 km Block ESS": treated_block_ess,
        "Conflict Year Standardized Estimate": float(result.params[mapping["Conflict year (2011)"]]) / pre_sd,
        "Early Recovery Standardized Estimate": float(result.params[mapping["Early recovery (2012-2014)"]]) / pre_sd,
        "Precursor Standardized Estimate": float(result.params[mapping["Precursor escalation (2008-2010)"]]) / pre_sd,
    }


def main() -> None:
    base = load_features()
    base = base.loc[base["Candidate All 2008-2011 Nearest Event Distance km"].gt(60)].copy()
    panel = pd.read_parquet(
        PANEL, columns=["National Grid Cell ID", "Year", OUTCOME, *WEATHER_CONTROLS]
    ).loc[lambda frame: frame["Year"].between(2000, 2024)]
    panel = panel.dropna(subset=[OUTCOME, *WEATHER_CONTROLS]).copy()
    panel["Analysis Period"] = period_from_year(panel["Year"])
    placebos = pd.read_csv(PLACEBOS).loc[lambda frame: frame["Eligible"]].copy()
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    rows = []
    for index, placebo in placebos.iterrows():
        east1, north1 = to_utm.transform(placebo["First Longitude"], placebo["First Latitude"])
        east2, north2 = to_utm.transform(placebo["Second Longitude"], placebo["Second Latitude"])
        print(f"Estimating {placebo['Placebo Placement ID']}", flush=True)
        estimates = estimate_one(base, panel, east1, north1, east2, north2)
        rows.append({**placebo.to_dict(), **estimates})
    results = pd.DataFrame(rows)
    results["Passes Estimation Support"] = (
        results["Maximum Absolute Weighted SMD"].le(BALANCE_THRESHOLD)
        & results["Treated 10 km Block ESS"].ge(MIN_TREATED_BLOCK_ESS)
    )
    actual = pd.read_csv(ACTUAL).set_index("Period")
    actual_conflict = float(actual.loc["Conflict year (2011)", "Standardized Estimate"])
    actual_recovery = float(actual.loc["Early recovery (2012-2014)", "Standardized Estimate"])
    valid = results.loc[results["Passes Estimation Support"]]
    conflict_p = (1 + (valid["Conflict Year Standardized Estimate"] <= actual_conflict).sum()) / (1 + len(valid))
    recovery_p = (1 + (valid["Early Recovery Standardized Estimate"] <= actual_recovery).sum()) / (1 + len(valid))
    summary = pd.DataFrame(
        [
            {
                "Outcome Period": "Conflict year (2011)",
                "Actual Standardized Estimate": actual_conflict,
                "Eligible Placebo Placements": len(placebos),
                "Support-Passing Placebo Placements": len(valid),
                "One-Sided Spatial Placebo p-value": conflict_p,
            },
            {
                "Outcome Period": "Early recovery (2012-2014)",
                "Actual Standardized Estimate": actual_recovery,
                "Eligible Placebo Placements": len(placebos),
                "Support-Passing Placebo Placements": len(valid),
                "One-Sided Spatial Placebo p-value": recovery_p,
            },
        ]
    )
    results.to_csv(OUT_DIR / "gate_b_longntl_placebo_site_estimates.csv", index=False)
    summary.to_csv(OUT_DIR / "gate_b_longntl_placebo_site_inference.csv", index=False)
    metadata = {
        "placebo_sample": "actual-event-clean frontier cells more than 60 km from every 2008-2011 event",
        "weights": "placement-specific outcome-independent binary overlap weights",
        "support_thresholds": {
            "maximum_absolute_smd": BALANCE_THRESHOLD,
            "minimum_treated_10km_block_ess": MIN_TREATED_BLOCK_ESS,
        },
        "p_value": "one-sided rank p=(1 + number of support-passing placebo estimates no greater than actual)/(1 + support-passing placebos)",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (OUT_DIR / "gate_b_longntl_placebo_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nSpatial placebo inference")
    print(summary.to_string(index=False))
    print("\nPlacebo support")
    print(results[["Placebo Placement ID", "Maximum Absolute Weighted SMD", "Treated 10 km Block ESS", "Conflict Year Standardized Estimate", "Passes Estimation Support"]].to_string(index=False))


if __name__ == "__main__":
    main()
