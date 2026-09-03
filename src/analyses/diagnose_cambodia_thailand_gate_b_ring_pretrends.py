#!/usr/bin/env python3
"""Diagnose Gate B ring pretrends under binary frontier-support overlap weights."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mj02-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from diagnose_cambodia_thailand_common_support import FIXED_PREDICTORS, PANEL, PRE_CLIMATE_PREDICTORS
from freeze_cambodia_thailand_gate_b_protocol import (
    OUT_DIR,
    RING_LABELS,
    WEIGHT_PREDICTORS,
    effective_sample_size,
    load_features,
    weighted_mean_variance,
)


OUTCOME = "Asinh Annual NPP-VIIRS-like Radiance"
REFERENCE_YEAR = 2007
PRE_YEARS = tuple(range(2000, REFERENCE_YEAR))
MATERIAL_THRESHOLD_SD = 0.10


def binary_support_weights(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    treated = ~sample["Conflict Distance Ring"].eq("Over 60 km")
    predictors = FIXED_PREDICTORS + PRE_CLIMATE_PREDICTORS
    x = StandardScaler().fit_transform(sample[predictors])
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")
    model.fit(x, treated.astype(int))
    propensity = np.clip(model.predict_proba(x)[:, 1], 1e-8, 1 - 1e-8)
    weight = np.where(treated, 1 - propensity, propensity)
    weight[treated] *= (len(sample) / 2) / weight[treated].sum()
    weight[~treated] *= (len(sample) / 2) / weight[~treated].sum()
    output = sample.copy()
    output["Within 60 km of 2011 Event"] = treated.to_numpy()
    output["Binary Frontier Propensity Score"] = propensity
    output["Gate B Binary Support Overlap Weight"] = weight

    rows = []
    treated_array = treated.to_numpy()
    for variable in predictors:
        treated_mean, treated_variance = weighted_mean_variance(
            output.loc[treated_array, variable].to_numpy(float), weight[treated_array]
        )
        control_mean, control_variance = weighted_mean_variance(
            output.loc[~treated_array, variable].to_numpy(float), weight[~treated_array]
        )
        denominator = np.sqrt((treated_variance + control_variance) / 2)
        smd = (treated_mean - control_mean) / denominator if denominator > 0 else 0.0
        rows.append(
            {
                "Variable": variable,
                "Weighted Exposed Mean": treated_mean,
                "Weighted Frontier Control Mean": control_mean,
                "Standardized Mean Difference": smd,
                "Absolute Standardized Mean Difference": abs(smd),
            }
        )
    return output, pd.DataFrame(rows)


def ring_support(weighted: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ring in RING_LABELS:
        group = weighted.loc[weighted["Conflict Distance Ring"].eq(ring)]
        weights = group["Gate B Binary Support Overlap Weight"].to_numpy(float)
        block_weights = group.groupby("Spatial Block ID")["Gate B Binary Support Overlap Weight"].sum()
        rows.append(
            {
                "Distance Ring": ring,
                "Grid Cells": len(group),
                "Spatial Blocks": group["Spatial Block ID"].nunique(),
                "Cell ESS under Binary Support Weights": effective_sample_size(weights),
                "10 km Block ESS under Binary Support Weights": effective_sample_size(
                    block_weights.to_numpy(float)
                ),
            }
        )
    return pd.DataFrame(rows)


def estimate_pretrends(weighted: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_parquet(
        PANEL, columns=["National Grid Cell ID", "Year", OUTCOME]
    ).loc[lambda frame: frame["Year"].between(2000, REFERENCE_YEAR)]
    keep = weighted[
        [
            "National Grid Cell ID",
            "Candidate Conflict Year 2011 Nearest Event Distance km",
            "Conflict Distance Ring",
            "Border Analysis Sector",
            "Spatial Block ID",
            "Gate B Binary Support Overlap Weight",
        ]
    ]
    frame = panel.merge(keep, on="National Grid Cell ID", validate="many_to_one").dropna(
        subset=[OUTCOME]
    )
    frame["Sector Year"] = frame["Border Analysis Sector"] + "__" + frame["Year"].astype(str)
    exog_columns = []
    for ring in RING_LABELS[:-1]:
        stem = ring.replace(" ", "").replace("-", "to")
        for year in PRE_YEARS:
            column = f"Ring{stem}Year{year}"
            frame[column] = (
                frame["Conflict Distance Ring"].eq(ring).astype(float)
                * frame["Year"].eq(year).astype(float)
            )
            exog_columns.append(column)
    outcome_sd = float(frame[OUTCOME].std(ddof=1))
    panel_frame = frame.set_index(["National Grid Cell ID", "Year"]).sort_index()
    exog = panel_frame[exog_columns].copy()
    exog.insert(0, "Constant", 1.0)
    other_effects = pd.DataFrame(
        {"Sector Year": pd.Categorical(panel_frame["Sector Year"]).codes},
        index=panel_frame.index,
    )
    model = PanelOLS(
        panel_frame[OUTCOME],
        exog,
        weights=panel_frame["Gate B Binary Support Overlap Weight"],
        entity_effects=True,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=False,
    )
    clusters = pd.DataFrame(
        {"Spatial Block": pd.Categorical(panel_frame["Spatial Block ID"]).codes},
        index=panel_frame.index,
    )
    result = model.fit(cov_type="clustered", clusters=clusters)
    coefficient_rows = []
    joint_rows = []
    for ring in RING_LABELS[:-1]:
        stem = ring.replace(" ", "").replace("-", "to")
        parameters = [f"Ring{stem}Year{year}" for year in PRE_YEARS]
        restriction = np.zeros((len(parameters), len(result.params)))
        for row, parameter in enumerate(parameters):
            restriction[row, result.params.index.get_loc(parameter)] = 1
        joint = result.wald_test(restriction)
        ring_coefficients = []
        for year, parameter in zip(PRE_YEARS, parameters, strict=True):
            estimate = float(result.params[parameter])
            standard_error = float(result.std_errors[parameter])
            standardized = estimate / outcome_sd
            ring_coefficients.append(standardized)
            coefficient_rows.append(
                {
                    "Distance Ring": ring,
                    "Year": year,
                    "Reference Year": REFERENCE_YEAR,
                    "Estimate": estimate,
                    "Standard Error": standard_error,
                    "Lower 95 CI": estimate - 1.96 * standard_error,
                    "Upper 95 CI": estimate + 1.96 * standard_error,
                    "Standardized Estimate": standardized,
                    "Standardized Standard Error": standard_error / outcome_sd,
                }
            )
        maximum = float(np.max(np.abs(ring_coefficients)))
        joint_rows.append(
            {
                "Distance Ring": ring,
                "Reference Ring": "Over 60 km",
                "Reference Year": REFERENCE_YEAR,
                "Joint Wald Statistic": float(joint.stat),
                "Joint Wald p-value": float(joint.pval),
                "Maximum Absolute Lead SD": maximum,
                "Below Material 0.10 SD Threshold": maximum < MATERIAL_THRESHOLD_SD,
            }
        )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(joint_rows)


def plot_pretrends(coefficients: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for axis, ring in zip(axes.flat, RING_LABELS[:-1], strict=True):
        part = coefficients.loc[coefficients["Distance Ring"].eq(ring)].sort_values("Year")
        axis.errorbar(
            part["Year"],
            part["Standardized Estimate"],
            yerr=1.96 * part["Standardized Standard Error"],
            fmt="o-",
            color="#2C6E9B",
            capsize=2,
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.axhline(MATERIAL_THRESHOLD_SD, color="#A64B3C", linestyle="--", linewidth=0.8)
        axis.axhline(-MATERIAL_THRESHOLD_SD, color="#A64B3C", linestyle="--", linewidth=0.8)
        axis.set_title(f"{ring} vs over 60 km")
        axis.set_xlabel("Pre-conflict year")
        axis.set_ylabel("Standardized gap vs 2007")
        axis.grid(alpha=0.2)
    fig.suptitle("Gate B LongNTL ring pretrend diagnostic", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_b_ring_pretrend_diagnostics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample = load_features()
    weighted, balance = binary_support_weights(sample)
    support = ring_support(weighted)
    coefficients, joint = estimate_pretrends(weighted)
    weighted[
        [
            "National Grid Cell ID",
            "Candidate Conflict Year 2011 Nearest Event Distance km",
            "Conflict Distance Ring",
            "Border Analysis Sector",
            "Spatial Block ID",
            "Within 60 km of 2011 Event",
            "Binary Frontier Propensity Score",
            "Gate B Binary Support Overlap Weight",
        ]
    ].to_parquet(OUT_DIR / "gate_b_binary_support_weights.parquet", index=False)
    balance.to_csv(OUT_DIR / "gate_b_binary_support_balance.csv", index=False)
    support.to_csv(OUT_DIR / "gate_b_binary_weight_ring_support.csv", index=False)
    coefficients.to_csv(OUT_DIR / "gate_b_ring_pretrend_coefficients.csv", index=False)
    joint.to_csv(OUT_DIR / "gate_b_ring_pretrend_joint_tests.csv", index=False)
    plot_pretrends(coefficients)
    metadata = {
        "status": "outcome-blind Gate B repair diagnostic",
        "primary_weight_estimand": "within 60 km of a 2011 event versus strict over-60-km frontier controls",
        "ring_role": "pre-specified exposure heterogeneity within the binary support weights",
        "weight_predictors": FIXED_PREDICTORS + PRE_CLIMATE_PREDICTORS,
        "fixed_effects": "grid cell and one-degree border-sector by year",
        "inference": "10 km spatial-block clustered covariance",
        "reference_year": REFERENCE_YEAR,
        "material_pretrend_threshold_sd": MATERIAL_THRESHOLD_SD,
        "post_2007_outcomes_loaded": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (OUT_DIR / "gate_b_ring_pretrend_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Binary support balance max abs SMD:", balance["Absolute Standardized Mean Difference"].max())
    print(support.to_string(index=False))
    print("\nRing pretrend tests")
    print(joint.to_string(index=False))


if __name__ == "__main__":
    main()
