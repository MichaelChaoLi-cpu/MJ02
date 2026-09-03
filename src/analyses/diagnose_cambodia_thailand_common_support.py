#!/usr/bin/env python3
"""Outcome-blind common-support diagnostics for the border-conflict experiment.

Only fixed geography and information dated no later than 2007 enter the design
features.  Post-2007 outcomes are never loaded by this script.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
EXPOSURE = ROOT / "data/processed/cambodia_national_ucdp_border_conflict_exposure_preprocessed.parquet"
COVARIATES = ROOT / "data/processed/cambodia_national_predetermined_covariates_preprocessed.parquet"
PANEL = ROOT / "data/processed/cambodia_national_annual_satellite_climate_panel_preprocessed.parquet"
CLIMATE = ROOT / "data/processed/cambodia_national_annual_climate_hazards_preprocessed.parquet"
OUT_DIR = ROOT / "data/exp/experiment-design/cambodia-thailand-common-support"

RADII_KM = (10, 20, 40, 60)
STRICT_CONTROL_DISTANCE_KM = 60
PROPENSITY_NEIGHBOURS_SEARCHED = 500
PROPENSITY_CALIPER_SD = 0.20
LOCAL_CONTROL_ANNULI_KM = {
    10: (20, 40),
    20: (40, 60),
    40: (60, 100),
    60: (80, 120),
}

FIXED_PREDICTORS = [
    "Longitude",
    "Latitude",
    "Distance to Cambodia Thailand Border km",
    "Mean Elevation m",
    "Elevation SD m",
    "Mean Slope Degrees",
    "Slope P90 Degrees",
    "Steep Terrain Share",
    "Historical Road Density Proxy km per km2",
    "Distance to Nearest Historical Road Proxy km",
    "Baseline Cropland Share",
    "Baseline Forest Share",
    "Baseline Grass Shrub Share",
    "Baseline Built Share",
    "Baseline Water Wetland Share",
    "Baseline Land Cover Stability Share",
    "Log Baseline Population 2000",
]

PRE_OUTCOME_PREDICTORS = [
    "Pre 2008 Mean Asinh LongNTL",
    "Pre 2008 Asinh LongNTL Trend per Year",
    "Pre 2008 Nonzero LongNTL Share",
    "Pre 2008 Mean Annual NPP",
    "Pre 2008 Annual NPP Trend per Year",
]

PRE_CLIMATE_PREDICTORS = [
    "Pre 2008 Mean Annual Precipitation mm",
    "Pre 2008 SD Annual Precipitation mm",
    "Pre 2008 Mean May October Precipitation mm",
    "Pre 2008 Mean Maximum Five Day Precipitation mm",
    "Pre 2008 Mean Maximum Consecutive Dry Days",
    "Pre 2008 Mean May October Maximum Temperature C",
    "Pre 2008 Mean Hot Day Count",
]

PREDICTORS = FIXED_PREDICTORS + PRE_OUTCOME_PREDICTORS + PRE_CLIMATE_PREDICTORS


def grouped_slope(frame: pd.DataFrame, value: str, output: str) -> pd.DataFrame:
    sample = frame[["National Grid Cell ID", "Year", value]].dropna().copy()
    group = sample.groupby("National Grid Cell ID", sort=False)
    x_bar = group["Year"].transform("mean")
    y_bar = group[value].transform("mean")
    sample["Numerator"] = (sample["Year"] - x_bar) * (sample[value] - y_bar)
    sample["Denominator"] = (sample["Year"] - x_bar) ** 2
    result = sample.groupby("National Grid Cell ID", sort=False)[
        ["Numerator", "Denominator"]
    ].sum()
    result[output] = result["Numerator"] / result["Denominator"]
    return result[[output]].reset_index()


def construct_pre_outcomes() -> pd.DataFrame:
    columns = [
        "National Grid Cell ID",
        "Year",
        "Climate Cell ID",
        "Asinh Annual NPP-VIIRS-like Radiance",
        "Any Nonzero Annual NPP-VIIRS-like Radiance",
        "Annual Land NPP Mean kg C per m2",
    ]
    panel = pd.read_parquet(PANEL, columns=columns)
    panel = panel.loc[panel["Year"].between(2000, 2007)].copy()
    if panel["Year"].max() > 2007:
        raise RuntimeError("Post-2007 observations entered the design feature sample")
    grouped = panel.groupby("National Grid Cell ID", sort=False)
    output = grouped.agg(
        **{
            "Climate Cell ID": ("Climate Cell ID", "first"),
            "Pre 2008 Mean Asinh LongNTL": (
                "Asinh Annual NPP-VIIRS-like Radiance",
                "mean",
            ),
            "Pre 2008 Nonzero LongNTL Share": (
                "Any Nonzero Annual NPP-VIIRS-like Radiance",
                "mean",
            ),
            "Pre 2008 Mean Annual NPP": ("Annual Land NPP Mean kg C per m2", "mean"),
        }
    ).reset_index()
    output = output.merge(
        grouped_slope(
            panel,
            "Asinh Annual NPP-VIIRS-like Radiance",
            "Pre 2008 Asinh LongNTL Trend per Year",
        ),
        on="National Grid Cell ID",
        how="left",
        validate="one_to_one",
    )
    output = output.merge(
        grouped_slope(
            panel,
            "Annual Land NPP Mean kg C per m2",
            "Pre 2008 Annual NPP Trend per Year",
        ),
        on="National Grid Cell ID",
        how="left",
        validate="one_to_one",
    )
    return output


def construct_pre_climate() -> pd.DataFrame:
    variables = [
        "Annual Precipitation Total mm",
        "May October Precipitation Total mm",
        "May October Maximum Five-Day Precipitation mm",
        "May October Maximum Consecutive Dry Days",
        "May October Maximum Daily Temperature C",
        "May October Hot Day Count",
    ]
    climate = pd.read_parquet(CLIMATE, columns=["Climate Cell ID", "Year", *variables])
    climate = climate.loc[climate["Year"].between(1991, 2007)].copy()
    grouped = climate.groupby("Climate Cell ID", sort=False)
    output = grouped.agg(
        **{
            "Pre 2008 Mean Annual Precipitation mm": (
                "Annual Precipitation Total mm",
                "mean",
            ),
            "Pre 2008 SD Annual Precipitation mm": (
                "Annual Precipitation Total mm",
                "std",
            ),
            "Pre 2008 Mean May October Precipitation mm": (
                "May October Precipitation Total mm",
                "mean",
            ),
            "Pre 2008 Mean Maximum Five Day Precipitation mm": (
                "May October Maximum Five-Day Precipitation mm",
                "mean",
            ),
            "Pre 2008 Mean Maximum Consecutive Dry Days": (
                "May October Maximum Consecutive Dry Days",
                "mean",
            ),
            "Pre 2008 Mean May October Maximum Temperature C": (
                "May October Maximum Daily Temperature C",
                "mean",
            ),
            "Pre 2008 Mean Hot Day Count": ("May October Hot Day Count", "mean"),
        }
    ).reset_index()
    return output


def weighted_mean_variance(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, variance


def balance_rows(
    x: pd.DataFrame,
    treated: np.ndarray,
    control_weights: np.ndarray,
    radius: int,
    reservoir: str,
    stage: str,
    treated_weights: np.ndarray | None = None,
) -> list[dict[str, object]]:
    rows = []
    if treated_weights is None:
        treated_weights = np.ones(int(treated.sum()), dtype=float)
    controls = ~treated
    for variable in PREDICTORS:
        treated_mean, treated_variance = weighted_mean_variance(
            x.loc[treated, variable].to_numpy(float), treated_weights
        )
        control_mean, control_variance = weighted_mean_variance(
            x.loc[controls, variable].to_numpy(float), control_weights
        )
        denominator = np.sqrt((treated_variance + control_variance) / 2)
        smd = (treated_mean - control_mean) / denominator if denominator > 0 else 0.0
        rows.append(
            {
                "Treatment Radius km": radius,
                "Control Reservoir": reservoir,
                "Balance Stage": stage,
                "Variable": variable,
                "Treated Mean": treated_mean,
                "Control Mean": control_mean,
                "Standardized Mean Difference": smd,
                "Absolute Standardized Mean Difference": abs(smd),
            }
        )
    return rows


def diagnose_one(
    features: pd.DataFrame, radius: int, reservoir: str
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    treated_flag = features[f"Candidate Conflict Year 2011 Exposure Within {radius} km"].astype(bool)
    if reservoir == "local_annulus":
        lower, upper = LOCAL_CONTROL_ANNULI_KM[radius]
        strict_control = features[
            "Candidate Conflict Year 2011 Nearest Event Distance km"
        ].gt(lower) & features["Candidate Conflict Year 2011 Nearest Event Distance km"].le(upper)
    else:
        strict_control = features["Candidate All 2008-2011 Nearest Event Distance km"].gt(
            STRICT_CONTROL_DISTANCE_KM
        )
    if reservoir == "frontier_60km":
        strict_control &= features["Within 60 km of Cambodia Thailand Border"].astype(bool)
    elif reservoir not in ("national", "local_annulus"):
        raise ValueError(reservoir)
    sample = features.loc[treated_flag | strict_control].copy().reset_index(drop=True)
    treated = sample[f"Candidate Conflict Year 2011 Exposure Within {radius} km"].to_numpy(bool)
    x = sample[PREDICTORS].copy()
    if x.isna().any().any():
        raise RuntimeError(
            f"Missing design features for radius={radius}, reservoir={reservoir}: "
            f"{x.isna().sum()[x.isna().sum().gt(0)].to_dict()}"
        )

    scaler = StandardScaler()
    standardized = scaler.fit_transform(x)
    model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
    model.fit(standardized, treated.astype(int))
    propensity = np.clip(model.predict_proba(standardized)[:, 1], 1e-6, 1 - 1e-6)
    logit = np.log(propensity / (1 - propensity))
    auc = roc_auc_score(treated.astype(int), propensity)
    control_propensity = propensity[~treated]
    treated_propensity = propensity[treated]
    support_lower = max(float(treated_propensity.min()), float(control_propensity.min()))
    support_upper = min(float(treated_propensity.max()), float(control_propensity.max()))
    treated_in_support = (treated_propensity >= support_lower) & (
        treated_propensity <= support_upper
    )

    controls_x = standardized[~treated]
    treated_x = standardized[treated]
    treated_logit = logit[treated]
    control_logit = logit[~treated]
    caliper = PROPENSITY_CALIPER_SD * float(np.std(logit, ddof=1))
    propensity_neighbours = NearestNeighbors(
        n_neighbors=min(PROPENSITY_NEIGHBOURS_SEARCHED, len(controls_x)), algorithm="auto"
    ).fit(control_logit[:, None])
    _, candidates = propensity_neighbours.kneighbors(treated_logit[:, None])
    selected_control = np.full(len(treated_x), -1, dtype=int)
    selected_distance = np.full(len(treated_x), np.nan, dtype=float)
    for index in range(len(treated_x)):
        eligible = np.abs(control_logit[candidates[index]] - treated_logit[index]) <= caliper
        if eligible.any():
            eligible_candidates = candidates[index][eligible]
            distances = np.sqrt(
                np.mean((controls_x[eligible_candidates] - treated_x[index]) ** 2, axis=1)
            )
            best = int(np.argmin(distances))
            selected_control[index] = int(eligible_candidates[best])
            selected_distance[index] = float(distances[best])

    matched_treated = selected_control >= 0
    control_weights = np.bincount(
        selected_control[matched_treated], minlength=int((~treated).sum())
    ).astype(float)
    before_weights = np.ones(int((~treated).sum()), dtype=float)
    before_balance = balance_rows(x, treated, before_weights, radius, reservoir, "before")

    overlap_treated_weights = 1 - treated_propensity
    overlap_control_weights = control_propensity
    overlap_balance = balance_rows(
        x,
        treated,
        overlap_control_weights,
        radius,
        reservoir,
        "overlap_weighted",
        treated_weights=overlap_treated_weights,
    )

    # Balance after matching is calculated on matched treated cells and their
    # duplicated-control weights.  Rebuild a compact matched sample so treated
    # and control rows remain correctly aligned for the generic balance helper.
    treated_frame = x.loc[treated].reset_index(drop=True).loc[matched_treated].copy()
    controls_frame = x.loc[~treated].reset_index(drop=True).copy()
    matched_sample = pd.concat([treated_frame, controls_frame], ignore_index=True)
    matched_flag = np.r_[np.ones(len(treated_frame), dtype=bool), np.zeros(len(controls_frame), dtype=bool)]
    after_balance = balance_rows(
        matched_sample,
        matched_flag,
        control_weights,
        radius,
        reservoir,
        "after",
    )

    treated_ids = sample.loc[treated, "National Grid Cell ID"].reset_index(drop=True)
    control_ids = sample.loc[~treated, "National Grid Cell ID"].reset_index(drop=True)
    pairs = pd.DataFrame(
        {
            "Treatment Radius km": radius,
            "Control Reservoir": reservoir,
            "Treated Grid Cell ID": treated_ids,
            "Matched": matched_treated,
            "Matched Control Grid Cell ID": [
                control_ids.iloc[index] if index >= 0 else pd.NA for index in selected_control
            ],
            "Scaled Covariate Distance": selected_distance,
            "Treated Propensity Score": treated_propensity,
            "Control Propensity Score": [
                control_propensity[index] if index >= 0 else np.nan for index in selected_control
            ],
        }
    )

    after_absolute = np.array(
        [row["Absolute Standardized Mean Difference"] for row in after_balance], dtype=float
    )
    overlap_absolute = np.array(
        [row["Absolute Standardized Mean Difference"] for row in overlap_balance], dtype=float
    )
    positive_weights = control_weights[control_weights > 0]
    control_ess = (
        float(positive_weights.sum() ** 2 / np.sum(positive_weights**2))
        if len(positive_weights)
        else 0.0
    )
    overlap_treated_ess = float(
        overlap_treated_weights.sum() ** 2 / np.sum(overlap_treated_weights**2)
    )
    overlap_control_ess = float(
        overlap_control_weights.sum() ** 2 / np.sum(overlap_control_weights**2)
    )
    treated_commune_weights = (
        sample.loc[treated, ["Commune Code"]]
        .assign(Weight=overlap_treated_weights)
        .groupby("Commune Code")["Weight"]
        .sum()
        .to_numpy()
    )
    control_commune_weights = (
        sample.loc[~treated, ["Commune Code"]]
        .assign(Weight=overlap_control_weights)
        .groupby("Commune Code")["Weight"]
        .sum()
        .to_numpy()
    )
    treated_commune_ess = float(
        treated_commune_weights.sum() ** 2 / np.sum(treated_commune_weights**2)
    )
    control_commune_ess = float(
        control_commune_weights.sum() ** 2 / np.sum(control_commune_weights**2)
    )
    block_columns = (sample["Grid Column"] // 10).astype(int)
    block_rows = (sample["Grid Row"] // 10).astype(int)
    block_id = block_columns.astype(str) + "_" + block_rows.astype(str)
    treated_block_weights = (
        pd.DataFrame({"Block": block_id[treated].to_numpy(), "Weight": overlap_treated_weights})
        .groupby("Block")["Weight"]
        .sum()
        .to_numpy()
    )
    control_block_weights = (
        pd.DataFrame({"Block": block_id[~treated].to_numpy(), "Weight": overlap_control_weights})
        .groupby("Block")["Weight"]
        .sum()
        .to_numpy()
    )
    treated_block_ess = float(
        treated_block_weights.sum() ** 2 / np.sum(treated_block_weights**2)
    )
    control_block_ess = float(
        control_block_weights.sum() ** 2 / np.sum(control_block_weights**2)
    )
    standardized_mde_80 = float(
        2.80 * np.sqrt(1 / treated_block_ess + 1 / control_block_ess)
    )
    treated_rows = sample.loc[treated]
    matched_control_ids = pairs.loc[pairs["Matched"], "Matched Control Grid Cell ID"].dropna()
    matched_control_rows = sample.loc[
        sample["National Grid Cell ID"].isin(matched_control_ids)
    ]
    summary = {
        "Treatment Radius km": radius,
        "Control Reservoir": reservoir,
        "Treated Grid Cells": int(treated.sum()),
        "Treated Communes": int(treated_rows["Commune Code"].nunique()),
        "Treated Provinces": int(treated_rows["Province Code"].nunique()),
        "Candidate Control Grid Cells": int((~treated).sum()),
        "Candidate Control Communes": int(sample.loc[~treated, "Commune Code"].nunique()),
        "Propensity AUC": float(auc),
        "Treated Within Propensity Support Share": float(treated_in_support.mean()),
        "Propensity Support Lower": support_lower,
        "Propensity Support Upper": support_upper,
        "Propensity Logit Caliper": caliper,
        "Matched Treated Grid Cells": int(matched_treated.sum()),
        "Matched Treated Share": float(matched_treated.mean()),
        "Unique Matched Control Grid Cells": int(matched_control_ids.nunique()),
        "Unique Matched Control Communes": int(matched_control_rows["Commune Code"].nunique()),
        "Matched Control Cell ESS": control_ess,
        "Median Scaled Covariate Distance": float(np.nanmedian(selected_distance)),
        "After Match Mean Absolute SMD": float(np.mean(after_absolute)),
        "After Match Maximum Absolute SMD": float(np.max(after_absolute)),
        "After Match Variables Above 0.10 SMD": int((after_absolute > 0.10).sum()),
        "After Match Variables Above 0.20 SMD": int((after_absolute > 0.20).sum()),
        "Overlap Weighted Treated ESS": overlap_treated_ess,
        "Overlap Weighted Control ESS": overlap_control_ess,
        "Overlap Weighted Mean Absolute SMD": float(np.mean(overlap_absolute)),
        "Overlap Weighted Maximum Absolute SMD": float(np.max(overlap_absolute)),
        "Overlap Weighted Variables Above 0.10 SMD": int((overlap_absolute > 0.10).sum()),
        "Overlap Weighted Treated Commune ESS": treated_commune_ess,
        "Overlap Weighted Control Commune ESS": control_commune_ess,
        "Overlap Weighted Treated 10km Block ESS": treated_block_ess,
        "Overlap Weighted Control 10km Block ESS": control_block_ess,
        "Screening Standardized MDE 80 Percent Power": standardized_mde_80,
    }
    weights = sample[
        [
            "National Grid Cell ID",
            "Province Code",
            "Province Name",
            "District Code",
            "District Name",
            "Commune Code",
            "Commune Name",
            "Grid Row",
            "Grid Column",
        ]
    ].copy()
    weights.insert(0, "Control Reservoir", reservoir)
    weights.insert(0, "Treatment Radius km", radius)
    weights["Treated"] = treated
    weights["Propensity Score"] = propensity
    weights["Overlap Weight"] = np.where(treated, 1 - propensity, propensity)
    weights["Matched Control Use Count"] = 0
    weights.loc[~treated, "Matched Control Use Count"] = control_weights.astype(int)
    return summary, before_balance + after_balance + overlap_balance, pairs, weights


def main() -> None:
    missing = [path for path in (EXPOSURE, COVARIATES, PANEL, CLIMATE) if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed = pd.read_parquet(COVARIATES)
    exposure_columns = [
        "National Grid Cell ID",
        "Candidate All 2008-2011 Nearest Event Distance km",
        "Candidate Conflict Year 2011 Nearest Event Distance km",
        *[
            f"Candidate Conflict Year 2011 Exposure Within {radius} km"
            for radius in RADII_KM
        ],
    ]
    exposure = pd.read_parquet(EXPOSURE, columns=exposure_columns)
    features = fixed.merge(exposure, on="National Grid Cell ID", validate="one_to_one")
    pre_outcomes = construct_pre_outcomes()
    features = features.merge(pre_outcomes, on="National Grid Cell ID", validate="one_to_one")
    pre_climate = construct_pre_climate()
    features = features.merge(pre_climate, on="Climate Cell ID", validate="many_to_one")
    if len(features) != 179_072:
        raise RuntimeError("Design feature merge changed national grid size")
    inventory = []
    for variable in PREDICTORS:
        timing = (
            "time invariant"
            if variable in FIXED_PREDICTORS
            else "2000-2007"
            if variable in PRE_OUTCOME_PREDICTORS
            else "1991-2007"
        )
        inventory.append(
            {
                "Variable": variable,
                "Timing": timing,
                "Minimum": float(features[variable].min()),
                "Maximum": float(features[variable].max()),
                "Missing Cells": int(features[variable].isna().sum()),
            }
        )
    pd.DataFrame(inventory).to_csv(OUT_DIR / "design_feature_inventory.csv", index=False)

    complete_design = ~features[PREDICTORS].isna().any(axis=1)
    excluded = features.loc[
        ~complete_design,
        [
            "National Grid Cell ID",
            "Province Code",
            "Province Name",
            "District Code",
            "District Name",
            "Commune Code",
            "Commune Name",
            *[
                f"Candidate Conflict Year 2011 Exposure Within {radius} km"
                for radius in RADII_KM
            ],
        ],
    ].copy()
    excluded["Exclusion Reason"] = "Incomplete 1991-2007 climate history"
    excluded.to_csv(OUT_DIR / "design_sample_exclusions.csv", index=False)
    if excluded[
        [f"Candidate Conflict Year 2011 Exposure Within {radius} km" for radius in RADII_KM]
    ].to_numpy(bool).any():
        raise RuntimeError("A treated grid cell has incomplete design features")
    features = features.loc[complete_design].copy().reset_index(drop=True)

    summaries = []
    balances = []
    pairs = []
    weights = []
    for radius in RADII_KM:
        for reservoir in ("local_annulus", "frontier_60km", "national"):
            print(f"Diagnosing radius={radius} km, reservoir={reservoir}", flush=True)
            summary, balance, matched_pairs, design_weights = diagnose_one(
                features, radius, reservoir
            )
            summaries.append(summary)
            balances.extend(balance)
            pairs.append(matched_pairs)
            weights.append(design_weights)

    summary_frame = pd.DataFrame(summaries)
    balance_frame = pd.DataFrame(balances)
    pair_frame = pd.concat(pairs, ignore_index=True)
    weight_frame = pd.concat(weights, ignore_index=True)
    summary_frame.to_csv(OUT_DIR / "common_support_summary.csv", index=False)
    balance_frame.to_csv(OUT_DIR / "covariate_balance.csv", index=False)
    pair_frame.to_parquet(OUT_DIR / "matched_pairs_diagnostic.parquet", index=False)
    weight_frame.to_parquet(OUT_DIR / "design_weights_diagnostic.parquet", index=False)

    metadata = {
        "status": "outcome-blind design diagnostic; not an effect estimate",
        "primary_event_phase": "UCDP conflict year 2011",
        "precursor_phase_role": "2008-2009 robustness only",
        "treatment_radii_km": list(RADII_KM),
        "strict_control_rule": "more than 60 km from every candidate 2008-2011 event",
        "control_reservoirs": {
            "local_annulus": (
                "radius-specific local outer rings: 20-40, 40-60, 60-100, and "
                "80-120 km for 10, 20, 40, and 60 km treatments"
            ),
            "frontier_60km": "strict controls within 60 km of the Cambodia-Thailand border",
            "national": "all strict controls in Cambodia",
        },
        "feature_timing": "fixed geography, 2000-2007 satellite outcomes, and 1991-2007 climate only",
        "matching": (
            "one-to-one nearest neighbour with replacement on standardized features, "
            "subject to 0.2 pooled-SD propensity-logit caliper"
        ),
        "overlap_weighting": "treated weight 1-p and control weight p from the same pre-treatment logistic propensity model",
        "screening_mde": "(1.96 + 0.84) times sqrt(1/treated 10-km-block ESS + 1/control 10-km-block ESS); descriptive design screen only",
        "post_2007_outcomes_loaded": False,
        "national_grid_cells": 179_072,
        "design_eligible_grid_cells": int(len(features)),
        "excluded_incomplete_pre_climate_cells": int(len(excluded)),
        "excluded_treated_cells": 0,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (OUT_DIR / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
