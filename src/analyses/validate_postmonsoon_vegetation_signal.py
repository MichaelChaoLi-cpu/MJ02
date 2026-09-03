#!/usr/bin/env python3
"""Validate the post-monsoon vegetation response to frozen monsoon structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


INPUT = Path("data/processed/cambodia_public_village_monsoon_ecology_panel_preprocessed.parquet")
OUTPUT = Path("data/exp/analysis/climate-welfare/postmonsoon-vegetation-validation")
ID = "National Village Point ID"
YEAR = "Year"
LON = "Point Longitude"
LAT = "Point Latitude"
EVI = "Village Buffer Mean November-February Mean EVI Anomaly Z"
NDVI = "Village Buffer Mean November-February Mean NDVI Anomaly Z"
RAIN = "Village Buffer Mean May October Precipitation Total mm Anomaly Z"
ONSET_B = "Village Buffer Mean Wet-Season Onset DOY Candidate B Anomaly Z"
DRY_B = "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z"
HEAT_B = "Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate B"
MODELS_B = {
    "Rainfall totals only": [RAIN],
    "Monsoon structure only": [ONSET_B, DRY_B, HEAT_B],
    "Joint rainfall and monsoon structure": [RAIN, ONSET_B, DRY_B, HEAT_B],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def spatial_block(frame: pd.DataFrame) -> pd.Series:
    return (
        np.floor((frame[LON] - 102.0) / 0.75).astype("Int64").astype(str)
        + "_"
        + np.floor((frame[LAT] - 10.0) / 0.75).astype("Int64").astype(str)
    )


def fit(frame: pd.DataFrame, outcome: str, variables: list[str], label: str) -> tuple[object, list[dict]]:
    sample = frame.dropna(subset=[outcome] + variables).copy()
    panel = sample.set_index([ID, YEAR]).sort_index()
    clusters = pd.DataFrame(
        {"Spatial Block": pd.Categorical(panel["Spatial Block"]).codes}, index=panel.index
    )
    result = PanelOLS(
        panel[outcome], panel[variables], entity_effects=True, time_effects=True, drop_absorbed=True
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)
    interval = result.conf_int(level=0.95)
    rows = [
        {
            "Specification": label,
            "Outcome": outcome,
            "Exposure": exposure,
            "Coefficient Outcome SD per Exposure Unit": float(result.params[exposure]),
            "Exposure Unit": "10 heat days" if exposure == HEAT_B else "1 SD",
            "Clustered Standard Error": float(result.std_errors[exposure]),
            "95 Percent CI Lower": float(interval.loc[exposure, "lower"]),
            "95 Percent CI Upper": float(interval.loc[exposure, "upper"]),
            "Probability Value": float(result.pvalues[exposure]),
            "Observations": int(result.nobs),
            "Villages": int(sample[ID].nunique()),
            "Within R2": float(result.rsquared_within),
            "Inclusive R2": float(result.rsquared_inclusive),
        }
        for exposure in variables
    ]
    return result, rows


def assign_folds(frame: pd.DataFrame) -> pd.Series:
    counts = frame.groupby("Spatial Block", observed=True).size().sort_values(ascending=False)
    totals = [0] * 5
    mapping = {}
    for block, count in counts.items():
        target = int(np.argmin(totals))
        mapping[str(block)] = target
        totals[target] += int(count)
    return frame["Spatial Block"].map(mapping).astype(int)


def pooled_prediction(train: pd.DataFrame, test: pd.DataFrame, outcome: str, variables: list[str]) -> np.ndarray:
    """Predict raw standardized anomalies with an inductive pooled model.

    Fixed-effects coefficients remain the inferential Model A estimates.  The
    held-out measurement comparison is deliberately separate: it uses an
    intercept and only variables observable for new villages and future years.
    """
    train_x = np.column_stack([np.ones(len(train)), train[variables].to_numpy(float)])
    test_x = np.column_stack([np.ones(len(test)), test[variables].to_numpy(float)])
    beta = np.linalg.lstsq(train_x, train[outcome].to_numpy(float), rcond=None)[0]
    return test_x @ beta


def crossfit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = frame.dropna(subset=[EVI] + MODELS_B["Joint rainfall and monsoon structure"]).copy()
    sample["Spatial Fold"] = assign_folds(sample)
    year_map = {
        int(year): fold
        for fold, years in enumerate(np.array_split(sorted(sample[YEAR].unique()), 5))
        for year in years
    }
    sample["Temporal Fold"] = sample[YEAR].map(year_map).astype(int)
    predictions = []
    fold_rows = []
    for spatial_fold in range(5):
        for temporal_fold in range(5):
            test = sample.loc[
                sample["Spatial Fold"].eq(spatial_fold)
                & sample["Temporal Fold"].eq(temporal_fold)
            ].copy()
            train = sample.loc[
                sample["Spatial Fold"].ne(spatial_fold)
                & sample["Temporal Fold"].ne(temporal_fold)
            ].copy()
            out = test[[ID, YEAR, EVI]].copy()
            for model, variables in MODELS_B.items():
                predicted = pooled_prediction(train, test, EVI, variables)
                out[f"Prediction: {model}"] = predicted
                error = test[EVI].to_numpy() - predicted
                fold_rows.append(
                    {
                        "Model": model,
                        "Spatial Fold": spatial_fold,
                        "Temporal Fold": temporal_fold,
                        "Observations": len(test),
                        "RMSE SD": float(np.sqrt(np.mean(error**2))),
                    }
                )
            predictions.append(out)
    predictions = pd.concat(predictions, ignore_index=True)
    comparison = []
    observed = predictions[EVI].to_numpy()
    for model in MODELS_B:
        predicted = predictions[f"Prediction: {model}"].to_numpy()
        error = observed - predicted
        comparison.append(
            {
                "Model": model,
                "Observations": len(observed),
                "Villages": predictions[ID].nunique(),
                "RMSE SD": float(np.sqrt(np.mean(error**2))),
                "MAE SD": float(np.mean(np.abs(error))),
                "R2 versus Zero-Anomaly Benchmark": float(1 - np.sum(error**2) / np.sum(observed**2)),
                "Squared Correlation": float(np.corrcoef(observed, predicted)[0, 1] ** 2),
            }
        )
    return pd.DataFrame(fold_rows), pd.DataFrame(comparison)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not (root / INPUT).exists():
        raise FileNotFoundError(root / INPUT)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    columns = [
        "Buffer Radius km", ID, YEAR, LON, LAT, EVI, NDVI, RAIN,
        ONSET_B, DRY_B, HEAT_B,
    ]
    frame = pd.read_parquet(root / INPUT, columns=columns)
    frame = frame.loc[frame[YEAR].between(2001, 2023)].copy()
    frame[HEAT_B] = frame[HEAT_B] / 10.0
    frame["Spatial Block"] = spatial_block(frame)
    rows: list[dict] = []
    for radius in (2, 5, 10):
        radius_frame = frame.loc[frame["Buffer Radius km"].eq(radius)]
        for outcome in (EVI, NDVI):
            _, coefficient_rows = fit(
                radius_frame,
                outcome,
                [RAIN, ONSET_B, DRY_B, HEAT_B],
                f"Candidate B, 35 C, {radius} km",
            )
            rows.extend(coefficient_rows)
    five = frame.loc[frame["Buffer Radius km"].eq(5)]
    for years in np.array_split(np.arange(2001, 2024), 5):
        excluded = f"{int(years.min())}-{int(years.max())}"
        _, coefficient_rows = fit(
            five.loc[~five[YEAR].isin(years)],
            EVI,
            [RAIN, ONSET_B, DRY_B, HEAT_B],
            f"Candidate B, 35 C, 5 km, excluding years {excluded}",
        )
        rows.extend(coefficient_rows)
    coefficients = pd.DataFrame(rows)
    coefficients.to_csv(output / "coefficients.csv", index=False)
    fold_metrics, comparison = crossfit(five)
    fold_metrics.to_csv(output / "crossfit_fold_metrics.csv", index=False)
    comparison.to_csv(output / "crossfit_model_comparison.csv", index=False)

    dry = coefficients.loc[coefficients["Exposure"].eq(DRY_B)].copy()
    core = dry.loc[
        dry["Specification"].isin(
            [
                "Candidate B, 35 C, 2 km", "Candidate B, 35 C, 5 km",
                "Candidate B, 35 C, 10 km",
            ]
        )
    ]
    exclusions = dry.loc[dry["Specification"].str.contains("excluding years")]
    rmse = comparison.set_index("Model")["RMSE SD"]
    improvement = float(
        (rmse["Rainfall totals only"] - rmse["Joint rainfall and monsoon structure"])
        / rmse["Rainfall totals only"]
        * 100
    )
    pass_rule = bool(
        core["Coefficient Outcome SD per Exposure Unit"].lt(0).all()
        and exclusions["Coefficient Outcome SD per Exposure Unit"].lt(0).all()
        and improvement > 0
    )
    summary = {
        "signal": "intraseasonal dry-spell anomaly to November-February vegetation anomaly",
        "core_evi_ndvi_radius_coefficients_all_negative": bool(
            core["Coefficient Outcome SD per Exposure Unit"].lt(0).all()
        ),
        "leave_one_temporal_block_out_evi_coefficients_all_negative": bool(
            exclusions["Coefficient Outcome SD per Exposure Unit"].lt(0).all()
        ),
        "joint_vs_rainfall_crossfit_rmse_improvement_percent": improvement,
        "validation_gate": "pass" if pass_rule else "not passed",
        "gate_rule": "negative dry-spell direction across EVI and NDVI at 2/5/10 km under the single Candidate B definition, negative in every temporal-block exclusion, and positive inductive spatial-temporal held-out RMSE improvement",
    }
    (output / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Post-monsoon vegetation validation\n\n"
        "This validation tests the prespecified November-February decomposition across sensors, "
        "buffer radii, temporal-block exclusions, and inductive spatial-temporal "
        "cross-fitting.\n\n"
        f"Validation gate: **{summary['validation_gate']}**\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(dry.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
