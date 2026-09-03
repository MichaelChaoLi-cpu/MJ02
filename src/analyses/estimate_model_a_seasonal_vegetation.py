#!/usr/bin/env python3
"""Test whether frozen monsoon structure predicts season-aligned vegetation response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


INPUT = Path(
    "data/processed/cambodia_public_village_monsoon_ecology_panel_preprocessed.parquet"
)
OUTPUT = Path("data/exp/analysis/climate-welfare/model-a-seasonal-vegetation")
ID = "National Village Point ID"
YEAR = "Year"
RADIUS = "Buffer Radius km"
LON = "Point Longitude"
LAT = "Point Latitude"
RAIN = "Village Buffer Mean May October Precipitation Total mm Anomaly Z"
ONSET = "Village Buffer Mean Wet-Season Onset DOY Candidate B Anomaly Z"
DRY = "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z"
HEAT = "Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate B"
PRIMARY = "Village Buffer Mean May-February Production-Season Mean EVI Anomaly Z"
MAY_OCT_EVI = "Village Buffer Mean May-October Mean EVI Anomaly Z"
NOV_FEB_EVI = "Village Buffer Mean November-February Mean EVI Anomaly Z"
PRIMARY_NDVI = "Village Buffer Mean May-February Production-Season Mean NDVI Anomaly Z"
MODELS = {
    "Rainfall totals only": [RAIN],
    "Monsoon structure only": [ONSET, DRY, HEAT],
    "Joint rainfall and monsoon structure": [RAIN, ONSET, DRY, HEAT],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def make_spatial_block(frame: pd.DataFrame, degrees: float = 0.75) -> pd.Series:
    lon = np.floor((frame[LON] - 102.0) / degrees).astype("Int64")
    lat = np.floor((frame[LAT] - 10.0) / degrees).astype("Int64")
    return lon.astype(str) + "_" + lat.astype(str)


def fit_panel(
    frame: pd.DataFrame, outcome: str, variables: list[str], model: str, role: str
) -> tuple[object, dict, list[dict]]:
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
            "Outcome": outcome,
            "Model": model,
            "Result Role": role,
            "Variable": variable,
            "Coefficient Outcome SD per Exposure Unit": float(result.params[variable]),
            "Exposure Unit": "10 heat days" if variable == HEAT else "1 SD",
            "Clustered Standard Error": float(result.std_errors[variable]),
            "95 Percent CI Lower": float(interval.loc[variable, "lower"]),
            "95 Percent CI Upper": float(interval.loc[variable, "upper"]),
            "Probability Value": float(result.pvalues[variable]),
            "Observations": int(result.nobs),
            "Villages": int(sample[ID].nunique()),
        }
        for variable in result.params.index
    ]
    summary = {
        "Outcome": outcome,
        "Model": model,
        "Result Role": role,
        "Observations": int(result.nobs),
        "Villages": int(sample[ID].nunique()),
        "Years": int(sample[YEAR].nunique()),
        "Spatial Blocks": int(sample["Spatial Block"].nunique()),
        "Within R2": float(result.rsquared_within),
        "Inclusive R2": float(result.rsquared_inclusive),
        "Residual Sum of Squares": float(result.resid_ss),
    }
    return result, summary, rows


def assign_spatial_folds(frame: pd.DataFrame, folds: int = 5) -> pd.Series:
    counts = frame.groupby("Spatial Block", observed=True).size().sort_values(ascending=False)
    totals = [0] * folds
    mapping: dict[str, int] = {}
    for block, count in counts.items():
        target = int(np.argmin(totals))
        mapping[str(block)] = target
        totals[target] += int(count)
    return frame["Spatial Block"].map(mapping).astype(int)


def pooled_prediction(
    train: pd.DataFrame, test: pd.DataFrame, outcome: str, variables: list[str]
) -> np.ndarray:
    """Inductively predict raw standardized anomalies from observable climate fields."""
    train_x = np.column_stack([np.ones(len(train)), train[variables].to_numpy(float)])
    test_x = np.column_stack([np.ones(len(test)), test[variables].to_numpy(float)])
    beta = np.linalg.lstsq(train_x, train[outcome].to_numpy(float), rcond=None)[0]
    return test_x @ beta


def crossfit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample = frame.dropna(subset=[PRIMARY] + MODELS["Joint rainfall and monsoon structure"]).copy()
    sample["Spatial Fold"] = assign_spatial_folds(sample)
    year_mapping = {
        int(year): fold
        for fold, years in enumerate(np.array_split(sorted(sample[YEAR].unique()), 5))
        for year in years
    }
    sample["Temporal Fold"] = sample[YEAR].map(year_mapping).astype(int)
    predictions: list[pd.DataFrame] = []
    metrics: list[dict] = []
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
            if test.empty:
                continue
            out = test[[ID, YEAR, "Spatial Fold", "Temporal Fold", PRIMARY]].copy()
            for model, variables in MODELS.items():
                predicted = pooled_prediction(train, test, PRIMARY, variables)
                error = test[PRIMARY].to_numpy(dtype=float) - predicted
                out[f"Predicted EVI Component: {model}"] = predicted
                metrics.append(
                    {
                        "Model": model,
                        "Spatial Fold": spatial_fold,
                        "Temporal Fold": temporal_fold,
                        "Evaluation Observations": int(len(test)),
                        "Training Observations": int(len(train)),
                        "RMSE SD": float(np.sqrt(np.mean(error**2))),
                        "MAE SD": float(np.mean(np.abs(error))),
                        "Squared Correlation": float(np.corrcoef(test[PRIMARY], predicted)[0, 1] ** 2),
                    }
                )
            predictions.append(out)
    predicted_frame = pd.concat(predictions, ignore_index=True)
    comparisons = []
    observed = predicted_frame[PRIMARY].to_numpy(dtype=float)
    for model in MODELS:
        predicted = predicted_frame[f"Predicted EVI Component: {model}"].to_numpy(dtype=float)
        error = observed - predicted
        comparisons.append(
            {
                "Model": model,
                "Cross-Fitted Observations": int(len(observed)),
                "Cross-Fitted Villages": int(predicted_frame[ID].nunique()),
                "Cross-Fitted RMSE SD": float(np.sqrt(np.mean(error**2))),
                "Cross-Fitted MAE SD": float(np.mean(np.abs(error))),
                "Cross-Fitted R2 versus Zero-Anomaly Benchmark": float(
                    1 - np.sum(error**2) / np.sum(observed**2)
                ),
                "Cross-Fitted Squared Correlation": float(np.corrcoef(observed, predicted)[0, 1] ** 2),
            }
        )
    predicted_frame["Cross-Fitted Climate-Implied EVI Loss SD"] = -predicted_frame[
        "Predicted EVI Component: Joint rainfall and monsoon structure"
    ]
    return predicted_frame, pd.DataFrame(metrics), pd.DataFrame(comparisons)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not (root / INPUT).exists():
        raise FileNotFoundError(root / INPUT)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    columns = [RADIUS, ID, YEAR, LON, LAT, PRIMARY, MAY_OCT_EVI, NOV_FEB_EVI, PRIMARY_NDVI]
    columns += sorted({variable for values in MODELS.values() for variable in values})
    frame = pd.read_parquet(root / INPUT, columns=columns)
    frame = frame.loc[frame[RADIUS].eq(5) & frame[YEAR].between(2001, 2023)].copy()
    frame[HEAT] = frame[HEAT] / 10.0
    frame["Spatial Block"] = make_spatial_block(frame)

    rows: list[dict] = []
    summaries: list[dict] = []
    primary_results: dict[str, object] = {}
    common = frame.dropna(subset=[PRIMARY] + MODELS["Joint rainfall and monsoon structure"]).copy()
    for model, variables in MODELS.items():
        result, summary, coefficient_rows = fit_panel(common, PRIMARY, variables, model, "primary")
        primary_results[model] = result
        summaries.append(summary)
        rows.extend(coefficient_rows)
    for outcome, role in (
        (MAY_OCT_EVI, "within-monsoon timing decomposition"),
        (NOV_FEB_EVI, "post-monsoon timing decomposition"),
        (PRIMARY_NDVI, "sensor robustness"),
    ):
        _, summary, coefficient_rows = fit_panel(
            frame,
            outcome,
            MODELS["Joint rainfall and monsoon structure"],
            "Joint rainfall and monsoon structure",
            role,
        )
        summaries.append(summary)
        rows.extend(coefficient_rows)

    predictions, fold_metrics, crossfit_comparison = crossfit(frame)
    model_comparison = pd.DataFrame(summaries).merge(crossfit_comparison, on="Model", how="left")
    coefficients = pd.DataFrame(rows)
    joint = primary_results["Joint rainfall and monsoon structure"]
    joint_variables = MODELS["Joint rainfall and monsoon structure"]
    restriction = np.zeros((3, len(joint_variables)))
    for row, variable in enumerate((ONSET, DRY, HEAT)):
        restriction[row, joint_variables.index(variable)] = 1
    wald = joint.wald_test(restriction=restriction)
    rmse = crossfit_comparison.set_index("Model")["Cross-Fitted RMSE SD"]
    improvement = float(
        (rmse["Rainfall totals only"] - rmse["Joint rainfall and monsoon structure"])
        / rmse["Rainfall totals only"]
        * 100
    )
    signs = {variable: bool(joint.params[variable] < 0) for variable in (ONSET, DRY, HEAT)}
    passes = improvement > 0 and float(wald.pval) < 0.05 and signs[DRY] and signs[HEAT]
    gate = (
        "pass: production-season vegetation supports incremental monsoon-structure information"
        if passes
        else "not passed: production-season vegetation does not meet the frozen incremental-information criteria"
    )
    summary = {
        "experiment": "Season-aligned dynamic vegetation Model A",
        "primary_outcome": PRIMARY,
        "sample_observations": int(len(common)),
        "sample_villages": int(common[ID].nunique()),
        "years": [2001, 2023],
        "basis": "additive linear effects fixed before seasonal vegetation estimation",
        "joint_monsoon_structure_wald_statistic": float(wald.stat),
        "joint_monsoon_structure_wald_p_value": float(wald.pval),
        "joint_vs_rainfall_crossfit_rmse_improvement_percent": improvement,
        "expected_direction_checks": signs,
        "ecological_gate": gate,
        "gate_rule": "positive strict cross-fitted RMSE improvement, joint monsoon-structure Wald p below 0.05, and negative dry-spell and absolute-heat coefficients",
    }
    coefficients.to_csv(output / "coefficients.csv", index=False)
    model_comparison.to_csv(output / "model_comparison.csv", index=False)
    fold_metrics.to_csv(output / "crossfit_fold_metrics.csv", index=False)
    predictions.to_parquet(output / "crossfit_predictions.parquet", index=False)
    (output / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Season-aligned dynamic vegetation Model A\n\n"
        "The primary outcome is mean EVI anomaly over the fixed May-to-February production "
        "season. The model compares rainfall totals, frozen monsoon structure, and their joint "
        "specification on one common 5 km sample. Cross-fitting excludes both each evaluation "
        "spatial fold and temporal fold from training.\n\n"
        f"Ecological gate: **{gate}**\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(coefficients.to_string(index=False))
    print(model_comparison.to_string(index=False))


if __name__ == "__main__":
    main()
