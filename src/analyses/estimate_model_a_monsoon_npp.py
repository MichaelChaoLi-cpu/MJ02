#!/usr/bin/env python3
"""Estimate the frozen first-stage monsoon-to-NPP experiment.

The primary experiment compares rainfall totals, monsoon structure, and their
joint model on one common 5 km village-year sample. Candidate B and the linear
additive basis are fixed before this script reads NPP. Candidate A, nonlinear
response shapes, and other buffer radii are reserved for later robustness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


INPUT = Path(
    "data/processed/cambodia_public_village_monsoon_npp_panel_preprocessed.parquet"
)
OUTPUT = Path("data/exp/analysis/climate-welfare/model-a-monsoon-npp")

ID = "National Village Point ID"
YEAR = "Year"
RADIUS = "Buffer Radius km"
LON = "Point Longitude"
LAT = "Point Latitude"
OUTCOME = "Village Buffer Mean Annual Land NPP Anomaly kg C per m2"
RAIN = "Village Buffer Mean May October Precipitation Total mm Anomaly Z"
ONSET = "Village Buffer Mean Wet-Season Onset DOY Candidate B Anomaly Z"
DRY = "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z"
HEAT = "Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate B"

MODELS = {
    "Rainfall totals only": [RAIN],
    "Monsoon structure only": [ONSET, DRY, HEAT],
    "Joint rainfall and monsoon structure": [RAIN, ONSET, DRY, HEAT],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def spatial_block(frame: pd.DataFrame, degrees: float = 0.75) -> pd.Series:
    lon_bin = np.floor((frame[LON] - 102.0) / degrees).astype("Int64")
    lat_bin = np.floor((frame[LAT] - 10.0) / degrees).astype("Int64")
    return lon_bin.astype(str) + "_" + lat_bin.astype(str)


def fit_panel(frame: pd.DataFrame, variables: list[str], model_name: str) -> tuple[object, dict]:
    panel = frame.set_index([ID, YEAR]).sort_index()
    clusters = pd.DataFrame(
        {"Spatial Block": pd.Categorical(panel["Spatial Block"]).codes}, index=panel.index
    )
    model = PanelOLS(
        panel[OUTCOME],
        panel[variables],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )
    result = model.fit(cov_type="clustered", clusters=clusters, debiased=True)
    summary = {
        "Model": model_name,
        "Observations": int(result.nobs),
        "Villages": int(frame[ID].nunique()),
        "Years": int(frame[YEAR].nunique()),
        "Spatial Blocks": int(frame["Spatial Block"].nunique()),
        "Within R2": float(result.rsquared_within),
        "Inclusive R2": float(result.rsquared_inclusive),
        "Residual Sum of Squares": float(result.resid_ss),
        "Village Fixed Effects": "yes",
        "Year Fixed Effects": "yes",
        "Inference": "clustered by fixed 0.75-degree spatial block",
    }
    return result, summary


def coefficient_rows(result: object, model_name: str, role: str = "primary") -> list[dict]:
    interval = result.conf_int(level=0.95)
    return [
        {
            "Model": model_name,
            "Result Role": role,
            "Variable": variable,
            "Coefficient kg C per m2 per Exposure Unit": float(result.params[variable]),
            "Exposure Unit": "10 heat days" if variable == HEAT else "1 SD",
            "Clustered Standard Error": float(result.std_errors[variable]),
            "95 Percent CI Lower": float(interval.loc[variable, "lower"]),
            "95 Percent CI Upper": float(interval.loc[variable, "upper"]),
            "Probability Value": float(result.pvalues[variable]),
        }
        for variable in result.params.index
    ]


def assign_balanced_spatial_folds(frame: pd.DataFrame, folds: int = 5) -> pd.Series:
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
    """Inductively predict raw outcome anomalies from observable climate fields."""
    train_x = np.column_stack([np.ones(len(train)), train[variables].to_numpy(float)])
    test_x = np.column_stack([np.ones(len(test)), test[variables].to_numpy(float)])
    beta = np.linalg.lstsq(train_x, train[outcome].to_numpy(float), rcond=None)[0]
    return test_x @ beta


def crossfit(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Restrict to the complete panel so every fold compares identical observations.
    counts = frame.groupby(ID, observed=True).size()
    supported_ids = counts[counts.eq(frame[YEAR].nunique())].index
    sample = frame.loc[frame[ID].isin(supported_ids)].copy()
    sample["Spatial Fold"] = assign_balanced_spatial_folds(sample)
    year_groups = {
        int(year): fold
        for fold, years in enumerate(np.array_split(sorted(sample[YEAR].unique()), 5))
        for year in years
    }
    sample["Temporal Fold"] = sample[YEAR].map(year_groups).astype(int)

    prediction_parts: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    all_variables = sorted({variable for variables in MODELS.values() for variable in variables})
    for spatial_fold in range(5):
        for temporal_fold in range(5):
            evaluation = sample.loc[
                sample["Spatial Fold"].eq(spatial_fold)
                & sample["Temporal Fold"].eq(temporal_fold)
            ].copy()
            training = sample.loc[
                sample["Spatial Fold"].ne(spatial_fold)
                & sample["Temporal Fold"].ne(temporal_fold)
            ].copy()
            if evaluation.empty or training.empty:
                continue
            out = evaluation[[ID, YEAR, "Spatial Fold", "Temporal Fold", OUTCOME]].copy()
            for model_name, variables in MODELS.items():
                prediction = pooled_prediction(training, evaluation, OUTCOME, variables)
                out[f"Predicted NPP Component: {model_name}"] = prediction
                error = evaluation[OUTCOME].to_numpy(dtype=float) - prediction
                fold_rows.append(
                    {
                        "Model": model_name,
                        "Spatial Fold": spatial_fold,
                        "Temporal Fold": temporal_fold,
                        "Evaluation Observations": int(len(evaluation)),
                        "Training Observations": int(len(training)),
                        "RMSE kg C per m2": float(np.sqrt(np.mean(error**2))),
                        "MAE kg C per m2": float(np.mean(np.abs(error))),
                        "Squared Correlation": float(
                            np.corrcoef(evaluation[OUTCOME], prediction)[0, 1] ** 2
                        ),
                    }
                )
            prediction_parts.append(out)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    comparison_rows = []
    for model_name in MODELS:
        prediction = predictions[f"Predicted NPP Component: {model_name}"].to_numpy()
        observed = predictions[OUTCOME].to_numpy()
        error = observed - prediction
        comparison_rows.append(
            {
                "Model": model_name,
                "Cross-Fitted Observations": int(len(observed)),
                "Cross-Fitted Villages": int(predictions[ID].nunique()),
                "Cross-Fitted RMSE kg C per m2": float(np.sqrt(np.mean(error**2))),
                "Cross-Fitted MAE kg C per m2": float(np.mean(np.abs(error))),
                "Cross-Fitted R2 versus Zero-Anomaly Benchmark": float(
                    1 - np.sum(error**2) / np.sum(observed**2)
                ),
                "Cross-Fitted Squared Correlation": float(np.corrcoef(observed, prediction)[0, 1] ** 2),
            }
        )
    predictions["Cross-Fitted Climate-Implied Ecological Loss kg C per m2"] = -predictions[
        "Predicted NPP Component: Joint rainfall and monsoon structure"
    ]
    return predictions, fold_metrics, pd.DataFrame(comparison_rows)


def lagged_alignment_diagnostic(frame: pd.DataFrame) -> tuple[object, dict]:
    variables = MODELS["Joint rainfall and monsoon structure"]
    lagged = frame[[ID, YEAR] + variables].copy()
    lagged[YEAR] = lagged[YEAR] + 1
    diagnostic = frame.drop(columns=variables).merge(lagged, on=[ID, YEAR], how="inner")
    result, summary = fit_panel(
        diagnostic, variables, "Prior-monsoon to next-calendar-year NPP diagnostic"
    )
    summary["Result Role"] = "temporal-alignment diagnostic; not primary"
    return result, summary


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    input_path = root / INPUT
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)

    columns = [RADIUS, ID, YEAR, LON, LAT, OUTCOME] + sorted(
        {variable for variables in MODELS.values() for variable in variables}
    )
    frame = pd.read_parquet(input_path, columns=columns)
    frame = frame.loc[frame[RADIUS].eq(5)].dropna(subset=[OUTCOME] + list(MODELS.values())[-1]).copy()
    frame[HEAT] = frame[HEAT] / 10.0
    frame["Spatial Block"] = spatial_block(frame)
    if frame.duplicated([ID, YEAR]).any():
        raise RuntimeError("Duplicate village-year keys in Model A sample")

    coefficient_data: list[dict] = []
    comparison_data: list[dict] = []
    fitted: dict[str, object] = {}
    for model_name, variables in MODELS.items():
        result, summary = fit_panel(frame, variables, model_name)
        fitted[model_name] = result
        comparison_data.append(summary)
        coefficient_data.extend(coefficient_rows(result, model_name))

    lagged_result, lagged_summary = lagged_alignment_diagnostic(frame)
    comparison_data.append(lagged_summary)
    coefficient_data.extend(
        coefficient_rows(
            lagged_result,
            "Prior-monsoon to next-calendar-year NPP diagnostic",
            role="temporal-alignment diagnostic; not primary",
        )
    )

    coefficients = pd.DataFrame(coefficient_data)
    comparison = pd.DataFrame(comparison_data)
    predictions, fold_metrics, crossfit_comparison = crossfit(frame)
    totals_rmse = float(
        crossfit_comparison.loc[
            crossfit_comparison["Model"].eq("Rainfall totals only"),
            "Cross-Fitted RMSE kg C per m2",
        ].iloc[0]
    )
    joint_rmse = float(
        crossfit_comparison.loc[
            crossfit_comparison["Model"].eq("Joint rainfall and monsoon structure"),
            "Cross-Fitted RMSE kg C per m2",
        ].iloc[0]
    )
    comparison = comparison.merge(crossfit_comparison, on="Model", how="left")
    coefficients.to_csv(output / "coefficients.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    fold_metrics.to_csv(output / "crossfit_fold_metrics.csv", index=False)
    predictions.to_parquet(output / "crossfit_predictions.parquet", index=False)
    sample_audit = pd.DataFrame(
        [
            {
                "Sample": "Primary complete-case 5 km village-year panel",
                "Observations": len(frame),
                "Villages": frame[ID].nunique(),
                "Years": frame[YEAR].nunique(),
                "Spatial Blocks": frame["Spatial Block"].nunique(),
            },
            {
                "Sample": "Strict cross-fitting support",
                "Observations": len(predictions),
                "Villages": predictions[ID].nunique(),
                "Years": predictions[YEAR].nunique(),
                "Spatial Blocks": frame.loc[frame[ID].isin(predictions[ID]), "Spatial Block"].nunique(),
            },
        ]
    )
    sample_audit.to_csv(output / "sample_audit.csv", index=False)

    same_year = fitted["Joint rainfall and monsoon structure"]
    pass_predictive = joint_rmse < totals_rmse
    expected_signs = {ONSET: -1, DRY: -1, HEAT: -1}
    sign_check = {
        variable: bool(np.sign(same_year.params[variable]) == expected_sign)
        for variable, expected_sign in expected_signs.items()
    }
    summary = {
        "experiment": "Model A: monsoon structure versus rainfall totals for annual NPP",
        "primary_sample_observations": int(len(frame)),
        "primary_sample_villages": int(frame[ID].nunique()),
        "primary_years": [int(frame[YEAR].min()), int(frame[YEAR].max())],
        "basis": "additive linear effects fixed before NPP estimation",
        "inference": "village and year fixed effects; uncertainty clustered by fixed 0.75-degree spatial block",
        "cross_fitting": "25 strict spatial-by-temporal evaluation cells; training excludes the evaluation spatial fold and temporal fold",
        "joint_rmse": joint_rmse,
        "rainfall_only_rmse": totals_rmse,
        "joint_rmse_improvement_percent": float((totals_rmse - joint_rmse) / totals_rmse * 100),
        "timing_adds_out_of_fold_information": pass_predictive,
        "primary_expected_sign_check": sign_check,
        "ecological_gate": (
            "provisional pass for prediction; proceed to dynamic vegetation and scale robustness"
            if pass_predictive and all(sign_check.values())
            else "not passed by the same-calendar-year annual-NPP specification; inspect temporal aggregation and dynamic vegetation before household models"
        ),
        "lagged_diagnostic_is_primary": False,
    }
    (output / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Model A monsoon-to-NPP experiment\n\n"
        "This experiment uses the frozen Candidate B climate measures and the primary absolute "
        "35 C heat-day definition on one common 5 km "
        "village-year sample. The primary model includes village and year fixed effects. "
        "The 25 cross-fitting cells exclude both the evaluation spatial fold and temporal fold "
        "from training. A prior-monsoon/next-calendar-year specification is reported only as a "
        "temporal-alignment diagnostic because annual NPP and the May-October production season "
        "do not share identical aggregation windows.\n\n"
        f"Ecological gate: **{summary['ecological_gate']}**\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(coefficients.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
