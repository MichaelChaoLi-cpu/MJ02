#!/usr/bin/env python3
"""Validate the CSES absolute-heat food-consumption signal in natural units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from sklearn.linear_model import LogisticRegression


CSES = Path("data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet")
ECOLOGY = Path("data/processed/cambodia_public_village_monsoon_ecology_panel_preprocessed.parquet")
OUTPUT = Path("data/exp/analysis/climate-welfare/cses-absolute-heat-food-validation")

ID = "National Village Point ID"
YEAR = "Last Complete May-October Season Year"
FOOD = "Real 2021 Food Consumption Value per Household Member Riels"
RAIN = "Rainfall Anomaly Z"
ONSET = "Onset Anomaly Z"
DRY = "Dry Spell Anomaly Z"
HEAT = "Absolute Heat Days per 10"
EXPOSURES = [RAIN, ONSET, DRY, HEAT]
COMPOSITION = [
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
]
LINKAGE_WEIGHT = "Linkage-Response Adjusted Survey Weight"
LINKAGE_FEATURES_CATEGORICAL = ["Survey Year", "Province Code", "Urban Rural"]
LINKAGE_FEATURES_NUMERIC = ["Household Size", "Agricultural Participation"] + COMPOSITION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["Log Real Food Consumption per Member"] = np.log(pd.to_numeric(frame[FOOD], errors="coerce"))
    frame["Wave by Interview Month"] = (
        frame["Survey Year"].astype(str) + "_" + frame["Interview Month"].astype("Int64").astype(str)
    )
    frame["Spatial Block"] = (
        np.floor((frame["Point Longitude"] - 102.0) / 0.75).astype("Int64").astype(str)
        + "_"
        + np.floor((frame["Point Latitude"] - 10.0) / 0.75).astype("Int64").astype(str)
    )
    return frame


def add_linkage_response_weights(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Construct outcome-blind stabilized inverse linkage-response weights."""
    result = frame.copy()
    categorical = pd.get_dummies(
        result[LINKAGE_FEATURES_CATEGORICAL].astype("string").fillna("missing"),
        drop_first=False,
        dtype=float,
    )
    numeric = result[LINKAGE_FEATURES_NUMERIC].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.fillna(numeric.median())
    scale = numeric.std(ddof=0).replace(0, 1.0)
    numeric = (numeric - numeric.mean()) / scale
    matrix = pd.concat([categorical, numeric], axis=1).to_numpy(float)
    linked = result["Climate Ecology Link Available"].eq(1).to_numpy()
    survey_weight = result["Household Survey Weight"].astype(float).to_numpy()
    model = LogisticRegression(C=1.0, max_iter=5_000, solver="lbfgs")
    model.fit(matrix, linked.astype(int), sample_weight=survey_weight)
    probability = np.clip(model.predict_proba(matrix)[:, 1], 0.02, 0.98)
    weighted_link_rate = float(np.average(linked, weights=survey_weight))
    multiplier = weighted_link_rate / probability
    linked_multiplier = multiplier[linked]
    lower, upper = np.quantile(linked_multiplier, [0.01, 0.99])
    multiplier = np.clip(multiplier, lower, upper)
    result["Linkage Response Probability"] = probability
    result["Linkage Response Weight Multiplier"] = multiplier
    result[LINKAGE_WEIGHT] = result["Household Survey Weight"].astype(float) * multiplier
    diagnostics = {
        "households": int(len(result)),
        "linked_households": int(linked.sum()),
        "survey_weighted_link_rate": weighted_link_rate,
        "linked_probability_p01": float(np.quantile(probability[linked], 0.01)),
        "linked_probability_median": float(np.median(probability[linked])),
        "linked_probability_p99": float(np.quantile(probability[linked], 0.99)),
        "linked_multiplier_clip": [float(lower), float(upper)],
        "target_limit": (
            "Sensitivity reweights observed linked households under an outcome-blind response model; "
            "it cannot recover provinces or village types with zero public-point support."
        ),
    }
    return result, diagnostics


def ecology_exposures(
    root: Path, radius: int, candidate: str = "B", threshold_c: int = 35
) -> pd.DataFrame:
    source = {
        RAIN: "Village Buffer Mean May October Precipitation Total mm Anomaly Z",
        ONSET: f"Village Buffer Mean Wet-Season Onset DOY Candidate {candidate} Anomaly Z",
        DRY: f"Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate {candidate} Anomaly Z",
        HEAT: (
            f"Village Buffer Mean Post-Onset Absolute Heat Day Count {threshold_c} C "
            f"Candidate {candidate}"
        ),
    }
    columns = [ID, "Year", "Buffer Radius km"] + list(source.values())
    frame = pd.read_parquet(root / ECOLOGY, columns=columns)
    frame = frame.loc[frame["Buffer Radius km"].eq(radius)].drop(columns="Buffer Radius km")
    frame = frame.rename(columns={value: key for key, value in source.items()})
    frame[HEAT] = frame[HEAT] / 10.0
    return frame


def fit(
    frame: pd.DataFrame,
    specification: str,
    controls: list[str] | None = None,
    village_effects: bool = False,
    weight_column: str = "Household Survey Weight",
) -> tuple[object, list[dict]]:
    controls = controls or []
    columns = ["Log Real Food Consumption per Member", weight_column, "Spatial Block"]
    columns += EXPOSURES + controls + ["Village Code", "District Code", "Wave by Interview Month"]
    sample = frame.dropna(subset=columns).copy()
    absorb = pd.DataFrame(index=sample.index)
    if village_effects:
        absorb["Village"] = sample["Village Code"].astype("category")
    else:
        absorb["District"] = sample["District Code"].astype("category")
    absorb["Wave by Interview Month"] = sample["Wave by Interview Month"].astype("category")
    regressors = EXPOSURES + controls
    result = AbsorbingLS(
        sample["Log Real Food Consumption per Member"].astype(float),
        sample[regressors].astype(float),
        absorb=absorb,
        weights=sample[weight_column].astype(float),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample["Spatial Block"]).codes,
        debiased=True,
    )
    interval = result.conf_int(level=0.95)
    rows = [
        {
            "Specification": specification,
            "Exposure": exposure,
            "Coefficient Log Points": float(result.params[exposure]),
            "Percent Change for Exposure Unit": float((np.exp(result.params[exposure]) - 1) * 100),
            "Exposure Unit": "10 days" if exposure == HEAT else "1 SD",
            "Clustered Standard Error": float(result.std_errors[exposure]),
            "95 Percent CI Lower": float(interval.loc[exposure, "lower"]),
            "95 Percent CI Upper": float(interval.loc[exposure, "upper"]),
            "Probability Value": float(result.pvalues[exposure]),
            "Observations": int(result.nobs),
            "Villages": int(sample["Village Code"].nunique()),
            "Spatial Blocks": int(sample["Spatial Block"].nunique()),
            "Weighting": weight_column,
        }
        for exposure in EXPOSURES
    ]
    return result, rows


def fit_conditional_future_placebo(frame: pd.DataFrame) -> tuple[object, list[dict]]:
    future_exposures = [f"Future {exposure}" for exposure in EXPOSURES]
    regressors = EXPOSURES + future_exposures
    columns = [
        "Log Real Food Consumption per Member", "Household Survey Weight", "Spatial Block",
        "District Code", "Wave by Interview Month",
    ] + regressors
    sample = frame.dropna(subset=columns).copy()
    absorb = pd.DataFrame(
        {
            "District": sample["District Code"].astype("category"),
            "Wave by Interview Month": sample["Wave by Interview Month"].astype("category"),
        },
        index=sample.index,
    )
    result = AbsorbingLS(
        sample["Log Real Food Consumption per Member"].astype(float),
        sample[regressors].astype(float),
        absorb=absorb,
        weights=sample["Household Survey Weight"].astype(float),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample["Spatial Block"]).codes,
        debiased=True,
    )
    interval = result.conf_int(level=0.95)
    rows = []
    for exposure in regressors:
        rows.append(
            {
                "Specification": "Current plus future season, clean interview months",
                "Exposure": exposure,
                "Coefficient Log Points": float(result.params[exposure]),
                "Percent Change for Exposure Unit": float((np.exp(result.params[exposure]) - 1) * 100),
                "Exposure Unit": "10 days" if exposure.endswith(HEAT) else "1 SD",
                "Clustered Standard Error": float(result.std_errors[exposure]),
                "95 Percent CI Lower": float(interval.loc[exposure, "lower"]),
                "95 Percent CI Upper": float(interval.loc[exposure, "upper"]),
                "Probability Value": float(result.pvalues[exposure]),
                "Observations": int(result.nobs),
                "Villages": int(sample["Village Code"].nunique()),
                "Spatial Blocks": int(sample["Spatial Block"].nunique()),
            }
        )
    return result, rows


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    for relative in (CSES, ECOLOGY):
        if not (root / relative).exists():
            raise FileNotFoundError(root / relative)
    output = root / OUTPUT
    output.mkdir(parents=True, exist_ok=True)

    cses_columns = [
        "Survey Year", "Interview Month", "Household ID", "Village Code", "District Code",
        "Province Code", "Household Size", "Agricultural Participation",
        "Climate Ecology Link Available", "Household Survey Weight", "Point Longitude",
        "Point Latitude", ID, YEAR, FOOD, "Urban Rural",
    ] + COMPOSITION
    base = pd.read_parquet(root / CSES, columns=cses_columns)
    base = prepare_frame(base)
    base, linkage_diagnostics = add_linkage_response_weights(base)
    (output / "linkage_response_weight_diagnostics.json").write_text(
        json.dumps(linkage_diagnostics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    base = base.loc[base["Climate Ecology Link Available"].eq(1)].copy()

    specifications: list[tuple[str, pd.DataFrame, list[str], bool]] = []
    for radius in (2, 5, 10):
        exposure = ecology_exposures(root, radius, "B", 35)
        sample = base.merge(exposure, left_on=[ID, YEAR], right_on=[ID, "Year"], how="left")
        specifications.append((f"Candidate B, 35 C, {radius} km", sample, [], False))
    primary = specifications[1][1]
    for threshold_c in (33, 37):
        threshold_exposure = ecology_exposures(root, 5, "B", threshold_c)
        threshold_sample = base.merge(
            threshold_exposure, left_on=[ID, YEAR], right_on=[ID, "Year"], how="left"
        )
        specifications.append(
            (f"Candidate B, {threshold_c} C, 5 km", threshold_sample, [], False)
        )
    specifications.append(
        ("Candidate B, 35 C, 5 km, composition adjusted", primary, COMPOSITION, False)
    )

    repeated_ids = (
        primary.groupby("Village Code", observed=True)["Survey Year"].nunique().loc[lambda x: x > 1].index
    )
    repeated = primary.loc[primary["Village Code"].isin(repeated_ids)].copy()
    specifications.append(
        ("Candidate B, 35 C, 5 km, repeated-village fixed effects", repeated, [], True)
    )

    rows: list[dict] = []
    for name, sample, controls, village_effects in specifications:
        _, coefficient_rows = fit(sample, name, controls=controls, village_effects=village_effects)
        rows.extend(coefficient_rows)
    _, coefficient_rows = fit(
        primary,
        "Candidate B, 35 C, 5 km, linkage-response weighted",
        weight_column=LINKAGE_WEIGHT,
    )
    rows.extend(coefficient_rows)

    # Leave-one-wave-out checks use the frozen 5 km Candidate B specification.
    for wave in sorted(primary["Survey Year"].unique()):
        _, coefficient_rows = fit(
            primary.loc[primary["Survey Year"].ne(wave)],
            f"Candidate B, 35 C, 5 km, excluding CSES {int(wave)}",
        )
        rows.extend(coefficient_rows)

    # Future-season placebo is cleanly future only for Jan-Apr and Nov-Dec interviews.
    future_exposure = ecology_exposures(root, 5, "B").rename(
        columns={column: f"Future {column}" for column in EXPOSURES}
    )
    future_base = primary.loc[primary["Interview Month"].isin([1, 2, 3, 4, 11, 12])].copy()
    future_base["Future Season Year"] = future_base[YEAR] + 1
    future = future_base.merge(
        future_exposure,
        left_on=[ID, "Future Season Year"],
        right_on=[ID, "Year"],
        how="left",
    )
    _, future_rows = fit_conditional_future_placebo(future)
    rows.extend(future_rows)

    coefficients = pd.DataFrame(rows)
    coefficients.to_csv(output / "coefficients.csv", index=False)
    heat = coefficients.loc[coefficients["Exposure"].eq(HEAT)].copy()
    heat.to_csv(output / "heat_specification_summary.csv", index=False)

    promotion_names = [
        "Candidate B, 35 C, 2 km",
        "Candidate B, 35 C, 5 km",
        "Candidate B, 35 C, 10 km",
        "Candidate B, 33 C, 5 km",
        "Candidate B, 35 C, 5 km, composition adjusted",
        "Candidate B, 35 C, 5 km, linkage-response weighted",
    ]
    promotion = heat.loc[heat["Specification"].isin(promotion_names)]
    boundary = heat.loc[
        heat["Specification"].isin(
            [
                "Candidate B, 37 C, 5 km",
                "Candidate B, 35 C, 5 km, repeated-village fixed effects",
            ]
        )
    ]
    leave_out = heat.loc[heat["Specification"].str.contains("excluding CSES")]
    placebo = coefficients.loc[
        coefficients["Exposure"].eq(f"Future {HEAT}")
        & coefficients["Specification"].eq("Current plus future season, clean interview months")
    ].iloc[0]
    primary_row = heat.loc[heat["Specification"].eq("Candidate B, 35 C, 5 km")].iloc[0]
    pass_rule = bool(
        primary_row["95 Percent CI Upper"] < 0
        and promotion["Coefficient Log Points"].lt(0).all()
        and leave_out["Coefficient Log Points"].lt(0).all()
        and placebo["95 Percent CI Lower"] <= 0 <= placebo["95 Percent CI Upper"]
    )
    summary = {
        "signal": "last-complete-season absolute heat days to log real food consumption per member",
        "primary_threshold_c": 35,
        "primary_reporting_unit": "10 additional post-onset heat days",
        "primary_percent_change_per_10_days": float(
            primary_row["Percent Change for Exposure Unit"]
        ),
        "primary_95_percent_log_interval": [
            float(primary_row["95 Percent CI Lower"]), float(primary_row["95 Percent CI Upper"])
        ],
        "promotion_specifications_all_negative": bool(
            promotion["Coefficient Log Points"].lt(0).all()
        ),
        "boundary_checks": boundary[
            ["Specification", "Coefficient Log Points", "95 Percent CI Lower", "95 Percent CI Upper"]
        ].to_dict(orient="records"),
        "leave_one_wave_out_all_negative": bool(leave_out["Coefficient Log Points"].lt(0).all()),
        "future_placebo_interval_contains_zero": bool(
            placebo["95 Percent CI Lower"] <= 0 <= placebo["95 Percent CI Upper"]
        ),
        "validation_gate": "pass" if pass_rule else "not passed",
        "gate_rule": "negative primary interval, negative direction at 2/5/10 km, at 33 C, with composition adjustment, and in every leave-one-wave-out check, plus a future-placebo interval containing zero; sparse 37 C and repeated-village estimates are reported as interpretation boundaries rather than promotion requirements",
    }
    (output / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Absolute-heat food-consumption validation\n\n"
        "This validation uses the 35 C, 5 km Candidate B estimate as primary and checks 33/37 C "
        "thresholds, 2/10 km buffers, composition adjustment, repeated-village effects, every "
        "leave-one-wave-out sample, and a clean future-season placebo.\n\n"
        f"Validation gate: **{summary['validation_gate']}**\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(heat.to_string(index=False))


if __name__ == "__main__":
    main()
