#!/usr/bin/env python3
"""Test direct associations between prior-year NDVI and household food consumption.

This exploratory diagnostic reuses the survey-anchored rolling 12-month NDVI
frame materialized from processed CSES, climate, and MODIS data.  It compares
district, commune, repeated-village, agricultural-household, and stable-commune
pseudo-panel specifications.  The estimand is associational and is not treated
as a causal mediation effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / (
    "data/exp/analysis/climate-welfare/rolling-multiseason-ndvi-food-test/"
    "household_rolling_12month_sample.parquet"
)
OUTPUT = ROOT / "data/exp/analysis/climate-welfare/ndvi-household-welfare-association"

FOOD = "Real 2021 Food Consumption Value per Household Member Riels"
WEIGHT = "Household Survey Weight"
TIME = "wave_month"
BLOCK = "spatial_block"
NDVI_METRICS = {
    "Annual mean absolute NDVI (0.01 units)": (
        "annual_mean_ndvi_absolute",
        0.01,
        "positive",
    ),
}
CLIMATE_CONTROLS = ["Annual heat days (10 days)", "Annual precipitation (1000 mm)"]
COMPOSITION = [
    "Household Size",
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
    "Agricultural Participation",
]


def prepare_households() -> pd.DataFrame:
    frame = pd.read_parquet(INPUT).copy()
    frame["Log real food consumption per member"] = np.log(
        pd.to_numeric(frame[FOOD], errors="coerce")
    )
    frame["Annual heat days (10 days)"] = (
        pd.to_numeric(frame["annual_heat_days_35"], errors="coerce") / 10.0
    )
    frame["Annual precipitation (1000 mm)"] = (
        pd.to_numeric(frame["annual_precipitation_mm"], errors="coerce") / 1000.0
    )
    for readable, (source, scale, _) in NDVI_METRICS.items():
        frame[readable] = pd.to_numeric(frame[source], errors="coerce") / scale
    frame["Agricultural Participation"] = (
        frame["Agricultural Participation"].astype("boolean").astype(float)
    )
    wave_count = frame.groupby("Village Code", observed=True)["Survey Year"].nunique()
    repeated = wave_count.loc[lambda value: value > 1].index
    frame["Repeated village"] = frame["Village Code"].isin(repeated)
    return frame


def fit_absorbing(
    frame: pd.DataFrame,
    outcome: str,
    exposure: str,
    area: str,
    controls: list[str],
    weight: str,
    cluster: str,
    specification: str,
    level: str,
) -> dict[str, object]:
    required = [outcome, exposure, area, TIME, weight, cluster] + controls
    sample = frame.dropna(subset=required).copy()
    absorb = pd.DataFrame(
        {
            "Area": sample[area].astype("category"),
            "Survey time": sample[TIME].astype("category"),
        },
        index=sample.index,
    )
    regressors = [exposure] + controls
    result = AbsorbingLS(
        sample[outcome].astype(float),
        sample[regressors].astype(float),
        absorb=absorb,
        weights=sample[weight].astype(float),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample[cluster]).codes,
        debiased=True,
    )
    interval = result.conf_int(level=0.95).loc[exposure]
    coefficient = float(result.params[exposure])
    return {
        "Evidence Level": level,
        "Specification": specification,
        "NDVI Metric": exposure,
        "Coefficient Log Points per 0.01 NDVI Unit": coefficient,
        "Percent Difference per 0.01 NDVI Unit": float(100.0 * np.expm1(coefficient)),
        "Clustered Standard Error": float(result.std_errors[exposure]),
        "95 Percent CI Lower": float(interval["lower"]),
        "95 Percent CI Upper": float(interval["upper"]),
        "Probability Value": float(result.pvalues[exposure]),
        "Observations": int(result.nobs),
        "Area Units": int(sample[area].nunique()),
        "Spatial Blocks": int(sample[cluster].nunique()),
        "Fixed Effects": f"{area} + survey-wave-by-interview-calendar-year-month",
        "Controls": ", ".join(controls),
        "Weighting": weight,
    }


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce")
    weights = pd.to_numeric(group[WEIGHT], errors="coerce")
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def modal_weighted_block(group: pd.DataFrame) -> str:
    totals = group.groupby(BLOCK, observed=True)[WEIGHT].sum()
    return str(totals.idxmax())


def build_commune_pseudopanel(frame: pd.DataFrame) -> pd.DataFrame:
    mean_columns = [
        "Log real food consumption per member",
        *NDVI_METRICS,
        *CLIMATE_CONTROLS,
        *COMPOSITION,
    ]
    rows: list[dict[str, object]] = []
    for (commune, time), group in frame.groupby(["Commune Code", TIME], observed=True):
        row: dict[str, object] = {
            "Commune Code": commune,
            TIME: time,
            "Survey Year": int(group["Survey Year"].iloc[0]),
            "Household Count": int(len(group)),
            "Survey Weight Sum": float(group[WEIGHT].sum()),
            BLOCK: modal_weighted_block(group),
        }
        for column in mean_columns:
            row[column] = weighted_mean(group, column)
        rows.append(row)
    panel = pd.DataFrame(rows)
    repeat_count = panel.groupby("Commune Code", observed=True)["Survey Year"].nunique()
    repeated = repeat_count.loc[lambda value: value > 1].index
    panel["Repeated commune"] = panel["Commune Code"].isin(repeated)
    return panel


def estimate_all(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = build_commune_pseudopanel(frame)
    rows: list[dict[str, object]] = []
    primary_controls = CLIMATE_CONTROLS
    composition_controls = CLIMATE_CONTROLS + COMPOSITION
    repeated = frame.loc[frame["Repeated village"]].copy()
    agricultural = frame.loc[frame["Agricultural Participation"].eq(1)].copy()
    pseudo = panel.loc[panel["Repeated commune"] & panel["Household Count"].ge(5)].copy()

    for exposure in NDVI_METRICS:
        rows.append(
            fit_absorbing(
                frame, "Log real food consumption per member", exposure,
                "Commune Code", [], WEIGHT, BLOCK,
                "All linked households, commune FE, NDVI only", "Household",
            )
        )
        rows.append(
            fit_absorbing(
                frame, "Log real food consumption per member", exposure,
                "District Code", primary_controls, WEIGHT, BLOCK,
                "All linked households, district FE", "Household",
            )
        )
        rows.append(
            fit_absorbing(
                frame, "Log real food consumption per member", exposure,
                "Commune Code", primary_controls, WEIGHT, BLOCK,
                "All linked households, commune FE", "Household",
            )
        )
        rows.append(
            fit_absorbing(
                frame, "Log real food consumption per member", exposure,
                "Commune Code", composition_controls, WEIGHT, BLOCK,
                "All linked households, commune FE, composition adjusted", "Household",
            )
        )
        rows.append(
            fit_absorbing(
                repeated, "Log real food consumption per member", exposure,
                "District Code", primary_controls, WEIGHT, BLOCK,
                "Repeated-village sample, district FE", "Household",
            )
        )
        rows.append(
            fit_absorbing(
                repeated, "Log real food consumption per member", exposure,
                "Village Code", primary_controls, WEIGHT, BLOCK,
                "Repeated-village sample, village FE", "Household",
            )
        )
        rows.append(
            fit_absorbing(
                agricultural, "Log real food consumption per member", exposure,
                "Commune Code", primary_controls, WEIGHT, BLOCK,
                "Agricultural households, commune FE", "Household",
            )
        )
        rows.append(
            fit_absorbing(
                pseudo, "Log real food consumption per member", exposure,
                "Commune Code", [], "Survey Weight Sum", BLOCK,
                "Repeated-commune pseudo-panel, NDVI only", "Commune survey-time pseudo-panel",
            )
        )
        rows.append(
            fit_absorbing(
                pseudo, "Log real food consumption per member", exposure,
                "Commune Code", primary_controls, "Survey Weight Sum", BLOCK,
                "Repeated-commune pseudo-panel, minimum five households", "Commune survey-time pseudo-panel",
            )
        )
        rows.append(
            fit_absorbing(
                pseudo, "Log real food consumption per member", exposure,
                "Commune Code", composition_controls, "Survey Weight Sum", BLOCK,
                "Repeated-commune pseudo-panel, composition adjusted", "Commune survey-time pseudo-panel",
            )
        )
    return pd.DataFrame(rows), panel


def support_summary(results: pd.DataFrame, frame: pd.DataFrame, panel: pd.DataFrame) -> dict[str, object]:
    primary_metric = "Annual mean absolute NDVI (0.01 units)"
    selected = results.loc[results["NDVI Metric"].eq(primary_metric)].set_index("Specification")
    commune = selected.loc["All linked households, commune FE"]
    pseudo = selected.loc["Repeated-commune pseudo-panel, minimum five households"]
    repeated = selected.loc["Repeated-village sample, village FE"]
    agricultural = selected.loc["Agricultural households, commune FE"]
    gate = {
        "commune_fe_positive_interval": bool(
            commune["Coefficient Log Points per 0.01 NDVI Unit"] > 0
            and commune["95 Percent CI Lower"] > 0
        ),
        "pseudo_panel_positive_interval": bool(
            pseudo["Coefficient Log Points per 0.01 NDVI Unit"] > 0
            and pseudo["95 Percent CI Lower"] > 0
        ),
        "repeated_village_positive_direction": bool(
            repeated["Coefficient Log Points per 0.01 NDVI Unit"] > 0
        ),
        "agricultural_household_positive_direction": bool(
            agricultural["Coefficient Log Points per 0.01 NDVI Unit"] > 0
        ),
    }
    gate["promotion_gate_passed"] = bool(all(gate.values()))
    return {
        "design": "Prior 12 complete months of village-buffer NDVI to interview food consumption",
        "reporting_unit": "0.01 NDVI index units and log food-consumption points",
        "households": int(len(frame)),
        "villages": int(frame["Village Code"].nunique()),
        "communes": int(frame["Commune Code"].nunique()),
        "pseudo_panel_cells_all": int(len(panel)),
        "primary_pseudo_panel_cells": int(
            (panel["Repeated commune"] & panel["Household Count"].ge(5)).sum()
        ),
        "gate": gate,
        "interpretation": (
            "Direct NDVI-household welfare association passed the frozen exploratory gate."
            if gate["promotion_gate_passed"]
            else "Direct NDVI-household welfare association did not pass the frozen exploratory gate."
        ),
        "claim_limit": "Associational repeated-cross-section and pseudo-panel evidence; not causal mediation.",
    }


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = prepare_households()
    results, panel = estimate_all(frame)
    summary = support_summary(results, frame, panel)
    results.to_csv(OUTPUT / "coefficients.csv", index=False)
    panel.to_parquet(OUTPUT / "commune_survey_time_pseudopanel.parquet", index=False)
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# NDVI and household welfare association diagnostic\n\n"
        "This exploratory diagnostic relates village-buffer NDVI over the twelve complete months "
        "before interview to real food consumption per household member. It compares district, "
        "commune, repeated-village, agricultural-household, and repeated-commune pseudo-panel "
        "specifications. Annual heat and precipitation are included as aligned climate controls. "
        "The estimates are associations and do not identify mediation or a causal NDVI effect.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
