#!/usr/bin/env python3
"""Test absolute-NDVI, household production, and food-consumption pathways.

Stage A relates mean absolute NDVI over the twelve complete months before
interview to own-produced food, crop production value, and crop value per
cultivated hectare. Stage B relates those production measures to total and
market-acquired food consumption on the same samples. The exercise is a set of
fixed-effect associations, not causal mediation.
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
OUTPUT = ROOT / "data/exp/analysis/climate-welfare/absolute-ndvi-production-food-pathways"

WEIGHT = "Household Survey Weight"
TIME = "wave_month"
BLOCK = "spatial_block"
NDVI = "Annual mean absolute NDVI (0.01 units)"
HEAT = "Annual heat days (10 days)"
RAIN = "Annual precipitation (1000 mm)"
CLIMATE = [HEAT, RAIN]

FOOD = "Real 2021 Food Consumption Value per Household Member Riels"
OWN = "Real 2021 Own Produced Food Value per Household Member Riels"
CROP = "Real 2021 Crop Production Value Riels"
CROP_HA = "Real 2021 Crop Production Value per Cultivated ha Riels"

OUTCOME_ANY_OWN = "Any positive own-produced food"
OUTCOME_LOG_OWN = "Log own-produced food per member among positive households"
OUTCOME_LOG_CROP = "Log1p crop production value"
OUTCOME_LOG_CROP_HA = "Log1p crop production value per cultivated ha"
PRIMARY_OUTCOMES = [
    OUTCOME_ANY_OWN,
    OUTCOME_LOG_OWN,
    OUTCOME_LOG_CROP,
    OUTCOME_LOG_CROP_HA,
]

COMPOSITION = [
    "Household Size",
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
    "Agricultural Participation",
]


def holm_adjust(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    valid = values.dropna().sort_values()
    m = len(valid)
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    running = 0.0
    for rank, (index, value) in enumerate(valid.items(), start=1):
        candidate = min(1.0, float(value) * (m - rank + 1))
        running = max(running, candidate)
        adjusted.loc[index] = running
    return adjusted


def prepare_frame() -> pd.DataFrame:
    frame = pd.read_parquet(INPUT).copy()
    frame[NDVI] = pd.to_numeric(frame["annual_mean_ndvi_absolute"], errors="coerce") / 0.01
    frame[HEAT] = pd.to_numeric(frame["annual_heat_days_35"], errors="coerce") / 10.0
    frame[RAIN] = pd.to_numeric(frame["annual_precipitation_mm"], errors="coerce") / 1000.0
    frame["Log total food per member"] = np.log(pd.to_numeric(frame[FOOD], errors="coerce"))

    own = pd.to_numeric(frame[OWN], errors="coerce")
    total = pd.to_numeric(frame[FOOD], errors="coerce")
    frame[OUTCOME_ANY_OWN] = own.gt(0).astype(float)
    frame[OUTCOME_LOG_OWN] = np.nan
    positive_own = own.gt(0)
    frame.loc[positive_own, OUTCOME_LOG_OWN] = np.log(own.loc[positive_own])
    market = total - own
    frame["Log market-acquired food per member"] = np.nan
    positive_market = market.gt(0)
    frame.loc[positive_market, "Log market-acquired food per member"] = np.log(
        market.loc[positive_market]
    )
    frame["Market-food accounting inconsistency"] = market.le(0)

    crop = pd.to_numeric(frame[CROP], errors="coerce")
    crop_ha = pd.to_numeric(frame[CROP_HA], errors="coerce")
    frame[OUTCOME_LOG_CROP] = np.where(crop.notna() & crop.ge(0), np.log1p(crop), np.nan)
    frame[OUTCOME_LOG_CROP_HA] = np.where(
        crop_ha.notna() & crop_ha.ge(0), np.log1p(crop_ha), np.nan
    )
    frame["Agricultural Participation"] = (
        frame["Agricultural Participation"].astype("boolean").astype(float)
    )
    wave_count = frame.groupby("Village Code", observed=True)["Survey Year"].nunique()
    repeated = wave_count.loc[lambda value: value > 1].index
    frame["Repeated village"] = frame["Village Code"].isin(repeated)
    return frame


def fit_model(
    frame: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    area: str,
) -> tuple[object, pd.DataFrame]:
    required = [outcome, WEIGHT, BLOCK, TIME, area] + regressors
    sample = frame.dropna(subset=required).copy()
    absorb = pd.DataFrame(
        {
            "Area": sample[area].astype("category"),
            "Survey time": sample[TIME].astype("category"),
        },
        index=sample.index,
    )
    result = AbsorbingLS(
        sample[outcome].astype(float),
        sample[regressors].astype(float),
        absorb=absorb,
        weights=sample[WEIGHT].astype(float),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample[BLOCK]).codes,
        debiased=True,
    )
    return result, sample


def coefficient_row(
    result: object,
    sample: pd.DataFrame,
    term: str,
    stage: str,
    outcome: str,
    specification: str,
    area: str,
) -> dict[str, object]:
    interval = result.conf_int(level=0.95).loc[term]
    coefficient = float(result.params[term])
    return {
        "Stage": stage,
        "Outcome": outcome,
        "Specification": specification,
        "Term": term,
        "Coefficient": coefficient,
        "Clustered Standard Error": float(result.std_errors[term]),
        "95 Percent CI Lower": float(interval["lower"]),
        "95 Percent CI Upper": float(interval["upper"]),
        "Probability Value": float(result.pvalues[term]),
        "Observations": int(result.nobs),
        "Area Units": int(sample[area].nunique()),
        "Spatial Blocks": int(sample[BLOCK].nunique()),
        "Fixed Effects": f"{area} + survey-wave-by-interview-calendar-year-month",
        "Survey Weighted": True,
    }


def stage_a(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    repeated = frame.loc[frame["Repeated village"]].copy()
    specifications = [
        ("District FE", frame, "District Code", CLIMATE),
        ("Commune FE", frame, "Commune Code", CLIMATE),
        ("Commune FE, composition adjusted", frame, "Commune Code", CLIMATE + COMPOSITION),
        ("Repeated-village sample, village FE", repeated, "Village Code", CLIMATE),
    ]
    for outcome in PRIMARY_OUTCOMES:
        for specification, sample_frame, area, controls in specifications:
            result, sample = fit_model(sample_frame, outcome, [NDVI] + controls, area)
            rows.append(
                coefficient_row(
                    result, sample, NDVI, "A: prior NDVI to household production",
                    outcome, specification, area,
                )
            )
    results = pd.DataFrame(rows)
    primary = results["Specification"].eq("Commune FE")
    results["Holm-Adjusted Probability Value Across Four Primary Outcomes"] = np.nan
    results.loc[primary, "Holm-Adjusted Probability Value Across Four Primary Outcomes"] = (
        holm_adjust(results.loc[primary, "Probability Value"])
    )
    return results


def stage_b(frame: pd.DataFrame, stage_a_results: pd.DataFrame) -> pd.DataFrame:
    pathway_specs = [
        ("Own-produced food", OUTCOME_LOG_OWN),
        ("Crop production value", OUTCOME_LOG_CROP),
        ("Crop production value per hectare", OUTCOME_LOG_CROP_HA),
    ]
    food_outcomes = [
        ("Total food consumption", "Log total food per member", True),
        ("Market-acquired food consumption", "Log market-acquired food per member", False),
    ]
    rows: list[dict[str, object]] = []
    for pathway, production in pathway_specs:
        for food_label, food_outcome, total_food in food_outcomes:
            required = [food_outcome, production, NDVI, *CLIMATE, WEIGHT, BLOCK, TIME, "Commune Code"]
            sample = frame.dropna(subset=required).copy()
            baseline, baseline_sample = fit_model(
                sample, food_outcome, [NDVI] + CLIMATE, "Commune Code"
            )
            augmented, augmented_sample = fit_model(
                sample, food_outcome, [NDVI, production] + CLIMATE, "Commune Code"
            )
            production_row = coefficient_row(
                augmented, augmented_sample, production,
                "B: household production to food consumption", food_outcome,
                f"{pathway} to {food_label}", "Commune Code",
            )
            production_row.update(
                {
                    "Pathway": pathway,
                    "Food Outcome": food_label,
                    "Part-Whole Accounting Relationship": bool(
                        pathway == "Own-produced food" and total_food
                    ),
                    "NDVI Coefficient Before Production Control": float(baseline.params[NDVI]),
                    "NDVI Coefficient After Production Control": float(augmented.params[NDVI]),
                    "Absolute NDVI Coefficient Attenuation": float(
                        abs(baseline.params[NDVI]) - abs(augmented.params[NDVI])
                    ),
                }
            )
            rows.append(production_row)
    results = pd.DataFrame(rows)
    nonmechanical = ~results["Part-Whole Accounting Relationship"]
    results["Holm-Adjusted Probability Value Across Nonmechanical Food Links"] = np.nan
    results.loc[nonmechanical, "Holm-Adjusted Probability Value Across Nonmechanical Food Links"] = (
        holm_adjust(results.loc[nonmechanical, "Probability Value"])
    )

    primary_a = stage_a_results.loc[
        stage_a_results["Specification"].eq("Commune FE")
    ].set_index("Outcome")
    a_map = {
        "Own-produced food": OUTCOME_LOG_OWN,
        "Crop production value": OUTCOME_LOG_CROP,
        "Crop production value per hectare": OUTCOME_LOG_CROP_HA,
    }
    results["Stage-A NDVI Coefficient"] = results["Pathway"].map(
        {key: float(primary_a.loc[value, "Coefficient"]) for key, value in a_map.items()}
    )
    results["Exploratory Coefficient Product"] = (
        results["Stage-A NDVI Coefficient"] * results["Coefficient"]
    )
    return results


def build_summary(
    frame: pd.DataFrame,
    a: pd.DataFrame,
    b: pd.DataFrame,
) -> dict[str, object]:
    primary_a = a.loc[a["Specification"].eq("Commune FE")].set_index("Outcome")
    repeated_a = a.loc[
        a["Specification"].eq("Repeated-village sample, village FE")
    ].set_index("Outcome")
    a_map = {
        "Own-produced food": OUTCOME_LOG_OWN,
        "Crop production value": OUTCOME_LOG_CROP,
        "Crop production value per hectare": OUTCOME_LOG_CROP_HA,
    }
    paths: dict[str, object] = {}
    for pathway, outcome in a_map.items():
        first = primary_a.loc[outcome]
        links = b.loc[
            b["Pathway"].eq(pathway)
            & ~b["Part-Whole Accounting Relationship"]
        ]
        positive_link = bool(
            (
                links["Coefficient"].gt(0)
                & links["Holm-Adjusted Probability Value Across Nonmechanical Food Links"].lt(0.05)
            ).any()
        )
        first_supported = bool(
            first["Coefficient"] > 0
            and first["Holm-Adjusted Probability Value Across Four Primary Outcomes"] < 0.05
            and repeated_a.loc[outcome, "Coefficient"] > 0
        )
        paths[pathway] = {
            "ndvi_to_production_supported": first_supported,
            "production_to_nonmechanical_food_link_supported": positive_link,
            "coherent_path_supported": bool(first_supported and positive_link),
        }
    return {
        "design": "Absolute NDVI in twelve complete pre-interview months to production and food outcomes",
        "ndvi_unit": "0.01 index units",
        "households": int(len(frame)),
        "villages": int(frame["Village Code"].nunique()),
        "own_food_positive_households": int(pd.to_numeric(frame[OWN], errors="coerce").gt(0).sum()),
        "crop_value_observed_households": int(pd.to_numeric(frame[CROP], errors="coerce").notna().sum()),
        "crop_value_per_ha_observed_households": int(
            pd.to_numeric(frame[CROP_HA], errors="coerce").notna().sum()
        ),
        "market_food_accounting_inconsistencies_excluded": int(
            frame["Market-food accounting inconsistency"].sum()
        ),
        "multiple_testing": {
            "stage_a": "Holm adjustment across four primary commune-FE NDVI coefficients",
            "stage_b": "Holm adjustment across five nonmechanical production-food links",
        },
        "pathway_gates": paths,
        "any_coherent_path_supported": bool(
            any(value["coherent_path_supported"] for value in paths.values())
        ),
        "claim_limit": (
            "Fixed-effect associations only. Own-produced food is a component of total food "
            "consumption, so that part-whole coefficient is not mechanism evidence."
        ),
    }


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = prepare_frame()
    a = stage_a(frame)
    b = stage_b(frame, a)
    summary = build_summary(frame, a, b)
    a.to_csv(OUTPUT / "stage_a_ndvi_to_production.csv", index=False)
    b.to_csv(OUTPUT / "stage_b_production_to_food.csv", index=False)
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Absolute NDVI, household production, and food-consumption pathways\n\n"
        "Stage A estimates NDVI associations with own-produced food, crop production value, "
        "and crop production value per hectare. Stage B tests production-food links on matched "
        "samples. Results use survey weights, area and exact survey-time fixed effects, and "
        "spatial-block clustered uncertainty. These are not causal mediation estimates.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nStage A primary commune-FE results")
    print(a.loc[a["Specification"].eq("Commune FE")].to_string(index=False))
    print("\nStage B results")
    print(b.to_string(index=False))


if __name__ == "__main__":
    main()
