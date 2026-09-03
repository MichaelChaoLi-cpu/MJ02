#!/usr/bin/env python3
"""Test dynamic MODIS cropland-weighted NDVI against household production and food."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analyses"))
import test_absolute_ndvi_production_food_pathways as pathway  # noqa: E402


CROPLAND = ROOT / "data/processed/cses_village_dynamic_cropland_ndvi_preprocessed.parquet"
OUTPUT = ROOT / "data/exp/analysis/climate-welfare/dynamic-cropland-ndvi-production-food-pathways"

MEASURES = {
    "Strict cropland (IGBP 12)": {
        "source": "Trailing Twelve-Month Strict-Cropland Weighted Absolute NDVI",
        "valid": "Strict-Cropland Valid NDVI Composites",
        "term": "Strict-cropland weighted absolute NDVI (0.01 units)",
        "role": "primary",
    },
    "Inclusive agriculture (IGBP 12+14)": {
        "source": "Trailing Twelve-Month Inclusive-Agriculture Weighted Absolute NDVI",
        "valid": "Inclusive-Agriculture Valid NDVI Composites",
        "term": "Inclusive-agriculture weighted absolute NDVI (0.01 units)",
        "role": "sensitivity",
    },
}


def holm_all_primary(results: pd.DataFrame) -> pd.Series:
    primary = results["Specification"].eq("Commune FE")
    adjusted = pd.Series(np.nan, index=results.index, dtype=float)
    adjusted.loc[primary] = pathway.holm_adjust(results.loc[primary, "Probability Value"])
    return adjusted


def main() -> None:
    if not CROPLAND.exists():
        raise FileNotFoundError(CROPLAND)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pathway.prepare_frame()
    overlay = pd.read_parquet(CROPLAND)
    overlay["Village Code"] = overlay["Village Code"].astype(str)
    overlay["interview_date"] = pd.to_datetime(overlay["interview_date"])
    frame["Village Code"] = frame["Village Code"].astype(str)
    frame["interview_date"] = pd.to_datetime(frame["interview_date"])
    frame = frame.merge(
        overlay,
        on=["Village Code", "interview_date"],
        how="left",
        validate="many_to_one",
    )

    stage_a_parts: list[pd.DataFrame] = []
    stage_b_parts: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    gates: dict[str, object] = {}
    for definition, spec in MEASURES.items():
        eligible = frame.loc[frame[spec["valid"]].ge(18)].copy()
        eligible[spec["term"]] = pd.to_numeric(eligible[spec["source"]], errors="coerce") / 0.01
        pathway.NDVI = spec["term"]
        a = pathway.stage_a(eligible)
        b = pathway.stage_b(eligible, a)
        a.insert(0, "NDVI Definition", definition)
        a.insert(1, "Inferential Role", spec["role"])
        b.insert(0, "NDVI Definition", definition)
        b.insert(1, "Inferential Role", spec["role"])
        stage_a_parts.append(a)
        stage_b_parts.append(b)

        primary = a.loc[a["Specification"].eq("Commune FE")].set_index("Outcome")
        repeated = a.loc[
            a["Specification"].eq("Repeated-village sample, village FE")
        ].set_index("Outcome")
        production_outcomes = [
            pathway.OUTCOME_LOG_OWN,
            pathway.OUTCOME_LOG_CROP,
            pathway.OUTCOME_LOG_CROP_HA,
        ]
        positive_primary = [
            outcome
            for outcome in production_outcomes
            if float(primary.loc[outcome, "Coefficient"]) > 0
            and float(
                primary.loc[
                    outcome,
                    "Holm-Adjusted Probability Value Across Four Primary Outcomes",
                ]
            ) < 0.05
            and float(repeated.loc[outcome, "Coefficient"]) > 0
        ]
        gates[definition] = {
            "positive_production_outcomes_passing_commune_holm_and_repeated_village_direction": (
                positive_primary
            ),
            "any_positive_production_signal_supported": bool(positive_primary),
        }
        coverage_rows.append(
            {
                "NDVI Definition": definition,
                "Inferential Role": spec["role"],
                "Households": len(eligible),
                "Villages": eligible["Village Code"].nunique(),
                "Village-Interview Windows": eligible[
                    ["Village Code", "interview_date"]
                ].drop_duplicates().shape[0],
                "Mean NDVI": eligible[spec["source"]].mean(),
                "SD NDVI": eligible[spec["source"]].std(),
                "P10 NDVI": eligible[spec["source"]].quantile(0.10),
                "Median NDVI": eligible[spec["source"]].median(),
                "P90 NDVI": eligible[spec["source"]].quantile(0.90),
            }
        )

    stage_a = pd.concat(stage_a_parts, ignore_index=True)
    stage_b = pd.concat(stage_b_parts, ignore_index=True)
    stage_a["Holm-Adjusted Probability Value Across Eight Primary Mask-Outcome Tests"] = (
        holm_all_primary(stage_a)
    )
    coverage = pd.DataFrame(coverage_rows)
    summary = {
        "design": (
            "Five-kilometre village-buffer MOD13Q1 NDVI weighted by same-year MCD12Q1 "
            "cropland share over the twelve complete months before interview"
        ),
        "scale": "coefficients are per 0.01 absolute NDVI; no NDVI standardization",
        "primary_mask": "IGBP class 12 Croplands",
        "sensitivity_mask": "IGBP classes 12 Croplands and 14 Cropland/Natural Vegetation Mosaics",
        "minimum_valid_composites": 18,
        "gates": gates,
        "screening_limit": (
            "This screen weights 1 km all-pixel MOD13Q1 means by cropland share. Exact 250 m "
            "cropland-pixel extraction should be run only if a coherent positive signal appears."
        ),
        "claim_limit": "fixed-effect associations, not causal mediation",
    }
    stage_a.to_csv(OUTPUT / "stage_a_cropland_ndvi_to_production.csv", index=False)
    stage_b.to_csv(OUTPUT / "stage_b_production_to_food.csv", index=False)
    coverage.to_csv(OUTPUT / "analysis_coverage_and_ndvi_distribution.csv", index=False)
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Dynamic MODIS cropland-weighted NDVI pathway screen\n\n"
        "The primary exposure uses MCD12Q1 IGBP class 12; classes 12+14 are a sensitivity "
        "definition. NDVI remains in absolute index units and is reported per 0.01. Results "
        "are exploratory fixed-effect associations and not causal mediation estimates.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nCoverage")
    print(coverage.to_string(index=False))
    print("\nPrimary commune-FE results")
    columns = [
        "NDVI Definition", "Outcome", "Coefficient", "Clustered Standard Error",
        "95 Percent CI Lower", "95 Percent CI Upper", "Probability Value",
        "Holm-Adjusted Probability Value Across Four Primary Outcomes",
        "Holm-Adjusted Probability Value Across Eight Primary Mask-Outcome Tests",
        "Observations",
    ]
    print(stage_a.loc[stage_a["Specification"].eq("Commune FE"), columns].to_string(index=False))


if __name__ == "__main__":
    main()
