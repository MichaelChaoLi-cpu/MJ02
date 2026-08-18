#!/usr/bin/env python3
"""Materialize the confirmed Direction 3 preprocessing decisions as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def clean_records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    records = frame[columns].where(pd.notna(frame[columns]), None).to_dict(orient="records")
    return [{str(key): value for key, value in record.items()} for record in records]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    exp = root / "data" / "exp" / "data-preprocessing"

    external = pd.read_csv(exp / "direction3_external_variable_dictionary.csv", dtype=str)
    cses = pd.read_csv(exp / "direction3_cses_variable_dictionary.csv", dtype=str)
    core_outcomes_path = exp / "direction3_core_outcome_dictionary.csv"
    core_outcomes = pd.read_csv(core_outcomes_path, dtype=str) if core_outcomes_path.exists() else pd.DataFrame()
    datasets: list[dict[str, object]] = []

    for dataset, group in external.groupby("dataset", sort=True):
        datasets.append(
            {
                "dataset": dataset,
                "output_path": group["output_path"].iloc[0],
                "preprocessing_script": "src/preprocessing/preprocess_direction3_external.py",
                "variables": clean_records(
                    group,
                    ["original_name", "readable_name", "transformation"],
                ),
            }
        )

    for module, group in cses.groupby("module", sort=True):
        datasets.append(
            {
                "dataset": f"cses_{module}",
                "module_title": group["module_title"].iloc[0],
                "output_path": group["output_path"].iloc[0],
                "preprocessing_script": "src/preprocessing/preprocess_direction3_cses.py",
                "source_manifest": "data/exp/data-preprocessing/direction3_cses_source_manifest.csv",
                "standardized_identifiers": [
                    "Survey Year",
                    "Survey Wave",
                    "Main Linked Sample",
                    "Household ID",
                    "Person ID",
                    "PSU",
                    "Province Code",
                    "District Code",
                    "Commune Code",
                    "Village Code",
                    "Geography Link Matched",
                    "Source Dataset",
                    "Source Submodule",
                    "Module Row ID",
                ],
                "variables": clean_records(
                    group,
                    [
                        "original_name",
                        "canonical_original_name",
                        "readable_name",
                        "latest_english_label",
                        "survey_waves",
                        "transformation",
                    ],
                ),
            }
        )

    decisions = {
        "schema_version": 1,
        "research_direction": "Direction 3: mine contamination and climate resilience",
        "confirmed_on": "2026-08-18",
        "confirmation": "User approved the default preprocessing design.",
        "global_decisions": {
            "main_linked_sample": "CSES 2007-2021",
            "archived_unlinked_sample": "CSES 2004 is retained in CSES modules but excluded from the initial mine-linked analysis.",
            "outcome_priority": {
                "primary": [
                    "cultivated land and agricultural production",
                    "agricultural costs and sales",
                    "food consumption and food security",
                ],
                "secondary": [
                    "broader consumption and vulnerability",
                    "education",
                ],
                "mechanisms_retained": [
                    "migration",
                    "nonagricultural activity",
                    "liabilities",
                    "durable assets",
                    "livestock",
                    "employment",
                    "housing",
                    "village infrastructure",
                ],
            },
            "climate_construction": "Retain continuous rainfall anomalies and bottom/top decile shock indicators.",
            "historical_climate_linkage": (
                "Preserve the strict commune-code result. In a separate enhanced layer, use audited "
                "unique hierarchical names, high-confidence fuzzy commune names (score >= 0.85 and "
                "margin >= 0.08), then explicitly flagged district or province rainfall fallback."
            ),
            "mine_date_correction": "Correct the single 1913 survey date to 2013; preserve the original date and correction flag.",
            "missing_data": "No imputation.",
            "outliers": "No winsorization.",
            "monetary_variables": (
                "Retain nominal source values and add 2021 constant-price riels. Use interview-month "
                "food CPI for food values, annual food CPI when month is unavailable, annual education "
                "CPI for education expenditure, and annual all-items CPI for agricultural monetary values."
            ),
            "mine_zero_interpretation": "An unmatched village means no recorded point in the public baseline, not verified mine-free status.",
            "row_grain": (
                "Preserve each raw module's row grain and add separate household-wave and person-wave "
                "concept-level outcome files; source modules remain unchanged."
            ),
        },
        "analysis_spines": [
            {
                "dataset": "direction3_psu_year_analysis_skeleton",
                "output_path": "data/processed/direction3_psu_year_analysis_skeleton_preprocessed.parquet",
                "unit": "PSU by survey wave",
                "preprocessing_script": "src/preprocessing/build_direction3_analysis_skeleton.py",
            },
            {
                "dataset": "direction3_household_year_spine",
                "output_path": "data/processed/direction3_household_year_spine_preprocessed.parquet",
                "unit": "household by survey wave",
                "preprocessing_script": "src/preprocessing/build_direction3_analysis_skeleton.py",
            },
            {
                "dataset": "direction3_household_core_outcomes",
                "output_path": "data/processed/direction3_household_core_outcomes_preprocessed.parquet",
                "unit": "household by survey wave",
                "preprocessing_script": "src/preprocessing/construct_direction3_core_outcomes.py",
            },
            {
                "dataset": "direction3_education_core_outcomes",
                "output_path": "data/processed/direction3_education_core_outcomes_preprocessed.parquet",
                "unit": "person by survey wave",
                "preprocessing_script": "src/preprocessing/construct_direction3_core_outcomes.py",
            },
        ],
        "enhanced_climate_linkage": {
            "output_path": "data/processed/direction3_psu_year_climate_enhanced_preprocessed.parquet",
            "preprocessing_script": "src/preprocessing/repair_direction3_climate_linkage.py",
            "strict_link_preserved": True,
            "validation_report": "data/exp/data-preprocessing/direction3_climate_linkage_repair_validation.csv",
        },
        "monetary_deflation": {
            "target_price_basis": "2021 annual mean CPI",
            "source_dataset": "IMF Consumer Price Index (CPI), IMF.STA:CPI(5.0.0)",
            "source_manifest": "data/exp/data-preprocessing/direction3_cpi_source_manifest.csv",
            "monthly_cpi_output": "data/processed/cambodia_cpi_monthly_preprocessed.parquet",
            "annual_cpi_output": "data/processed/cambodia_cpi_annual_preprocessed.parquet",
            "preprocessing_script": "src/preprocessing/acquire_cambodia_cpi.py",
            "validation_report": "data/exp/data-preprocessing/direction3_cpi_validation.csv",
        },
        "constructed_outcomes": (
            clean_records(
                core_outcomes,
                ["variable_name", "full_name", "role", "unit", "construction"],
            )
            if not core_outcomes.empty
            else []
        ),
        "datasets": datasets,
        "validation_reports": [
            "data/exp/data-preprocessing/direction3_processed_validation.csv",
            "data/exp/data-preprocessing/direction3_climate_linkage_repair_validation.csv",
            "data/exp/data-preprocessing/direction3_core_outcome_validation.csv",
            "data/exp/data-preprocessing/direction3_cpi_validation.csv",
        ],
    }

    destination = exp / "decisions.json"
    destination.write_text(json.dumps(decisions, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(f"datasets={len(datasets)}")
    print(f"variable_decisions={sum(len(item['variables']) for item in datasets):,}")
    print(f"output={destination.relative_to(root)}")


if __name__ == "__main__":
    main()
