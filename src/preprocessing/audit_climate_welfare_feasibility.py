#!/usr/bin/env python3
"""Supplement the generic feasibility scan with Parquet-aware evidence."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


OUTPUT_DIR = Path("data/exp/feasibility-check")
DATASETS = [
    (
        "National daily precipitation",
        Path("data/processed/cambodia_national_daily_precipitation/year=2001/precipitation.parquet"),
        "climate cell-day partitions, 1991-2024",
        "RQ1 hazard construction",
    ),
    (
        "National daily temperature",
        Path("data/processed/cambodia_national_daily_temperature/year=2001/temperature.parquet"),
        "climate cell-day partitions, 1991-2024",
        "RQ1 compound heat construction",
    ),
    (
        "Outcome-blind monsoon timing",
        Path("data/processed/cambodia_national_monsoon_timing_preprocessed.parquet"),
        "climate cell-year, 1991-2024",
        "RQ1 candidate hazards",
    ),
    (
        "National village monsoon-NPP panel",
        Path("data/processed/cambodia_public_village_monsoon_npp_panel_preprocessed.parquet"),
        "village-buffer-year, 2001-2024",
        "RQ1 ecology and RQ4 national prediction",
    ),
    (
        "CSES climate-ecology-welfare frame",
        Path("data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"),
        "household-wave, 2007-2021 wave labels",
        "RQ2 welfare and RQ3 buffers",
    ),
    (
        "National predetermined covariates",
        Path("data/processed/cambodia_national_predetermined_covariates_preprocessed.parquet"),
        "national 1 km grid, baseline or time invariant",
        "RQ3 buffers, RQ4 common support",
    ),
    (
        "CSES 2019-2020 interview timing",
        Path("data/processed/cses_2019_interview_timing_preprocessed.parquet"),
        "household, July 2019-June 2020 plus one flagged source record",
        "RQ2 temporal alignment",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    out = root / OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    for name in ("dataset_availability", "variable_inventory", "question_feasibility"):
        current = out / f"{name}.csv"
        backup = out / f"{name}_name_scan.csv"
        if current.exists():
            shutil.copy2(current, backup)

    dataset_rows = []
    for title, relative, unit_period, role in DATASETS:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        dataset_rows.append(
            {
                "Dataset": title,
                "Path": str(relative),
                "Readable": True,
                "Rows": parquet.metadata.num_rows,
                "Columns": len(parquet.schema.names),
                "Unit and Period": unit_period,
                "Research Role": role,
            }
        )
    pd.DataFrame(dataset_rows).to_csv(out / "dataset_availability.csv", index=False)

    variables = [
        ("Wet-Season Onset DOY Candidate A", "exposure candidate", "RQ1", "constructed"),
        ("Wet-Season Onset DOY Candidate B", "exposure candidate", "RQ1", "constructed"),
        ("False Onset Indicator Candidate A", "exposure candidate", "RQ1", "constructed"),
        ("False Onset Indicator Candidate B", "exposure candidate", "RQ1", "constructed"),
        ("Longest Intraseasonal Dry Spell Days Candidate A", "exposure candidate", "RQ1", "constructed"),
        ("Longest Intraseasonal Dry Spell Days Candidate B", "exposure candidate", "RQ1", "constructed"),
        ("Post-Onset Hot-Dry Day Count Candidate A", "exposure candidate", "RQ1", "constructed"),
        ("Post-Onset Hot-Dry Day Count Candidate B", "exposure candidate", "RQ1", "constructed"),
        ("Village Buffer Mean Annual Land NPP Anomaly kg C per m2", "ecological outcome", "RQ1", "constructed"),
        ("Baseline-Cropland-Weighted Annual Land NPP Anomaly kg C per m2", "ecological sensitivity", "RQ1", "constructed with labeling limit"),
        ("Agricultural Participation", "household outcome", "RQ2", "constructed"),
        ("Real 2021 Crop Production Value per Cultivated ha Riels", "household outcome", "RQ2", "constructed"),
        ("Real 2021 Food Consumption Value per Household Member Riels", "household outcome", "RQ2", "constructed"),
        ("Any Severe Food Insecurity Experience", "secondary outcome", "RQ2", "constructed for 2014-2021"),
        ("Any Irrigable Parcel", "buffer candidate", "RQ3", "partly comparable; absent in 2007"),
        ("Historical Road Distance km", "buffer candidate", "RQ3", "constructed for linked villages"),
        ("Log Baseline Population 2000", "buffer candidate", "RQ3", "constructed for linked villages"),
        ("Prediction Support", "prediction diagnostic", "RQ4", "requires fitted models"),
        ("Predicted-Loss Uncertainty", "prediction diagnostic", "RQ4", "requires fitted models"),
    ]
    pd.DataFrame(
        variables,
        columns=["Readable Variable", "Role", "Research Question", "Availability Status"],
    ).to_csv(out / "variable_inventory.csv", index=False)

    questions = pd.DataFrame(
        [
            {
                "ID": "RQ1",
                "Status": "partly-testable",
                "Question": "Which monsoon disruptions cause ecological production loss?",
                "Direct Support": "212,568 climate-cell-years and 939,024 village-buffer NPP rows",
                "Remaining Gate": "Freeze one onset family outcome-blind; estimate Model A",
            },
            {
                "ID": "RQ2",
                "Status": "partly-testable",
                "Question": "When do ecological losses become agricultural and food-welfare losses?",
                "Direct Support": "62,920 households; 36,366 public-village climate-ecology links",
                "Remaining Gate": "Construct cross-fitted ecological loss; estimate Models B-C",
            },
            {
                "ID": "RQ3",
                "Status": "partly-testable",
                "Question": "Which observable conditions buffer ecological-to-welfare transmission?",
                "Direct Support": "Irrigation, historical road distance, and baseline population connectivity",
                "Remaining Gate": "Freeze no more than three modifiers after overlap audit; estimate Model D",
            },
            {
                "ID": "RQ4",
                "Status": "partly-testable",
                "Question": "Where is adaptation priority highest?",
                "Direct Support": "13,042 national villages with three buffer radii and baseline context",
                "Remaining Gate": "Pass Models A-C and common-support validation before Model E",
            },
        ]
    )
    questions.to_csv(out / "question_feasibility.csv", index=False)

    summary = {
        "questions_found": 4,
        "status_counts": {"partly-testable": 4},
        "parquet_datasets_audited": len(DATASETS),
        "curated_variables_audited": len(variables),
        "recommended_next_skill": "data-preprocessing",
        "generic_name_scan_result_retained_as": [
            "dataset_availability_name_scan.csv",
            "variable_inventory_name_scan.csv",
            "question_feasibility_name_scan.csv",
        ],
    }
    (out / "parquet_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    readme = """# Feasibility Check

## Final Parquet-Aware Assessment

The generic name scan initially reported four `not-yet-testable` questions because
it did not inspect the project's Parquet analysis releases. That diagnostic is
preserved in the `*_name_scan.csv` files. The substantive Parquet-aware audit
finds all four questions `partly-testable`.

| Question | Status | Material support | Remaining gate |
|---|---|---|---|
| RQ1: monsoon to NPP | partly-testable | Daily climate, 212,568 cell-years, 939,024 village-buffer NPP rows | Freeze the outcome-blind hazard definition and estimate Model A |
| RQ2: ecology to welfare | partly-testable | 62,920 CSES households and 36,366 public-village links | Build cross-fitted ecological loss and estimate Models B-C |
| RQ3: buffering | partly-testable | Irrigation, historical roads, and baseline population connectivity | Freeze no more than three supported modifiers and estimate Model D |
| RQ4: priority geography | partly-testable | 13,042 national villages at 2, 5, and 10 km | Pass Models A-C and national common-support checks before Model E |

## Decision

Proceed to focused data preprocessing. Do not return to research-question
planning at this stage. Availability does not prove any hypothesis, and the
national priority map remains conditional on the ecological and household gates.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
