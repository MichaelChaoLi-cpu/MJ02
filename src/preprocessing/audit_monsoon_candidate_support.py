#!/usr/bin/env python3
"""Audit monsoon candidates without reading ecological or household outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


INPUT = Path("data/processed/cambodia_public_village_monsoon_npp_panel_preprocessed.parquet")
OUTPUT = Path("data/exp/data-preprocessing/climate-welfare/monsoon-candidate-support")

FIELDS = {
    "Onset DOY A": "Village Buffer Mean Wet-Season Onset DOY Candidate A",
    "Onset Z A": "Village Buffer Mean Wet-Season Onset DOY Candidate A Anomaly Z",
    "False Onset Share A": "Village Buffer Mean False Onset Indicator Candidate A",
    "Longest Dry Spell A": "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate A",
    "Longest Dry Spell Z A": "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate A Anomaly Z",
    "Hot-Dry Days A": "Village Buffer Mean Post-Onset Hot-Dry Day Count Candidate A",
    "Hot-Dry Days Z A": "Village Buffer Mean Post-Onset Hot-Dry Day Count Candidate A Anomaly Z",
    "Onset DOY B": "Village Buffer Mean Wet-Season Onset DOY Candidate B",
    "Onset Z B": "Village Buffer Mean Wet-Season Onset DOY Candidate B Anomaly Z",
    "False Onset Share B": "Village Buffer Mean False Onset Indicator Candidate B",
    "Longest Dry Spell B": "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B",
    "Longest Dry Spell Z B": "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z",
    "Hot-Dry Days B": "Village Buffer Mean Post-Onset Hot-Dry Day Count Candidate B",
    "Hot-Dry Days Z B": "Village Buffer Mean Post-Onset Hot-Dry Day Count Candidate B Anomaly Z",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source = root / INPUT
    if not source.exists():
        raise FileNotFoundError(source)
    columns = ["National Village Point ID", "Buffer Radius km", "Year", *FIELDS.values()]
    frame = pd.read_parquet(source, columns=columns)
    frame = frame.loc[frame["Buffer Radius km"].eq(5)].copy()
    out = root / OUTPUT
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for short, field in FIELDS.items():
        values = frame[field]
        two_way = (
            values
            - frame.groupby("National Village Point ID", observed=True)[field].transform("mean")
            - frame.groupby("Year", observed=True)[field].transform("mean")
            + values.mean()
        )
        rows.append(
            {
                "Candidate Measure": short,
                "Readable Variable": field,
                "Observed Rows": int(values.notna().sum()),
                "Observed Share": float(values.notna().mean()),
                "Mean": float(values.mean()),
                "SD": float(values.std()),
                "Two-Way Demeaned SD": float(two_way.std()),
                "P01": float(values.quantile(0.01)),
                "P50": float(values.quantile(0.50)),
                "P99": float(values.quantile(0.99)),
            }
        )
    support = pd.DataFrame(rows)
    support.to_csv(out / "candidate_support.csv", index=False)

    correlation_pairs = [
        ("Onset DOY", FIELDS["Onset DOY A"], FIELDS["Onset DOY B"]),
        ("Longest Dry Spell", FIELDS["Longest Dry Spell A"], FIELDS["Longest Dry Spell B"]),
        ("Hot-Dry Days", FIELDS["Hot-Dry Days A"], FIELDS["Hot-Dry Days B"]),
    ]
    correlations = pd.DataFrame(
        [
            {
                "Measure": label,
                "Candidate A Variable": a,
                "Candidate B Variable": b,
                "Correlation": float(frame[[a, b]].corr().iloc[0, 1]),
            }
            for label, a, b in correlation_pairs
        ]
    )
    correlations.to_csv(out / "candidate_a_b_correlations.csv", index=False)

    yearly_fields = [
        FIELDS["Onset DOY A"], FIELDS["Onset DOY B"],
        FIELDS["False Onset Share A"], FIELDS["False Onset Share B"],
        FIELDS["Longest Dry Spell A"], FIELDS["Longest Dry Spell B"],
        FIELDS["Hot-Dry Days A"], FIELDS["Hot-Dry Days B"],
    ]
    frame.groupby("Year", observed=True)[yearly_fields].mean().reset_index().to_csv(
        out / "national_means_by_year.csv", index=False
    )

    recommendation = {
        "primary_onset_family": "Candidate B",
        "primary_continuous_measures": [
            "Village Buffer Mean Wet-Season Onset DOY Candidate B Anomaly Z",
            "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z",
            "Village Buffer Mean Post-Onset Hot-Dry Day Count Candidate B Anomaly Z",
        ],
        "diagnostic_only": [
            "Village Buffer Mean False Onset Indicator Candidate B",
        ],
        "robustness_family": "Candidate A",
        "rationale": [
            "Candidate B has near-complete coverage and material within-village, net-of-year variation.",
            "Its average onset occurs in early May and follows a regionally grounded 30 mm over three days persistence definition.",
            "Candidate B false onset is too rare for a main continuous regressor and remains diagnostic.",
            "Candidate A and B dry-spell and hot-dry measures correlate above 0.97, supporting definition robustness.",
        ],
        "outcomes_read": False,
        "npp_columns_read": False,
        "household_columns_read": False,
    }
    (out / "recommendation.json").write_text(
        json.dumps(recommendation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out / "README.md").write_text(
        "# Outcome-blind monsoon candidate support\n\n"
        "Candidate B is recommended for the primary continuous hazard family: local onset "
        "anomaly, longest post-onset dry-spell anomaly, and post-onset hot-dry-day anomaly. "
        "Candidate A is the fixed definition robustness family. Candidate B false onset is "
        "retained as a diagnostic because it is rare. No NPP or household outcome column was "
        "read in this audit.\n",
        encoding="utf-8",
    )
    print(json.dumps(recommendation, indent=2, ensure_ascii=False))
    print(support.to_string(index=False))


if __name__ == "__main__":
    main()
