#!/usr/bin/env python3
"""Recover actual interview year and month for the CSES 2019-2020 wave."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd


SOURCE = Path("data/raw/CSE/CSES2019.zip")
MEMBER = "CSES2019/S01-17_HHOtherInfo.dta"
OUTPUT = Path("data/processed/cses_2019_interview_timing_preprocessed.parquet")
AUDIT = Path("data/exp/data-preprocessing/climate-welfare/cses-2019-interview-timing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source = root / SOURCE
    if not source.exists():
        raise FileNotFoundError(source)
    with zipfile.ZipFile(source) as archive:
        raw = archive.read(MEMBER)
    frame = pd.read_stata(
        io.BytesIO(raw),
        columns=["hhid", "psu", "yearsur", "monthsur"],
        convert_categoricals=False,
    ).rename(
        columns={
            "hhid": "Household ID",
            "psu": "PSU",
            "yearsur": "Interview Calendar Year",
            "monthsur": "Interview Month",
        }
    )
    frame.insert(0, "Survey Wave", "2019")
    frame["Household ID"] = frame["Household ID"].astype("string")
    frame["PSU"] = frame["PSU"].astype("string")
    frame["Interview Calendar Year"] = pd.to_numeric(
        frame["Interview Calendar Year"], errors="coerce"
    ).astype("Int16")
    frame["Interview Month"] = pd.to_numeric(frame["Interview Month"], errors="coerce").astype(
        "Int8"
    )
    if frame["Household ID"].duplicated().any():
        raise RuntimeError("Duplicate household IDs in CSES 2019 timing source")
    if not frame["Interview Calendar Year"].isin([2019, 2020]).all():
        raise RuntimeError("Unexpected interview calendar year in CSES 2019 timing source")
    if not frame["Interview Month"].between(1, 12).all():
        raise RuntimeError("Invalid interview month in CSES 2019 timing source")

    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    audit = root / AUDIT
    audit.mkdir(parents=True, exist_ok=True)
    coverage = (
        frame.groupby(["Interview Calendar Year", "Interview Month"], observed=True)
        .size()
        .rename("Households")
        .reset_index()
    )
    coverage.to_csv(audit / "coverage_by_calendar_month.csv", index=False)
    metadata = {
        "source_wave_label": "CSES 2019",
        "actual_field_period": "July 2019-June 2020",
        "rows": int(len(frame)),
        "households": int(frame["Household ID"].nunique()),
        "calendar_years": sorted(frame["Interview Calendar Year"].unique().astype(int).tolist()),
        "timing_fields": ["yearsur", "monthsur"],
        "raw_data_modified": False,
    }
    (audit / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (audit / "README.md").write_text(
        "# CSES 2019-2020 interview timing\n\n"
        f"Recovered actual interview calendar year and month for {len(frame):,} households. "
        "The wave runs from July 2019 through June 2020. One source record reports December "
        "2020 and is retained with a source-warning flag in downstream auditing rather than "
        "silently changed.\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
