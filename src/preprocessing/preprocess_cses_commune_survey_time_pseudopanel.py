#!/usr/bin/env python3
"""Build the CSES commune-by-survey-time climate-food pseudo-panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
OUTPUT = ROOT / "data/processed/cses_commune_survey_time_climate_food_pseudopanel_preprocessed.parquet"
AUDIT_DIR = ROOT / "data/exp/data-preprocessing/cses-commune-survey-time-pseudopanel"

WEIGHT = "Household Survey Weight"
FOOD = "Real 2021 Food Consumption Value per Household Member Riels"
HEAT = "Last Complete Season Absolute Heat Day Count 35 C Candidate B"
RAIN = "Last Complete Season May-October Precipitation Anomaly Z"
ONSET = "Last Complete Season Wet-Season Onset Anomaly Z Candidate B"
DRY = "Last Complete Season Longest Dry Spell Anomaly Z Candidate B"

GROUP_KEYS = [
    "Commune Code",
    "Survey Year",
    "Survey Wave",
    "Interview Calendar Year",
    "Interview Month",
    "Last Complete May-October Season Year",
]

WEIGHTED_VARIABLES = {
    "Survey-Weighted Mean Log Real Food Consumption per Member": "Log Real Food Consumption per Member",
    "Survey-Weighted Mean Real Food Consumption per Member Riels": FOOD,
    "Survey-Weighted Mean Absolute Heat Days 35 C": HEAT,
    "Survey-Weighted Mean May-October Precipitation Anomaly Z": RAIN,
    "Survey-Weighted Mean Wet-Season Onset Anomaly Z": ONSET,
    "Survey-Weighted Mean Longest Intraseasonal Dry Spell Anomaly Z": DRY,
    "Survey-Weighted Mean Household Size": "Household Size",
    "Survey-Weighted Mean Female Household Member Share": "Female Household Member Share",
    "Survey-Weighted Mean Household Member Age Years": "Mean Household Member Age Years",
    "Survey-Weighted Mean Child Age 0-14 Share": "Child Age 0-14 Share",
    "Survey-Weighted Mean Older Age 65 Plus Share": "Older Age 65 Plus Share",
    "Survey-Weighted Mean Household Dependency Ratio": "Household Dependency Ratio",
    "Survey-Weighted Agricultural Participation Share": "Agricultural Participation Indicator",
    "Survey-Weighted Urban Household Share": "Urban Household Indicator",
}


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(frame[column], errors="coerce")
    weights = pd.to_numeric(frame[WEIGHT], errors="coerce")
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def first_nonmissing(series: pd.Series) -> object:
    values = series.dropna()
    return values.iloc[0] if len(values) else None


def prepare_households() -> tuple[pd.DataFrame, dict[str, int]]:
    columns = [
        "Survey Year", "Survey Wave", "Interview Calendar Year", "Interview Month",
        "Last Complete May-October Season Year", "Household ID", "Province Code",
        "Province Name", "District Code", "District Name", "Commune Code", "Commune Name",
        "Village Code", "Urban Rural", "Climate Ecology Link Available", WEIGHT, FOOD,
        HEAT, RAIN, ONSET, DRY, "Household Size", "Female Household Member Share",
        "Mean Household Member Age Years", "Child Age 0-14 Share", "Older Age 65 Plus Share",
        "Household Dependency Ratio", "Agricultural Participation", "Point Longitude",
        "Point Latitude",
    ]
    frame = pd.read_parquet(INPUT, columns=columns)
    counts = {"input_households": int(len(frame))}
    frame = frame.loc[frame["Climate Ecology Link Available"].eq(1)].copy()
    counts["linked_households"] = int(len(frame))
    required = GROUP_KEYS + [WEIGHT, FOOD, HEAT, RAIN, ONSET, DRY, "Point Longitude", "Point Latitude"]
    complete = frame[required].notna().all(axis=1)
    complete &= pd.to_numeric(frame[WEIGHT], errors="coerce").gt(0)
    complete &= pd.to_numeric(frame[FOOD], errors="coerce").gt(0)
    frame = frame.loc[complete].copy()
    counts["analysis_eligible_households"] = int(len(frame))
    frame["Log Real Food Consumption per Member"] = np.log(frame[FOOD].astype(float))
    frame["Agricultural Participation Indicator"] = frame["Agricultural Participation"].astype(float)
    frame["Urban Household Indicator"] = frame["Urban Rural"].astype("string").str.startswith("1").astype(float)
    return frame, counts


def build_panel(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for keys, part in frame.groupby(GROUP_KEYS, observed=True, sort=True, dropna=False):
        record = dict(zip(GROUP_KEYS, keys, strict=True))
        record.update(
            {
                "Province Code": first_nonmissing(part["Province Code"]),
                "District Code": first_nonmissing(part["District Code"]),
                "Household Count": int(len(part)),
                "Village Count": int(part["Village Code"].nunique()),
                "Survey Weight Sum": float(part[WEIGHT].sum()),
                "Survey-Weighted Mean Sampled Longitude": weighted_mean(part, "Point Longitude"),
                "Survey-Weighted Mean Sampled Latitude": weighted_mean(part, "Point Latitude"),
            }
        )
        for readable, source in WEIGHTED_VARIABLES.items():
            record[readable] = weighted_mean(part, source)
        records.append(record)
    panel = pd.DataFrame(records)
    panel["Survey-Time Fixed-Effect ID"] = (
        panel["Survey Wave"].astype(str)
        + "_" + panel["Interview Calendar Year"].astype(int).astype(str)
        + "_" + panel["Interview Month"].astype(int).astype(str).str.zfill(2)
    )
    panel["Primary Minimum Five Households"] = panel["Household Count"].ge(5)
    panel["Robustness Minimum Ten Households"] = panel["Household Count"].ge(10)

    commune = (
        frame.groupby("Commune Code", observed=True)
        .agg(
            **{
                "Commune Survey Wave Count": ("Survey Year", "nunique"),
                "Commune Reference Longitude": ("Point Longitude", "mean"),
                "Commune Reference Latitude": ("Point Latitude", "mean"),
            }
        )
        .reset_index()
    )
    commune["Repeated Commune Indicator"] = commune["Commune Survey Wave Count"].ge(2)
    commune["Spatial Block ID"] = (
        np.floor((commune["Commune Reference Longitude"] - 102.0) / 0.75).astype("Int64").astype(str)
        + "_"
        + np.floor((commune["Commune Reference Latitude"] - 10.0) / 0.75).astype("Int64").astype(str)
    )
    panel = panel.merge(commune, on="Commune Code", how="left", validate="many_to_one")
    panel = panel.sort_values(GROUP_KEYS).reset_index(drop=True)
    return panel


def write_audit(frame: pd.DataFrame, panel: pd.DataFrame, counts: dict[str, int]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    coverage = (
        panel.groupby("Survey Year", observed=True)
        .agg(
            Cells=("Commune Code", "size"),
            Communes=("Commune Code", "nunique"),
            Households=("Household Count", "sum"),
            Primary_Cells=("Primary Minimum Five Households", "sum"),
            Robustness_Cells=("Robustness Minimum Ten Households", "sum"),
        )
        .reset_index()
    )
    coverage.to_csv(AUDIT_DIR / "coverage_by_survey_wave.csv", index=False)
    panel["Household Count"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_frame(
        "Household Count"
    ).to_csv(AUDIT_DIR / "cell_size_summary.csv")

    roles = {
        "Survey-Weighted Mean Log Real Food Consumption per Member": "Primary pseudo-panel outcome",
        "Survey-Weighted Mean Real Food Consumption per Member Riels": "Natural-unit outcome support",
        "Survey-Weighted Mean Absolute Heat Days 35 C": "Primary pseudo-panel exposure",
        "Survey-Weighted Mean May-October Precipitation Anomaly Z": "Climate control",
        "Survey-Weighted Mean Wet-Season Onset Anomaly Z": "Climate control",
        "Survey-Weighted Mean Longest Intraseasonal Dry Spell Anomaly Z": "Climate control",
        "Survey Weight Sum": "Pseudo-panel estimation weight",
        "Household Count": "Primary and robustness sample gate",
        "Commune Code": "Stable area fixed-effect identifier",
        "Survey-Time Fixed-Effect ID": "Survey timing fixed-effect identifier",
        "Spatial Block ID": "Inference cluster identifier",
    }
    variable_rows = []
    for column in panel.columns:
        variable_rows.append(
            {
                "readable_name": column,
                "dtype": str(panel[column].dtype),
                "null_pct": float(panel[column].isna().mean() * 100),
                "role": roles.get(column, "Supporting pseudo-panel variable"),
                "is_final_variable": "yes",
            }
        )
    pd.DataFrame(variable_rows).to_csv(AUDIT_DIR / "variable_list.csv", index=False)

    metadata = {
        **counts,
        "output_rows": int(len(panel)),
        "output_columns": int(panel.shape[1]),
        "communes": int(panel["Commune Code"].nunique()),
        "repeated_communes": int(
            panel.loc[panel["Repeated Commune Indicator"], "Commune Code"].nunique()
        ),
        "primary_cells_minimum_five": int(panel["Primary Minimum Five Households"].sum()),
        "robustness_cells_minimum_ten": int(panel["Robustness Minimum Ten Households"].sum()),
        "primary_households_represented": int(
            panel.loc[panel["Primary Minimum Five Households"], "Household Count"].sum()
        ),
        "construction": "Commune by survey wave by interview calendar year and month",
        "missing_data": "No imputation; weighted means use available values for secondary composition fields",
    }
    (AUDIT_DIR / "processing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (AUDIT_DIR / "README.md").write_text(
        "# CSES Commune Survey-Time Climate-Food Pseudo-Panel\n\n"
        "This release aggregates linked CSES households to stable commune-by-survey-time cells. "
        "All outcomes, exposures, and composition measures are survey-weighted. The release keeps "
        "all cells and provides separate minimum-five and minimum-ten household eligibility flags.\n\n"
        f"- Eligible households: {len(frame):,}\n"
        f"- Output cells: {len(panel):,}\n"
        f"- Communes: {panel['Commune Code'].nunique():,}\n"
        f"- Primary cells (at least five households): {panel['Primary Minimum Five Households'].sum():,}\n",
        encoding="utf-8",
    )


def validate(panel: pd.DataFrame, counts: dict[str, int]) -> None:
    assert counts["linked_households"] == 46_445
    assert counts["analysis_eligible_households"] == 46_445
    assert not panel.duplicated(GROUP_KEYS).any()
    assert panel["Household Count"].sum() == counts["analysis_eligible_households"]
    assert panel["Commune Code"].nunique() == 1_298
    assert panel.loc[panel["Repeated Commune Indicator"], "Commune Code"].nunique() == 994
    assert panel["Primary Minimum Five Households"].mean() > 0.98
    assert panel["Robustness Minimum Ten Households"].mean() > 0.95
    assert panel["Survey-Weighted Mean Log Real Food Consumption per Member"].notna().all()
    assert panel["Survey-Weighted Mean Absolute Heat Days 35 C"].notna().all()


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)
    households, counts = prepare_households()
    panel = build_panel(households)
    validate(panel, counts)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT, index=False, compression="zstd")
    write_audit(households, panel, counts)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Rows={len(panel):,}; columns={panel.shape[1]}; communes={panel['Commune Code'].nunique():,}")
    print(
        f"Primary cells={panel['Primary Minimum Five Households'].sum():,}; "
        f"robustness cells={panel['Robustness Minimum Ten Households'].sum():,}"
    )


if __name__ == "__main__":
    main()
