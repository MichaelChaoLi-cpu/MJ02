#!/usr/bin/env python3
"""Finalize the public mine, climate, and CSES geography layers for Direction 3.

The exploratory builders retain source-style names so that they can be audited.
This script applies the confirmed, English-readable schema and writes the
analysis-facing Parquet files under ``data/processed``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


MINE_RECORD_NAMES = {
    "baseline_record_id": "Mine Baseline Record ID",
    "village_code_full": "Village Code",
    "commune_code_full": "Commune Code",
    "province_name": "Province Name",
    "district_name": "District Name",
    "commune_name": "Commune Name",
    "village_name": "Village Name",
    "fear_level": "Fear Level",
    "land_class": "Land Class",
    "proximity": "Proximity",
    "operator": "Mine Operator",
    "survey_date_raw": "Mine Survey Date Original",
    "survey_date_suspect": "Mine Survey Date Corrected Flag",
    "survey_date_candidate_corrected": "Mine Survey Date",
    "longitude": "Mine Point Longitude",
    "latitude": "Mine Point Latitude",
    "geometry": "Mine Point Geometry",
}

MINE_EXPOSURE_NAMES = {
    "village_code_full": "Village Code",
    "commune_code_full": "Commune Code",
    "province_name": "Province Name",
    "district_name": "District Name",
    "commune_name": "Commune Name",
    "village_name": "Village Name",
    "contamination_record_count": "Mine Baseline Record Count",
    "survey_date_first": "First Mine Survey Date",
    "survey_date_last": "Last Mine Survey Date",
    "suspect_date_count": "Corrected Mine Survey Date Count",
    "contamination_longitude_mean": "Mean Mine Point Longitude",
    "contamination_latitude_mean": "Mean Mine Point Latitude",
    "operator_count": "Mine Operator Count",
    "log_contamination_record_count": "Log Mine Baseline Record Count",
    "fear_high_count": "High Fear Record Count",
    "fear_low_count": "Low Fear Record Count",
    "fear_medium_count": "Medium Fear Record Count",
    "fear_none_count": "No Fear Record Count",
    "fear_missing_count": "Missing Fear Record Count",
    "proximity_far_count": "Far Proximity Record Count",
    "proximity_near_count": "Near Proximity Record Count",
    "proximity_very_far_count": "Very Far Proximity Record Count",
    "proximity_very_near_count": "Very Near Proximity Record Count",
    "proximity_missing_count": "Missing Proximity Record Count",
    "land_class_a1_count": "Land Class A1 Record Count",
    "land_class_a2_count": "Land Class A2 Record Count",
    "land_class_a2_1_count": "Land Class A2 1 Record Count",
    "land_class_a2_2_count": "Land Class A2 2 Record Count",
    "land_class_a3_count": "Land Class A3 Record Count",
    "land_class_a4_count": "Land Class A4 Record Count",
    "land_class_b1_count": "Land Class B1 Record Count",
    "land_class_b1_1_count": "Land Class B1 1 Record Count",
    "land_class_b1_2_count": "Land Class B1 2 Record Count",
    "land_class_b1_3_count": "Land Class B1 3 Record Count",
    "land_class_b1_4_count": "Land Class B1 4 Record Count",
    "land_class_b1_5_count": "Land Class B1 5 Record Count",
    "land_class_b2_count": "Land Class B2 Record Count",
    "operators": "Mine Operators",
}

CLIMATE_MONTH_NAMES = {
    "commune_code_full": "Commune Code",
    "commune_name_2014": "Commune Name 2014",
    "year": "Year",
    "month": "Month",
    "rainfall_mm": "Rainfall mm",
    "grid_cell_count": "Climate Grid Cell Count",
    "extraction_method": "Climate Extraction Method",
    "calendar_month_mean_mm": "Calendar Month Mean Rainfall mm",
    "calendar_month_sd_mm": "Calendar Month Rainfall SD mm",
    "monthly_rainfall_anomaly_z": "Monthly Rainfall Anomaly Z",
}

CLIMATE_YEAR_NAMES = {
    "commune_code_full": "Commune Code",
    "commune_name_2014": "Commune Name 2014",
    "year": "Year",
    "annual_rainfall_mm": "Annual Rainfall mm",
    "observed_months": "Observed Climate Months",
    "grid_cell_count": "Climate Grid Cell Count",
    "extraction_method": "Climate Extraction Method",
    "may_oct_rainfall_mm": "May October Rainfall mm",
    "annual_rainfall_anomaly_z": "Annual Rainfall Anomaly Z",
    "annual_rainfall_bottom_decile": "Annual Rainfall Bottom Decile",
    "annual_rainfall_top_decile": "Annual Rainfall Top Decile",
    "may_oct_rainfall_anomaly_z": "May October Rainfall Anomaly Z",
    "may_oct_rainfall_bottom_decile": "May October Rainfall Bottom Decile",
    "may_oct_rainfall_top_decile": "May October Rainfall Top Decile",
}

CROSSWALK_NAMES = {
    "survey_year": "Survey Wave",
    "psu": "PSU",
    "province_code": "Province Code Component",
    "district_code": "District Code Component",
    "commune_code": "Commune Code Component",
    "village_code": "Village Code Component",
    "province_name": "Province Name",
    "district_name": "District Name",
    "commune_name": "Commune Name",
    "village_name": "Village Name",
    "urban_rural": "Urban Rural",
    "survey_month": "Survey Month",
    "commune_code_full": "Commune Code",
    "village_code_full": "Village Code",
    "mine_baseline_match": "Mine Baseline Recorded",
    "source_dataset": "Source Dataset",
}


def rename_and_validate(frame: pd.DataFrame, mapping: dict[str, str], dataset: str) -> pd.DataFrame:
    missing = sorted(set(mapping).difference(frame.columns))
    extra = sorted(set(frame.columns).difference(mapping))
    if missing or extra:
        raise ValueError(f"{dataset}: schema mismatch; missing={missing}, extra={extra}")
    output = frame.rename(columns=mapping)
    if len(output.columns) != len(set(output.columns)):
        raise ValueError(f"{dataset}: readable names are not unique")
    if any(not column.isascii() for column in output.columns):
        raise ValueError(f"{dataset}: non-ASCII readable column name")
    return output


def dictionary_rows(dataset: str, output_path: Path, mapping: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "dataset": dataset,
            "output_path": str(output_path),
            "original_name": original,
            "readable_name": readable,
            "transformation": "rename only; values preserved",
        }
        for original, readable in mapping.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    exp = root / "data" / "exp"
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    specifications = [
        (
            "mine_baseline_records",
            exp / "data-preprocessing" / "mine_baseline_records_candidate.parquet",
            processed / "mine_baseline_records_preprocessed.parquet",
            MINE_RECORD_NAMES,
            True,
        ),
        (
            "mine_exposure_village",
            exp / "data-preprocessing" / "mine_exposure_village_candidate.parquet",
            processed / "mine_exposure_village_preprocessed.parquet",
            MINE_EXPOSURE_NAMES,
            False,
        ),
        (
            "climate_commune_month",
            exp / "data-preprocessing" / "climate_commune_month_candidate.parquet",
            processed / "climate_commune_month_preprocessed.parquet",
            CLIMATE_MONTH_NAMES,
            False,
        ),
        (
            "climate_commune_year",
            exp / "data-preprocessing" / "climate_commune_year_candidate.parquet",
            processed / "climate_commune_year_preprocessed.parquet",
            CLIMATE_YEAR_NAMES,
            False,
        ),
        (
            "cses_village_crosswalk",
            exp / "feasibility-check" / "cses_village_crosswalk_candidate.csv",
            processed / "cses_village_crosswalk_preprocessed.parquet",
            CROSSWALK_NAMES,
            False,
        ),
    ]

    rows: list[dict[str, str]] = []
    for dataset, source, destination, mapping, geospatial in specifications:
        if source.suffix == ".csv":
            frame = pd.read_csv(source, dtype=str)
            frame["mine_baseline_match"] = frame["mine_baseline_match"].str.lower().eq("true")
        else:
            frame = gpd.read_parquet(source) if geospatial else pd.read_parquet(source)
        output = rename_and_validate(frame, mapping, dataset)
        if dataset == "cses_village_crosswalk":
            output["PSU"] = output["PSU"].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(5)
            for column, width in [("Commune Code", 6), ("Village Code", 8)]:
                output[column] = output[column].astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(width)
        output.to_parquet(destination, index=False)
        rows.extend(dictionary_rows(dataset, destination.relative_to(root), mapping))
        print(f"{dataset}: rows={len(output):,}, columns={len(output.columns)}, output={destination.relative_to(root)}")

    dictionary = pd.DataFrame(rows)
    dictionary_path = exp / "data-preprocessing" / "direction3_external_variable_dictionary.csv"
    dictionary.to_csv(dictionary_path, index=False)
    print(f"dictionary_rows={len(dictionary):,}")


if __name__ == "__main__":
    main()
