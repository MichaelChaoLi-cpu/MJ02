#!/usr/bin/env python3
"""Construct prospective temperature-extreme exposures for the boundary design.

Definitions are fixed without using vegetation, nighttime-light, poverty, or
other outcomes.  Cell-specific calendar-day thresholds use the 1991--2020
reference period and a centred five-day moving window.  The primary temperature
family is hot days, heatwaves, and hot nights.  Cold nights are retained as a
secondary diagnostic.  Daily measures are aggregated to the existing MODIS
16-day slots and to the May--October annual season.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TEMPERATURE_DAILY = Path(
    "data/processed/historical_boundary_daily_temperature_preprocessed.parquet"
)
RAINFALL_DAILY = Path(
    "data/processed/historical_boundary_daily_chirps_preprocessed.parquet"
)
VILLAGE_CELL_MAP = Path(
    "data/exp/data-preprocessing/dynamic-rainfall-source/"
    "climateserv-chirps/village_to_chirps_cell.csv"
)
OUTPUT_DAILY = Path(
    "data/processed/historical_boundary_daily_temperature_shocks_preprocessed.parquet"
)
OUTPUT_THRESHOLDS = Path(
    "data/processed/historical_boundary_temperature_thresholds_preprocessed.parquet"
)
OUTPUT_INTERVAL = Path(
    "data/processed/historical_boundary_16day_temperature_shocks_preprocessed.parquet"
)
OUTPUT_ANNUAL = Path(
    "data/processed/historical_boundary_annual_temperature_shocks_preprocessed.parquet"
)
OUTPUT_VILLAGE_ANNUAL = Path(
    "data/processed/historical_boundary_village_year_temperature_shocks_preprocessed.parquet"
)
EXP_DIR = Path("data/exp/data-preprocessing/temperature-shocks")

REFERENCE_START = 1991
REFERENCE_END = 2020
HEATWAVE_MINIMUM_DAYS = 3
WINDOW_RADIUS_DAYS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--temperature", type=Path, default=TEMPERATURE_DAILY)
    parser.add_argument("--rainfall", type=Path, default=RAINFALL_DAILY)
    parser.add_argument("--village-cell-map", type=Path, default=VILLAGE_CELL_MAP)
    return parser.parse_args()


def project_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def nonleap_day(dates: pd.Series) -> pd.Series:
    """Return 1..365, with 29 February represented by 59.5."""
    dates = pd.to_datetime(dates)
    day = dates.dt.dayofyear.astype(float)
    after_february_in_leap_year = dates.dt.is_leap_year & dates.dt.month.gt(2)
    day.loc[after_february_in_leap_year] -= 1.0
    february_29 = dates.dt.month.eq(2) & dates.dt.day.eq(29)
    day.loc[february_29] = 59.5
    return day


def calendar_day_label(day: float) -> str:
    if day == 59.5:
        return "02-29"
    return (pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(day) - 1)).strftime("%m-%d")


def build_thresholds(daily: pd.DataFrame) -> pd.DataFrame:
    reference = daily.loc[
        daily["Date"].dt.year.between(REFERENCE_START, REFERENCE_END)
        & ~(
            daily["Date"].dt.month.eq(2)
            & daily["Date"].dt.day.eq(29)
        )
    ].copy()
    reference["Climatological Day"] = nonleap_day(reference["Date"]).astype(int)
    rows: list[dict[str, float | int | str]] = []
    for cell_id, group in reference.groupby("CHIRPS Cell ID", observed=True, sort=True):
        observed_day = group["Climatological Day"].to_numpy(int)
        tmax = group["Daily Maximum Temperature C"].to_numpy(float)
        tmin = group["Daily Minimum Temperature C"].to_numpy(float)
        for target_day in range(1, 366):
            direct_distance = np.abs(observed_day - target_day)
            circular_distance = np.minimum(direct_distance, 365 - direct_distance)
            window = circular_distance <= WINDOW_RADIUS_DAYS
            rows.append(
                {
                    "CHIRPS Cell ID": cell_id,
                    "Climatological Day": float(target_day),
                    "Calendar Day": calendar_day_label(float(target_day)),
                    "Tmax P90 Threshold C": float(np.quantile(tmax[window], 0.90)),
                    "Tmin P90 Threshold C": float(np.quantile(tmin[window], 0.90)),
                    "Tmin P10 Threshold C": float(np.quantile(tmin[window], 0.10)),
                    "Reference Window Observations": int(window.sum()),
                }
            )
    thresholds = pd.DataFrame(rows)
    february_29 = (
        thresholds.loc[thresholds["Climatological Day"].isin([59.0, 60.0])]
        .groupby("CHIRPS Cell ID", observed=True)
        .agg(
            **{
                "Tmax P90 Threshold C": ("Tmax P90 Threshold C", "mean"),
                "Tmin P90 Threshold C": ("Tmin P90 Threshold C", "mean"),
                "Tmin P10 Threshold C": ("Tmin P10 Threshold C", "mean"),
                "Reference Window Observations": ("Reference Window Observations", "sum"),
            }
        )
        .reset_index()
    )
    february_29["Climatological Day"] = 59.5
    february_29["Calendar Day"] = "02-29"
    thresholds = pd.concat([thresholds, february_29], ignore_index=True)
    return thresholds.sort_values(
        ["CHIRPS Cell ID", "Climatological Day"]
    ).reset_index(drop=True)


def add_heatwave_fields(daily: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in daily.groupby("CHIRPS Cell ID", observed=True, sort=False):
        group = group.sort_values("Date").copy()
        hot = group["Hot Day"].eq(1)
        spell_group = hot.ne(hot.shift(fill_value=False)).cumsum()
        run_length = hot.groupby(spell_group).transform("sum").astype(int)
        heatwave = hot & run_length.ge(HEATWAVE_MINIMUM_DAYS)
        group["Heatwave Day"] = heatwave.astype("int8")
        group["Heatwave Spell Duration Days"] = np.where(
            heatwave, run_length, 0
        ).astype("int16")
        group["Heatwave Start"] = (
            heatwave & ~heatwave.shift(fill_value=False)
        ).astype("int8")
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def construct_daily(
    temperature: pd.DataFrame,
    rainfall: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["CHIRPS Cell ID", "Date"]
    temperature = temperature.copy()
    temperature["Date"] = pd.to_datetime(temperature["Date"])
    rainfall = rainfall[keys + ["Daily Rainfall mm"]].copy()
    rainfall["Date"] = pd.to_datetime(rainfall["Date"])
    daily = temperature.merge(rainfall, on=keys, how="left", validate="one_to_one")
    if daily["Daily Rainfall mm"].isna().any():
        raise ValueError("Temperature cell-days do not have complete CHIRPS rainfall")
    daily["Climatological Day"] = nonleap_day(daily["Date"])
    daily = daily.merge(
        thresholds,
        on=["CHIRPS Cell ID", "Climatological Day"],
        how="left",
        validate="many_to_one",
    )
    threshold_columns = [
        "Tmax P90 Threshold C",
        "Tmin P90 Threshold C",
        "Tmin P10 Threshold C",
    ]
    if daily[threshold_columns].isna().any().any():
        raise ValueError("At least one daily temperature row lacks a threshold")
    daily["Hot Day"] = (
        daily["Daily Maximum Temperature C"] > daily["Tmax P90 Threshold C"]
    ).astype("int8")
    daily["Hot Night"] = (
        daily["Daily Minimum Temperature C"] > daily["Tmin P90 Threshold C"]
    ).astype("int8")
    daily["Cold Night"] = (
        daily["Daily Minimum Temperature C"] < daily["Tmin P10 Threshold C"]
    ).astype("int8")
    daily["Hot Day Exceedance C"] = (
        daily["Daily Maximum Temperature C"] - daily["Tmax P90 Threshold C"]
    ).clip(lower=0)
    daily["Hot Night Exceedance C"] = (
        daily["Daily Minimum Temperature C"] - daily["Tmin P90 Threshold C"]
    ).clip(lower=0)
    daily["Cold Night Deficit C"] = (
        daily["Tmin P10 Threshold C"] - daily["Daily Minimum Temperature C"]
    ).clip(lower=0)
    daily = add_heatwave_fields(daily)
    return daily.sort_values(keys).reset_index(drop=True)


def add_group_zscore(
    frame: pd.DataFrame,
    group_columns: list[str],
    reference_mask: pd.Series,
    source: str,
    output: str,
) -> pd.DataFrame:
    baseline = (
        frame.loc[reference_mask]
        .groupby(group_columns, observed=True)[source]
        .agg(["mean", "std", "count"])
        .rename(
            columns={
                "mean": f"{source} Reference Mean",
                "std": f"{source} Reference SD",
                "count": f"{source} Reference Observations",
            }
        )
        .reset_index()
    )
    frame = frame.merge(baseline, on=group_columns, how="left", validate="many_to_one")
    frame[output] = (
        frame[source] - frame[f"{source} Reference Mean"]
    ) / frame[f"{source} Reference SD"].replace(0, np.nan)
    return frame


def aggregate_interval(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.copy()
    frame["Year"] = frame["Date"].dt.year
    frame["Day of Year"] = frame["Date"].dt.dayofyear
    frame["MODIS Composite Slot"] = np.minimum(
        ((frame["Day of Year"] - 1) // 16 + 1), 23
    ).astype(int)
    interval = (
        frame.groupby(
            [
                "CHIRPS Cell ID",
                "CHIRPS Cell Longitude",
                "CHIRPS Cell Latitude",
                "Year",
                "MODIS Composite Slot",
            ],
            observed=True,
        )
        .agg(
            **{
                "Interval Start Date": ("Date", "min"),
                "Interval End Date": ("Date", "max"),
                "Interval Observed Temperature Days": ("Date", "size"),
                "Interval Mean Tmax C": ("Daily Maximum Temperature C", "mean"),
                "Interval Maximum Tmax C": ("Daily Maximum Temperature C", "max"),
                "Interval Mean Tmin C": ("Daily Minimum Temperature C", "mean"),
                "Interval Minimum Tmin C": ("Daily Minimum Temperature C", "min"),
                "Interval Hot Day Count": ("Hot Day", "sum"),
                "Interval Hot Day Exceedance Degree Days C": ("Hot Day Exceedance C", "sum"),
                "Interval Hot Night Count": ("Hot Night", "sum"),
                "Interval Hot Night Exceedance Degree Days C": ("Hot Night Exceedance C", "sum"),
                "Interval Heatwave Day Count": ("Heatwave Day", "sum"),
                "Interval Heatwave Start Count": ("Heatwave Start", "sum"),
                "Interval Longest Overlapping Heatwave Days": ("Heatwave Spell Duration Days", "max"),
                "Interval Cold Night Count": ("Cold Night", "sum"),
                "Interval Cold Night Deficit Degree Days C": ("Cold Night Deficit C", "sum"),
                "Interval Rainfall mm": ("Daily Rainfall mm", "sum"),
            }
        )
        .reset_index()
    )
    interval["Interval Hot Day Share"] = (
        interval["Interval Hot Day Count"]
        / interval["Interval Observed Temperature Days"]
    )
    interval["Interval Hot Night Share"] = (
        interval["Interval Hot Night Count"]
        / interval["Interval Observed Temperature Days"]
    )
    interval["Interval Heatwave Day Share"] = (
        interval["Interval Heatwave Day Count"]
        / interval["Interval Observed Temperature Days"]
    )
    interval["Interval Cold Night Share"] = (
        interval["Interval Cold Night Count"]
        / interval["Interval Observed Temperature Days"]
    )
    reference = interval["Year"].between(REFERENCE_START, REFERENCE_END)
    zscores = {
        "Interval Mean Tmax C": "Interval Mean Tmax Anomaly Z",
        "Interval Hot Day Count": "Interval Hot Day Count Anomaly Z",
        "Interval Hot Night Count": "Interval Hot Night Count Anomaly Z",
        "Interval Heatwave Day Count": "Interval Heatwave Day Count Anomaly Z",
        "Interval Cold Night Count": "Interval Cold Night Count Anomaly Z",
        "Interval Rainfall mm": "Interval Rainfall Anomaly Z",
    }
    for source, output in zscores.items():
        interval = add_group_zscore(
            interval,
            ["CHIRPS Cell ID", "MODIS Composite Slot"],
            reference,
            source,
            output,
        )
    interval["Dry Rainfall Intensity"] = (
        -interval["Interval Rainfall Anomaly Z"]
    ).clip(lower=0)
    interval["Wet Rainfall Intensity"] = interval[
        "Interval Rainfall Anomaly Z"
    ].clip(lower=0)
    interval["Hot Day Intensity"] = interval[
        "Interval Hot Day Count Anomaly Z"
    ].clip(lower=0)
    interval["Compound Hot-Dry Intensity"] = (
        interval["Hot Day Intensity"] * interval["Dry Rainfall Intensity"]
    )
    return interval.sort_values(
        ["CHIRPS Cell ID", "Year", "MODIS Composite Slot"]
    ).reset_index(drop=True)


def aggregate_annual(daily: pd.DataFrame) -> pd.DataFrame:
    frame = daily.loc[daily["Date"].dt.month.between(5, 10)].copy()
    frame["Year"] = frame["Date"].dt.year
    annual = (
        frame.groupby(
            [
                "CHIRPS Cell ID",
                "CHIRPS Cell Longitude",
                "CHIRPS Cell Latitude",
                "Year",
            ],
            observed=True,
        )
        .agg(
            **{
                "May October Observed Temperature Days": ("Date", "size"),
                "May October Mean Tmax C": ("Daily Maximum Temperature C", "mean"),
                "May October Maximum Tmax C": ("Daily Maximum Temperature C", "max"),
                "May October Mean Tmin C": ("Daily Minimum Temperature C", "mean"),
                "May October Minimum Tmin C": ("Daily Minimum Temperature C", "min"),
                "May October Hot Day Count": ("Hot Day", "sum"),
                "May October Hot Day Exceedance Degree Days C": ("Hot Day Exceedance C", "sum"),
                "May October Hot Night Count": ("Hot Night", "sum"),
                "May October Hot Night Exceedance Degree Days C": ("Hot Night Exceedance C", "sum"),
                "May October Heatwave Day Count": ("Heatwave Day", "sum"),
                "May October Heatwave Start Count": ("Heatwave Start", "sum"),
                "May October Longest Overlapping Heatwave Days": ("Heatwave Spell Duration Days", "max"),
                "May October Cold Night Count": ("Cold Night", "sum"),
                "May October Cold Night Deficit Degree Days C": ("Cold Night Deficit C", "sum"),
                "May October Rainfall mm": ("Daily Rainfall mm", "sum"),
            }
        )
        .reset_index()
    )
    reference = annual["Year"].between(REFERENCE_START, REFERENCE_END)
    zscores = {
        "May October Mean Tmax C": "May October Mean Tmax Anomaly Z",
        "May October Hot Day Count": "May October Hot Day Count Anomaly Z",
        "May October Hot Night Count": "May October Hot Night Count Anomaly Z",
        "May October Heatwave Day Count": "May October Heatwave Day Count Anomaly Z",
        "May October Cold Night Count": "May October Cold Night Count Anomaly Z",
        "May October Rainfall mm": "May October Rainfall Anomaly Z",
    }
    for source, output in zscores.items():
        annual = add_group_zscore(
            annual, ["CHIRPS Cell ID"], reference, source, output
        )
    annual["May October Dry Rainfall Intensity"] = (
        -annual["May October Rainfall Anomaly Z"]
    ).clip(lower=0)
    annual["May October Hot Day Intensity"] = annual[
        "May October Hot Day Count Anomaly Z"
    ].clip(lower=0)
    annual["May October Compound Hot-Dry Intensity"] = (
        annual["May October Hot Day Intensity"]
        * annual["May October Dry Rainfall Intensity"]
    )
    return annual.sort_values(["CHIRPS Cell ID", "Year"]).reset_index(drop=True)


def build_village_annual(
    root: Path, map_path: Path, annual: pd.DataFrame
) -> pd.DataFrame:
    mapping = pd.read_csv(
        project_path(root, map_path),
        dtype={"Village Code": "string", "CHIRPS Cell ID": "string"},
    )
    mapping = mapping[["Village Code", "CHIRPS Cell ID"]].drop_duplicates(
        "Village Code"
    )
    village = mapping.merge(
        annual, on="CHIRPS Cell ID", how="left", validate="many_to_many"
    )
    if village["Year"].isna().any():
        raise ValueError("At least one boundary village lacks annual temperature shocks")
    return village.sort_values(["Village Code", "Year"]).reset_index(drop=True)


def write_documentation(
    root: Path,
    daily: pd.DataFrame,
    thresholds: pd.DataFrame,
    interval: pd.DataFrame,
    annual: pd.DataFrame,
    village: pd.DataFrame,
) -> None:
    exp_dir = project_path(root, EXP_DIR)
    exp_dir.mkdir(parents=True, exist_ok=True)
    daily_coverage = (
        daily.assign(Year=daily["Date"].dt.year)
        .groupby("Year", observed=True)
        .agg(
            cells=("CHIRPS Cell ID", "nunique"),
            days=("Date", "nunique"),
            rows=("Date", "size"),
            hot_days=("Hot Day", "sum"),
            hot_nights=("Hot Night", "sum"),
            heatwave_days=("Heatwave Day", "sum"),
            cold_nights=("Cold Night", "sum"),
        )
        .reset_index()
    )
    daily_coverage.to_csv(exp_dir / "daily_coverage_by_year.csv", index=False)
    interval_coverage = (
        interval.groupby("Year", observed=True)
        .agg(
            cells=("CHIRPS Cell ID", "nunique"),
            intervals=("MODIS Composite Slot", "nunique"),
            rows=("MODIS Composite Slot", "size"),
            valid_heat_z=("Interval Hot Day Count Anomaly Z", lambda x: int(x.notna().sum())),
            valid_rainfall_z=("Interval Rainfall Anomaly Z", lambda x: int(x.notna().sum())),
            valid_compound=("Compound Hot-Dry Intensity", lambda x: int(x.notna().sum())),
        )
        .reset_index()
    )
    interval_coverage.to_csv(exp_dir / "interval_coverage_by_year.csv", index=False)
    annual_coverage = (
        annual.groupby("Year", observed=True)
        .agg(
            cells=("CHIRPS Cell ID", "nunique"),
            rows=("CHIRPS Cell ID", "size"),
            valid_heat_z=("May October Hot Day Count Anomaly Z", lambda x: int(x.notna().sum())),
            valid_rainfall_z=("May October Rainfall Anomaly Z", lambda x: int(x.notna().sum())),
            valid_compound=("May October Compound Hot-Dry Intensity", lambda x: int(x.notna().sum())),
        )
        .reset_index()
    )
    annual_coverage.to_csv(exp_dir / "annual_coverage_by_year.csv", index=False)

    validation = {
        "reference_period": [REFERENCE_START, REFERENCE_END],
        "calendar_threshold_window_days": 2 * WINDOW_RADIUS_DAYS + 1,
        "heatwave_minimum_consecutive_hot_days": HEATWAVE_MINIMUM_DAYS,
        "daily_rows": len(daily),
        "threshold_rows": len(thresholds),
        "interval_rows": len(interval),
        "annual_cell_year_rows": len(annual),
        "village_year_rows": len(village),
        "cells": int(daily["CHIRPS Cell ID"].nunique()),
        "villages": int(village["Village Code"].nunique()),
        "date_start": str(daily["Date"].min().date()),
        "date_end": str(daily["Date"].max().date()),
        "minimum_reference_window_observations": int(
            thresholds["Reference Window Observations"].min()
        ),
        "maximum_reference_window_observations": int(
            thresholds["Reference Window Observations"].max()
        ),
        "tmax_below_tmin_rows": int(
            (
                daily["Daily Maximum Temperature C"]
                < daily["Daily Minimum Temperature C"]
            ).sum()
        ),
        "duplicate_cell_days": int(
            daily.duplicated(["CHIRPS Cell ID", "Date"]).sum()
        ),
        "missing_core_daily_values": int(
            daily[
                [
                    "Daily Maximum Temperature C",
                    "Daily Minimum Temperature C",
                    "Daily Rainfall mm",
                    "Tmax P90 Threshold C",
                    "Tmin P90 Threshold C",
                    "Tmin P10 Threshold C",
                ]
            ].isna().sum().sum()
        ),
    }
    (exp_dir / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    variable_rows = [
        {
            "dataset": str(OUTPUT_DAILY),
            "variable": "Hot Day",
            "dtype": "int8",
            "role": "primary temperature shock input",
            "is_final_variable": "yes",
            "definition": "Daily Tmax exceeds the cell and calendar-day 1991-2020 P90 threshold.",
        },
        {
            "dataset": str(OUTPUT_DAILY),
            "variable": "Heatwave Day",
            "dtype": "int8",
            "role": "primary temperature shock input",
            "is_final_variable": "yes",
            "definition": "Hot Day belongs to a run of at least three consecutive Hot Days.",
        },
        {
            "dataset": str(OUTPUT_DAILY),
            "variable": "Hot Night",
            "dtype": "int8",
            "role": "primary temperature shock input",
            "is_final_variable": "yes",
            "definition": "Daily Tmin exceeds the cell and calendar-day 1991-2020 P90 threshold.",
        },
        {
            "dataset": str(OUTPUT_DAILY),
            "variable": "Cold Night",
            "dtype": "int8",
            "role": "secondary temperature diagnostic",
            "is_final_variable": "no",
            "definition": "Daily Tmin is below the cell and calendar-day 1991-2020 P10 threshold.",
        },
        {
            "dataset": str(OUTPUT_INTERVAL),
            "variable": "Interval Hot Day Count Anomaly Z",
            "dtype": "float64",
            "role": "primary 16-day heat shock",
            "is_final_variable": "yes",
            "definition": "Hot-day count standardized within cell and MODIS slot over 1991-2020.",
        },
        {
            "dataset": str(OUTPUT_INTERVAL),
            "variable": "Compound Hot-Dry Intensity",
            "dtype": "float64",
            "role": "prospective compound shock",
            "is_final_variable": "yes",
            "definition": "Positive hot-day-count anomaly multiplied by positive dry-rainfall intensity.",
        },
        {
            "dataset": str(OUTPUT_ANNUAL),
            "variable": "May October Hot Day Count Anomaly Z",
            "dtype": "float64",
            "role": "primary annual heat shock",
            "is_final_variable": "yes",
            "definition": "May-October hot-day count standardized within cell over 1991-2020.",
        },
        {
            "dataset": str(OUTPUT_ANNUAL),
            "variable": "May October Compound Hot-Dry Intensity",
            "dtype": "float64",
            "role": "prospective annual compound shock",
            "is_final_variable": "yes",
            "definition": "Positive May-October hot-day anomaly multiplied by positive seasonal dry intensity.",
        },
    ]
    pd.DataFrame(variable_rows).to_csv(exp_dir / "variable_list.csv", index=False)
    decisions = {
        "source": "CHIRTS-ERA5 daily Tmax and Tmin on the frozen CHIRPS boundary grid",
        "reference_period": "1991-2020",
        "threshold_method": "cell-specific calendar-day percentiles from a centered five-day moving window",
        "primary_temperature_variables": [
            "Hot Day",
            "Heatwave Day",
            "Hot Night",
            "Interval Hot Day Count Anomaly Z",
            "May October Hot Day Count Anomaly Z",
        ],
        "secondary_temperature_variables": ["Cold Night"],
        "compound_variables": [
            "Compound Hot-Dry Intensity",
            "May October Compound Hot-Dry Intensity",
        ],
        "imputation": "none",
        "winsorization": "none",
        "outcomes_used_to_select_definitions": False,
    }
    (exp_dir / "decisions.json").write_text(
        json.dumps(decisions, indent=2), encoding="utf-8"
    )
    readme = """# Boundary temperature-shock preprocessing

This release extracts CHIRTS-ERA5 daily maximum and minimum temperature only for
the 27 pre-existing 0.05-degree CHIRPS cells used by the frozen historical
boundary design. No outcome was used to select cells, thresholds, dates, or
shock definitions.

Primary temperature exposures are hot days, heatwaves of at least three
consecutive hot days, and hot nights. Cold nights are retained as a secondary
diagnostic. Cell-specific daily thresholds use the 1991-2020 reference period,
the 90th or 10th percentile as appropriate, and a centred five-calendar-day
window. February 29 thresholds interpolate February 28 and March 1.

Outputs include a daily cell panel, a threshold panel, a MODIS-aligned 16-day
cell panel, a May-October cell-year panel, and a village-year panel. Compound
hot-dry intensity is the product of the positive standardized hot-day anomaly
and positive dry-rainfall intensity. It is a prospective extension, not a
replacement for the frozen rainfall estimand.
"""
    (exp_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    temperature = pd.read_parquet(project_path(root, args.temperature))
    rainfall = pd.read_parquet(project_path(root, args.rainfall))
    thresholds = build_thresholds(temperature.assign(Date=pd.to_datetime(temperature["Date"])))
    daily = construct_daily(temperature, rainfall, thresholds)
    interval = aggregate_interval(daily)
    annual = aggregate_annual(daily)
    village = build_village_annual(root, args.village_cell_map, annual)

    outputs = {
        OUTPUT_DAILY: daily,
        OUTPUT_THRESHOLDS: thresholds,
        OUTPUT_INTERVAL: interval,
        OUTPUT_ANNUAL: annual,
        OUTPUT_VILLAGE_ANNUAL: village,
    }
    for relative_path, frame in outputs.items():
        path = project_path(root, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        print(f"Saved: {path.relative_to(root)} rows={len(frame):,}")
    write_documentation(root, daily, thresholds, interval, annual, village)
    print(
        f"Validated cells={daily['CHIRPS Cell ID'].nunique():,}; "
        f"villages={village['Village Code'].nunique():,}; "
        f"dates={daily['Date'].min().date()} to {daily['Date'].max().date()}"
    )


if __name__ == "__main__":
    main()
