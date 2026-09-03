#!/usr/bin/env python3
"""Align daily CHIRPS rainfall to MODIS composites and freeze extreme events.

This preprocessing step is outcome-blind with respect to historical-side
contrasts.  Extreme dry and wet events are defined solely from interval-aligned
rainfall relative to a fixed 1991--2020 cell-by-calendar-slot reference.  It
creates a village-composite panel, a stacked event panel, and event-level loss
and recovery outcomes without estimating any Southwest-minus-West coefficient.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VEGETATION_PANEL = Path("data/processed/historical_boundary_16day_vegetation_preprocessed.parquet")
RAINFALL_PANEL = Path("data/processed/historical_boundary_daily_chirps_preprocessed.parquet")
CELL_MAP = Path("data/exp/data-preprocessing/dynamic-rainfall-source/climateserv-chirps/village_to_chirps_cell.csv")
DEFAULT_PANEL_OUTPUT = Path("data/processed/historical_boundary_16day_climate_vegetation_preprocessed.parquet")
DEFAULT_STACK_OUTPUT = Path("data/processed/historical_boundary_vegetation_event_stack_preprocessed.parquet")
DEFAULT_EVENT_OUTPUT = Path("data/processed/historical_boundary_vegetation_event_outcomes_preprocessed.parquet")
EXP_DIR = Path("data/exp/data-preprocessing/high-frequency-boundary-events")

REFERENCE_START = 1991
REFERENCE_END = 2020
ANALYSIS_START = 2001
ANALYSIS_END = 2021
DRY_THRESHOLD = -1.5
WET_THRESHOLD = 1.5
EVENT_MIN = -5
EVENT_MAX = 8
PRE_PERIODS = (-3, -2, -1)
IMMEDIATE_PERIODS = (0, 1, 2)
CUMULATIVE_PERIODS = (0, 1, 2, 3, 4, 5)
RECOVERY_TOLERANCE_SD = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--panel-output", type=Path, default=DEFAULT_PANEL_OUTPUT)
    parser.add_argument("--stack-output", type=Path, default=DEFAULT_STACK_OUTPUT)
    parser.add_argument("--event-output", type=Path, default=DEFAULT_EVENT_OUTPUT)
    return parser.parse_args()


def project_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def nominal_windows() -> pd.DataFrame:
    rows = []
    for year in range(REFERENCE_START, ANALYSIS_END + 1):
        starts = [pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=16 * slot) for slot in range(23)]
        for slot_index, start in enumerate(starts, start=1):
            end = starts[slot_index] - pd.Timedelta(days=1) if slot_index < len(starts) else pd.Timestamp(year=year, month=12, day=31)
            rows.append({
                "Year": year,
                "MODIS Composite Slot": slot_index,
                "Composite Date": start,
                "Rainfall Window End": end,
                "Rainfall Window Days": int((end - start).days + 1),
            })
    return pd.DataFrame(rows)


def aggregate_rainfall(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily["Year"] = daily["Date"].dt.year
    daily["Day of Year"] = daily["Date"].dt.dayofyear
    daily["MODIS Composite Slot"] = np.minimum(((daily["Day of Year"] - 1) // 16 + 1), 23).astype(int)
    interval = (
        daily.groupby(["CHIRPS Cell ID", "Year", "MODIS Composite Slot"], observed=True)
        .agg(
            **{
                "Interval Rainfall mm": ("Daily Rainfall mm", "sum"),
                "Observed Rainfall Days": ("Daily Rainfall mm", "count"),
            }
        )
        .reset_index()
    )
    windows = nominal_windows()
    interval = interval.merge(windows, on=["Year", "MODIS Composite Slot"], how="left", validate="many_to_one")
    interval.loc[
        interval["Observed Rainfall Days"].ne(interval["Rainfall Window Days"]),
        "Interval Rainfall mm",
    ] = np.nan
    reference = interval["Year"].between(REFERENCE_START, REFERENCE_END)
    baseline = (
        interval.loc[reference]
        .groupby(["CHIRPS Cell ID", "MODIS Composite Slot"], observed=True)["Interval Rainfall mm"]
        .agg(["mean", "std", "count"])
        .rename(columns={
            "mean": "Rainfall Slot Mean 1991-2020 mm",
            "std": "Rainfall Slot SD 1991-2020 mm",
            "count": "Rainfall Slot Valid Years 1991-2020",
        })
        .reset_index()
    )
    interval = interval.merge(baseline, on=["CHIRPS Cell ID", "MODIS Composite Slot"], how="left", validate="many_to_one")
    interval["Interval-Aligned Rainfall Anomaly Z"] = (
        interval["Interval Rainfall mm"] - interval["Rainfall Slot Mean 1991-2020 mm"]
    ) / interval["Rainfall Slot SD 1991-2020 mm"]
    interval["Dry Rainfall Extreme"] = interval["Interval-Aligned Rainfall Anomaly Z"].le(DRY_THRESHOLD).astype("int8")
    interval["Wet Rainfall Extreme"] = interval["Interval-Aligned Rainfall Anomaly Z"].ge(WET_THRESHOLD).astype("int8")
    return interval


def freeze_events(interval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = interval.loc[interval["Year"].between(ANALYSIS_START, ANALYSIS_END)].copy()
    analysis = analysis.sort_values(["CHIRPS Cell ID", "Composite Date"]).reset_index(drop=True)
    analysis["Composite Sequence"] = analysis.groupby("CHIRPS Cell ID", observed=True).cumcount()
    analysis["Any Rainfall Extreme"] = analysis[["Dry Rainfall Extreme", "Wet Rainfall Extreme"]].max(axis=1)
    analysis["Retained Nonoverlapping Event"] = np.int8(0)
    events = []
    for cell_id, group in analysis.groupby("CHIRPS Cell ID", observed=True, sort=False):
        candidates = group.loc[group["Any Rainfall Extreme"].eq(1), "Composite Sequence"].to_numpy(int)
        maximum = int(group["Composite Sequence"].max())
        last_retained = -10_000
        for _, source in group.loc[group["Any Rainfall Extreme"].eq(1)].iterrows():
            position = int(source["Composite Sequence"])
            # The frozen rule requires three event-free pre-periods and no
            # overlapping post-event windows.  Earlier exploratory code was
            # stricter than the frozen text by excluding any other extreme in
            # the full -5...+8 window; this implementation follows the record.
            clean_preperiod = not bool(np.any((candidates >= position - 3) & (candidates <= position - 1)))
            nonoverlapping_post = position > last_retained + EVENT_MAX
            complete_window = position + EVENT_MIN >= 0 and position + EVENT_MAX <= maximum
            if not (clean_preperiod and nonoverlapping_post and complete_window):
                continue
            shock = "Dry" if int(source["Dry Rainfall Extreme"]) == 1 else "Wet"
            event_id = f"{cell_id}_{source['Composite Date']:%Y%m%d}_{shock.lower()}"
            events.append({
                "Event ID": event_id,
                "CHIRPS Cell ID": cell_id,
                "Event Date": source["Composite Date"],
                "Event Composite Sequence": position,
                "Shock Family": shock,
                "Event Rainfall Anomaly Z": float(source["Interval-Aligned Rainfall Anomaly Z"]),
            })
            mask = analysis["CHIRPS Cell ID"].eq(cell_id) & analysis["Composite Sequence"].eq(position)
            analysis.loc[mask, "Retained Nonoverlapping Event"] = np.int8(1)
            last_retained = position
    return analysis, pd.DataFrame(events)


def build_stack(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    relative = pd.DataFrame({"Event Time": range(EVENT_MIN, EVENT_MAX + 1)})
    targets = events.merge(relative, how="cross")
    targets["Composite Sequence"] = targets["Event Composite Sequence"] + targets["Event Time"]
    stack = targets.merge(
        panel,
        on=["CHIRPS Cell ID", "Composite Sequence"],
        how="left",
        validate="many_to_many",
    )
    stack = stack.sort_values(["Event ID", "Village Code", "Event Time"]).reset_index(drop=True)
    return stack


def event_metrics(group: pd.DataFrame, outcome: str, prefix: str) -> dict[str, float | int]:
    series = group.set_index("Event Time")[outcome]
    pre = series.reindex(PRE_PERIODS)
    immediate = series.reindex(IMMEDIATE_PERIODS)
    cumulative = series.reindex(CUMULATIVE_PERIODS)
    complete_pre = bool(pre.notna().all())
    complete_immediate = bool(immediate.notna().all())
    pre_mean = float(pre.mean()) if complete_pre else np.nan
    immediate_mean = float(immediate.mean()) if complete_immediate else np.nan
    minimum_period = int(immediate.idxmin()) if complete_immediate else -1
    recovery_time = np.nan
    recovery_event_period = np.nan
    recovered_by_eight = 0
    if complete_pre and complete_immediate:
        threshold = pre_mean - RECOVERY_TOLERANCE_SD
        for period in range(minimum_period, EVENT_MAX):
            pair = series.reindex([period, period + 1])
            if pair.notna().all() and bool((pair >= threshold).all()):
                recovery_time = float(period - minimum_period)
                recovery_event_period = float(period)
                recovered_by_eight = 1
                break
    cumulative_loss = (
        float(np.maximum(pre_mean - cumulative.to_numpy(float), 0.0).sum())
        if complete_pre and cumulative.notna().all() else np.nan
    )
    return {
        f"{prefix} Pre-Event Mean SD": pre_mean,
        f"{prefix} Immediate Vegetation Loss SD": pre_mean - immediate_mean if complete_pre and complete_immediate else np.nan,
        f"{prefix} Cumulative Negative Vegetation Loss SD": cumulative_loss,
        f"{prefix} Minimum Event Period": minimum_period if minimum_period >= 0 else np.nan,
        f"{prefix} Recovery Event Period": recovery_event_period,
        f"{prefix} Recovery Time Composites": recovery_time,
        f"{prefix} Recovered by Period 8": recovered_by_eight,
        f"{prefix} Recovery Right Censored": int(complete_pre and complete_immediate and not recovered_by_eight),
        f"{prefix} Restricted Recovery Time through Period 8": (
            recovery_time if recovered_by_eight else 9.0
        ) if complete_pre and complete_immediate else np.nan,
    }


def build_event_outcomes(stack: pd.DataFrame) -> pd.DataFrame:
    identifiers = [
        "Event ID", "Event Date", "Shock Family", "Event Rainfall Anomaly Z",
        "CHIRPS Cell ID", "Village Code", "Village Name", "Longitude", "Latitude",
        "Higher-Repression Southwest Zone", "Historical Repression Side",
        "Signed Distance to Historical Repression Boundary km",
        "Absolute Distance to Historical Repression Boundary km",
        "Historical Boundary Segment", "Linked Climate Commune Code",
        "Linked Climate Commune Name", "Cross-Side CHIRPS Cell",
    ]
    rows = []
    for _, group in stack.groupby(["Event ID", "Village Code"], observed=True, sort=False):
        first = group.iloc[0]
        row = {column: first[column] for column in identifiers}
        row.update(event_metrics(group, "High-Frequency EVI Anomaly Z", "EVI"))
        row.update(event_metrics(group, "High-Frequency NDVI Anomaly Z", "NDVI"))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    vegetation = pd.read_parquet(project_path(root, VEGETATION_PANEL))
    rainfall = pd.read_parquet(project_path(root, RAINFALL_PANEL))
    cell_map = pd.read_csv(project_path(root, CELL_MAP), dtype={"Village Code": "string", "CHIRPS Cell ID": "string"})
    cell_map = cell_map[["Village Code", "CHIRPS Cell ID"]].drop_duplicates("Village Code")
    vegetation["Village Code"] = vegetation["Village Code"].astype("string")
    vegetation = vegetation.merge(cell_map, on="Village Code", how="left", validate="many_to_one")
    if vegetation["CHIRPS Cell ID"].isna().any():
        raise ValueError("At least one vegetation village lacks a CHIRPS-cell assignment")

    interval_all = aggregate_rainfall(rainfall)
    interval, events = freeze_events(interval_all)
    vegetation["Composite Date"] = pd.to_datetime(vegetation["Composite Date"])
    panel = vegetation.merge(
        interval,
        on=["CHIRPS Cell ID", "Year", "MODIS Composite Slot", "Composite Date"],
        how="left",
        validate="many_to_one",
    )
    side_support = panel.groupby("CHIRPS Cell ID", observed=True)["Higher-Repression Southwest Zone"].nunique()
    panel["Cross-Side CHIRPS Cell"] = panel["CHIRPS Cell ID"].map(side_support.eq(2)).astype("int8")
    if panel["Interval-Aligned Rainfall Anomaly Z"].isna().all():
        raise ValueError("Rainfall did not align to the vegetation composites")

    stack = build_stack(panel.loc[panel["Cross-Side CHIRPS Cell"].eq(1)].copy(), events.loc[events["CHIRPS Cell ID"].isin(side_support[side_support.eq(2)].index)].copy())
    event_outcomes = build_event_outcomes(stack)

    panel_output = project_path(root, args.panel_output)
    stack_output = project_path(root, args.stack_output)
    event_output = project_path(root, args.event_output)
    for output in (panel_output, stack_output, event_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_output, index=False)
    stack.to_parquet(stack_output, index=False)
    event_outcomes.to_parquet(event_output, index=False)

    exp_dir = project_path(root, EXP_DIR)
    exp_dir.mkdir(parents=True, exist_ok=True)
    interval_all.to_csv(exp_dir / "interval_rainfall_1991_2021.csv", index=False)
    events.to_csv(exp_dir / "retained_nonoverlapping_events.csv", index=False)
    coverage = (
        panel.groupby("Year", observed=True)
        .agg(
            composites=("Composite Date", "nunique"),
            villages=("Village Code", "nunique"),
            rows=("Village Code", "size"),
            evi_available=("High-Frequency EVI Anomaly Z", lambda x: int(x.notna().sum())),
            rainfall_available=("Interval-Aligned Rainfall Anomaly Z", lambda x: int(x.notna().sum())),
            retained_event_village_rows=("Retained Nonoverlapping Event", "sum"),
        )
        .reset_index()
    )
    coverage["evi_coverage_share"] = coverage["evi_available"] / coverage["rows"]
    coverage["rainfall_coverage_share"] = coverage["rainfall_available"] / coverage["rows"]
    coverage.to_csv(exp_dir / "coverage_by_year.csv", index=False)
    event_support = (
        event_outcomes.groupby("Shock Family", observed=True)
        .agg(
            events=("Event ID", "nunique"),
            rainfall_cells=("CHIRPS Cell ID", "nunique"),
            villages=("Village Code", "nunique"),
            village_events=("Event ID", "size"),
            southwest_village_events=("Higher-Repression Southwest Zone", "sum"),
        )
        .reset_index()
    )
    event_support["west_village_events"] = event_support["village_events"] - event_support["southwest_village_events"]
    event_support.to_csv(exp_dir / "event_support.csv", index=False)
    print(f"Saved: {panel_output.relative_to(root)}")
    print(f"Saved: {stack_output.relative_to(root)}")
    print(f"Saved: {event_output.relative_to(root)}")
    print(coverage.to_string(index=False))
    print(event_support.to_string(index=False))


if __name__ == "__main__":
    main()
