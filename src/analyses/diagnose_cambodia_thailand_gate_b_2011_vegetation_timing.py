#!/usr/bin/env python3
"""Diagnose whether 2011 vegetation anomalies align with dated conflict or monsoon timing."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mj02-matplotlib")

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from diagnose_cambodia_thailand_common_support import PANEL
from freeze_cambodia_thailand_gate_b_protocol import OUT_DIR, load_features
from test_cambodia_thailand_gate_b_placebo_sites import make_weights
from test_cambodia_thailand_gate_b_sector_robustness import (
    distance_to_sites,
    event_sector_coordinates,
)


ROOT = Path(__file__).resolve().parents[2]
VEGETATION = ROOT / "data/processed/cambodia_national_modis_vegetation_16day"
CLIMATE = ROOT / "data/processed/cambodia_national_16day_climate_hazards_preprocessed.parquet"
OUTCOMES = ("Mean EVI", "Mean NDVI")
CLIMATE_CONTROLS = (
    "Composite Precipitation Total mm Anomaly Z",
    "Composite Maximum One-Day Precipitation mm Anomaly Z",
    "Composite Maximum Daily Temperature C Anomaly Z",
    "Composite Hot Day Count Anomaly Z",
)


def load_vegetation(ids: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect()
    con.register("selected_ids", ids[["National Grid Cell ID"]])
    baseline_glob = (VEGETATION / "year=200[1-7]/vegetation.parquet").as_posix()
    event_file = (VEGETATION / "year=2011/vegetation.parquet").as_posix()
    baseline = con.execute(
        f'''SELECT v."National Grid Cell ID", v."MODIS Composite Slot",
                   avg(v."Mean EVI") AS "Baseline Mean EVI",
                   avg(v."Mean NDVI") AS "Baseline Mean NDVI",
                   avg(v."EVI Valid Pixel Share") AS "Baseline EVI Valid Share",
                   avg(v."NDVI Valid Pixel Share") AS "Baseline NDVI Valid Share"
            FROM read_parquet('{baseline_glob}') v
            INNER JOIN selected_ids i USING ("National Grid Cell ID")
            GROUP BY v."National Grid Cell ID", v."MODIS Composite Slot"'''
    ).fetch_df()
    event = con.execute(
        f'''SELECT v."National Grid Cell ID", v."Composite Date", v."MODIS Composite Slot",
                   v."Mean EVI", v."Mean NDVI", v."EVI Valid Pixel Share", v."NDVI Valid Pixel Share"
            FROM read_parquet('{event_file}') v
            INNER JOIN selected_ids i USING ("National Grid Cell ID")'''
    ).fetch_df()
    con.close()
    return event.merge(
        baseline,
        on=["National Grid Cell ID", "MODIS Composite Slot"],
        validate="one_to_one",
    )


def climate_links(ids: pd.DataFrame) -> pd.DataFrame:
    links = pd.read_parquet(
        PANEL, columns=["National Grid Cell ID", "Year", "Climate Cell ID"]
    )
    links = links.loc[links["Year"].eq(2000), ["National Grid Cell ID", "Climate Cell ID"]]
    links = ids[["National Grid Cell ID"]].merge(links, on="National Grid Cell ID", validate="one_to_one")
    climate = pd.read_parquet(
        CLIMATE,
        columns=["Climate Cell ID", "Year", "MODIS Composite Slot", *CLIMATE_CONTROLS],
    ).loc[lambda frame: frame["Year"].eq(2011)]
    return links.merge(climate.drop(columns="Year"), on="Climate Cell ID", validate="many_to_many")


def scenario_frame(
    base: pd.DataFrame,
    sites: np.ndarray,
    vegetation: pd.DataFrame,
    climate: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    distance = distance_to_sites(base, sites)
    strict_control = base["Candidate All 2008-2011 Nearest Event Distance km"].gt(60).to_numpy()
    treated = distance <= 60
    keep = treated | strict_control
    sample = base.loc[keep].copy()
    distance = distance[keep]
    treated = distance <= 60
    weights, max_smd, block_ess = make_weights(sample, treated)
    sample["Analysis Weight"] = weights
    sample["Continuous Conflict Dose"] = np.maximum(0, 1 - distance / 60)
    frame = (
        vegetation.merge(
            sample[
                [
                    "National Grid Cell ID",
                    "Border Analysis Sector",
                    "Spatial Block ID",
                    "Analysis Weight",
                    "Continuous Conflict Dose",
                ]
            ],
            on="National Grid Cell ID",
            validate="many_to_one",
        )
        .merge(climate, on=["National Grid Cell ID", "MODIS Composite Slot"], validate="one_to_one")
    )
    return frame, {
        "Grid Cells": sample["National Grid Cell ID"].nunique(),
        "Treated Grid Cells": int(treated.sum()),
        "Maximum Absolute Weighted SMD": max_smd,
        "Treated 10 km Block ESS": block_ess,
    }


def estimate_slots(frame: pd.DataFrame, scenario: str, diagnostics: dict[str, float]) -> list[dict[str, object]]:
    rows = []
    sector_dummies = pd.get_dummies(frame["Border Analysis Sector"], drop_first=True, dtype=float)
    for outcome in OUTCOMES:
        baseline = f"Baseline {outcome}"
        valid_share = "EVI Valid Pixel Share" if outcome == "Mean EVI" else "NDVI Valid Pixel Share"
        frame[f"{outcome} Anomaly"] = frame[outcome] - frame[baseline]
        outcome_sd = float(frame[baseline].std(ddof=1))
        for slot, part in frame.groupby("MODIS Composite Slot", sort=True):
            columns = ["Continuous Conflict Dose", *CLIMATE_CONTROLS]
            x = pd.concat([part[columns], sector_dummies.loc[part.index]], axis=1)
            x = sm.add_constant(x, has_constant="add")
            complete = part[f"{outcome} Anomaly"].notna() & part[valid_share].ge(0.25)
            complete &= x.notna().all(axis=1)
            result = sm.WLS(
                part.loc[complete, f"{outcome} Anomaly"],
                x.loc[complete],
                weights=part.loc[complete, "Analysis Weight"],
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": part.loc[complete, "Spatial Block ID"]},
            )
            estimate = float(result.params["Continuous Conflict Dose"]) / outcome_sd
            standard_error = float(result.bse["Continuous Conflict Dose"]) / outcome_sd
            rows.append(
                {
                    "Scenario": scenario,
                    "Outcome": outcome.replace("Mean ", ""),
                    "Composite Slot": int(slot),
                    "Composite Date": part["Composite Date"].iloc[0],
                    "Standardized Dose Estimate": estimate,
                    "Standardized Standard Error": standard_error,
                    "Standardized Lower 95 CI": estimate - 1.96 * standard_error,
                    "Standardized Upper 95 CI": estimate + 1.96 * standard_error,
                    "Observations": int(complete.sum()),
                    **diagnostics,
                }
            )
    return rows


def plot_results(results: pd.DataFrame) -> None:
    scenarios = results["Scenario"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(13, 3.8 * len(scenarios)), sharex=True)
    if len(scenarios) == 1:
        axes = np.asarray([axes])
    for row_index, scenario in enumerate(scenarios):
        for column_index, outcome in enumerate(("EVI", "NDVI")):
            axis = axes[row_index, column_index]
            part = results.loc[
                results["Scenario"].eq(scenario) & results["Outcome"].eq(outcome)
            ].sort_values("Composite Slot")
            axis.errorbar(
                part["Composite Slot"],
                part["Standardized Dose Estimate"],
                yerr=1.96 * part["Standardized Standard Error"],
                fmt="o-",
                markersize=3,
                capsize=2,
                color="#2C6E9B",
            )
            axis.axhline(0, color="black", linewidth=0.8)
            if "Preah Vihear" in scenario:
                axis.axvspan(3, 4, color="#A64B3C", alpha=0.13)
            else:
                axis.axvspan(8, 9, color="#A64B3C", alpha=0.13)
            axis.axvspan(17, 19, color="#6A8EAE", alpha=0.10)
            axis.set_title(f"{scenario}: {outcome}")
            axis.set_ylabel("2011 anomaly dose coefficient (baseline SD)")
            axis.set_xlabel("MODIS 16-day composite slot")
            axis.grid(alpha=0.2)
    fig.suptitle("Vegetation timing around the 2011 Cambodia-Thailand conflict", y=0.998)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_b_2011_vegetation_timing.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    base = load_features()
    vegetation = load_vegetation(base[["National Grid Cell ID"]])
    climate = climate_links(base[["National Grid Cell ID"]])
    rows = []
    for name, sites in event_sector_coordinates():
        scenario = "Preah Vihear sector" if "Preah Vihear" in name else "Ta Moan/Ta Krabey sector"
        print(f"Estimating vegetation timing: {scenario}", flush=True)
        frame, diagnostics = scenario_frame(base, sites, vegetation, climate)
        rows.extend(estimate_slots(frame, scenario, diagnostics))
    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "gate_b_2011_vegetation_timing_coefficients.csv", index=False)
    plot_results(results)
    metadata = {
        "outcome": "2011 16-day EVI or NDVI minus the same cell-slot mean in 2001-2007",
        "conflict_windows": {
            "Preah Vihear": "slots 3-4, covering February 2 to March 5",
            "Ta Moan/Ta Krabey": "slots 8-9, covering April 23 to May 24",
        },
        "monsoon_flood_diagnostic_window": "slots 17-19, covering September 14 to October 31",
        "controls": list(CLIMATE_CONTROLS) + ["one-degree border-sector fixed effects"],
        "inference": "10 km spatial-block clustered covariance",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (OUT_DIR / "gate_b_2011_vegetation_timing_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    windows = results.loc[results["Composite Slot"].isin([3, 4, 8, 9, 17, 18, 19])]
    print(windows[["Scenario", "Outcome", "Composite Slot", "Standardized Dose Estimate", "Standardized Lower 95 CI", "Standardized Upper 95 CI"]].to_string(index=False))


if __name__ == "__main__":
    main()
