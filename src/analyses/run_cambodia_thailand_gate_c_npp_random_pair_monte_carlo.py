#!/usr/bin/env python3
"""Run 1,000 accepted random-two-sector Gate C NPP spatial simulations.

Protocol: MILI-D-20260822-017. Candidate assignments are screened and frozen
without loading NPP outcomes. Only after 1,000 assignments pass the fixed
balance and effective-block rules does outcome estimation begin.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mj02-matplotlib")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from diagnose_cambodia_thailand_common_support import FIXED_PREDICTORS, PRE_CLIMATE_PREDICTORS
from estimate_cambodia_thailand_gate_c_npp import (
    MAIN_PERIOD,
    NPP_NATURAL,
    POST,
    TARGET,
    fit_model,
)
from freeze_cambodia_thailand_gate_b_protocol import load_features


ROOT = Path(__file__).resolve().parents[2]
BORDER = ROOT / "data/processed/cambodia_thailand_shared_border_preprocessed.parquet"
SOURCE_PANEL = ROOT / "data/processed/cambodia_national_annual_satellite_climate_panel_preprocessed.parquet"
ACTUAL_SECTORS = ROOT / "data/exp/experiment-design/cambodia-thailand-gate-b/gate_b_actual_conflict_sectors.csv"
ACTUAL_RESULTS = ROOT / "data/exp/experiments/cambodia-thailand-gate-c-npp/gate_c_npp_main_model_tidy.csv"
OUT = ROOT / "data/exp/experiments/cambodia-thailand-gate-c-npp/random-pair-monte-carlo"
ASSIGNMENTS = OUT / "gate_c_npp_random_pair_accepted_assignments.csv"
BASE_CELLS = OUT / "gate_c_npp_random_pair_base_cells.parquet"
WEIGHTS = OUT / "gate_c_npp_random_pair_weights_float32.npy"
DOSES = OUT / "gate_c_npp_random_pair_doses_float32.npy"
FREEZE_METADATA = OUT / "gate_c_npp_random_pair_freeze_metadata.json"
ESTIMATES = OUT / "gate_c_npp_random_pair_estimates.csv"
INFERENCE = OUT / "gate_c_npp_random_pair_inference.csv"
WORKBOOK = OUT / "gate_c_npp_random_pair_summary.xlsx"
FIGURE = OUT / "gate_c_npp_random_pair_distribution.png"
RUN_METADATA = OUT / "gate_c_npp_random_pair_run_metadata.json"

CELL = "National Grid Cell ID"
YEAR = "Year"
CLIMATE = "Climate Cell ID"
BLOCK = "Spatial Block ID"
WEIGHT = "Gate B Binary Support Overlap Weight"
DOSE = "Continuous Conflict Dose"
DRY_RAIN = "May October Dry Rainfall Intensity"
PREDICTORS = FIXED_PREDICTORS + PRE_CLIMATE_PREDICTORS

SIMULATIONS = 1000
SEED = 2011
MIN_SEPARATION_KM = 30.0
ACTUAL_EXCLUSION_KM = 120.0
DOSE_RADIUS_KM = 60.0
BALANCE_THRESHOLD = 0.10
MIN_TREATED_BLOCK_ESS = 10.0
WORKERS = 4

_WORKER_PANEL: pd.DataFrame | None = None
_WORKER_WEIGHTS: np.ndarray | None = None
_WORKER_DOSES: np.ndarray | None = None
_ROWS_PER_CELL: int | None = None


def allowed_intervals(length_km: float, actual_positions: np.ndarray) -> list[tuple[float, float]]:
    blocked = sorted(
        (
            max(0.0, float(position) - ACTUAL_EXCLUSION_KM),
            min(length_km, float(position) + ACTUAL_EXCLUSION_KM),
        )
        for position in actual_positions
    )
    merged: list[list[float]] = []
    for start, end in blocked:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    allowed = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            allowed.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < length_km:
        allowed.append((cursor, length_km))
    return [(start, end) for start, end in allowed if end > start]


def sample_position(rng: np.random.Generator, intervals: list[tuple[float, float]]) -> float:
    lengths = np.array([end - start for start, end in intervals], dtype=float)
    draw = rng.uniform(0, lengths.sum())
    cumulative = 0.0
    for (start, end), length in zip(intervals, lengths, strict=True):
        if draw <= cumulative + length:
            return float(start + draw - cumulative)
        cumulative += length
    return float(intervals[-1][1])


def overlap_weights_and_support(
    x: np.ndarray,
    treated: np.ndarray,
    block_codes: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    if treated.sum() == 0 or (~treated).sum() == 0:
        raise ValueError("Candidate assignment has an empty exposure group")
    model = LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs").fit(
        x, treated.astype(int)
    )
    propensity = np.clip(model.predict_proba(x)[:, 1], 1e-8, 1 - 1e-8)
    weights = np.where(treated, 1 - propensity, propensity)
    weights[treated] *= (len(treated) / 2) / weights[treated].sum()
    weights[~treated] *= (len(treated) / 2) / weights[~treated].sum()
    treated_weights = weights[treated]
    control_weights = weights[~treated]
    treated_x = x[treated]
    control_x = x[~treated]
    treated_mean = np.average(treated_x, axis=0, weights=treated_weights)
    control_mean = np.average(control_x, axis=0, weights=control_weights)
    treated_variance = np.average(
        (treated_x - treated_mean) ** 2, axis=0, weights=treated_weights
    )
    control_variance = np.average(
        (control_x - control_mean) ** 2, axis=0, weights=control_weights
    )
    denominator = np.sqrt((treated_variance + control_variance) / 2)
    smd = np.divide(
        treated_mean - control_mean,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    block_weight = np.bincount(
        block_codes[treated], weights=treated_weights, minlength=int(block_codes.max()) + 1
    )
    block_weight = block_weight[block_weight > 0]
    block_ess = float(block_weight.sum() ** 2 / np.sum(block_weight**2))
    return weights, float(np.max(np.abs(smd))), block_ess


def freeze_assignments() -> pd.DataFrame:
    if all(path.exists() for path in [ASSIGNMENTS, BASE_CELLS, WEIGHTS, DOSES, FREEZE_METADATA]):
        frozen = pd.read_csv(ASSIGNMENTS)
        if len(frozen) != SIMULATIONS:
            raise RuntimeError("Existing frozen assignment count differs from protocol")
        print("Using existing outcome-independent frozen assignment set", flush=True)
        return frozen

    OUT.mkdir(parents=True, exist_ok=True)
    border = gpd.read_parquet(BORDER).geometry.iloc[0]
    length_km = float(border.length / 1000)
    actual = pd.read_csv(ACTUAL_SECTORS)["Projected Border Position km"].to_numpy(float)
    intervals = allowed_intervals(length_km, actual)
    if sum(end - start for start, end in intervals) <= MIN_SEPARATION_KM:
        raise RuntimeError("Eligible border domain is too short for two sectors")

    base = load_features()
    base = base.loc[
        base["Candidate All 2008-2011 Nearest Event Distance km"].gt(60)
    ].reset_index(drop=True)
    base[[CELL, BLOCK]].assign(**{"Placebo Base Index": np.arange(len(base))}).to_parquet(
        BASE_CELLS, index=False
    )
    x = StandardScaler().fit_transform(base[PREDICTORS])
    block_codes = pd.Categorical(base[BLOCK]).codes
    east = base["Grid Centre Easting m"].to_numpy(float)
    north = base["Grid Centre Northing m"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    to_wgs = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)

    weight_temp = WEIGHTS.with_suffix(".tmp.npy")
    dose_temp = DOSES.with_suffix(".tmp.npy")
    weight_matrix = np.lib.format.open_memmap(
        weight_temp, mode="w+", dtype="float32", shape=(SIMULATIONS, len(base))
    )
    dose_matrix = np.lib.format.open_memmap(
        dose_temp, mode="w+", dtype="float32", shape=(SIMULATIONS, len(base))
    )
    rows = []
    attempts = 0
    while len(rows) < SIMULATIONS:
        attempts += 1
        first = sample_position(rng, intervals)
        second = sample_position(rng, intervals)
        if abs(first - second) < MIN_SEPARATION_KM:
            continue
        if second < first:
            first, second = second, first
        anchor1 = border.interpolate(first * 1000)
        anchor2 = border.interpolate(second * 1000)
        distance = np.minimum(
            np.hypot(east - anchor1.x, north - anchor1.y),
            np.hypot(east - anchor2.x, north - anchor2.y),
        ) / 1000
        treated = distance <= DOSE_RADIUS_KM
        try:
            weights, max_smd, block_ess = overlap_weights_and_support(
                x, treated, block_codes
            )
        except ValueError:
            continue
        if max_smd > BALANCE_THRESHOLD or block_ess < MIN_TREATED_BLOCK_ESS:
            continue
        dose = np.maximum(0.0, 1 - distance / DOSE_RADIUS_KM)
        accepted = len(rows)
        weight_matrix[accepted] = weights.astype("float32")
        dose_matrix[accepted] = dose.astype("float32")
        lon1, lat1 = to_wgs.transform(anchor1.x, anchor1.y)
        lon2, lat2 = to_wgs.transform(anchor2.x, anchor2.y)
        rows.append(
            {
                "Simulation ID": accepted + 1,
                "Attempt ID": attempts,
                "First Border Position km": first,
                "Second Border Position km": second,
                "Sector Separation km": second - first,
                "First Longitude": lon1,
                "First Latitude": lat1,
                "Second Longitude": lon2,
                "Second Latitude": lat2,
                "Maximum Absolute Weighted SMD": max_smd,
                "Treated 10 km Block ESS": block_ess,
                "Treated Grid Cells": int(treated.sum()),
                "Treated Spatial Blocks": int(np.unique(block_codes[treated]).size),
                "Mean Conflict Dose All Cells": float(dose.mean()),
                "Mean Conflict Dose Exposed Cells": float(dose[treated].mean()),
            }
        )
        if len(rows) % 50 == 0:
            weight_matrix.flush()
            dose_matrix.flush()
            print(
                f"Frozen {len(rows)}/{SIMULATIONS} accepted assignments "
                f"after {attempts} attempts",
                flush=True,
            )
    weight_matrix.flush()
    dose_matrix.flush()
    del weight_matrix, dose_matrix
    weight_temp.replace(WEIGHTS)
    dose_temp.replace(DOSES)
    frozen = pd.DataFrame(rows)
    frozen.to_csv(ASSIGNMENTS, index=False)
    metadata = {
        "status": "outcome-independent frozen random-pair assignments",
        "human_approval_record": "MILI-D-20260822-017",
        "simulations": SIMULATIONS,
        "seed": SEED,
        "border_length_km": length_km,
        "eligible_border_intervals_km": intervals,
        "actual_sector_positions_km": actual.tolist(),
        "actual_exclusion_km": ACTUAL_EXCLUSION_KM,
        "minimum_pair_separation_km": MIN_SEPARATION_KM,
        "maximum_pair_separation_km": None,
        "dose_radius_km": DOSE_RADIUS_KM,
        "support_thresholds": {
            "maximum_absolute_weighted_SMD": BALANCE_THRESHOLD,
            "minimum_treated_10km_block_ESS": MIN_TREATED_BLOCK_ESS,
        },
        "attempts_required": attempts,
        "base_grid_cells": len(base),
        "NPP_outcomes_loaded": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    FREEZE_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Frozen assignments saved before outcome loading: {ASSIGNMENTS.relative_to(ROOT)}")
    return frozen


def initialize_worker() -> None:
    global _WORKER_PANEL, _WORKER_WEIGHTS, _WORKER_DOSES, _ROWS_PER_CELL
    base_cells = pd.read_parquet(BASE_CELLS)
    panel = pd.read_parquet(
        SOURCE_PANEL,
        columns=[CELL, YEAR, CLIMATE, NPP_NATURAL, DRY_RAIN],
    )
    panel = panel.loc[
        panel[YEAR].between(2001, 2007) | panel[YEAR].between(2012, 2024)
    ].dropna(subset=[NPP_NATURAL, DRY_RAIN])
    panel = panel.merge(base_cells, on=CELL, validate="many_to_one")
    panel = panel.sort_values(["Placebo Base Index", YEAR]).reset_index(drop=True)
    counts = panel.groupby("Placebo Base Index")[YEAR].nunique()
    if not counts.eq(20).all() or len(counts) != len(base_cells):
        raise RuntimeError("Monte Carlo worker panel is not balanced at 20 years per base cell")
    panel[POST] = panel[YEAR].between(2012, 2024)
    panel[MAIN_PERIOD] = True
    _WORKER_PANEL = panel
    _WORKER_WEIGHTS = np.load(WEIGHTS, mmap_mode="r")
    _WORKER_DOSES = np.load(DOSES, mmap_mode="r")
    _ROWS_PER_CELL = 20


def estimate_simulation(simulation_index: int) -> dict[str, object]:
    if _WORKER_PANEL is None or _WORKER_WEIGHTS is None or _WORKER_DOSES is None:
        raise RuntimeError("Worker is not initialized")
    frame = _WORKER_PANEL.copy()
    frame[WEIGHT] = np.repeat(_WORKER_WEIGHTS[simulation_index], _ROWS_PER_CELL)
    frame[DOSE] = np.repeat(_WORKER_DOSES[simulation_index], _ROWS_PER_CELL)
    result, _ = fit_model(
        frame,
        NPP_NATURAL,
        DRY_RAIN,
        "Random-pair Monte Carlo: natural-unit NPP x dry-rainfall intensity",
    )
    target = result.loc[result["Term"].eq(TARGET)].iloc[0]
    return {
        "Simulation ID": simulation_index + 1,
        "Target Estimate kg C per m2": float(target["Estimate"]),
        "Clustered Standard Error": float(target["Clustered Standard Error"]),
        "Model-based p-value": float(target["p-value"]),
        "Observations": int(target["Observations"]),
        "Spatial Blocks": int(target["Spatial Blocks"]),
    }


def estimate_all(assignments: pd.DataFrame) -> pd.DataFrame:
    completed: dict[int, dict[str, object]] = {}
    model_columns = [
        "Simulation ID",
        "Target Estimate kg C per m2",
        "Clustered Standard Error",
        "Model-based p-value",
        "Observations",
        "Spatial Blocks",
    ]
    if ESTIMATES.exists():
        checkpoint = pd.read_csv(ESTIMATES)
        checkpoint = checkpoint[model_columns]
        completed = {
            int(row["Simulation ID"]): row.to_dict() for _, row in checkpoint.iterrows()
        }
        print(f"Resuming from {len(completed)} completed simulations", flush=True)
    remaining = [index for index in range(SIMULATIONS) if index + 1 not in completed]
    if remaining:
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=initialize_worker) as executor:
            futures = {
                executor.submit(estimate_simulation, index): index for index in remaining
            }
            for future in as_completed(futures):
                record = future.result()
                completed[int(record["Simulation ID"])] = record
                if len(completed) % 20 == 0:
                    checkpoint = pd.DataFrame(completed.values()).sort_values("Simulation ID")
                    checkpoint.to_csv(ESTIMATES, index=False)
                    print(f"Estimated {len(completed)}/{SIMULATIONS} simulations", flush=True)
    estimates = pd.DataFrame(completed.values())[model_columns].sort_values("Simulation ID")
    if len(estimates) != SIMULATIONS:
        raise RuntimeError("Monte Carlo estimation did not complete all frozen assignments")
    estimates = assignments.merge(estimates, on="Simulation ID", validate="one_to_one")
    estimates.to_csv(ESTIMATES, index=False)
    return estimates


def write_outputs(estimates: pd.DataFrame) -> pd.DataFrame:
    actual_results = pd.read_csv(ACTUAL_RESULTS)
    actual = float(
        actual_results.loc[
            actual_results["Specification"].eq(
                "Primary: natural-unit NPP x dry-rainfall intensity"
            )
            & actual_results["Term"].eq(TARGET),
            "Estimate",
        ].iloc[0]
    )
    simulated = estimates["Target Estimate kg C per m2"]
    negative_count = int((simulated <= actual).sum())
    positive_count = int((simulated >= actual).sum())
    absolute_count = int((simulated.abs() >= abs(actual)).sum())
    one_sided_p = (1 + negative_count) / (1 + SIMULATIONS)
    upper_tail_p = (1 + positive_count) / (1 + SIMULATIONS)
    two_sided_equal_tail_p = min(1.0, 2 * min(one_sided_p, upper_tail_p))
    zero_centered_absolute_p = (1 + absolute_count) / (1 + SIMULATIONS)
    summary = pd.DataFrame(
        [
            {
                "Actual Target Estimate kg C per m2": actual,
                "Accepted Monte Carlo Simulations": SIMULATIONS,
                "Simulated Estimates At Least As Negative As Actual": negative_count,
                "One-Sided Negative-Tail Monte Carlo p-value": one_sided_p,
                "One-Sided Monte Carlo Standard Error": np.sqrt(
                    one_sided_p * (1 - one_sided_p) / SIMULATIONS
                ),
                "Upper-Tail Monte Carlo p-value": upper_tail_p,
                "Two-Sided Equal-Tail Monte Carlo p-value": two_sided_equal_tail_p,
                "Simulated Estimates At Least As Large in Absolute Value": absolute_count,
                "Zero-Centered Absolute-Value Diagnostic p-value": zero_centered_absolute_p,
                "Passes Approved One-Sided 5% Rule": bool(one_sided_p < 0.05),
                "Simulated Mean": float(simulated.mean()),
                "Simulated SD": float(simulated.std(ddof=1)),
                "Simulated P01": float(simulated.quantile(0.01)),
                "Simulated P05": float(simulated.quantile(0.05)),
                "Simulated Median": float(simulated.median()),
                "Simulated P95": float(simulated.quantile(0.95)),
                "Simulated P99": float(simulated.quantile(0.99)),
                "Correlation with Sector Separation km": float(
                    estimates["Target Estimate kg C per m2"].corr(
                        estimates["Sector Separation km"]
                    )
                ),
                "Correlation with Treated Grid Cells": float(
                    estimates["Target Estimate kg C per m2"].corr(
                        estimates["Treated Grid Cells"]
                    )
                ),
                "Correlation with Mean Exposed Conflict Dose": float(
                    estimates["Target Estimate kg C per m2"].corr(
                        estimates["Mean Conflict Dose Exposed Cells"]
                    )
                ),
            }
        ]
    )
    summary.to_csv(INFERENCE, index=False)
    with pd.ExcelWriter(WORKBOOK, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Inference Summary")
        estimates.to_excel(writer, index=False, sheet_name="Simulation Estimates")
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            sheet.sheet_view.showGridLines = False
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for column_cells in sheet.columns:
                width = min(34, max(12, max(len(str(cell.value or "")) for cell in column_cells) + 2))
                sheet.column_dimensions[column_cells[0].column_letter].width = width

    fig, axis = plt.subplots(figsize=(7.4, 4.5))
    axis.hist(simulated, bins=40, color="#7894A8", edgecolor="white", linewidth=0.5)
    axis.axvline(actual, color="#B23A48", linewidth=1.8, label="Actual conflict placement")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axvline(
        simulated.median(),
        color="#495057",
        linewidth=1.0,
        linestyle="--",
        label="Monte Carlo median",
    )
    axis.set_xlabel("Conflict dose x post x drought estimate (kg C per m2)")
    axis.set_ylabel("Monte Carlo assignments")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "status": "completed 1,000-draw random-pair spatial Monte Carlo",
        "human_approval_record": "MILI-D-20260822-017",
        "workers": WORKERS,
        "p_value": "add-one finite Monte Carlo correction",
        "primary_rule": "one-sided negative-tail p-value below 0.05",
        "interpretation_limit": (
            "Passing this rule validates spatial unusualness under the declared placebo "
            "universe; it does not make actual conflict placement random or independently "
            "replicated."
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    RUN_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\nMonte Carlo inference")
    print(summary.to_string(index=False))
    print(f"\nSaved {INFERENCE.relative_to(ROOT)}")
    print(f"Saved {WORKBOOK.relative_to(ROOT)}")
    print(f"Saved {FIGURE.relative_to(ROOT)}")
    return summary


def main() -> None:
    assignments = freeze_assignments()
    estimates = estimate_all(assignments)
    write_outputs(estimates)


if __name__ == "__main__":
    main()
