#!/usr/bin/env python3
"""Run a 1,000-draw random-two-sector village compound-shock placebo.

The frozen border anchors come from the earlier outcome-independent random-pair
protocol, but all village assignments and all NPP estimates are rebuilt for the
current area-level ITT design. Each draw selects spatially compact placebo zones
of 25 and 63 villages, applies 2008 and 2011 pseudo-onset years, and estimates the
same full drought/heat/compound lower-order hierarchy. The Monte Carlo estimator
uses group-balanced weights, village and sector-year fixed effects, and four
predetermined covariate-specific linear trends. The actual placement is estimated
with exactly the same placebo specification.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/exp/.matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_predetermined_balance_and_common_support import build_sample  # noqa: E402


ANCHORS = (
    ROOT
    / "data/exp/experiments/cambodia-thailand-gate-c-npp/random-pair-monte-carlo"
    / "gate_c_npp_random_pair_accepted_assignments.csv"
)
PANEL = ROOT / "data/processed/cambodia_public_village_npp_conflict_panel_candidate_preprocessed.parquet"
OUT = ROOT / "data/exp/experiments/cambodia-thailand-village-area-itt-climate/random-pair-placebo"
ASSIGNMENTS = OUT / "random_pair_village_assignments.csv"
FREEZE_METADATA = OUT / "random_pair_village_assignment_metadata.json"
ESTIMATES = OUT / "random_pair_compound_estimates.csv"
INFERENCE = OUT / "random_pair_compound_inference.csv"
FIGURE = OUT / "random_pair_compound_distribution.png"
RUN_METADATA = OUT / "random_pair_compound_run_metadata.json"

ID = "National Village Point ID"
YEAR = "Year"
NPP = "Village Buffer Mean Annual Land NPP Anomaly Z 2001-2020"
DRY = "Village Buffer Mean May October Dry Rainfall Intensity"
HEAT = "Village Buffer Mean May October Heat Intensity"
COMPOUND = "Village Buffer Mean May October Compound Hot-Dry Intensity"
SECTOR = "Candidate Conflict Sector"

PRIMARY_RADIUS_KM = 5
SIMULATIONS = 1000
SEED = 20260823
WORKERS = 4
SECTOR_PLAN = {
    "Preah Vihear": {"villages": 25, "first_year": 2008},
    "Ta Moan-Ta Krabey": {"villages": 63, "first_year": 2011},
}
CONTROL_FEATURES = [
    "Pre-conflict NPP mean",
    "Pre-conflict NPP trend",
    "Baseline cropland share",
    "Log baseline population",
]
STRUCTURAL_TERMS = [
    "Drought",
    "Area x drought",
    "Post x drought",
    "Area x post x drought",
    "Heat",
    "Area x heat",
    "Post x heat",
    "Area x post x heat",
    "Compound",
    "Area x compound",
    "Post x compound",
    "Area x post x compound",
    "Area x post",
    *[f"{variable} x linear year" for variable in CONTROL_FEATURES],
]
TARGET_INDEX = STRUCTURAL_TERMS.index("Area x post x compound")

_CONTEXT: dict[str, object] | None = None
_ASSIGNMENT_MAP: dict[int, dict[str, list[str]]] | None = None


def haversine_km(lon1: float, lat1: float, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    lon1r = np.radians(lon1)
    lat1r = np.radians(lat1)
    lon2r = np.radians(lon2)
    lat2r = np.radians(lat2)
    term = (
        np.sin((lat2r - lat1r) / 2) ** 2
        + np.cos(lat1r) * np.cos(lat2r) * np.sin((lon2r - lon1r) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(term, 0, 1)))


def freeze_assignments() -> pd.DataFrame:
    """Freeze village zones without reading post-conflict NPP."""
    if ASSIGNMENTS.exists() and FREEZE_METADATA.exists():
        frozen = pd.read_csv(ASSIGNMENTS, dtype={ID: str})
        if frozen["Simulation ID"].nunique() != SIMULATIONS or len(frozen) != SIMULATIONS * 88:
            raise RuntimeError("Existing village placebo assignment file is inconsistent")
        print("Using existing frozen village placebo assignments", flush=True)
        return frozen

    OUT.mkdir(parents=True, exist_ok=True)
    anchors = pd.read_csv(ANCHORS)
    if len(anchors) != SIMULATIONS:
        raise RuntimeError(f"Expected {SIMULATIONS} frozen anchor pairs; found {len(anchors)}")
    _, controls = build_sample()
    controls = controls[[ID, "Point Longitude", "Point Latitude"]].copy()
    controls[ID] = controls[ID].astype(str)
    ids = controls[ID].to_numpy(str)
    longitude = controls["Point Longitude"].to_numpy(float)
    latitude = controls["Point Latitude"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for index, anchor in anchors.iterrows():
        simulation_id = int(anchor["Simulation ID"])
        anchor_pairs = [
            (float(anchor["First Longitude"]), float(anchor["First Latitude"]), "First"),
            (float(anchor["Second Longitude"]), float(anchor["Second Latitude"]), "Second"),
        ]
        if rng.integers(0, 2) == 1:
            anchor_pairs.reverse()
        selected: set[str] = set()
        for (sector, plan), (anchor_lon, anchor_lat, source_anchor) in zip(
            SECTOR_PLAN.items(), anchor_pairs, strict=True
        ):
            distance = haversine_km(anchor_lon, anchor_lat, longitude, latitude)
            order = np.argsort(distance)
            chosen_indices: list[int] = []
            for candidate_index in order:
                candidate_id = ids[candidate_index]
                if candidate_id in selected:
                    continue
                chosen_indices.append(int(candidate_index))
                selected.add(candidate_id)
                if len(chosen_indices) == int(plan["villages"]):
                    break
            if len(chosen_indices) != int(plan["villages"]):
                raise RuntimeError(f"Could not form placebo sector {sector} in draw {simulation_id}")
            for rank, candidate_index in enumerate(chosen_indices, start=1):
                rows.append(
                    {
                        "Simulation ID": simulation_id,
                        "Conflict Sector": sector,
                        "Pseudo First Conflict Year": int(plan["first_year"]),
                        ID: ids[candidate_index],
                        "Nearest-village Rank": rank,
                        "Anchor Source": source_anchor,
                        "Anchor Longitude": anchor_lon,
                        "Anchor Latitude": anchor_lat,
                        "Village Longitude": float(longitude[candidate_index]),
                        "Village Latitude": float(latitude[candidate_index]),
                        "Anchor-to-village Distance km": float(distance[candidate_index]),
                        "Original Anchor Separation km": float(anchor["Sector Separation km"]),
                    }
                )
        if (index + 1) % 100 == 0:
            print(f"Frozen village zones {index + 1}/{SIMULATIONS}", flush=True)

    frozen = pd.DataFrame(rows)
    frozen.to_csv(ASSIGNMENTS, index=False)
    metadata = {
        "status": "outcome-independent village placebo zones frozen",
        "human_decision_record": "MILI-D-20260823-009",
        "anchor_source": str(ANCHORS.relative_to(ROOT)),
        "simulations": SIMULATIONS,
        "seed_for_sector_size_anchor_swap": SEED,
        "sector_plan": SECTOR_PLAN,
        "assignment_rule": "nearest eligible original-control villages to each anchor; no village overlap within draw",
        "actual_conflict_villages_excluded_from_placebo_universe": True,
        "maximum_anchor_to_village_distance_km": float(frozen["Anchor-to-village Distance km"].max()),
        "p95_anchor_to_village_distance_km": float(frozen["Anchor-to-village Distance km"].quantile(0.95)),
        "post_conflict_NPP_loaded": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    FREEZE_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Frozen assignments saved: {ASSIGNMENTS.relative_to(ROOT)}", flush=True)
    return frozen


def load_context() -> dict[str, object]:
    treated, controls = build_sample()
    pool = pd.concat([treated, controls], ignore_index=True).copy()
    pool[ID] = pool[ID].astype(str)
    pool = pool.drop_duplicates(ID).reset_index(drop=True)
    panel = pd.read_parquet(
        PANEL,
        columns=[ID, YEAR, NPP, DRY, HEAT, COMPOUND],
        filters=[("Buffer Radius km", "=", PRIMARY_RADIUS_KM)],
    )
    panel[ID] = panel[ID].astype(str)
    panel = panel.loc[panel[ID].isin(set(pool[ID]))].copy()
    years = np.sort(panel[YEAR].unique().astype(int))
    if list(years) != list(range(2001, 2025)):
        raise RuntimeError("Placebo panel must cover every year from 2001 through 2024")
    ids = pool[ID].to_numpy(str)
    arrays: dict[str, np.ndarray] = {}
    for variable in [NPP, DRY, HEAT, COMPOUND]:
        pivot = panel.pivot(index=ID, columns=YEAR, values=variable).reindex(index=ids, columns=years)
        if pivot.isna().any().any():
            raise RuntimeError(f"Unbalanced placebo panel for {variable}")
        arrays[variable] = pivot.to_numpy(float)
    controls_matrix = pool[CONTROL_FEATURES].to_numpy(float)
    controls_matrix = (controls_matrix - controls_matrix.mean(axis=0)) / controls_matrix.std(axis=0, ddof=0)
    id_to_index = {village_id: index for index, village_id in enumerate(ids)}
    actual_sector_ids = {
        sector: treated.loc[treated[SECTOR].eq(sector), ID].astype(str).tolist()
        for sector in SECTOR_PLAN
    }
    return {
        "ids": ids,
        "id_to_index": id_to_index,
        "years": years,
        "outcome": arrays[NPP],
        "dry": arrays[DRY],
        "heat": arrays[HEAT],
        "compound": arrays[COMPOUND],
        "controls_matrix": controls_matrix,
        "actual_sector_ids": actual_sector_ids,
        "original_control_ids": controls[ID].astype(str).tolist(),
    }


def double_demean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Absorb village and year effects for a balanced panel with village weights."""
    total_weight = weights.sum()
    entity_mean = values.mean(axis=1, keepdims=True)
    time_mean = np.einsum("i,it...->t...", weights, values) / total_weight
    overall_mean = np.einsum("i,i...->...", weights, entity_mean[:, 0]) / total_weight
    return values - entity_mean - time_mean[None, ...] + overall_mean


def sector_crossproducts(
    context: dict[str, object],
    treated_ids: list[str],
    control_ids: list[str],
    first_year: int,
    sector_share: float,
) -> tuple[np.ndarray, np.ndarray]:
    id_to_index = context["id_to_index"]
    treated_index = np.array([id_to_index[value] for value in treated_ids], dtype=int)
    control_index = np.array([id_to_index[value] for value in control_ids], dtype=int)
    index = np.r_[treated_index, control_index]
    area = np.r_[np.ones(len(treated_index)), np.zeros(len(control_index))][:, None]
    weights = np.r_[
        np.full(len(treated_index), sector_share / (2 * len(treated_index))),
        np.full(len(control_index), sector_share / (2 * len(control_index))),
    ]
    years = np.asarray(context["years"], dtype=int)
    post = (years >= first_year).astype(float)[None, :]
    area_post = area * post
    dry = np.asarray(context["dry"])[index]
    heat = np.asarray(context["heat"])[index]
    compound = np.asarray(context["compound"])[index]
    linear_year = ((years - years.mean()) / years.std(ddof=0))[None, :]
    predetermined = np.asarray(context["controls_matrix"])[index]
    columns = [
        dry,
        area * dry,
        post * dry,
        area_post * dry,
        heat,
        area * heat,
        post * heat,
        area_post * heat,
        compound,
        area * compound,
        post * compound,
        area_post * compound,
        np.broadcast_to(area_post, dry.shape),
        *[predetermined[:, column][:, None] * linear_year for column in range(predetermined.shape[1])],
    ]
    design = np.stack(columns, axis=2)
    outcome = np.asarray(context["outcome"])[index]
    design = double_demean(design, weights)
    outcome = double_demean(outcome, weights)
    flat_design = design.reshape(-1, design.shape[2])
    flat_outcome = outcome.reshape(-1)
    repeated_weights = np.repeat(weights, design.shape[1])
    xtx = flat_design.T @ (flat_design * repeated_weights[:, None])
    xty = flat_design.T @ (flat_outcome * repeated_weights)
    return xtx, xty


def estimate_pair(
    context: dict[str, object],
    sector_ids: dict[str, list[str]],
    control_ids: list[str],
) -> tuple[float, int]:
    xtx = np.zeros((len(STRUCTURAL_TERMS), len(STRUCTURAL_TERMS)))
    xty = np.zeros(len(STRUCTURAL_TERMS))
    total_treated = sum(len(values) for values in sector_ids.values())
    for sector, plan in SECTOR_PLAN.items():
        sector_share = len(sector_ids[sector]) / total_treated
        sector_xtx, sector_xty = sector_crossproducts(
            context,
            sector_ids[sector],
            control_ids,
            int(plan["first_year"]),
            sector_share,
        )
        xtx += sector_xtx
        xty += sector_xty
    scale = np.sqrt(np.clip(np.diag(xtx), 1e-20, None))
    correlation = xtx / np.outer(scale, scale)
    standardized_rhs = xty / scale
    standardized_beta, _, rank, _ = np.linalg.lstsq(correlation, standardized_rhs, rcond=1e-10)
    beta = standardized_beta / scale
    return float(beta[TARGET_INDEX]), int(rank)


def initialize_worker() -> None:
    global _CONTEXT, _ASSIGNMENT_MAP
    _CONTEXT = load_context()
    assignments = pd.read_csv(ASSIGNMENTS, dtype={ID: str})
    _ASSIGNMENT_MAP = {}
    for simulation_id, frame in assignments.groupby("Simulation ID", sort=False):
        _ASSIGNMENT_MAP[int(simulation_id)] = {
            sector: frame.loc[frame["Conflict Sector"].eq(sector), ID].astype(str).tolist()
            for sector in SECTOR_PLAN
        }


def estimate_simulation(simulation_id: int) -> dict[str, object]:
    if _CONTEXT is None or _ASSIGNMENT_MAP is None:
        raise RuntimeError("Placebo worker is not initialized")
    sector_ids = _ASSIGNMENT_MAP[simulation_id]
    pseudo_treated = set(sector_ids["Preah Vihear"]).union(sector_ids["Ta Moan-Ta Krabey"])
    controls = [value for value in _CONTEXT["original_control_ids"] if value not in pseudo_treated]
    estimate, rank = estimate_pair(_CONTEXT, sector_ids, controls)
    return {
        "Simulation ID": simulation_id,
        "Compound Triple-Interaction Estimate": estimate,
        "Model Matrix Rank": rank,
        "Placebo Treated Villages": len(pseudo_treated),
        "Placebo Control Villages": len(controls),
    }


def estimate_actual(context: dict[str, object]) -> tuple[float, int]:
    return estimate_pair(
        context,
        context["actual_sector_ids"],
        context["original_control_ids"],
    )


def run_simulations() -> tuple[pd.DataFrame, float, int]:
    context = load_context()
    actual, actual_rank = estimate_actual(context)
    completed: dict[int, dict[str, object]] = {}
    if ESTIMATES.exists():
        checkpoint = pd.read_csv(ESTIMATES)
        completed = {
            int(row["Simulation ID"]): row.to_dict() for _, row in checkpoint.iterrows()
        }
        print(f"Resuming from {len(completed)} placebo estimates", flush=True)
    remaining = [simulation_id for simulation_id in range(1, SIMULATIONS + 1) if simulation_id not in completed]
    if remaining:
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=initialize_worker) as executor:
            futures = {executor.submit(estimate_simulation, simulation_id): simulation_id for simulation_id in remaining}
            for future in as_completed(futures):
                row = future.result()
                completed[int(row["Simulation ID"])] = row
                if len(completed) % 25 == 0:
                    pd.DataFrame(completed.values()).sort_values("Simulation ID").to_csv(ESTIMATES, index=False)
                    print(f"Estimated {len(completed)}/{SIMULATIONS} placebo pairs", flush=True)
    estimates = pd.DataFrame(completed.values()).sort_values("Simulation ID")
    if len(estimates) != SIMULATIONS:
        raise RuntimeError("The placebo run did not complete 1,000 estimates")
    assignments = pd.read_csv(ASSIGNMENTS)
    assignment_summary = assignments.groupby("Simulation ID", as_index=False).agg(
        **{
            "Maximum Anchor-to-village Distance km": ("Anchor-to-village Distance km", "max"),
            "Mean Anchor-to-village Distance km": ("Anchor-to-village Distance km", "mean"),
            "Original Anchor Separation km": ("Original Anchor Separation km", "first"),
        }
    )
    estimates = estimates.merge(assignment_summary, on="Simulation ID", validate="one_to_one")
    estimates.to_csv(ESTIMATES, index=False)
    return estimates, actual, actual_rank


def write_outputs(estimates: pd.DataFrame, actual: float, actual_rank: int) -> pd.DataFrame:
    simulated = estimates["Compound Triple-Interaction Estimate"]
    negative_count = int((simulated <= actual).sum())
    absolute_count = int((simulated.abs() >= abs(actual)).sum())
    one_sided = (negative_count + 1) / (SIMULATIONS + 1)
    two_sided_absolute = (absolute_count + 1) / (SIMULATIONS + 1)
    summary = pd.DataFrame(
        [
            {
                "Actual Compound Triple-Interaction Estimate": actual,
                "Actual Model Matrix Rank": actual_rank,
                "Accepted Random Two-Sector Draws": SIMULATIONS,
                "Placebo Estimates At Least As Negative As Actual": negative_count,
                "Add-one One-Sided Negative-Tail p-value": one_sided,
                "Placebo Estimates At Least As Large in Absolute Value": absolute_count,
                "Add-one Two-Sided Absolute p-value": two_sided_absolute,
                "Passes Approved One-Sided 5% Rule": bool(one_sided < 0.05),
                "Placebo Mean": float(simulated.mean()),
                "Placebo SD": float(simulated.std(ddof=1)),
                "Placebo P01": float(simulated.quantile(0.01)),
                "Placebo P05": float(simulated.quantile(0.05)),
                "Placebo Median": float(simulated.median()),
                "Placebo P95": float(simulated.quantile(0.95)),
                "Placebo P99": float(simulated.quantile(0.99)),
            }
        ]
    )
    summary.to_csv(INFERENCE, index=False)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.hist(simulated, bins=42, color="#7894A8", edgecolor="white", linewidth=0.5)
    ax.axvline(actual, color="#B23A48", linewidth=2.0, label="Actual conflict sectors")
    ax.axvline(0, color="#4A4A4A", linewidth=0.8)
    ax.axvline(simulated.median(), color="#4A4A4A", linestyle="--", linewidth=1.1, label="Placebo median")
    ax.set_xlabel("Compound hot–dry triple-interaction estimate (NPP SD)")
    ax.set_ylabel("Random two-sector assignments")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E3E7EA", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=180, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "completed 1,000-draw current-village compound spatial placebo",
        "human_decision_record": "MILI-D-20260823-009",
        "simulations": SIMULATIONS,
        "workers": WORKERS,
        "placebo_model": {
            "outcome": "Annual Land NPP Anomaly Z",
            "structural_terms": STRUCTURAL_TERMS,
            "fixed_effects": "village and sector-year",
            "weights": "within-sector equal treated/control total mass; actual and placebo use identical rule",
            "controls": "four standardized predetermined covariates interacted with linear year",
            "buffer_km": PRIMARY_RADIUS_KM,
        },
        "comparison_to_primary_model": (
            "The placebo specification preserves the full climate-interaction hierarchy but replaces "
            "full control-by-year indicators and sector calibration with a computationally tractable "
            "common specification applied identically to actual and all placebo placements."
        ),
        "p_value_rule": "add-one one-sided negative tail below 0.05; two-sided absolute p also reported",
        "post_result_validation": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    RUN_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nSpatial placebo inference")
    print(summary.to_string(index=False))
    print(f"\nSaved: {INFERENCE.relative_to(ROOT)}")
    print(f"Saved: {FIGURE.relative_to(ROOT)}")
    return summary


def main() -> None:
    freeze_assignments()
    estimates, actual, actual_rank = run_simulations()
    write_outputs(estimates, actual, actual_rank)


if __name__ == "__main__":
    main()
