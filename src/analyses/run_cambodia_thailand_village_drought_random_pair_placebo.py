#!/usr/bin/env python3
"""Run the frozen 1,000-assignment drought random-two-sector placebo.

Plan: Test whether the positive 5 km affected-area drought-slope change is
unusually concentrated at the documented conflict geography.
Framework: AnaSOP Sections 5-7 frozen spatial-placebo protocol using compact
25- and 63-village pseudo sectors, common group-balanced estimation, the full
lower-order drought hierarchy, stack fixed effects, and a one-sided positive-
tail add-one probability.
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
from run_cambodia_thailand_village_compound_random_pair_placebo import (  # noqa: E402
    COMPOUND,
    CONTROL_FEATURES,
    DRY,
    ID,
    NPP,
    PANEL,
    YEAR,
    double_demean,
    haversine_km,
    load_context,
)
from table_predetermined_balance_and_common_support import build_sample  # noqa: E402


OUT = (
    ROOT
    / "data/exp/experiments/cambodia-thailand-village-area-itt-climate"
    / "drought-random-pair-placebo"
)
ASSIGNMENTS = OUT / "drought_random_pair_village_assignments.csv"
FREEZE_METADATA = OUT / "drought_random_pair_assignment_metadata.json"
ESTIMATES = OUT / "drought_random_pair_estimates.csv"
INFERENCE = OUT / "drought_random_pair_inference.csv"
FIGURE = OUT / "drought_random_pair_distribution.png"
RUN_METADATA = OUT / "drought_random_pair_run_metadata.json"

SIMULATIONS = 1000
SEED = 20260823
WORKERS = 4
PRIMARY_RADIUS_KM = 5
SECTOR_PLAN = {
    "Preah Vihear": {"villages": 25, "first_year": 2008},
    "Ta Moan-Ta Krabey": {"villages": 63, "first_year": 2011},
}
TERMS = [
    "Drought",
    "Area x drought",
    "Post x drought",
    "Area x post x drought",
    "Area x post",
    *[f"{variable} x linear year" for variable in CONTROL_FEATURES],
]
TARGET_INDEX = TERMS.index("Area x post x drought")

_CONTEXT: dict[str, object] | None = None
_ASSIGNMENT_MAP: dict[int, dict[str, list[str]]] | None = None


def freeze_unique_assignments() -> pd.DataFrame:
    """Freeze unique compact pseudo sectors without opening post-conflict NPP."""
    if ASSIGNMENTS.exists() and FREEZE_METADATA.exists():
        frozen = pd.read_csv(ASSIGNMENTS, dtype={ID: str})
        signatures = assignment_signatures(frozen)
        if (
            frozen["Simulation ID"].nunique() != SIMULATIONS
            or len(frozen) != SIMULATIONS * 88
            or len(signatures) != SIMULATIONS
        ):
            raise RuntimeError("Existing drought placebo assignments are inconsistent")
        print("Using existing frozen unique drought-placebo assignments", flush=True)
        return frozen

    OUT.mkdir(parents=True, exist_ok=True)
    _, controls = build_sample()
    controls = controls[[ID, "Point Longitude", "Point Latitude"]].drop_duplicates(ID).copy()
    controls[ID] = controls[ID].astype(str)
    ids = controls[ID].to_numpy(str)
    longitude = controls["Point Longitude"].to_numpy(float)
    latitude = controls["Point Latitude"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    accepted = 0
    attempts = 0

    while accepted < SIMULATIONS:
        attempts += 1
        anchor_indices = rng.choice(len(ids), size=2, replace=False)
        anchor_pairs = [
            (int(index), float(longitude[index]), float(latitude[index]))
            for index in anchor_indices
        ]
        if int(rng.integers(0, 2)) == 1:
            anchor_pairs.reverse()
        selected: set[str] = set()
        draw_rows: list[dict[str, object]] = []
        sector_ids: dict[str, list[str]] = {}
        anchor_separation = float(
            haversine_km(
                anchor_pairs[0][1],
                anchor_pairs[0][2],
                np.array([anchor_pairs[1][1]]),
                np.array([anchor_pairs[1][2]]),
            )[0]
        )

        for (sector, plan), (anchor_index, anchor_lon, anchor_lat) in zip(
            SECTOR_PLAN.items(), anchor_pairs, strict=True
        ):
            distance = haversine_km(anchor_lon, anchor_lat, longitude, latitude)
            chosen_indices: list[int] = []
            for candidate_index in np.argsort(distance):
                candidate_id = ids[int(candidate_index)]
                if candidate_id in selected:
                    continue
                chosen_indices.append(int(candidate_index))
                selected.add(candidate_id)
                if len(chosen_indices) == int(plan["villages"]):
                    break
            if len(chosen_indices) != int(plan["villages"]):
                raise RuntimeError(f"Could not form pseudo sector {sector}")
            chosen_ids = [ids[index] for index in chosen_indices]
            sector_ids[sector] = chosen_ids
            for rank, candidate_index in enumerate(chosen_indices, start=1):
                draw_rows.append(
                    {
                        "Simulation ID": accepted + 1,
                        "Conflict Sector": sector,
                        "Pseudo First Conflict Year": int(plan["first_year"]),
                        ID: ids[candidate_index],
                        "Nearest-village Rank": rank,
                        "Anchor Village ID": ids[anchor_index],
                        "Anchor Longitude": anchor_lon,
                        "Anchor Latitude": anchor_lat,
                        "Village Longitude": float(longitude[candidate_index]),
                        "Village Latitude": float(latitude[candidate_index]),
                        "Anchor-to-village Distance km": float(distance[candidate_index]),
                        "Anchor Separation km": anchor_separation,
                    }
                )

        signature = tuple(
            tuple(sorted(sector_ids[sector])) for sector in SECTOR_PLAN
        )
        if signature in seen:
            continue
        seen.add(signature)
        accepted += 1
        rows.extend(draw_rows)
        if accepted % 100 == 0:
            print(f"Frozen unique assignments {accepted}/{SIMULATIONS}", flush=True)

    frozen = pd.DataFrame(rows)
    frozen.to_csv(ASSIGNMENTS, index=False)
    metadata = {
        "status": "outcome-independent unique drought-placebo assignments frozen",
        "human_decision_record": "MILI-D-20260823-013",
        "simulations": SIMULATIONS,
        "unique_assignments": len(seen),
        "random_seed": SEED,
        "generation_attempts": attempts,
        "eligible_anchor_and_village_universe": int(len(controls)),
        "sector_plan": SECTOR_PLAN,
        "assignment_rule": (
            "sample two distinct eligible comparison-village anchors; randomly map the "
            "25/2008 and 63/2011 sector plans to anchors; take nearest non-overlapping "
            "eligible villages; reject duplicate sector-specific village assignments"
        ),
        "actual_conflict_villages_excluded_from_pseudo_treatment": True,
        "actual_sector_separation_preserved": False,
        "only_nonoverlap_requirement": "pseudo sectors cannot share village IDs within a draw",
        "maximum_anchor_to_village_distance_km": float(
            frozen["Anchor-to-village Distance km"].max()
        ),
        "p95_anchor_to_village_distance_km": float(
            frozen["Anchor-to-village Distance km"].quantile(0.95)
        ),
        "post_conflict_NPP_loaded_during_freeze": False,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    FREEZE_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Frozen assignments saved: {ASSIGNMENTS.relative_to(ROOT)}", flush=True)
    return frozen


def assignment_signatures(
    assignments: pd.DataFrame,
) -> set[tuple[tuple[str, ...], tuple[str, ...]]]:
    signatures: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for _, frame in assignments.groupby("Simulation ID", sort=False):
        signatures.add(
            tuple(
                tuple(sorted(frame.loc[frame["Conflict Sector"].eq(sector), ID].astype(str)))
                for sector in SECTOR_PLAN
            )
        )
    return signatures


def drought_sector_crossproducts(
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
    drought = np.asarray(context["dry"])[index]
    linear_year = ((years - years.mean()) / years.std(ddof=0))[None, :]
    predetermined = np.asarray(context["controls_matrix"])[index]
    columns = [
        drought,
        area * drought,
        post * drought,
        area_post * drought,
        np.broadcast_to(area_post, drought.shape),
        *[
            predetermined[:, column][:, None] * linear_year
            for column in range(predetermined.shape[1])
        ],
    ]
    design = double_demean(np.stack(columns, axis=2), weights)
    outcome = double_demean(np.asarray(context["outcome"])[index], weights)
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
    xtx = np.zeros((len(TERMS), len(TERMS)))
    xty = np.zeros(len(TERMS))
    total_treated = sum(len(values) for values in sector_ids.values())
    for sector, plan in SECTOR_PLAN.items():
        sector_share = len(sector_ids[sector]) / total_treated
        sector_xtx, sector_xty = drought_sector_crossproducts(
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
    standardized_beta, _, rank, _ = np.linalg.lstsq(
        correlation,
        standardized_rhs,
        rcond=1e-10,
    )
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
        raise RuntimeError("Drought-placebo worker is not initialized")
    sector_ids = _ASSIGNMENT_MAP[simulation_id]
    pseudo_treated = set(sector_ids["Preah Vihear"]).union(
        sector_ids["Ta Moan-Ta Krabey"]
    )
    controls = [
        value for value in _CONTEXT["original_control_ids"] if value not in pseudo_treated
    ]
    estimate, rank = estimate_pair(_CONTEXT, sector_ids, controls)
    return {
        "Simulation ID": simulation_id,
        "Drought Triple-Interaction Estimate": estimate,
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
    actual_estimate, actual_rank = estimate_actual(context)
    completed: dict[int, dict[str, object]] = {}
    if ESTIMATES.exists():
        prior = pd.read_csv(ESTIMATES)
        for row in prior.to_dict("records"):
            completed[int(row["Simulation ID"])] = row
        if completed:
            print(f"Resuming from {len(completed)} drought-placebo estimates", flush=True)
    remaining = [value for value in range(1, SIMULATIONS + 1) if value not in completed]
    if remaining:
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=initialize_worker) as executor:
            futures = {executor.submit(estimate_simulation, value): value for value in remaining}
            for future in as_completed(futures):
                row = future.result()
                completed[int(row["Simulation ID"])] = row
                if len(completed) % 50 == 0:
                    pd.DataFrame(completed.values()).sort_values("Simulation ID").to_csv(
                        ESTIMATES,
                        index=False,
                    )
                if len(completed) % 100 == 0:
                    print(f"Estimated {len(completed)}/{SIMULATIONS} placebo pairs", flush=True)
    estimates = pd.DataFrame(completed.values()).sort_values("Simulation ID").reset_index(drop=True)
    estimates.to_csv(ESTIMATES, index=False)
    if len(estimates) != SIMULATIONS:
        raise RuntimeError("Drought placebo run did not complete 1,000 estimates")
    if estimates["Model Matrix Rank"].ne(len(TERMS)).any():
        raise RuntimeError("At least one drought placebo model is rank deficient")
    if not np.isfinite(estimates["Drought Triple-Interaction Estimate"]).all():
        raise RuntimeError("At least one drought placebo estimate is non-finite")
    return estimates, actual_estimate, actual_rank


def summarize_and_plot(
    estimates: pd.DataFrame,
    actual_estimate: float,
    actual_rank: int,
) -> pd.DataFrame:
    placebo = estimates["Drought Triple-Interaction Estimate"].astype(float)
    positive_count = int(placebo.ge(actual_estimate).sum())
    absolute_count = int(placebo.abs().ge(abs(actual_estimate)).sum())
    positive_p = (positive_count + 1) / (SIMULATIONS + 1)
    absolute_p = (absolute_count + 1) / (SIMULATIONS + 1)
    passed = bool(actual_estimate > 0 and positive_p < 0.05)
    inference = pd.DataFrame(
        [
            {
                "Actual Common-Specification Drought Estimate": actual_estimate,
                "Actual Model Matrix Rank": actual_rank,
                "Placebo Assignments": SIMULATIONS,
                "Unique Placebo Assignments": SIMULATIONS,
                "Placebo Estimates At Least Actual": positive_count,
                "Add-One One-Sided Positive-Tail p-value": positive_p,
                "Actual Positive-Tail Rank Including Actual": positive_count + 1,
                "Placebo Absolute Estimates At Least Actual": absolute_count,
                "Add-One Two-Sided Absolute p-value": absolute_p,
                "Placebo Mean": float(placebo.mean()),
                "Placebo Median": float(placebo.median()),
                "Placebo SD": float(placebo.std(ddof=1)),
                "Placebo 2.5 Percentile": float(placebo.quantile(0.025)),
                "Placebo 97.5 Percentile": float(placebo.quantile(0.975)),
                "Approved Pass Rule": "actual estimate > 0 and one-sided positive-tail p < 0.05",
                "Passes Approved Spatial-Specificity Rule": passed,
            }
        ]
    )
    inference.to_csv(INFERENCE, index=False)

    fig, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    axis.hist(placebo, bins=35, color="#A7B0B5", edgecolor="white", linewidth=0.6)
    axis.axvline(
        actual_estimate,
        color="#C4493D",
        linewidth=2.2,
        label=f"Actual conflict geography: {actual_estimate:.3f}",
    )
    axis.text(
        0.98,
        0.94,
        f"Positive-tail rank: {positive_count + 1}/{SIMULATIONS + 1}\n"
        f"Add-one p = {positive_p:.3f}\n"
        f"Approved rule: {'pass' if passed else 'fail'}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    axis.set_xlabel("Drought slope-change coefficient under random two-sector assignment")
    axis.set_ylabel("Number of assignments")
    axis.legend(frameon=False, loc="upper left", fontsize=8)
    axis.grid(True, axis="y", color="#E3E7EA", linewidth=0.6)
    axis.grid(False, axis="x")
    fig.savefig(FIGURE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return inference


def main() -> None:
    frozen = freeze_unique_assignments()
    estimates, actual_estimate, actual_rank = run_simulations()
    inference = summarize_and_plot(estimates, actual_estimate, actual_rank)
    freeze_metadata = json.loads(FREEZE_METADATA.read_text(encoding="utf-8"))
    metadata = {
        "status": "completed 1,000-unique-assignment drought spatial placebo",
        "human_decision_record": "MILI-D-20260823-013",
        "primary_buffer_km": PRIMARY_RADIUS_KM,
        "outcome": NPP,
        "hazard": DRY,
        "unused_extension_field": COMPOUND,
        "actual_and_placebo_model": {
            "terms": TERMS,
            "fixed_effects": "village and sector-year absorbed by weighted double demeaning",
            "weights": "within-sector equal total treated and comparison mass",
            "predetermined_adjustment": "four baseline variables interacted with linear year",
            "timing": SECTOR_PLAN,
        },
        "assignment_freeze": freeze_metadata,
        "inference": {
            "primary_tail": "positive",
            "p_value": "(1 + count(placebo estimate >= actual estimate)) / 1001",
            "approved_pass_rule": "actual common estimate > 0 and one-sided positive-tail p < 0.05",
            "two_sided_absolute_tail": "reported descriptively",
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    RUN_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {ASSIGNMENTS.relative_to(ROOT)}")
    print(f"Saved: {ESTIMATES.relative_to(ROOT)}")
    print(f"Saved: {INFERENCE.relative_to(ROOT)}")
    print(f"Saved: {FIGURE.relative_to(ROOT)}")
    print("\nDrought random-two-sector inference")
    print(inference.to_string(index=False))


if __name__ == "__main__":
    main()
