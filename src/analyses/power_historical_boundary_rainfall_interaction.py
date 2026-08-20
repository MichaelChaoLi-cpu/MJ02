#!/usr/bin/env python3
"""Outcome-blind design power for the historical-boundary rainfall interaction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


TREATMENT = "Higher-Repression Southwest Zone"
DISTANCE = "Signed Distance to Historical Repression Boundary km"
SEGMENT = "Historical Boundary Segment"
SPI12 = "Interview Month SPI 12 Month"
ANNUAL_RAINFALL = "Annual Rainfall Anomaly Z (1991-2020)"
RAINFALL_SHOCKS = (SPI12, ANNUAL_RAINFALL)
PRICE = "12 Month Change in Local Relative Log Wholesale Rice Price"
BANDWIDTHS_KM = (2, 5, 10, 15, 20, 30)
SCENARIOS = (
    ("iid diagnostic", 0.0, 0.0),
    ("moderate local dependence", 0.25, 0.25),
    ("strong local dependence", 0.25, 0.50),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exp/feasibility-check/historical-boundary"),
    )
    parser.add_argument("--simulations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def collapse_shocks(root: Path) -> pd.DataFrame:
    columns = ["Survey Wave", "PSU", *RAINFALL_SHOCKS, PRICE]
    shocks = pd.read_parquet(
        root / "data/processed/direction3_household_conflict_shock_preprocessed.parquet",
        columns=columns,
    )
    shocks["Survey Wave"] = shocks["Survey Wave"].astype(str)
    shocks["PSU"] = shocks["PSU"].astype(str)
    variation = shocks.groupby(["Survey Wave", "PSU"])[[*RAINFALL_SHOCKS, PRICE]].nunique(
        dropna=False
    )
    if variation.gt(1).any().any():
        raise RuntimeError("Shock values vary within a PSU-wave")
    return shocks.drop_duplicates(["Survey Wave", "PSU"])


def residualized_interaction(
    data: pd.DataFrame, shock_name: str
) -> tuple[np.ndarray, float, int, float]:
    treatment = data[TREATMENT].to_numpy(dtype=float)
    shock = data[shock_name].to_numpy(dtype=float)
    shock = (shock - shock.mean()) / shock.std(ddof=0)
    distance = data[DISTANCE].to_numpy(dtype=float)
    interaction = treatment * shock

    wave = pd.get_dummies(data["Survey Wave"], prefix="wave", drop_first=True, dtype=float)
    segment = pd.get_dummies(data[SEGMENT].astype(int), prefix="segment", drop_first=True, dtype=float)
    nuisance = np.column_stack(
        [
            np.ones(len(data)),
            treatment,
            shock,
            distance,
            treatment * distance,
            wave.to_numpy(),
            segment.to_numpy(),
        ]
    )
    residualized = interaction - nuisance @ (np.linalg.pinv(nuisance) @ interaction)
    denominator = float(residualized @ residualized)
    rank = int(np.linalg.matrix_rank(np.column_stack([nuisance, interaction])))
    condition = float(np.linalg.cond(np.column_stack([nuisance, interaction])))
    if denominator < 1e-10:
        raise RuntimeError("The treatment-by-shock interaction is not identified")
    return residualized, denominator, rank, condition


def simulated_errors(
    data: pd.DataFrame,
    simulations: int,
    rng: np.random.Generator,
    village_share: float,
    local_cell_wave_share: float,
) -> np.ndarray:
    individual_share = 1.0 - village_share - local_cell_wave_share
    if individual_share < 0:
        raise ValueError("Dependence shares exceed one")
    n = len(data)
    errors = np.sqrt(individual_share) * rng.standard_normal((simulations, n))

    village_codes, village_index = np.unique(
        data["Matched Public Village Code"].astype(str), return_inverse=True
    )
    if village_share:
        village_effect = rng.standard_normal((simulations, len(village_codes)))
        errors += np.sqrt(village_share) * village_effect[:, village_index]

    local_cell_wave = (
        data[TREATMENT].astype(int).astype(str)
        + "-"
        + data[SEGMENT].astype(int).astype(str)
        + "-"
        + data["Survey Wave"].astype(str)
    )
    local_cell_wave_codes, local_cell_wave_index = np.unique(
        local_cell_wave, return_inverse=True
    )
    if local_cell_wave_share:
        local_cell_wave_effect = rng.standard_normal(
            (simulations, len(local_cell_wave_codes))
        )
        errors += (
            np.sqrt(local_cell_wave_share)
            * local_cell_wave_effect[:, local_cell_wave_index]
        )
    return errors


def power_grid(data: pd.DataFrame, simulations: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for shock_index, shock_name in enumerate(RAINFALL_SHOCKS):
        for bandwidth in BANDWIDTHS_KM:
            sample = data.loc[
                data[f"Historical-Boundary Common Support {bandwidth} km"].eq(1)
                & data[shock_name].notna()
            ].copy()
            if sample.empty:
                continue
            residualized, denominator, rank, condition = residualized_interaction(
                sample, shock_name
            )
            for scenario_index, (scenario, village_share, local_cell_wave_share) in enumerate(
                SCENARIOS
            ):
                rng = np.random.default_rng(
                    seed + 1000 * shock_index + 100 * bandwidth + scenario_index
                )
                errors = simulated_errors(
                    sample,
                    simulations,
                    rng,
                    village_share,
                    local_cell_wave_share,
                )
                theta = errors @ residualized / denominator
                standard_error = float(theta.std(ddof=1))
                rows.append(
                    {
                        "Shock": shock_name,
                        "Bandwidth km": bandwidth,
                        "Dependence Scenario": scenario,
                        "Village Error Share": village_share,
                        "Local Cell-Wave Error Share": local_cell_wave_share,
                        "PSU-Wave Rows": len(sample),
                        "Southwest PSU-Wave Rows": int(sample[TREATMENT].eq(1).sum()),
                        "West PSU-Wave Rows": int(sample[TREATMENT].eq(0).sum()),
                        "Unique Matched Villages": sample["Matched Public Village Code"].nunique(),
                        "Survey Waves": sample["Survey Wave"].nunique(),
                        "Boundary Segments": sample[SEGMENT].nunique(),
                        "Design Matrix Rank": rank,
                        "Design Matrix Condition Number": condition,
                        "Residualized Interaction Information": denominator,
                        "Simulated Standard Error": standard_error,
                        "MDE 80 Percent Power SD": (norm.ppf(0.975) + norm.ppf(0.80))
                        * standard_error,
                        "MDE 90 Percent Power SD": (norm.ppf(0.975) + norm.ppf(0.90))
                        * standard_error,
                        "Simulations": simulations,
                    }
                )
    return pd.DataFrame(rows)


def activation_gate(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bandwidth in BANDWIDTHS_KM:
        support = data.loc[data[f"Historical-Boundary Common Support {bandwidth} km"].eq(1)]
        for shock, role in (
            (SPI12, "interview-aligned drought"),
            (ANNUAL_RAINFALL, "full-wave annual rainfall"),
            (PRICE, "current local rice price"),
        ):
            available = support.loc[support[shock].notna()]
            waves_by_side = available.groupby(TREATMENT)["Survey Wave"].nunique()
            minimum_side_waves = int(waves_by_side.min()) if len(waves_by_side) == 2 else 0
            status = (
                "proceed to power and continuity gates"
                if shock in RAINFALL_SHOCKS and minimum_side_waves >= 5
                else "fail current temporal-support gate"
            )
            rows.append(
                {
                    "Bandwidth km": bandwidth,
                    "Shock": shock,
                    "Role": role,
                    "Available PSU-Wave Rows": len(available),
                    "Available Survey Waves": available["Survey Wave"].nunique(),
                    "Minimum Side-Specific Survey Waves": minimum_side_waves,
                    "Activation Status": status,
                }
            )
    rows.append(
        {
            "Bandwidth km": pd.NA,
            "Shock": "International Rice Price Shock",
            "Role": "candidate external price",
            "Available PSU-Wave Rows": 0,
            "Available Survey Waves": 0,
            "Minimum Side-Specific Survey Waves": 0,
            "Activation Status": "not acquired; not tested",
        }
    )
    return pd.DataFrame(rows)


def write_plot(results: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    colors = {
        "iid diagnostic": "#7f7f7f",
        "moderate local dependence": "#2166ac",
        "strong local dependence": "#b2182b",
    }
    titles = {SPI12: "Interview-aligned SPI12", ANNUAL_RAINFALL: "Annual rainfall anomaly"}
    for ax, shock_name in zip(axes, RAINFALL_SHOCKS, strict=True):
        shock_results = results.loc[results["Shock"].eq(shock_name)]
        for scenario, group in shock_results.groupby("Dependence Scenario"):
            ax.plot(
                group["Bandwidth km"],
                group["MDE 80 Percent Power SD"],
                marker="o",
                linewidth=1.8,
                color=colors[scenario],
                label=scenario,
            )
        ax.set_xlabel("Candidate boundary bandwidth (km)")
        ax.set_title(titles[shock_name])
        ax.grid(True, color="#d0d0d0", linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#333333")
    axes[0].set_ylabel("80% power minimum detectable effect (SD)")
    axes[1].legend(frameon=True)
    fig.suptitle("Outcome-blind rainfall-interaction design power")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    boundary = pd.read_parquet(root / "data/processed/historical_boundary_design_preprocessed.parquet")
    boundary["Survey Wave"] = boundary["Survey Wave"].astype(str)
    boundary["PSU"] = boundary["PSU"].astype(str)
    data = boundary.merge(
        collapse_shocks(root), on=["Survey Wave", "PSU"], how="left", validate="one_to_one"
    )

    results = power_grid(data, args.simulations, args.seed)
    gate = activation_gate(data)
    results.to_csv(output / "rainfall_interaction_design_power.csv", index=False)
    gate.to_csv(output / "shock_activation_gate.csv", index=False)
    write_plot(results, output / "rainfall_interaction_design_power.png")

    strong_dependence = results.loc[
        results["Dependence Scenario"].eq("strong local dependence")
    ]
    best = strong_dependence.loc[strong_dependence["MDE 80 Percent Power SD"].idxmin()]
    readme = f"""# Outcome-Blind Boundary Interaction Power

This simulation reads treatment, distance, boundary segment, survey identifiers, and shock values
only. It does not read or estimate any contemporary outcome.

- Simulations per bandwidth and dependence scenario: {args.simulations:,}
- Rainfall shocks: {SPI12}; {ANNUAL_RAINFALL}
- Candidate bandwidths: {', '.join(map(str, BANDWIDTHS_KM))} km
- Error scenarios: independent diagnostic; moderate village and local side-by-segment-by-wave
  dependence; strong local dependence.
- Best strong-dependence 80% power MDE in the tested grid: {best['MDE 80 Percent Power SD']:.3f}
  residual-outcome SD at {int(best['Bandwidth km'])} km.
- The MDE is not a pass/fail result until the smallest effect of substantive interest is set
  independently of observed outcomes.
- Current local rice-price exposure fails the temporal-support gate at every tested bandwidth.
- International Rice Price Shock remains untested because it has not been acquired.

Interpret the MDE as the differential standardized outcome response to a one-standard-deviation
change in the named rainfall measure under the specified fixed design. The simulation is
deliberately conservative at the PSU-wave level and does not count household or person records as
independent assignment units.
"""
    (output / "README_power.md").write_text(readme, encoding="utf-8")
    print(strong_dependence[["Shock", "Bandwidth km", "PSU-Wave Rows", "Survey Waves", "MDE 80 Percent Power SD"]].to_string(index=False))
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
