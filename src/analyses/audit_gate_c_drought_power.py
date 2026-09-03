#!/usr/bin/env python3
"""Prospective design-rank and power gate for the Gate C drought interaction."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mj02-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/cambodia_thailand_gate_c_annual_panel_preprocessed.parquet"
OUT = ROOT / "data/exp/experiment-design/cambodia-thailand-gate-c"
DROUGHT = "May October Dry Rainfall Intensity"
EXTREME_RAIN = "May October Extreme Wet Rainfall Intensity"
HEAT = "May October Heat Intensity"
LONGNTL = "Asinh Annual NPP-VIIRS-like Radiance"
NPP = "Annual Land NPP Anomaly Z 2001-2020"
POWER = OUT / "gate_c_drought_prospective_power.csv"
DESIGN = OUT / "gate_c_drought_design_rank.csv"
FIGURE = OUT / "gate_c_drought_power_curve.png"
METADATA = OUT / "gate_c_drought_power_metadata.json"
CROSSWALK = OUT / "gate_c_effect_scale_human_review.csv"
OUTCOME_AUDIT = OUT / "gate_c_outcome_scale_audit.csv"
SESOI = 0.20
ALPHA = 0.05
TARGET_POWER = 0.80
SIMULATIONS = 5000
SEED = 2011


def weighted_block_year(frame: pd.DataFrame) -> pd.DataFrame:
    group = ["Spatial Block ID", "Border Analysis Sector", "Year"]
    weight = frame["Gate B Binary Support Overlap Weight"].to_numpy(float)
    work = frame[group].copy()
    work["Weight"] = weight
    variables = ["Continuous Conflict Dose", DROUGHT, EXTREME_RAIN, HEAT]
    for variable in variables:
        work[variable] = frame[variable].to_numpy(float) * weight
    sums = work.groupby(group, as_index=False).sum(numeric_only=True)
    for variable in variables:
        sums[variable] = sums[variable] / sums["Weight"]
    for outcome in (LONGNTL, NPP):
        outcome_work = frame[group].copy()
        valid = frame[outcome].notna().to_numpy()
        outcome_work["Outcome Numerator"] = np.where(
            valid, frame[outcome].fillna(0).to_numpy(float) * weight, 0.0
        )
        outcome_work["Outcome Weight"] = np.where(valid, weight, 0.0)
        outcome_sum = outcome_work.groupby(group, as_index=False).sum(numeric_only=True)
        outcome_sum[outcome] = np.divide(
            outcome_sum["Outcome Numerator"],
            outcome_sum["Outcome Weight"],
            out=np.full(len(outcome_sum), np.nan),
            where=outcome_sum["Outcome Weight"].gt(0),
        )
        sums = sums.merge(outcome_sum[[*group, outcome]], on=group, validate="one_to_one")
    sums["Post"] = sums["Year"].between(2012, 2024).astype(float)
    sums["Dose x Post"] = sums["Continuous Conflict Dose"] * sums["Post"]
    sums["Post x Drought"] = sums["Post"] * sums[DROUGHT]
    sums["Dose x Drought"] = sums["Continuous Conflict Dose"] * sums[DROUGHT]
    sums["Dose x Post x Drought"] = (
        sums["Continuous Conflict Dose"] * sums["Post"] * sums[DROUGHT]
    )
    return sums.sort_values(["Spatial Block ID", "Year"]).reset_index(drop=True)


def subtract_group_mean(values: np.ndarray, groups: np.ndarray, weights: np.ndarray) -> np.ndarray:
    output = values.copy()
    group_count = int(groups.max()) + 1
    denominators = np.bincount(groups, weights=weights, minlength=group_count)
    for column in range(output.shape[1]):
        numerators = np.bincount(
            groups, weights=weights * output[:, column], minlength=group_count
        )
        means = np.divide(
            numerators,
            denominators,
            out=np.zeros_like(numerators, dtype=float),
            where=denominators > 0,
        )
        output[:, column] -= means[groups]
    return output


def absorb_fixed_effects(
    values: np.ndarray,
    block_groups: np.ndarray,
    sector_year_groups: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    output = values.copy()
    for _ in range(200):
        previous = output.copy()
        output = subtract_group_mean(output, block_groups, weights)
        output = subtract_group_mean(output, sector_year_groups, weights)
        if np.max(np.abs(output - previous)) < 1e-11:
            break
    return output


def residualized_target(block_year: pd.DataFrame) -> tuple[np.ndarray, dict[str, float]]:
    target = block_year[["Dose x Post x Drought"]].to_numpy(float)
    lower_order = block_year[
        [
            DROUGHT,
            "Dose x Post",
            "Post x Drought",
            "Dose x Drought",
            EXTREME_RAIN,
            HEAT,
        ]
    ].to_numpy(float)
    values = np.column_stack([target, lower_order])
    block_groups = pd.Categorical(block_year["Spatial Block ID"]).codes
    sector_year_groups = pd.Categorical(
        block_year["Border Analysis Sector"] + "__" + block_year["Year"].astype(str)
    ).codes
    weights = np.ones(len(block_year), dtype=float)
    absorbed = absorb_fixed_effects(values, block_groups, sector_year_groups, weights)
    target_absorbed = absorbed[:, 0]
    nuisance_absorbed = absorbed[:, 1:]
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        nuisance_absorbed, target_absorbed, rcond=None
    )
    target_residual = target_absorbed - nuisance_absorbed @ coefficients
    raw_variance = float(np.var(target_absorbed, ddof=1))
    residual_variance = float(np.var(target_residual, ddof=1))
    diagnostics = {
        "Block-Year Observations": len(block_year),
        "Spatial Blocks": block_year["Spatial Block ID"].nunique(),
        "Years": block_year["Year"].nunique(),
        "Nuisance Rank": int(rank),
        "Nuisance Columns": nuisance_absorbed.shape[1],
        "Nuisance Condition Number": float(singular_values[0] / singular_values[-1]),
        "Absorbed Target Variance": raw_variance,
        "Residual Target Variance": residual_variance,
        "Target Variance Retained Share": residual_variance / raw_variance,
        "Target VIF": raw_variance / residual_variance,
        "Residual Target Sum of Squares": float(target_residual @ target_residual),
    }
    return target_residual, diagnostics


def simulate_standard_error(
    target: np.ndarray,
    block_codes: np.ndarray,
    year_codes: np.ndarray,
    rho: float,
    simulations: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    blocks = int(block_codes.max()) + 1
    years = int(year_codes.max()) + 1
    denominator = float(target @ target)
    estimates: list[np.ndarray] = []
    batch_size = 250
    for start in range(0, simulations, batch_size):
        batch = min(batch_size, simulations - start)
        innovations = rng.normal(size=(batch, blocks, years))
        errors = np.empty_like(innovations)
        errors[:, :, 0] = innovations[:, :, 0]
        scale = np.sqrt(1 - rho**2)
        for index in range(1, years):
            errors[:, :, index] = rho * errors[:, :, index - 1] + scale * innovations[:, :, index]
        row_errors = errors[:, block_codes, year_codes]
        estimates.append((row_errors @ target) / denominator)
    return float(np.std(np.concatenate(estimates), ddof=1))


def power_for_effect(effect: np.ndarray, standard_error: float) -> np.ndarray:
    critical = norm.ppf(1 - ALPHA / 2)
    return norm.cdf(-critical - effect / standard_error) + 1 - norm.cdf(
        critical - effect / standard_error
    )


def preconflict_residual_scale(
    block_year: pd.DataFrame,
    outcome: str,
    grid_pre_sd: float,
) -> float:
    pre = block_year.loc[block_year["Year"].between(2000, 2007)].dropna(
        subset=[outcome, DROUGHT, EXTREME_RAIN, HEAT]
    ).copy()
    values = np.column_stack(
        [
            pre[outcome].to_numpy(float) / grid_pre_sd,
            pre[[DROUGHT, EXTREME_RAIN, HEAT]].to_numpy(float),
        ]
    )
    block_groups = pd.Categorical(pre["Spatial Block ID"]).codes
    sector_year_groups = pd.Categorical(
        pre["Border Analysis Sector"] + "__" + pre["Year"].astype(str)
    ).codes
    absorbed = absorb_fixed_effects(
        values, block_groups, sector_year_groups, np.ones(len(pre), dtype=float)
    )
    y = absorbed[:, 0]
    x = absorbed[:, 1:]
    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coefficients
    return float(np.std(residual, ddof=x.shape[1] + 1))


def plot_power(power: pd.DataFrame) -> None:
    effects = np.linspace(0, 0.4, 161)
    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    colors = {"LongNTL": "#2C6E9B", "Annual NPP": "#B54A3A"}
    linestyles = {0.0: ":", 0.5: "-", 0.8: "--"}
    for row in power.itertuples(index=False):
        rho = float(row.AR1_Rho)
        standard_error = float(row.Null_Standard_Error)
        axis.plot(
            effects,
            power_for_effect(effects, standard_error),
            color=colors[row.Outcome_Calibration],
            linestyle=linestyles[rho],
            label=f"{row.Outcome_Calibration}, rho={rho:.1f}",
        )
    axis.axhline(TARGET_POWER, color="black", linestyle="--", linewidth=0.9)
    axis.axvline(SESOI, color="#666666", linestyle=":", linewidth=1.1)
    axis.set_xlabel("Absolute drought-amplification effect (outcome SD)")
    axis.set_ylabel("Two-sided rejection probability")
    axis.set_ylim(0, 1.02)
    axis.set_title("Prospective Gate C drought power at the 10 km block-year level")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(PANEL)
    frame = frame.loc[frame["Gate C Main Pre-Post Period"]].copy()
    block_year = weighted_block_year(frame)
    counts = block_year.groupby("Spatial Block ID")["Year"].nunique()
    if not counts.eq(21).all():
        raise RuntimeError("Power panel is not balanced at 21 years per spatial block")
    target, diagnostics = residualized_target(block_year)
    block_codes = pd.Categorical(block_year["Spatial Block ID"]).codes
    year_codes = pd.Categorical(block_year["Year"], ordered=True).codes
    outcome_scales = {}
    for label, outcome in (("LongNTL", LONGNTL), ("Annual NPP", NPP)):
        grid_pre = frame.loc[frame["Year"].between(2000, 2007), outcome].dropna()
        grid_pre_sd = float(grid_pre.std(ddof=1))
        outcome_scales[label] = {
            "grid_pre_sd": grid_pre_sd,
            "block_year_residual_sd": preconflict_residual_scale(
                block_year, outcome, grid_pre_sd
            ),
        }
    rows = []
    effects = np.array([0.10, 0.20, 0.30])
    for rho in (0.0, 0.5, 0.8):
        unit_standard_error = simulate_standard_error(
            target,
            block_codes,
            year_codes,
            rho,
            SIMULATIONS,
            SEED + int(rho * 10),
        )
        for label, calibration in outcome_scales.items():
            standard_error = unit_standard_error * calibration["block_year_residual_sd"]
            effect_power = power_for_effect(effects, standard_error)
            mde80 = (norm.ppf(1 - ALPHA / 2) + norm.ppf(TARGET_POWER)) * standard_error
            rows.append(
                {
                    "Outcome Calibration": label,
                    "Pre-conflict Grid Outcome SD": calibration["grid_pre_sd"],
                    "Pre-conflict Block-Year Residual SD in Grid SD Units": calibration[
                        "block_year_residual_sd"
                    ],
                    "AR1 Rho": rho,
                    "Unit-Residual Null Standard Error": unit_standard_error,
                    "Null Standard Error": standard_error,
                    "80 Percent MDE": mde80,
                    "Power at 0.10 SD": effect_power[0],
                    "Power at 0.20 SD": effect_power[1],
                    "Power at 0.30 SD": effect_power[2],
                    "Passes 0.20 SD Power Gate": bool(effect_power[1] >= TARGET_POWER),
                }
            )
    power = pd.DataFrame(rows)
    design = pd.DataFrame([diagnostics])
    review_effects = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])
    crosswalk_rows = []
    for label, calibration in outcome_scales.items():
        standard_error = float(
            power.loc[
                power["Outcome Calibration"].eq(label) & power["AR1 Rho"].eq(0.5),
                "Null Standard Error",
            ].iloc[0]
        )
        unit_label = (
            "asinh radiance units"
            if label == "LongNTL"
            else "annual NPP anomaly-Z units"
        )
        for effect, rejection_probability in zip(
            review_effects, power_for_effect(review_effects, standard_error), strict=True
        ):
            crosswalk_rows.append(
                {
                    "Outcome": label,
                    "Candidate Effect in Pre-conflict Outcome SD": effect,
                    "Equivalent Outcome-Scale Slope Change": (
                        effect * calibration["grid_pre_sd"]
                    ),
                    "Outcome-Scale Unit": unit_label,
                    "Power under AR1 Rho 0.5": rejection_probability,
                    "Power at Least 0.80": bool(rejection_probability >= 0.80),
                    "Estimand Scale": (
                        "event-location-versus-zero-dose contrast per one-unit increase "
                        "in dry-rainfall intensity"
                    ),
                }
            )
    crosswalk = pd.DataFrame(crosswalk_rows)
    pre = frame.loc[frame["Year"].between(2000, 2007)]
    outcome_audit = pd.DataFrame(
        [
            {
                "Outcome": "LongNTL",
                "Pre-conflict Observations": pre[LONGNTL].notna().sum(),
                "Pre-conflict Missing Share": pre[LONGNTL].isna().mean(),
                "Pre-conflict Mean": pre[LONGNTL].mean(),
                "Pre-conflict SD": pre[LONGNTL].std(ddof=1),
                "Pre-conflict Zero Share": pre[LONGNTL].eq(0).mean(),
                "Scale Warning": (
                    "Continuous LongNTL is extremely zero-inflated on this frontier; "
                    "standardized effects are not uniform percentage changes."
                ),
            },
            {
                "Outcome": "Annual NPP",
                "Pre-conflict Observations": pre[NPP].notna().sum(),
                "Pre-conflict Missing Share": pre[NPP].isna().mean(),
                "Pre-conflict Mean": pre[NPP].mean(),
                "Pre-conflict SD": pre[NPP].std(ddof=1),
                "Pre-conflict Zero Share": pre[NPP].eq(0).mean(),
                "Scale Warning": (
                    "The source outcome is already an anomaly Z score; the reported "
                    "empirical pre-conflict SD is not exactly one."
                ),
            },
        ]
    )
    power.to_csv(POWER, index=False)
    design.to_csv(DESIGN, index=False)
    crosswalk.to_csv(CROSSWALK, index=False)
    outcome_audit.to_csv(OUTCOME_AUDIT, index=False)
    plot_power(power.rename(columns=lambda value: value.replace(" ", "_").replace("-", "_")))
    metadata = {
        "status": "prospective target-blind Gate C drought power gate",
        "analysis_level": "10 km spatial block by year",
        "periods": "2000-2007 versus 2012-2024; 2008-2011 excluded",
        "target": "continuous conflict dose x post-conflict x drought intensity",
        "lower_order_terms": [
            "drought intensity",
            "conflict dose x post-conflict",
            "post-conflict x drought intensity",
            "conflict dose x drought intensity",
        ],
        "fixed_effects": "10 km spatial block and border-sector by year",
        "cohazard_controls": [EXTREME_RAIN, HEAT],
        "simulation": {
            "outcome calibration": "pre-conflict block-year residual SD after block and sector-year fixed effects plus drought, extreme-rainfall, and heat controls; no post-conflict outcome or target interaction opened",
            "AR1 rho scenarios": [0.0, 0.5, 0.8],
            "simulations": SIMULATIONS,
            "seed": SEED,
            "alpha": ALPHA,
            "candidate_target_power_pending_human_review": TARGET_POWER,
            "candidate_SESOI_pending_human_review": SESOI,
        },
        "screening_rule": "provisional analyst default pending explicit human review: power at an absolute 0.20 outcome-SD amplification effect is at least 0.80 under AR1 rho 0.5; rho 0.0 and 0.8 are sensitivity cases and are not ordered as conservative versus optimistic",
        "post_conflict_outcomes_opened": False,
        "pre_conflict_outcomes_used_only_for_variance_calibration": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Design rank")
    print(design.to_string(index=False))
    print("\nProspective power")
    print(power.to_string(index=False))


if __name__ == "__main__":
    main()
