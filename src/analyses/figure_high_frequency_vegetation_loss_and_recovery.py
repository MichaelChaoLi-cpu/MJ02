#!/usr/bin/env python3
"""Generate the gated internal high-frequency vegetation loss and recovery figure.

Plan: AnaSOP Sections 5.7, 6.14, and 8.
The script first publishes an activation audit. It writes treatment-side
estimates and an internal diagnostic figure only when every frozen support and
precision gate passes; this stopped experiment never writes to formal results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _high_frequency_recovery_models import (
    EVENT_TIMES,
    event_time_rows,
    fit_dynamic_path,
    fit_scalar_outcome,
    linear_contrast,
    pretrend_test,
    scalar_result_row,
)


ROOT = Path(__file__).resolve().parents[2]
STACK_PATH = ROOT / "data/processed/historical_boundary_vegetation_event_stack_preprocessed.parquet"
EVENT_PATH = ROOT / "data/processed/historical_boundary_vegetation_event_outcomes_preprocessed.parquet"
OUTPUT = ROOT / "data/exp/internal_output_archive/figures/Figure_high_frequency_vegetation_loss_and_recovery.png"
EXP_DIR = ROOT / "data/exp/high-frequency-boundary-recovery"

PRIMARY_OUTCOME = "High-Frequency EVI Anomaly Z"
MIN_CROSS_SIDE_CELLS = 6
MIN_VILLAGES = 120
MIN_DRY_EVENTS = 30
MIN_WET_EVENTS = 20
MIN_OBSERVATION_COVERAGE = 0.85
MAX_MDE_SD = 0.20
ALPHA_95_PLUS_POWER_80 = 1.959964 + 0.841621

SPEC_COLORS = {"Primary": "#242424", "Within commune": "#B4523A"}
SHOCK_COLORS = {"Dry": "#B87924", "Wet": "#347E9B"}


def immediate_weights() -> dict[str, float]:
    weights = {f"Southwest x k={period}": -1.0 / 3.0 for period in (0, 1, 2)}
    weights.update({f"Southwest x k={period}": 1.0 / 3.0 for period in (-3, -2)})
    # k=-1 is the omitted reference and therefore contributes zero.
    return weights


def activation_audit(stack: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    observation_coverage = float(stack[PRIMARY_OUTCOME].notna().mean())
    event_counts = events.groupby("Shock Family", observed=True)["Event ID"].nunique()
    preliminary = fit_dynamic_path(stack, "Dry", PRIMARY_OUTCOME, "Primary")
    immediate = linear_contrast(preliminary, immediate_weights())
    mde = ALPHA_95_PLUS_POWER_80 * immediate["standard_error"]
    diagnostics = [
        ("Cross-side CHIRPS cells", int(events["CHIRPS Cell ID"].nunique()), MIN_CROSS_SIDE_CELLS, ">="),
        ("Villages in event stack", int(stack["Village Code"].nunique()), MIN_VILLAGES, ">="),
        ("Retained dry events", int(event_counts.get("Dry", 0)), MIN_DRY_EVENTS, ">="),
        ("Retained wet events", int(event_counts.get("Wet", 0)), MIN_WET_EVENTS, ">="),
        ("Observed EVI event-window row share", observation_coverage, MIN_OBSERVATION_COVERAGE, ">="),
        ("Blinded 80% MDE for immediate differential loss (SD)", float(mde), MAX_MDE_SD, "<="),
    ]
    rows = []
    for diagnostic, value, threshold, direction in diagnostics:
        passed = value >= threshold if direction == ">=" else value <= threshold
        rows.append({
            "diagnostic": diagnostic,
            "value": value,
            "threshold": threshold,
            "direction": direction,
            "pass": bool(passed),
        })
    return pd.DataFrame(rows), preliminary


def build_results(stack: pd.DataFrame, events: pd.DataFrame, dry_primary: object) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_fits = {}
    path_frames = []
    for outcome in (PRIMARY_OUTCOME, "High-Frequency NDVI Anomaly Z"):
        specifications = ("Primary", "Within commune") if outcome == PRIMARY_OUTCOME else ("Primary",)
        for shock in ("Dry", "Wet"):
            for specification in specifications:
                key = (shock, outcome, specification)
                fit = dry_primary if key == ("Dry", PRIMARY_OUTCOME, "Primary") else fit_dynamic_path(
                    stack, shock, outcome, specification
                )
                path_fits[key] = fit
                frame = event_time_rows(fit)
                statistic, p_value, degrees = pretrend_test(fit)
                frame["pretrend_chi2"] = statistic
                frame["pretrend_df"] = degrees
                frame["pretrend_p_value"] = p_value
                path_frames.append(frame)
    paths = pd.concat(path_frames, ignore_index=True)

    metrics = (
        ("Immediate Vegetation Loss SD", "Immediate loss", "outcome SD"),
        ("Cumulative Negative Vegetation Loss SD", "Cumulative negative loss, periods 0-5", "sum of outcome SD"),
        ("Restricted Recovery Time through Period 8", "Restricted recovery time", "16-day composites"),
        ("Recovered by Period 8", "Recovered by period 8", "probability"),
    )
    rows = []
    for prefix in ("EVI", "NDVI"):
        specifications = ("Primary", "Within commune") if prefix == "EVI" else ("Primary",)
        for shock in ("Dry", "Wet"):
            for specification in specifications:
                for suffix, estimand, scale in metrics:
                    # NDVI is a sensor-construction check; retain only its loss
                    # metrics to prevent the table from becoming redundant.
                    if prefix == "NDVI" and suffix not in {
                        "Immediate Vegetation Loss SD", "Cumulative Negative Vegetation Loss SD"
                    }:
                        continue
                    outcome = f"{prefix} {suffix}"
                    fit = fit_scalar_outcome(events, shock, outcome, specification)
                    row = scalar_result_row(fit, estimand, scale)
                    row["outcome_family"] = prefix
                    path_outcome = PRIMARY_OUTCOME if prefix == "EVI" else "High-Frequency NDVI Anomaly Z"
                    path_fit = path_fits[(shock, path_outcome, specification)]
                    _, p_value, _ = pretrend_test(path_fit)
                    row["pretrend_p_value"] = p_value
                    rows.append(row)
    return paths, pd.DataFrame(rows)


def plot_path(ax: plt.Axes, paths: pd.DataFrame, shock: str, panel_label: str) -> None:
    ax.axhspan(-0.20, 0.20, color="#DDE9D8", alpha=0.72, zorder=0)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.axvline(-0.5, color="#777777", linewidth=0.8, linestyle="--")
    for specification, offset in (("Primary", -0.08), ("Within commune", 0.08)):
        subset = paths.loc[
            paths["shock_family"].eq(shock)
            & paths["outcome"].eq(PRIMARY_OUTCOME)
            & paths["specification"].eq(specification)
        ].sort_values("event_time")
        x = subset["event_time"].to_numpy(float) + offset
        y = subset["estimate"].to_numpy(float)
        low = subset["ci_low"].to_numpy(float)
        high = subset["ci_high"].to_numpy(float)
        ax.errorbar(
            x, y, yerr=[y - low, high - y], fmt="o-", linewidth=1.25,
            markersize=3.8, capsize=2.0, color=SPEC_COLORS[specification],
            label=specification,
        )
    primary = paths.loc[
        paths["shock_family"].eq(shock)
        & paths["outcome"].eq(PRIMARY_OUTCOME)
        & paths["specification"].eq("Primary")
    ].iloc[0]
    ax.set_title(f"{panel_label}  {shock.lower()} rainfall extremes (pretrend p={primary['pretrend_p_value']:.2f})", loc="left", fontweight="bold")
    ax.set_xlabel("16-day composites from the rainfall extreme")
    ax.set_ylabel("Southwest-minus-West EVI response (SD)")
    ax.set_xticks(EVENT_TIMES)
    ax.grid(True, color="#E5E5E5", linewidth=0.55)
    ax.legend(frameon=False, fontsize=8, ncol=2)


def forest(ax: plt.Axes, rows: pd.DataFrame, estimand: str, panel_label: str, xlabel: str) -> None:
    subset = rows.loc[rows["outcome_family"].eq("EVI") & rows["estimand"].eq(estimand)].copy()
    order = [("Dry", "Primary"), ("Dry", "Within commune"), ("Wet", "Primary"), ("Wet", "Within commune")]
    labels = ["Dry: primary", "Dry: within commune", "Wet: primary", "Wet: within commune"]
    for y, ((shock, specification), label) in enumerate(zip(order, labels)):
        row = subset.loc[subset["shock_family"].eq(shock) & subset["specification"].eq(specification)].iloc[0]
        ax.errorbar(
            row["estimate"], y,
            xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
            fmt="o", color=SHOCK_COLORS[shock], ecolor=SHOCK_COLORS[shock], capsize=3, markersize=5,
        )
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(f"{panel_label}  {estimand.lower()}", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.55)


def main() -> None:
    stack = pd.read_parquet(STACK_PATH)
    events = pd.read_parquet(EVENT_PATH)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    audit, dry_primary = activation_audit(stack, events)
    audit.to_csv(EXP_DIR / "blinded_activation_gate.csv", index=False)
    if not bool(audit["pass"].all()):
        print(audit.to_string(index=False))
        raise SystemExit("High-frequency effect estimation stopped: at least one frozen activation gate failed")

    paths, scalar = build_results(stack, events, dry_primary)
    paths.to_csv(EXP_DIR / "dynamic_event_path_estimates.csv", index=False)
    scalar.to_csv(EXP_DIR / "event_loss_and_recovery_estimates.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=True)
    plot_path(axes[0, 0], paths, "Dry", "a")
    plot_path(axes[0, 1], paths, "Wet", "b")
    forest(axes[1, 0], scalar, "Immediate loss", "c", "Differential immediate vegetation loss (SD)")
    forest(axes[1, 1], scalar, "Recovered by period 8", "d", "Difference in recovery probability")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(audit.to_string(index=False))
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
