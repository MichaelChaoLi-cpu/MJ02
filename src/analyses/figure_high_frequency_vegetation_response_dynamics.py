#!/usr/bin/env python3
"""Generate high-frequency distributed rainfall-response dynamics.

The activated design uses all 16-day rainfall variation within cross-side
CHIRPS cells.  Village fixed effects and CHIRPS-cell-by-date fixed effects make
the historical-side interaction compare villages facing exactly the same
rainfall history at the same date.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _high_frequency_recovery_models import (
    distributed_lag_rows,
    distributed_pretrend_test,
    fit_distributed_lag,
    linear_contrast,
)


ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data/processed/historical_boundary_16day_climate_vegetation_preprocessed.parquet"
OUTPUT = ROOT / "data/exp/legacy-results/figures/Figure_high_frequency_vegetation_response_dynamics.png"
EXP_DIR = ROOT / "data/exp/high-frequency-boundary-dynamics"
EVI = "High-Frequency EVI Anomaly Z"
NDVI = "High-Frequency NDVI Anomaly Z"
MDE_FACTOR = 1.959964 + 0.841621

SPEC_COLORS = {"Primary": "#242424", "Within commune": "#B4523A"}
SHOCK_COLORS = {"Dry": "#B87924", "Wet": "#347E9B"}


def weights(shock: str, estimand: str) -> dict[str, float]:
    if estimand == "Immediate average response, periods 0-2":
        periods, scale = (0, 1, 2), 1.0 / 3.0
    elif estimand == "Cumulative response, periods 0-5":
        periods, scale = range(0, 6), 1.0
    elif estimand == "Late response, periods 6-8":
        periods, scale = (6, 7, 8), 1.0 / 3.0
    else:
        raise ValueError(estimand)
    return {f"Southwest x {shock} k={period}": scale for period in periods}


def result_row(fit: object, shock: str, outcome_family: str, estimand: str) -> dict[str, object]:
    result = linear_contrast(fit, weights(shock, estimand))
    _, pretrend_p, _ = distributed_pretrend_test(fit, shock)
    return {
        "outcome_family": outcome_family,
        "shock_family": shock,
        "estimand": estimand,
        "specification": fit.specification,
        "scale": "EVI/NDVI SD per 1-SD rainfall-intensity change" if "Cumulative" not in estimand else "sum of EVI/NDVI SD responses",
        **result,
        "pretrend_p_value": pretrend_p,
        "observations": fit.observations,
        "dates": fit.events,
        "villages": fit.villages,
        "rainfall_groups": fit.groups,
        "inference": "Two-way clustered by village and composite date",
    }


def activation_audit(panel: pd.DataFrame, preliminary: object) -> pd.DataFrame:
    sample = panel.loc[panel["Cross-Side CHIRPS Cell"].eq(1)]
    immediate = linear_contrast(preliminary, weights("Dry", "Immediate average response, periods 0-2"))
    rows = [
        ("Cross-side CHIRPS cells", int(sample["CHIRPS Cell ID"].nunique()), 6, ">="),
        ("Villages on cross-side cells", int(sample["Village Code"].nunique()), 120, ">="),
        ("Composite dates retained by model", preliminary.events, 400, ">="),
        ("Observed EVI row share on cross-side cells", float(sample[EVI].notna().mean()), 0.55, ">="),
        ("Blinded 80% MDE for dry immediate response (SD)", MDE_FACTOR * immediate["standard_error"], 0.20, "<="),
    ]
    audit = []
    for diagnostic, value, threshold, direction in rows:
        passed = value >= threshold if direction == ">=" else value <= threshold
        audit.append({"diagnostic": diagnostic, "value": value, "threshold": threshold, "direction": direction, "pass": bool(passed)})
    return pd.DataFrame(audit)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Add the journal-style lowercase panel label without a panel title."""
    ax.text(
        -0.10,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def plot_path(ax: plt.Axes, paths: pd.DataFrame, shock: str, panel: str) -> None:
    ax.axhspan(-0.20, 0.20, color="#DDE9D8", alpha=0.72, zorder=0)
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.axvline(-0.5, color="#777777", linewidth=0.8, linestyle="--")
    for specification, offset in (("Primary", -0.07), ("Within commune", 0.07)):
        subset = paths.loc[
            paths["shock_family"].eq(shock)
            & paths["outcome"].eq(EVI)
            & paths["specification"].eq(specification)
        ].sort_values("event_time")
        x = subset["event_time"].to_numpy(float) + offset
        y = subset["estimate"].to_numpy(float)
        low = subset["ci_low"].to_numpy(float)
        high = subset["ci_high"].to_numpy(float)
        ax.errorbar(
            x, y, yerr=[y - low, high - y], fmt="o-", linewidth=1.25,
            markersize=3.8, capsize=2, color=SPEC_COLORS[specification], label=specification,
        )
    add_panel_label(ax, panel)
    ax.set_xlabel("16-day response lag")
    ax.set_ylabel("Southwest-minus-West EVI slope (SD)")
    ax.set_xticks(range(-3, 9))
    ax.grid(True, color="#E5E5E5", linewidth=0.55)
    ax.legend(frameon=False, fontsize=8, ncol=2)


def forest(ax: plt.Axes, summary: pd.DataFrame, estimand: str, panel: str, include_ndvi: bool) -> None:
    subset = summary.loc[summary["estimand"].eq(estimand)].copy()
    order = [
        ("EVI", "Dry", "Primary"), ("EVI", "Dry", "Within commune"),
        ("EVI", "Wet", "Primary"), ("EVI", "Wet", "Within commune"),
    ]
    if include_ndvi:
        order.extend([("NDVI", "Dry", "Primary"), ("NDVI", "Wet", "Primary")])
    labels = [f"{outcome} {shock.lower()}: {spec.lower()}" for outcome, shock, spec in order]
    for y, ((outcome, shock, specification), label) in enumerate(zip(order, labels)):
        row = subset.loc[
            subset["outcome_family"].eq(outcome)
            & subset["shock_family"].eq(shock)
            & subset["specification"].eq(specification)
        ].iloc[0]
        ax.errorbar(
            row["estimate"], y,
            xerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
            fmt="o", color=SHOCK_COLORS[shock], ecolor=SHOCK_COLORS[shock], capsize=3, markersize=5,
        )
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.axvspan(-0.20, 0.20, color="#DDE9D8", alpha=0.72, zorder=0)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Southwest-minus-West response per 1-SD shock intensity")
    add_panel_label(ax, panel)
    ax.grid(True, axis="x", color="#E5E5E5", linewidth=0.55)


def main() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    primary = fit_distributed_lag(panel, EVI, "Primary")
    audit = activation_audit(panel, primary)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(EXP_DIR / "blinded_activation_gate.csv", index=False)
    if not bool(audit["pass"].all()):
        print(audit.to_string(index=False))
        raise SystemExit("Distributed-lag estimation stopped: activation gate failed")

    fits = {
        ("EVI", "Primary"): primary,
        ("EVI", "Within commune"): fit_distributed_lag(panel, EVI, "Within commune"),
        ("NDVI", "Primary"): fit_distributed_lag(panel, NDVI, "Primary"),
    }
    path_frames = []
    for (outcome_family, specification), fit in fits.items():
        frame = distributed_lag_rows(fit)
        for shock in ("Dry", "Wet"):
            statistic, p_value, degrees = distributed_pretrend_test(fit, shock)
            mask = frame["shock_family"].eq(shock)
            frame.loc[mask, "pretrend_chi2"] = statistic
            frame.loc[mask, "pretrend_df"] = degrees
            frame.loc[mask, "pretrend_p_value"] = p_value
        path_frames.append(frame)
    paths = pd.concat(path_frames, ignore_index=True)
    paths.to_csv(EXP_DIR / "distributed_lag_path_estimates.csv", index=False)

    rows = []
    for (outcome_family, specification), fit in fits.items():
        estimands = (
            "Immediate average response, periods 0-2",
            "Cumulative response, periods 0-5",
            "Late response, periods 6-8",
        ) if outcome_family == "EVI" else ("Immediate average response, periods 0-2",)
        for shock in ("Dry", "Wet"):
            for estimand in estimands:
                rows.append(result_row(fit, shock, outcome_family, estimand))
    summary = pd.DataFrame(rows)
    summary.to_csv(EXP_DIR / "distributed_lag_summary_estimates.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=True)
    plot_path(axes[0, 0], paths, "Dry", "a")
    plot_path(axes[0, 1], paths, "Wet", "b")
    forest(axes[1, 0], summary, "Immediate average response, periods 0-2", "c", include_ndvi=True)
    forest(axes[1, 1], summary, "Late response, periods 6-8", "d", include_ndvi=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(audit.to_string(index=False))
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
