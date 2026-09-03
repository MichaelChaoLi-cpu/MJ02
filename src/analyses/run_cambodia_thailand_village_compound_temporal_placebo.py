#!/usr/bin/env python3
"""Run the synchronized-onset temporal placebo for village NPP.

Plan: Hold villages, sector-specific calibration weights, outcome, climate
variables, and the full compound hot-dry model fixed while synchronously moving
the two sector onset years across every admissible annual placement.
Framework: AnaSOP Sections 5-7 timing gate and conflict-conditioned climate-
sensitivity model; the diagnostic feeds the planned robustness/placebo table.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/exp/.matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_conflict_conditioned_climate_sensitivity import (  # noqa: E402
    ID,
    YEAR,
    add_control_by_year_terms,
    build_stacked_panel,
    coefficient_row,
    fit_panel,
)
from figure_flood_timing_and_sector_influence import (  # noqa: E402
    TARGET,
    build_compound_terms,
    renormalize_weights,
)


OUT_DIR = (
    ROOT
    / "data/exp/experiments/cambodia-thailand-village-area-itt-climate"
    / "temporal-placebo"
)
ESTIMATES_OUT = OUT_DIR / "synchronized_onset_temporal_placebo_estimates.csv"
INFERENCE_OUT = OUT_DIR / "synchronized_onset_temporal_placebo_inference.csv"
PRE_CONFLICT_ESTIMATES_OUT = OUT_DIR / "strict_pre_conflict_placebo_estimates.csv"
FIGURE_OUT = OUT_DIR / "synchronized_onset_temporal_placebo.png"
METADATA_OUT = OUT_DIR / "synchronized_onset_temporal_placebo_metadata.json"

ACTUAL_PREAH_YEAR = 2008
ACTUAL_TA_MOAN_YEAR = 2011
SECTOR_GAP_YEARS = ACTUAL_TA_MOAN_YEAR - ACTUAL_PREAH_YEAR
MINIMUM_PRE_YEARS = 3
MINIMUM_POST_YEARS = 3
PANEL_FIRST_YEAR = 2001
PANEL_LAST_YEAR = 2024
PREAH_CANDIDATE_YEARS = list(
    range(
        PANEL_FIRST_YEAR + MINIMUM_PRE_YEARS,
        PANEL_LAST_YEAR - SECTOR_GAP_YEARS - MINIMUM_POST_YEARS + 2,
    )
)
STRICT_PRE_CONFLICT_SHIFTS = [-5, -4, -3, -2]


def fit_onset_pair(stacked: pd.DataFrame, preah_year: int) -> dict[str, object]:
    ta_moan_year = preah_year + SECTOR_GAP_YEARS
    sample = stacked.copy()
    onset = sample["Conflict Sector"].map(
        {"Preah Vihear": preah_year, "Ta Moan-Ta Krabey": ta_moan_year}
    )
    if onset.isna().any():
        raise RuntimeError("Unexpected conflict-sector label")
    sample["First Conflict Year"] = onset.astype(int)
    sample["Post Conflict Period"] = sample[YEAR].ge(sample["First Conflict Year"]).astype(int)
    sample = renormalize_weights(sample)
    structural_terms, target = build_compound_terms(sample)
    if target != TARGET:
        raise RuntimeError("Temporal-placebo target differs from the frozen compound target")
    control_terms = add_control_by_year_terms(sample)
    fitted = fit_panel(sample, [*structural_terms, *control_terms])
    row = coefficient_row(
        fitted,
        target,
        Model="Synchronized-onset temporal placebo",
        Hazard="Compound hot-dry",
        **{
            "Buffer Radius km": 5,
            "Estimand": "Post-period change in affected-area compound-shock slope",
        },
    )
    row.update(
        {
            "Preah Vihear Assigned Onset": preah_year,
            "Ta Moan-Ta Krabey Assigned Onset": ta_moan_year,
            "Common Shift From Actual Years": preah_year - ACTUAL_PREAH_YEAR,
            "Assignment Type": "Actual conflict timing"
            if preah_year == ACTUAL_PREAH_YEAR
            else "Earlier alternative timing"
            if preah_year < ACTUAL_PREAH_YEAR
            else "Later alternative timing",
            "Treated Villages": int(sample.loc[sample["Area ITT"].eq(1), ID].nunique()),
            "Control Villages": int(sample.loc[sample["Area ITT"].eq(0), ID].nunique()),
        }
    )
    return row


def fit_strict_pre_conflict_placebo(stacked: pd.DataFrame, shift: int) -> dict[str, object]:
    actual_onset = stacked["Conflict Sector"].map(
        {"Preah Vihear": ACTUAL_PREAH_YEAR, "Ta Moan-Ta Krabey": ACTUAL_TA_MOAN_YEAR}
    )
    sample = stacked.loc[stacked[YEAR].lt(actual_onset)].copy()
    pseudo_onset = sample["Conflict Sector"].map(
        {
            "Preah Vihear": ACTUAL_PREAH_YEAR + shift,
            "Ta Moan-Ta Krabey": ACTUAL_TA_MOAN_YEAR + shift,
        }
    )
    sample["First Conflict Year"] = pseudo_onset.astype(int)
    sample["Post Conflict Period"] = sample[YEAR].ge(sample["First Conflict Year"]).astype(int)
    support = (
        sample.groupby(["Conflict Sector", "Post Conflict Period"])[YEAR]
        .nunique()
        .rename("Years")
    )
    if len(support) != 4 or int(support.min()) < 2:
        raise RuntimeError(f"Insufficient strict pre-conflict support for shift {shift}")
    sample = renormalize_weights(sample)
    structural_terms, target = build_compound_terms(sample)
    control_terms = add_control_by_year_terms(sample)
    fitted = fit_panel(sample, [*structural_terms, *control_terms])
    row = coefficient_row(
        fitted,
        target,
        Model="Strict pre-conflict temporal placebo",
        Hazard="Compound hot-dry",
        **{
            "Buffer Radius km": 5,
            "Estimand": "Pre-conflict pseudo-post change in affected-area compound-shock slope",
        },
    )
    row.update(
        {
            "Preah Vihear Pseudo Onset": ACTUAL_PREAH_YEAR + shift,
            "Ta Moan-Ta Krabey Pseudo Onset": ACTUAL_TA_MOAN_YEAR + shift,
            "Common Shift From Actual Years": shift,
            "Latest Included Preah Vihear Year": ACTUAL_PREAH_YEAR - 1,
            "Latest Included Ta Moan-Ta Krabey Year": ACTUAL_TA_MOAN_YEAR - 1,
            "Minimum Years Per Sector-Period Cell": int(support.min()),
            "Treated Villages": int(sample.loc[sample["Area ITT"].eq(1), ID].nunique()),
            "Control Villages": int(sample.loc[sample["Area ITT"].eq(0), ID].nunique()),
        }
    )
    return row


def build_inference(estimates: pd.DataFrame, strict: pd.DataFrame) -> pd.DataFrame:
    actual = estimates.loc[estimates["Assignment Type"].eq("Actual conflict timing")].iloc[0]
    placebo = estimates.loc[~estimates["Assignment Type"].eq("Actual conflict timing")].copy()
    actual_estimate = float(actual["Estimate"])
    return pd.DataFrame(
        [
            {
                "Diagnostic": "Full-panel synchronized timing scan",
                "Actual Estimate": actual_estimate,
                "Alternative Assignments": int(len(placebo)),
                "Alternatives No Greater Than Actual": int(placebo["Estimate"].le(actual_estimate).sum()),
                "Actual Negative-Tail Rank Including Actual": int(placebo["Estimate"].le(actual_estimate).sum()) + 1,
                "Negative Significant Strict Placebos": np.nan,
                "Strict Placebos": np.nan,
                "Interpretation": "Descriptive timing alignment only; alternative post periods may include actual post-conflict years.",
            },
            {
                "Diagnostic": "Strict pre-conflict placebo",
                "Actual Estimate": actual_estimate,
                "Alternative Assignments": np.nan,
                "Alternatives No Greater Than Actual": np.nan,
                "Actual Negative-Tail Rank Including Actual": np.nan,
                "Negative Significant Strict Placebos": int(
                    (strict["95% CI Upper"].lt(0)).sum()
                ),
                "Strict Placebos": int(len(strict)),
                "Interpretation": "All observations are censored before each sector's documented conflict onset.",
            },
        ]
    )


def draw_figure(estimates: pd.DataFrame, strict: pd.DataFrame, inference: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), constrained_layout=True)

    ordered = estimates.sort_values("Preah Vihear Assigned Onset")
    colors = np.where(
        ordered["Assignment Type"].eq("Actual conflict timing"),
        "#C4493D",
        np.where(
            ordered["Assignment Type"].eq("Earlier alternative timing"),
            "#2F6B8A",
            "#9AA3A8",
        ),
    )
    axes[0].axhline(0, color="#5A5A5A", linewidth=0.9)
    axes[0].vlines(
        ordered["Preah Vihear Assigned Onset"],
        ordered["95% CI Lower"],
        ordered["95% CI Upper"],
        color=colors,
        linewidth=1.2,
        alpha=0.9,
    )
    axes[0].scatter(
        ordered["Preah Vihear Assigned Onset"],
        ordered["Estimate"],
        c=colors,
        s=34,
        zorder=3,
    )
    axes[0].axvline(ACTUAL_PREAH_YEAR, color="#C4493D", linestyle="--", linewidth=1.0)
    axes[0].set_xticks(PREAH_CANDIDATE_YEARS[::2])
    axes[0].set_xlabel("Assigned Preah Vihear onset year\n(Ta Moan–Ta Krabey onset is three years later)")
    axes[0].set_ylabel("Compound hot–dry slope-change coefficient")
    axes[0].grid(True, color="#E3E7EA", linewidth=0.6)

    strict_ordered = strict.sort_values("Common Shift From Actual Years")
    strict_y = np.arange(len(strict_ordered))
    strict_labels = [
        f"{int(row['Preah Vihear Pseudo Onset'])}/{int(row['Ta Moan-Ta Krabey Pseudo Onset'])}"
        for _, row in strict_ordered.iterrows()
    ]
    axes[1].axvline(0, color="#5A5A5A", linewidth=0.9)
    axes[1].errorbar(
        strict_ordered["Estimate"],
        strict_y,
        xerr=[
            strict_ordered["Estimate"] - strict_ordered["95% CI Lower"],
            strict_ordered["95% CI Upper"] - strict_ordered["Estimate"],
        ],
        fmt="o",
        color="#2F6B8A",
        ecolor="#2F6B8A",
        capsize=3,
        linewidth=1.4,
    )
    axes[1].set_yticks(strict_y, strict_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Strict pre-conflict placebo coefficient")
    axes[1].set_ylabel("Pseudo onset years\n(Preah Vihear / Ta Moan–Ta Krabey)")
    axes[1].grid(True, axis="x", color="#E3E7EA", linewidth=0.6)
    axes[1].grid(False, axis="y")

    actual = ordered.loc[ordered["Assignment Type"].eq("Actual conflict timing"), "Estimate"].iloc[0]
    reference = ordered.loc[~ordered["Assignment Type"].eq("Actual conflict timing"), "Estimate"]
    sns.histplot(reference, bins=min(10, len(reference)), color="#9AA3A8", edgecolor="white", ax=axes[2])
    axes[2].axvline(actual, color="#C4493D", linewidth=2.0, label=f"Actual timing: {actual:.2f}")
    all_row = inference.loc[
        inference["Diagnostic"].eq("Full-panel synchronized timing scan")
    ].iloc[0]
    axes[2].text(
        0.98,
        0.94,
        "Descriptive negative-tail rank: "
        f"{int(all_row['Actual Negative-Tail Rank Including Actual'])}/{len(ordered)}",
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    axes[2].set_xlabel("Alternative-timing compound hot–dry coefficient")
    axes[2].set_ylabel("Number of synchronized assignments")
    axes[2].legend(frameon=False, loc="upper left", fontsize=8)
    axes[2].grid(True, axis="y", color="#E3E7EA", linewidth=0.6)
    axes[2].grid(False, axis="x")

    for label, axis in zip("abc", axes):
        axis.text(-0.12, 1.05, label, transform=axis.transAxes, fontsize=12, fontweight="bold", va="top")
    FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stacked = build_stacked_panel(5)
    rows: list[dict[str, object]] = []
    for index, preah_year in enumerate(PREAH_CANDIDATE_YEARS, start=1):
        print(
            f"Estimating synchronized onset {preah_year}/{preah_year + SECTOR_GAP_YEARS} "
            f"({index}/{len(PREAH_CANDIDATE_YEARS)})",
            flush=True,
        )
        rows.append(fit_onset_pair(stacked, preah_year))
    estimates = pd.DataFrame(rows)
    estimates.to_csv(ESTIMATES_OUT, index=False)
    strict_rows: list[dict[str, object]] = []
    for index, shift in enumerate(STRICT_PRE_CONFLICT_SHIFTS, start=1):
        print(
            f"Estimating strict pre-conflict placebo shift {shift} "
            f"({index}/{len(STRICT_PRE_CONFLICT_SHIFTS)})",
            flush=True,
        )
        strict_rows.append(fit_strict_pre_conflict_placebo(stacked, shift))
    strict = pd.DataFrame(strict_rows)
    strict.to_csv(PRE_CONFLICT_ESTIMATES_OUT, index=False)
    inference = build_inference(estimates, strict)
    inference.to_csv(INFERENCE_OUT, index=False)
    draw_figure(estimates, strict, inference)
    metadata = {
        "status": "completed exact synchronized-onset temporal placebo",
        "human_decision_record": "MILI-D-20260823-011",
        "actual_sector_years": {
            "Preah Vihear": ACTUAL_PREAH_YEAR,
            "Ta Moan-Ta Krabey": ACTUAL_TA_MOAN_YEAR,
        },
        "assignment_rule": (
            "Enumerate every synchronized annual onset pair that preserves the observed "
            "three-year sector gap and at least three pre and three post years per sector"
        ),
        "candidate_preah_years": PREAH_CANDIDATE_YEARS,
        "unique_joint_assignments": len(PREAH_CANDIDATE_YEARS),
        "repeat_sampling_used": False,
        "primary_buffer_km": 5,
        "villages_and_calibration_weights": "identical across actual and alternative timings",
        "model": (
            "full drought, heat, and compound lower-order hierarchy; stack-village and "
            "stack-year fixed effects; four predetermined controls interacted flexibly "
            "with year; 10 km spatial-block clustered covariance"
        ),
        "timing_scan_interpretation": (
            "descriptive only because alternative post periods can include documented "
            "post-conflict years; no randomization p-value is assigned"
        ),
        "strict_pre_conflict_placebo": {
            "common_shifts": STRICT_PRE_CONFLICT_SHIFTS,
            "censoring": "each sector is censored immediately before its documented onset",
            "minimum_period_support": "at least two years in every sector-by-pseudo-period cell",
            "automatic_pass_rule": None,
        },
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    METADATA_OUT.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {ESTIMATES_OUT.relative_to(ROOT)}")
    print(f"Saved: {PRE_CONFLICT_ESTIMATES_OUT.relative_to(ROOT)}")
    print(f"Saved: {INFERENCE_OUT.relative_to(ROOT)}")
    print(f"Saved: {FIGURE_OUT.relative_to(ROOT)}")
    print("\nTemporal-placebo inference")
    print(inference.to_string(index=False))


if __name__ == "__main__":
    main()
