#!/usr/bin/env python3
"""Monsoon Definition and Exposure Support.

Plan: Document the single approved onset timing, dry-spell support, absolute heat at
33/35/37 C, and national temporal variation before outcome interpretation.
Framework: AnaSOP outcome-blind climate-definition and exposure-support audit.
All displayed exposure units are natural days or calendar day of year.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/cambodia_national_monsoon_timing_preprocessed.parquet"
ANALYSIS_DIR = ROOT / "data/exp/analysis/climate-welfare/monsoon-definition-exposure-support"
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/figures/Figure_monsoon_definition_and_exposure_support.png"

BLUE = "#2a6f97"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
GOLD = "#e9c46a"
GRAY = "#7a858c"

ONSET_B = "Wet-Season Onset DOY Candidate B"
DRY_B = "Longest Intraseasonal Dry Spell Days Candidate B"
HEAT_33 = "Post-Onset Absolute Heat Day Count 33 C Candidate B"
HEAT_35 = "Post-Onset Absolute Heat Day Count 35 C Candidate B"
HEAT_37 = "Post-Onset Absolute Heat Day Count 37 C Candidate B"


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.07, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    observed = np.sort(values.dropna().to_numpy(float))
    probability = np.arange(1, len(observed) + 1) / len(observed)
    return observed, probability


def main() -> None:
    columns = ["Climate Cell ID", "Year", ONSET_B, DRY_B, HEAT_33, HEAT_35, HEAT_37]
    frame = pd.read_parquet(INPUT, columns=columns)
    frame = frame.loc[frame["Year"].between(1991, 2024)].copy()
    annual = (
        frame.groupby("Year", observed=True)
        .agg(
            onset_b=(ONSET_B, "mean"),
            heat_35_mean=(HEAT_35, "mean"),
            heat_35_p10=(HEAT_35, lambda x: x.quantile(0.10)),
            heat_35_p90=(HEAT_35, lambda x: x.quantile(0.90)),
            valid_cells=("Climate Cell ID", "nunique"),
        )
        .reset_index()
    )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    annual.to_csv(ANALYSIS_DIR / "national_annual_exposure_support.csv", index=False)
    summary_rows = []
    for variable in [ONSET_B, DRY_B, HEAT_33, HEAT_35, HEAT_37]:
        values = frame[variable].dropna()
        summary_rows.append(
            {
                "Variable": variable,
                "Observed Rows": len(values),
                "Observed Share": len(values) / len(frame),
                "Mean": values.mean(),
                "SD": values.std(),
                "P10": values.quantile(0.10),
                "Median": values.median(),
                "P90": values.quantile(0.90),
            }
        )
    pd.DataFrame(summary_rows).to_csv(ANALYSIS_DIR / "exposure_distribution_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)

    # a: annual onset timing under the single frozen definition
    ax = axes[0, 0]
    ax.plot(annual["Year"], annual["onset_b"], color=ORANGE, linewidth=1.8,
            marker="o", markersize=3.3, label="Approved rule: 30 mm / 3 days")
    ax.set_ylabel("National mean onset day of year")
    ax.set_xlabel("Year")
    ax.set_xlim(1991, 2024)
    ax.grid(color="#d9dde0", linewidth=0.6, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    add_panel_label(ax, "a")

    # b: dry-spell distributions in natural days
    ax = axes[0, 1]
    upper_dry = int(np.ceil(frame[DRY_B].quantile(0.995)))
    bins = np.arange(0, upper_dry + 2, 1)
    ax.hist(frame[DRY_B].dropna(), bins=bins, density=True, histtype="step",
            linewidth=1.8, color=ORANGE, label="Approved rule")
    ax.set_xlim(0, upper_dry)
    ax.set_xlabel("Longest intraseasonal dry spell (days)")
    ax.set_ylabel("Density")
    ax.grid(axis="y", color="#d9dde0", linewidth=0.6, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    add_panel_label(ax, "b")

    # c: threshold-specific absolute heat exposure
    ax = axes[1, 0]
    for variable, label, color in [
        (HEAT_33, "≥33°C", GOLD),
        (HEAT_35, "≥35°C", ORANGE),
        (HEAT_37, "≥37°C", "#9d174d"),
    ]:
        x, y = ecdf(frame[variable])
        ax.plot(x, y, linewidth=1.9, color=color, label=label)
    upper_heat = float(frame[HEAT_33].quantile(0.995))
    ax.set_xlim(0, upper_heat)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Post-onset absolute heat days")
    ax.set_ylabel("Cumulative share")
    ax.grid(color="#d9dde0", linewidth=0.6, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    add_panel_label(ax, "c")

    # d: national temporal variation in the primary 35 C definition
    ax = axes[1, 1]
    ax.fill_between(
        annual["Year"], annual["heat_35_p10"], annual["heat_35_p90"],
        color="#f4b49f", alpha=0.40, linewidth=0, label="Cell P10–P90",
    )
    ax.plot(annual["Year"], annual["heat_35_mean"], color=ORANGE,
            linewidth=2.0, marker="o", markersize=3.3, label="National mean")
    ax.set_xlim(1991, 2024)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("Post-onset days ≥35°C")
    ax.grid(color="#d9dde0", linewidth=0.6, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    add_panel_label(ax, "d")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
