#!/usr/bin/env python3
"""Nighttime Activity Independent Validation.

Plan: Display the observed-VIIRS historical-boundary rainfall-response estimate,
the mandatory within-modern-commune confirmation, frozen outcome, coverage,
weighting, rainfall, and bandwidth checks, and the pre-specified SESOI.
Framework: AnaSOP Sections 5.5, 6.10, and the nighttime-activity independent-
validation workflow in Section 7.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from experiment_viirs_historical_boundary_shock_response import (
    prepare_panel,
    run_experiment,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    ROOT
    / "data/results/figures/Figure_nighttime_activity_independent_validation.png"
)
SESOI = 0.20

CORE_ORDER = [
    "Primary 5 km",
    "Within-commune confirmation 5 km",
    "Annual rainfall alternative 5 km",
    "Annual median radiance 5 km",
    "Any nonzero radiance 5 km",
    "At least 40 cloud-free observations 5 km",
    "Triangular distance weights 5 km",
]

CORE_LABELS = {
    "Primary 5 km": "Primary: mean radiance",
    "Within-commune confirmation 5 km": "Within-commune confirmation",
    "Annual rainfall alternative 5 km": "Annual rainfall",
    "Annual median radiance 5 km": "Median radiance",
    "Any nonzero radiance 5 km": "Any nonzero radiance",
    "At least 40 cloud-free observations 5 km": "≥40 cloud-free observations",
    "Triangular distance weights 5 km": "Triangular distance weights",
}

BANDWIDTH_ORDER = [
    "Fixed bandwidth 2 km",
    "Primary 5 km",
    "Fixed bandwidth 10 km",
    "Fixed bandwidth 15 km",
    "Fixed bandwidth 20 km",
    "Fixed bandwidth 30 km",
]

BANDWIDTH_LABELS = {
    "Fixed bandwidth 2 km": "2 km",
    "Primary 5 km": "5 km (primary)",
    "Fixed bandwidth 10 km": "10 km",
    "Fixed bandwidth 15 km": "15 km",
    "Fixed bandwidth 20 km": "20 km",
    "Fixed bandwidth 30 km": "30 km",
}


def ordered_rows(
    estimates: pd.DataFrame,
    order: list[str],
    labels: dict[str, str],
) -> pd.DataFrame:
    indexed = estimates.set_index("specification", drop=False)
    missing = [name for name in order if name not in indexed.index]
    if missing:
        raise ValueError(f"Missing frozen specifications: {missing}")
    selected = indexed.loc[order].copy().reset_index(drop=True)
    selected["display_label"] = selected["specification"].map(labels)
    return selected


def style_axis(axis: plt.Axes) -> None:
    axis.axvspan(-SESOI, SESOI, color="#E8EFE4", alpha=0.92, zorder=0)
    axis.axvline(0, color="#2D2D2D", linewidth=0.95, zorder=1)
    axis.axvline(-SESOI, color="#71896A", linewidth=0.85, linestyle="--", zorder=1)
    axis.axvline(SESOI, color="#71896A", linewidth=0.85, linestyle="--", zorder=1)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.55, zorder=0)
    axis.set_xlim(-0.23, 0.23)
    axis.set_xticks(np.arange(-0.2, 0.201, 0.1))
    axis.tick_params(axis="both", labelsize=8.5)
    for spine in axis.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.9)


def forest_panel(
    axis: plt.Axes,
    rows: pd.DataFrame,
    *,
    primary_highlight: bool,
) -> None:
    display = rows.iloc[::-1].reset_index(drop=True)
    positions = np.arange(len(display))
    for position, row in display.iterrows():
        is_primary = row["specification"] == "Primary 5 km"
        is_confirmation = bool(row["confirmation_model"])
        if is_primary and primary_highlight:
            color, marker, size = "#C94C35", "o", 6.2
        elif is_confirmation:
            color, marker, size = "#111111", "D", 5.6
        else:
            color, marker, size = "#3C78B5", "o", 5.2
        axis.errorbar(
            row["standardized_estimate"],
            position,
            xerr=[
                [row["standardized_estimate"] - row["standardized_ci_low"]],
                [row["standardized_ci_high"] - row["standardized_estimate"]],
            ],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.25,
            capsize=2.7,
            markersize=size,
            zorder=3,
        )
    axis.set_yticks(positions, display["display_label"], fontsize=8.5)
    axis.set_xlabel(
        "Southwest − West rainfall response\n"
        "(within-cell outcome SD per 1-SD shock)",
        fontsize=9,
    )
    style_axis(axis)


def main() -> None:
    estimates = run_experiment(prepare_panel())
    core = ordered_rows(estimates, CORE_ORDER, CORE_LABELS)
    bandwidth = ordered_rows(estimates, BANDWIDTH_ORDER, BANDWIDTH_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4))
    forest_panel(axes[0], core, primary_highlight=True)
    forest_panel(axes[1], bandwidth, primary_highlight=True)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="#C94C35",
                linewidth=1.25,
                markersize=5.5,
                label="Primary",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="#111111",
                linewidth=1.25,
                markersize=5.0,
                label="Mandatory confirmation",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="#3C78B5",
                linewidth=1.25,
                markersize=5.0,
                label="Pre-specified check",
            ),
            Patch(facecolor="#E8EFE4", edgecolor="none", label="±0.20 SD SESOI"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.55, 0.995),
        frameon=True,
        fontsize=7.6,
        ncol=4,
    )

    for label, axis in zip("ab", axes, strict=True):
        axis.text(
            -0.12,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.90,
        bottom=0.13,
        wspace=0.43,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    primary = estimates.loc[estimates["specification"].eq("Primary 5 km")].iloc[0]
    confirmation = estimates.loc[
        estimates["specification"].eq("Within-commune confirmation 5 km")
    ].iloc[0]
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(
        "Primary standardized interaction: "
        f"{primary['standardized_estimate']:.4f} "
        f"[{primary['standardized_ci_low']:.4f}, {primary['standardized_ci_high']:.4f}]"
    )
    print(
        "Confirmation standardized interaction: "
        f"{confirmation['standardized_estimate']:.4f} "
        f"[{confirmation['standardized_ci_low']:.4f}, "
        f"{confirmation['standardized_ci_high']:.4f}]"
    )
    print(
        "SESOI audit: "
        f"{int(estimates['sesoi_classification'].eq('substantively precise null').sum())} "
        f"of {len(estimates)} frozen specifications are inside ±{SESOI:.2f}."
    )


if __name__ == "__main__":
    main()
