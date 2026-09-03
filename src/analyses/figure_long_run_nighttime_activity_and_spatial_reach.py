#!/usr/bin/env python3
"""Long-Run Nighttime Activity and Spatial Reach.

Plan: Show full-period, source-stage, cumulative-bandwidth, and mirrored-strip estimates.
Framework: AnaSOP Sections 5.5, 6.10, and 7 long-run nighttime validation workflow.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from longntl_boundary_estimation import ROOT, estimate_all


OUTPUT = ROOT / "data/exp/legacy-results/figures/Figure_long_run_nighttime_activity_and_spatial_reach.png"
COLOR_PRIMARY = "#8B1E3F"
COLOR_CONFIRM = "#1F5A7A"
COLOR_SECONDARY = "#6B7785"
SESOI = 0.20


def forest(axis: plt.Axes, data: pd.DataFrame, labels: list[str]) -> None:
    y = np.arange(len(data))[::-1]
    for position, (_, row) in zip(y, data.iterrows()):
        color = COLOR_CONFIRM if row["confirmation_model"] else COLOR_PRIMARY
        axis.errorbar(
            row["common_standardized_estimate"],
            position,
            xerr=[[row["common_standardized_estimate"] - row["common_standardized_ci_low"]],
                  [row["common_standardized_ci_high"] - row["common_standardized_estimate"]]],
            fmt="D" if row["confirmation_model"] else "o",
            color=color,
            ecolor=color,
            capsize=2.5,
            markersize=4.8,
            linewidth=1.1,
        )
    axis.set_yticks(y, labels, fontsize=8)


def main() -> None:
    results = estimate_all()
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), constrained_layout=True)
    for axis in axes[0, :]:
        axis.axvspan(-SESOI, SESOI, color="#EDF2E8", alpha=0.8, zorder=0)
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.axvline(-SESOI, color="#85947C", linewidth=0.7, linestyle="--")
        axis.axvline(SESOI, color="#85947C", linewidth=0.7, linestyle="--")
        axis.grid(axis="x", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[1, :]:
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    names = [
        "Full period primary 5 km",
        "Full period within-commune 5 km",
        "Full period triangular 5 km",
        "Annual rainfall alternative 5 km",
        "Any nonzero LongNTL primary 5 km",
        "Any nonzero LongNTL within-commune 5 km",
    ]
    data = results.set_index("specification").loc[names].reset_index()
    forest(axes[0, 0], data, ["Primary", "Within commune", "Triangular weights",
                              "Annual rainfall", "Any nonzero", "Any nonzero, commune"])
    axes[0, 0].set_xlabel("Southwest - West response (within-outcome SD)", fontsize=9)

    names = [
        "Reconstructed-period primary 5 km",
        "Reconstructed-period within-commune 5 km",
        "Observed-composite-period primary 5 km",
        "Observed-composite-period within-commune 5 km",
        "Observed VIIRS benchmark: Primary 5 km",
        "Observed VIIRS benchmark: Within-commune confirmation 5 km",
    ]
    labels = ["LongNTL 2000-12", "LongNTL 2000-12, commune", "LongNTL 2013-24",
              "LongNTL 2013-24, commune", "Observed VIIRS", "Observed VIIRS, commune"]
    data = results.set_index("specification").loc[names].reset_index()
    forest(axes[0, 1], data, labels)
    axes[0, 1].set_xlabel("Southwest - West response (within-outcome SD)", fontsize=9)

    names = ["Fixed cumulative bandwidth 2 km", "Full period primary 5 km"] + [
        f"Fixed cumulative bandwidth {bandwidth} km" for bandwidth in (10, 15, 20, 30)
    ]
    data = results.set_index("specification").loc[names].reset_index()
    x = np.array([2, 5, 10, 15, 20, 30])
    y = data["common_standardized_estimate"].to_numpy()
    low = data["common_standardized_ci_low"].to_numpy()
    high = data["common_standardized_ci_high"].to_numpy()
    axes[1, 0].errorbar(x, y, yerr=[y - low, high - y], fmt="o-", color=COLOR_SECONDARY,
                        capsize=2.5, linewidth=1.2, markersize=4.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xlabel("Cumulative distance from boundary (km)", fontsize=9)
    axes[1, 0].set_ylabel("Southwest - West response (within-cell SD)", fontsize=9)
    axes[1, 0].grid(axis="y", alpha=0.22)

    names = [f"Mirrored strip {low}-{high} km" for low, high in ((0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30))]
    data = results.set_index("specification").loc[names].reset_index()
    x = np.arange(len(data))
    y = data["common_standardized_estimate"].to_numpy()
    low = data["common_standardized_ci_low"].to_numpy()
    high = data["common_standardized_ci_high"].to_numpy()
    axes[1, 1].errorbar(x, y, yerr=[y - low, high - y], fmt="o-", color=COLOR_SECONDARY,
                        capsize=2.5, linewidth=1.2, markersize=4.8)
    axes[1, 1].set_xticks(x, ["0-2", "2-5", "5-10", "10-15", "15-20", "20-30"], rotation=25)
    axes[1, 1].set_xlabel("Non-overlapping mirrored strip (km)", fontsize=9)
    axes[1, 1].set_ylabel("Descriptive response contrast (within-cell SD)", fontsize=9)
    axes[1, 1].grid(axis="y", alpha=0.22)

    for label, axis in zip("abcd", axes.flat):
        panel_label = axis.text(-0.09, 1.03, label, transform=axis.transAxes, fontsize=12,
                                fontweight="bold", va="top", ha="right")
        panel_label.set_in_layout(False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
