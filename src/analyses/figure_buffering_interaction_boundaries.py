#!/usr/bin/env python3
"""Buffering Interaction Boundaries.

Plan: Show absolute-heat food-consumption responses at low and high levels of
three prespecified modifiers, alongside the multiplicity-adjusted interaction
test. These are heterogeneity boundaries, not intervention effects.
Framework: AnaSOP Appendix buffering model with Candidate B, 35 C, 5 km heat,
survey weights, district and wave-by-month effects, spatial-block clustering,
and Holm correction across three interactions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data/exp/analysis/climate-welfare/absolute-heat-food-buffering"
INTERACTIONS = INPUT_DIR / "buffer_interactions.csv"
MARGINALS = INPUT_DIR / "buffer_marginal_effects.csv"
OUTPUT = ROOT / "data/exp/legacy-results/figures/Figure_buffering_interaction_boundaries.png"

BLUE = "#2a6f97"
TEAL = "#2a9d8f"

PANELS = [
    (
        "Irrigation",
        "Irrigable land availability",
        {
            "No irrigable parcel": "No irrigable parcel",
            "Any irrigable parcel": "Any irrigable parcel",
        },
    ),
    (
        "Historical Road Access",
        "Historical road access",
        {
            "Village 25th percentile": "Better access\n(lower road distance)",
            "Village 75th percentile": "Poorer access\n(higher road distance)",
        },
    ),
    (
        "Baseline Settlement Connectivity",
        "Baseline settlement connectivity",
        {
            "Village 25th percentile": "Lower connectivity\n(25th percentile)",
            "Village 75th percentile": "Higher connectivity\n(75th percentile)",
        },
    ),
]


def panel_header(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")
    ax.text(0.0, 1.08, title, transform=ax.transAxes, fontsize=10.5,
            va="top", ha="left")


def main() -> None:
    interactions = pd.read_csv(INTERACTIONS)
    marginals = pd.read_csv(MARGINALS)

    plot_rows = []
    for buffer, title, label_map in PANELS:
        rows = marginals.loc[marginals["Buffer"].eq(buffer)].copy()
        rows["Display Label"] = rows["Modifier Level"].map(label_map)
        if rows["Display Label"].isna().any() or len(rows) != 2:
            raise ValueError(f"Unexpected marginal-effect rows for {buffer}")
        plot_rows.append(rows)
    figure_values = pd.concat(plot_rows, ignore_index=True)
    figure_values.to_csv(INPUT_DIR / "buffering_figure_values.csv", index=False)

    global_low = float(figure_values["95 Percent CI Lower Percent"].min())
    global_high = float(figure_values["95 Percent CI Upper Percent"].max())
    span = global_high - global_low
    x_limits = (global_low - 0.08 * span, global_high + 0.12 * span)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.2), constrained_layout=True, sharex=True)

    for index, (ax, (buffer, title, _), rows) in enumerate(zip(axes, PANELS, plot_rows)):
        work = rows.reset_index(drop=True)
        y = np.array([1.0, 0.0])
        estimates = work["Absolute Heat Effect Percent"].to_numpy(float)
        lower = work["95 Percent CI Lower Percent"].to_numpy(float)
        upper = work["95 Percent CI Upper Percent"].to_numpy(float)
        ax.axvline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
        for row_index, color in enumerate([BLUE, TEAL]):
            ax.errorbar(
                estimates[row_index], y[row_index],
                xerr=[
                    [estimates[row_index] - lower[row_index]],
                    [upper[row_index] - estimates[row_index]],
                ],
                fmt="o", color=color, ecolor=color, markersize=6,
                elinewidth=1.6, capsize=4, zorder=2,
            )
            ax.text(
                estimates[row_index], y[row_index] + 0.17,
                f"{estimates[row_index]:.2f}%", ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold",
            )
        interaction = interactions.loc[interactions["Buffer"].eq(buffer)]
        if len(interaction) != 1:
            raise ValueError(f"Expected one interaction row for {buffer}")
        interaction = interaction.iloc[0]
        ax.text(
            0.02, 0.965,
            (
                f"Interaction β = {interaction['Coefficient Log Points']:+.3f}\n"
                f"95% CI [{interaction['95 Percent CI Lower']:+.3f}, "
                f"{interaction['95 Percent CI Upper']:+.3f}]\n"
                f"Holm p = {interaction['Holm Adjusted Probability Value']:.3f}"
            ),
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#d5dade", "linewidth": 0.6,
                  "boxstyle": "round,pad=0.35", "alpha": 0.92},
        )
        ax.set_yticks(y, work["Display Label"])
        ax.set_ylim(-0.55, 1.75)
        ax.set_xlim(*x_limits)
        ax.set_xlabel("Food-consumption change per 10 heat days (%)")
        ax.grid(axis="x", color="#d9dde0", linewidth=0.65, linestyle="--")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel_header(ax, chr(ord("a") + index), title)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(interactions[[
        "Buffer", "Coefficient Log Points", "95 Percent CI Lower", "95 Percent CI Upper",
        "Raw Probability Value", "Holm Adjusted Probability Value",
        "Holm Significant at 0.05", "Observations", "Villages",
    ]].to_string(index=False))
    print(figure_values[[
        "Buffer", "Display Label", "Absolute Heat Effect Percent",
        "95 Percent CI Lower Percent", "95 Percent CI Upper Percent",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
