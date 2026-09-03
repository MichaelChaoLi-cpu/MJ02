#!/usr/bin/env python3
"""Climate-shock distributions and correlations.

Plan: document the natural-unit support of the prespecified stage-1 climate
exposures and the dependence among them before interpreting coefficients.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/cses_public_village_cropland_npp_absolute_climate_panel_preprocessed.parquet"
OUTPUT = ROOT / "data/results/figures/Figure_climate_shock_distributions_and_correlations.png"
EVIDENCE = ROOT / "data/exp/analysis/climate-npp/climate-shock-distributions-and-correlations"

HEAT_DAYS = "Village Buffer Mean Annual Heat Days at or Above 35 C"
HEAT_DD = "Village Buffer Mean Annual Heat Degree-Days Above 35 C"
RX5DAY = "Village Buffer Mean Annual Maximum Consecutive Five-Day Precipitation Rx5day mm"
DRY_DAYS = "Village Buffer Mean Annual Maximum Consecutive Dry Days Below 1 mm"
RAIN_TOTAL = "Village Buffer Mean Annual Precipitation Total mm"

VARIABLES = [HEAT_DAYS, HEAT_DD, RX5DAY, DRY_DAYS, RAIN_TOTAL]
SHORT_LABELS = {
    HEAT_DAYS: "Heat days\n≥35°C",
    HEAT_DD: "Degree-days\n>35°C",
    RX5DAY: "Rx5day",
    DRY_DAYS: "Maximum\ndry spell",
    RAIN_TOTAL: "Annual\nrainfall",
}

DARK = "#40505A"
GRID = "#DDE3E5"
CORRELATION_CMAP = LinearSegmentedColormap.from_list(
    "blue_green_white_yellow_red",
    ["#2166AC", "#1A9850", "#FFFFFF", "#FEE08B", "#D73027"],
    N=256,
)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.012,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111111",
        zorder=10,
    )


def title_inside(ax: plt.Axes, title: str) -> None:
    ax.text(
        0.985,
        0.97,
        title,
        transform=ax.transAxes,
        fontsize=10.2,
        fontweight="bold",
        ha="right",
        va="top",
        color="#173C5C",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.6},
        zorder=10,
    )


def style_distribution_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=0.55, linestyle="--", alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.2, colors=DARK, length=2.5)
    for spine in ax.spines.values():
        spine.set_color("#89969C")
        spine.set_linewidth(0.65)


def histogram(
    ax: plt.Axes,
    values: pd.Series,
    xlabel: str,
    title: str,
    panel: str,
    color: str,
) -> None:
    clean = values.dropna().to_numpy(float)
    upper = np.quantile(clean, 0.995)
    shown = clean[clean <= upper]
    ax.hist(shown, bins=35, density=True, color=color, alpha=0.82, edgecolor="white", linewidth=0.35)
    median = float(np.median(clean))
    ax.axvline(median, color="#27343A", linewidth=1.15, linestyle="--")
    ax.text(
        median,
        ax.get_ylim()[1] * 0.77,
        f"Median {median:,.1f}",
        rotation=90,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#27343A",
    )
    ax.set_xlim(left=0, right=upper)
    ax.set_xlabel(xlabel, fontsize=8.7, color=DARK)
    ax.set_ylabel("Density", fontsize=8.7, color=DARK)
    panel_label(ax, panel)
    title_inside(ax, title)
    style_distribution_axis(ax)


def main() -> None:
    frame = pd.read_parquet(INPUT, columns=VARIABLES).dropna(subset=VARIABLES).copy()

    evidence_stats = frame[VARIABLES].describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    ).T
    correlation = frame[VARIABLES].corr(method="pearson")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    evidence_stats.to_csv(EVIDENCE / "climate_shock_natural_unit_distributions.csv")
    correlation.to_csv(EVIDENCE / "climate_shock_pearson_correlations.csv")

    heat_correlation = float(correlation.loc[HEAT_DAYS, HEAT_DD])
    summary = {
        "complete_village_year_observations": int(len(frame)),
        "heat_days_degree_days_pearson_correlation": heat_correlation,
        "distribution_plot_upper_limit": "99.5th percentile; all observations retained in evidence tables and correlations",
    }
    (EVIDENCE / "climate_shock_distributions_and_correlations_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    fig, axes = plt.subplot_mosaic(
        [["a", "b", "e", "e"], ["c", "d", "e", "e"]],
        figsize=(15.2, 7.9),
        facecolor="white",
        gridspec_kw={
            "left": 0.06,
            "right": 0.965,
            "top": 0.965,
            "bottom": 0.095,
            "wspace": 0.27,
            "hspace": 0.25,
        },
    )
    ax_a = axes["a"]
    ax_b = axes["b"]
    ax_c = axes["c"]
    ax_d = axes["d"]
    ax_e = axes["e"]

    heat_x = frame[HEAT_DAYS].to_numpy(float)
    heat_y = frame[HEAT_DD].to_numpy(float)
    heat_plot = ax_a.hexbin(
        heat_x,
        heat_y,
        gridsize=43,
        mincnt=1,
        bins="log",
        cmap="YlGnBu",
        linewidths=0,
    )
    ax_a.set_xlabel("Annual days at or above 35°C", fontsize=8.7, color=DARK)
    ax_a.set_ylabel("Annual degree-days above 35°C", fontsize=8.7, color=DARK)
    panel_label(ax_a, "a")
    title_inside(ax_a, f"Heat measures (r = {heat_correlation:.2f})")
    style_distribution_axis(ax_a)
    heat_bar = fig.colorbar(heat_plot, ax=ax_a, orientation="vertical", pad=0.018, shrink=0.72, aspect=24)
    heat_bar.set_label("Village-year density (log)", fontsize=7.7, color=DARK, labelpad=4)
    heat_bar.ax.tick_params(labelsize=7.2, colors=DARK, length=2)
    heat_bar.outline.set_linewidth(0.45)

    histogram(ax_b, frame[RX5DAY], "Annual Rx5day (mm)", "Extreme five-day rainfall", "b", "#377EB8")
    histogram(ax_c, frame[DRY_DAYS], "Maximum consecutive dry days", "Annual dry-spell length", "c", "#2A9D63")
    histogram(ax_d, frame[RAIN_TOTAL], "Annual precipitation total (mm)", "Annual rainfall", "d", "#E4A72D")

    matrix = correlation.to_numpy(float)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    image = ax_e.imshow(matrix, cmap=CORRELATION_CMAP, norm=norm, interpolation="nearest")
    labels = [SHORT_LABELS[column] for column in VARIABLES]
    positions = np.arange(len(labels))
    ax_e.set_xticks(positions, labels=labels, fontsize=8.0, rotation=38, ha="right", color=DARK)
    ax_e.set_yticks(positions, labels=labels, fontsize=8.0, color=DARK)
    ax_e.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax_e.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax_e.grid(which="minor", color="white", linewidth=1.8)
    ax_e.tick_params(which="minor", bottom=False, left=False)
    ax_e.tick_params(which="major", length=0)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix[row, column]
            text_color = "white" if abs(value) >= 0.63 else "#263238"
            ax_e.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8.2, color=text_color)
    panel_label(ax_e, "e")
    title_inside(ax_e, "Pearson correlations")
    correlation_bar = fig.colorbar(image, ax=ax_e, orientation="horizontal", pad=0.11, shrink=0.84, aspect=32)
    correlation_bar.set_label("Correlation coefficient", fontsize=8.0, color=DARK, labelpad=4)
    correlation_bar.ax.tick_params(labelsize=7.4, colors=DARK, length=2)
    correlation_bar.outline.set_linewidth(0.45)
    for spine in ax_e.spines.values():
        spine.set_color("#89969C")
        spine.set_linewidth(0.65)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Complete observations: {len(frame):,}")
    print(f"Heat-days / degree-days correlation: {heat_correlation:.4f}")
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved evidence: {EVIDENCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
