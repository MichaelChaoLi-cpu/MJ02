#!/usr/bin/env python3
"""Absolute Heat Exposure and Household Food Consumption.

Plan: Show the full-sample estimate, same-sample village decomposition,
commune pseudo-panel estimates, and temporal validation.
Framework: AnaSOP Model B household and commune pseudo-panel specifications
with survey weights and 0.75-degree spatial-block-clustered uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LEGACY_INPUT = ROOT / "data/exp/analysis/climate-welfare/cses-absolute-heat-food-validation/coefficients.csv"
REPAIR_INPUT = ROOT / "data/exp/analysis/climate-welfare/household-heat-identification-repair/coefficients.csv"
OUTPUT = ROOT / "data/exp/internal_output_archive/obsolete-formal-results-20260824/figures/Figure_absolute_heat_exposure_and_household_food_consumption.png"
HEAT = "Absolute Heat Days per 10"


def percent(value: pd.Series | float) -> pd.Series | float:
    return 100.0 * (np.exp(value) - 1.0)


def rows_for(frame: pd.DataFrame, labels: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for label, specification, exposure in labels:
        selected = frame.loc[
            frame["Specification"].eq(specification) & frame["Exposure"].eq(exposure)
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one row for {specification!r}, {exposure!r}; found {len(selected)}")
        row = selected.iloc[0].copy()
        row["Label"] = label
        rows.append(row)
    return pd.DataFrame(rows)


def forest(ax: plt.Axes, frame: pd.DataFrame, colors: list[str] | str) -> None:
    work = frame.iloc[::-1].reset_index(drop=True)
    estimate = percent(work["Coefficient Log Points"].astype(float)).to_numpy()
    lower = percent(work["95 Percent CI Lower"].astype(float)).to_numpy()
    upper = percent(work["95 Percent CI Upper"].astype(float)).to_numpy()
    palette = [colors] * len(work) if isinstance(colors, str) else list(reversed(colors))
    y = np.arange(len(work))
    ax.axvline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    for index in range(len(work)):
        ax.errorbar(
            estimate[index], y[index],
            xerr=[[estimate[index] - lower[index]], [upper[index] - estimate[index]]],
            fmt="o", color=palette[index], ecolor=palette[index], markersize=5,
            elinewidth=1.4, capsize=3, zorder=2,
        )
    ax.set_yticks(y, work["Label"])
    ax.set_xlabel("Food-consumption change per 10 heat days (%)")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    legacy = pd.read_csv(LEGACY_INPUT)
    repair = pd.read_csv(REPAIR_INPUT)
    primary = rows_for(
        repair,
        [
            ("All linked, district FE", "All linked households, district FE", HEAT),
            ("All linked, commune FE", "All linked households, commune FE", HEAT),
        ],
    )
    sample_structure = rows_for(
        repair,
        [
            ("Repeated villages, district FE", "Repeated-village sample, district FE", HEAT),
            ("Repeated villages, village FE", "Repeated-village sample, village FE", HEAT),
            ("Single-wave villages, district FE", "Single-wave-village sample, district FE", HEAT),
            (
                "Repeated minus single-wave difference",
                "Formal repeated-versus-single-wave heat contrast",
                "Heat X Repeated Village",
            ),
        ],
    )
    pseudo = rows_for(
        repair,
        [
            ("Minimum 5 households", "Commune pseudo-panel, minimum 5 households", HEAT),
            ("Minimum 10 households", "Commune pseudo-panel, minimum 10 households", HEAT),
            (
                "+ composition",
                "Commune pseudo-panel, minimum 5 households, composition adjusted",
                HEAT,
            ),
        ],
    )
    wave_labels = []
    for year in (2007, 2009, 2011, 2013, 2014, 2016, 2017, 2019, 2021):
        wave_labels.append(
            (f"Exclude {year}", f"Candidate B, 35 C, 5 km, excluding CSES {year}", HEAT)
        )
    wave_labels.extend(
        [
            ("Current, placebo sample", "Current plus future season, clean interview months", HEAT),
            ("Future season", "Current plus future season, clean interview months", f"Future {HEAT}"),
        ]
    )
    timing = rows_for(legacy, wave_labels)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2), constrained_layout=True)
    forest(axes[0, 0], primary, ["#335c81", "#006d5b"])
    axes[0, 0].set_xlim(-5.2, 1.2)
    estimate = float(percent(primary.iloc[1]["Coefficient Log Points"]))
    axes[0, 0].annotate(
        f"{estimate:.2f}%", xy=(estimate, 0), xytext=(0, 18),
        textcoords="offset points", ha="center", va="bottom",
        fontsize=10, color="#006d5b", fontweight="bold",
    )
    forest(
        axes[0, 1], sample_structure,
        ["#335c81", "#c06c2b", "#335c81", "#888888"],
    )
    axes[0, 1].text(
        0.98, 0.92, "Formal difference: p = 0.734",
        transform=axes[0, 1].transAxes, ha="right", va="top",
        fontsize=8.8, color="#555555",
    )
    forest(axes[1, 0], pseudo, ["#006d5b", "#335c81", "#c06c2b"])
    forest(axes[1, 1], timing, ["#335c81"] * 10 + ["#888888"])

    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
