#!/usr/bin/env python3
"""Plot a non-promoted prototype of the post-monsoon vegetation response."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data/exp/analysis/climate-welfare/postmonsoon-vegetation-validation"
COEFFICIENTS = INPUT_DIR / "coefficients.csv"
CROSSFIT = INPUT_DIR / "crossfit_model_comparison.csv"
OUTPUT = ROOT / "data/exp/figure-previews/Figure_intraseasonal_dry_spells_and_post_monsoon_vegetation_loss.png"

EVI = "Village Buffer Mean November-February Mean EVI Anomaly Z"
NDVI = "Village Buffer Mean November-February Mean NDVI Anomaly Z"
DRY_SPELL = "Longest Intraseasonal Dry Spell Days"
COLORS = {"EVI": "#2F7D5B", "NDVI": "#3C6E9E"}


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def dry_spell_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["Exposure"].str.contains(DRY_SPELL, regex=False)].copy()


def outcome_name(value: str) -> str:
    return "EVI" if value == EVI else "NDVI"


def forest(
    ax: plt.Axes,
    frame: pd.DataFrame,
    labels: list[str],
    panel: str,
    colors: list[str],
    xlim: tuple[float, float] = (-0.12, 0.025),
) -> None:
    y = np.arange(len(frame))
    estimates = frame["Coefficient SD per 1 SD"].to_numpy(float)
    lower = frame["95 Percent CI Lower"].to_numpy(float)
    upper = frame["95 Percent CI Upper"].to_numpy(float)
    for index in range(len(frame)):
        ax.errorbar(
            estimates[index],
            y[index],
            xerr=[[estimates[index] - lower[index]], [upper[index] - estimates[index]]],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            markersize=5.5,
            capsize=3,
            linewidth=1.4,
            zorder=3,
        )
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks(y, labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel("Vegetation response (SD per 1-SD longer dry spell)")
    ax.grid(True, axis="x", color="#E4E4E4", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, panel)


def primary_panel(ax: plt.Axes, coefficients: pd.DataFrame) -> None:
    selected = dry_spell_rows(coefficients).loc[
        lambda x: x["Specification"].eq("Candidate B, 5 km")
        & x["Outcome"].isin([EVI, NDVI])
    ].copy()
    selected["order"] = selected["Outcome"].map({EVI: 0, NDVI: 1})
    selected = selected.sort_values("order")
    labels = [outcome_name(value) for value in selected["Outcome"]]
    forest(ax, selected, labels, "a", [COLORS[label] for label in labels], xlim=(-0.105, 0.015))
    for y, (_, row) in enumerate(selected.iterrows()):
        ax.text(
            row["95 Percent CI Upper"] + 0.003,
            y,
            f'{row["Coefficient SD per 1 SD"]:.3f}',
            ha="left",
            va="center",
            fontsize=8.2,
            color="#222222",
        )
    ax.set_ylabel("Primary 5-km outcome")


def radius_panel(ax: plt.Axes, coefficients: pd.DataFrame) -> None:
    selected = dry_spell_rows(coefficients).loc[
        lambda x: x["Specification"].isin(
            ["Candidate B, 2 km", "Candidate B, 5 km", "Candidate B, 10 km"]
        )
        & x["Outcome"].isin([EVI, NDVI])
    ].copy()
    order = []
    for outcome in [EVI, NDVI]:
        for radius in ["2 km", "5 km", "10 km"]:
            order.append((outcome, f"Candidate B, {radius}"))
    selected["order"] = selected.apply(
        lambda row: order.index((row["Outcome"], row["Specification"])), axis=1
    )
    selected = selected.sort_values("order")
    labels = [
        f"{outcome_name(outcome)} · {specification.removeprefix('Candidate B, ')}"
        for outcome, specification in zip(selected["Outcome"], selected["Specification"])
    ]
    colors = [COLORS[outcome_name(value)] for value in selected["Outcome"]]
    forest(ax, selected, labels, "b", colors)
    ax.set_ylabel("Village-buffer sensitivity")


def definition_time_panel(ax: plt.Axes, coefficients: pd.DataFrame) -> None:
    selected = dry_spell_rows(coefficients).loc[lambda x: x["Outcome"].eq(EVI)].copy()
    wanted = [
        "Candidate B, 5 km",
        "Candidate A, 5 km",
        "Candidate B, 5 km, excluding years 2001-2005",
        "Candidate B, 5 km, excluding years 2006-2010",
        "Candidate B, 5 km, excluding years 2011-2015",
        "Candidate B, 5 km, excluding years 2016-2019",
        "Candidate B, 5 km, excluding years 2020-2023",
    ]
    selected = selected.loc[selected["Specification"].isin(wanted)].copy()
    selected["order"] = selected["Specification"].map({value: index for index, value in enumerate(wanted)})
    selected = selected.sort_values("order")
    labels = [
        "Candidate B",
        "Candidate A",
        "Exclude 2001–05",
        "Exclude 2006–10",
        "Exclude 2011–15",
        "Exclude 2016–19",
        "Exclude 2020–23",
    ]
    colors = ["#2F7D5B", "#7A7A7A"] + ["#77A98D"] * 5
    forest(ax, selected, labels, "c", colors)
    ax.axhline(1.5, color="#BBBBBB", linewidth=0.7)
    ax.set_ylabel("Definition and time-block sensitivity (EVI)")


def crossfit_panel(ax: plt.Axes, crossfit: pd.DataFrame) -> None:
    order = [
        "Rainfall totals only",
        "Joint rainfall and monsoon structure",
        "Monsoon structure only",
    ]
    selected = crossfit.set_index("Model").loc[order].reset_index()
    labels = ["Rainfall totals only", "Rainfall + structure", "Monsoon structure only"]
    values = selected["RMSE SD"].to_numpy(float)
    y = np.arange(len(selected))
    colors = ["#9B9B9B", "#2F7D5B", "#6AA184"]
    ax.hlines(y, 0.590, values, color=colors, linewidth=2.0, alpha=0.72)
    ax.scatter(values, y, s=48, color=colors, zorder=3)
    baseline = values[0]
    for index, value in enumerate(values):
        label = f"{value:.3f}"
        if index > 0:
            improvement = 100 * (baseline - value) / baseline
            label += f"  (−{improvement:.2f}%)"
        ax.text(value + 0.0008, index, label, va="center", fontsize=8.4)
    ax.set_yticks(y, labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0.590, 0.622)
    ax.set_xlabel("Strict cross-fitted EVI RMSE (SD; lower is better)")
    ax.set_ylabel("Held-out model")
    ax.grid(True, axis="x", color="#E4E4E4", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    panel_label(ax, "d")


def main() -> None:
    coefficients = pd.read_csv(COEFFICIENTS)
    crossfit = pd.read_csv(CROSSFIT)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    primary_panel(axes[0, 0], coefficients)
    radius_panel(axes[0, 1], coefficients)
    definition_time_panel(axes[1, 0], coefficients)
    crossfit_panel(axes[1, 1], crossfit)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
