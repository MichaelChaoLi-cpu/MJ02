#!/usr/bin/env python3
"""Exposure Distributions and Common Support.

Plan: Test whether the main analytical sample has usable distributions and
overlap in historical conflict and contemporary rainfall, drought, and price
exposures.
Framework: AnaSOP Sections 5.1, 5.2, 6.2, 6.5, and the common-support audit
step in Section 7. Conflict groups are diagnostic only, not estimands.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    ROOT / "data/processed/direction3_household_conflict_shock_preprocessed.parquet"
)
OUTPUT_PATH = (
    ROOT / "data/results/figures/Figure_exposure_distributions_and_common_support.png"
)

YEAR = "Survey Year"
MONTH = "Survey Month Numeric"
RESOLUTION = "Climate Geography Resolution"
GEOGRAPHY = "Climate Geography Code"
CONFLICT = "Log Bombing Unique Locations per 100 km2"
ANNUAL_RAIN = "Annual Rainfall Anomaly Z (1991-2020)"
SPI12 = "Interview Month SPI 12 Month"
PRICE = "Local Relative Log Wholesale Rice Price"
GROUP = "Historical conflict group"
GROUP_ORDER = ["Low", "Middle", "High"]
PALETTE = {"Low": "#2B8CBE", "Middle": "#7BCCC4", "High": "#D95F0E"}


def assign_conflict_groups(
    values: pd.Series, lower_cut: float, upper_cut: float
) -> pd.Series:
    return pd.cut(
        values,
        bins=[float("-inf"), lower_cut, upper_cut, float("inf")],
        labels=GROUP_ORDER,
        include_lowest=True,
    )


def build_samples() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, float]]:
    columns = [YEAR, MONTH, RESOLUTION, GEOGRAPHY, CONFLICT, ANNUAL_RAIN, SPI12, PRICE]
    households = pd.read_parquet(INPUT_PATH, columns=columns)
    communes = households.loc[households[RESOLUTION].eq("commune")].copy()

    conflict_variation = communes.groupby(GEOGRAPHY)[CONFLICT].nunique(dropna=True)
    assert conflict_variation.max() == 1
    geography = (
        communes[[GEOGRAPHY, CONFLICT]]
        .drop_duplicates(GEOGRAPHY)
        .dropna(subset=[CONFLICT])
        .reset_index(drop=True)
    )
    assert len(geography) == 1_490
    lower_cut = float(geography[CONFLICT].quantile(1 / 3))
    upper_cut = float(geography[CONFLICT].quantile(2 / 3))
    assert lower_cut < upper_cut
    geography[GROUP] = assign_conflict_groups(geography[CONFLICT], lower_cut, upper_cut)

    definitions = {
        ANNUAL_RAIN: [GEOGRAPHY, YEAR],
        SPI12: [GEOGRAPHY, YEAR, MONTH],
        PRICE: [GEOGRAPHY, YEAR, MONTH],
    }
    samples: dict[str, pd.DataFrame] = {}
    support: dict[str, float] = {}
    for variable, keys in definitions.items():
        candidate = communes[keys + [CONFLICT, variable]].drop_duplicates(keys)
        support[variable] = float(candidate[variable].notna().mean())
        sample = candidate.dropna(subset=[CONFLICT, variable]).copy()
        sample[GROUP] = assign_conflict_groups(sample[CONFLICT], lower_cut, upper_cut)
        assert sample[GROUP].notna().all()
        samples[variable] = sample

    assert len(samples[ANNUAL_RAIN]) == 4_791
    assert len(samples[SPI12]) == 4_197
    assert len(samples[PRICE]) == 2_761
    return geography, samples, {
        "lower_cut": lower_cut,
        "upper_cut": upper_cut,
        **support,
    }


def style_main_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9CA3AF")
    ax.tick_params(colors="#374151", labelsize=8.5)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.set_ylabel("Density", fontsize=9, color="#374151")


def add_overall_boxplot(ax: plt.Axes, frame: pd.DataFrame, variable: str) -> None:
    inset = ax.inset_axes([0.12, 0.72, 0.83, 0.20])
    inset.set_facecolor((1, 1, 1, 0.92))
    sns.boxplot(
        data=frame,
        x=variable,
        ax=inset,
        color="#4C78A8",
        width=0.45,
        linewidth=0.8,
        fliersize=1.5,
    )
    inset.set(xlabel=None, ylabel=None, yticks=[])
    inset.tick_params(axis="x", labelsize=7, colors="#4B5563", length=2)
    inset.spines[["top", "right", "left"]].set_visible(False)
    inset.spines["bottom"].set_color("#D1D5DB")


def add_group_boxplot(ax: plt.Axes, frame: pd.DataFrame, variable: str) -> None:
    inset = ax.inset_axes([0.12, 0.67, 0.83, 0.25])
    inset.set_facecolor((1, 1, 1, 0.92))
    sns.boxplot(
        data=frame,
        x=variable,
        y=GROUP,
        hue=GROUP,
        order=GROUP_ORDER,
        hue_order=GROUP_ORDER,
        palette=PALETTE,
        dodge=False,
        ax=inset,
        width=0.55,
        linewidth=0.7,
        fliersize=1.2,
        legend=False,
    )
    inset.set(xlabel=None, ylabel=None)
    inset.tick_params(axis="x", labelsize=7, colors="#4B5563", length=2)
    inset.tick_params(axis="y", labelsize=7, colors="#4B5563", length=0)
    inset.spines[["top", "right", "left"]].set_visible(False)
    inset.spines["bottom"].set_color("#D1D5DB")


def plot_conflict_panel(
    ax: plt.Axes, frame: pd.DataFrame, lower_cut: float, upper_cut: float
) -> None:
    sns.histplot(
        data=frame,
        x=CONFLICT,
        bins=34,
        stat="density",
        color="#4C78A8",
        alpha=0.58,
        edgecolor="white",
        linewidth=0.35,
        ax=ax,
    )
    ax.axvline(lower_cut, color="#374151", linestyle="--", linewidth=1.0)
    ax.axvline(upper_cut, color="#374151", linestyle="--", linewidth=1.0)
    add_overall_boxplot(ax, frame, CONFLICT)
    ax.set_xlabel("Log bombing-location density per 100 km²", fontsize=9)
    ax.text(
        0.98,
        0.96,
        f"N = {len(frame):,} communes",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#374151",
    )


def plot_shock_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    variable: str,
    xlabel: str,
    support: float,
) -> None:
    value_min = float(frame[variable].min())
    value_max = float(frame[variable].max())
    for group in GROUP_ORDER:
        group_frame = frame.loc[frame[GROUP].eq(group)]
        sns.histplot(
            data=group_frame,
            x=variable,
            bins=34,
            binrange=(value_min, value_max),
            stat="density",
            element="step",
            fill=False,
            linewidth=1.35,
            color=PALETTE[group],
            ax=ax,
        )
    add_group_boxplot(ax, frame, variable)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.text(
        0.98,
        0.96,
        f"N = {len(frame):,} · support = {support:.1%}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color="#374151",
    )


def main() -> None:
    geography, samples, diagnostics = build_samples()
    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    panel_specs = [
        (axes[0, 0], "a", "Historical conflict exposure"),
        (axes[0, 1], "b", "Annual rainfall anomaly"),
        (axes[1, 0], "c", "Interview-aligned drought conditions"),
        (axes[1, 1], "d", "Local wholesale rice-price pressure"),
    ]
    for ax, label, descriptor in panel_specs:
        style_main_axis(ax)
        ax.text(
            -0.08,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )
        ax.text(
            0.0,
            1.04,
            descriptor,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="semibold",
            va="top",
        )

    plot_conflict_panel(
        axes[0, 0], geography, diagnostics["lower_cut"], diagnostics["upper_cut"]
    )
    plot_shock_panel(
        axes[0, 1],
        samples[ANNUAL_RAIN],
        ANNUAL_RAIN,
        "Annual rainfall anomaly z-score",
        diagnostics[ANNUAL_RAIN],
    )
    plot_shock_panel(
        axes[1, 0],
        samples[SPI12],
        SPI12,
        "Interview-month SPI-12",
        diagnostics[SPI12],
    )
    plot_shock_panel(
        axes[1, 1],
        samples[PRICE],
        PRICE,
        "Local relative log wholesale rice price",
        diagnostics[PRICE],
    )

    legend_handles = [Patch(facecolor=PALETTE[group], label=group) for group in GROUP_ORDER]
    figure.legend(
        handles=legend_handles,
        title="Historical conflict tercile",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        title_fontsize=8.5,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.96, bottom=0.085, wspace=0.18, hspace=0.27)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print("Panels: 4 (a-d), no figure title")
    print(
        "Conflict tercile cut points: "
        f"{diagnostics['lower_cut']:.3f}, {diagnostics['upper_cut']:.3f}"
    )
    print(f"Conflict sample: {len(geography):,} communes")
    for variable in [ANNUAL_RAIN, SPI12, PRICE]:
        print(
            f"{variable}: N={len(samples[variable]):,}; "
            f"support={diagnostics[variable]:.1%}"
        )


if __name__ == "__main__":
    main()
