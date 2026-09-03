#!/usr/bin/env python3
"""Household linkage and consumption support by survey wave.

Plan: document released and geocoded household support, prior-year NPP linkage,
consumption complete cases, exact interview-to-NPP timing, and questionnaire
regime changes before interpreting stage-2 coefficients.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/cses_household_cropland_npp_analysis_preprocessed.parquet"
OUTPUT = ROOT / "data/results/figures/Figure_household_linkage_and_consumption_support_by_wave.png"
EVIDENCE = ROOT / "data/exp/analysis/climate-npp/household-linkage-and-consumption-support-by-wave"

WAVES = ["2004", "2007", "2009", "2011-12", "2013", "2014", "2016", "2017", "2019", "2021"]
WAVE_LABELS = ["2004", "2007", "2009", "2011–12", "2013", "2014", "2016", "2017", "2019", "2021"]

REGIME_SHORT = {
    "2004 diagnostic": "2004 diagnostic",
    "2007 integrated housing recall": "2007 instrument",
    "2009-2013 recall plus housing": "2009–13 regime",
    "2014-2017 expanded recall plus housing": "2014–17 regime",
    "2019-2021 itemized recall plus housing and education": "2019–21 regime",
}
REGIME_COLORS = {
    "2004 diagnostic": "#B5BDC1",
    "2007 integrated housing recall": "#547A9B",
    "2009-2013 recall plus housing": "#2F80A2",
    "2014-2017 expanded recall plus housing": "#3A9D8F",
    "2019-2021 itemized recall plus housing and education": "#D6A84B",
}

NAVY = "#183B56"
BLUE = "#2F80A2"
TEAL = "#3A9D8F"
ORANGE = "#D97757"
GOLD = "#D6A84B"
GRAY = "#B5BDC1"
DARK = "#48545B"
GRID = "#D8DEE2"


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
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.2},
        zorder=20,
    )


def title_inside(ax: plt.Axes, title: str) -> None:
    ax.text(
        0.985,
        0.965,
        title,
        transform=ax.transAxes,
        fontsize=10.1,
        fontweight="bold",
        ha="right",
        va="top",
        color=NAVY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 1.4},
        zorder=20,
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", color=GRID, linewidth=0.55, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.2, colors=DARK, length=2.5)
    for spine in ax.spines.values():
        spine.set_color("#89969C")
        spine.set_linewidth(0.65)


def main() -> None:
    columns = [
        "Survey Wave",
        "Household ID",
        "Main Linked Sample",
        "Geography Link Matched",
        "Public Village Point Matched",
        "Total Consumption Main Outcome Eligible",
        "Prior-Year Strict-Cropland NPP",
        "Stage 2 Total Consumption Complete Case",
        "Stage 2 Food Consumption Complete Case",
        "Total Consumption Instrument Regime",
        "Interview Calendar Year",
        "Prior NPP Calendar Year",
    ]
    frame = pd.read_parquet(INPUT, columns=columns).copy()
    frame["Public Village Point Matched Flag"] = frame["Public Village Point Matched"].fillna(0).eq(1)
    frame["Prior-Year NPP Linked"] = frame["Prior-Year Strict-Cropland NPP"].notna()
    frame["Survey Wave"] = pd.Categorical(frame["Survey Wave"], categories=WAVES, ordered=True)

    summary = (
        frame.groupby("Survey Wave", observed=False)
        .agg(
            **{
                "Released Households": ("Household ID", "size"),
                "Main Analysis-Frame Households": ("Main Linked Sample", "sum"),
                "Geography Matched Households": ("Geography Link Matched", "sum"),
                "Public-Point Matched Households": ("Public Village Point Matched Flag", "sum"),
                "Outcome-Eligible Households": ("Total Consumption Main Outcome Eligible", "sum"),
                "Prior-Year NPP Linked Households": ("Prior-Year NPP Linked", "sum"),
                "Total-Consumption Complete Cases": ("Stage 2 Total Consumption Complete Case", "sum"),
                "Food-Consumption Complete Cases": ("Stage 2 Food Consumption Complete Case", "sum"),
                "Interview Year Minimum": ("Interview Calendar Year", "min"),
                "Interview Year Median": ("Interview Calendar Year", "median"),
                "Interview Year Maximum": ("Interview Calendar Year", "max"),
                "Prior NPP Year Minimum": ("Prior NPP Calendar Year", "min"),
                "Prior NPP Year Median": ("Prior NPP Calendar Year", "median"),
                "Prior NPP Year Maximum": ("Prior NPP Calendar Year", "max"),
                "Instrument Regime": ("Total Consumption Instrument Regime", "first"),
            }
        )
        .reset_index()
    )
    denominator = summary["Outcome-Eligible Households"].replace(0, np.nan)
    summary["Public-Point Match Rate"] = summary["Public-Point Matched Households"] / denominator
    summary["Prior-Year NPP Link Rate"] = summary["Prior-Year NPP Linked Households"] / denominator
    summary["Total-Consumption Complete Rate"] = summary["Total-Consumption Complete Cases"] / denominator
    summary["Food-Consumption Complete Rate"] = summary["Food-Consumption Complete Cases"] / denominator

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    summary.to_csv(EVIDENCE / "household_linkage_consumption_support_by_wave.csv", index=False)
    regime_table = summary[["Survey Wave", "Instrument Regime", "Released Households"]].copy()
    regime_table["Readable Instrument Regime"] = regime_table["Instrument Regime"].map(REGIME_SHORT)
    regime_table.to_csv(EVIDENCE / "consumption_instrument_regime_by_wave.csv", index=False)
    evidence_summary = {
        "released_households": int(summary["Released Households"].sum()),
        "public_point_matched_households": int(summary["Public-Point Matched Households"].sum()),
        "prior_year_npp_linked_households": int(summary["Prior-Year NPP Linked Households"].sum()),
        "total_consumption_complete_cases": int(summary["Total-Consumption Complete Cases"].sum()),
        "food_consumption_complete_cases": int(summary["Food-Consumption Complete Cases"].sum()),
        "waves_without_public_point_linkage": summary.loc[
            summary["Public-Point Matched Households"].eq(0), "Survey Wave"
        ].astype(str).tolist(),
        "exact_timing_rule": "Prior NPP calendar year equals interview calendar year minus one",
    }
    (EVIDENCE / "household_linkage_and_consumption_support_summary.json").write_text(
        json.dumps(evidence_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    x = np.arange(len(WAVES))
    fig, axes = plt.subplot_mosaic(
        [["a", "a", "b", "b"], ["c", "c", "d", "d"]],
        figsize=(15.2, 8.4),
        facecolor="white",
        gridspec_kw={
            "left": 0.065,
            "right": 0.975,
            "top": 0.97,
            "bottom": 0.10,
            "wspace": 0.24,
            "hspace": 0.28,
        },
    )

    ax = axes["a"]
    width = 0.38
    ax.bar(
        x - width / 2,
        summary["Released Households"],
        width,
        color=GRAY,
        label="Released households",
        zorder=2,
    )
    ax.bar(
        x + width / 2,
        summary["Public-Point Matched Households"],
        width,
        color=TEAL,
        label="Public-village-point matched",
        zorder=2,
    )
    ax.set_xticks(x, WAVE_LABELS)
    ax.set_ylabel("Households", fontsize=8.8, color=DARK)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    ax.legend(loc="upper right", bbox_to_anchor=(0.99, 0.88), frameon=False, fontsize=8.0, ncol=2)
    panel_label(ax, "a")
    title_inside(ax, "Released and geographically matched samples")
    style_axis(ax)

    ax = axes["b"]
    rate_series = [
        ("Public-Point Match Rate", "Public-point matched", BLUE, "o"),
        ("Prior-Year NPP Link Rate", "Prior-year NPP linked", TEAL, "s"),
        ("Total-Consumption Complete Rate", "Total complete", NAVY, "D"),
        ("Food-Consumption Complete Rate", "Food complete", ORANGE, "^"),
    ]
    for column, label, color, marker in rate_series:
        ax.plot(
            x,
            summary[column],
            color=color,
            marker=marker,
            markersize=4.2,
            linewidth=1.45,
            label=label,
            zorder=3,
        )
    ax.set_xticks(x, WAVE_LABELS)
    ax.set_ylim(0.55, 0.91)
    ax.set_ylabel("Share of outcome-eligible households", fontsize=8.8, color=DARK)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=7.7, ncol=2)
    panel_label(ax, "b")
    title_inside(ax, "Linkage and complete-case retention")
    style_axis(ax)

    ax = axes["c"]
    timing = summary.dropna(subset=["Interview Year Median", "Prior NPP Year Median"]).copy()
    timing_x = timing.index.to_numpy(float)
    for xpos, (_, row) in zip(timing_x, timing.iterrows(), strict=True):
        ax.vlines(
            xpos,
            row["Prior NPP Year Median"],
            row["Interview Year Median"],
            color="#AAB2B6",
            linewidth=1.2,
            zorder=1,
        )
        ax.vlines(
            xpos - 0.08,
            row["Interview Year Minimum"],
            row["Interview Year Maximum"],
            color=ORANGE,
            linewidth=2.0,
            zorder=2,
        )
        ax.vlines(
            xpos + 0.08,
            row["Prior NPP Year Minimum"],
            row["Prior NPP Year Maximum"],
            color=TEAL,
            linewidth=2.0,
            zorder=2,
        )
    ax.scatter(timing_x - 0.08, timing["Interview Year Median"], s=30, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.scatter(timing_x + 0.08, timing["Prior NPP Year Median"], s=30, color=TEAL, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(x, WAVE_LABELS)
    ax.set_ylabel("Calendar year", fontsize=8.8, color=DARK)
    ax.set_ylim(2004.5, 2022.0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=ORANGE, label="Interview year", markersize=4.5),
            Line2D([0], [0], marker="o", color=TEAL, label="Prior NPP year", markersize=4.5),
        ],
        loc="upper left",
        bbox_to_anchor=(0.02, 0.88),
        frameon=False,
        fontsize=8.0,
        ncol=2,
    )
    panel_label(ax, "c")
    title_inside(ax, "Exact interview-to-NPP timing")
    style_axis(ax)

    ax = axes["d"]
    regimes = summary["Instrument Regime"].astype(str)
    bar_colors = [REGIME_COLORS[regime] for regime in regimes]
    ax.bar(
        x,
        summary["Total-Consumption Complete Cases"],
        width=0.62,
        color=bar_colors,
        zorder=2,
    )
    ax.scatter(
        x,
        summary["Food-Consumption Complete Cases"],
        marker="D",
        s=29,
        facecolor="white",
        edgecolor="#26343B",
        linewidth=0.9,
        zorder=3,
        label="Food-consumption complete cases",
    )
    ax.set_xticks(x, WAVE_LABELS)
    ax.set_ylabel("Complete-case households", fontsize=8.8, color=DARK)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    regime_handles = [
        Patch(facecolor=color, edgecolor="none", label=REGIME_SHORT[regime])
        for regime, color in REGIME_COLORS.items()
    ]
    regime_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#26343B",
            label="Food complete",
            markersize=5,
        )
    )
    ax.legend(
        handles=regime_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.88),
        frameon=False,
        fontsize=7.0,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    panel_label(ax, "d")
    title_inside(ax, "Analytical counts by instrument regime")
    style_axis(ax)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved evidence: {EVIDENCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
