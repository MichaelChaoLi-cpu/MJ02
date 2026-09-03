#!/usr/bin/env python3
"""Data Linkage and Analytical Support.

Plan: Show CSES spatial linkage by wave, raw-versus-weighted predetermined
balance, and the concentration of common-support weights.
Framework: AnaSOP Section 5 balance/overlap gate, Section 6 linkage
diagnostics, and Section 7 Step 2.  This diagnostic does not estimate any
post-conflict outcome effect.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
CSES_PANEL = ROOT / "data/processed/cambodia_thailand_cses_conflict_climate_panel_candidate_preprocessed.parquet"
DESIGN_DIR = ROOT / "data/exp/experiment-design/cambodia-thailand-village-common-support"
BALANCE = DESIGN_DIR / "predetermined_balance_diagnostics.csv"
WEIGHTS = DESIGN_DIR / "village_common_support_weights.csv"
SUMMARY = DESIGN_DIR / "common_support_summary.json"
OUT = ROOT / "data/exp/legacy-results/figures/Figure_data_linkage_and_analytical_support.png"

PRIMARY_RADIUS_KM = 5
BLUE = "#2166ac"
ORANGE = "#ef8a62"
GRAY = "#969696"
RED = "#b2182b"

SHORT_NAMES = {
    "Pre-conflict NPP mean": "NPP mean",
    "Pre-conflict NPP trend": "NPP trend",
    "Pre-conflict NPP SD": "NPP variability",
    "Pre-conflict rainfall mean": "Rainfall mean",
    "Pre-conflict rainfall SD": "Rainfall variability",
    "Pre-conflict dry intensity": "Dry intensity",
    "Pre-conflict heat intensity": "Heat intensity",
    "Border distance": "Border distance",
    "Elevation": "Elevation",
    "Slope": "Slope",
    "Historical-road distance": "Historical-road distance",
    "Baseline cropland share": "Cropland share",
    "Log baseline population": "Log population",
}


def build_linkage() -> pd.DataFrame:
    columns = [
        "Survey Year",
        "Village Code",
        "Buffer Radius km",
        "National Public Village Point Matched",
        "Candidate Affected District",
    ]
    frame = pd.read_parquet(CSES_PANEL, columns=columns)
    frame = frame[frame["Buffer Radius km"].eq(PRIMARY_RADIUS_KM)].copy()
    if frame.duplicated(["Survey Year", "Village Code"]).any():
        raise RuntimeError("The primary-radius CSES release is not unique by village-year")
    linkage = frame.groupby("Survey Year", observed=True).agg(
        Village_years=("Village Code", "size"),
        Linked_village_years=("National Public Village Point Matched", "sum"),
        Candidate_treated_village_years=("Candidate Affected District", "sum"),
    ).reset_index()
    linkage["Linkage rate"] = linkage["Linked_village_years"] / linkage["Village_years"]
    return linkage


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.975,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        fontweight="bold",
        zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.6},
    )


def plot_balance(ax: plt.Axes, balance: pd.DataFrame) -> None:
    data = balance.copy()
    data["Variable label"] = data["Variable"].map(SHORT_NAMES).fillna(data["Variable"])
    data["Raw absolute SMD"] = data["Raw standardized mean difference"].abs()
    data["Weighted absolute SMD"] = data["Weighted standardized mean difference"].abs()
    data = data.sort_values("Raw absolute SMD", ascending=True).reset_index(drop=True)
    y = np.arange(len(data))
    ax.hlines(y, data["Weighted absolute SMD"], data["Raw absolute SMD"], color="#d9d9d9", linewidth=1.5)
    ax.scatter(data["Raw absolute SMD"], y, s=38, color=GRAY, edgecolor="white", linewidth=0.4, label="Raw")
    ax.scatter(data["Weighted absolute SMD"], y, s=42, marker="D", color=BLUE, edgecolor="white", linewidth=0.4, label="Weighted")
    ax.axvline(0.10, color=ORANGE, linewidth=1.0, linestyle="--")
    ax.axvline(0.20, color=RED, linewidth=1.0, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels(data["Variable label"], fontsize=8.5)
    ax.set_xlabel("Absolute standardized mean difference")
    ax.set_xlim(0, max(0.45, data["Raw absolute SMD"].max() * 1.06))
    ax.grid(axis="x", color="#e5e5e5", linestyle="--", linewidth=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    sns.despine(ax=ax)


def plot_linkage(ax: plt.Axes, linkage: pd.DataFrame) -> None:
    x = np.arange(len(linkage))
    width = 0.72
    ax.bar(x, linkage["Village_years"], width=width, color="#d9d9d9", label="All CSES village-years")
    ax.bar(x, linkage["Linked_village_years"], width=width, color=BLUE, label="Linked to public point")
    for xpos, linked, rate in zip(x, linkage["Linked_village_years"], linkage["Linkage rate"]):
        ax.text(xpos, linked + linkage["Village_years"].max() * 0.025, f"{rate:.0%}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(linkage["Survey Year"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Village-year records")
    ax.set_ylim(0, linkage["Village_years"].max() * 1.15)
    ax.grid(axis="y", color="#e5e5e5", linestyle="--", linewidth=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=8.2)
    sns.despine(ax=ax)


def plot_weight_concentration(ax: plt.Axes, weights: pd.DataFrame, summary: dict) -> None:
    controls = weights[weights["Analysis role"].eq("Eligible weighted control")].copy()
    controls = controls.sort_values("Control calibration weight", ascending=False).reset_index(drop=True)
    cumulative = controls["Control calibration weight"].cumsum().to_numpy()
    rank = np.arange(1, len(controls) + 1)
    n95 = int(summary["mapped_controls_reaching_95_percent_weight"])
    ax.plot(rank, cumulative, color=BLUE, linewidth=2.0)
    ax.axhline(0.95, color=ORANGE, linewidth=1.0, linestyle="--")
    ax.axvline(n95, color=ORANGE, linewidth=1.0, linestyle="--")
    ax.scatter([n95], [cumulative[n95 - 1]], s=44, color=ORANGE, edgecolor="white", linewidth=0.5, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(1, len(controls))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Control villages ranked by weight (log scale)")
    ax.set_ylabel("Cumulative control weight")
    ax.grid(color="#e5e5e5", linestyle="--", linewidth=0.6)
    ax.text(
        0.04,
        0.73,
        f"88 candidate treated villages\n{len(controls):,} eligible controls\n{n95} controls carry 95% of weight\nESS = {summary['control_effective_sample_size']:.1f}",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        linespacing=1.35,
    )
    sns.despine(ax=ax)


def main() -> None:
    linkage = build_linkage()
    balance = pd.read_csv(BALANCE)
    weights = pd.read_csv(WEIGHTS, low_memory=False)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(13.6, 8.6))
    axes = fig.subplot_mosaic(
        [["a", "b"], ["a", "c"]],
        gridspec_kw={"width_ratios": [1.15, 1.0], "height_ratios": [1.0, 1.0]},
    )
    fig.subplots_adjust(left=0.16, right=0.985, top=0.975, bottom=0.10, wspace=0.23, hspace=0.30)

    plot_balance(axes["a"], balance)
    plot_linkage(axes["b"], linkage)
    plot_weight_concentration(axes["c"], weights, summary)
    for label, ax in axes.items():
        add_panel_label(ax, label)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT.relative_to(ROOT)}")
    print(linkage.to_string(index=False))


if __name__ == "__main__":
    main()
