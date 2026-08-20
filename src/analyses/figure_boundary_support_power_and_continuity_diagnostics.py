#!/usr/bin/env python3
"""Boundary Support Power and Continuity Diagnostics.

Plan: Summarize side-specific support, predetermined continuity, effective-unit power,
and the modern-boundary safeguard under the frozen historical-boundary design.
Framework: AnaSOP Sections 5.3-5.4, 6.8, and the support/continuity/power workflow
in Section 7. No NPP or other outcome column is read.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
PANEL = (
    ROOT / "data/processed/historical_boundary_annual_spatial_climate_preprocessed.parquet"
)
ANNUAL_DIAGNOSTICS = (
    ROOT / "data/exp/feasibility-check/historical-boundary-annual-spatial"
)
IDENTIFICATION_DIAGNOSTICS = (
    ROOT / "data/exp/feasibility-check/historical-boundary-identification"
)
OUTPUT = (
    ROOT
    / "data/results/figures/Figure_boundary_support_power_and_continuity_diagnostics.png"
)

BANDWIDTHS = [2, 5, 10, 15, 20, 30]
PRIMARY_BANDWIDTH = 5
SESOI = 0.20
COLORS = {"Southwest": "#C94C35", "West": "#3C78B5"}
DESIGN_COLUMNS = [
    "Village Code",
    "Year",
    "Historical Repression Side",
    "Absolute Distance to Historical Repression Boundary km",
    "May October Rainfall Anomaly Z (1991-2020)",
]


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(
        True,
        axis=grid_axis,
        color="#D9D9D9",
        linewidth=0.55,
        zorder=0,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")


def side_support(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for bandwidth in BANDWIDTHS:
        sample = panel.loc[
            panel["Absolute Distance to Historical Repression Boundary km"].le(bandwidth)
        ]
        for side in ("West", "Southwest"):
            side_sample = sample.loc[sample["Historical Repression Side"].eq(side)]
            rows.append(
                {
                    "bandwidth_km": bandwidth,
                    "side": side,
                    "villages": side_sample["Village Code"].nunique(),
                    "rainfall_sd": side_sample[
                        "May October Rainfall Anomaly Z (1991-2020)"
                    ].std(),
                }
            )
    return pd.DataFrame(rows)


def plot_support(ax: plt.Axes, support: pd.DataFrame) -> None:
    ax.axvspan(4.55, 5.45, color="#F7E6A6", alpha=0.75, zorder=0)
    rainfall_ax = ax.twinx()
    for side in ("West", "Southwest"):
        data = support.loc[support["side"].eq(side)]
        ax.plot(
            data["bandwidth_km"],
            data["villages"],
            color=COLORS[side],
            marker="o",
            linewidth=1.8,
            markersize=5,
            zorder=3,
        )
        rainfall_ax.plot(
            data["bandwidth_km"],
            data["rainfall_sd"],
            color=COLORS[side],
            marker="s",
            markerfacecolor="white",
            linestyle="--",
            linewidth=1.25,
            markersize=4.5,
            zorder=3,
        )
    ax.set_xlabel("Symmetric boundary bandwidth (km)")
    ax.set_ylabel("Unique villages")
    rainfall_ax.tick_params(axis="y", labelsize=8)
    ax.set_xticks(BANDWIDTHS)
    ax.set_ylim(bottom=0)
    rainfall_ax.set_ylim(0.80, 1.08)
    rainfall_ax.grid(False)
    rainfall_ax.spines["right"].set_linewidth(0.9)
    rainfall_ax.spines["right"].set_color("#333333")
    style_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], color=COLORS["West"], marker="o", label="West villages"),
            Line2D(
                [0], [0], color=COLORS["Southwest"], marker="o", label="Southwest villages"
            ),
            Line2D(
                [0],
                [0],
                color="#555555",
                marker="s",
                markerfacecolor="white",
                linestyle="--",
                label="Rainfall SD (right axis)",
            ),
            Patch(facecolor="#F7E6A6", label="5 km primary"),
        ],
        loc="upper left",
        frameon=True,
        fontsize=7.4,
        ncol=2,
    )


def plot_continuity(ax: plt.Axes, continuity: pd.DataFrame) -> None:
    data = continuity.loc[
        continuity["bandwidth_km"].eq(PRIMARY_BANDWIDTH)
        & ~continuity["family"].str.contains("timing-ambiguous", na=False)
    ].copy()
    family_order = {
        "historical climate normal": 0,
        "soil": 1,
        "terrain and geography": 2,
    }
    data["family_order"] = data["family"].map(family_order)
    data = data.sort_values(["family_order", "label"], ascending=[False, False])
    y = np.arange(len(data))
    ax.axvspan(-0.25, 0.25, color="#E8EFE4", alpha=0.85, zorder=0)
    ax.axvline(0, color="#222222", linewidth=1.0, zorder=1)
    ax.axvline(-0.25, color="#71896A", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0.25, color="#71896A", linewidth=0.8, linestyle="--", zorder=1)
    for index, (_, row) in enumerate(data.iterrows()):
        color = "#D18A24" if row["review_status"] == "review" else "#4C78A8"
        ax.errorbar(
            row["standardized_discontinuity"],
            index,
            xerr=[
                [row["standardized_discontinuity"] - row["ci95_low"]],
                [row["ci95_high"] - row["standardized_discontinuity"]],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.15,
            capsize=2.2,
            markersize=4.5,
            zorder=3,
        )
    ax.set_yticks(y, data["label"], fontsize=7.2)
    ax.set_xlabel("Standardized Southwest-side discontinuity (95% CI)")
    ax.set_xlim(-0.75, 1.08)
    style_axis(ax, "x")
    ax.legend(
        handles=[
            Patch(facecolor="#E8EFE4", label="±0.25 SD review band"),
            Line2D([0], [0], marker="o", color="#D18A24", linestyle="", label="Review item"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=7.4,
    )


def plot_power(
    ax: plt.Axes,
    power: pd.DataFrame,
    commune_power: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> None:
    data = power.loc[power["dependence_scenario"].eq("strong clustered dependence")].copy()
    effective_ax = ax.twinx()
    may_oct = data.loc[data["shock"].eq("May-October rainfall anomaly")].sort_values(
        "bandwidth_km"
    )
    annual = data.loc[data["shock"].eq("Annual rainfall anomaly")].sort_values(
        "bandwidth_km"
    )
    effective_ax.plot(
        may_oct["bandwidth_km"],
        may_oct["iid_equivalent_village_years"],
        color="#A9A9A9",
        marker=".",
        linestyle=":",
        linewidth=1.2,
        zorder=1,
    )
    ax.axhline(SESOI, color="#A33A2B", linewidth=1.2, linestyle="--", zorder=2)
    ax.axvspan(4.55, 5.45, color="#F7E6A6", alpha=0.75, zorder=0)
    ax.plot(
        may_oct["bandwidth_km"],
        may_oct["mde_80_standardized_outcome_per_one_sd_shock"],
        color="#2D6A4F",
        marker="o",
        linewidth=1.8,
        markersize=5,
        zorder=4,
    )
    ax.plot(
        annual["bandwidth_km"],
        annual["mde_80_standardized_outcome_per_one_sd_shock"],
        color="#79589F",
        marker="s",
        linestyle="--",
        linewidth=1.45,
        markersize=4.5,
        zorder=4,
    )
    confirmation = commune_power.loc[
        commune_power["sample"].eq(
            "cross-side communes plus commune-by-year fixed effects"
        )
    ].iloc[0]
    worst_segment = leave_one_out.loc[
        ~leave_one_out["excluded_segment"].astype(str).eq("none")
    ].sort_values("mde_80_outcome_sd_per_one_sd_shock", ascending=False).iloc[0]
    ax.scatter(
        [5],
        [confirmation["mde_80_outcome_sd_per_one_sd_shock"]],
        marker="D",
        s=42,
        color="#111111",
        zorder=5,
    )
    ax.scatter(
        [5],
        [worst_segment["mde_80_outcome_sd_per_one_sd_shock"]],
        marker="^",
        s=45,
        color="#D18A24",
        zorder=5,
    )
    ax.annotate(
        "commune safeguard",
        (5, confirmation["mde_80_outcome_sd_per_one_sd_shock"]),
        xytext=(7, -12),
        textcoords="offset points",
        fontsize=7,
    )
    ax.annotate(
        "worst segment omission",
        (5, worst_segment["mde_80_outcome_sd_per_one_sd_shock"]),
        xytext=(8, 6),
        textcoords="offset points",
        fontsize=7,
    )
    ax.set_xlabel("Symmetric boundary bandwidth (km)")
    ax.set_ylabel("80% MDE (outcome SD per 1-SD shock)")
    effective_ax.set_ylabel("IID-equivalent village-years", color="#777777")
    effective_ax.tick_params(axis="y", colors="#777777")
    ax.set_xticks(BANDWIDTHS)
    ax.set_ylim(0.145, 0.218)
    effective_ax.set_ylim(bottom=0)
    effective_ax.grid(False)
    effective_ax.spines["right"].set_linewidth(0.9)
    effective_ax.spines["right"].set_color("#777777")
    style_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#2D6A4F", marker="o", label="May–October MDE"),
            Line2D(
                [0], [0], color="#79589F", marker="s", linestyle="--", label="Annual MDE"
            ),
            Line2D([0], [0], color="#A33A2B", linestyle="--", label="0.20 SD SESOI"),
            Line2D([0], [0], color="#A9A9A9", linestyle=":", label="Effective units"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=7.2,
        ncol=2,
    )


def plot_modern_boundary(
    ax: plt.Axes, coincidence: pd.DataFrame, commune_power: pd.DataFrame
) -> None:
    data = coincidence.loc[
        coincidence["modern_feature"].eq("modern commune internal boundaries")
    ].sort_values("distance_threshold_km")
    ax.plot(
        data["distance_threshold_km"],
        data["boundary_length_share_within_threshold"],
        color="#C94C35",
        marker="o",
        linewidth=1.8,
        label="Historical boundary",
    )
    ax.plot(
        data["distance_threshold_km"],
        data["province_1km_grid_point_share_within_threshold"],
        color="#3C78B5",
        marker="s",
        linestyle="--",
        linewidth=1.5,
        label="Province grid reference",
    )
    ax.axvline(1, color="#777777", linewidth=0.8, linestyle=":")
    one_km = data.loc[data["distance_threshold_km"].eq(1)].iloc[0]
    ax.scatter(
        [1],
        [one_km["boundary_length_share_within_threshold"]],
        color="#C94C35",
        s=34,
        zorder=4,
    )
    confirmation = commune_power.loc[
        commune_power["sample"].eq(
            "cross-side communes plus commune-by-year fixed effects"
        )
    ].iloc[0]
    ax.text(
        0.97,
        0.07,
        "Mandatory confirmation\n"
        f"{int(confirmation['climate_communes'])} cross-side communes · "
        f"{int(confirmation['villages'])} villages\n"
        f"80% MDE = {confirmation['mde_80_outcome_sd_per_one_sd_shock']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#888888"},
    )
    ax.set_xlabel("Distance to a modern commune boundary (km)")
    ax.set_ylabel("Share within threshold")
    ax.set_xticks(data["distance_threshold_km"])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 0.84)
    style_axis(ax)
    ax.legend(loc="upper left", frameon=True, fontsize=7.5)


def main() -> None:
    panel = pd.read_parquet(PANEL, columns=DESIGN_COLUMNS)
    support = side_support(panel)
    continuity = pd.read_csv(
        IDENTIFICATION_DIAGNOSTICS / "predetermined_covariate_continuity.csv"
    )
    power = pd.read_csv(ANNUAL_DIAGNOSTICS / "annual_spatial_blinded_power.csv")
    coincidence = pd.read_csv(
        IDENTIFICATION_DIAGNOSTICS / "modern_boundary_coincidence.csv"
    )
    commune_power = pd.read_csv(
        IDENTIFICATION_DIAGNOSTICS / "modern_commune_restriction_power.csv"
    )
    leave_one_out = pd.read_csv(
        IDENTIFICATION_DIAGNOSTICS / "boundary_segment_leave_one_out_power.csv"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.4))
    plot_support(axes[0, 0], support)
    plot_continuity(axes[0, 1], continuity)
    plot_power(axes[1, 0], power, commune_power, leave_one_out)
    plot_modern_boundary(axes[1, 1], coincidence, commune_power)

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(
            -0.11,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

    fig.subplots_adjust(left=0.12, right=0.94, top=0.98, bottom=0.09, wspace=0.55, hspace=0.34)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print("Outcome columns read: none")
    print(
        "5 km support: "
        f"{int(support.loc[(support.bandwidth_km == 5) & (support.side == 'West'), 'villages'].iloc[0])} West, "
        f"{int(support.loc[(support.bandwidth_km == 5) & (support.side == 'Southwest'), 'villages'].iloc[0])} Southwest villages"
    )


if __name__ == "__main__":
    main()
