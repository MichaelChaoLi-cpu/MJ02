#!/usr/bin/env python3
"""Historical Repression and Multi-Hazard Climate Sensitivity.

Plan: Compare the frozen rainfall-response evidence with the activated annual and
16-day heat-response estimates while excluding the underpowered annual compound model.
Framework: AnaSOP Sections 5.8, 6.16, and the temperature-response workflow in Section 7.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from _temperature_response_models import fit_confirmatory_family


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/exp/legacy-results/figures/Figure_historical_repression_and_multi_hazard_climate_sensitivity.png"
ESTIMATE_OUTPUT = ROOT / "data/exp/temperature-response/formal_confirmatory_estimates.csv"
ANNUAL_RAIN = ROOT / "data/exp/legacy-results/tables/Table_historical_boundary_shock_response_estimates.xlsx"
HF_RAIN = ROOT / "data/exp/legacy-results/tables/Table_high_frequency_rainfall_response_dynamics.xlsx"
COLORS = {"Primary": "#B33B2E", "Within climate commune": "#2B6F9F"}
MARKERS = {"Primary": "o", "Within climate commune": "s"}


def parse_interval(value: str) -> tuple[float, float]:
    low, high = str(value).strip("[]").split(",")
    return float(low), float(high)


def rainfall_rows() -> pd.DataFrame:
    annual = pd.read_excel(ANNUAL_RAIN, sheet_name="Shock Response")
    annual = annual.loc[
        annual["Outcome"].eq("Annual land NPP anomaly (standardized)")
        & annual["Shock"].eq("May–October rainfall anomaly (1 SD)")
        & annual["Bandwidth (km)"].eq(5)
    ].copy()
    annual["specification"] = annual["Sample"].map(
        lambda value: "Within climate commune" if str(value).startswith("9 cross-side") else "Primary"
    )
    annual[["ci_low", "ci_high"]] = annual["95% CI"].map(parse_interval).tolist()
    annual_rows = pd.DataFrame(
        {
            "panel": "Annual land NPP",
            "estimand": "Growing-season rainfall",
            "specification": annual["specification"],
            "estimate": annual["Interaction estimate"].astype(float),
            "ci_low": annual["ci_low"].astype(float),
            "ci_high": annual["ci_high"].astype(float),
        }
    )

    hf = pd.read_excel(HF_RAIN, sheet_name="Response Dynamics")
    hf = hf.loc[
        hf["Outcome"].eq("EVI")
        & hf["Shock intensity"].eq("Dry")
        & hf["Estimand"].eq("Immediate average response, periods 0-2")
    ].copy()
    hf["specification"] = hf["Specification"].replace(
        {"Within commune": "Within climate commune"}
    )
    hf[["ci_low", "ci_high"]] = hf["95% CI"].map(parse_interval).tolist()
    hf_rows = pd.DataFrame(
        {
            "panel": "16-day EVI",
            "estimand": "Dry rainfall, immediate",
            "specification": hf["specification"],
            "estimate": hf["Estimate"].astype(float),
            "ci_low": hf["ci_low"].astype(float),
            "ci_high": hf["ci_high"].astype(float),
        }
    )
    return pd.concat([annual_rows, hf_rows], ignore_index=True)


def temperature_rows() -> pd.DataFrame:
    estimates = fit_confirmatory_family()
    ESTIMATE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(ESTIMATE_OUTPUT, index=False)
    label_map = {
        "Annual hot days": ("Annual land NPP", "Hot days"),
        "16-day hot days": ("16-day EVI", "Hot days"),
        "16-day compound hot-dry": ("16-day EVI", "Compound hot-dry"),
    }
    rows = []
    for row in estimates.itertuples(index=False):
        panel, label = label_map[row.estimand]
        rows.append(
            {
                "panel": panel,
                "estimand": label,
                "specification": row.specification,
                "estimate": row.estimate,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
            }
        )
    return pd.DataFrame(rows)


def draw_panel(ax: plt.Axes, data: pd.DataFrame, labels: list[str], panel_label: str, descriptor: str) -> None:
    ax.axvspan(-0.20, 0.20, color="#ECE9E1", alpha=0.72, zorder=0)
    ax.axvline(0, color="#555555", linewidth=0.9, zorder=1)
    offsets = {"Primary": 0.12, "Within climate commune": -0.12}
    positions = {label: len(labels) - index - 1 for index, label in enumerate(labels)}
    for specification in ("Primary", "Within climate commune"):
        subset = data.loc[data["specification"].eq(specification)]
        for row in subset.itertuples(index=False):
            y = positions[row.estimand] + offsets[specification]
            ax.errorbar(
                row.estimate,
                y,
                xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                fmt=MARKERS[specification],
                color=COLORS[specification],
                markersize=5.2,
                capsize=2.5,
                elinewidth=1.4,
                zorder=3,
            )
    ax.set_yticks([positions[label] for label in labels], labels)
    ax.set_ylim(-0.65, len(labels) - 0.35)
    ax.set_xlim(-0.22, 0.22)
    ax.grid(axis="x", color="#D8D8D8", linewidth=0.65)
    ax.set_xlabel("Southwest-minus-West response difference (outcome SD)")
    ax.text(-0.14, 1.04, panel_label, transform=ax.transAxes, fontweight="bold", fontsize=12)
    ax.text(0.00, 1.04, descriptor, transform=ax.transAxes, fontsize=10.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#444444")


def main() -> None:
    data = pd.concat([rainfall_rows(), temperature_rows()], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), gridspec_kw={"wspace": 0.42})
    draw_panel(
        axes[0],
        data.loc[data["panel"].eq("Annual land NPP")],
        ["Growing-season rainfall", "Hot days"],
        "a",
        "Annual land NPP",
    )
    draw_panel(
        axes[1],
        data.loc[data["panel"].eq("16-day EVI")],
        ["Dry rainfall, immediate", "Hot days", "Compound hot-dry"],
        "b",
        "16-day EVI",
    )
    handles = [
        plt.Line2D([], [], marker=MARKERS[name], color=COLORS[name], linestyle="none", label=name)
        for name in ("Primary", "Within climate commune")
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.subplots_adjust(bottom=0.20, left=0.15, right=0.98, top=0.88)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved estimates: {ESTIMATE_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
