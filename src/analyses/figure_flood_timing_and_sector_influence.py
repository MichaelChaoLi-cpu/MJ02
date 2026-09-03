#!/usr/bin/env python3
"""Flood Timing and Sector Influence.

Plan: Diagnose whether the 5 km compound hot-dry coefficient reflects 2011
inundation, transition-year coding, or one of the two conflict sectors.
Framework: AnaSOP Sections 5-7 full climate-interaction hierarchy with
sector-stratified calibration, village and stack-year fixed effects,
predetermined control-by-year terms, and spatial-block clustered inference.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/exp/.matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_conflict_conditioned_climate_sensitivity import (  # noqa: E402
    COMPOUND,
    CONTROL_FEATURES,
    DRY,
    HEAT,
    ID,
    SECTOR_YEARS,
    YEAR,
    add_control_by_year_terms,
    add_hazard_terms,
    build_stacked_panel,
    coefficient_row,
    fit_panel,
)


FLOOD = ROOT / "data/processed/cambodia_national_2011_gfd_flood_exposure_preprocessed.parquet"
CROSSWALK = ROOT / "data/processed/cambodia_public_village_buffer_grid_crosswalk/radius_5_km.parquet"
OUT = ROOT / "data/exp/legacy-results/figures/Figure_flood_timing_and_sector_influence.png"
AUDIT_DIR = ROOT / "data/exp/experiments/cambodia-thailand-village-area-itt-climate/flood-timing-sector"
ESTIMATES_OUT = AUDIT_DIR / "flood_timing_sector_estimates.csv"
EXPOSURE_OUT = AUDIT_DIR / "flood_exposure_summary.csv"
VILLAGE_FLOOD_OUT = AUDIT_DIR / "village_2011_flood_exposure_5km.csv"
METADATA_OUT = AUDIT_DIR / "model_metadata.json"

FLOOD_SHARE = "2011 Maximum Flooded Share Excluding Permanent Water"
CLEAR_SHARE = "Event 3853 Clear Observation Share"
TARGET = "Compound hot–dry: area x post x hazard"
PRIMARY_RADIUS_KM = 5
FLOOD_ADJUSTMENT_YEARS = [2011, 2012, 2013]


def aggregate_village_flood() -> pd.DataFrame:
    crosswalk = pd.read_parquet(CROSSWALK, columns=[ID, "National Grid Cell ID"])
    flood = pd.read_parquet(
        FLOOD,
        columns=["National Grid Cell ID", FLOOD_SHARE, CLEAR_SHARE],
    )
    merged = crosswalk.merge(flood, on="National Grid Cell ID", how="left", validate="many_to_one")
    village = merged.groupby(ID, as_index=False).agg(
        **{
            FLOOD_SHARE: (FLOOD_SHARE, "mean"),
            CLEAR_SHARE: (CLEAR_SHARE, "mean"),
            "Buffer Grid Cells": ("National Grid Cell ID", "size"),
            "Flood-observed Grid Cells": (FLOOD_SHARE, "count"),
            "Clear-observed Grid Cells": (CLEAR_SHARE, "count"),
        }
    )
    return village


def renormalize_weights(sample: pd.DataFrame) -> pd.DataFrame:
    output = sample.copy()
    sector_counts = (
        output.loc[output["Area ITT"].eq(1), ["Conflict Sector", ID]]
        .drop_duplicates()
        .groupby("Conflict Sector")
        .size()
    )
    total_treated = int(sector_counts.sum())
    for sector, treated_count in sector_counts.items():
        sector_share = treated_count / total_treated
        for area_value in [0, 1]:
            mask = output["Conflict Sector"].eq(sector) & output["Area ITT"].eq(area_value)
            current = output.loc[mask, "Analysis Weight"].sum() / output.loc[mask, YEAR].nunique()
            if current <= 0:
                raise RuntimeError(f"Empty analysis-weight arm for {sector}, area={area_value}")
            output.loc[mask, "Analysis Weight"] *= (sector_share / 2) / current
    return output


def build_compound_terms(sample: pd.DataFrame) -> tuple[list[str], str]:
    drought_terms = add_hazard_terms(sample, DRY, "Drought control")
    heat_terms = add_hazard_terms(sample, HEAT, "Heat control")
    compound_terms = add_hazard_terms(sample, COMPOUND, "Compound hot–dry")
    terms = [*drought_terms, *heat_terms, *compound_terms]
    direct_terms = [term for term in terms if term.endswith(": area x post")]
    keep_direct = direct_terms[0]
    for term in direct_terms[1:]:
        sample.drop(columns=term, inplace=True)
    terms = [term for term in terms if not term.endswith(": area x post") or term == keep_direct]
    return terms, compound_terms[-1]


def fit_specification(
    sample: pd.DataFrame,
    specification: str,
    family: str,
    flood_adjusted: bool = False,
) -> dict[str, object]:
    model_sample = renormalize_weights(sample)
    structural_terms, target = build_compound_terms(model_sample)
    extra_terms: list[str] = []
    if flood_adjusted:
        for year in FLOOD_ADJUSTMENT_YEARS:
            flood_term = f"Flood share x {year}"
            clear_term = f"Clear observation x {year}"
            model_sample[flood_term] = model_sample[FLOOD_SHARE] * model_sample[YEAR].eq(year).astype(int)
            model_sample[clear_term] = model_sample[CLEAR_SHARE] * model_sample[YEAR].eq(year).astype(int)
            extra_terms.extend([flood_term, clear_term])
    control_terms = add_control_by_year_terms(model_sample)
    fitted = fit_panel(model_sample, [*structural_terms, *extra_terms, *control_terms])
    row = coefficient_row(
        fitted,
        target,
        Model="Compound hot–dry amplification",
        Hazard="Compound hot–dry",
        **{
            "Buffer Radius km": PRIMARY_RADIUS_KM,
            "Estimand": "Post-conflict change in affected-area compound-shock slope",
        },
    )
    row.update(
        {
            "Specification Family": family,
            "Specification": specification,
            "Treated Villages": int(model_sample.loc[model_sample["Area ITT"].eq(1), ID].nunique()),
            "Control Villages": int(model_sample.loc[model_sample["Area ITT"].eq(0), ID].nunique()),
            "Conflict Sectors": int(model_sample.loc[model_sample["Area ITT"].eq(1), "Conflict Sector"].nunique()),
            "Flood Adjusted": flood_adjusted,
            "Included Years": ", ".join(map(str, sorted(model_sample[YEAR].unique()))),
        }
    )
    return row


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    finite = values.notna() & weights.notna()
    return float(np.average(values.loc[finite], weights=weights.loc[finite]))


def exposure_summary(stacked: pd.DataFrame) -> pd.DataFrame:
    village = stacked.drop_duplicates(["Conflict Sector", ID]).copy()
    rows: list[dict[str, object]] = []
    for (sector, area), frame in village.groupby(["Conflict Sector", "Area ITT"], observed=True):
        rows.append(
            {
                "Conflict Sector": sector,
                "Group": "Affected area" if area == 1 else "Calibrated comparison",
                "Villages": int(frame[ID].nunique()),
                "Weighted Maximum Flooded Share": weighted_mean(frame[FLOOD_SHARE], frame["Analysis Weight"]),
                "Weighted Clear Observation Share": weighted_mean(frame[CLEAR_SHARE], frame["Analysis Weight"]),
                "Weighted Below 10 Percent Flooded Share": weighted_mean(
                    (frame[FLOOD_SHARE].lt(0.10) & frame[CLEAR_SHARE].ge(0.80)).astype(float),
                    frame["Analysis Weight"],
                ),
            }
        )
    return pd.DataFrame(rows)


def forest(ax: plt.Axes, frame: pd.DataFrame, label_column: str, colors: list[str]) -> None:
    data = frame.reset_index(drop=True)
    y = np.arange(len(data))
    for index, row in data.iterrows():
        ax.errorbar(
            row["Estimate"],
            y[index],
            xerr=[[row["Estimate"] - row["95% CI Lower"]], [row["95% CI Upper"] - row["Estimate"]]],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=3,
            elinewidth=1.5,
            markersize=6,
        )
    ax.axvline(0, color="#5A5A5A", linewidth=0.9)
    ax.set_yticks(y, data[label_column])
    ax.invert_yaxis()
    ax.grid(True, axis="x", color="#E3E7EA", linewidth=0.6)
    ax.grid(False, axis="y")


def annotate_estimates(ax: plt.Axes, frame: pd.DataFrame) -> None:
    """Print point estimates without obscuring confidence intervals."""
    data = frame.reset_index(drop=True)
    for index, row in data.iterrows():
        ax.annotate(
            f"{row['Estimate']:.2f}",
            xy=(row["Estimate"], index),
            xytext=(0, -11),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7,
            color="#333333",
        )


def draw_figure(exposure: pd.DataFrame, estimates: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.4), constrained_layout=True)

    exposure = exposure.copy()
    exposure["Label"] = exposure["Conflict Sector"].str.replace("Ta Moan-Ta Krabey", "Ta Moan–Ta Krabey", regex=False)
    sectors = list(SECTOR_YEARS)
    positions = np.arange(len(sectors))
    group_styles = {
        "Affected area": ("#C4493D", -0.10),
        "Calibrated comparison": ("#2F6B8A", 0.10),
    }
    for group, (color, offset) in group_styles.items():
        part = exposure.loc[exposure["Group"].eq(group)].set_index("Conflict Sector").loc[sectors]
        axes[0].scatter(
            part["Weighted Maximum Flooded Share"] * 100,
            positions + offset,
            color=color,
            s=42,
            label=group,
            zorder=3,
        )
    axes[0].set_yticks(positions, ["Preah Vihear", "Ta Moan–Ta Krabey"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Weighted 2011 maximum flooded share (%)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    axes[0].set_title("Observed inundation", fontsize=10, pad=10)
    axes[0].grid(True, axis="x", color="#E3E7EA", linewidth=0.6)
    axes[0].grid(False, axis="y")

    timing = estimates.loc[estimates["Specification Family"].eq("Flood and timing")].copy()
    timing_order = [
        "Primary",
        "Flood-observed sample",
        "Flood-adjusted 2011–2013",
        "Low-flood sample",
        "Exclude sector onset years",
        "Exclude 2011–2013",
    ]
    timing["order"] = timing["Specification"].map({value: index for index, value in enumerate(timing_order)})
    timing = timing.sort_values("order")
    forest(axes[1], timing, "Specification", ["#C4493D"] + ["#6B7C85"] * (len(timing) - 1))
    axes[1].set_xlabel("Compound hot–dry slope-change coefficient")
    axes[1].set_title("Flood and timing checks", fontsize=10, pad=10)

    sector = estimates.loc[estimates["Specification Family"].eq("Sector influence")].copy()
    sector_order = ["Pooled sectors", "Preah Vihear only", "Ta Moan–Ta Krabey only"]
    sector["order"] = sector["Specification"].map({value: index for index, value in enumerate(sector_order)})
    sector = sector.sort_values("order")
    forest(axes[2], sector, "Specification", ["#C4493D", "#A65A2E", "#7B4F9D"])
    axes[2].set_xscale("symlog", linthresh=2, linscale=1.4)
    axes[2].set_xlim(-40, 0.3)
    axes[2].set_xticks([-40, -20, -10, -5, -2, 0])
    axes[2].set_xticklabels(["−40", "−20", "−10", "−5", "−2", "0"])
    axes[2].set_xlabel("Compound hot–dry coefficient (symmetric-log scale)")
    axes[2].set_title("Conflict-sector influence", fontsize=10, pad=10)
    annotate_estimates(axes[2], sector)

    for label, ax in zip("abc", axes):
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    village_flood = aggregate_village_flood()
    village_flood.to_csv(VILLAGE_FLOOD_OUT, index=False)
    stacked = build_stacked_panel(PRIMARY_RADIUS_KM).merge(village_flood, on=ID, how="left", validate="many_to_one")
    exposure = exposure_summary(stacked)
    exposure.to_csv(EXPOSURE_OUT, index=False)

    rows: list[dict[str, object]] = []
    rows.append(fit_specification(stacked, "Primary", "Flood and timing"))
    observed = stacked.loc[stacked[CLEAR_SHARE].gt(0)].copy()
    rows.append(fit_specification(observed, "Flood-observed sample", "Flood and timing"))
    rows.append(
        fit_specification(
            observed,
            "Flood-adjusted 2011–2013",
            "Flood and timing",
            flood_adjusted=True,
        )
    )
    low_flood = stacked.loc[stacked[CLEAR_SHARE].ge(0.80) & stacked[FLOOD_SHARE].lt(0.10)].copy()
    rows.append(fit_specification(low_flood, "Low-flood sample", "Flood and timing"))
    onset_mask = np.zeros(len(stacked), dtype=bool)
    for sector, first_year in SECTOR_YEARS.items():
        onset_mask |= stacked["Conflict Sector"].eq(sector) & stacked[YEAR].eq(first_year)
    rows.append(
        fit_specification(
            stacked.loc[~onset_mask].copy(),
            "Exclude sector onset years",
            "Flood and timing",
        )
    )
    rows.append(
        fit_specification(
            stacked.loc[~stacked[YEAR].isin(FLOOD_ADJUSTMENT_YEARS)].copy(),
            "Exclude 2011–2013",
            "Flood and timing",
        )
    )

    rows.append(fit_specification(stacked, "Pooled sectors", "Sector influence"))
    for sector, label in [
        ("Preah Vihear", "Preah Vihear only"),
        ("Ta Moan-Ta Krabey", "Ta Moan–Ta Krabey only"),
    ]:
        rows.append(
            fit_specification(
                stacked.loc[stacked["Conflict Sector"].eq(sector)].copy(),
                label,
                "Sector influence",
            )
        )
    estimates = pd.DataFrame(rows)
    estimates.to_csv(ESTIMATES_OUT, index=False)
    metadata = {
        "status": "first flood, timing, and sector-influence experiment",
        "human_decision_record": "MILI-D-20260823-010",
        "primary_buffer_km": PRIMARY_RADIUS_KM,
        "flood_source_fields": [FLOOD_SHARE, CLEAR_SHARE],
        "village_flood_aggregation": "mean of Cambodia 1 km land-grid-cell values within the frozen 5 km village buffer",
        "flood_adjustment_years": FLOOD_ADJUSTMENT_YEARS,
        "low_flood_rule": "clear observation share >= 0.80 and maximum flooded share < 0.10",
        "weights": "sector-stratified calibration renormalized within retained sector and treatment arm",
        "fixed_effects": "stack-village and stack-year",
        "predetermined_controls": CONTROL_FEATURES,
        "inference": "10 km spatial-block clustered covariance with debiasing",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    METADATA_OUT.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    draw_figure(exposure, estimates)
    print(f"Saved: {OUT.relative_to(ROOT)}")
    print(f"Saved: {ESTIMATES_OUT.relative_to(ROOT)}")
    print("\nFlood exposure")
    print(exposure.to_string(index=False))
    print("\nCompound hot-dry estimates")
    print(
        estimates[
            [
                "Specification Family",
                "Specification",
                "Estimate",
                "95% CI Lower",
                "95% CI Upper",
                "p-value",
                "Treated Villages",
                "Control Villages",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
