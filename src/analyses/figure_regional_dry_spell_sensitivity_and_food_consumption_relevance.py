#!/usr/bin/env python3
"""Regional Dry-Spell Sensitivity and Food-Consumption Relevance.

Plan: Compare six frozen outcome-blind regions on the NPP change associated
with the pooled P10-to-P90 dry-spell contrast and the expanded-control food-
consumption association. The two axes remain separate evidence dimensions.
Framework: AnaSOP Sections 5-7 regional interaction models, natural-unit
translations, national reference lines, fixed classification rules, survey
weights, exact survey-time effects, and village-clustered uncertainty.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_cropland_npp_and_household_consumption import (
    COMPOSITION_CONTROLS,
    FOOD_FLAG,
    FOOD_OUTCOME,
    ID,
    REGION,
    SOCIOECONOMIC_CONTROLS,
    estimate_national_ladder,
    estimate_regional_slopes,
    load_data,
)


ROOT = Path(__file__).resolve().parents[2]
STAGE1_PANEL = ROOT / "data/processed/cses_public_village_cropland_npp_absolute_climate_panel_preprocessed.parquet"
REGIONS = ROOT / "data/processed/outcome_blind_spatial_regions_preprocessed.parquet"
REGIONAL_CLIMATE = ROOT / "data/exp/analysis/climate-npp/outcome-blind-regions-and-zonal-climate-to-npp-responses/regional_climate_to_npp_coefficients.csv"
NATIONAL_CLIMATE = ROOT / "data/exp/analysis/climate-npp/national-climate-to-npp-responses/national_climate_npp_coefficients.csv"
OUTPUT = ROOT / "data/results/figures/Figure_regional_dry_spell_sensitivity_and_food_consumption_relevance.png"
EVIDENCE = ROOT / "data/exp/analysis/climate-npp/regional-dry-spell-sensitivity-and-food-consumption-relevance"

YEAR = "Year"
NPP = "Annual Strict-Cropland Mean NPP kg C per m2"
HEAT = "Village Buffer Mean Annual Heat Days at or Above 35 C"
RX5DAY = "Village Buffer Mean Annual Maximum Consecutive Five-Day Precipitation Rx5day mm"
DRY = "Village Buffer Mean Annual Maximum Consecutive Dry Days Below 1 mm"
RAIN = "Village Buffer Mean Annual Precipitation Total mm"

REGION_COLORS = ["#173F5F", "#2F80A2", "#3A9D8F", "#D4A72C", "#D97757", "#9B4A63"]
DARK = "#37444A"
GRID = "#D9DEE1"


def stage1_support() -> tuple[pd.DataFrame, float, float, float]:
    columns = [ID, YEAR, NPP, HEAT, RX5DAY, DRY, RAIN]
    panel = pd.read_parquet(STAGE1_PANEL, columns=columns)
    for column in [YEAR, NPP, HEAT, RX5DAY, DRY, RAIN]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.loc[panel[YEAR].between(2001, 2021)].dropna(subset=columns).copy()
    regions = pd.read_parquet(REGIONS, columns=[ID, REGION])
    panel = panel.merge(regions, on=ID, how="left", validate="many_to_one")
    if panel[REGION].isna().any():
        raise ValueError("Stage-1 analytical villages are missing frozen SKATER regions")
    p10 = float(panel[DRY].quantile(0.10))
    p90 = float(panel[DRY].quantile(0.90))
    national_mean = float(panel[NPP].mean())
    return panel, p10, p90, national_mean


def expanded_food_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    households = load_data()
    controls = [*COMPOSITION_CONTROLS, *SOCIOECONOMIC_CONTROLS]
    regional, equality = estimate_regional_slopes(
        households,
        FOOD_OUTCOME,
        FOOD_FLAG,
        "Food consumption",
        controls=controls,
        specification="Frozen SKATER regional slope + socioeconomic controls",
    )
    national = estimate_national_ladder(
        households,
        FOOD_OUTCOME,
        FOOD_FLAG,
        "Food consumption",
    )
    national_row = national.loc[national["Specification"].eq("+ Socioeconomic controls")].iloc[0]
    required = [ID, REGION, "Exact Survey Time", FOOD_OUTCOME, *controls]
    sample = households.loc[households[FOOD_FLAG].fillna(False).astype(bool)].dropna(subset=required)
    support = (
        sample.groupby(REGION, as_index=False)
        .agg(Households=(ID, "size"), Household_Villages=(ID, "nunique"))
        .rename(columns={REGION: "Region ID"})
    )
    return regional, equality, national_row, support


def assemble_evidence() -> tuple[pd.DataFrame, dict[str, float]]:
    panel, p10, p90, national_mean = stage1_support()
    contrast = p90 - p10
    region_means = (
        panel.groupby(REGION, as_index=False)
        .agg(
            Region_Mean_NPP=(NPP, "mean"),
            Stage1_Observations=(ID, "size"),
            Stage1_Villages=(ID, "nunique"),
        )
        .rename(columns={REGION: "Region ID"})
    )

    climate = pd.read_csv(REGIONAL_CLIMATE)
    climate = climate.loc[
        climate["Algorithm"].eq("SKATER") & climate["Exposure"].eq("Maximum dry spell")
    ].copy()
    climate = climate[
        [
            "Region ID",
            "Estimate",
            "Clustered Standard Error",
            "95 Percent CI Lower",
            "95 Percent CI Upper",
            "Probability Value",
        ]
    ].rename(
        columns={
            "Estimate": "Dry_Spell_Coefficient_per_10_Days",
            "Clustered Standard Error": "Dry_Spell_Clustered_SE",
            "95 Percent CI Lower": "Dry_Spell_CI_Lower",
            "95 Percent CI Upper": "Dry_Spell_CI_Upper",
            "Probability Value": "Dry_Spell_P_Value",
        }
    )

    food, equality, national_food, support = expanded_food_results()
    food = food[
        [
            "Region ID",
            "NPP Coefficient",
            "Clustered Standard Error",
            "95 Percent CI Lower",
            "95 Percent CI Upper",
            "Probability Value",
            "Percent Difference per 0.1 NPP",
            "Percent Difference 95 Percent CI Lower",
            "Percent Difference 95 Percent CI Upper",
        ]
    ].rename(
        columns={
            "NPP Coefficient": "Food_NPP_Coefficient",
            "Clustered Standard Error": "Food_NPP_Clustered_SE",
            "95 Percent CI Lower": "Food_NPP_CI_Lower",
            "95 Percent CI Upper": "Food_NPP_CI_Upper",
            "Probability Value": "Food_NPP_P_Value",
            "Percent Difference per 0.1 NPP": "Food_Percent_Difference",
            "Percent Difference 95 Percent CI Lower": "Food_Percent_CI_Lower",
            "Percent Difference 95 Percent CI Upper": "Food_Percent_CI_Upper",
        }
    )

    evidence = climate.merge(region_means, on="Region ID", validate="one_to_one")
    evidence = evidence.merge(food, on="Region ID", validate="one_to_one")
    evidence = evidence.merge(support, on="Region ID", validate="one_to_one")
    multiplier = contrast / 10.0
    evidence["P10_P90_Dry_Spell_Contrast_Days"] = contrast
    evidence["P10_P90_NPP_Change_kg_C_per_m2"] = (
        evidence["Dry_Spell_Coefficient_per_10_Days"] * multiplier
    )
    evidence["P10_P90_NPP_Change_kg_C_per_ha"] = (
        evidence["P10_P90_NPP_Change_kg_C_per_m2"] * 10000.0
    )
    evidence["P10_P90_NPP_Change_Percent"] = (
        100.0 * evidence["P10_P90_NPP_Change_kg_C_per_m2"] / evidence["Region_Mean_NPP"]
    )
    evidence["P10_P90_NPP_Loss_Percent"] = -evidence["P10_P90_NPP_Change_Percent"]
    evidence["P10_P90_NPP_Loss_CI_Lower"] = (
        -100.0 * evidence["Dry_Spell_CI_Upper"] * multiplier / evidence["Region_Mean_NPP"]
    )
    evidence["P10_P90_NPP_Loss_CI_Upper"] = (
        -100.0 * evidence["Dry_Spell_CI_Lower"] * multiplier / evidence["Region_Mean_NPP"]
    )

    national_climate = pd.read_csv(NATIONAL_CLIMATE)
    national_dry = national_climate.loc[
        national_climate["Model"].eq("Primary heat-day specification")
        & national_climate["Term"].eq("Maximum dry spell per 10 days")
    ].iloc[0]
    national_loss = -100.0 * float(national_dry["Estimate"]) * multiplier / national_mean
    national_food_percent = float(national_food["Percent Difference per 0.1 NPP"])

    ecological_elevated = (
        evidence["P10_P90_NPP_Loss_Percent"].gt(national_loss)
        & evidence["P10_P90_NPP_Loss_CI_Lower"].gt(0)
    )
    household_elevated = (
        evidence["Food_Percent_Difference"].gt(national_food_percent)
        & evidence["Food_Percent_CI_Lower"].gt(0)
    )
    evidence["Evidence_Type"] = "Mixed or uncertain"
    evidence.loc[ecological_elevated & household_elevated, "Evidence_Type"] = "Jointly elevated"
    evidence.loc[ecological_elevated & ~household_elevated, "Evidence_Type"] = "Ecological sensitivity only"
    evidence.loc[~ecological_elevated & household_elevated, "Evidence_Type"] = "Household relevance only"
    evidence.loc[
        evidence["P10_P90_NPP_Loss_CI_Lower"].gt(0)
        & evidence["Food_Percent_Difference"].lt(0),
        "Evidence_Type",
    ] = "Discordant"
    evidence.insert(0, "Region", "R" + evidence["Region ID"].astype(int).astype(str))

    metadata = {
        "dry_spell_p10_days": p10,
        "dry_spell_p90_days": p90,
        "dry_spell_contrast_days": contrast,
        "national_mean_npp_kg_c_per_m2": national_mean,
        "national_dry_spell_coefficient_per_10_days": float(national_dry["Estimate"]),
        "national_dry_spell_loss_percent": national_loss,
        "national_expanded_control_food_percent_difference": national_food_percent,
        "expanded_control_regional_food_equality_p_value": float(equality["Probability Value"].iloc[0]),
    }
    return evidence.sort_values("Region ID").reset_index(drop=True), metadata


def draw_figure(evidence: pd.DataFrame, metadata: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 7.1), facecolor="white")
    ax.axvline(
        metadata["national_dry_spell_loss_percent"],
        color="#6F7C82",
        linewidth=1.1,
        linestyle="--",
        zorder=1,
    )
    ax.axhline(
        metadata["national_expanded_control_food_percent_difference"],
        color="#6F7C82",
        linewidth=1.1,
        linestyle="--",
        zorder=1,
    )
    for index, row in evidence.iterrows():
        x = float(row["P10_P90_NPP_Loss_Percent"])
        y = float(row["Food_Percent_Difference"])
        xerr = np.array(
            [[x - float(row["P10_P90_NPP_Loss_CI_Lower"])],
             [float(row["P10_P90_NPP_Loss_CI_Upper"]) - x]]
        )
        yerr = np.array(
            [[y - float(row["Food_Percent_CI_Lower"])],
             [float(row["Food_Percent_CI_Upper"]) - y]]
        )
        size = 75.0 + 185.0 * float(row["Households"]) / float(evidence["Households"].max())
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor=REGION_COLORS[index],
            elinewidth=1.35,
            capsize=3.0,
            alpha=0.78,
            zorder=2,
        )
        ax.scatter(
            x,
            y,
            s=size,
            color=REGION_COLORS[index],
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        offsets = {1: (-14, 9), 2: (8, 9), 3: (8, -14), 4: (8, 9), 5: (8, -15), 6: (8, 9)}
        dx, dy = offsets[int(row["Region ID"])]
        ax.annotate(
            row["Region"],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=10.2,
            fontweight="bold",
            color=REGION_COLORS[index],
        )

    ax.text(
        metadata["national_dry_spell_loss_percent"],
        ax.get_ylim()[1] if ax.get_ylim()[1] else 1,
        " National dry-spell benchmark",
        fontsize=8.4,
        color="#59656B",
        ha="left",
        va="top",
    )
    ax.text(
        0.99,
        metadata["national_expanded_control_food_percent_difference"],
        "National food-consumption benchmark ",
        transform=ax.get_yaxis_transform(),
        fontsize=8.4,
        color="#59656B",
        ha="right",
        va="bottom",
    )
    ax.grid(color=GRID, linewidth=0.6, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("NPP decline over the pooled P10–P90 dry-spell contrast (% of regional mean)", fontsize=10.2)
    ax.set_ylabel("Food-consumption difference per 0.1 kg C m⁻² higher prior-year NPP (%)", fontsize=10.2)
    ax.tick_params(labelsize=9.2, colors=DARK)
    for spine in ax.spines.values():
        spine.set_color("#89969C")
        spine.set_linewidth(0.75)
    ax.margins(x=0.10, y=0.14)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    evidence, metadata = assemble_evidence()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(EVIDENCE / "regional_priority_evidence.csv", index=False)
    (EVIDENCE / "regional_priority_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    draw_figure(evidence, metadata)
    print(evidence[[
        "Region",
        "P10_P90_NPP_Loss_Percent",
        "Food_Percent_Difference",
        "Food_Percent_CI_Lower",
        "Food_Percent_CI_Upper",
        "Households",
        "Evidence_Type",
    ]].to_string(index=False))
    print(json.dumps(metadata, indent=2))
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved evidence: {EVIDENCE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
