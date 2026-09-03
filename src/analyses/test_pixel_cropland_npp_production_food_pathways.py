#!/usr/bin/env python3
"""Test pixel-level cropland MOD17 NPP against CSES production and food outcomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analyses"))
import test_absolute_ndvi_production_food_pathways as pathway  # noqa: E402


CSES = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
PIXEL_NPP = ROOT / "data/processed/cses_public_village_pixel_cropland_npp_annual_preprocessed.parquet"
CLIMATE_PANEL = ROOT / "data/processed/cambodia_public_village_monsoon_npp_panel_preprocessed.parquet"
OUTPUT = ROOT / "data/exp/analysis/climate-welfare/pixel-cropland-npp-production-food-pathways"

ID = "National Village Point ID"
WEIGHT = "Household Survey Weight"
FOOD = "Real 2021 Food Consumption Value per Household Member Riels"
OWN = "Real 2021 Own Produced Food Value per Household Member Riels"
CROP = "Real 2021 Crop Production Value Riels"
CROP_HA = "Real 2021 Crop Production Value per Cultivated ha Riels"
HEAT_SOURCE = "Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate B"
RAIN_SOURCE = "Village Buffer Mean Annual Precipitation Total mm"
HEAT = "Prior-year post-onset heat days (10 days)"
RAIN = "Prior-year annual precipitation (1000 mm)"

SUPPORT_THRESHOLDS = (1, 4, 10, 20)
DEFINITIONS = {
    "Strict cropland (IGBP 12)": {
        "value": "Annual Strict-Cropland Mean NPP kg C per m2",
        "valid": "Strict-Cropland Valid NPP 500m Pixel Count",
    },
    "Inclusive agriculture (IGBP 12+14)": {
        "value": "Annual Inclusive-Agriculture Mean NPP kg C per m2",
        "valid": "Inclusive-Agriculture Valid NPP 500m Pixel Count",
    },
}


def load_frame() -> pd.DataFrame:
    households = pd.read_parquet(CSES)
    households = households.loc[
        households["Climate Ecology Link Available"].eq(1)
        & pd.to_numeric(households[FOOD], errors="coerce").gt(0)
        & pd.to_numeric(households[WEIGHT], errors="coerce").gt(0)
    ].copy()
    households[ID] = households[ID].astype(str)
    households["Prior Calendar Year"] = (
        households["Interview Calendar Year"].astype(int) - 1
    )
    households["wave_month"] = (
        households["Survey Wave"].astype(str)
        + "_"
        + households["Interview Calendar Year"].astype("Int64").astype(str)
        + "-"
        + households["Interview Month"].astype("Int64").astype(str)
    )
    households["spatial_block"] = (
        np.floor((households["Point Longitude"] - 102.0) / 0.75)
        .astype("Int64")
        .astype(str)
        + "_"
        + np.floor((households["Point Latitude"] - 10.0) / 0.75)
        .astype("Int64")
        .astype(str)
    )

    npp = pd.read_parquet(PIXEL_NPP).rename(columns={"Year": "Prior Calendar Year"})
    npp[ID] = npp[ID].astype(str)
    npp_columns = [ID, "Prior Calendar Year"] + [
        column
        for definition in DEFINITIONS.values()
        for column in (definition["value"], definition["valid"])
    ]
    npp = npp[npp_columns]

    climate = pd.read_parquet(
        CLIMATE_PANEL,
        columns=[ID, "Year", "Buffer Radius km", HEAT_SOURCE, RAIN_SOURCE],
    )
    climate = climate.loc[climate["Buffer Radius km"].eq(5)].drop(
        columns="Buffer Radius km"
    )
    climate = climate.rename(columns={"Year": "Prior Calendar Year"})
    climate[ID] = climate[ID].astype(str)
    climate[HEAT] = pd.to_numeric(climate[HEAT_SOURCE], errors="coerce") / 10.0
    climate[RAIN] = pd.to_numeric(climate[RAIN_SOURCE], errors="coerce") / 1000.0
    climate = climate[[ID, "Prior Calendar Year", HEAT, RAIN]]

    frame = households.merge(
        npp, on=[ID, "Prior Calendar Year"], how="inner", validate="many_to_one"
    ).merge(
        climate, on=[ID, "Prior Calendar Year"], how="left", validate="many_to_one"
    )
    frame["Log total food per member"] = np.log(
        pd.to_numeric(frame[FOOD], errors="coerce")
    )
    own = pd.to_numeric(frame[OWN], errors="coerce")
    total = pd.to_numeric(frame[FOOD], errors="coerce")
    frame[pathway.OUTCOME_ANY_OWN] = own.gt(0).astype(float)
    frame[pathway.OUTCOME_LOG_OWN] = np.nan
    positive_own = own.gt(0)
    frame.loc[positive_own, pathway.OUTCOME_LOG_OWN] = np.log(own.loc[positive_own])
    market = total - own
    frame["Log market-acquired food per member"] = np.nan
    positive_market = market.gt(0)
    frame.loc[positive_market, "Log market-acquired food per member"] = np.log(
        market.loc[positive_market]
    )
    frame["Market-food accounting inconsistency"] = market.le(0)

    crop = pd.to_numeric(frame[CROP], errors="coerce")
    crop_ha = pd.to_numeric(frame[CROP_HA], errors="coerce")
    frame[pathway.OUTCOME_LOG_CROP] = np.where(
        crop.notna() & crop.ge(0), np.log1p(crop), np.nan
    )
    frame[pathway.OUTCOME_LOG_CROP_HA] = np.where(
        crop_ha.notna() & crop_ha.ge(0), np.log1p(crop_ha), np.nan
    )
    frame["Agricultural Participation"] = (
        frame["Agricultural Participation"].astype("boolean").astype(float)
    )
    wave_count = frame.groupby("Village Code", observed=True)["Survey Year"].nunique()
    repeated = wave_count.loc[lambda value: value > 1].index
    frame["Repeated village"] = frame["Village Code"].isin(repeated)
    return frame


def support_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for definition, spec in DEFINITIONS.items():
        for threshold in SUPPORT_THRESHOLDS:
            eligible = (
                pd.to_numeric(frame[spec["value"]], errors="coerce").notna()
                & pd.to_numeric(frame[spec["valid"]], errors="coerce").ge(threshold)
            )
            rows.append(
                {
                    "NPP Definition": definition,
                    "Minimum Valid Cropland Pixels": threshold,
                    "Households": int(eligible.sum()),
                    "Household Share": float(eligible.mean()),
                    "Villages": int(frame.loc[eligible, "Village Code"].nunique()),
                    "Communes": int(frame.loc[eligible, "Commune Code"].nunique()),
                    "Repeated-Village Households": int(
                        (eligible & frame["Repeated village"]).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def all_screen_holm(stage_a: pd.DataFrame) -> pd.Series:
    primary = stage_a["Specification"].eq("Commune FE")
    adjusted = pd.Series(np.nan, index=stage_a.index, dtype=float)
    adjusted.loc[primary] = pathway.holm_adjust(
        stage_a.loc[primary, "Probability Value"]
    )
    return adjusted


def stability_summary(stage_a: pd.DataFrame) -> pd.DataFrame:
    primary = stage_a.loc[stage_a["Specification"].eq("Commune FE")].copy()
    repeated = stage_a.loc[
        stage_a["Specification"].eq("Repeated-village sample, village FE")
    ][
        [
            "NPP Definition",
            "Minimum Valid Cropland Pixels",
            "Outcome",
            "Coefficient",
        ]
    ].rename(columns={"Coefficient": "Repeated-Village Coefficient"})
    merged = primary.merge(
        repeated,
        on=["NPP Definition", "Minimum Valid Cropland Pixels", "Outcome"],
        how="left",
        validate="one_to_one",
    )
    merged["Positive Commune Direction"] = merged["Coefficient"].gt(0)
    merged["Positive Repeated-Village Direction"] = merged[
        "Repeated-Village Coefficient"
    ].gt(0)
    merged["Within-Specification Holm Below 0.05"] = merged[
        "Holm-Adjusted Probability Value Across Four Primary Outcomes"
    ].lt(0.05)
    merged["Stable Positive Signal"] = (
        merged["Positive Commune Direction"]
        & merged["Positive Repeated-Village Direction"]
        & merged["Within-Specification Holm Below 0.05"]
    )
    return merged


def main() -> None:
    required = [CSES, PIXEL_NPP, CLIMATE_PANEL]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    support = support_table(frame)

    pathway.HEAT = HEAT
    pathway.RAIN = RAIN
    pathway.CLIMATE = [HEAT, RAIN]
    stage_a_parts: list[pd.DataFrame] = []
    stage_b_parts: list[pd.DataFrame] = []
    for definition, spec in DEFINITIONS.items():
        for threshold in SUPPORT_THRESHOLDS:
            sample = frame.loc[
                pd.to_numeric(frame[spec["value"]], errors="coerce").notna()
                & pd.to_numeric(frame[spec["valid"]], errors="coerce").ge(threshold)
            ].copy()
            term = f"Prior-year {definition} NPP (0.1 kg C per m2)"
            sample[term] = pd.to_numeric(sample[spec["value"]], errors="coerce") / 0.1
            pathway.NDVI = term
            a = pathway.stage_a(sample)
            b = pathway.stage_b(sample, a)
            for results in (a, b):
                results.insert(0, "NPP Definition", definition)
                results.insert(1, "Minimum Valid Cropland Pixels", threshold)
            stage_a_parts.append(a)
            stage_b_parts.append(b)
            print(
                f"Estimated {definition}, minimum {threshold} pixels; households={len(sample):,}",
                flush=True,
            )

    stage_a = pd.concat(stage_a_parts, ignore_index=True)
    stage_b = pd.concat(stage_b_parts, ignore_index=True)
    stage_a[
        "Holm-Adjusted Probability Value Across All 32 Commune-FE Screen Tests"
    ] = all_screen_holm(stage_a)
    stability = stability_summary(stage_a)

    outcomes = [
        pathway.OUTCOME_ANY_OWN,
        pathway.OUTCOME_LOG_OWN,
        pathway.OUTCOME_LOG_CROP,
        pathway.OUTCOME_LOG_CROP_HA,
    ]
    stable_counts = (
        stability.groupby("Outcome", observed=True)["Stable Positive Signal"]
        .agg(["sum", "count"])
        .reindex(outcomes)
    )
    summary = {
        "design": (
            "Prior complete calendar-year native 500 m MOD17 NPP from same-year MCD12Q1 "
            "cropland pixels within 5 km of CSES-linked villages"
        ),
        "npp_unit": "0.1 kg C per m2 per year; not standardized",
        "land_cover_definitions": list(DEFINITIONS),
        "minimum_valid_pixel_thresholds": list(SUPPORT_THRESHOLDS),
        "threshold_status": "all reported; no single human-approved primary threshold",
        "households_before_pixel_support_gate": int(len(frame)),
        "villages_before_pixel_support_gate": int(frame["Village Code"].nunique()),
        "stable_positive_signal_counts": {
            outcome: {
                "passing_specifications": int(stable_counts.loc[outcome, "sum"]),
                "tested_specifications": int(stable_counts.loc[outcome, "count"]),
            }
            for outcome in outcomes
        },
        "any_stable_positive_production_signal": bool(
            stability.loc[
                stability["Outcome"].isin(
                    [
                        pathway.OUTCOME_LOG_OWN,
                        pathway.OUTCOME_LOG_CROP,
                        pathway.OUTCOME_LOG_CROP_HA,
                    ]
                ),
                "Stable Positive Signal",
            ].any()
        ),
        "multiple_testing": {
            "within_specification": "Holm across four production outcomes",
            "global_screen": "Holm across 32 commune-FE mask-threshold-outcome tests",
        },
        "claim_limit": (
            "Associational repeated cross-section and repeated-village fixed-effect screen; "
            "MOD17 NPP is biological productivity, not harvested crop yield."
        ),
    }
    support.to_csv(OUTPUT / "sample_support.csv", index=False)
    stage_a.to_csv(OUTPUT / "stage_a_pixel_npp_to_production.csv", index=False)
    stage_b.to_csv(OUTPUT / "stage_b_production_to_food.csv", index=False)
    stability.to_csv(OUTPUT / "production_signal_stability.csv", index=False)
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Pixel-level cropland NPP, production, and food pathway screen\n\n"
        "The exposure is prior-calendar-year native 500 m MOD17 NPP restricted with annual "
        "MCD12Q1 cropland classes inside five-kilometre village buffers. Results report all "
        "declared support thresholds and remain exploratory fixed-effect associations.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSample support")
    print(support.to_string(index=False))
    print("\nCommune-FE production results")
    columns = [
        "NPP Definition",
        "Minimum Valid Cropland Pixels",
        "Outcome",
        "Coefficient",
        "Clustered Standard Error",
        "95 Percent CI Lower",
        "95 Percent CI Upper",
        "Probability Value",
        "Holm-Adjusted Probability Value Across Four Primary Outcomes",
        "Holm-Adjusted Probability Value Across All 32 Commune-FE Screen Tests",
        "Observations",
    ]
    print(
        stage_a.loc[stage_a["Specification"].eq("Commune FE"), columns].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
