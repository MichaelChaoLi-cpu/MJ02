#!/usr/bin/env python3
"""Test same-production-year cropland NPP against aligned CSES crop outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
HOUSEHOLDS = ROOT / "data/processed/cses_household_reference_year_crop_production_preprocessed.parquet"
DEFAULT_PIXEL_NPP = ROOT / "data/processed/cses_public_village_pixel_cropland_npp_annual_preprocessed.parquet"
CLIMATE = ROOT / "data/processed/cambodia_public_village_monsoon_ecology_panel_preprocessed.parquet"
DEFAULT_OUTPUT = ROOT / "data/exp/analysis/climate-welfare/reference-year-pixel-cropland-npp-production"

ID = "National Village Point ID"
YEAR = "Crop Production Reference Year"
WEIGHT = "Household Survey Weight"
TIME = "Survey-wave interview time"
BLOCK = "Spatial block"
HEAT_SOURCE = "Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate B"
RAIN_SOURCE = "Village Buffer Mean May October Precipitation Total mm Anomaly Z"
HEAT = "Production-year post-onset heat days per 10"
RAIN = "Production-year May-October rainfall anomaly Z"

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
OUTCOMES = {
    "Log1p crop production quantity kg": "Crop Production Quantity kg",
    "Log1p crop yield kg per ha": "Crop Yield kg per ha",
    "Log1p real crop production value": "Real 2021 Crop Production Value Riels",
    "Log1p real crop value per cultivated ha": "Real 2021 Crop Production Value per Cultivated ha Riels",
}
COMPOSITION = [
    "Household Size",
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
]


def holm_adjust(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    valid = values.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    running = 0.0
    total = len(valid)
    for rank, (index, value) in enumerate(valid.items(), start=1):
        running = max(running, min(1.0, float(value) * (total - rank + 1)))
        adjusted.loc[index] = running
    return adjusted


def load_frame(pixel_npp: Path, buffer_radius_km: float) -> pd.DataFrame:
    households = pd.read_parquet(HOUSEHOLDS).copy()
    households[ID] = households[ID].astype(str)
    households[TIME] = (
        households["Survey Wave"].astype(str)
        + "_"
        + households["Interview Calendar Year"].astype("Int64").astype(str)
        + "-"
        + households["Interview Month"].astype("Int64").astype(str)
    )
    households[BLOCK] = (
        np.floor((households["Point Longitude"] - 102.0) / 0.75)
        .astype("Int64")
        .astype(str)
        + "_"
        + np.floor((households["Point Latitude"] - 10.0) / 0.75)
        .astype("Int64")
        .astype(str)
    )
    for outcome, source in OUTCOMES.items():
        values = pd.to_numeric(households[source], errors="coerce")
        households[outcome] = np.where(values.ge(0), np.log1p(values), np.nan)

    npp = pd.read_parquet(pixel_npp).rename(columns={"Year": YEAR})
    npp[ID] = npp[ID].astype(str)
    npp_columns = [ID, YEAR] + [
        column
        for definition in DEFINITIONS.values()
        for column in (definition["value"], definition["valid"])
    ]
    npp = npp[npp_columns]

    climate = pd.read_parquet(
        CLIMATE,
        columns=[ID, "Year", "Buffer Radius km", HEAT_SOURCE, RAIN_SOURCE],
    )
    climate = climate.loc[climate["Buffer Radius km"].eq(buffer_radius_km)].drop(
        columns="Buffer Radius km"
    )
    climate = climate.rename(columns={"Year": YEAR})
    climate[ID] = climate[ID].astype(str)
    climate[HEAT] = pd.to_numeric(climate[HEAT_SOURCE], errors="coerce") / 10.0
    climate[RAIN] = pd.to_numeric(climate[RAIN_SOURCE], errors="coerce")
    climate = climate[[ID, YEAR, HEAT, RAIN]]

    frame = households.merge(
        npp, on=[ID, YEAR], how="inner", validate="many_to_one"
    ).merge(climate, on=[ID, YEAR], how="left", validate="many_to_one")
    repeated = (
        frame.groupby("Village Code", observed=True)["Survey Wave"].nunique().gt(1)
    )
    frame["Repeated village"] = frame["Village Code"].isin(repeated[repeated].index)
    return frame


def fit_model(
    frame: pd.DataFrame,
    outcome: str,
    term: str,
    area: str,
    controls: list[str],
) -> tuple[object, pd.DataFrame]:
    required = [outcome, term, area, TIME, YEAR, WEIGHT, BLOCK, *controls]
    sample = frame.dropna(subset=required).copy()
    sample = sample.loc[pd.to_numeric(sample[WEIGHT], errors="coerce").gt(0)].copy()
    absorb = pd.DataFrame(
        {
            "Area": sample[area].astype("category"),
            "Survey time": sample[TIME].astype("category"),
            "Production year": sample[YEAR].astype("category"),
        },
        index=sample.index,
    )
    model = AbsorbingLS(
        sample[outcome].astype(float),
        sample[[term, *controls]].astype(float),
        absorb=absorb,
        weights=sample[WEIGHT].astype(float),
        drop_absorbed=True,
    )
    result = model.fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample[BLOCK]).codes,
        debiased=True,
    )
    return result, sample


def estimate(frame: pd.DataFrame, term: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = [
        ("District FE", frame, "District Code", [HEAT, RAIN]),
        ("Commune FE", frame, "Commune Code", [HEAT, RAIN]),
        (
            "Commune FE, composition adjusted",
            frame,
            "Commune Code",
            [HEAT, RAIN, *COMPOSITION],
        ),
        (
            "Repeated-village sample, village FE",
            frame.loc[frame["Repeated village"]].copy(),
            "Village Code",
            [HEAT, RAIN],
        ),
    ]
    for outcome in OUTCOMES:
        for specification, sample_frame, area, controls in specifications:
            result, sample = fit_model(sample_frame, outcome, term, area, controls)
            interval = result.conf_int(level=0.95).loc[term]
            rows.append(
                {
                    "Outcome": outcome,
                    "Specification": specification,
                    "Term": term,
                    "Coefficient": float(result.params[term]),
                    "Clustered Standard Error": float(result.std_errors[term]),
                    "95 Percent CI Lower": float(interval["lower"]),
                    "95 Percent CI Upper": float(interval["upper"]),
                    "Probability Value": float(result.pvalues[term]),
                    "Observations": int(result.nobs),
                    "Area Units": int(sample[area].nunique()),
                    "Spatial Blocks": int(sample[BLOCK].nunique()),
                    "Fixed Effects": (
                        f"{area} + survey-wave-by-interview-year-month + production year"
                    ),
                    "Controls": ", ".join(controls),
                    "Survey Weighted": True,
                }
            )
    results = pd.DataFrame(rows)
    primary = results["Specification"].eq("Commune FE")
    results["Holm-Adjusted Probability Value Across Four Outcomes"] = np.nan
    results.loc[primary, "Holm-Adjusted Probability Value Across Four Outcomes"] = (
        holm_adjust(results.loc[primary, "Probability Value"])
    )
    return results


def main(pixel_npp: Path, output_dir: Path, buffer_radius_km: float) -> None:
    for path in (HOUSEHOLDS, pixel_npp, CLIMATE):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_frame(pixel_npp, buffer_radius_km)
    support_rows: list[dict[str, object]] = []
    result_parts: list[pd.DataFrame] = []
    for definition, columns in DEFINITIONS.items():
        for threshold in SUPPORT_THRESHOLDS:
            eligible = (
                pd.to_numeric(frame[columns["value"]], errors="coerce").notna()
                & pd.to_numeric(frame[columns["valid"]], errors="coerce").ge(threshold)
            )
            sample = frame.loc[eligible].copy()
            term = f"Same-year {definition} NPP per 0.1 kg C per m2"
            sample[term] = pd.to_numeric(sample[columns["value"]], errors="coerce") / 0.1
            results = estimate(sample, term)
            results.insert(0, "NPP Definition", definition)
            results.insert(1, "Minimum Valid Cropland Pixels", threshold)
            result_parts.append(results)
            support_rows.append(
                {
                    "NPP Definition": definition,
                    "Minimum Valid Cropland Pixels": threshold,
                    "Households": int(len(sample)),
                    "Villages": int(sample["Village Code"].nunique()),
                    "Communes": int(sample["Commune Code"].nunique()),
                    "Production Years": int(sample[YEAR].nunique()),
                    "Repeated-Village Households": int(sample["Repeated village"].sum()),
                }
            )
            print(f"Estimated {definition}, threshold {threshold}: {len(sample):,} households", flush=True)

    results = pd.concat(result_parts, ignore_index=True)
    support = pd.DataFrame(support_rows)
    primary = results["Specification"].eq("Commune FE")
    results["Global Holm-Adjusted Probability Value Across 32 Primary Tests"] = np.nan
    results.loc[primary, "Global Holm-Adjusted Probability Value Across 32 Primary Tests"] = (
        holm_adjust(results.loc[primary, "Probability Value"])
    )
    commune = results.loc[primary].copy()
    repeated = results.loc[
        results["Specification"].eq("Repeated-village sample, village FE"),
        ["NPP Definition", "Minimum Valid Cropland Pixels", "Outcome", "Coefficient"],
    ].rename(columns={"Coefficient": "Repeated-Village Coefficient"})
    stability = commune.merge(
        repeated,
        on=["NPP Definition", "Minimum Valid Cropland Pixels", "Outcome"],
        how="left",
        validate="one_to_one",
    )
    stability["Positive in Commune FE"] = stability["Coefficient"].gt(0)
    stability["Positive in Repeated-Village FE"] = stability[
        "Repeated-Village Coefficient"
    ].gt(0)
    stability["Within-Specification Holm Below 0.05"] = stability[
        "Holm-Adjusted Probability Value Across Four Outcomes"
    ].lt(0.05)
    stability["Stable Positive Signal"] = (
        stability["Positive in Commune FE"]
        & stability["Positive in Repeated-Village FE"]
        & stability["Within-Specification Holm Below 0.05"]
    )

    summary = {
        "design": (
            "Latest complete CSES crop-production reference year linked to same-year "
            "native 500 m MOD17 NPP over annual MCD12Q1 cropland pixels in "
            f"{buffer_radius_km:g} km buffers"
        ),
        "buffer_radius_km": buffer_radius_km,
        "npp_unit": "0.1 kg C per m2 per year; not standardized",
        "outcomes": list(OUTCOMES),
        "land_cover_definitions": list(DEFINITIONS),
        "minimum_valid_pixel_thresholds": list(SUPPORT_THRESHOLDS),
        "households_before_pixel_support_gate": int(len(frame)),
        "villages_before_pixel_support_gate": int(frame["Village Code"].nunique()),
        "any_stable_positive_signal": bool(stability["Stable Positive Signal"].any()),
        "stable_positive_signal_count": int(stability["Stable Positive Signal"].sum()),
        "multiple_testing": {
            "within_specification": "Holm across four agricultural outcomes",
            "global_screen": "Holm across 32 commune-FE mask-threshold-outcome tests",
        },
        "claim_limit": (
            "Weighted fixed-effect association, not causal mediation; MOD17 NPP measures "
            "vegetation carbon production rather than harvested crop yield"
        ),
    }
    support.to_csv(output_dir / "sample_support.csv", index=False)
    results.to_csv(output_dir / "same_year_npp_production_estimates.csv", index=False)
    stability.to_csv(output_dir / "production_signal_stability.csv", index=False)
    (output_dir / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Reference-year cropland NPP and CSES agricultural production\n\n"
        "This timing-repair analysis links each household's latest complete crop-production "
        "reference year to same-year 500 m cropland NPP within "
        f"{buffer_radius_km:g} km. It reports strict and inclusive "
        "agricultural land masks and all four declared minimum-pixel thresholds.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    display = [
        "NPP Definition",
        "Minimum Valid Cropland Pixels",
        "Outcome",
        "Coefficient",
        "Clustered Standard Error",
        "95 Percent CI Lower",
        "95 Percent CI Upper",
        "Probability Value",
        "Holm-Adjusted Probability Value Across Four Outcomes",
        "Global Holm-Adjusted Probability Value Across 32 Primary Tests",
        "Observations",
    ]
    print("\nCommune-FE results")
    print(results.loc[primary, display].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixel-npp", type=Path, default=DEFAULT_PIXEL_NPP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--buffer-radius-km", type=float, default=5.0)
    arguments = parser.parse_args()
    main(arguments.pixel_npp, arguments.output_dir, arguments.buffer_radius_km)
