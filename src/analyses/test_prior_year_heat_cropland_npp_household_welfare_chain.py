#!/usr/bin/env python3
"""Test a prior-year heat -> cropland NPP -> household-welfare evidence chain.

This is a pair of temporally aligned fixed-effect associations, not a 2SLS or
causal mediation estimator. Stage 1 uses one row per linked village-prior-year.
Stage 2 uses household outcomes and assigns the common village-prior-year NPP
exposure to every linked household in that village and survey timing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
HOUSEHOLDS = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
NPP = ROOT / "data/processed/cses_public_village_pixel_cropland_npp_annual_preprocessed.parquet"
CLIMATE = ROOT / "data/processed/cambodia_public_village_monsoon_ecology_panel_preprocessed.parquet"
OUTPUT = ROOT / "data/exp/analysis/climate-welfare/prior-year-heat-cropland-npp-household-welfare-chain"

ID = "National Village Point ID"
YEAR = "Prior Calendar Year"
WEIGHT = "Household Survey Weight"
TIME = "Survey-wave interview time"
BLOCK = "Spatial block"

HEAT_SOURCE = "Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate B"
RAIN_SOURCE = "Village Buffer Mean May October Precipitation Total mm Anomaly Z"
ONSET_SOURCE = "Village Buffer Mean Wet-Season Onset DOY Candidate B Anomaly Z"
DRY_SOURCE = "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z"
HEAT = "Prior-year post-onset heat days per 10"
RAIN = "Prior-year May-October precipitation anomaly Z"
ONSET = "Prior-year wet-season onset anomaly Z"
DRY = "Prior-year longest intraseasonal dry-spell anomaly Z"
CLIMATE_CONTROLS = [HEAT, RAIN, ONSET, DRY]

FOOD_SOURCE = "Real 2021 Food Consumption Value per Household Member Riels"
SEVERE_SOURCE = "Any Severe Food Insecurity Experience"
SEVERITY_SOURCE = "Food Insecurity Severity Sum"
OUTCOMES = {
    "Log real food consumption per household member": {
        "source": FOOD_SOURCE,
        "expected_direction": "positive",
    },
    "Any severe food insecurity experience": {
        "source": SEVERE_SOURCE,
        "expected_direction": "negative",
    },
    "Food insecurity severity sum": {
        "source": SEVERITY_SOURCE,
        "expected_direction": "negative",
    },
}

COMPOSITION = [
    "Household Size",
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
    "Rural household",
]
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


def holm_adjust(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=numeric.index, dtype=float)
    running = 0.0
    total = len(valid)
    for rank, (index, value) in enumerate(valid.items(), start=1):
        running = max(running, min(1.0, float(value) * (total - rank + 1)))
        adjusted.loc[index] = running
    return adjusted


def load_frame() -> pd.DataFrame:
    households = pd.read_parquet(HOUSEHOLDS)
    households = households.loc[
        households["Climate Ecology Link Available"].eq(1)
        & households[ID].notna()
        & pd.to_numeric(households[WEIGHT], errors="coerce").gt(0)
    ].copy()
    households[ID] = households[ID].astype(str)
    households[YEAR] = households["Interview Calendar Year"].astype(int) - 1
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
    households["Rural household"] = (
        pd.to_numeric(households["Urban Rural"], errors="coerce").eq(2).astype(float)
    )
    food = pd.to_numeric(households[FOOD_SOURCE], errors="coerce")
    households["Log real food consumption per household member"] = np.where(
        food.gt(0), np.log(food), np.nan
    )
    households["Any severe food insecurity experience"] = pd.to_numeric(
        households[SEVERE_SOURCE], errors="coerce"
    )
    households["Food insecurity severity sum"] = pd.to_numeric(
        households[SEVERITY_SOURCE], errors="coerce"
    )

    npp = pd.read_parquet(NPP).rename(columns={"Year": YEAR})
    npp[ID] = npp[ID].astype(str)
    npp_columns = [ID, YEAR] + [
        column
        for definition in DEFINITIONS.values()
        for column in (definition["value"], definition["valid"])
    ]
    npp = npp[npp_columns]

    climate = pd.read_parquet(
        CLIMATE,
        columns=[
            ID,
            "Year",
            "Buffer Radius km",
            HEAT_SOURCE,
            RAIN_SOURCE,
            ONSET_SOURCE,
            DRY_SOURCE,
        ],
    )
    climate = climate.loc[climate["Buffer Radius km"].eq(5)].drop(
        columns="Buffer Radius km"
    )
    climate = climate.rename(columns={"Year": YEAR})
    climate[ID] = climate[ID].astype(str)
    climate[HEAT] = pd.to_numeric(climate[HEAT_SOURCE], errors="coerce") / 10.0
    climate[RAIN] = pd.to_numeric(climate[RAIN_SOURCE], errors="coerce")
    climate[ONSET] = pd.to_numeric(climate[ONSET_SOURCE], errors="coerce")
    climate[DRY] = pd.to_numeric(climate[DRY_SOURCE], errors="coerce")
    climate = climate[[ID, YEAR, *CLIMATE_CONTROLS]]

    frame = households.merge(
        npp, on=[ID, YEAR], how="inner", validate="many_to_one"
    ).merge(climate, on=[ID, YEAR], how="inner", validate="many_to_one")
    repeat_count = frame.groupby("Village Code", observed=True)["Survey Wave"].nunique()
    repeated = repeat_count.loc[lambda values: values.gt(1)].index
    frame["Repeated village"] = frame["Village Code"].isin(repeated)
    return frame


def load_full_ecological_panel() -> pd.DataFrame:
    """Load all 2001-2020 years for the same CSES-linked public village points."""
    npp = pd.read_parquet(NPP).rename(columns={"Year": YEAR})
    npp[ID] = npp[ID].astype(str)
    npp = npp.loc[pd.to_numeric(npp[YEAR], errors="coerce").between(2001, 2020)].copy()
    npp[BLOCK] = (
        np.floor((npp["Point Longitude"] - 102.0) / 0.75)
        .astype("Int64")
        .astype(str)
        + "_"
        + np.floor((npp["Point Latitude"] - 10.0) / 0.75)
        .astype("Int64")
        .astype(str)
    )
    climate = pd.read_parquet(
        CLIMATE,
        columns=[
            ID,
            "Year",
            "Buffer Radius km",
            HEAT_SOURCE,
            RAIN_SOURCE,
            ONSET_SOURCE,
            DRY_SOURCE,
        ],
    )
    climate = climate.loc[climate["Buffer Radius km"].eq(5)].drop(
        columns="Buffer Radius km"
    )
    climate = climate.rename(columns={"Year": YEAR})
    climate[ID] = climate[ID].astype(str)
    climate[HEAT] = pd.to_numeric(climate[HEAT_SOURCE], errors="coerce") / 10.0
    climate[RAIN] = pd.to_numeric(climate[RAIN_SOURCE], errors="coerce")
    climate[ONSET] = pd.to_numeric(climate[ONSET_SOURCE], errors="coerce")
    climate[DRY] = pd.to_numeric(climate[DRY_SOURCE], errors="coerce")
    return npp.merge(
        climate[[ID, YEAR, *CLIMATE_CONTROLS]],
        on=[ID, YEAR],
        how="inner",
        validate="one_to_one",
    )


def fit_absorbed(
    frame: pd.DataFrame,
    outcome: str,
    regressors: list[str],
    absorb_columns: list[str],
    weighted: bool,
) -> tuple[object, pd.DataFrame]:
    required = [outcome, BLOCK, *regressors, *absorb_columns]
    if weighted:
        required.append(WEIGHT)
    sample = frame.dropna(subset=required).copy()
    if weighted:
        sample = sample.loc[pd.to_numeric(sample[WEIGHT], errors="coerce").gt(0)]
    absorb = pd.DataFrame(
        {
            f"FE {index}": sample[column].astype("category")
            for index, column in enumerate(absorb_columns, start=1)
        },
        index=sample.index,
    )
    model = AbsorbingLS(
        sample[outcome].astype(float),
        sample[regressors].astype(float),
        absorb=absorb,
        weights=sample[WEIGHT].astype(float) if weighted else None,
        drop_absorbed=True,
    )
    result = model.fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample[BLOCK]).codes,
        debiased=True,
    )
    return result, sample


def coefficient_row(
    result: object,
    sample: pd.DataFrame,
    term: str,
    stage: str,
    outcome: str,
    specification: str,
    fixed_effects: str,
    weighted: bool,
) -> dict[str, object]:
    interval = result.conf_int(level=0.95).loc[term]
    return {
        "Stage": stage,
        "Outcome": outcome,
        "Specification": specification,
        "Term": term,
        "Coefficient": float(result.params[term]),
        "Clustered Standard Error": float(result.std_errors[term]),
        "95 Percent CI Lower": float(interval["lower"]),
        "95 Percent CI Upper": float(interval["upper"]),
        "Probability Value": float(result.pvalues[term]),
        "Observations": int(result.nobs),
        "Villages": int(sample[ID].nunique()),
        "Years or Survey Times": int(
            sample[YEAR].nunique() if stage.startswith("1") else sample[TIME].nunique()
        ),
        "Spatial Blocks": int(sample[BLOCK].nunique()),
        "Fixed Effects": fixed_effects,
        "Survey Weighted": weighted,
    }


def stage_one(frame: pd.DataFrame, full_panel: pd.DataFrame) -> pd.DataFrame:
    survey_preceding = (
        frame.sort_values([ID, YEAR])
        .drop_duplicates([ID, YEAR])
        .copy()
    )
    rows: list[dict[str, object]] = []
    panels = [
        ("Exact survey-preceding village-years", survey_preceding),
        ("Full 2001-2020 linked-village panel", full_panel),
    ]
    for panel_scope, village_year in panels:
        for definition, columns in DEFINITIONS.items():
            for threshold in SUPPORT_THRESHOLDS:
                sample = village_year.loc[
                    pd.to_numeric(village_year[columns["value"]], errors="coerce").notna()
                    & pd.to_numeric(village_year[columns["valid"]], errors="coerce").ge(threshold)
                ].copy()
                outcome = f"Prior-year {definition} NPP kg C per m2"
                sample[outcome] = pd.to_numeric(sample[columns["value"]], errors="coerce")
                result, used = fit_absorbed(
                    sample,
                    outcome,
                    CLIMATE_CONTROLS,
                    [ID, YEAR],
                    weighted=False,
                )
                row = coefficient_row(
                    result,
                    used,
                    HEAT,
                    "1: prior-year heat to same-year cropland NPP",
                    outcome,
                    "Village and year FE with monsoon controls",
                    "national public village point + prior calendar year",
                    False,
                )
                row.update(
                    {
                        "Panel Scope": panel_scope,
                        "NPP Definition": definition,
                        "Minimum Valid Cropland Pixels": threshold,
                        "Expected Direction": "negative",
                        "Direction Supported": bool(result.params[HEAT] < 0),
                        "Nominal Probability Below 0.05": bool(result.pvalues[HEAT] < 0.05),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def stage_two(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for definition, columns in DEFINITIONS.items():
        for threshold in SUPPORT_THRESHOLDS:
            eligible = (
                pd.to_numeric(frame[columns["value"]], errors="coerce").notna()
                & pd.to_numeric(frame[columns["valid"]], errors="coerce").ge(threshold)
            )
            available = frame.loc[eligible].copy()
            term = f"Prior-year {definition} NPP per 0.1 kg C per m2"
            available[term] = pd.to_numeric(
                available[columns["value"]], errors="coerce"
            ) / 0.1
            specifications = [
                (
                    "Commune FE, monsoon adjusted without heat",
                    available,
                    "Commune Code",
                    [RAIN, ONSET, DRY],
                ),
                (
                    "Commune FE, heat and monsoon adjusted",
                    available,
                    "Commune Code",
                    CLIMATE_CONTROLS,
                ),
                (
                    "Commune FE, climate and composition adjusted",
                    available,
                    "Commune Code",
                    [*CLIMATE_CONTROLS, *COMPOSITION],
                ),
                (
                    "Repeated-village sample, village FE",
                    available.loc[available["Repeated village"]].copy(),
                    "Village Code",
                    CLIMATE_CONTROLS,
                ),
            ]
            for specification, sample_frame, area, controls in specifications:
                for outcome, metadata in OUTCOMES.items():
                    result, used = fit_absorbed(
                        sample_frame,
                        outcome,
                        [term, *controls],
                        [area, TIME],
                        weighted=True,
                    )
                    row = coefficient_row(
                        result,
                        used,
                        term,
                        "2: prior-year cropland NPP to household welfare",
                        outcome,
                        specification,
                        f"{area} + survey-wave-by-interview-year-month",
                        True,
                    )
                    coefficient = float(result.params[term])
                    expected = metadata["expected_direction"]
                    row.update(
                        {
                            "NPP Definition": definition,
                            "Minimum Valid Cropland Pixels": threshold,
                            "Expected Direction": expected,
                            "Direction Supported": bool(
                                coefficient > 0 if expected == "positive" else coefficient < 0
                            ),
                        }
                    )
                    rows.append(row)
    results = pd.DataFrame(rows)
    results["Holm-Adjusted Probability Value Across Three Welfare Outcomes"] = np.nan
    group_keys = [
        "NPP Definition",
        "Minimum Valid Cropland Pixels",
        "Specification",
    ]
    for _, indexes in results.groupby(group_keys, observed=True).groups.items():
        results.loc[indexes, "Holm-Adjusted Probability Value Across Three Welfare Outcomes"] = (
            holm_adjust(results.loc[indexes, "Probability Value"])
        )
    results["Direction and Holm Supported"] = (
        results["Direction Supported"]
        & results["Holm-Adjusted Probability Value Across Three Welfare Outcomes"].lt(0.05)
    )
    return results


def main() -> None:
    for path in (HOUSEHOLDS, NPP, CLIMATE):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    full_panel = load_full_ecological_panel()
    first = stage_one(frame, full_panel)
    second = stage_two(frame)

    primary_second = second.loc[
        second["Specification"].eq(
            "Commune FE, climate and composition adjusted"
        )
    ].copy()
    chain_rows = first.loc[
        first["Panel Scope"].eq("Exact survey-preceding village-years"),
        [
            "NPP Definition",
            "Minimum Valid Cropland Pixels",
            "Coefficient",
            "Probability Value",
            "Direction Supported",
            "Nominal Probability Below 0.05",
        ]
    ].rename(
        columns={
            "Coefficient": "Heat-to-NPP Coefficient",
            "Probability Value": "Heat-to-NPP Probability Value",
            "Direction Supported": "Heat-to-NPP Negative Direction",
            "Nominal Probability Below 0.05": "Heat-to-NPP Probability Below 0.05",
        }
    )
    chain = primary_second.merge(
        chain_rows,
        on=["NPP Definition", "Minimum Valid Cropland Pixels"],
        how="left",
        validate="many_to_one",
    )
    chain["Two-Link Directional Chain Supported"] = (
        chain["Heat-to-NPP Negative Direction"]
        & chain["Heat-to-NPP Probability Below 0.05"]
        & chain["Direction and Holm Supported"]
    )

    support_rows: list[dict[str, object]] = []
    for definition, columns in DEFINITIONS.items():
        for threshold in SUPPORT_THRESHOLDS:
            eligible = (
                pd.to_numeric(frame[columns["value"]], errors="coerce").notna()
                & pd.to_numeric(frame[columns["valid"]], errors="coerce").ge(threshold)
            )
            sample = frame.loc[eligible]
            support_rows.append(
                {
                    "NPP Definition": definition,
                    "Minimum Valid Cropland Pixels": threshold,
                    "Households": int(len(sample)),
                    "Villages": int(sample[ID].nunique()),
                    "Village Prior-Years": int(
                        sample[[ID, YEAR]].drop_duplicates().shape[0]
                    ),
                    "Food Consumption Observed": int(
                        sample["Log real food consumption per household member"].notna().sum()
                    ),
                    "Severe Food Insecurity Observed": int(
                        sample["Any severe food insecurity experience"].notna().sum()
                    ),
                    "Food Insecurity Severity Observed": int(
                        sample["Food insecurity severity sum"].notna().sum()
                    ),
                }
            )
    support = pd.DataFrame(support_rows)
    summary = {
        "design": (
            "Stage 1: prior-calendar-year heat to same-year 5 km pixel-level cropland NPP "
            "at linked village-year scale. Stage 2: the same prior-year NPP assigned to "
            "each linked household and related to living-quality outcomes."
        ),
        "not_2sls_or_causal_mediation": True,
        "ndvi_included": False,
        "households_before_pixel_support_gate": int(len(frame)),
        "villages_before_pixel_support_gate": int(frame[ID].nunique()),
        "stage_one_exact_negative_and_nominally_significant_specs": int(
            (
                first["Panel Scope"].eq("Exact survey-preceding village-years")
                & first["Direction Supported"]
                & first["Nominal Probability Below 0.05"]
            ).sum()
        ),
        "stage_one_full_panel_negative_and_nominally_significant_specs": int(
            (
                first["Panel Scope"].eq("Full 2001-2020 linked-village panel")
                & first["Direction Supported"]
                & first["Nominal Probability Below 0.05"]
            ).sum()
        ),
        "stage_one_specs_per_panel": int(len(first) / 2),
        "stage_two_direction_and_holm_supported_primary_rows": int(
            primary_second["Direction and Holm Supported"].sum()
        ),
        "stage_two_primary_rows": int(len(primary_second)),
        "two_link_directional_chain_supported_rows": int(
            chain["Two-Link Directional Chain Supported"].sum()
        ),
        "two_link_rows": int(len(chain)),
        "promotion_rule": (
            "A coherent row requires negative heat-to-NPP at p<0.05 and an expected-direction "
            "NPP-to-welfare coefficient passing Holm adjustment across three welfare outcomes."
        ),
        "interpretation_limit": (
            "Sequential fixed-effect associations only; NPP is not instrumented and the "
            "analysis does not identify causal mediation."
        ),
    }
    first.to_csv(OUTPUT / "stage_1_prior_year_heat_to_cropland_npp.csv", index=False)
    second.to_csv(OUTPUT / "stage_2_prior_year_cropland_npp_to_household_welfare.csv", index=False)
    chain.to_csv(OUTPUT / "two_link_chain_assessment.csv", index=False)
    support.to_csv(OUTPUT / "sample_support.csv", index=False)
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Prior-year heat, cropland NPP, and household welfare chain\n\n"
        "The two regressions share the calendar year immediately before household interview. "
        "Stage 1 uses unique linked village-years; Stage 2 uses survey-weighted household "
        "observations with common village-year NPP exposure. NDVI is excluded.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nStage 1")
    print(
        first[
            [
                "NPP Definition",
                "Panel Scope",
                "Minimum Valid Cropland Pixels",
                "Coefficient",
                "Clustered Standard Error",
                "95 Percent CI Lower",
                "95 Percent CI Upper",
                "Probability Value",
                "Observations",
                "Villages",
            ]
        ].to_string(index=False)
    )
    print("\nStage 2: climate and composition adjusted commune FE")
    print(
        primary_second[
            [
                "NPP Definition",
                "Minimum Valid Cropland Pixels",
                "Outcome",
                "Coefficient",
                "Clustered Standard Error",
                "95 Percent CI Lower",
                "95 Percent CI Upper",
                "Probability Value",
                "Holm-Adjusted Probability Value Across Three Welfare Outcomes",
                "Observations",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
