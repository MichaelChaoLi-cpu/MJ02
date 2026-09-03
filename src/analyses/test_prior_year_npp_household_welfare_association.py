#!/usr/bin/env python3
"""Test prior-calendar-year absolute NPP and household food consumption.

The exposure is the mean absolute annual MODIS NPP level within a 5 km public-
village buffer during the complete calendar year before interview.  Models use
survey weights, exact survey-time fixed effects, and 0.75-degree spatial-block
clustered uncertainty.  The estimates are associational, not causal mediation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
CSES = ROOT / "data/processed/cses_climate_ecology_welfare_frame_preprocessed.parquet"
NPP_PANEL = ROOT / "data/processed/cambodia_public_village_monsoon_npp_panel_preprocessed.parquet"
OUTPUT = ROOT / "data/exp/analysis/climate-welfare/prior-year-npp-household-welfare-association"

ID = "National Village Point ID"
FOOD = "Real 2021 Food Consumption Value per Household Member Riels"
WEIGHT = "Household Survey Weight"
NPP_SOURCE = "Village Buffer Mean Annual Land NPP Mean kg C per m2"
NPP = "Prior-year absolute NPP (0.1 kg C per m2)"
HEAT_SOURCE = "Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate B"
HEAT = "Prior-year post-onset heat days (10 days)"
RAIN_SOURCE = "Village Buffer Mean Annual Precipitation Total mm"
RAIN = "Prior-year annual precipitation (1000 mm)"
TIME = "Survey-time fixed-effect ID"
BLOCK = "Spatial block ID"
CLIMATE_CONTROLS = [HEAT, RAIN]
COMPOSITION = [
    "Household Size",
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
    "Agricultural Participation",
]


def load_frame() -> pd.DataFrame:
    household_columns = [
        "Survey Year", "Survey Wave", "Interview Calendar Year", "Interview Month",
        "Household ID", "Village Code", "Commune Code", "District Code",
        "Climate Ecology Link Available", "Point Longitude", "Point Latitude", ID,
        FOOD, WEIGHT, *COMPOSITION,
    ]
    households = pd.read_parquet(CSES, columns=household_columns)
    households = households.loc[
        households["Climate Ecology Link Available"].eq(1)
        & pd.to_numeric(households[FOOD], errors="coerce").gt(0)
        & pd.to_numeric(households[WEIGHT], errors="coerce").gt(0)
    ].copy()
    households["Prior Calendar Year"] = households["Interview Calendar Year"].astype(int) - 1
    households["Log real food consumption per member"] = np.log(
        pd.to_numeric(households[FOOD], errors="coerce")
    )
    households[TIME] = (
        households["Survey Wave"].astype(str)
        + "_"
        + households["Interview Calendar Year"].astype("Int64").astype(str)
        + "_"
        + households["Interview Month"].astype("Int64").astype(str)
    )
    households[BLOCK] = (
        np.floor((households["Point Longitude"] - 102.0) / 0.75).astype("Int64").astype(str)
        + "_"
        + np.floor((households["Point Latitude"] - 10.0) / 0.75).astype("Int64").astype(str)
    )
    households["Agricultural Participation"] = (
        households["Agricultural Participation"].astype("boolean").astype(float)
    )

    npp_columns = [
        ID, "Year", "Buffer Radius km", NPP_SOURCE, HEAT_SOURCE, RAIN_SOURCE,
    ]
    npp = pd.read_parquet(NPP_PANEL, columns=npp_columns)
    npp = npp.loc[npp["Buffer Radius km"].eq(5)].drop(columns="Buffer Radius km")
    npp = npp.rename(columns={"Year": "Prior Calendar Year"})
    npp[NPP] = pd.to_numeric(npp[NPP_SOURCE], errors="coerce") / 0.1
    npp[HEAT] = pd.to_numeric(npp[HEAT_SOURCE], errors="coerce") / 10.0
    npp[RAIN] = pd.to_numeric(npp[RAIN_SOURCE], errors="coerce") / 1000.0
    npp = npp[[ID, "Prior Calendar Year", NPP, HEAT, RAIN]]

    frame = households.merge(npp, on=[ID, "Prior Calendar Year"], how="inner")
    wave_count = frame.groupby("Village Code", observed=True)["Survey Year"].nunique()
    repeated = wave_count.loc[lambda value: value > 1].index
    frame["Repeated village"] = frame["Village Code"].isin(repeated)
    return frame


def fit_absorbing(
    frame: pd.DataFrame,
    area: str,
    controls: list[str],
    weight: str,
    specification: str,
    evidence_level: str,
) -> dict[str, object]:
    outcome = "Log real food consumption per member"
    required = [outcome, NPP, area, TIME, BLOCK, weight] + controls
    sample = frame.dropna(subset=required).copy()
    regressors = [NPP] + controls
    absorb = pd.DataFrame(
        {
            "Area": sample[area].astype("category"),
            "Survey time": sample[TIME].astype("category"),
        },
        index=sample.index,
    )
    result = AbsorbingLS(
        sample[outcome].astype(float),
        sample[regressors].astype(float),
        absorb=absorb,
        weights=sample[weight].astype(float),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample[BLOCK]).codes,
        debiased=True,
    )
    interval = result.conf_int(level=0.95).loc[NPP]
    coefficient = float(result.params[NPP])
    return {
        "Evidence Level": evidence_level,
        "Specification": specification,
        "Coefficient Log Points per 0.1 kg C per m2 NPP": coefficient,
        "Percent Difference per 0.1 kg C per m2 NPP": float(100.0 * np.expm1(coefficient)),
        "Clustered Standard Error": float(result.std_errors[NPP]),
        "95 Percent CI Lower": float(interval["lower"]),
        "95 Percent CI Upper": float(interval["upper"]),
        "Probability Value": float(result.pvalues[NPP]),
        "Observations": int(result.nobs),
        "Area Units": int(sample[area].nunique()),
        "Spatial Blocks": int(sample[BLOCK].nunique()),
        "Fixed Effects": f"{area} + survey-wave-by-interview-calendar-year-month",
        "Controls": ", ".join(controls),
        "Weighting": weight,
    }


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(group[column], errors="coerce")
    weights = pd.to_numeric(group[WEIGHT], errors="coerce")
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return np.nan
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def modal_weighted_block(group: pd.DataFrame) -> str:
    totals = group.groupby(BLOCK, observed=True)[WEIGHT].sum()
    return str(totals.idxmax())


def build_pseudopanel(frame: pd.DataFrame) -> pd.DataFrame:
    means = ["Log real food consumption per member", NPP, HEAT, RAIN, *COMPOSITION]
    rows: list[dict[str, object]] = []
    for (commune, time), group in frame.groupby(["Commune Code", TIME], observed=True):
        row: dict[str, object] = {
            "Commune Code": commune,
            TIME: time,
            "Survey Year": int(group["Survey Year"].iloc[0]),
            "Household Count": int(len(group)),
            "Survey Weight Sum": float(group[WEIGHT].sum()),
            BLOCK: modal_weighted_block(group),
        }
        for column in means:
            row[column] = weighted_mean(group, column)
        rows.append(row)
    panel = pd.DataFrame(rows)
    repeat_count = panel.groupby("Commune Code", observed=True)["Survey Year"].nunique()
    repeated = repeat_count.loc[lambda value: value > 1].index
    panel["Repeated commune"] = panel["Commune Code"].isin(repeated)
    return panel


def estimate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeated = frame.loc[frame["Repeated village"]].copy()
    agricultural = frame.loc[frame["Agricultural Participation"].eq(1)].copy()
    panel = build_pseudopanel(frame)
    pseudo = panel.loc[panel["Repeated commune"] & panel["Household Count"].ge(5)].copy()
    rows = [
        fit_absorbing(
            frame, "Commune Code", [], WEIGHT,
            "All linked households, commune FE, NPP only", "Household",
        ),
        fit_absorbing(
            frame, "District Code", CLIMATE_CONTROLS, WEIGHT,
            "All linked households, district FE", "Household",
        ),
        fit_absorbing(
            frame, "Commune Code", CLIMATE_CONTROLS, WEIGHT,
            "All linked households, commune FE", "Household",
        ),
        fit_absorbing(
            frame, "Commune Code", CLIMATE_CONTROLS + COMPOSITION, WEIGHT,
            "All linked households, commune FE, composition adjusted", "Household",
        ),
        fit_absorbing(
            repeated, "District Code", CLIMATE_CONTROLS, WEIGHT,
            "Repeated-village sample, district FE", "Household",
        ),
        fit_absorbing(
            repeated, "Village Code", CLIMATE_CONTROLS, WEIGHT,
            "Repeated-village sample, village FE", "Household",
        ),
        fit_absorbing(
            agricultural, "Commune Code", CLIMATE_CONTROLS, WEIGHT,
            "Agricultural households, commune FE", "Household",
        ),
        fit_absorbing(
            pseudo, "Commune Code", [], "Survey Weight Sum",
            "Repeated-commune pseudo-panel, NPP only", "Commune survey-time pseudo-panel",
        ),
        fit_absorbing(
            pseudo, "Commune Code", CLIMATE_CONTROLS, "Survey Weight Sum",
            "Repeated-commune pseudo-panel, minimum five households", "Commune survey-time pseudo-panel",
        ),
        fit_absorbing(
            pseudo, "Commune Code", CLIMATE_CONTROLS + COMPOSITION, "Survey Weight Sum",
            "Repeated-commune pseudo-panel, composition adjusted", "Commune survey-time pseudo-panel",
        ),
    ]
    return pd.DataFrame(rows), panel


def summarize(results: pd.DataFrame, frame: pd.DataFrame, panel: pd.DataFrame) -> dict[str, object]:
    selected = results.set_index("Specification")
    commune = selected.loc["All linked households, commune FE"]
    pseudo = selected.loc["Repeated-commune pseudo-panel, minimum five households"]
    repeated = selected.loc["Repeated-village sample, village FE"]
    agricultural = selected.loc["Agricultural households, commune FE"]
    coefficient = "Coefficient Log Points per 0.1 kg C per m2 NPP"
    gate = {
        "commune_fe_positive_interval": bool(
            commune[coefficient] > 0 and commune["95 Percent CI Lower"] > 0
        ),
        "pseudo_panel_positive_interval": bool(
            pseudo[coefficient] > 0 and pseudo["95 Percent CI Lower"] > 0
        ),
        "repeated_village_positive_direction": bool(repeated[coefficient] > 0),
        "agricultural_household_positive_direction": bool(agricultural[coefficient] > 0),
    }
    gate["promotion_gate_passed"] = bool(all(gate.values()))
    return {
        "design": "Absolute village-buffer NPP in the complete calendar year before interview",
        "npp_reporting_unit": "0.1 kg C per m2",
        "households": int(len(frame)),
        "villages": int(frame["Village Code"].nunique()),
        "communes": int(frame["Commune Code"].nunique()),
        "prior_year_range": [int(frame["Prior Calendar Year"].min()), int(frame["Prior Calendar Year"].max())],
        "npp_distribution_kg_c_per_m2": {
            key: float(value)
            for key, value in frame[NPP].mul(0.1).describe()[["mean", "std", "min", "50%", "max"]].items()
        },
        "pseudo_panel_cells_all": int(len(panel)),
        "primary_pseudo_panel_cells": int(
            (panel["Repeated commune"] & panel["Household Count"].ge(5)).sum()
        ),
        "gate": gate,
        "interpretation": (
            "Prior-year NPP-household welfare association passed the frozen exploratory gate."
            if gate["promotion_gate_passed"]
            else "Prior-year NPP-household welfare association did not pass the frozen exploratory gate."
        ),
        "claim_limit": "Associational repeated-cross-section and pseudo-panel evidence; not a causal productivity effect.",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    results, panel = estimate(frame)
    summary = summarize(results, frame, panel)
    results.to_csv(OUTPUT / "coefficients.csv", index=False)
    panel.to_parquet(OUTPUT / "commune_survey_time_pseudopanel.parquet", index=False)
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Prior-year NPP and household welfare association diagnostic\n\n"
        "The exposure is the mean absolute annual MODIS NPP level in the 5 km village buffer "
        "during the complete calendar year before interview. The outcome is log real food "
        "consumption per household member. Results compare household and commune pseudo-panel "
        "fixed-effect specifications and are interpreted as associations only.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
