#!/usr/bin/env python3
"""Fit same-year annual temperature and precipitation models for cropland NPP."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
NPP = ROOT / "data/processed/cses_public_village_pixel_cropland_npp_annual_preprocessed.parquet"
ANNUAL = ROOT / "data/processed/cses_village_buffer_annual_satellite_preprocessed.parquet"
CROSSWALK = ROOT / "data/processed/cses_to_cambodia_public_village_point_crosswalk_preprocessed.parquet"
OUTPUT = ROOT / "data/exp/analysis/climate-welfare/annual-climate-cropland-npp-model"

ID = "National Village Point ID"
YEAR = "Year"
BLOCK = "Spatial block"
RAIN_SOURCE = "Buffer Mean Annual Precipitation Total mm"
TMAX_SOURCE = "Buffer Mean Annual Mean Daily Maximum Temperature C"
TMIN_SOURCE = "Buffer Mean Annual Mean Daily Minimum Temperature C"
SUPPORT_THRESHOLDS = (1, 10, 20)
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
MODELS = {
    "Linear annual climate": ["Temperature deviation C", "Precipitation deviation per 100 mm"],
    "Nonlinear annual climate": [
        "Temperature deviation C",
        "Precipitation deviation per 100 mm",
        "Temperature deviation squared",
        "Precipitation deviation squared",
        "Temperature X precipitation deviation",
    ],
    "Maximum-minimum temperature decomposition": [
        "Maximum-temperature deviation C",
        "Minimum-temperature deviation C",
        "Precipitation deviation per 100 mm",
    ],
}


def load_panel() -> pd.DataFrame:
    crosswalk = pd.read_parquet(CROSSWALK, columns=["Village Code", ID]).dropna(subset=[ID])
    annual = pd.read_parquet(
        ANNUAL,
        columns=[
            "Village Code",
            "Buffer Radius km",
            YEAR,
            RAIN_SOURCE,
            TMAX_SOURCE,
            TMIN_SOURCE,
        ],
    )
    annual = annual.loc[
        annual["Buffer Radius km"].eq(5)
        & annual[YEAR].between(2001, 2020)
    ].drop(columns="Buffer Radius km")
    annual = annual.merge(crosswalk, on="Village Code", how="inner", validate="many_to_one")
    annual = annual.groupby([ID, YEAR], observed=True).agg(
        **{
            RAIN_SOURCE: (RAIN_SOURCE, "mean"),
            TMAX_SOURCE: (TMAX_SOURCE, "mean"),
            TMIN_SOURCE: (TMIN_SOURCE, "mean"),
        }
    ).reset_index()

    npp = pd.read_parquet(NPP)
    npp[ID] = npp[ID].astype(str)
    annual[ID] = annual[ID].astype(str)
    panel = npp.merge(annual, on=[ID, YEAR], how="inner", validate="one_to_one")
    panel = panel.loc[panel[YEAR].between(2001, 2020)].copy()
    panel[BLOCK] = (
        np.floor((panel["Point Longitude"] - 102.0) / 0.75).astype("Int64").astype(str)
        + "_"
        + np.floor((panel["Point Latitude"] - 10.0) / 0.75).astype("Int64").astype(str)
    )
    panel["Annual mean temperature C"] = (
        pd.to_numeric(panel[TMAX_SOURCE], errors="coerce")
        + pd.to_numeric(panel[TMIN_SOURCE], errors="coerce")
    ) / 2.0
    for source, target, scale in [
        ("Annual mean temperature C", "Temperature deviation C", 1.0),
        (TMAX_SOURCE, "Maximum-temperature deviation C", 1.0),
        (TMIN_SOURCE, "Minimum-temperature deviation C", 1.0),
        (RAIN_SOURCE, "Precipitation deviation per 100 mm", 100.0),
    ]:
        values = pd.to_numeric(panel[source], errors="coerce")
        local_mean = values.groupby(panel[ID], observed=True).transform("mean")
        panel[target] = (values - local_mean) / scale
    panel["Temperature deviation squared"] = panel["Temperature deviation C"] ** 2
    panel["Precipitation deviation squared"] = panel[
        "Precipitation deviation per 100 mm"
    ] ** 2
    panel["Temperature X precipitation deviation"] = (
        panel["Temperature deviation C"]
        * panel["Precipitation deviation per 100 mm"]
    )
    return panel


def fit(panel: pd.DataFrame, outcome: str, regressors: list[str]) -> tuple[object, pd.DataFrame]:
    required = [outcome, ID, YEAR, BLOCK, *regressors]
    sample = panel.dropna(subset=required).copy()
    absorb = pd.DataFrame(
        {
            "Village": sample[ID].astype("category"),
            "Year": sample[YEAR].astype("category"),
        },
        index=sample.index,
    )
    result = AbsorbingLS(
        sample[outcome].astype(float),
        sample[regressors].astype(float),
        absorb=absorb,
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample[BLOCK]).codes,
        debiased=True,
    )
    return result, sample


def main() -> None:
    for path in (NPP, ANNUAL, CROSSWALK):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    coefficient_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    for definition, columns in DEFINITIONS.items():
        for threshold in SUPPORT_THRESHOLDS:
            sample = panel.loc[
                pd.to_numeric(panel[columns["value"]], errors="coerce").notna()
                & pd.to_numeric(panel[columns["valid"]], errors="coerce").ge(threshold)
            ].copy()
            outcome = f"{definition} NPP kg C per m2"
            sample[outcome] = pd.to_numeric(sample[columns["value"]], errors="coerce")
            for model_name, regressors in MODELS.items():
                result, used = fit(sample, outcome, regressors)
                restriction = np.eye(len(regressors))
                joint = result.wald_test(restriction, np.zeros(len(regressors)))
                model_rows.append(
                    {
                        "NPP Definition": definition,
                        "Minimum Valid Cropland Pixels": threshold,
                        "Model": model_name,
                        "Observations": int(result.nobs),
                        "Villages": int(used[ID].nunique()),
                        "Years": int(used[YEAR].nunique()),
                        "Spatial Blocks": int(used[BLOCK].nunique()),
                        "R Squared": float(result.rsquared),
                        "Absorbed R Squared": float(result.absorbed_rsquared),
                        "Joint Climate Wald Statistic": float(joint.stat),
                        "Joint Climate Degrees of Freedom": int(joint.df),
                        "Joint Climate Probability Value": float(joint.pval),
                    }
                )
                intervals = result.conf_int(level=0.95)
                for term in regressors:
                    coefficient_rows.append(
                        {
                            "NPP Definition": definition,
                            "Minimum Valid Cropland Pixels": threshold,
                            "Model": model_name,
                            "Term": term,
                            "Coefficient": float(result.params[term]),
                            "Clustered Standard Error": float(result.std_errors[term]),
                            "95 Percent CI Lower": float(intervals.loc[term, "lower"]),
                            "95 Percent CI Upper": float(intervals.loc[term, "upper"]),
                            "Probability Value": float(result.pvalues[term]),
                            "Observations": int(result.nobs),
                        }
                    )
    coefficients = pd.DataFrame(coefficient_rows)
    models = pd.DataFrame(model_rows)
    summary = {
        "design": (
            "Same-year annual temperature and precipitation model for 5 km pixel-level "
            "cropland NPP, with linked-village and calendar-year fixed effects"
        ),
        "years": [2001, 2020],
        "temperature": (
            "Annual mean temperature proxy calculated as the mean of annual daily maximum "
            "and minimum temperature, expressed as within-village degrees C deviation"
        ),
        "precipitation": "Annual total expressed as within-village deviation per 100 mm",
        "joint_climate_significant_model_count": int(
            models["Joint Climate Probability Value"].lt(0.05).sum()
        ),
        "model_count": int(len(models)),
        "interpretation_limit": "Associational annual climate production function, not causal",
    }
    coefficients.to_csv(OUTPUT / "annual_climate_npp_coefficients.csv", index=False)
    models.to_csv(OUTPUT / "annual_climate_npp_model_fit.csv", index=False)
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Annual climate model for pixel-level cropland NPP\n\n"
        "Annual temperature and precipitation in year t are fitted to cropland NPP in the "
        "same year using linked-village and year fixed effects. Linear, nonlinear, and "
        "maximum/minimum-temperature decompositions are reported.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nModel fit")
    print(models.to_string(index=False))
    print("\nCoefficients: threshold 10")
    print(coefficients.loc[coefficients["Minimum Valid Cropland Pixels"].eq(10)].to_string(index=False))


if __name__ == "__main__":
    main()
