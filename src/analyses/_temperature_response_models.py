"""Shared estimators for the activated historical-boundary temperature extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
ANNUAL_PANEL = ROOT / "data/processed/historical_boundary_annual_local_temperature_analysis_preprocessed.parquet"
HF_PANEL = ROOT / "data/processed/historical_boundary_16day_climate_vegetation_temperature_preprocessed.parquet"

TREATMENT = "Higher-Repression Southwest Zone"
DISTANCE = "Signed Distance to Historical Repression Boundary km"
ABS_DISTANCE = "Absolute Distance to Historical Repression Boundary km"
VILLAGE = "Village Code"
YEAR = "Year"
DATE = "Composite Date"
SEGMENT = "Historical Boundary Segment"
COMMUNE = "Linked Climate Commune Code"
ADMIN_COMMUNE = "Commune Code"
CELL = "CHIRPS Cell ID"
NPP = "Annual Land NPP Anomaly Z 2001-2020"
EVI = "High-Frequency EVI Anomaly Z"
NDVI = "High-Frequency NDVI Anomaly Z"
ANNUAL_HOT = "May October Hot Day Count Anomaly Z"
ANNUAL_HEATWAVE = "May October Heatwave Day Count Anomaly Z"
ANNUAL_HOT_NIGHT = "May October Hot Night Count Anomaly Z"
HF_HOT = "Interval Hot Day Count Anomaly Z"
HF_HEATWAVE = "Interval Heatwave Day Count Anomaly Z"
HF_HOT_NIGHT = "Interval Hot Night Count Anomaly Z"
HF_COMPOUND = "Compound Hot-Dry Intensity"
HF_HOT_INTENSITY = "Hot Day Intensity"
HF_DRY_INTENSITY = "Dry Rainfall Intensity"
SESOI = 0.20


@dataclass(frozen=True)
class Estimate:
    design: str
    outcome: str
    shock: str
    specification: str
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    p_value: float
    observations: int
    villages: int
    periods: int
    cross_side_groups: int
    shock_sd: float
    scale: str
    fixed_effects: str
    inference: str
    equivalence: str

    def as_record(self) -> dict[str, object]:
        return asdict(self)


def _cross_side_values(data: pd.DataFrame, group_columns: list[str]) -> set[object]:
    village = data.drop_duplicates(VILLAGE)
    counts = village.groupby(group_columns, observed=True)[TREATMENT].nunique()
    selected = counts[counts.eq(2)].index
    if len(group_columns) == 1:
        return set(selected.astype(str))
    return set(selected)


def prepare_annual() -> pd.DataFrame:
    columns = [
        VILLAGE,
        YEAR,
        ADMIN_COMMUNE,
        COMMUNE,
        TREATMENT,
        DISTANCE,
        ABS_DISTANCE,
        SEGMENT,
        "NPP Complete 2001-2020 Baseline",
        NPP,
        ANNUAL_HOT,
        ANNUAL_HEATWAVE,
        ANNUAL_HOT_NIGHT,
    ]
    data = pd.read_parquet(ANNUAL_PANEL, columns=columns)
    data = data.loc[
        data[YEAR].between(2001, 2021)
        & data[ABS_DISTANCE].le(5)
        & data["NPP Complete 2001-2020 Baseline"].eq(1)
    ].copy()
    data[VILLAGE] = data[VILLAGE].astype("string").str.zfill(8)
    data[ADMIN_COMMUNE] = data[ADMIN_COMMUNE].astype("string").str.zfill(6)
    data[COMMUNE] = data[COMMUNE].astype("string")
    data["_district_year"] = data[ADMIN_COMMUNE].str[:4] + "|" + data[YEAR].astype(str)
    data["_segment_year"] = data[SEGMENT].astype(str) + "|" + data[YEAR].astype(str)
    data["_commune_year"] = data[COMMUNE] + "|" + data[YEAR].astype(str)
    cross = _cross_side_values(data, [COMMUNE])
    data["_cross_side_commune"] = data[COMMUNE].isin(cross)
    return data


def prepare_high_frequency() -> pd.DataFrame:
    columns = [
        VILLAGE,
        DATE,
        YEAR,
        COMMUNE,
        CELL,
        TREATMENT,
        DISTANCE,
        ABS_DISTANCE,
        SEGMENT,
        "Cross-Side CHIRPS Cell",
        EVI,
        NDVI,
        HF_HOT,
        HF_HEATWAVE,
        HF_HOT_NIGHT,
        HF_HOT_INTENSITY,
        HF_DRY_INTENSITY,
        HF_COMPOUND,
    ]
    data = pd.read_parquet(HF_PANEL, columns=columns)
    data[DATE] = pd.to_datetime(data[DATE])
    data = data.loc[
        data[YEAR].between(2001, 2021)
        & data[ABS_DISTANCE].le(5)
        & data["Cross-Side CHIRPS Cell"].eq(1)
    ].copy()
    data[VILLAGE] = data[VILLAGE].astype("string").str.zfill(8)
    data[COMMUNE] = data[COMMUNE].astype("string")
    data[CELL] = data[CELL].astype("string")
    data["_cell_commune"] = data[CELL] + "|" + data[COMMUNE]
    cross_pairs = _cross_side_values(data, [CELL, COMMUNE])
    pairs = pd.MultiIndex.from_frame(data[[CELL, COMMUNE]])
    data["_cross_side_cell_commune"] = pairs.isin(cross_pairs)
    data["_cell_date"] = data[CELL] + "|" + data[DATE].astype(str)
    data["_cell_commune_date"] = data["_cell_commune"] + "|" + data[DATE].astype(str)
    return data


def _fit(
    data: pd.DataFrame,
    *,
    design: str,
    outcome: str,
    shock: str,
    specification: str,
    compound: bool = False,
    bandwidth_km: float = 5,
) -> Estimate:
    if specification not in {"Primary", "Within climate commune"}:
        raise ValueError(f"Unknown specification: {specification}")
    bandwidth_sample = data.loc[data[ABS_DISTANCE].le(bandwidth_km)].copy()
    sample = bandwidth_sample.copy()
    if specification == "Within climate commune":
        if design == "Annual NPP":
            cross = _cross_side_values(sample, [COMMUNE])
            sample = sample.loc[sample[COMMUNE].isin(cross)].copy()
        else:
            cross = _cross_side_values(sample, [CELL, COMMUNE])
            pairs = pd.MultiIndex.from_frame(sample[[CELL, COMMUNE]])
            sample = sample.loc[pairs.isin(cross)].copy()

    required = [outcome, shock, TREATMENT, DISTANCE, VILLAGE]
    if compound:
        required += [HF_HOT_INTENSITY, HF_DRY_INTENSITY]
    sample = sample.dropna(subset=required).copy()
    if sample.empty:
        raise ValueError(f"No observations for {design}, {shock}, {specification}")

    treatment = sample[TREATMENT].astype(float)
    distance = sample[DISTANCE].astype(float)
    shock_sd = float(bandwidth_sample.dropna(subset=[outcome, shock])[shock].std(ddof=1))
    if shock_sd <= 0:
        raise ValueError(f"No shock variation for {shock}")

    if compound:
        components = {
            "hot": sample[HF_HOT_INTENSITY].astype(float),
            "dry": sample[HF_DRY_INTENSITY].astype(float),
            "compound": sample[shock].astype(float),
        }
        exog: dict[str, pd.Series] = {}
        for name, values in components.items():
            exog[f"_treat_{name}"] = treatment * values
            exog[f"_distance_{name}"] = distance * values
            exog[f"_treat_distance_{name}"] = treatment * distance * values
        target = "_treat_compound"
    else:
        values = sample[shock].astype(float)
        exog = {
            "_treat_shock": treatment * values,
            "_distance_shock": distance * values,
            "_treat_distance_shock": treatment * distance * values,
        }
        target = "_treat_shock"
        if design == "Annual NPP" and specification == "Primary":
            exog = {"_shock": values, **exog}

    exog_frame = pd.DataFrame(exog, index=sample.index)
    if design == "Annual NPP":
        effects = [VILLAGE, "_segment_year"]
        if specification == "Within climate commune":
            effects.append("_commune_year")
        clusters = pd.DataFrame(
            {
                "village": pd.factorize(sample[VILLAGE])[0],
                "district_year": pd.factorize(sample["_district_year"])[0],
            },
            index=sample.index,
        )
        period_count = int(sample[YEAR].nunique())
        group_count = int(sample.loc[sample["_cross_side_commune"], COMMUNE].nunique())
        fixed_effects = "Village; boundary-segment-by-year"
        if specification == "Within climate commune":
            fixed_effects += "; climate-commune-by-year"
        inference = "Village + district-by-year two-way clustered"
    else:
        group_date = "_cell_date" if specification == "Primary" else "_cell_commune_date"
        effects = [VILLAGE, group_date]
        clusters = pd.DataFrame(
            {
                "village": pd.factorize(sample[VILLAGE])[0],
                "date": pd.factorize(sample[DATE])[0],
            },
            index=sample.index,
        )
        period_count = int(sample[DATE].nunique())
        group_count = int(sample[CELL].nunique())
        fixed_effects = "Village; climate-cell-by-date"
        if specification == "Within climate commune":
            fixed_effects = "Village; climate-cell-by-climate-commune-by-date"
        inference = "Village + composite-date two-way clustered"

    fitted = AbsorbingLS(
        dependent=sample[outcome].astype(float),
        exog=exog_frame,
        absorb=sample[effects].astype("category"),
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)
    interval = fitted.conf_int().loc[target]
    estimate = float(fitted.params[target] * shock_sd)
    standard_error = float(fitted.std_errors[target] * shock_sd)
    ci_low = float(interval["lower"] * shock_sd)
    ci_high = float(interval["upper"] * shock_sd)
    equivalence = (
        "95% CI inside +/-0.20 SD" if ci_low > -SESOI and ci_high < SESOI
        else "95% CI not inside +/-0.20 SD"
    )
    return Estimate(
        design=design,
        outcome=outcome,
        shock=shock,
        specification=specification,
        estimate=estimate,
        standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=float(fitted.pvalues[target]),
        observations=int(fitted.nobs),
        villages=int(sample[VILLAGE].nunique()),
        periods=period_count,
        cross_side_groups=group_count,
        shock_sd=shock_sd,
        scale="Outcome SD per one sample-SD shock",
        fixed_effects=fixed_effects,
        inference=inference,
        equivalence=equivalence,
    )


def fit_annual(shock: str, specification: str, bandwidth_km: float = 5) -> Estimate:
    return _fit(
        prepare_annual(),
        design="Annual NPP",
        outcome=NPP,
        shock=shock,
        specification=specification,
        bandwidth_km=bandwidth_km,
    )


def fit_high_frequency(
    shock: str,
    specification: str,
    outcome: str = EVI,
    compound: bool = False,
    bandwidth_km: float = 5,
) -> Estimate:
    return _fit(
        prepare_high_frequency(),
        design="16-day vegetation",
        outcome=outcome,
        shock=shock,
        specification=specification,
        compound=compound,
        bandwidth_km=bandwidth_km,
    )


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def fit_confirmatory_family() -> pd.DataFrame:
    records = []
    models = [
        ("Annual hot days", lambda specification: fit_annual(ANNUAL_HOT, specification)),
        ("16-day hot days", lambda specification: fit_high_frequency(HF_HOT, specification)),
        (
            "16-day compound hot-dry",
            lambda specification: fit_high_frequency(
                HF_COMPOUND, specification, compound=True
            ),
        ),
    ]
    for estimand, estimator in models:
        for specification in ("Primary", "Within climate commune"):
            record = estimator(specification).as_record()
            record["estimand"] = estimand
            records.append(record)
    results = pd.DataFrame(records)
    primary = results["specification"].eq("Primary")
    results["holm_primary_p_value"] = np.nan
    results.loc[primary, "holm_primary_p_value"] = holm_adjust(
        results.loc[primary, "p_value"]
    )
    mapping = results.loc[primary].set_index("estimand")["holm_primary_p_value"]
    results["holm_primary_p_value"] = results["estimand"].map(mapping)
    return results
