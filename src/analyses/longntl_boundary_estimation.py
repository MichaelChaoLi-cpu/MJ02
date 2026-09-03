#!/usr/bin/env python3
"""Shared LongNTL historical-boundary estimation for planned outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/longntl_v2_historical_boundary_climate_preprocessed.parquet"
VIIRS_RESULTS = ROOT / "data/exp/viirs-boundary-experiment/viirs_boundary_shock_response_estimates.csv"
OUTPUT = ROOT / "data/exp/longntl-v2-boundary-experiment"

CELL = "Grid Cell ID"
YEAR = "Year"
COMMUNE = "Linked Climate Commune Code"
SEGMENT = "Historical Boundary Segment"
TREATMENT = "Higher-Repression Southwest Zone"
DISTANCE = "Signed Distance to Historical Repression Boundary km"
ABS_DISTANCE = "Absolute Distance to Historical Repression Boundary km"
OUTCOME = "Asinh Annual NPP-VIIRS-like Radiance"
EXTENSIVE = "Any Nonzero Annual NPP-VIIRS-like Radiance"
PRIMARY_SHOCK = "May October Rainfall Anomaly Z (1991-2020)"
ANNUAL_SHOCK = "Annual Rainfall Anomaly Z (1991-2020)"
SESOI = 0.20
STRIPS = ((0, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30))


def prepare_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL).copy()
    panel = panel.loc[
        panel[TREATMENT].notna()
        & panel[DISTANCE].notna()
        & panel["Published Kampong Speu Replication Frame"].eq(1)
    ].copy()
    panel[TREATMENT] = panel[TREATMENT].astype("int8")
    panel[COMMUNE] = panel[COMMUNE].astype("string")
    panel["_district"] = panel[COMMUNE].str[:4]
    panel["_district_year"] = panel["_district"] + "_" + panel[YEAR].astype("string")
    panel["_segment_year"] = panel[SEGMENT].astype("string") + "_" + panel[YEAR].astype("string")
    primary_cells = panel.loc[
        panel["Historical-Boundary Common Support 5 km"].eq(1)
    ].drop_duplicates(CELL)
    side_count = primary_cells.groupby(COMMUNE, observed=True)[TREATMENT].nunique()
    panel["_cross_side_commune"] = panel[COMMUNE].isin(side_count.loc[side_count.eq(2)].index)
    return panel


def within_cell_sd(sample: pd.DataFrame, outcome: str) -> float:
    reference = sample[[CELL, outcome]].dropna()
    demeaned = reference[outcome].astype(float) - reference.groupby(CELL)[outcome].transform("mean")
    degrees = len(reference) - reference[CELL].nunique()
    if degrees <= 0:
        raise RuntimeError(f"Insufficient within-cell degrees of freedom for {outcome}")
    return float(np.sqrt(np.square(demeaned).sum() / degrees))


def classify(low: float, high: float) -> str:
    if low >= -SESOI and high <= SESOI:
        return "inside +/-0.20 SD"
    if low <= -SESOI and high >= SESOI:
        return "inconclusive relative to +/-0.20 SD"
    return "bounded but not equivalent"


def finish_record(
    fitted: object,
    sample: pd.DataFrame,
    *,
    target: str,
    specification: str,
    model_family: str,
    outcome: str,
    shock: str,
    period: str,
    source_stage: str,
    support_type: str,
    distance_range: str,
    bandwidth_km: float | None,
    confirmation: bool,
    triangular: bool,
    common_sd: float,
    period_sd: float,
    fixed_effects: str,
    interpretation: str,
) -> dict[str, object]:
    interval = fitted.conf_int().loc[target]
    estimate = float(fitted.params[target])
    low = float(interval["lower"])
    high = float(interval["upper"])
    cells = sample.drop_duplicates(CELL)
    commune_sides = cells.groupby(COMMUNE, observed=True)[TREATMENT].nunique()
    return {
        "specification": specification,
        "model_family": model_family,
        "outcome": outcome,
        "shock": shock,
        "period": period,
        "source_stage": source_stage,
        "support_type": support_type,
        "distance_range_km": distance_range,
        "bandwidth_km": bandwidth_km,
        "confirmation_model": confirmation,
        "triangular_weights": triangular,
        "estimate": estimate,
        "standard_error": float(fitted.std_errors[target]),
        "ci_low": low,
        "ci_high": high,
        "p_value": float(fitted.pvalues[target]),
        "common_standardized_estimate": estimate / common_sd,
        "common_standardized_ci_low": low / common_sd,
        "common_standardized_ci_high": high / common_sd,
        "period_standardized_estimate": estimate / period_sd,
        "period_standardized_ci_low": low / period_sd,
        "period_standardized_ci_high": high / period_sd,
        "common_sd_reference": common_sd,
        "period_sd_reference": period_sd,
        "sesoi_classification": classify(low / common_sd, high / common_sd),
        "observations": int(fitted.nobs),
        "grid_cells": int(sample[CELL].nunique()),
        "climate_communes": int(sample[COMMUNE].nunique()),
        "cross_side_communes": int(commune_sides.eq(2).sum()),
        "boundary_segments": int(sample[SEGMENT].nunique()),
        "district_year_clusters": int(sample["_district_year"].nunique()),
        "fixed_effects": fixed_effects,
        "inference": "Two-way clustered by grid cell and district-by-year",
        "interpretation": interpretation,
    }


def fit_cumulative(
    panel: pd.DataFrame,
    *,
    specification: str,
    outcome: str = OUTCOME,
    shock: str = PRIMARY_SHOCK,
    bandwidth: int = 5,
    start_year: int = 2000,
    end_year: int = 2024,
    source_stage: str = "full harmonized period",
    confirmation: bool = False,
    triangular: bool = False,
    common_sd: float,
) -> dict[str, object]:
    sample = panel.loc[
        panel[f"Historical-Boundary Common Support {bandwidth} km"].eq(1)
        & panel[YEAR].between(start_year, end_year)
    ].copy()
    if confirmation:
        sample = sample.loc[sample["_cross_side_commune"]].copy()
    required = [outcome, shock, CELL, YEAR, COMMUNE, SEGMENT, TREATMENT, DISTANCE]
    sample = sample.dropna(subset=required).copy()
    sample["_shock"] = sample[shock].astype(float)
    sample["_treat_shock"] = sample[TREATMENT] * sample["_shock"]
    sample["_distance_shock"] = sample[DISTANCE] * sample["_shock"]
    sample["_treat_distance_shock"] = sample[TREATMENT] * sample[DISTANCE] * sample["_shock"]
    exog = ["_treat_shock", "_distance_shock", "_treat_distance_shock"]
    absorb_names = [CELL, "_segment_year"]
    if confirmation:
        sample["_commune_year"] = sample[COMMUNE] + "_" + sample[YEAR].astype("string")
        absorb_names.append("_commune_year")
    else:
        exog.insert(0, "_shock")
    weights = None
    if triangular:
        weights = (1 - sample[ABS_DISTANCE] / bandwidth).clip(lower=1e-8)
    clusters = pd.DataFrame(
        {
            "cell": pd.factorize(sample[CELL])[0],
            "district_year": pd.factorize(sample["_district_year"])[0],
        },
        index=sample.index,
    )
    fitted = AbsorbingLS(
        dependent=sample[outcome].astype(float),
        exog=sample[exog].astype(float),
        absorb=sample[absorb_names].astype("category"),
        weights=weights,
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)
    period_sd = within_cell_sd(sample, outcome)
    return finish_record(
        fitted,
        sample,
        target="_treat_shock",
        specification=specification,
        model_family="cumulative boundary corridor",
        outcome=outcome,
        shock=shock,
        period=f"{start_year}-{end_year}",
        source_stage=source_stage,
        support_type="cumulative bandwidth",
        distance_range=f"0-{bandwidth}",
        bandwidth_km=float(bandwidth),
        confirmation=confirmation,
        triangular=triangular,
        common_sd=common_sd,
        period_sd=period_sd,
        fixed_effects=(
            "Grid cell; boundary segment x year; climate commune x year"
            if confirmation
            else "Grid cell; boundary segment x year"
        ),
        interpretation="Local boundary estimand" if bandwidth == 5 else "Bandwidth sensitivity",
    )


def fit_strip(
    panel: pd.DataFrame,
    *,
    low: int,
    high: int,
    common_sd: float,
) -> dict[str, object]:
    sample = panel.loc[
        panel[YEAR].between(2000, 2024)
        & panel[ABS_DISTANCE].gt(low)
        & panel[ABS_DISTANCE].le(high)
    ].dropna(subset=[OUTCOME, PRIMARY_SHOCK, CELL, COMMUNE, SEGMENT, TREATMENT]).copy()
    raw_cells = sample[CELL].nunique()
    sample["_absolute_distance_bin"] = np.floor(sample[ABS_DISTANCE]).astype(int)
    cell_support = sample.drop_duplicates(CELL)
    mirrored_support = cell_support.groupby(
        [SEGMENT, "_absolute_distance_bin"], observed=True
    )[TREATMENT].nunique()
    matched_keys = mirrored_support.loc[mirrored_support.eq(2)].index
    sample_keys = pd.MultiIndex.from_frame(sample[[SEGMENT, "_absolute_distance_bin"]])
    sample = sample.loc[sample_keys.isin(matched_keys)].copy()
    if sample.empty:
        raise RuntimeError(f"No matched mirrored support remains in strip {low}-{high} km")
    sample["_shock"] = sample[PRIMARY_SHOCK].astype(float)
    sample["_treat_shock"] = sample[TREATMENT] * sample["_shock"]
    sample["_mirror_year"] = (
        sample[SEGMENT].astype("string")
        + "_"
        + sample["_absolute_distance_bin"].astype("string")
        + "_"
        + sample[YEAR].astype("string")
    )
    clusters = pd.DataFrame(
        {
            "cell": pd.factorize(sample[CELL])[0],
            "district_year": pd.factorize(sample["_district_year"])[0],
        },
        index=sample.index,
    )
    fitted = AbsorbingLS(
        dependent=sample[OUTCOME].astype(float),
        exog=sample[["_shock", "_treat_shock"]].astype(float),
        absorb=sample[[CELL, "_mirror_year"]].astype("category"),
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)
    record = finish_record(
        fitted,
        sample,
        target="_treat_shock",
        specification=f"Mirrored strip {low}-{high} km",
        model_family="mirrored distance strip",
        outcome=OUTCOME,
        shock=PRIMARY_SHOCK,
        period="2000-2024",
        source_stage="full harmonized period",
        support_type="non-overlapping mirrored strip",
        distance_range=f"{low}-{high}",
        bandwidth_km=None,
        confirmation=False,
        triangular=False,
        common_sd=common_sd,
        period_sd=within_cell_sd(sample, OUTCOME),
        fixed_effects="Grid cell; boundary segment x 1-km absolute-distance bin x year",
        interpretation="Descriptive spatial gradient; not a local RD estimand",
    )
    record["sesoi_classification"] = "descriptive; formal equivalence not asserted"
    record["matched_mirrored_groups"] = int(len(matched_keys))
    record["unmatched_grid_cells_excluded"] = int(raw_cells - sample[CELL].nunique())
    return record


def append_observed_viirs(results: list[dict[str, object]]) -> None:
    viirs = pd.read_csv(VIIRS_RESULTS)
    for name in ["Primary 5 km", "Within-commune confirmation 5 km"]:
        row = viirs.loc[viirs["specification"].eq(name)].iloc[0]
        results.append(
            {
                "specification": f"Observed VIIRS benchmark: {name}",
                "model_family": "observed VIIRS benchmark",
                "outcome": row["outcome"],
                "shock": row["shock"],
                "period": "2013-2021",
                "source_stage": "observed EOG VIIRS benchmark",
                "support_type": "cumulative bandwidth",
                "distance_range_km": "0-5",
                "bandwidth_km": 5.0,
                "confirmation_model": bool(row["confirmation_model"]),
                "triangular_weights": False,
                "estimate": row["estimate"],
                "standard_error": row["standard_error"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "p_value": row["p_value"],
                "common_standardized_estimate": row["standardized_estimate"],
                "common_standardized_ci_low": row["standardized_ci_low"],
                "common_standardized_ci_high": row["standardized_ci_high"],
                "period_standardized_estimate": row["standardized_estimate"],
                "period_standardized_ci_low": row["standardized_ci_low"],
                "period_standardized_ci_high": row["standardized_ci_high"],
                "common_sd_reference": row["outcome_sd_reference"],
                "period_sd_reference": row["outcome_sd_reference"],
                "sesoi_classification": row["sesoi_classification"],
                "observations": int(row["observations"]),
                "grid_cells": int(row["grid_cells"]),
                "climate_communes": int(row["climate_communes"]),
                "cross_side_communes": int(row["cross_side_communes"]),
                "boundary_segments": int(row["boundary_segments"]),
                "district_year_clusters": int(row["district_year_clusters"]),
                "fixed_effects": (
                    "Grid cell; boundary segment x year; climate commune x year"
                    if bool(row["confirmation_model"])
                    else "Grid cell; boundary segment x year"
                ),
                "inference": "Two-way clustered by grid cell and district-by-year",
                "interpretation": "External observed-sensor measurement benchmark",
            }
        )


def estimate_all() -> pd.DataFrame:
    panel = prepare_panel()
    primary = panel.loc[panel["Historical-Boundary Common Support 5 km"].eq(1)]
    common_sd = within_cell_sd(primary, OUTCOME)
    extensive_sd = within_cell_sd(primary, EXTENSIVE)
    results: list[dict[str, object]] = []

    specifications = [
        dict(specification="Full period primary 5 km", common_sd=common_sd),
        dict(specification="Full period within-commune 5 km", confirmation=True, common_sd=common_sd),
        dict(specification="Full period triangular 5 km", triangular=True, common_sd=common_sd),
        dict(specification="Reconstructed-period primary 5 km", end_year=2012, source_stage="reconstructed", common_sd=common_sd),
        dict(specification="Reconstructed-period within-commune 5 km", end_year=2012, source_stage="reconstructed", confirmation=True, common_sd=common_sd),
        dict(specification="Observed-composite-period primary 5 km", start_year=2013, source_stage="observed annual composite", common_sd=common_sd),
        dict(specification="Observed-composite-period within-commune 5 km", start_year=2013, source_stage="observed annual composite", confirmation=True, common_sd=common_sd),
        dict(specification="Annual rainfall alternative 5 km", shock=ANNUAL_SHOCK, common_sd=common_sd),
        dict(specification="Any nonzero LongNTL primary 5 km", outcome=EXTENSIVE, common_sd=extensive_sd),
        dict(specification="Any nonzero LongNTL within-commune 5 km", outcome=EXTENSIVE, confirmation=True, common_sd=extensive_sd),
    ]
    specifications.extend(
        dict(specification=f"Fixed cumulative bandwidth {bandwidth} km", bandwidth=bandwidth, common_sd=common_sd)
        for bandwidth in (2, 10, 15, 20, 30)
    )
    for specification in specifications:
        print(f"Estimating {specification['specification']}", flush=True)
        results.append(fit_cumulative(panel, **specification))
    for low, high in STRIPS:
        print(f"Estimating mirrored strip {low}-{high} km", flush=True)
        results.append(fit_strip(panel, low=low, high=high, common_sd=common_sd))
    append_observed_viirs(results)
    output = pd.DataFrame(results)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT / "longntl_boundary_response_estimates.csv", index=False)
    return output


if __name__ == "__main__":
    estimates = estimate_all()
    print(estimates[["specification", "common_standardized_estimate", "common_standardized_ci_low", "common_standardized_ci_high"]].to_string(index=False))
