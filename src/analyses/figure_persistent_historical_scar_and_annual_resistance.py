#!/usr/bin/env python3
"""Persistent Historical Scar and Annual Resistance.

Plan: Estimate same-support NPP and nighttime-activity level/trend discontinuities and
place them beside the completed annual rainfall-response equivalence estimates.
Framework: AnaSOP Sections 5.7, 6.13, 6.15, and the corresponding Section 7 steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[2]
NPP_PANEL = ROOT / "data/processed/historical_boundary_annual_spatial_climate_preprocessed.parquet"
VIIRS_PANEL = ROOT / "data/processed/viirs_historical_boundary_climate_preprocessed.parquet"
SOCIAL_PANEL = ROOT / "data/processed/grasse_historical_boundary_level_outcomes_preprocessed.parquet"
NPP_ANNUAL_TABLE = ROOT / "data/exp/legacy-results/tables/Table_historical_boundary_shock_response_estimates.xlsx"
VIIRS_ANNUAL_TABLE = ROOT / "data/exp/legacy-results/tables/Table_nighttime_activity_independent_validation_estimates.xlsx"
OUTPUT = ROOT / "data/exp/legacy-results/figures/Figure_persistent_historical_scar_and_annual_resistance.png"
EXP_DIR = ROOT / "data/exp/same-support-scar-resistance"

TREATMENT = "Higher-Repression Southwest Zone"
DISTANCE = "Signed Distance to Historical Repression Boundary km"
ABS_DISTANCE = "Absolute Distance to Historical Repression Boundary km"
SEGMENT = "Historical Boundary Segment"
LONGITUDE = "Longitude"
LATITUDE = "Latitude"
YEAR = "Year"
COMMUNE = "Linked Climate Commune Code"
PRIMARY_BANDWIDTH = 5
PRIMARY_CONLEY_KM = 20.0

COLORS = {"West": "#3C78B5", "Southwest": "#C94C35"}
SPEC_COLORS = {"Primary": "#222222", "Within commune": "#B05A3C"}


@dataclass(frozen=True)
class RDFit:
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    p_value: float
    observations: int
    southwest_units: int
    west_units: int
    communes: int
    segments: int
    outcome_sd: float
    standardized_estimate: float
    standardized_ci_low: float
    standardized_ci_high: float
    params: np.ndarray
    columns: tuple[str, ...]


def unit_slope(group: pd.DataFrame, outcome: str, minimum_years: int) -> float:
    valid = group[[YEAR, outcome]].dropna().drop_duplicates(YEAR)
    if len(valid) < minimum_years:
        return np.nan
    centered = valid[YEAR].to_numpy(float) - valid[YEAR].mean()
    return float(np.polyfit(centered, valid[outcome].to_numpy(float), 1)[0])


def prepare_npp() -> pd.DataFrame:
    mean_col = "Village 2001-2020 Mean Annual Land NPP kg C per m2"
    annual_col = "Annual Land NPP Mean kg C per m2"
    columns = [
        "Village Code", LONGITUDE, LATITUDE, YEAR, COMMUNE, SEGMENT, TREATMENT,
        DISTANCE, ABS_DISTANCE, mean_col, annual_col,
        "Historical-Boundary Common Support 10 km",
    ]
    panel = pd.read_parquet(NPP_PANEL, columns=columns)
    panel = panel.loc[panel["Historical-Boundary Common Support 10 km"].eq(1)].copy()
    records = []
    for unit, group in panel.groupby("Village Code", observed=True, sort=False):
        first = group.iloc[0]
        records.append({
            "unit": str(unit), "domain": "Land NPP", LONGITUDE: first[LONGITUDE],
            LATITUDE: first[LATITUDE], COMMUNE: str(first[COMMUNE]), SEGMENT: str(first[SEGMENT]),
            TREATMENT: int(first[TREATMENT]), DISTANCE: float(first[DISTANCE]),
            ABS_DISTANCE: float(first[ABS_DISTANCE]),
            "level": float(group[mean_col].dropna().iloc[0]),
            "trend": 10.0 * unit_slope(group, annual_col, 15),
        })
    return pd.DataFrame(records)


def prepare_viirs() -> pd.DataFrame:
    annual_col = "Asinh Annual Mean Radiance"
    columns = [
        "Grid Cell ID", LONGITUDE, LATITUDE, YEAR, COMMUNE, SEGMENT, TREATMENT,
        DISTANCE, ABS_DISTANCE, annual_col, "Historical-Boundary Common Support 10 km",
    ]
    panel = pd.read_parquet(VIIRS_PANEL, columns=columns)
    panel = panel.loc[panel["Historical-Boundary Common Support 10 km"].eq(1)].copy()
    records = []
    for unit, group in panel.groupby("Grid Cell ID", observed=True, sort=False):
        first = group.iloc[0]
        records.append({
            "unit": str(unit), "domain": "Nighttime activity", LONGITUDE: first[LONGITUDE],
            LATITUDE: first[LATITUDE], COMMUNE: str(first[COMMUNE]), SEGMENT: str(first[SEGMENT]),
            TREATMENT: int(first[TREATMENT]), DISTANCE: float(first[DISTANCE]),
            ABS_DISTANCE: float(first[ABS_DISTANCE]),
            "level": float(group[annual_col].mean()),
            "trend": 10.0 * unit_slope(group, annual_col, 7),
        })
    return pd.DataFrame(records)


def prepare_social() -> pd.DataFrame:
    data = pd.read_parquet(SOCIAL_PANEL).copy()
    commune_link = pd.read_parquet(
        NPP_PANEL, columns=["Village Code", "Linked Climate Commune Code"]
    ).drop_duplicates("Village Code")
    data = data.merge(commune_link, on="Village Code", how="left", validate="one_to_one")
    data = data.rename(columns={"Linked Climate Commune Code": COMMUNE})
    data[COMMUNE] = data[COMMUNE].astype("string")
    data[SEGMENT] = data[SEGMENT].astype("string")
    data[ABS_DISTANCE] = data[DISTANCE].abs()
    data["unit"] = data["Village Code"].astype(str)
    data["domain"] = "Village welfare"
    return data


def haversine_matrix(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    lon_r = np.radians(lon)
    lat_r = np.radians(lat)
    dlon = lon_r[:, None] - lon_r[None, :]
    dlat = lat_r[:, None] - lat_r[None, :]
    a = np.sin(dlat / 2.0) ** 2 + (
        np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def design_matrix(sample: pd.DataFrame, confirmation: bool) -> tuple[np.ndarray, tuple[str, ...]]:
    pieces = [
        pd.Series(1.0, index=sample.index, name="Intercept"),
        sample[TREATMENT].astype(float).rename("Southwest"),
        sample[DISTANCE].astype(float).rename("Distance"),
        (sample[TREATMENT] * sample[DISTANCE]).astype(float).rename("Southwest x distance"),
    ]
    segment_dummies = pd.get_dummies(sample[SEGMENT].astype(str), prefix="Segment", drop_first=True, dtype=float)
    pieces.extend(segment_dummies[col] for col in segment_dummies.columns)
    if confirmation:
        commune_dummies = pd.get_dummies(sample[COMMUNE].astype(str), prefix="Commune", drop_first=True, dtype=float)
        pieces.extend(commune_dummies[col] for col in commune_dummies.columns)
    frame = pd.concat(pieces, axis=1)
    matrix = frame.to_numpy(float)
    mandatory = list(range(4))
    if np.linalg.matrix_rank(matrix[:, mandatory]) != len(mandatory):
        raise ValueError("Core treatment and distance terms are not separately identified")
    selected = mandatory.copy()
    current_rank = len(selected)
    for candidate in range(4, matrix.shape[1]):
        trial = selected + [candidate]
        trial_rank = np.linalg.matrix_rank(matrix[:, trial])
        if trial_rank > current_rank:
            selected.append(candidate)
            current_rank = trial_rank
    reduced = frame.iloc[:, selected]
    return reduced.to_numpy(float), tuple(reduced.columns.astype(str))


def fit_rd(
    data: pd.DataFrame,
    outcome: str,
    confirmation: bool,
    cutoff_km: float = PRIMARY_CONLEY_KM,
    bandwidth_km: float = PRIMARY_BANDWIDTH,
) -> RDFit:
    sample = data.loc[data[ABS_DISTANCE].le(bandwidth_km)].dropna(
        subset=[outcome, TREATMENT, DISTANCE, SEGMENT, LONGITUDE, LATITUDE, COMMUNE]
    ).copy()
    if confirmation:
        cross_side = sample.groupby(COMMUNE, observed=True)[TREATMENT].nunique()
        sample = sample.loc[sample[COMMUNE].isin(cross_side.loc[cross_side.eq(2)].index)].copy()
    y = sample[outcome].to_numpy(float)
    outcome_sd = float(np.std(y, ddof=1))
    x, columns = design_matrix(sample, confirmation)
    rank = np.linalg.matrix_rank(x)
    if rank != x.shape[1]:
        raise ValueError(f"Rank-deficient level model: rank={rank}, columns={x.shape[1]}")
    bread = np.linalg.inv(x.T @ x)
    params = bread @ (x.T @ y)
    residual = y - x @ params
    distance_km = haversine_matrix(sample[LONGITUDE].to_numpy(float), sample[LATITUDE].to_numpy(float))
    kernel = np.clip(1.0 - distance_km / cutoff_km, 0.0, 1.0)
    scores = x * residual[:, None]
    meat = scores.T @ kernel @ scores
    correction = len(sample) / max(len(sample) - x.shape[1], 1)
    covariance = correction * bread @ meat @ bread
    target = columns.index("Southwest")
    estimate = float(params[target])
    standard_error = float(np.sqrt(max(covariance[target, target], 0.0)))
    ci_low = estimate - 1.96 * standard_error
    ci_high = estimate + 1.96 * standard_error
    p_value = float(2.0 * norm.sf(abs(estimate / standard_error))) if standard_error > 0 else np.nan
    return RDFit(
        estimate=estimate, standard_error=standard_error, ci_low=ci_low, ci_high=ci_high,
        p_value=p_value, observations=len(sample),
        southwest_units=int(sample[TREATMENT].eq(1).sum()), west_units=int(sample[TREATMENT].eq(0).sum()),
        communes=int(sample[COMMUNE].nunique()), segments=int(sample[SEGMENT].nunique()),
        outcome_sd=outcome_sd, standardized_estimate=estimate / outcome_sd,
        standardized_ci_low=ci_low / outcome_sd, standardized_ci_high=ci_high / outcome_sd,
        params=params, columns=columns,
    )


def build_level_estimates(npp: pd.DataFrame, viirs: pd.DataFrame, social: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for data, domain in ((npp, "Land NPP"), (viirs, "Nighttime activity")):
        for outcome, estimand in (("level", "Long-run level"), ("trend", "Linear trend per decade")):
            for confirmation, specification in ((False, "Primary"), (True, "Within commune")):
                fit = fit_rd(data, outcome, confirmation)
                rows.append({
                    "evidence_component": "Persistent scar", "domain": domain,
                    "estimand": estimand, "specification": specification,
                    "period": "2001-2020" if domain == "Land NPP" and outcome == "level" else
                              "2001-2021" if domain == "Land NPP" else "2013-2021",
                    "bandwidth_km": PRIMARY_BANDWIDTH, "scale": "Natural units",
                    "estimate": fit.estimate, "standard_error": fit.standard_error,
                    "ci_low": fit.ci_low, "ci_high": fit.ci_high, "p_value": fit.p_value,
                    "standardized_estimate": fit.standardized_estimate,
                    "standardized_ci_low": fit.standardized_ci_low,
                    "standardized_ci_high": fit.standardized_ci_high,
                    "compatibility_frontier_sd": max(abs(fit.standardized_ci_low), abs(fit.standardized_ci_high)),
                    "observations": fit.observations, "southwest_units": fit.southwest_units,
                    "west_units": fit.west_units, "communes": fit.communes,
                    "segments": fit.segments, "inference": f"Conley spatial HAC, Bartlett {PRIMARY_CONLEY_KM:g} km",
                })
    for outcome, estimand in (
        ("Village Poverty Rate Percent", "Village poverty rate"),
        ("Mean Years of Schooling", "Mean years of schooling"),
        ("Adult Literacy Rate Percent", "Adult literacy rate"),
    ):
        for confirmation, specification in ((False, "Primary"), (True, "Within commune")):
            fit = fit_rd(social, outcome, confirmation)
            rows.append({
                "evidence_component": "Persistent scar", "domain": "Village welfare",
                "estimand": estimand, "specification": specification,
                "period": "Public replication cross-section", "bandwidth_km": PRIMARY_BANDWIDTH,
                "scale": "Natural units", "estimate": fit.estimate,
                "standard_error": fit.standard_error, "ci_low": fit.ci_low,
                "ci_high": fit.ci_high, "p_value": fit.p_value,
                "standardized_estimate": fit.standardized_estimate,
                "standardized_ci_low": fit.standardized_ci_low,
                "standardized_ci_high": fit.standardized_ci_high,
                "compatibility_frontier_sd": max(abs(fit.standardized_ci_low), abs(fit.standardized_ci_high)),
                "observations": fit.observations, "southwest_units": fit.southwest_units,
                "west_units": fit.west_units, "communes": fit.communes,
                "segments": fit.segments, "inference": f"Conley spatial HAC, Bartlett {PRIMARY_CONLEY_KM:g} km",
            })
    return pd.DataFrame(rows)


def build_annual_estimates() -> pd.DataFrame:
    npp = pd.read_excel(NPP_ANNUAL_TABLE, sheet_name="Shock Response")
    npp = npp.loc[npp["Outcome"].eq("Annual land NPP anomaly (standardized)")].head(2)
    viirs = pd.read_excel(VIIRS_ANNUAL_TABLE, sheet_name="NTL Validation").head(2)
    rows = []
    for domain, frame, scale_col in (
        ("Land NPP", npp, "Scale"), ("Nighttime activity", viirs, "Standardized scale")
    ):
        for idx, row in frame.reset_index(drop=True).iterrows():
            low, high = [float(value.strip()) for value in str(row["95% CI"]).strip("[]").split(",")]
            rows.append({
                "evidence_component": "Annual resistance", "domain": domain,
                "estimand": "Rainfall-response difference", "specification": "Primary" if idx == 0 else "Within commune",
                "period": "2001-2021" if domain == "Land NPP" else "2013-2021",
                "bandwidth_km": float(row["Bandwidth (km)"]), "scale": str(row[scale_col]),
                "estimate": float(row["Interaction estimate"]), "standard_error": np.nan,
                "ci_low": low, "ci_high": high, "p_value": np.nan,
                "standardized_estimate": float(row["Interaction estimate"]),
                "standardized_ci_low": low, "standardized_ci_high": high,
                "compatibility_frontier_sd": max(abs(low), abs(high)),
                "observations": np.nan, "southwest_units": np.nan, "west_units": np.nan,
                "communes": np.nan, "segments": np.nan, "inference": str(row["Inference"]),
            })
    return pd.DataFrame(rows)


def binned_panel(ax: plt.Axes, data: pd.DataFrame, outcome: str, ylabel: str) -> None:
    sample = data.loc[data[ABS_DISTANCE].le(PRIMARY_BANDWIDTH)].dropna(subset=[outcome]).copy()
    sample["bin"] = np.floor((sample[DISTANCE] + 5.0) / 0.5) * 0.5 - 5.0 + 0.25
    for treatment, side in ((0, "West"), (1, "Southwest")):
        side_data = sample.loc[sample[TREATMENT].eq(treatment)]
        bins = side_data.groupby("bin", observed=True).agg(x=(DISTANCE, "mean"), y=(outcome, "mean"), n=(outcome, "size")).reset_index()
        ax.scatter(bins["x"], bins["y"], s=np.sqrt(bins["n"]) * 11, color=COLORS[side], alpha=0.78, edgecolor="white", linewidth=0.4)
        coefficients = np.polyfit(side_data[DISTANCE], side_data[outcome], 1)
        xgrid = np.linspace(side_data[DISTANCE].min(), side_data[DISTANCE].max(), 100)
        ax.plot(xgrid, np.polyval(coefficients, xgrid), color=COLORS[side], linewidth=2.0, label=side)
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_xlim(-5, 5)
    ax.set_xlabel("Signed distance to historical boundary (km)")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#E2E2E2", linewidth=0.55)
    ax.legend(frameon=False, fontsize=8)


def forest_panel(ax: plt.Axes, estimates: pd.DataFrame, annual: bool = False) -> None:
    if annual:
        order = [
            ("Land NPP", "Primary"), ("Land NPP", "Within commune"),
            ("Nighttime activity", "Primary"), ("Nighttime activity", "Within commune"),
        ]
        labels = ["NPP: primary", "NPP: within commune", "VIIRS: primary", "VIIRS: within commune"]
        ax.axvspan(-0.20, 0.20, color="#DCE9D8", alpha=0.7, zorder=0)
    else:
        order = [
            ("Village welfare", "Village poverty rate", "Primary"), ("Village welfare", "Village poverty rate", "Within commune"),
            ("Village welfare", "Mean years of schooling", "Primary"), ("Village welfare", "Mean years of schooling", "Within commune"),
            ("Land NPP", "Long-run level", "Primary"), ("Land NPP", "Long-run level", "Within commune"),
            ("Nighttime activity", "Long-run level", "Primary"), ("Nighttime activity", "Long-run level", "Within commune"),
            ("Land NPP", "Linear trend per decade", "Primary"), ("Land NPP", "Linear trend per decade", "Within commune"),
            ("Nighttime activity", "Linear trend per decade", "Primary"), ("Nighttime activity", "Linear trend per decade", "Within commune"),
        ]
        labels = ["Poverty: primary", "Poverty: within commune", "Schooling: primary", "Schooling: within commune",
                  "NPP level: primary", "NPP level: within commune", "VIIRS level: primary", "VIIRS level: within commune",
                  "NPP trend: primary", "NPP trend: within commune", "VIIRS trend: primary", "VIIRS trend: within commune"]
    rows = []
    for key in order:
        mask = estimates["domain"].eq(key[0]) & estimates["specification"].eq(key[-1])
        if not annual:
            mask &= estimates["estimand"].eq(key[1])
        rows.append(estimates.loc[mask].iloc[0])
    for y, (row, label) in enumerate(zip(rows, labels)):
        color = SPEC_COLORS[row["specification"]]
        ax.errorbar(row["standardized_estimate"], y,
                    xerr=[[row["standardized_estimate"] - row["standardized_ci_low"]],
                          [row["standardized_ci_high"] - row["standardized_estimate"]]],
                    fmt="o", color=color, ecolor=color, capsize=2.5, markersize=4.5, zorder=2)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Southwest-minus-West difference (outcome SD)" + ("\nper 1-SD rainfall shock" if annual else ""))
    ax.grid(True, axis="x", color="#E2E2E2", linewidth=0.55)


def main() -> None:
    npp = prepare_npp()
    viirs = prepare_viirs()
    social = prepare_social()
    level = build_level_estimates(npp, viirs, social)
    annual = build_annual_estimates()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([level, annual], ignore_index=True)
    combined.to_csv(EXP_DIR / "same_support_scar_and_annual_resistance_estimates.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 10.2), constrained_layout=True)
    binned_panel(axes[0, 0], social, "Village Poverty Rate Percent", "Village poverty rate (%)")
    binned_panel(axes[0, 1], social, "Mean Years of Schooling", "Mean years of schooling")
    forest_panel(axes[1, 0], level, annual=False)
    forest_panel(axes[1, 1], annual, annual=True)
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
        for spine in ax.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.8)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"Saved: {(EXP_DIR / 'same_support_scar_and_annual_resistance_estimates.csv').relative_to(ROOT)}")
    print(combined[["evidence_component", "domain", "estimand", "specification", "standardized_estimate", "standardized_ci_low", "standardized_ci_high", "compatibility_frontier_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()
