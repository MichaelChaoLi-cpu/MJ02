#!/usr/bin/env python3
"""Conflict-Conditioned Climate Sensitivity.

Plan: Estimate the area-level ITT direct effect, the post-conflict change in
drought sensitivity, heat and compound extensions, and the mandatory 2/5/10 km
scale sensitivity.
Framework: AnaSOP Sections 5-7 stacked village fixed-effects model with full
lower-order interactions, stack-year effects, predetermined control-by-year
terms, sector-stratified outcome-blind calibration, and spatial-block clustered
inference.
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
from linearmodels.panel import PanelOLS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_predetermined_balance_and_common_support import (  # noqa: E402
    build_sample,
    sector_calibration,
)


PANEL = ROOT / "data/processed/cambodia_public_village_npp_conflict_panel_candidate_preprocessed.parquet"
OUT = ROOT / "data/exp/legacy-results/figures/Figure_conflict_conditioned_climate_sensitivity.png"
AUDIT_DIR = ROOT / "data/exp/experiments/cambodia-thailand-village-area-itt-climate"
COEFFICIENT_OUT = AUDIT_DIR / "climate_model_structural_coefficients.csv"
SUMMARY_OUT = AUDIT_DIR / "headline_and_sensitivity_estimates.csv"
RESPONSE_OUT = AUDIT_DIR / "post_conflict_drought_response_lines.csv"
METADATA_OUT = AUDIT_DIR / "model_metadata.json"

ID = "National Village Point ID"
YEAR = "Year"
RADIUS = "Buffer Radius km"
NPP = "Village Buffer Mean Annual Land NPP Anomaly Z 2001-2020"
DRY = "Village Buffer Mean May October Dry Rainfall Intensity"
HEAT = "Village Buffer Mean May October Heat Intensity"
COMPOUND = "Village Buffer Mean May October Compound Hot-Dry Intensity"
LON = "Point Longitude"
LAT = "Point Latitude"
TREATMENT = "Candidate Affected District"
SECTOR = "Candidate Conflict Sector"

SECTOR_YEARS = {"Preah Vihear": 2008, "Ta Moan-Ta Krabey": 2011}
RADII = [2, 5, 10]
PRIMARY_RADIUS_KM = 5
CONTROL_FEATURES = [
    "Pre-conflict NPP mean",
    "Pre-conflict NPP trend",
    "Baseline cropland share",
    "Log baseline population",
]
HAZARD_LABELS = {
    DRY: "Drought",
    HEAT: "Heat",
    COMPOUND: "Compound hot–dry",
}


def normalize_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.sum() <= 0:
        raise RuntimeError("Analysis weights have no positive mass")
    return values / values.sum()


def prepare_design() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    treated, controls = build_sample()
    design = sector_calibration(treated, controls)
    baseline = pd.concat([treated, controls], ignore_index=True)
    for variable in CONTROL_FEATURES:
        mean = baseline[variable].mean()
        sd = baseline[variable].std(ddof=0)
        baseline[f"z {variable}"] = (baseline[variable] - mean) / sd
    return treated, controls, design["sector_control_weights"], baseline


def build_stacked_panel(radius_km: int) -> pd.DataFrame:
    treated, controls, sector_control_weights, baseline = prepare_design()
    baseline = baseline[
        [ID, *[f"z {variable}" for variable in CONTROL_FEATURES]]
    ].copy()
    baseline[ID] = baseline[ID].astype(str)
    panel = pd.read_parquet(
        PANEL,
        columns=[ID, YEAR, RADIUS, NPP, DRY, HEAT, COMPOUND, LON, LAT, TREATMENT, SECTOR],
        filters=[(RADIUS, "=", radius_km)],
    )
    panel = panel.dropna(subset=[ID, YEAR, NPP, DRY, HEAT, COMPOUND, LON, LAT]).copy()
    panel[ID] = panel[ID].astype(str)
    panel = panel.merge(baseline, on=ID, how="inner", validate="many_to_one")
    control_ids = controls[ID].astype(str).tolist()
    frames: list[pd.DataFrame] = []

    for sector, first_year in SECTOR_YEARS.items():
        sector_treated = treated.loc[treated[SECTOR].eq(sector)].copy()
        sector_share = len(sector_treated) / len(treated)
        treated_weights = pd.Series(
            sector_share / len(sector_treated),
            index=sector_treated[ID].astype(str),
        )
        control_weights = pd.Series(
            normalize_positive(sector_control_weights[sector]) * sector_share,
            index=controls[ID].astype(str),
        )
        weight_map = pd.concat([treated_weights, control_weights])
        selected_ids = set(sector_treated[ID].astype(str)).union(control_ids)
        stack = panel.loc[panel[ID].isin(selected_ids)].copy()
        stack["Conflict Sector"] = sector
        stack["First Conflict Year"] = first_year
        stack["Area ITT"] = stack[ID].isin(set(sector_treated[ID].astype(str))).astype(int)
        stack["Post Conflict Period"] = stack[YEAR].ge(first_year).astype(int)
        stack["Analysis Weight"] = stack[ID].map(weight_map)
        if stack["Analysis Weight"].isna().any():
            raise RuntimeError(f"Missing analysis weights in {sector} stack")
        stack = stack.loc[stack["Analysis Weight"].gt(0)].copy()
        stack["Stack Village ID"] = sector + "__" + stack[ID]
        stack["Stack Year"] = sector + "__" + stack[YEAR].astype(str)
        stack["Spatial Block ID"] = (
            np.floor(stack[LON].to_numpy(float) * 10).astype(int).astype(str)
            + "_"
            + np.floor(stack[LAT].to_numpy(float) * 10).astype(int).astype(str)
        )
        frames.append(stack)

    stacked = pd.concat(frames, ignore_index=True)
    if stacked.duplicated(["Stack Village ID", YEAR]).any():
        raise RuntimeError("Stacked panel is not unique by stack-village and year")
    return stacked


def add_control_by_year_terms(sample: pd.DataFrame) -> list[str]:
    terms: list[str] = []
    reference_year = int(sample[YEAR].min())
    for variable in CONTROL_FEATURES:
        standardized = f"z {variable}"
        for year in sorted(sample[YEAR].unique()):
            if int(year) == reference_year:
                continue
            term = f"{standardized} x year {int(year)}"
            sample[term] = sample[standardized] * sample[YEAR].eq(year).astype(int)
            terms.append(term)
    return terms


def fit_panel(sample: pd.DataFrame, terms: list[str]) -> object:
    panel = sample.set_index(["Stack Village ID", YEAR]).sort_index()
    other_effects = pd.DataFrame(
        {"Stack x year": pd.Categorical(panel["Stack Year"]).codes},
        index=panel.index,
    )
    clusters = pd.DataFrame(
        {"10 km spatial block": pd.Categorical(panel["Spatial Block ID"]).codes},
        index=panel.index,
    )
    return PanelOLS(
        panel[NPP].astype(float),
        panel[terms].astype(float),
        weights=panel["Analysis Weight"].astype(float),
        entity_effects=True,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)


def coefficient_row(fitted: object, term: str, **fields: object) -> dict[str, object]:
    confidence = fitted.conf_int(level=0.95)
    return {
        **fields,
        "Term": term,
        "Estimate": float(fitted.params[term]),
        "Clustered Standard Error": float(fitted.std_errors[term]),
        "95% CI Lower": float(confidence.loc[term, "lower"]),
        "95% CI Upper": float(confidence.loc[term, "upper"]),
        "p-value": float(fitted.pvalues[term]),
        "Observations": int(fitted.nobs),
    }


def fit_direct(stacked: pd.DataFrame, radius_km: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    sample = stacked.copy()
    target = "Area ITT x post"
    sample[target] = sample["Area ITT"] * sample["Post Conflict Period"]
    control_terms = add_control_by_year_terms(sample)
    fitted = fit_panel(sample, [target, *control_terms])
    row = coefficient_row(
        fitted,
        target,
        Model="Direct area-level ITT",
        Hazard="None",
        **{"Buffer Radius km": radius_km, "Estimand": "Post-conflict NPP level difference"},
    )
    return row, [row]


def add_hazard_terms(sample: pd.DataFrame, hazard: str, prefix: str) -> list[str]:
    names = [
        f"{prefix}: hazard",
        f"{prefix}: area x hazard",
        f"{prefix}: post x hazard",
        f"{prefix}: area x post",
        f"{prefix}: area x post x hazard",
    ]
    sample[names[0]] = sample[hazard]
    sample[names[1]] = sample["Area ITT"] * sample[hazard]
    sample[names[2]] = sample["Post Conflict Period"] * sample[hazard]
    sample[names[3]] = sample["Area ITT"] * sample["Post Conflict Period"]
    sample[names[4]] = sample["Area ITT"] * sample["Post Conflict Period"] * sample[hazard]
    return names


def linear_combination(fitted: object, terms: list[str]) -> tuple[float, float, float, float]:
    vector = np.zeros(len(fitted.params))
    for term in terms:
        vector[fitted.params.index.get_loc(term)] = 1.0
    estimate = float(vector @ fitted.params.to_numpy())
    variance = float(vector @ fitted.cov.to_numpy() @ vector)
    standard_error = float(np.sqrt(max(variance, 0)))
    return estimate, standard_error, estimate - 1.96 * standard_error, estimate + 1.96 * standard_error


def fit_climate(
    stacked: pd.DataFrame,
    hazard: str,
    radius_km: int,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[tuple[str, str], tuple[float, float, float, float]] | None,
]:
    sample = stacked.copy()
    label = HAZARD_LABELS[hazard]
    target_terms = add_hazard_terms(sample, hazard, label)
    structural_terms = list(target_terms)
    if hazard == COMPOUND:
        # Estimate the incremental compound component while retaining the full
        # lower-order drought and heat response structures.
        structural_terms = [
            *add_hazard_terms(sample, DRY, "Drought control"),
            *add_hazard_terms(sample, HEAT, "Heat control"),
            *target_terms,
        ]
        # The direct area-by-post term is shared across the three hierarchies.
        duplicate_direct = [term for term in structural_terms if term.endswith(": area x post")]
        keep_direct = duplicate_direct[0]
        for term in duplicate_direct[1:]:
            sample.drop(columns=term, inplace=True)
        structural_terms = [term for term in structural_terms if not term.endswith(": area x post") or term == keep_direct]
    control_terms = add_control_by_year_terms(sample)
    fitted = fit_panel(sample, [*structural_terms, *control_terms])
    target = target_terms[-1]
    summary = coefficient_row(
        fitted,
        target,
        Model=f"{label} amplification",
        Hazard=label,
        **{"Buffer Radius km": radius_km, "Estimand": "Post-conflict change in affected-area climate slope"},
    )
    structural_rows = [
        coefficient_row(
            fitted,
            term,
            Model=f"{label} amplification",
            Hazard=label,
            **{"Buffer Radius km": radius_km, "Estimand": "Structural coefficient"},
        )
        for term in structural_terms
    ]
    response: dict[tuple[str, str], tuple[float, float, float, float]] | None = None
    if hazard == DRY:
        response = {
            ("Calibrated comparison", "Before conflict"): linear_combination(fitted, [target_terms[0]]),
            ("Affected area", "Before conflict"): linear_combination(
                fitted,
                [target_terms[0], target_terms[1]],
            ),
            ("Calibrated comparison", "After conflict"): linear_combination(
                fitted,
                [target_terms[0], target_terms[2]],
            ),
            ("Affected area", "After conflict"): linear_combination(
                fitted,
                [target_terms[0], target_terms[1], target_terms[2], target_terms[4]],
            ),
        }
    return summary, structural_rows, response


def build_response_lines(
    stacked: pd.DataFrame,
    slopes: dict[tuple[str, str], tuple[float, float, float, float]],
) -> pd.DataFrame:
    weights = normalize_positive(stacked["Analysis Weight"].to_numpy(float))
    order = np.argsort(stacked[DRY].to_numpy(float))
    values = stacked[DRY].to_numpy(float)[order]
    cumulative = np.cumsum(weights[order])
    upper = float(values[np.searchsorted(cumulative, 0.95, side="left")])
    grid = np.linspace(0, upper, 60)
    rows: list[dict[str, object]] = []
    for (group, period), (slope, slope_se, _, _) in slopes.items():
        for drought in grid:
            estimate = slope * drought
            se = slope_se * drought
            rows.append(
                {
                    "Group": group,
                    "Period": period,
                    "Line Label": f"{group}, {period.lower()}",
                    "Drought Intensity": drought,
                    "Predicted NPP Change": estimate,
                    "95% CI Lower": estimate - 1.96 * se,
                    "95% CI Upper": estimate + 1.96 * se,
                    "Post-conflict Slope": slope,
                }
            )
    return pd.DataFrame(rows)


def forest(ax: plt.Axes, data: pd.DataFrame, labels: list[str], colors: list[str], y_offsets: np.ndarray | None = None) -> None:
    frame = data.reset_index(drop=True)
    y = np.arange(len(frame), dtype=float) if y_offsets is None else y_offsets
    for index, row in frame.iterrows():
        ax.errorbar(
            row["Estimate"],
            y[index],
            xerr=[[row["Estimate"] - row["95% CI Lower"]], [row["95% CI Upper"] - row["Estimate"]]],
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            elinewidth=1.6,
            capsize=3,
            markersize=6,
        )
    ax.axvline(0, color="#5A5A5A", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.grid(True, axis="x", color="#E3E7EA", linewidth=0.6)
    ax.grid(False, axis="y")


def draw_figure(summary: pd.DataFrame, response: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.4), constrained_layout=True)
    axes = axes.ravel()

    headline = pd.concat(
        [
            summary.loc[
                summary["Model"].eq("Direct area-level ITT") & summary["Buffer Radius km"].eq(PRIMARY_RADIUS_KM)
            ],
            summary.loc[
                summary["Model"].eq("Drought amplification") & summary["Buffer Radius km"].eq(PRIMARY_RADIUS_KM)
            ],
        ],
        ignore_index=True,
    )
    forest(
        axes[0],
        headline,
        ["Post-conflict level", "Drought amplification"],
        ["#5B5B5B", "#C4493D"],
    )
    axes[0].set_xlabel("Coefficient (NPP SD)")
    axes[0].text(0.98, 0.95, "Primary 5 km estimates", transform=axes[0].transAxes, ha="right", va="top", fontsize=10)

    response_colors = {"Affected area": "#C4493D", "Calibrated comparison": "#2F6B8A"}
    period_styles = {"Before conflict": "--", "After conflict": "-"}
    for (group, period), frame in response.groupby(["Group", "Period"], observed=True):
        frame = frame.sort_values("Drought Intensity")
        axes[1].plot(
            frame["Drought Intensity"],
            frame["Predicted NPP Change"],
            color=response_colors[group],
            linewidth=2 if period == "After conflict" else 1.6,
            linestyle=period_styles[period],
            label=f"{group}, {period.lower()}",
        )
        axes[1].fill_between(
            frame["Drought Intensity"],
            frame["95% CI Lower"],
            frame["95% CI Upper"],
            color=response_colors[group],
            alpha=0.10 if period == "Before conflict" else 0.13,
            linewidth=0,
        )
    axes[1].axhline(0, color="#5A5A5A", linewidth=0.8)
    axes[1].set_xlabel("May–October drought intensity")
    axes[1].set_ylabel("Predicted NPP change (SD)")
    axes[1].legend(frameon=False, loc="lower left", fontsize=8)
    axes[1].text(0.98, 0.95, "Drought response before and after conflict", transform=axes[1].transAxes, ha="right", va="top", fontsize=10)

    extensions = summary.loc[
        summary["Model"].isin(["Drought amplification", "Heat amplification", "Compound hot–dry amplification"])
        & summary["Buffer Radius km"].eq(PRIMARY_RADIUS_KM)
    ].copy()
    extensions["order"] = extensions["Model"].map(
        {"Drought amplification": 0, "Heat amplification": 1, "Compound hot–dry amplification": 2}
    )
    extensions = extensions.sort_values("order")
    forest(
        axes[2],
        extensions,
        ["Drought", "Heat", "Compound hot–dry"],
        ["#C4493D", "#D99642", "#7B4F9D"],
    )
    axes[2].set_xlabel("Slope-change coefficient (NPP SD)")
    axes[2].text(0.98, 0.95, "Climate-shock extensions", transform=axes[2].transAxes, ha="right", va="top", fontsize=10)

    scale = summary.loc[summary["Model"].eq("Drought amplification")].sort_values("Buffer Radius km")
    forest(
        axes[3],
        scale,
        [f"{int(radius)} km" for radius in scale["Buffer Radius km"]],
        ["#2F6B8A", "#C4493D", "#6B8E5A"],
    )
    axes[3].set_xlabel("Drought-amplification coefficient (NPP SD)")
    axes[3].text(0.98, 0.95, "Village-buffer sensitivity", transform=axes[3].transAxes, ha="right", va="top", fontsize=10)

    for label, ax in zip("abcd", axes):
        ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    stacked_by_radius = {radius: build_stacked_panel(radius) for radius in RADII}
    summary_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    drought_response: dict[tuple[str, str], tuple[float, float, float, float]] | None = None

    for radius in RADII:
        print(f"Estimating direct model at {radius} km", flush=True)
        direct, rows = fit_direct(stacked_by_radius[radius], radius)
        summary_rows.append(direct)
        coefficient_rows.extend(rows)

    for radius in RADII:
        print(f"Estimating drought amplification at {radius} km", flush=True)
        estimate, rows, response = fit_climate(stacked_by_radius[radius], DRY, radius)
        summary_rows.append(estimate)
        coefficient_rows.extend(rows)
        if radius == PRIMARY_RADIUS_KM:
            drought_response = response

    for hazard in [HEAT, COMPOUND]:
        print(f"Estimating {HAZARD_LABELS[hazard]} amplification at 5 km", flush=True)
        estimate, rows, _ = fit_climate(stacked_by_radius[PRIMARY_RADIUS_KM], hazard, PRIMARY_RADIUS_KM)
        summary_rows.append(estimate)
        coefficient_rows.extend(rows)

    if drought_response is None:
        raise RuntimeError("Primary drought-response slopes were not produced")
    summary = pd.DataFrame(summary_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    response = build_response_lines(stacked_by_radius[PRIMARY_RADIUS_KM], drought_response)
    summary.to_csv(SUMMARY_OUT, index=False)
    coefficients.to_csv(COEFFICIENT_OUT, index=False)
    response.to_csv(RESPONSE_OUT, index=False)
    metadata = {
        "status": "first area-level ITT climate-amplification experiment",
        "human_decision_record": "MILI-D-20260823-008",
        "primary_outcome": "Annual Land NPP Anomaly Z",
        "primary_hazard": "May-October Dry Rainfall Intensity",
        "primary_buffer_km": PRIMARY_RADIUS_KM,
        "treatment": "Candidate Affected Area ITT",
        "sector_first_years": SECTOR_YEARS,
        "weights": "sector-stratified capped calibration based on 2001-2007 outcomes and predetermined variables",
        "fixed_effects": "stack-village and stack-year",
        "predetermined_controls": CONTROL_FEATURES,
        "control_function": "standardized predetermined controls interacted with calendar-year indicators",
        "structural_hierarchy": "hazard, area x hazard, post x hazard, area x post, area x post x hazard",
        "compound_model": "retains full separate drought and heat response structures",
        "inference": "10 km spatial-block clustered covariance with debiasing",
        "interpretation": "affected-district area-level ITT; not verified village shelling or evacuation",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    METADATA_OUT.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    draw_figure(summary, response)
    print(f"Saved: {OUT.relative_to(ROOT)}")
    print(f"Saved: {SUMMARY_OUT.relative_to(ROOT)}")
    print("\nHeadline estimates")
    print(
        summary[
            ["Model", "Buffer Radius km", "Estimate", "95% CI Lower", "95% CI Upper", "p-value"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
