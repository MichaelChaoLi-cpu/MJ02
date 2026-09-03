#!/usr/bin/env python3
"""Conflict and Village NPP Dynamics.

Plan: Show sector-specific affected-area and calibrated-comparison NPP paths,
then estimate the pooled stacked event-time differences.
Framework: AnaSOP Sections 5-7 area-level ITT event study. The preferred
sector-stratified calibration is constructed only from 2001-2007 outcomes and
predetermined variables; post-conflict NPP is opened only after that design is
fixed.
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
OUT = ROOT / "data/exp/legacy-results/figures/Figure_conflict_and_village_npp_dynamics.png"
AUDIT_DIR = ROOT / "data/exp/experiments/cambodia-thailand-village-area-itt-npp"
EVENT_OUT = AUDIT_DIR / "event_study_coefficients.csv"
DIRECT_OUT = AUDIT_DIR / "direct_effect_summary.csv"
PATH_OUT = AUDIT_DIR / "sector_weighted_npp_paths.csv"
METADATA_OUT = AUDIT_DIR / "model_metadata.json"

ID = "National Village Point ID"
YEAR = "Year"
RADIUS = "Buffer Radius km"
NPP = "Village Buffer Mean Annual Land NPP Anomaly Z 2001-2020"
LON = "Point Longitude"
LAT = "Point Latitude"
TREATMENT = "Candidate Affected District"
SECTOR = "Candidate Conflict Sector"

PRIMARY_RADIUS_KM = 5
REFERENCE_EVENT_TIME = -1
EVENT_TIMES = list(range(-7, 14))
SECTOR_YEARS = {
    "Preah Vihear": 2008,
    "Ta Moan-Ta Krabey": 2011,
}
SECTOR_LABELS = {
    "Preah Vihear": "Preah Vihear sector",
    "Ta Moan-Ta Krabey": "Ta Moan–Ta Krabey sector",
}


def build_stacked_panel() -> pd.DataFrame:
    """Build two outcome-independent sector stacks with calibrated weights."""
    treated, controls = build_sample()
    design = sector_calibration(treated, controls)
    panel = pd.read_parquet(
        PANEL,
        columns=[ID, YEAR, RADIUS, NPP, LON, LAT, TREATMENT, SECTOR],
        filters=[(RADIUS, "=", PRIMARY_RADIUS_KM)],
    )
    panel = panel.dropna(subset=[ID, YEAR, NPP, LON, LAT]).copy()
    control_ids = controls[ID].astype(str).tolist()
    sector_frames: list[pd.DataFrame] = []

    for sector, first_year in SECTOR_YEARS.items():
        sector_treated = treated.loc[treated[SECTOR].eq(sector)].copy()
        sector_share = len(sector_treated) / len(treated)
        treated_weights = pd.Series(
            sector_share / len(sector_treated),
            index=sector_treated[ID].astype(str),
        )
        control_weights = pd.Series(
            np.asarray(design["sector_control_weights"][sector], dtype=float) * sector_share,
            index=controls[ID].astype(str),
        )
        weight_map = pd.concat([treated_weights, control_weights])
        selected_ids = set(sector_treated[ID].astype(str)).union(control_ids)
        stack = panel.loc[panel[ID].astype(str).isin(selected_ids)].copy()
        stack[ID] = stack[ID].astype(str)
        stack["Conflict Sector"] = sector
        stack["First Conflict Year"] = first_year
        stack["Area ITT"] = stack[ID].isin(set(sector_treated[ID].astype(str))).astype(int)
        stack["Post Conflict Period"] = stack[YEAR].ge(first_year).astype(int)
        stack["Event Time"] = stack[YEAR] - first_year
        stack["Analysis Weight"] = stack[ID].map(weight_map)
        stack["Analysis Role"] = np.where(stack["Area ITT"].eq(1), "Affected area", "Calibrated comparison")
        stack["Stack Village ID"] = sector + "__" + stack[ID]
        stack["Stack Year"] = sector + "__" + stack[YEAR].astype(str)
        stack["Spatial Block ID"] = (
            np.floor(stack[LON].to_numpy(float) * 10).astype(int).astype(str)
            + "_"
            + np.floor(stack[LAT].to_numpy(float) * 10).astype(int).astype(str)
        )
        if stack["Analysis Weight"].isna().any():
            raise RuntimeError(f"Missing analysis weights in {sector} stack")
        # The capped logistic calibration can underflow negligible control
        # weights to exact zero. PanelOLS requires strictly positive weights,
        # so those zero-mass rows are omitted without changing the estimand.
        stack = stack.loc[stack["Analysis Weight"].gt(0)].copy()
        expected_villages = len(sector_treated) + int(np.count_nonzero(control_weights.to_numpy() > 0))
        if stack[ID].nunique() != expected_villages:
            raise RuntimeError(
                f"Unexpected {sector} support: {stack[ID].nunique()} villages, expected {expected_villages}"
            )
        sector_frames.append(stack)

    stacked = pd.concat(sector_frames, ignore_index=True)
    if stacked.duplicated(["Stack Village ID", YEAR]).any():
        raise RuntimeError("Stacked panel is not unique by stack-village and year")
    return stacked


def weighted_paths(stacked: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (sector, year, role), frame in stacked.groupby(
        ["Conflict Sector", YEAR, "Analysis Role"], observed=True
    ):
        rows.append(
            {
                "Conflict Sector": sector,
                "Year": int(year),
                "Analysis Role": role,
                "Weighted Mean NPP Anomaly Z": float(np.average(frame[NPP], weights=frame["Analysis Weight"])),
                "Villages": int(frame[ID].nunique()),
                "Weight Sum": float(frame["Analysis Weight"].sum()),
            }
        )
    return pd.DataFrame(rows)


def fit_event_study(stacked: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    sample = stacked.loc[stacked["Event Time"].between(min(EVENT_TIMES), max(EVENT_TIMES))].copy()
    terms: list[str] = []
    for event_time in EVENT_TIMES:
        if event_time == REFERENCE_EVENT_TIME:
            continue
        term = f"Area ITT x event {event_time:+d}"
        sample[term] = sample["Area ITT"] * sample["Event Time"].eq(event_time).astype(int)
        terms.append(term)

    panel = sample.set_index(["Stack Village ID", YEAR]).sort_index()
    other_effects = pd.DataFrame(
        {"Stack x year": pd.Categorical(panel["Stack Year"]).codes},
        index=panel.index,
    )
    clusters = pd.DataFrame(
        {"10 km spatial block": pd.Categorical(panel["Spatial Block ID"]).codes},
        index=panel.index,
    )
    fitted = PanelOLS(
        panel[NPP].astype(float),
        panel[terms].astype(float),
        weights=panel["Analysis Weight"].astype(float),
        entity_effects=True,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)
    confidence = fitted.conf_int(level=0.95)
    rows: list[dict[str, object]] = []
    for event_time in EVENT_TIMES:
        if event_time == REFERENCE_EVENT_TIME:
            rows.append(
                {
                    "Event Time": event_time,
                    "Estimate": 0.0,
                    "Clustered Standard Error": 0.0,
                    "95% CI Lower": 0.0,
                    "95% CI Upper": 0.0,
                    "p-value": np.nan,
                    "Reference Period": True,
                }
            )
            continue
        term = f"Area ITT x event {event_time:+d}"
        rows.append(
            {
                "Event Time": event_time,
                "Estimate": float(fitted.params[term]),
                "Clustered Standard Error": float(fitted.std_errors[term]),
                "95% CI Lower": float(confidence.loc[term, "lower"]),
                "95% CI Upper": float(confidence.loc[term, "upper"]),
                "p-value": float(fitted.pvalues[term]),
                "Reference Period": False,
            }
        )

    pre_terms = [f"Area ITT x event {event_time:+d}" for event_time in range(-7, -1)]
    restriction = np.zeros((len(pre_terms), len(fitted.params)))
    for row_index, term in enumerate(pre_terms):
        restriction[row_index, fitted.params.index.get_loc(term)] = 1.0
    pre_test = fitted.wald_test(restriction)
    leave_one_lead_out: dict[str, float] = {}
    for omitted in pre_terms:
        retained = [term for term in pre_terms if term != omitted]
        retained_restriction = np.zeros((len(retained), len(fitted.params)))
        for row_index, term in enumerate(retained):
            retained_restriction[row_index, fitted.params.index.get_loc(term)] = 1.0
        leave_one_lead_out[omitted] = float(fitted.wald_test(retained_restriction).pval)
    diagnostics: dict[str, object] = {
        "observations": int(fitted.nobs),
        "stack_villages": int(sample["Stack Village ID"].nunique()),
        "unique_physical_villages": int(sample[ID].nunique()),
        "treated_villages": int(sample.loc[sample["Area ITT"].eq(1), ID].nunique()),
        "control_villages": int(sample.loc[sample["Area ITT"].eq(0), ID].nunique()),
        "spatial_blocks": int(sample["Spatial Block ID"].nunique()),
        "pretrend_wald_statistic": float(pre_test.stat),
        "pretrend_wald_df": int(len(pre_terms)),
        "pretrend_wald_p_value": float(pre_test.pval),
        "largest_leave_one_lead_out_p_value": float(max(leave_one_lead_out.values())),
        "lead_whose_omission_maximizes_p_value": str(max(leave_one_lead_out, key=leave_one_lead_out.get)),
        "within_r_squared": float(fitted.rsquared_within),
    }
    return pd.DataFrame(rows), diagnostics


def fit_linear_pretrend(stacked: pd.DataFrame) -> dict[str, float]:
    """Estimate a complementary differential linear trend in the common pre-period."""
    sample = stacked.loc[stacked["Event Time"].between(min(EVENT_TIMES), REFERENCE_EVENT_TIME)].copy()
    term = "Area ITT x linear pretrend"
    sample[term] = sample["Area ITT"] * sample["Event Time"]
    panel = sample.set_index(["Stack Village ID", YEAR]).sort_index()
    other_effects = pd.DataFrame(
        {"Stack x year": pd.Categorical(panel["Stack Year"]).codes},
        index=panel.index,
    )
    clusters = pd.DataFrame(
        {"10 km spatial block": pd.Categorical(panel["Spatial Block ID"]).codes},
        index=panel.index,
    )
    fitted = PanelOLS(
        panel[NPP].astype(float),
        panel[[term]].astype(float),
        weights=panel["Analysis Weight"].astype(float),
        entity_effects=True,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)
    confidence = fitted.conf_int(level=0.95)
    return {
        "linear_pretrend_estimate_per_year": float(fitted.params[term]),
        "linear_pretrend_standard_error": float(fitted.std_errors[term]),
        "linear_pretrend_ci_lower": float(confidence.loc[term, "lower"]),
        "linear_pretrend_ci_upper": float(confidence.loc[term, "upper"]),
        "linear_pretrend_p_value": float(fitted.pvalues[term]),
    }


def fit_direct_effect(stacked: pd.DataFrame) -> pd.DataFrame:
    sample = stacked.loc[stacked["Event Time"].between(min(EVENT_TIMES), max(EVENT_TIMES))].copy()
    term = "Area ITT x post"
    sample[term] = sample["Area ITT"] * sample["Post Conflict Period"]
    panel = sample.set_index(["Stack Village ID", YEAR]).sort_index()
    other_effects = pd.DataFrame(
        {"Stack x year": pd.Categorical(panel["Stack Year"]).codes},
        index=panel.index,
    )
    clusters = pd.DataFrame(
        {"10 km spatial block": pd.Categorical(panel["Spatial Block ID"]).codes},
        index=panel.index,
    )
    fitted = PanelOLS(
        panel[NPP].astype(float),
        panel[[term]].astype(float),
        weights=panel["Analysis Weight"].astype(float),
        entity_effects=True,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)
    confidence = fitted.conf_int(level=0.95)
    return pd.DataFrame(
        [
            {
                "Estimand": "Area-level ITT post-conflict NPP difference",
                "Estimate": float(fitted.params[term]),
                "Clustered Standard Error": float(fitted.std_errors[term]),
                "95% CI Lower": float(confidence.loc[term, "lower"]),
                "95% CI Upper": float(confidence.loc[term, "upper"]),
                "p-value": float(fitted.pvalues[term]),
                "Observations": int(fitted.nobs),
                "Stack Villages": int(sample["Stack Village ID"].nunique()),
                "Physical Villages": int(sample[ID].nunique()),
                "Spatial Blocks": int(sample["Spatial Block ID"].nunique()),
                "Fixed Effects": "stack-village; stack-year",
                "Inference": "10 km spatial-block clustered; debiased",
                "Primary Buffer km": PRIMARY_RADIUS_KM,
            }
        ]
    )


def draw_figure(paths: pd.DataFrame, event: pd.DataFrame, diagnostics: dict[str, object]) -> None:
    sns.set_theme(style="whitegrid", context="paper")
    colors = {"Affected area": "#C4493D", "Calibrated comparison": "#2F6B8A"}
    fig = plt.figure(figsize=(11.2, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.18])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]

    for ax, sector in zip(axes[:2], SECTOR_YEARS):
        subset = paths.loc[paths["Conflict Sector"].eq(sector)]
        for role in ["Affected area", "Calibrated comparison"]:
            line = subset.loc[subset["Analysis Role"].eq(role)].sort_values("Year")
            ax.plot(
                line["Year"],
                line["Weighted Mean NPP Anomaly Z"],
                color=colors[role],
                linewidth=2.0,
                label=role,
            )
        ax.axvline(SECTOR_YEARS[sector], color="#4A4A4A", linewidth=1.1, linestyle="--")
        ax.axhline(0, color="#9A9A9A", linewidth=0.7)
        ax.text(0.98, 0.95, SECTOR_LABELS[sector], transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean NPP anomaly (SD)")
        ax.set_xlim(2001, 2024)
        ax.set_xticks([2001, 2005, 2009, 2013, 2017, 2021, 2024])
        ax.grid(True, color="#E3E7EA", linewidth=0.6)

    event = event.sort_values("Event Time")
    pre = event["Event Time"].lt(0)
    axes[2].fill_between(
        event["Event Time"],
        event["95% CI Lower"],
        event["95% CI Upper"],
        color="#B8CDD8",
        alpha=0.45,
        linewidth=0,
    )
    axes[2].plot(event["Event Time"], event["Estimate"], color="#2F6B8A", linewidth=1.4)
    axes[2].scatter(
        event.loc[pre, "Event Time"],
        event.loc[pre, "Estimate"],
        color="#2F6B8A",
        s=28,
        zorder=3,
    )
    axes[2].scatter(
        event.loc[~pre, "Event Time"],
        event.loc[~pre, "Estimate"],
        color="#C4493D",
        s=28,
        zorder=3,
    )
    axes[2].scatter([REFERENCE_EVENT_TIME], [0], facecolor="white", edgecolor="#2F6B8A", s=38, zorder=4)
    axes[2].axhline(0, color="#4A4A4A", linewidth=0.9)
    axes[2].axvline(-0.5, color="#8A8A8A", linewidth=1.0, linestyle="--")
    axes[2].text(0.98, 0.95, "Stacked event study", transform=axes[2].transAxes, ha="right", va="top", fontsize=10)
    axes[2].text(
        0.02,
        0.06,
        f"Joint pre-period p = {diagnostics['pretrend_wald_p_value']:.3f}",
        transform=axes[2].transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#303030",
    )
    axes[2].set_xlabel("Years relative to sector conflict onset")
    axes[2].set_ylabel("Affected-area difference (SD)")
    axes[2].set_xticks(EVENT_TIMES)
    axes[2].grid(True, color="#E3E7EA", linewidth=0.6)

    for label, ax in zip("abc", axes):
        ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    stacked = build_stacked_panel()
    paths = weighted_paths(stacked)
    event, diagnostics = fit_event_study(stacked)
    diagnostics.update(fit_linear_pretrend(stacked))
    direct = fit_direct_effect(stacked)
    paths.to_csv(PATH_OUT, index=False)
    event.to_csv(EVENT_OUT, index=False)
    direct.to_csv(DIRECT_OUT, index=False)
    metadata = {
        "status": "first area-level ITT NPP dynamics experiment",
        "human_decision_record": "MILI-D-20260823-007",
        "treatment": "Candidate Affected Area ITT",
        "sector_first_years": SECTOR_YEARS,
        "primary_buffer_km": PRIMARY_RADIUS_KM,
        "event_window": [min(EVENT_TIMES), max(EVENT_TIMES)],
        "reference_event_time": REFERENCE_EVENT_TIME,
        "weights": "sector-stratified capped calibration based on 2001-2007 outcomes and predetermined variables",
        "fixed_effects": "stack-village and stack-year",
        "inference": "10 km spatial-block clustered covariance with debiasing",
        "interpretation": "affected-district area-level ITT; not verified village shelling or evacuation",
        "diagnostics": diagnostics,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    METADATA_OUT.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    draw_figure(paths, event, diagnostics)
    print(f"Saved: {OUT.relative_to(ROOT)}")
    print(f"Saved: {EVENT_OUT.relative_to(ROOT)}")
    print(f"Saved: {DIRECT_OUT.relative_to(ROOT)}")
    print(
        "Pre-period joint test: "
        f"Wald({diagnostics['pretrend_wald_df']})={diagnostics['pretrend_wald_statistic']:.3f}, "
        f"p={diagnostics['pretrend_wald_p_value']:.4f}"
    )
    print(
        "Linear pretrend: "
        f"{diagnostics['linear_pretrend_estimate_per_year']:.4f} SD/year "
        f"[95% CI {diagnostics['linear_pretrend_ci_lower']:.4f}, "
        f"{diagnostics['linear_pretrend_ci_upper']:.4f}], "
        f"p={diagnostics['linear_pretrend_p_value']:.4f}"
    )
    row = direct.iloc[0]
    print(
        "Direct area-level ITT: "
        f"{row['Estimate']:.4f} SD "
        f"[95% CI {row['95% CI Lower']:.4f}, {row['95% CI Upper']:.4f}], "
        f"p={row['p-value']:.4f}"
    )


if __name__ == "__main__":
    main()
