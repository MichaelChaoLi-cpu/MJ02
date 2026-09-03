#!/usr/bin/env python3
"""Distinguish localized conflict, broad spillover, and northern regional change.

This post-placebo diagnostic uses only the source-defined administrative groups
frozen in AnaSOP Sections 5-7 and the official-source ledger. It does not revise
the primary treatment after viewing NPP. Each conflict-sector stack separates
the direct district, the rest of the same humanitarian-affected province, other
northern Thai-border provinces, and the remaining eligible border pool.
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
from linearmodels.panel import PanelOLS
from scipy import stats

sys.path.insert(0, str(ROOT / "src/analyses"))
from table_predetermined_balance_and_common_support import build_sample  # noqa: E402


PANEL = (
    ROOT
    / "data/processed/cambodia_public_village_npp_conflict_panel_candidate_preprocessed.parquet"
)
GRID = ROOT / "data/processed/cambodia_national_1km_grid_preprocessed.parquet"
OUT = (
    ROOT
    / "data/exp/experiments/cambodia-thailand-village-area-itt-climate"
    / "northern-spillover-vs-region"
)
ASSIGNMENTS_OUT = OUT / "official_geography_group_assignments.csv"
COUNTS_OUT = OUT / "official_geography_group_support.csv"
COEFFICIENTS_OUT = OUT / "northern_spillover_model_coefficients.csv"
SLOPES_OUT = OUT / "group_drought_slope_levels_and_changes.csv"
CONTRASTS_OUT = OUT / "group_slope_change_pairwise_contrasts.csv"
JOINT_TESTS_OUT = OUT / "group_slope_change_joint_tests.csv"
FIGURE_OUT = OUT / "northern_spillover_vs_regional_pattern.png"
METADATA_OUT = OUT / "model_metadata.json"

ID = "National Village Point ID"
YEAR = "Year"
NPP = "Village Buffer Mean Annual Land NPP Anomaly Z 2001-2020"
DRY = "Village Buffer Mean May October Dry Rainfall Intensity"
RADIUS = "Buffer Radius km"
PRIMARY_RADIUS_KM = 5

CONTROL_FEATURES = [
    "Pre-conflict NPP mean",
    "Pre-conflict NPP trend",
    "Baseline cropland share",
    "Log baseline population",
]
GROUPS = [
    "Direct conflict district",
    "Same-province response area",
    "Adjacent northern-border provinces",
    "Other eligible border pool",
]
REFERENCE_GROUP = GROUPS[-1]
GROUP_PREFIX = {
    "Direct conflict district": "Direct",
    "Same-province response area": "SameProvince",
    "Adjacent northern-border provinces": "AdjacentNorth",
}
NORTHERN_PROVINCES = {"Preah Vihear", "Oddar Meanchey", "Banteay Meanchey"}
SECTOR_PLAN = {
    "Preah Vihear": {
        "onset": 2008,
        "direct_province": "Preah Vihear",
        "direct_district": "Choam Ksant",
        "other_direct_province": "Oddar Meanchey",
        "other_direct_district": "Banteay Ampil",
    },
    "Ta Moan-Ta Krabey": {
        "onset": 2011,
        "direct_province": "Oddar Meanchey",
        "direct_district": "Banteay Ampil",
        "other_direct_province": "Preah Vihear",
        "other_direct_district": "Choam Ksant",
    },
}


def classify_group(frame: pd.DataFrame, plan: dict[str, object]) -> pd.Series:
    direct = frame["Province Name"].eq(plan["direct_province"]) & frame[
        "District Name"
    ].eq(plan["direct_district"])
    same_province = frame["Province Name"].eq(plan["direct_province"]) & ~direct
    adjacent_north = frame["Province Name"].isin(NORTHERN_PROVINCES) & ~direct & ~same_province
    group = pd.Series(REFERENCE_GROUP, index=frame.index, dtype="string")
    group.loc[adjacent_north] = "Adjacent northern-border provinces"
    group.loc[same_province] = "Same-province response area"
    group.loc[direct] = "Direct conflict district"
    return group


def build_assignments() -> pd.DataFrame:
    treated, controls = build_sample()
    baseline = pd.concat([treated, controls], ignore_index=True).drop_duplicates(ID).copy()
    baseline[ID] = baseline[ID].astype(str)
    rows: list[pd.DataFrame] = []
    sector_mass = {
        sector: int(
            baseline["Candidate Conflict Sector"].eq(sector).sum()
        )
        for sector in SECTOR_PLAN
    }
    total_direct = sum(sector_mass.values())

    for sector, plan in SECTOR_PLAN.items():
        other_direct = baseline["Province Name"].eq(plan["other_direct_province"]) & baseline[
            "District Name"
        ].eq(plan["other_direct_district"])
        part = baseline.loc[~other_direct].copy()
        part["Conflict Sector Stack"] = sector
        part["First Conflict Year"] = int(plan["onset"])
        part["Official Geography Group"] = classify_group(part, plan)
        counts = part["Official Geography Group"].value_counts()
        missing = set(GROUPS).difference(counts.index)
        if missing:
            raise RuntimeError(f"{sector} is missing frozen groups: {sorted(missing)}")
        sector_share = sector_mass[sector] / total_direct
        part["Group-balanced Stack Weight"] = part["Official Geography Group"].map(
            {group: sector_share / (len(GROUPS) * int(counts[group])) for group in GROUPS}
        )
        rows.append(part)

    assignments = pd.concat(rows, ignore_index=True)
    if assignments.duplicated(["Conflict Sector Stack", ID]).any():
        raise RuntimeError("Assignments are not unique within conflict-sector stack")
    if not assignments["Group-balanced Stack Weight"].gt(0).all():
        raise RuntimeError("All frozen stack weights must be positive")
    return assignments


def add_spatial_blocks(assignments: pd.DataFrame) -> pd.DataFrame:
    panel_static = pd.read_parquet(
        PANEL,
        columns=[ID, YEAR, RADIUS, "National Grid Cell ID"],
        filters=[(RADIUS, "=", PRIMARY_RADIUS_KM), (YEAR, "=", 2001)],
    ).drop_duplicates(ID)
    panel_static[ID] = panel_static[ID].astype(str)
    grid = pd.read_parquet(GRID, columns=["National Grid Cell ID", "Grid Row", "Grid Column"])
    grid["Spatial Block ID"] = (
        (grid["Grid Column"].astype(int) // 10).astype(str)
        + "_"
        + (grid["Grid Row"].astype(int) // 10).astype(str)
    )
    output = assignments.merge(
        panel_static[[ID, "National Grid Cell ID"]],
        on=ID,
        how="left",
        validate="many_to_one",
    ).merge(
        grid[["National Grid Cell ID", "Spatial Block ID"]],
        on="National Grid Cell ID",
        how="left",
        validate="many_to_one",
    )
    if output["Spatial Block ID"].isna().any():
        raise RuntimeError("Missing spatial block in official-geography assignments")
    return output


def build_panel(assignments: pd.DataFrame) -> pd.DataFrame:
    panel = pd.read_parquet(
        PANEL,
        columns=[ID, YEAR, RADIUS, NPP, DRY],
        filters=[(RADIUS, "=", PRIMARY_RADIUS_KM)],
    )
    panel[ID] = panel[ID].astype(str)
    frame = assignments.merge(
        panel[[ID, YEAR, NPP, DRY]],
        on=ID,
        how="left",
        validate="many_to_many",
    )
    expected_years = list(range(2001, 2025))
    if sorted(frame[YEAR].unique().tolist()) != expected_years:
        raise RuntimeError("Diagnostic panel must cover every year from 2001 through 2024")
    if frame[[NPP, DRY, *CONTROL_FEATURES]].isna().any().any():
        raise RuntimeError("Diagnostic panel contains missing model fields")

    frame["Panel Village ID"] = frame["Conflict Sector Stack"] + "__" + frame[ID]
    frame["Sector Year"] = frame["Conflict Sector Stack"] + "__" + frame[YEAR].astype(str)
    frame["Post"] = frame[YEAR].ge(frame["First Conflict Year"]).astype(float)
    frame["Post x drought"] = frame["Post"] * frame[DRY]
    frame["Linear Year"] = (frame[YEAR] - frame[YEAR].mean()) / frame[YEAR].std(ddof=0)
    for variable in CONTROL_FEATURES:
        mean = frame.drop_duplicates("Panel Village ID")[variable].mean()
        sd = frame.drop_duplicates("Panel Village ID")[variable].std(ddof=0)
        frame[f"{variable} standardized"] = (frame[variable] - mean) / sd
        frame[f"{variable} x linear year"] = (
            frame[f"{variable} standardized"] * frame["Linear Year"]
        )
    for group, prefix in GROUP_PREFIX.items():
        indicator = frame["Official Geography Group"].eq(group).astype(float)
        frame[f"{prefix} x drought"] = indicator * frame[DRY]
        frame[f"{prefix} x post"] = indicator * frame["Post"]
        frame[f"{prefix} x post x drought"] = indicator * frame["Post"] * frame[DRY]
    return frame


def linear_combination(
    result: object,
    coefficients: dict[str, float],
) -> dict[str, float | bool]:
    vector = pd.Series(0.0, index=result.params.index)
    for term, value in coefficients.items():
        vector.loc[term] = value
    estimate = float(vector @ result.params)
    variance = float(vector.to_numpy() @ result.cov.to_numpy() @ vector.to_numpy())
    standard_error = float(np.sqrt(max(variance, 0.0)))
    statistic = estimate / standard_error if standard_error > 0 else np.nan
    p_value = float(2 * stats.norm.sf(abs(statistic))) if np.isfinite(statistic) else np.nan
    return {
        "Estimate": estimate,
        "Clustered Standard Error": standard_error,
        "95% CI Lower": estimate - 1.96 * standard_error,
        "95% CI Upper": estimate + 1.96 * standard_error,
        "p-value": p_value,
        "Significant at 5%": bool(p_value < 0.05) if np.isfinite(p_value) else False,
    }


def fit_specification(frame: pd.DataFrame, specification: str) -> tuple[object, pd.DataFrame]:
    regressors = [
        DRY,
        "Post x drought",
        *[f"{prefix} x drought" for prefix in GROUP_PREFIX.values()],
        *[f"{prefix} x post" for prefix in GROUP_PREFIX.values()],
        *[f"{prefix} x post x drought" for prefix in GROUP_PREFIX.values()],
        *[f"{variable} x linear year" for variable in CONTROL_FEATURES],
    ]
    panel = frame.set_index(["Panel Village ID", YEAR]).sort_index()
    other_effects = pd.DataFrame(
        {"Sector Year": pd.Categorical(panel["Sector Year"]).codes},
        index=panel.index,
    )
    clusters = pd.DataFrame(
        {"10 km spatial block": pd.Categorical(panel["Spatial Block ID"]).codes},
        index=panel.index,
    )
    model = PanelOLS(
        panel[NPP].astype(float),
        panel[regressors].astype(float),
        weights=panel["Group-balanced Stack Weight"].astype(float),
        entity_effects=True,
        other_effects=other_effects,
        drop_absorbed=True,
        check_rank=True,
    )
    fitted = model.fit(cov_type="clustered", clusters=clusters, debiased=True)
    confidence = fitted.conf_int(level=0.95)
    coefficients = pd.DataFrame(
        {
            "Specification": specification,
            "Term": fitted.params.index,
            "Estimate": fitted.params.to_numpy(float),
            "Clustered Standard Error": fitted.std_errors.to_numpy(float),
            "95% CI Lower": confidence["lower"].to_numpy(float),
            "95% CI Upper": confidence["upper"].to_numpy(float),
            "p-value": fitted.pvalues.to_numpy(float),
            "Observations": int(fitted.nobs),
            "Panel Villages": int(frame["Panel Village ID"].nunique()),
            "Original Villages": int(frame[ID].nunique()),
            "Spatial Blocks": int(frame["Spatial Block ID"].nunique()),
        }
    )
    return fitted, coefficients


def summarize_slopes(result: object, specification: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in GROUPS:
        pre_terms = {DRY: 1.0}
        change_terms = {"Post x drought": 1.0}
        if group in GROUP_PREFIX:
            prefix = GROUP_PREFIX[group]
            pre_terms[f"{prefix} x drought"] = 1.0
            change_terms[f"{prefix} x post x drought"] = 1.0
        post_terms = pre_terms | {}
        for term, value in change_terms.items():
            post_terms[term] = post_terms.get(term, 0.0) + value
        for estimand, terms in [
            ("Pre-period drought slope", pre_terms),
            ("Post-period drought slope", post_terms),
            ("Post-minus-pre drought-slope change", change_terms),
        ]:
            row = linear_combination(result, terms)
            row.update(
                {
                    "Specification": specification,
                    "Official Geography Group": group,
                    "Estimand": estimand,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_contrasts(result: object, specification: str) -> pd.DataFrame:
    change_term = {
        REFERENCE_GROUP: None,
        **{
            group: f"{prefix} x post x drought"
            for group, prefix in GROUP_PREFIX.items()
        },
    }
    pairs = [
        (GROUPS[0], GROUPS[1]),
        (GROUPS[0], GROUPS[2]),
        (GROUPS[0], GROUPS[3]),
        (GROUPS[1], GROUPS[2]),
        (GROUPS[1], GROUPS[3]),
        (GROUPS[2], GROUPS[3]),
    ]
    rows: list[dict[str, object]] = []
    for left, right in pairs:
        terms: dict[str, float] = {}
        if change_term[left] is not None:
            terms[str(change_term[left])] = 1.0
        if change_term[right] is not None:
            terms[str(change_term[right])] = terms.get(str(change_term[right]), 0.0) - 1.0
        row = linear_combination(result, terms)
        row.update(
            {
                "Specification": specification,
                "Left Group": left,
                "Right Group": right,
                "Contrast": f"{left} minus {right}",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_joint_tests(result: object, specification: str) -> pd.DataFrame:
    parameters = result.params.index
    direct = parameters.get_loc("Direct x post x drought")
    same = parameters.get_loc("SameProvince x post x drought")
    adjacent = parameters.get_loc("AdjacentNorth x post x drought")

    equality = np.zeros((2, len(parameters)))
    equality[0, direct] = 1
    equality[0, same] = -1
    equality[1, direct] = 1
    equality[1, adjacent] = -1
    all_reference = np.zeros((3, len(parameters)))
    all_reference[0, direct] = 1
    all_reference[1, same] = 1
    all_reference[2, adjacent] = 1
    rows = []
    for label, restriction in [
        ("Equal slope changes across the three northern groups", equality),
        ("All three northern-group changes equal the other-pool change", all_reference),
    ]:
        test = result.wald_test(restriction)
        rows.append(
            {
                "Specification": specification,
                "Joint Test": label,
                "Statistic": float(np.asarray(test.stat).squeeze()),
                "Degrees of Freedom": int(restriction.shape[0]),
                "p-value": float(test.pval),
                "Rejects at 5%": bool(test.pval < 0.05),
            }
        )
    return pd.DataFrame(rows)


def make_figure(slopes: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    group_colors = {
        GROUPS[0]: "#B33A3A",
        GROUPS[1]: "#D1843B",
        GROUPS[2]: "#4C78A8",
        GROUPS[3]: "#6F767B",
    }
    pooled_slopes = slopes.loc[
        slopes["Specification"].eq("Pooled sectors")
        & slopes["Estimand"].eq("Post-minus-pre drought-slope change")
    ].copy()
    pooled_slopes["Official Geography Group"] = pd.Categorical(
        pooled_slopes["Official Geography Group"], categories=GROUPS, ordered=True
    )
    pooled_slopes = pooled_slopes.sort_values("Official Geography Group")
    selected_contrasts = contrasts.loc[
        contrasts["Specification"].eq("Pooled sectors")
        & contrasts["Right Group"].eq(REFERENCE_GROUP)
    ].copy()
    selected_contrasts["Left Group"] = pd.Categorical(
        selected_contrasts["Left Group"], categories=GROUPS[:-1], ordered=True
    )
    selected_contrasts = selected_contrasts.sort_values("Left Group")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    y = np.arange(len(pooled_slopes))[::-1]
    for ypos, (_, row) in zip(y, pooled_slopes.iterrows(), strict=True):
        color = group_colors[str(row["Official Geography Group"])]
        axes[0].errorbar(
            row["Estimate"],
            ypos,
            xerr=[[row["Estimate"] - row["95% CI Lower"]], [row["95% CI Upper"] - row["Estimate"]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
        )
    axes[0].axvline(0, color="#555555", linewidth=0.8)
    axes[0].set_yticks(y, pooled_slopes["Official Geography Group"])
    axes[0].set_xlabel("Post-minus-pre drought-slope change (NPP SD)")
    axes[0].set_title("a  Change within each official-geography group", loc="left", fontweight="bold")

    y2 = np.arange(len(selected_contrasts))[::-1]
    labels = [
        value.replace(" provinces", "\nprovinces")
        for value in selected_contrasts["Left Group"].astype(str)
    ]
    for ypos, (_, row) in zip(y2, selected_contrasts.iterrows(), strict=True):
        color = group_colors[str(row["Left Group"])]
        axes[1].errorbar(
            row["Estimate"],
            ypos,
            xerr=[[row["Estimate"] - row["95% CI Lower"]], [row["95% CI Upper"] - row["Estimate"]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            markersize=6,
        )
    axes[1].axvline(0, color="#555555", linewidth=0.8)
    axes[1].set_yticks(y2, labels)
    axes[1].set_xlabel("Slope-change difference versus other border pool")
    axes[1].set_title("b  Northern contrasts against the reference", loc="left", fontweight="bold")

    for axis in axes:
        axis.grid(axis="x", color="#E1E5E8", linewidth=0.7)
        axis.grid(False, axis="y")
        for spine in axis.spines.values():
            spine.set_color("#AEB5BA")
            spine.set_linewidth(0.8)
    fig.savefig(FIGURE_OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assignments = add_spatial_blocks(build_assignments())
    assignments[
        [
            "Conflict Sector Stack",
            "First Conflict Year",
            ID,
            "Public Village Name",
            "Province Name",
            "District Name",
            "Official Geography Group",
            "Group-balanced Stack Weight",
            "Point Longitude",
            "Point Latitude",
            "Spatial Block ID",
        ]
    ].to_csv(ASSIGNMENTS_OUT, index=False)
    support = (
        assignments.groupby(["Conflict Sector Stack", "Official Geography Group"], as_index=False)
        .agg(
            Villages=(ID, "nunique"),
            Provinces=("Province Name", "nunique"),
            Districts=("District Name", "nunique"),
            **{
                "10 km Spatial Blocks": ("Spatial Block ID", "nunique"),
                "Total Analysis Weight": ("Group-balanced Stack Weight", "sum"),
                "Mean Latitude": ("Point Latitude", "mean"),
                "Mean Border Distance km": ("Border distance", "mean"),
            },
        )
    )
    support.to_csv(COUNTS_OUT, index=False)
    frame = build_panel(assignments)

    coefficient_frames = []
    slope_frames = []
    contrast_frames = []
    joint_frames = []
    specifications = [("Pooled sectors", frame)] + [
        (sector, frame.loc[frame["Conflict Sector Stack"].eq(sector)].copy())
        for sector in SECTOR_PLAN
    ]
    for name, sample in specifications:
        fitted, coefficients = fit_specification(sample, name)
        coefficient_frames.append(coefficients)
        slope_frames.append(summarize_slopes(fitted, name))
        contrast_frames.append(summarize_contrasts(fitted, name))
        joint_frames.append(summarize_joint_tests(fitted, name))

    coefficient_output = pd.concat(coefficient_frames, ignore_index=True)
    slope_output = pd.concat(slope_frames, ignore_index=True)
    contrast_output = pd.concat(contrast_frames, ignore_index=True)
    joint_output = pd.concat(joint_frames, ignore_index=True)
    coefficient_output.to_csv(COEFFICIENTS_OUT, index=False)
    slope_output.to_csv(SLOPES_OUT, index=False)
    contrast_output.to_csv(CONTRASTS_OUT, index=False)
    joint_output.to_csv(JOINT_TESTS_OUT, index=False)
    make_figure(slope_output, contrast_output)

    metadata = {
        "status": "completed post-placebo northern spillover versus regional-change diagnostic",
        "human_decision_record": "MILI-D-20260823-015",
        "official_source_ledger": str(
            (
                ROOT
                / "data/exp/experiment-design/cambodia-thailand-northern-spillover"
                / "official_conflict_geography_source_ledger.csv"
            ).relative_to(ROOT)
        ),
        "primary_buffer_km": PRIMARY_RADIUS_KM,
        "outcome": NPP,
        "hazard": DRY,
        "groups": GROUPS,
        "sector_plan": SECTOR_PLAN,
        "comparison_universe": "existing eligible villages within 200 km of Cambodia-Thailand border",
        "weights": "equal group mass within each stack; stack mass follows 25/88 and 63/88 direct-sector shares",
        "fixed_effects": "stack-village and stack-year",
        "controls": "four predetermined standardized covariates interacted with a linear year trend",
        "inference": "10 km spatial-block clustered covariance with debiasing",
        "interpretation": (
            "post-placebo diagnostic only; same-province group is possible spillover or humanitarian-response geography, "
            "not verified direct exposure; no result can retroactively pass the original spatial-specificity gate"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    METADATA_OUT.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    pooled_changes = slope_output.loc[
        slope_output["Specification"].eq("Pooled sectors")
        & slope_output["Estimand"].eq("Post-minus-pre drought-slope change")
    ]
    pooled_contrasts = contrast_output.loc[
        contrast_output["Specification"].eq("Pooled sectors")
    ]
    print("\nFrozen official-geography support")
    print(support.to_string(index=False))
    print("\nPooled group-specific drought-slope changes")
    print(pooled_changes.to_string(index=False))
    print("\nPooled pairwise slope-change contrasts")
    print(pooled_contrasts.to_string(index=False))
    print("\nJoint tests")
    print(joint_output.to_string(index=False))
    print(f"\nSaved figure: {FIGURE_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
