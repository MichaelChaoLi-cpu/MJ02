#!/usr/bin/env python3
"""Mundlak between-within NPP-welfare decomposition with spatial regimes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyhdfe
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analyses"))
import test_annual_climate_cropland_npp_model as annual_model  # noqa: E402
import test_household_confirmatory_province_precipitation_slopes as confirmation  # noqa: E402
import test_spatial_clustering_cropland_npp_household_welfare as spatial  # noqa: E402


OUTPUT = ROOT / "data/exp/analysis/climate-welfare/mundlak-between-within-spatial-regimes"
GEODETECTOR = ROOT / "data/exp/analysis/climate-welfare/geodetector-gwr-coefficient-strata"
VILLAGE_CONTEXT = GEODETECTOR / "village_geodetector_analysis_frame.csv"
STRATA_SUMMARY = GEODETECTOR / "validated_province_precipitation_strata_summary.csv"

ID = spatial.ID
OUTCOME = spatial.FOOD
CURRENT_NPP = spatial.NPP_TERM
BETWEEN_NPP = "Village long-run mean cropland NPP per 0.1 kg C per m2"
WITHIN_NPP = "Annual deviation from village long-run mean NPP per 0.1 kg C per m2"
WEIGHT = spatial.WEIGHT
BLOCK = spatial.BLOCK
TIME = spatial.TIME
COMPOUND = confirmation.COMPOUND
PROVINCE_STRATUM = "Province stratum"
PERIOD = confirmation.PERIOD
SURVEY_YEAR = confirmation.SURVEY_YEAR
PERMUTATIONS = 999
SEED = 20260824

TIME_VARYING_CONTROLS = [spatial.TEMP, spatial.RAIN, *spatial.COMPOSITION]
TIME_INVARIANT_CONTROLS = [
    "Long-run annual mean temperature",
    "Long-run annual precipitation",
    "Baseline cropland share",
    "Baseline population",
    "Historical road distance",
    "Mean elevation",
    "Mean slope",
]
OTHER_CONTROLS = [*TIME_VARYING_CONTROLS, *TIME_INVARIANT_CONTROLS]


def load_frame() -> tuple[pd.DataFrame, list[str], list[str]]:
    households, _, temporal_groups_frame = confirmation.load_frame()
    context = pd.read_csv(VILLAGE_CONTEXT, dtype={"Province Code": str})
    context = context[
        [
            ID,
            "Stratum: Province strata",
            *TIME_INVARIANT_CONTROLS,
        ]
    ].rename(columns={"Stratum: Province strata": PROVINCE_STRATUM})
    annual = annual_model.load_panel()
    value = annual_model.DEFINITIONS["Strict cropland (IGBP 12)"]["value"]
    valid = annual_model.DEFINITIONS["Strict cropland (IGBP 12)"]["valid"]
    annual = annual.loc[
        pd.to_numeric(annual[value], errors="coerce").notna()
        & pd.to_numeric(annual[valid], errors="coerce").ge(10)
    ].copy()
    village_mean = (
        annual.groupby(ID, observed=True)[value]
        .mean()
        .div(0.1)
        .rename(BETWEEN_NPP)
        .reset_index()
    )
    frame = households.merge(context, on=ID, how="left", validate="many_to_one")
    frame = frame.merge(village_mean, on=ID, how="left", validate="many_to_one")
    frame[WITHIN_NPP] = frame[CURRENT_NPP] - frame[BETWEEN_NPP]
    full_groups = sorted(frame[COMPOUND].dropna().unique())
    temporal_groups = sorted(temporal_groups_frame[COMPOUND].tolist())
    return frame, full_groups, temporal_groups


def residualize(
    sample: pd.DataFrame,
    matrix: np.ndarray,
    fixed_effects: list[str],
    weights: np.ndarray,
) -> np.ndarray:
    ids = np.column_stack(
        [pd.Categorical(sample[column]).codes for column in fixed_effects]
    )
    algorithm = pyhdfe.create(
        ids,
        drop_singletons=False,
        compute_degrees=False,
        residualize_method="map",
    )
    return algorithm.residualize(matrix, weights=weights[:, None])


def select_independent_controls(
    focal: np.ndarray,
    candidates: np.ndarray,
    names: list[str],
    weights: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str]]:
    weighted = np.sqrt(weights)[:, None] * focal
    current_rank = np.linalg.matrix_rank(weighted)
    if current_rank < focal.shape[1]:
        raise RuntimeError("Focal between-within terms are not jointly identified")
    selected_positions: list[int] = []
    for position in range(candidates.shape[1]):
        proposed = np.column_stack(
            [weighted, np.sqrt(weights) * candidates[:, position]]
        )
        proposed_rank = np.linalg.matrix_rank(proposed)
        if proposed_rank > current_rank:
            selected_positions.append(position)
            weighted = proposed
            current_rank = proposed_rank
    return (
        candidates[:, selected_positions],
        [names[position] for position in selected_positions],
        [name for position, name in enumerate(names) if position not in selected_positions],
    )


def fit_between_regimes(
    frame: pd.DataFrame,
    group_column: str,
    groups: list[str],
    fixed_effects: list[str],
    sample_label: str,
    model_label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = [
        OUTCOME,
        BETWEEN_NPP,
        WITHIN_NPP,
        WEIGHT,
        BLOCK,
        group_column,
        *fixed_effects,
        *OTHER_CONTROLS,
    ]
    sample = frame.loc[frame[group_column].isin(groups)].dropna(subset=required).copy()
    groups = [group for group in groups if sample[group_column].eq(group).any()]
    between = sample[BETWEEN_NPP].to_numpy(float)
    between_group = np.column_stack(
        [between * sample[group_column].eq(group).to_numpy(float) for group in groups]
    )
    within = sample[WITHIN_NPP].to_numpy(float)[:, None]
    focal = np.column_stack([between_group, within])
    candidate_controls = sample[OTHER_CONTROLS].to_numpy(float)
    y = sample[OUTCOME].to_numpy(float)
    weights = sample[WEIGHT].to_numpy(float)
    clusters = sample[BLOCK].astype(str).to_numpy()
    residualized = residualize(
        sample,
        np.column_stack([y, focal, candidate_controls]),
        fixed_effects,
        weights,
    )
    y_tilde = residualized[:, 0]
    focal_tilde = residualized[:, 1 : 1 + focal.shape[1]]
    candidate_tilde = residualized[:, 1 + focal.shape[1] :]
    control_tilde, retained_controls, dropped_controls = select_independent_controls(
        focal_tilde, candidate_tilde, OTHER_CONTROLS, weights
    )
    design = np.column_stack([focal_tilde, control_tilde])
    xtw = design.T * weights
    bread = np.linalg.pinv(xtw @ design)
    beta = bread @ (xtw @ y_tilde)
    residual = y_tilde - design @ beta
    covariance, _ = confirmation.cluster_covariance(
        design, residual, weights, clusters, bread
    )
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0))
    clusters_n = int(pd.Series(clusters).nunique())
    degrees = max(clusters_n - 1, 1)
    critical = float(stats.t.ppf(0.975, degrees))
    rows = []
    for position, group in enumerate(groups):
        subset = sample.loc[sample[group_column].eq(group)]
        statistic = beta[position] / standard_error[position]
        pvalue = float(2 * stats.t.sf(abs(statistic), degrees))
        rows.append(
            {
                "Model": model_label,
                "Sample": sample_label,
                "Spatial regime": group,
                "Households": int(len(subset)),
                "Villages": int(subset[ID].nunique()),
                "Between-village coefficient": float(beta[position]),
                "Clustered standard error": float(standard_error[position]),
                "95 percent CI lower": float(beta[position] - critical * standard_error[position]),
                "95 percent CI upper": float(beta[position] + critical * standard_error[position]),
                "Probability value": pvalue,
            }
        )
    estimates = pd.DataFrame(rows)
    estimates["BH-adjusted probability value"] = confirmation.bh_adjust(
        estimates["Probability value"]
    )
    estimates["FDR significant"] = estimates["BH-adjusted probability value"].lt(0.05)

    within_position = len(groups)
    within_statistic = beta[within_position] / standard_error[within_position]
    within_pvalue = float(2 * stats.t.sf(abs(within_statistic), degrees))
    within_interval = [
        float(beta[within_position] - critical * standard_error[within_position]),
        float(beta[within_position] + critical * standard_error[within_position]),
    ]
    restrictions = np.zeros((max(len(groups) - 1, 0), len(beta)))
    if len(groups) > 1:
        for row in range(len(groups) - 1):
            restrictions[row, row + 1] = 1.0
            restrictions[row, 0] = -1.0
        contrast = restrictions @ beta
        contrast_covariance = restrictions @ covariance @ restrictions.T
        inverse_contrast_covariance = np.linalg.pinv(contrast_covariance)
        observed_wald = float(contrast.T @ inverse_contrast_covariance @ contrast)

        # Restricted score bootstrap imposes one common between-village slope.
        global_between_tilde = focal_tilde[:, : len(groups)].sum(axis=1, keepdims=True)
        restricted_design = np.column_stack(
            [global_between_tilde, focal_tilde[:, [within_position]], control_tilde]
        )
        restricted_xtw = restricted_design.T * weights
        restricted_bread = np.linalg.pinv(restricted_xtw @ restricted_design)
        restricted_beta = restricted_bread @ (restricted_xtw @ y_tilde)
        restricted_residual = y_tilde - restricted_design @ restricted_beta
        _, restricted_scores = confirmation.cluster_covariance(
            design, restricted_residual, weights, clusters, bread
        )
        null_beta = np.concatenate(
            [
                np.repeat(restricted_beta[0], len(groups)),
                restricted_beta[1:],
            ]
        )
        rng = np.random.default_rng(
            SEED + sum(ord(character) for character in sample_label + model_label)
        )
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(PERMUTATIONS, restricted_scores.shape[0]),
        )
        perturbation = bread @ restricted_scores.T @ signs.T
        bootstrap_contrast = restrictions @ (null_beta[:, None] + perturbation)
        bootstrap_wald = np.einsum(
            "ib,ij,jb->b",
            bootstrap_contrast,
            inverse_contrast_covariance,
            bootstrap_contrast,
        )
        bootstrap_pvalue = float(
            (1 + np.sum(bootstrap_wald >= observed_wald)) / (PERMUTATIONS + 1)
        )
        contrast_condition = float(np.linalg.cond(contrast_covariance))
        restricted_global_between = float(restricted_beta[0])
    else:
        observed_wald = np.nan
        bootstrap_pvalue = np.nan
        contrast_condition = np.nan
        restricted_global_between = float(beta[0])

    between_position = 0
    difference = beta[between_position] - beta[within_position]
    difference_variance = (
        covariance[between_position, between_position]
        + covariance[within_position, within_position]
        - 2 * covariance[between_position, within_position]
    )
    difference_se = float(np.sqrt(max(difference_variance, 0)))
    difference_p = float(
        2 * stats.t.sf(abs(difference / difference_se), degrees)
    ) if difference_se > 0 and len(groups) == 1 else np.nan
    summary = {
        "Model": model_label,
        "Sample": sample_label,
        "Households": int(len(sample)),
        "Villages": int(sample[ID].nunique()),
        "Spatial regimes": len(groups),
        "Spatial blocks": clusters_n,
        "Restricted common between-village coefficient": restricted_global_between,
        "Global between-village coefficient": float(beta[0]) if len(groups) == 1 else np.nan,
        "Global between-village clustered standard error": (
            float(standard_error[0]) if len(groups) == 1 else np.nan
        ),
        "Global between-village 95 percent CI lower": (
            float(beta[0] - critical * standard_error[0]) if len(groups) == 1 else np.nan
        ),
        "Global between-village 95 percent CI upper": (
            float(beta[0] + critical * standard_error[0]) if len(groups) == 1 else np.nan
        ),
        "Global between-village probability value": (
            float(2 * stats.t.sf(abs(beta[0] / standard_error[0]), degrees))
            if len(groups) == 1 and standard_error[0] > 0
            else np.nan
        ),
        "Within-village annual deviation coefficient": float(beta[within_position]),
        "Within-village clustered standard error": float(standard_error[within_position]),
        "Within-village 95 percent CI lower": within_interval[0],
        "Within-village 95 percent CI upper": within_interval[1],
        "Within-village probability value": within_pvalue,
        "Global between-minus-within difference": float(difference) if len(groups) == 1 else np.nan,
        "Global between-minus-within probability value": difference_p,
        "Equal between-slope Wald statistic": observed_wald,
        "Equal between-slope wild-cluster score bootstrap probability value": bootstrap_pvalue,
        "Wild-cluster score bootstrap draws": PERMUTATIONS if len(groups) > 1 else 0,
        "Contrast covariance condition number": contrast_condition,
        "Weighted normal matrix condition number": float(np.linalg.cond(xtw @ design)),
        "Retained controls": retained_controls,
        "Dropped absorbed or collinear controls": dropped_controls,
    }
    return estimates, summary


def correlation_summary(
    compound_full: pd.DataFrame,
    compound_early: pd.DataFrame,
    compound_late: pd.DataFrame,
    gwr_summary: pd.DataFrame,
) -> dict[str, object]:
    gwr = gwr_summary[[COMPOUND, "Mean_local_coefficient"]].rename(
        columns={COMPOUND: "Spatial regime", "Mean_local_coefficient": "GWR coefficient"}
    )
    full = compound_full.merge(gwr, on="Spatial regime", how="inner", validate="one_to_one")
    pearson = stats.pearsonr(full["Between-village coefficient"], full["GWR coefficient"])
    spearman = stats.spearmanr(full["Between-village coefficient"], full["GWR coefficient"])
    temporal = compound_early[["Spatial regime", "Between-village coefficient"]].rename(
        columns={"Between-village coefficient": "Early coefficient"}
    ).merge(
        compound_late[["Spatial regime", "Between-village coefficient"]].rename(
            columns={"Between-village coefficient": "Late coefficient"}
        ),
        on="Spatial regime",
        how="inner",
        validate="one_to_one",
    )
    temporal_pearson = stats.pearsonr(temporal["Early coefficient"], temporal["Late coefficient"])
    temporal_spearman = stats.spearmanr(temporal["Early coefficient"], temporal["Late coefficient"])
    return {
        "household_between_vs_gwr_pearson_correlation": float(pearson.statistic),
        "household_between_vs_gwr_pearson_probability_value": float(pearson.pvalue),
        "household_between_vs_gwr_spearman_correlation": float(spearman.statistic),
        "household_between_vs_gwr_spearman_probability_value": float(spearman.pvalue),
        "early_late_between_slope_pearson_correlation": float(temporal_pearson.statistic),
        "early_late_between_slope_pearson_probability_value": float(temporal_pearson.pvalue),
        "early_late_between_slope_spearman_correlation": float(temporal_spearman.statistic),
        "early_late_between_slope_spearman_probability_value": float(temporal_spearman.pvalue),
        "early_late_same_sign_share": float(
            np.mean(np.sign(temporal["Early coefficient"]) == np.sign(temporal["Late coefficient"]))
        ),
        "compound_strata_compared": int(len(temporal)),
    }


def temporal_slope_summary(
    early: pd.DataFrame,
    late: pd.DataFrame,
    label: str,
) -> dict[str, object]:
    paired = early[["Spatial regime", "Between-village coefficient"]].rename(
        columns={"Between-village coefficient": "Early coefficient"}
    ).merge(
        late[["Spatial regime", "Between-village coefficient"]].rename(
            columns={"Between-village coefficient": "Late coefficient"}
        ),
        on="Spatial regime",
        how="inner",
        validate="one_to_one",
    )
    pearson = stats.pearsonr(paired["Early coefficient"], paired["Late coefficient"])
    spearman = stats.spearmanr(paired["Early coefficient"], paired["Late coefficient"])
    return {
        "spatial_partition": label,
        "groups_compared": int(len(paired)),
        "early_late_pearson_correlation": float(pearson.statistic),
        "early_late_pearson_probability_value": float(pearson.pvalue),
        "early_late_spearman_correlation": float(spearman.statistic),
        "early_late_spearman_probability_value": float(spearman.pvalue),
        "early_late_same_sign_share": float(
            np.mean(
                np.sign(paired["Early coefficient"])
                == np.sign(paired["Late coefficient"])
            )
        ),
    }


def make_figure(
    global_estimates: pd.DataFrame,
    global_summary: dict[str, object],
    province_estimates: pd.DataFrame,
    province_early: pd.DataFrame,
    province_late: pd.DataFrame,
    compound_full: pd.DataFrame,
    compound_early: pd.DataFrame,
    compound_late: pd.DataFrame,
    gwr_summary: pd.DataFrame,
    correlations: dict[str, object],
    province_temporal: dict[str, object],
    path: Path,
) -> None:
    fig = plt.figure(figsize=(12.8, 9.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.45], hspace=0.38, wspace=0.32)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    between_row = global_estimates.iloc[0]
    component_labels = ["Between villages\n(long-run mean)", "Within village\n(annual deviation)"]
    component_beta = [
        between_row["Between-village coefficient"],
        global_summary["Within-village annual deviation coefficient"],
    ]
    component_lower = [
        between_row["95 percent CI lower"],
        global_summary["Within-village 95 percent CI lower"],
    ]
    component_upper = [
        between_row["95 percent CI upper"],
        global_summary["Within-village 95 percent CI upper"],
    ]
    y = np.arange(2)
    ax_a.hlines(y, component_lower, component_upper, color="#555555", linewidth=1.4)
    ax_a.scatter(component_beta, y, color=["#a33d3d", "#376f9f"], s=40, zorder=3)
    ax_a.axvline(0, color="#444444", linewidth=0.9)
    ax_a.set_yticks(y, component_labels, fontsize=8)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Log-food coefficient per 0.1 kg C m⁻² NPP", fontsize=8)
    ax_a.tick_params(axis="x", labelsize=7)
    ax_a.grid(axis="x", color="#d3d3d3", linestyle=(0, (2, 3)), linewidth=0.5)
    ax_a.text(-0.16, 1.04, "a", transform=ax_a.transAxes, fontsize=12, fontweight="bold", va="top")
    ax_a.text(-0.07, 1.04, "Global between–within decomposition", transform=ax_a.transAxes, fontsize=9.2, fontweight="bold", va="top")

    province = province_estimates.sort_values("Between-village coefficient").reset_index(drop=True)
    province = province.merge(
        province_early[["Spatial regime", "Between-village coefficient"]].rename(
            columns={"Between-village coefficient": "Early coefficient"}
        ),
        on="Spatial regime",
        how="left",
        validate="one_to_one",
    ).merge(
        province_late[["Spatial regime", "Between-village coefficient"]].rename(
            columns={"Between-village coefficient": "Late coefficient"}
        ),
        on="Spatial regime",
        how="left",
        validate="one_to_one",
    )
    py = np.arange(len(province))
    ax_b.hlines(py, province["95 percent CI lower"], province["95 percent CI upper"], color="#777777", linewidth=0.9)
    ax_b.scatter(
        province["Between-village coefficient"], py, color="#333333", s=24,
        zorder=4, label="Full sample",
    )
    ax_b.scatter(
        province["Early coefficient"], py - 0.13, color="#3b76a4", s=18,
        marker="^", zorder=3, label="2007–2017",
    )
    ax_b.scatter(
        province["Late coefficient"], py + 0.13, color="#b35c44", s=18,
        marker="v", zorder=3, label="2019–2021",
    )
    ax_b.axvline(0, color="#444444", linewidth=0.9)
    ax_b.set_yticks(py, province["Spatial regime"], fontsize=6.7)
    ax_b.set_xlabel("Between-village coefficient", fontsize=8)
    ax_b.tick_params(axis="x", labelsize=7)
    ax_b.grid(axis="x", color="#d3d3d3", linestyle=(0, (2, 3)), linewidth=0.5)
    ax_b.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax_b.text(
        0.88,
        0.98,
        (
            f"Early–late Spearman ρ = "
            f"{province_temporal['early_late_spearman_correlation']:.2f}\n"
            f"same sign = {100 * province_temporal['early_late_same_sign_share']:.0f}%"
        ),
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
    )
    ax_b.text(-0.16, 1.04, "b", transform=ax_b.transAxes, fontsize=12, fontweight="bold", va="top")
    ax_b.text(-0.07, 1.04, "Province slopes and temporal stability", transform=ax_b.transAxes, fontsize=9.2, fontweight="bold", va="top")

    gwr = gwr_summary[[COMPOUND, "Mean_local_coefficient"]].rename(
        columns={COMPOUND: "Spatial regime", "Mean_local_coefficient": "GWR coefficient"}
    )
    full = compound_full.merge(gwr, on="Spatial regime", how="inner", validate="one_to_one")
    temporal = compound_early[["Spatial regime", "Between-village coefficient"]].rename(
        columns={"Between-village coefficient": "Early"}
    ).merge(
        compound_late[["Spatial regime", "Between-village coefficient"]].rename(
            columns={"Between-village coefficient": "Late"}
        ),
        on="Spatial regime",
        how="inner",
        validate="one_to_one",
    )
    ax_c.scatter(
        full["GWR coefficient"],
        full["Between-village coefficient"],
        color="#3b76a4",
        alpha=0.82,
        s=30,
        label=(
            f"Full household vs GWR: Spearman ρ = "
            f"{correlations['household_between_vs_gwr_spearman_correlation']:.2f}"
        ),
    )
    limits = [
        float(min(full["GWR coefficient"].min(), full["Between-village coefficient"].min())),
        float(max(full["GWR coefficient"].max(), full["Between-village coefficient"].max())),
    ]
    padding = 0.06 * (limits[1] - limits[0])
    limits = [limits[0] - padding, limits[1] + padding]
    ax_c.plot(limits, limits, color="#777777", linestyle=(0, (4, 3)), linewidth=0.9)
    ax_c.axhline(0, color="#b0b0b0", linewidth=0.7)
    ax_c.axvline(0, color="#b0b0b0", linewidth=0.7)
    ax_c.set_xlim(limits)
    ax_c.set_ylim(limits)
    ax_c.set_xlabel("GWR compound mean coefficient", fontsize=8)
    ax_c.set_ylabel("Household between-village compound coefficient", fontsize=8)
    ax_c.tick_params(labelsize=7)
    ax_c.grid(color="#d3d3d3", linestyle=(0, (2, 3)), linewidth=0.5)
    ax_c.legend(frameon=False, fontsize=8, loc="upper left")
    ax_c.text(-0.07, 1.03, "c", transform=ax_c.transAxes, fontsize=12, fontweight="bold", va="top")
    ax_c.text(0.00, 1.03, "Alignment of long-run household and GWR spatial regimes", transform=ax_c.transAxes, fontsize=9.2, fontweight="bold", va="top")
    for _, row in full.loc[
        np.abs(full["Between-village coefficient"] - full["GWR coefficient"])
        .nlargest(4)
        .index
    ].iterrows():
        ax_c.annotate(
            row["Spatial regime"].split(" | ")[0],
            (row["GWR coefficient"], row["Between-village coefficient"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.3,
        )
    for ax in [ax_a, ax_b, ax_c]:
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, full_compound_groups, temporal_compound_groups = load_frame()
    frame["National"] = "National"
    province_groups = sorted(frame[PROVINCE_STRATUM].dropna().unique())
    province_period_support = (
        frame.groupby([PROVINCE_STRATUM, PERIOD], observed=True)
        .agg(Households=(ID, "size"), Villages=(ID, "nunique"))
        .reset_index()
    )
    province_support_pivot = province_period_support.pivot(
        index=PROVINCE_STRATUM,
        columns=PERIOD,
        values=["Households", "Villages"],
    ).fillna(0)
    temporal_province_groups = sorted(
        province_support_pivot.index[
            (
                province_support_pivot["Households"]
                >= confirmation.MIN_TEMPORAL_HOUSEHOLDS
            ).all(axis=1)
            & (
                province_support_pivot["Villages"]
                >= confirmation.MIN_TEMPORAL_VILLAGES
            ).all(axis=1)
        ]
    )

    global_estimates, global_summary = fit_between_regimes(
        frame,
        "National",
        ["National"],
        [COMPOUND, TIME],
        "Full 2007-2021",
        "Global Mundlak decomposition",
    )
    province_estimates, province_summary = fit_between_regimes(
        frame,
        PROVINCE_STRATUM,
        province_groups,
        [PROVINCE_STRATUM, TIME],
        "Full 2007-2021",
        "Province-specific between slopes",
    )
    province_early, province_early_summary = fit_between_regimes(
        frame.loc[frame[PERIOD].eq("2007-2017")],
        PROVINCE_STRATUM,
        temporal_province_groups,
        [PROVINCE_STRATUM, TIME],
        "2007-2017",
        "Province-specific between slopes",
    )
    province_late, province_late_summary = fit_between_regimes(
        frame.loc[frame[PERIOD].eq("2019-2021")],
        PROVINCE_STRATUM,
        temporal_province_groups,
        [PROVINCE_STRATUM, TIME],
        "2019-2021",
        "Province-specific between slopes",
    )
    compound_full, compound_full_summary = fit_between_regimes(
        frame,
        COMPOUND,
        full_compound_groups,
        [COMPOUND, TIME],
        "Full 2007-2021",
        "Province-precipitation between slopes",
    )
    compound_early, compound_early_summary = fit_between_regimes(
        frame.loc[frame[PERIOD].eq("2007-2017")],
        COMPOUND,
        temporal_compound_groups,
        [COMPOUND, TIME],
        "2007-2017",
        "Province-precipitation between slopes",
    )
    compound_late, compound_late_summary = fit_between_regimes(
        frame.loc[frame[PERIOD].eq("2019-2021")],
        COMPOUND,
        temporal_compound_groups,
        [COMPOUND, TIME],
        "2019-2021",
        "Province-precipitation between slopes",
    )
    all_estimates = pd.concat(
        [
            global_estimates,
            province_estimates,
            province_early,
            province_late,
            compound_full,
            compound_early,
            compound_late,
        ],
        ignore_index=True,
    )
    gwr_summary = pd.read_csv(STRATA_SUMMARY)
    gwr_summary = gwr_summary.loc[
        gwr_summary["Meets minimum village support"].astype(str).str.lower().eq("true")
    ]
    correlations = correlation_summary(
        compound_full, compound_early, compound_late, gwr_summary
    )
    province_temporal = temporal_slope_summary(
        province_early, province_late, "Province strata"
    )
    assessment = {
        "global_between_coefficient_nonzero_at_0_05": bool(
            global_estimates.iloc[0]["Probability value"] < 0.05
        ),
        "global_within_coefficient_nonzero_at_0_05": bool(
            global_summary["Within-village probability value"] < 0.05
        ),
        "province_between_slope_heterogeneity_wild_probability_value": province_summary[
            "Equal between-slope wild-cluster score bootstrap probability value"
        ],
        "province_early_heterogeneity_wild_probability_value": province_early_summary[
            "Equal between-slope wild-cluster score bootstrap probability value"
        ],
        "province_late_heterogeneity_wild_probability_value": province_late_summary[
            "Equal between-slope wild-cluster score bootstrap probability value"
        ],
        "province_between_slopes_temporally_stable_at_0_05": bool(
            province_temporal["early_late_spearman_probability_value"] < 0.05
            and province_temporal["early_late_spearman_correlation"] > 0
            and province_temporal["early_late_same_sign_share"] >= 0.75
        ),
        "compound_between_slope_heterogeneity_wild_probability_value": compound_full_summary[
            "Equal between-slope wild-cluster score bootstrap probability value"
        ],
        "household_between_slopes_align_with_gwr_at_0_05": bool(
            correlations["household_between_vs_gwr_spearman_probability_value"] < 0.05
        ),
        "early_late_between_slopes_stable_at_0_05": bool(
            correlations["early_late_between_slope_spearman_probability_value"] < 0.05
            and correlations["early_late_between_slope_spearman_correlation"] > 0
        ),
    }
    summary = {
        "design": (
            "Mundlak decomposition of village long-run mean cropland NPP and annual deviations; "
            "survey weights; spatial-block clustering; stratum and survey-time fixed effects; "
            "time-varying household-climate and time-invariant geographic controls"
        ),
        "global_model": global_summary,
        "province_model": province_summary,
        "province_early_model": province_early_summary,
        "province_late_model": province_late_summary,
        "province_temporal_stability": province_temporal,
        "compound_full_model": compound_full_summary,
        "compound_early_model": compound_early_summary,
        "compound_late_model": compound_late_summary,
        "cross_design_and_temporal_stability": correlations,
        "assessment": assessment,
        "interpretation_limit": (
            "Between-village coefficients describe adjusted long-run spatial associations and "
            "remain vulnerable to unobserved place characteristics. Within-village coefficients "
            "describe annual associations, not exogenous productivity shocks."
        ),
    }
    all_estimates.to_csv(OUTPUT / "mundlak_between_regime_estimates.csv", index=False)
    province_period_support.to_csv(
        OUTPUT / "province_temporal_support.csv", index=False
    )
    pd.DataFrame(
        [
            global_summary,
            province_summary,
            province_early_summary,
            province_late_summary,
            compound_full_summary,
            compound_early_summary,
            compound_late_summary,
        ]
    ).to_csv(OUTPUT / "mundlak_model_tests.csv", index=False)
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Mundlak between-within NPP-welfare spatial regimes\n\n"
        "The experiment separates persistent between-village NPP geography from annual "
        "within-village NPP deviations. Province and province-by-precipitation slopes are "
        "also tested for early-versus-late temporal stability. All estimates remain "
        "associational.\n",
        encoding="utf-8",
    )
    make_figure(
        global_estimates,
        global_summary,
        province_estimates,
        province_early,
        province_late,
        compound_full,
        compound_early,
        compound_late,
        gwr_summary,
        correlations,
        province_temporal,
        OUTPUT / "mundlak_between_within_and_spatial_stability.png",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nProvince estimates")
    print(province_estimates.sort_values("Between-village coefficient").to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
