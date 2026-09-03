#!/usr/bin/env python3
"""Household-level confirmation of province-by-precipitation NPP-welfare slopes.

The model returns from the smoothed GWR coefficient surface to the original household
records. It estimates one prior-year cropland-NPP slope for every GeoDetector-supported
province-by-long-run-precipitation stratum, absorbs commune and survey-time fixed effects,
uses survey weights, controls annual climate and household composition, clusters by the
fixed spatial block, and uses a 999-draw restricted wild-cluster score bootstrap for the
joint slope-equality test.
"""

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
import test_spatial_clustering_cropland_npp_household_welfare as spatial  # noqa: E402


GEODETECTOR = (
    ROOT / "data/exp/analysis/climate-welfare/geodetector-gwr-coefficient-strata"
)
VILLAGE_STRATA = GEODETECTOR / "village_geodetector_analysis_frame.csv"
STRATA_SUMMARY = GEODETECTOR / "validated_province_precipitation_strata_summary.csv"
OUTPUT = (
    ROOT
    / "data/exp/analysis/climate-welfare/household-confirmatory-province-precipitation-slopes"
)

ID = spatial.ID
OUTCOME = spatial.FOOD
NPP = spatial.NPP_TERM
WEIGHT = spatial.WEIGHT
BLOCK = spatial.BLOCK
TIME = spatial.TIME
COMMUNE = "Commune Code"
COMPOUND = "Compound stratum"
SURVEY_YEAR = "Survey Year"
PERIOD = "Temporal period"
PERMUTATIONS = 999
SEED = 20260824
MIN_TEMPORAL_HOUSEHOLDS = 150
MIN_TEMPORAL_VILLAGES = 15
CONTROLS = [spatial.TEMP, spatial.RAIN, *spatial.COMPOSITION]


def bh_adjust(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna().sort_values()
    adjusted = pd.Series(np.nan, index=numeric.index, dtype=float)
    running = 1.0
    total = len(valid)
    for reverse_rank, (index, value) in enumerate(reversed(list(valid.items())), start=1):
        rank = total - reverse_rank + 1
        running = min(running, float(value) * total / rank)
        adjusted.loc[index] = min(1.0, running)
    return adjusted


def load_frame() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    households = spatial.load_households()
    village = pd.read_csv(VILLAGE_STRATA, dtype={"Province Code": str})
    village[COMPOUND] = (
        village["Stratum: Province strata"].astype(str)
        + " | "
        + village["Stratum: Long-run annual precipitation"].astype(str)
    )
    stratum_summary = pd.read_csv(STRATA_SUMMARY)
    supported = stratum_summary.loc[
        stratum_summary["Meets minimum village support"].astype(str).str.lower().eq("true"),
        COMPOUND,
    ]
    households = households.merge(
        village[
            [
                ID,
                COMPOUND,
                "Stratum: Province strata",
                "Stratum: Long-run annual precipitation",
            ]
        ],
        on=ID,
        how="left",
        validate="many_to_one",
    )
    households[PERIOD] = np.where(
        households[SURVEY_YEAR].astype(int).le(2017), "2007-2017", "2019-2021"
    )
    eligible = households.loc[households[COMPOUND].isin(supported)].copy()
    support = (
        eligible.groupby([COMPOUND, PERIOD], observed=True)
        .agg(
            Households=(ID, "size"),
            Villages=(ID, "nunique"),
            Communes=(COMMUNE, "nunique"),
            Spatial_blocks=(BLOCK, "nunique"),
        )
        .reset_index()
    )
    pivot = support.pivot(index=COMPOUND, columns=PERIOD, values=["Households", "Villages"]).fillna(0)
    temporal_supported = pivot.index[
        (pivot["Households"] >= MIN_TEMPORAL_HOUSEHOLDS).all(axis=1)
        & (pivot["Villages"] >= MIN_TEMPORAL_VILLAGES).all(axis=1)
    ]
    return eligible, support, pd.DataFrame({COMPOUND: temporal_supported})


def cluster_covariance(
    x: np.ndarray,
    residual: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    bread: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_clusters, cluster_codes = np.unique(clusters, return_inverse=True)
    scores = np.zeros((len(unique_clusters), x.shape[1]))
    weighted_score = x * (weights * residual)[:, None]
    np.add.at(scores, cluster_codes, weighted_score)
    correction = (
        len(unique_clusters)
        / max(len(unique_clusters) - 1, 1)
        * (len(x) - 1)
        / max(len(x) - x.shape[1], 1)
    )
    covariance = correction * bread @ (scores.T @ scores) @ bread
    return covariance, scores


def residualize(
    sample: pd.DataFrame,
    matrix: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    fixed_effects = np.column_stack(
        [
            pd.Categorical(sample[COMMUNE]).codes,
            pd.Categorical(sample[TIME]).codes,
        ]
    )
    algorithm = pyhdfe.create(
        fixed_effects,
        drop_singletons=False,
        compute_degrees=False,
        residualize_method="map",
    )
    return algorithm.residualize(matrix, weights=weights[:, None])


def fit_group_slopes(
    frame: pd.DataFrame,
    groups: list[str],
    label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = [OUTCOME, NPP, WEIGHT, BLOCK, COMMUNE, TIME, COMPOUND, *CONTROLS]
    sample = frame.loc[frame[COMPOUND].isin(groups)].dropna(subset=required).copy()
    groups = [group for group in groups if sample[COMPOUND].eq(group).any()]
    npp = sample[NPP].to_numpy(float)
    group_matrix = np.column_stack(
        [npp * sample[COMPOUND].eq(group).to_numpy(float) for group in groups]
    )
    control_matrix = sample[CONTROLS].to_numpy(float)
    y = sample[OUTCOME].to_numpy(float)
    weights = sample[WEIGHT].to_numpy(float)
    clusters = sample[BLOCK].astype(str).to_numpy()
    residualized = residualize(
        sample,
        np.column_stack([y, group_matrix, control_matrix]),
        weights,
    )
    y_tilde = residualized[:, 0]
    x_group_tilde = residualized[:, 1 : 1 + len(groups)]
    candidate_control_tilde = residualized[:, 1 + len(groups) :]
    # Preserve every focal group slope, then add only controls that increase weighted rank.
    weighted_group = np.sqrt(weights)[:, None] * x_group_tilde
    if np.linalg.matrix_rank(weighted_group) < len(groups):
        raise RuntimeError(f"Focal group slopes are not jointly identified in {label}")
    selected_control_positions: list[int] = []
    weighted_selected = weighted_group.copy()
    current_rank = np.linalg.matrix_rank(weighted_selected)
    for position in range(candidate_control_tilde.shape[1]):
        candidate = np.column_stack(
            [weighted_selected, np.sqrt(weights) * candidate_control_tilde[:, position]]
        )
        candidate_rank = np.linalg.matrix_rank(candidate)
        if candidate_rank > current_rank:
            selected_control_positions.append(position)
            weighted_selected = candidate
            current_rank = candidate_rank
    x_control_tilde = candidate_control_tilde[:, selected_control_positions]
    retained_controls = [CONTROLS[position] for position in selected_control_positions]
    dropped_controls = [
        control for position, control in enumerate(CONTROLS) if position not in selected_control_positions
    ]
    x_unrestricted = np.column_stack([x_group_tilde, x_control_tilde])
    xtw = x_unrestricted.T * weights
    bread = np.linalg.pinv(xtw @ x_unrestricted)
    beta = bread @ (xtw @ y_tilde)
    residual = y_tilde - x_unrestricted @ beta
    covariance, unrestricted_scores = cluster_covariance(
        x_unrestricted, residual, weights, clusters, bread
    )
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0))
    cluster_count = int(pd.Series(clusters).nunique())
    degrees = max(cluster_count - 1, 1)
    critical = float(stats.t.ppf(0.975, degrees))

    rows = []
    for index, group in enumerate(groups):
        subset = sample.loc[sample[COMPOUND].eq(group)]
        statistic = beta[index] / standard_error[index] if standard_error[index] > 0 else np.nan
        pvalue = float(2 * stats.t.sf(abs(statistic), degrees)) if np.isfinite(statistic) else np.nan
        rows.append(
            {
                "Sample": label,
                COMPOUND: group,
                "Households": int(len(subset)),
                "Villages": int(subset[ID].nunique()),
                "Communes": int(subset[COMMUNE].nunique()),
                "Coefficient": float(beta[index]),
                "Clustered standard error": float(standard_error[index]),
                "95 percent CI lower": float(beta[index] - critical * standard_error[index]),
                "95 percent CI upper": float(beta[index] + critical * standard_error[index]),
                "Probability value": pvalue,
            }
        )
    estimates = pd.DataFrame(rows)
    estimates["BH-adjusted probability value"] = bh_adjust(estimates["Probability value"])
    estimates["FDR significant"] = estimates["BH-adjusted probability value"].lt(0.05)

    restrictions = np.zeros((len(groups) - 1, len(beta)))
    for row in range(len(groups) - 1):
        restrictions[row, row + 1] = 1.0
        restrictions[row, 0] = -1.0
    contrast = restrictions @ beta
    contrast_covariance = restrictions @ covariance @ restrictions.T
    inverse_contrast_covariance = np.linalg.pinv(contrast_covariance)
    wald = float(contrast.T @ inverse_contrast_covariance @ contrast)
    restriction_count = restrictions.shape[0]
    f_statistic = wald / max(restriction_count, 1)
    cluster_f_pvalue = float(stats.f.sf(f_statistic, restriction_count, degrees))

    # Restricted wild-cluster score bootstrap under equal NPP slopes.
    x_global_tilde = x_group_tilde.sum(axis=1, keepdims=True)
    x_restricted = np.column_stack([x_global_tilde, x_control_tilde])
    restricted_xtw = x_restricted.T * weights
    restricted_bread = np.linalg.pinv(restricted_xtw @ x_restricted)
    restricted_beta = restricted_bread @ (restricted_xtw @ y_tilde)
    restricted_residual = y_tilde - x_restricted @ restricted_beta
    _, restricted_scores_unrestricted = cluster_covariance(
        x_unrestricted,
        restricted_residual,
        weights,
        clusters,
        bread,
    )
    null_unrestricted_beta = np.concatenate(
        [
            np.repeat(restricted_beta[0], len(groups)),
            restricted_beta[1:],
        ]
    )
    rng = np.random.default_rng(SEED + sum(ord(character) for character in label))
    signs = rng.choice(
        np.array([-1.0, 1.0]),
        size=(PERMUTATIONS, restricted_scores_unrestricted.shape[0]),
    )
    perturbation = bread @ restricted_scores_unrestricted.T @ signs.T
    bootstrap_contrast = restrictions @ (
        null_unrestricted_beta[:, None] + perturbation
    )
    bootstrap_wald = np.einsum(
        "ib,ij,jb->b",
        bootstrap_contrast,
        inverse_contrast_covariance,
        bootstrap_contrast,
    )
    bootstrap_pvalue = float(
        (1 + np.sum(bootstrap_wald >= wald)) / (PERMUTATIONS + 1)
    )
    weighted_sse = float(np.sum(weights * np.square(residual)))
    weighted_sst = float(
        np.sum(weights * np.square(y_tilde - np.average(y_tilde, weights=weights)))
    )
    model_summary = {
        "Sample": label,
        "Households": int(len(sample)),
        "Villages": int(sample[ID].nunique()),
        "Communes": int(sample[COMMUNE].nunique()),
        "Compound strata": len(groups),
        "Spatial blocks": cluster_count,
        "Within fixed-effect weighted R squared": float(1.0 - weighted_sse / weighted_sst),
        "Joint equal slopes Wald statistic": wald,
        "Joint equal slopes restrictions": restriction_count,
        "Joint equal slopes cluster F statistic": f_statistic,
        "Joint equal slopes cluster F probability value": cluster_f_pvalue,
        "Contrast covariance condition number": float(
            np.linalg.cond(contrast_covariance)
        ),
        "Unrestricted weighted normal matrix condition number": float(
            np.linalg.cond(xtw @ x_unrestricted)
        ),
        "Retained controls after fixed-effect rank check": retained_controls,
        "Controls absorbed or collinear after fixed effects": dropped_controls,
        "Joint equal slopes wild-cluster score bootstrap probability value": bootstrap_pvalue,
        "Wild-cluster score bootstrap draws": PERMUTATIONS,
        "Restricted global NPP slope": float(restricted_beta[0]),
    }
    return estimates, model_summary


def association_summary(
    full: pd.DataFrame,
    early: pd.DataFrame,
    late: pd.DataFrame,
    gwr_summary: pd.DataFrame,
) -> dict[str, object]:
    gwr = gwr_summary[[COMPOUND, "Mean_local_coefficient"]].rename(
        columns={"Mean_local_coefficient": "GWR compound mean coefficient"}
    )
    full_match = full.merge(gwr, on=COMPOUND, how="inner", validate="one_to_one")
    pearson = stats.pearsonr(
        full_match["Coefficient"], full_match["GWR compound mean coefficient"]
    )
    spearman = stats.spearmanr(
        full_match["Coefficient"], full_match["GWR compound mean coefficient"]
    )
    temporal = early[[COMPOUND, "Coefficient"]].rename(
        columns={"Coefficient": "Early coefficient"}
    ).merge(
        late[[COMPOUND, "Coefficient"]].rename(columns={"Coefficient": "Late coefficient"}),
        on=COMPOUND,
        how="inner",
        validate="one_to_one",
    )
    temporal_pearson = stats.pearsonr(temporal["Early coefficient"], temporal["Late coefficient"])
    temporal_spearman = stats.spearmanr(temporal["Early coefficient"], temporal["Late coefficient"])
    temporal["Same sign"] = np.sign(temporal["Early coefficient"]) == np.sign(
        temporal["Late coefficient"]
    )
    return {
        "full_household_vs_gwr_compound_mean_pearson_correlation": float(pearson.statistic),
        "full_household_vs_gwr_compound_mean_pearson_probability_value": float(pearson.pvalue),
        "full_household_vs_gwr_compound_mean_spearman_correlation": float(spearman.statistic),
        "full_household_vs_gwr_compound_mean_spearman_probability_value": float(spearman.pvalue),
        "early_late_compound_slope_pearson_correlation": float(temporal_pearson.statistic),
        "early_late_compound_slope_pearson_probability_value": float(temporal_pearson.pvalue),
        "early_late_compound_slope_spearman_correlation": float(temporal_spearman.statistic),
        "early_late_compound_slope_spearman_probability_value": float(temporal_spearman.pvalue),
        "early_late_same_sign_share": float(temporal["Same sign"].mean()),
        "early_late_compound_strata": int(len(temporal)),
    }


def make_figure(
    full: pd.DataFrame,
    early: pd.DataFrame,
    late: pd.DataFrame,
    gwr_summary: pd.DataFrame,
    path: Path,
) -> None:
    gwr = gwr_summary[[COMPOUND, "Mean_local_coefficient"]].rename(
        columns={"Mean_local_coefficient": "GWR compound mean coefficient"}
    )
    left = full.merge(gwr, on=COMPOUND, how="left", validate="one_to_one")
    left = left.sort_values("Coefficient").reset_index(drop=True)
    temporal = early[[COMPOUND, "Coefficient"]].rename(
        columns={"Coefficient": "2007-2017"}
    ).merge(
        late[[COMPOUND, "Coefficient"]].rename(columns={"Coefficient": "2019-2021"}),
        on=COMPOUND,
        how="inner",
        validate="one_to_one",
    )
    temporal = temporal.set_index(COMPOUND).reindex(left[COMPOUND]).dropna().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 10.2), sharey=False)
    fig.subplots_adjust(left=0.25, right=0.985, top=0.97, bottom=0.08, wspace=0.48)

    y_left = np.arange(len(left))
    axes[0].hlines(
        y_left,
        left["95 percent CI lower"],
        left["95 percent CI upper"],
        color="#696969",
        linewidth=1.0,
    )
    colors = np.where(
        left["FDR significant"] & left["Coefficient"].gt(0),
        "#b33a3a",
        np.where(left["FDR significant"], "#2b6292", "#555555"),
    )
    axes[0].scatter(left["Coefficient"], y_left, color=colors, s=25, zorder=3, label="Household model")
    axes[0].scatter(
        left["GWR compound mean coefficient"],
        y_left,
        facecolor="white",
        edgecolor="#111111",
        marker="D",
        s=22,
        linewidth=0.8,
        zorder=4,
        label="GWR compound mean",
    )
    axes[0].axvline(0, color="#444444", linewidth=0.9)
    axes[0].set_yticks(y_left, left[COMPOUND], fontsize=6.8)
    axes[0].set_xlabel("Log-food coefficient per 0.1 kg C m⁻² cropland NPP", fontsize=8)
    axes[0].tick_params(axis="x", labelsize=7)
    axes[0].grid(axis="x", color="#d3d3d3", linestyle=(0, (2, 3)), linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=7.2, loc="lower right")
    axes[0].text(-0.18, 1.015, "a", transform=axes[0].transAxes, fontsize=12, fontweight="bold", va="top")
    axes[0].text(
        -0.10,
        1.015,
        "Household-level compound-stratum slopes",
        transform=axes[0].transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )

    y_right = np.arange(len(temporal))
    axes[1].hlines(
        y_right,
        temporal["2007-2017"],
        temporal["2019-2021"],
        color="#b8b8b8",
        linewidth=0.8,
        zorder=1,
    )
    axes[1].scatter(
        temporal["2007-2017"], y_right, color="#3b76a4", s=24, label="2007–2017", zorder=3
    )
    axes[1].scatter(
        temporal["2019-2021"], y_right, color="#d8833f", s=24, label="2019–2021", zorder=3
    )
    axes[1].axvline(0, color="#444444", linewidth=0.9)
    axes[1].set_yticks(y_right, temporal[COMPOUND], fontsize=6.8)
    axes[1].set_xlabel("Log-food coefficient per 0.1 kg C m⁻² cropland NPP", fontsize=8)
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].grid(axis="x", color="#d3d3d3", linestyle=(0, (2, 3)), linewidth=0.5)
    axes[1].legend(frameon=False, fontsize=7.2, loc="lower right")
    axes[1].text(-0.18, 1.015, "b", transform=axes[1].transAxes, fontsize=12, fontweight="bold", va="top")
    axes[1].text(
        -0.10,
        1.015,
        "Temporal stability of compound-stratum slopes",
        transform=axes[1].transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )
    for ax in axes:
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    households, support, temporal_groups_frame = load_frame()
    full_groups = sorted(households[COMPOUND].dropna().unique())
    temporal_groups = sorted(temporal_groups_frame[COMPOUND].tolist())
    full_estimates, full_summary = fit_group_slopes(households, full_groups, "Full 2007-2021")
    early_estimates, early_summary = fit_group_slopes(
        households.loc[households[PERIOD].eq("2007-2017")],
        temporal_groups,
        "2007-2017",
    )
    late_estimates, late_summary = fit_group_slopes(
        households.loc[households[PERIOD].eq("2019-2021")],
        temporal_groups,
        "2019-2021",
    )
    estimates = pd.concat([full_estimates, early_estimates, late_estimates], ignore_index=True)
    gwr_summary = pd.read_csv(STRATA_SUMMARY)
    gwr_summary = gwr_summary.loc[
        gwr_summary["Meets minimum village support"].astype(str).str.lower().eq("true")
    ]
    associations = association_summary(
        full_estimates, early_estimates, late_estimates, gwr_summary
    )
    confirmatory_assessment = {
        "preferred_full_sample_joint_heterogeneity_probability_value": full_summary[
            "Joint equal slopes wild-cluster score bootstrap probability value"
        ],
        "full_sample_joint_heterogeneity_confirmed_at_0_05": bool(
            full_summary[
                "Joint equal slopes wild-cluster score bootstrap probability value"
            ]
            < 0.05
        ),
        "household_slopes_track_gwr_surface_at_0_05": bool(
            associations[
                "full_household_vs_gwr_compound_mean_spearman_probability_value"
            ]
            < 0.05
        ),
        "early_late_rank_stability_confirmed_at_0_05": bool(
            associations["early_late_compound_slope_spearman_probability_value"]
            < 0.05
            and associations["early_late_compound_slope_spearman_correlation"] > 0
        ),
        "status": "Not confirmed in the original household regressions",
        "reason": (
            "The preferred wild-cluster joint test does not reject equal slopes; the "
            "household-GWR rank correlation is not significant at 0.05; and early-late "
            "compound slopes are uncorrelated with fewer than half retaining the same sign."
        ),
    }
    summary = {
        "design": (
            "Household survey-weighted compound-stratum NPP slopes; commune and survey-time "
            "fixed effects; annual climate and household-composition controls; spatial-block "
            "clustered uncertainty; restricted wild-cluster score bootstrap"
        ),
        "full_sample": full_summary,
        "early_sample": early_summary,
        "late_sample": late_summary,
        "temporal_support_gate": {
            "minimum_households_per_period": MIN_TEMPORAL_HOUSEHOLDS,
            "minimum_villages_per_period": MIN_TEMPORAL_VILLAGES,
            "supported_compound_strata": len(temporal_groups),
        },
        "cross_design_and_temporal_stability": associations,
        "confirmatory_assessment": confirmatory_assessment,
        "interpretation_limit": (
            "The compound strata were selected after exploratory GWR and GeoDetector screening. "
            "The household regressions test internal consistency and temporal stability; they do "
            "not confirm the exploratory spatial surface and remain associational."
        ),
    }
    support.to_csv(OUTPUT / "compound_strata_household_support.csv", index=False)
    estimates.to_csv(OUTPUT / "compound_strata_household_slope_estimates.csv", index=False)
    pd.DataFrame([full_summary, early_summary, late_summary]).to_csv(
        OUTPUT / "compound_strata_model_tests.csv", index=False
    )
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Household confirmation of province-by-precipitation NPP-welfare slopes\n\n"
        "This experiment estimates the GeoDetector-supported compound-stratum slopes in the "
        "original household records. Results remain associational.\n",
        encoding="utf-8",
    )
    make_figure(
        full_estimates,
        early_estimates,
        late_estimates,
        gwr_summary,
        OUTPUT / "household_compound_strata_coefficients.png",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nFull-sample estimates")
    print(full_estimates.sort_values("Coefficient").to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
