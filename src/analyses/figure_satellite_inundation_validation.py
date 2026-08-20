#!/usr/bin/env python3
"""Satellite Inundation Validation.

Plan: Compare extreme-wet rainfall with observed inundation and test whether
the conflict-conditioned agricultural response is reproduced by satellite
flood measures on a common 2009-2017 coverage sample.
Framework: AnaSOP Sections 5.1-5.2, 6.2-6.6, and the satellite-validation
step in Section 7. Satellite inundation is secondary convergence evidence,
not a coequal full-period shock measure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    ROOT / "data/processed/direction3_household_conflict_shock_preprocessed.parquet"
)
OUTPUT_PATH = ROOT / "data/exp/internal_output_archive/figures/Figure_satellite_inundation_validation.png"

YEAR = "Survey Year"
RESOLUTION = "Climate Geography Resolution"
GEOGRAPHY = "Climate Geography Code"
PROVINCE = "Province Code Component"
WEIGHT = "Household Survey Weight"
HOUSEHOLD_SIZE = "Household Size"
AGRICULTURAL_HOUSEHOLD = "Agricultural Household"
CONFLICT = "Log Bombing Unique Locations per 100 km2"
EXTREME_WET = "Annual Rainfall Extreme Wet Shock"
FLOOD_YEAR = "Survey Year Maximum Flooded Geography Share"
FLOOD_12M = "Preceding 12 Month Maximum Flooded Geography Share"
FLOOD_DURATION = "Survey Year Maximum Flood Duration Days"
CROP_YIELD = "Crop Yield kg per ha"
LOSS_SHARE = "Post Harvest Loss Share"

COMMON_COVERAGE = [FLOOD_YEAR, FLOOD_12M, FLOOD_DURATION]
SHOCK_LABELS = {
    EXTREME_WET: "Wet rainfall",
    FLOOD_YEAR: "Survey-year share",
    FLOOD_12M: "Prior-12m share",
    FLOOD_DURATION: "Flood duration",
}
SHOCK_LEGEND = {
    EXTREME_WET: "Extreme-wet rainfall (0→1)",
    FLOOD_YEAR: "Survey-year flooded share (1 SD)",
    FLOOD_12M: "Prior-12m flooded share (1 SD)",
    FLOOD_DURATION: "Flood duration (1 SD)",
}
SHOCK_STYLES = {
    EXTREME_WET: {"color": "#D95F0E", "marker": "^"},
    FLOOD_YEAR: {"color": "#2B8CBE", "marker": "o"},
    FLOOD_12M: {"color": "#31A354", "marker": "s"},
    FLOOD_DURATION: {"color": "#756BB1", "marker": "D"},
}
OUTCOME_LABELS = {
    CROP_YIELD: "Yield",
    LOSS_SHARE: "Loss share",
}


@dataclass(frozen=True)
class EstimateSpec:
    outcome: str
    shock: str
    transform: str


@dataclass
class Estimate:
    spec: EstimateSpec
    coefficient: float
    standard_error: float
    sample_size: int
    cluster_count: int
    survey_waves: tuple[int, ...]

    @property
    def lower(self) -> float:
        return self.coefficient - 1.96 * self.standard_error

    @property
    def upper(self) -> float:
        return self.coefficient + 1.96 * self.standard_error


ESTIMATE_SPECS = [
    EstimateSpec(outcome, shock, transform)
    for outcome, transform in [(CROP_YIELD, "asinh"), (LOSS_SHARE, "level")]
    for shock in [EXTREME_WET, FLOOD_YEAR, FLOOD_12M, FLOOD_DURATION]
]


def weighted_standardize(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    standard_deviation = float(np.sqrt(variance))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0:
        raise ValueError("Cannot standardize a variable with zero or invalid variance")
    return (values - mean) / standard_deviation


def transform_outcome(values: np.ndarray, transform: str) -> np.ndarray:
    if transform == "asinh":
        if np.nanmin(values) < 0:
            raise ValueError("Crop yield unexpectedly contains negative values")
        return np.arcsinh(values)
    if transform == "level":
        if np.nanmin(values) < 0 or np.nanmax(values) > 1:
            raise ValueError("Post-harvest loss share falls outside [0, 1]")
        return values
    raise ValueError(f"Unknown outcome transform: {transform}")


def prepare_common_sample(data: pd.DataFrame, outcome: str) -> pd.DataFrame:
    required = [
        outcome,
        EXTREME_WET,
        *COMMON_COVERAGE,
        CONFLICT,
        WEIGHT,
        HOUSEHOLD_SIZE,
        GEOGRAPHY,
        PROVINCE,
        YEAR,
        AGRICULTURAL_HOUSEHOLD,
    ]
    sample = data.loc[data[RESOLUTION].eq("commune"), required].dropna().copy()
    sample = sample.loc[
        sample[WEIGHT].gt(0) & sample[AGRICULTURAL_HOUSEHOLD].astype(bool)
    ].copy()
    if not sample[EXTREME_WET].isin([0, 1, False, True]).all():
        raise ValueError("Extreme-wet rainfall indicator is not binary")
    if not sample[YEAR].between(2009, 2017).all():
        raise ValueError("Common satellite sample unexpectedly includes unsupported waves")
    if sample[GEOGRAPHY].nunique() < 50:
        raise ValueError(f"Too few clusters for {outcome}")
    return sample


def fit_estimate(data: pd.DataFrame, spec: EstimateSpec) -> Estimate:
    sample = prepare_common_sample(data, spec.outcome)
    weights = sample[WEIGHT].to_numpy(dtype=float)
    conflict_z = weighted_standardize(sample[CONFLICT].to_numpy(dtype=float), weights)
    raw_shock = sample[spec.shock].to_numpy(dtype=float)
    shock_model = (
        raw_shock
        if spec.shock == EXTREME_WET
        else weighted_standardize(raw_shock, weights)
    )
    outcome = transform_outcome(
        sample[spec.outcome].to_numpy(dtype=float), spec.transform
    )
    outcome_z = weighted_standardize(outcome, weights)

    exogenous = pd.DataFrame(
        {
            "shock": shock_model,
            "conflict_x_shock": conflict_z * shock_model,
            "household_size": sample[HOUSEHOLD_SIZE].to_numpy(dtype=float),
        },
        index=sample.index,
    )
    province_wave = (
        sample[PROVINCE].astype(str) + "_" + sample[YEAR].astype(int).astype(str)
    )
    absorbed = pd.DataFrame(
        {
            "geography": sample[GEOGRAPHY].astype("category"),
            "province_wave": province_wave.astype("category"),
        },
        index=sample.index,
    )
    fitted = AbsorbingLS(
        dependent=pd.Series(outcome_z, index=sample.index, name="outcome_z"),
        exog=exogenous,
        absorb=absorbed,
        weights=pd.Series(weights, index=sample.index),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=sample[[GEOGRAPHY]],
        debiased=True,
    )
    if "conflict_x_shock" not in fitted.params.index:
        raise ValueError(
            f"Interaction absorbed for {OUTCOME_LABELS[spec.outcome]} × "
            f"{SHOCK_LABELS[spec.shock]}"
        )
    return Estimate(
        spec=spec,
        coefficient=float(fitted.params["conflict_x_shock"]),
        standard_error=float(fitted.std_errors["conflict_x_shock"]),
        sample_size=len(sample),
        cluster_count=int(sample[GEOGRAPHY].nunique()),
        survey_waves=tuple(sorted(sample[YEAR].astype(int).unique())),
    )


def geography_year_validation_sample(data: pd.DataFrame) -> pd.DataFrame:
    columns = [YEAR, GEOGRAPHY, EXTREME_WET, FLOOD_YEAR, FLOOD_12M, FLOOD_DURATION]
    sample = data.loc[data[RESOLUTION].eq("commune"), columns].dropna().copy()
    sample = sample.drop_duplicates([GEOGRAPHY, YEAR, EXTREME_WET, FLOOD_YEAR, FLOOD_DURATION])
    uniqueness = sample.groupby([GEOGRAPHY, YEAR])[
        [EXTREME_WET, FLOOD_YEAR, FLOOD_DURATION]
    ].nunique()
    if not uniqueness.le(1).all().all():
        raise ValueError("Survey-year satellite exposure is not unique within geography-year")
    sample = sample.drop_duplicates([GEOGRAPHY, YEAR])
    if not sample[YEAR].between(2009, 2017).all():
        raise ValueError("Validation scatter unexpectedly includes unsupported waves")
    return sample


def plot_scatter(ax: plt.Axes, sample: pd.DataFrame) -> float:
    rng = np.random.default_rng(20260819)
    x = sample[EXTREME_WET].astype(int).to_numpy(dtype=float)
    jitter = rng.uniform(-0.13, 0.13, size=len(sample))
    scatter = ax.scatter(
        x + jitter,
        sample[FLOOD_YEAR],
        c=sample[FLOOD_DURATION],
        cmap="viridis",
        s=13,
        alpha=0.28,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    for wet_value in [0, 1]:
        values = sample.loc[sample[EXTREME_WET].astype(int).eq(wet_value), FLOOD_YEAR]
        mean = float(values.mean())
        confidence_interval = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
        ax.errorbar(
            wet_value,
            mean,
            yerr=confidence_interval,
            fmt="o",
            markersize=7,
            color="#111827",
            ecolor="#111827",
            elinewidth=1.5,
            capsize=3,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=4,
        )
    correlation = float(sample[[EXTREME_WET, FLOOD_YEAR]].corr().iloc[0, 1])
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 1], ["No", "Yes"])
    ax.set_xlabel("Annual extreme-wet rainfall")
    ax.set_ylabel("Survey-year flooded geography share")
    ax.text(-0.10, 1.06, "a", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.0, 1.05, "Rainfall–inundation correspondence", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    ax.text(
        0.98,
        0.97,
        f"geography-years = {len(sample):,} · r = {correlation:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#4B5563",
    )
    colorbar = ax.figure.colorbar(scatter, ax=ax, pad=0.025, fraction=0.055)
    colorbar.set_label("Maximum flood duration (days)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#9CA3AF")
    ax.tick_params(colors="#374151", labelsize=8)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    return correlation


def plot_forest(ax: plt.Axes, estimates: list[Estimate]) -> None:
    positions = np.arange(len(estimates))[::-1]
    labels = [
        f"{OUTCOME_LABELS[item.spec.outcome]} · {SHOCK_LABELS[item.spec.shock]}"
        for item in estimates
    ]
    for position, estimate in zip(positions, estimates, strict=True):
        style = SHOCK_STYLES[estimate.spec.shock]
        ax.errorbar(
            estimate.coefficient,
            position,
            xerr=1.96 * estimate.standard_error,
            fmt=style["marker"],
            markersize=5.8,
            color=style["color"],
            ecolor=style["color"],
            elinewidth=1.35,
            capsize=2.8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
        )
    ax.axvline(0, color="#6B7280", linewidth=0.9, linestyle="--")
    ax.axhline(3.5, color="#D1D5DB", linewidth=0.8)
    ax.set_yticks(positions, labels)
    ax.set_ylim(-0.7, len(estimates) - 0.2)
    ax.set_xlabel("Conflict × shock estimate (outcome SD)")
    ax.text(-0.10, 1.06, "b", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    ax.text(0.0, 1.05, "Agricultural amplification on common coverage", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
    sample_sizes = sorted({item.sample_size for item in estimates})
    clusters = sorted({item.cluster_count for item in estimates})
    ax.text(
        0.98,
        0.97,
        f"N = {min(sample_sizes):,}–{max(sample_sizes):,} · clusters = {min(clusters):,}–{max(clusters):,}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#4B5563",
    )
    handles = [
        Line2D(
            [0],
            [0],
            color=SHOCK_STYLES[shock]["color"],
            marker=SHOCK_STYLES[shock]["marker"],
            linestyle="none",
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=SHOCK_LEGEND[shock],
        )
        for shock in [EXTREME_WET, FLOOD_YEAR, FLOOD_12M, FLOOD_DURATION]
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=7.3,
        handletextpad=0.35,
        labelspacing=0.35,
    )
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.tick_params(axis="x", colors="#374151", labelsize=8)
    ax.tick_params(axis="y", colors="#374151", labelsize=8, length=0)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.grid(axis="y", visible=False)


def main() -> None:
    columns = [
        YEAR,
        RESOLUTION,
        GEOGRAPHY,
        PROVINCE,
        WEIGHT,
        HOUSEHOLD_SIZE,
        AGRICULTURAL_HOUSEHOLD,
        CONFLICT,
        EXTREME_WET,
        FLOOD_YEAR,
        FLOOD_12M,
        FLOOD_DURATION,
        CROP_YIELD,
        LOSS_SHARE,
    ]
    data = pd.read_parquet(INPUT_PATH, columns=columns)
    assert len(data) == 62_920
    validation_sample = geography_year_validation_sample(data)
    estimates = [fit_estimate(data, spec) for spec in ESTIMATE_SPECS]

    sns.set_theme(style="whitegrid", context="paper")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 6.0),
        gridspec_kw={"width_ratios": [1.0, 1.30]},
    )
    correlation = plot_scatter(axes[0], validation_sample)
    plot_forest(axes[1], estimates)
    figure.subplots_adjust(left=0.08, right=0.985, top=0.92, bottom=0.12, wspace=0.48)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")
    print("Panels: 2 (a-b), no figure title or footer note")
    print(
        f"a Rainfall-inundation correspondence: geography-years={len(validation_sample):,}, "
        f"r={correlation:.6f}, waves={sorted(validation_sample[YEAR].astype(int).unique())}"
    )
    for estimate in estimates:
        print(
            f"b {OUTCOME_LABELS[estimate.spec.outcome]} | "
            f"{SHOCK_LABELS[estimate.spec.shock]}: "
            f"N={estimate.sample_size:,}, clusters={estimate.cluster_count:,}, "
            f"beta={estimate.coefficient:.6f}, SE={estimate.standard_error:.6f}, "
            f"waves={list(estimate.survey_waves)}"
        )


if __name__ == "__main__":
    main()
