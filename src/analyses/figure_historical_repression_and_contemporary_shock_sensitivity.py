#!/usr/bin/env python3
"""Historical Repression and Contemporary Shock Sensitivity.

Plan: Estimate and display the frozen first annual land-NPP outcome model, mandatory
within-modern-commune confirmation, fixed bandwidths, alternative rainfall, triangular
weights, river-distance adjustment, outcome-quality adjustment, and standardized SESOI.
Framework: AnaSOP Sections 5.3-5.4, 6.8-6.9, and the annual land-productivity workflow
in Section 7.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[2]
PANEL = (
    ROOT / "data/processed/historical_boundary_annual_spatial_climate_preprocessed.parquet"
)
OUTPUT = (
    ROOT
    / "data/results/figures/Figure_historical_repression_and_contemporary_shock_sensitivity.png"
)

PRIMARY_OUTCOME = "Annual Land NPP Anomaly kg C per m2"
STANDARDIZED_OUTCOME = "Annual Land NPP Anomaly Z 2001-2020"
PRIMARY_SHOCK = "May October Rainfall Anomaly Z (1991-2020)"
ALTERNATIVE_SHOCK = "Annual Rainfall Anomaly Z (1991-2020)"
QUALITY = "Mean NPP QC Filled Growing-Season Days Percent"
RIVER = "Log One Plus Distance to River m"
DISTANCE = "Signed Distance to Historical Repression Boundary km"
ABS_DISTANCE = "Absolute Distance to Historical Repression Boundary km"
TREATMENT = "Higher-Repression Southwest Zone"
COMMUNE = "Linked Climate Commune Code"
VILLAGE = "Village Code"
SEGMENT = "Historical Boundary Segment"
YEAR = "Year"

BANDWIDTHS = [2, 5, 10, 15, 20, 30]
PRIMARY_BANDWIDTH = 5
SESOI = 0.20
COLORS = {"West": "#3C78B5", "Southwest": "#C94C35"}

READ_COLUMNS = [
    VILLAGE,
    YEAR,
    COMMUNE,
    SEGMENT,
    TREATMENT,
    DISTANCE,
    ABS_DISTANCE,
    PRIMARY_SHOCK,
    ALTERNATIVE_SHOCK,
    PRIMARY_OUTCOME,
    STANDARDIZED_OUTCOME,
    QUALITY,
    RIVER,
]


@dataclass(frozen=True)
class Estimate:
    label: str
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    p_value: float
    sample_size: int
    villages: int
    district_year_clusters: int
    bandwidth_km: int
    outcome: str
    shock: str
    sample: str
    weights: str
    adjustment: str


@dataclass(frozen=True)
class FittedModel:
    estimate: Estimate
    params: pd.Series
    covariance: pd.DataFrame
    critical_value: float
    sample: pd.DataFrame


def prepared_panel() -> pd.DataFrame:
    data = pd.read_parquet(PANEL, columns=READ_COLUMNS).copy()
    data["_district"] = data[COMMUNE].astype(str).str[:4]
    data["_district_year"] = data["_district"] + "_" + data[YEAR].astype(str)
    data["_segment_year"] = data[SEGMENT].astype(str) + "_" + data[YEAR].astype(str)
    first_year = data.loc[data[YEAR].eq(data[YEAR].min())]
    side_count = first_year.groupby(COMMUNE, observed=True)[TREATMENT].nunique()
    cross_side_communes = set(side_count.loc[side_count.eq(2)].index.astype(str))
    data["_cross_side_commune"] = data[COMMUNE].astype(str).isin(cross_side_communes)
    return data


def fit_model(
    panel: pd.DataFrame,
    *,
    label: str,
    outcome: str = PRIMARY_OUTCOME,
    shock: str = PRIMARY_SHOCK,
    bandwidth_km: int = PRIMARY_BANDWIDTH,
    confirmation: bool = False,
    triangular: bool = False,
    modifier: str | None = None,
) -> FittedModel:
    required = [
        outcome,
        shock,
        VILLAGE,
        YEAR,
        COMMUNE,
        SEGMENT,
        TREATMENT,
        DISTANCE,
        ABS_DISTANCE,
        "_district_year",
        "_segment_year",
    ]
    if modifier is not None:
        required.append(modifier)
    sample = panel.loc[panel[ABS_DISTANCE].le(bandwidth_km)].copy()
    if confirmation:
        sample = sample.loc[sample["_cross_side_commune"]].copy()
    sample = sample.dropna(subset=required).copy()
    if sample.empty:
        raise ValueError(f"No complete observations for {label}")

    sample["_shock"] = sample[shock].astype(float)
    sample["_treat_shock"] = sample[TREATMENT] * sample["_shock"]
    sample["_distance_shock"] = sample[DISTANCE] * sample["_shock"]
    sample["_treat_distance_shock"] = (
        sample[TREATMENT] * sample[DISTANCE] * sample["_shock"]
    )
    exog_names = ["_treat_shock", "_distance_shock", "_treat_distance_shock"]
    if not confirmation:
        exog_names.insert(0, "_shock")

    if modifier is not None:
        sample["_modifier"] = sample[modifier] - sample[modifier].mean()
        sample["_shock_modifier"] = sample["_shock"] * sample["_modifier"]
        sample["_treat_shock_modifier"] = (
            sample[TREATMENT] * sample["_shock"] * sample["_modifier"]
        )
        exog_names.extend(["_shock_modifier", "_treat_shock_modifier"])
        if modifier == QUALITY:
            sample["_treat_modifier"] = sample[TREATMENT] * sample["_modifier"]
            exog_names.extend(["_modifier", "_treat_modifier"])

    absorb_names = [VILLAGE, "_segment_year"]
    if confirmation:
        sample["_commune_year"] = sample[COMMUNE].astype(str) + "_" + sample[YEAR].astype(str)
        absorb_names.append("_commune_year")
    absorb = sample[absorb_names].astype("category")
    clusters = pd.DataFrame(
        {
            "village": pd.factorize(sample[VILLAGE])[0],
            "district_year": pd.factorize(sample["_district_year"])[0],
        },
        index=sample.index,
    )
    weights = None
    weight_label = "Equal village-year"
    if triangular:
        weights = (1 - sample[ABS_DISTANCE] / bandwidth_km).clip(lower=1e-8)
        weight_label = "Triangular distance"

    fitted = AbsorbingLS(
        dependent=sample[outcome],
        exog=sample[exog_names],
        absorb=absorb,
        weights=weights,
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=clusters,
        debiased=True,
    )
    target = "_treat_shock"
    interval = fitted.conf_int().loc[target]
    critical = float(student_t.ppf(0.975, fitted.df_resid))
    adjustment = "None"
    if modifier == QUALITY:
        adjustment = "NPP quality interactions"
    elif modifier == RIVER:
        adjustment = "River-distance interactions"
    estimate = Estimate(
        label=label,
        estimate=float(fitted.params[target]),
        standard_error=float(fitted.std_errors[target]),
        ci_low=float(interval["lower"]),
        ci_high=float(interval["upper"]),
        p_value=float(fitted.pvalues[target]),
        sample_size=int(fitted.nobs),
        villages=int(sample[VILLAGE].nunique()),
        district_year_clusters=int(sample["_district_year"].nunique()),
        bandwidth_km=bandwidth_km,
        outcome=outcome,
        shock=shock,
        sample="Cross-side modern communes" if confirmation else "All eligible villages",
        weights=weight_label,
        adjustment=adjustment,
    )
    return FittedModel(
        estimate=estimate,
        params=fitted.params.copy(),
        covariance=fitted.cov.copy(),
        critical_value=critical,
        sample=sample,
    )


def response_slope(model: FittedModel, side: str) -> tuple[float, float]:
    if side == "West":
        return float(model.params["_shock"]), float(
            np.sqrt(model.covariance.loc["_shock", "_shock"])
        )
    vector = np.array([1.0, 1.0])
    covariance = model.covariance.loc[
        ["_shock", "_treat_shock"], ["_shock", "_treat_shock"]
    ].to_numpy()
    slope = float(model.params["_shock"] + model.params["_treat_shock"])
    standard_error = float(np.sqrt(vector @ covariance @ vector))
    return slope, standard_error


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.55, zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#333333")


def plot_response(ax: plt.Axes, model: FittedModel) -> None:
    low, high = model.sample["_shock"].quantile([0.02, 0.98])
    shock_grid = np.linspace(low, high, 120)
    ax.axhline(0, color="#333333", linewidth=0.9, zorder=1)
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle=":", zorder=1)
    for side in ("West", "Southwest"):
        slope, standard_error = response_slope(model, side)
        response = slope * shock_grid
        margin = model.critical_value * standard_error * np.abs(shock_grid)
        ax.fill_between(
            shock_grid,
            response - margin,
            response + margin,
            color=COLORS[side],
            alpha=0.16,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            shock_grid,
            response,
            color=COLORS[side],
            linewidth=2.0,
            label=side,
            zorder=3,
        )
    ax.set_xlabel("May–October rainfall anomaly (SD)")
    ax.set_ylabel("Model-implied partial NPP response\n(kg C m$^{-2}$)")
    style_axis(ax)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.text(
        0.02,
        0.04,
        f"5 km primary · {model.estimate.villages} villages · 2001–2021",
        transform=ax.transAxes,
        fontsize=7.8,
        ha="left",
        va="bottom",
    )


def forest_plot(
    ax: plt.Axes,
    estimates: list[Estimate],
    *,
    xlabel: str,
    equivalence: bool = False,
) -> None:
    ordered = list(reversed(estimates))
    y = np.arange(len(ordered))
    if equivalence:
        ax.axvspan(-SESOI, SESOI, color="#E8EFE4", alpha=0.90, zorder=0)
        ax.axvline(-SESOI, color="#71896A", linewidth=0.8, linestyle="--")
        ax.axvline(SESOI, color="#71896A", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#333333", linewidth=0.9, zorder=1)
    for index, estimate in enumerate(ordered):
        color = "#111111" if "Confirmation" in estimate.label else "#3C78B5"
        marker = "D" if "Confirmation" in estimate.label else "o"
        ax.errorbar(
            estimate.estimate,
            index,
            xerr=[
                [estimate.estimate - estimate.ci_low],
                [estimate.ci_high - estimate.estimate],
            ],
            fmt=marker,
            color=color,
            ecolor=color,
            elinewidth=1.25,
            capsize=2.5,
            markersize=5,
            zorder=3,
        )
    ax.set_yticks(y, [item.label for item in ordered], fontsize=7.6)
    ax.set_xlabel(xlabel)
    style_axis(ax, "x")
    if equivalence:
        ax.legend(
            handles=[Patch(facecolor="#E8EFE4", label="±0.20 SD SESOI")],
            loc="lower right",
            frameon=True,
            fontsize=7.6,
        )


def plot_bandwidth(ax: plt.Axes, estimates: list[Estimate]) -> None:
    data = pd.DataFrame([item.__dict__ for item in estimates]).sort_values("bandwidth_km")
    ax.axhline(0, color="#333333", linewidth=0.9, zorder=1)
    ax.axvspan(4.55, 5.45, color="#F7E6A6", alpha=0.75, zorder=0)
    ax.fill_between(
        data["bandwidth_km"],
        data["ci_low"],
        data["ci_high"],
        color="#3C78B5",
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        data["bandwidth_km"],
        data["estimate"],
        color="#3C78B5",
        marker="o",
        linewidth=1.8,
        markersize=5,
        zorder=3,
    )
    ax.set_xticks(BANDWIDTHS)
    ax.set_xlabel("Symmetric boundary bandwidth (km)")
    ax.set_ylabel("Southwest − West rainfall response\n(kg C m$^{-2}$ per 1-SD shock)")
    style_axis(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#3C78B5", marker="o", label="Estimate and 95% CI"),
            Patch(facecolor="#F7E6A6", label="5 km primary"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=7.6,
    )


def main() -> None:
    panel = prepared_panel()
    primary = fit_model(panel, label="Primary 5 km")
    confirmation = fit_model(panel, label="Confirmation 5 km", confirmation=True)
    annual = fit_model(panel, label="Annual rainfall", shock=ALTERNATIVE_SHOCK)
    triangular = fit_model(panel, label="Triangular weights", triangular=True)
    river = fit_model(panel, label="River-distance adjusted", modifier=RIVER)
    quality = fit_model(panel, label="NPP-quality adjusted", modifier=QUALITY)
    bandwidth_models = [
        fit_model(panel, label=f"{bandwidth} km", bandwidth_km=bandwidth)
        for bandwidth in BANDWIDTHS
    ]
    standardized = [
        fit_model(panel, label="Primary 5 km", outcome=STANDARDIZED_OUTCOME),
        fit_model(
            panel,
            label="Confirmation 5 km",
            outcome=STANDARDIZED_OUTCOME,
            confirmation=True,
        ),
        fit_model(
            panel,
            label="Annual rainfall",
            outcome=STANDARDIZED_OUTCOME,
            shock=ALTERNATIVE_SHOCK,
        ),
        fit_model(
            panel,
            label="Triangular weights",
            outcome=STANDARDIZED_OUTCOME,
            triangular=True,
        ),
        fit_model(
            panel,
            label="River-distance adjusted",
            outcome=STANDARDIZED_OUTCOME,
            modifier=RIVER,
        ),
        fit_model(
            panel,
            label="NPP-quality adjusted",
            outcome=STANDARDIZED_OUTCOME,
            modifier=QUALITY,
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.5))
    plot_response(axes[0, 0], primary)
    forest_plot(
        axes[0, 1],
        [
            primary.estimate,
            confirmation.estimate,
            annual.estimate,
            triangular.estimate,
            river.estimate,
            quality.estimate,
        ],
        xlabel="Southwest − West rainfall response\n(kg C m$^{-2}$ per 1-SD shock)",
    )
    plot_bandwidth(axes[1, 0], [model.estimate for model in bandwidth_models])
    forest_plot(
        axes[1, 1],
        [model.estimate for model in standardized],
        xlabel="Southwest − West rainfall response\n(outcome SD per 1-SD shock)",
        equivalence=True,
    )

    for label, ax in zip("abcd", axes.flat, strict=True):
        ax.text(
            -0.11,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.98, bottom=0.09, wspace=0.47, hspace=0.35)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(
        "Primary natural-unit interaction: "
        f"{primary.estimate.estimate:.6f} "
        f"[{primary.estimate.ci_low:.6f}, {primary.estimate.ci_high:.6f}]"
    )
    print(
        "Confirmation natural-unit interaction: "
        f"{confirmation.estimate.estimate:.6f} "
        f"[{confirmation.estimate.ci_low:.6f}, {confirmation.estimate.ci_high:.6f}]"
    )
    print(
        "Primary standardized interaction: "
        f"{standardized[0].estimate.estimate:.4f} "
        f"[{standardized[0].estimate.ci_low:.4f}, {standardized[0].estimate.ci_high:.4f}]"
    )
    print(
        "Confirmation standardized interaction: "
        f"{standardized[1].estimate.estimate:.4f} "
        f"[{standardized[1].estimate.ci_low:.4f}, {standardized[1].estimate.ci_high:.4f}]"
    )
    print(
        "Primary inference: village + district-year two-way clustered SE; "
        f"N={primary.estimate.sample_size}; villages={primary.estimate.villages}; "
        f"district-years={primary.estimate.district_year_clusters}"
    )


if __name__ == "__main__":
    main()
