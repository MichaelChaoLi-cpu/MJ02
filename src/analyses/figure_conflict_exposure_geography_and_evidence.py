#!/usr/bin/env python3
"""Select outcome-blind village support and map its spatial distribution.

This is a prospective design diagnostic.  It reads NPP only for 2001--2007 and
never reads post-conflict NPP.  The current affected-district indicator remains
provisional, so the output does not freeze treatment or authorize causal
estimation.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import least_squares
from scipy.special import expit
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/cambodia_public_village_npp_conflict_panel_candidate_preprocessed.parquet"
POINTS = ROOT / "data/processed/cambodia_public_village_points_preprocessed.parquet"
CSES_CROSSWALK = ROOT / "data/processed/cses_to_cambodia_public_village_point_crosswalk_preprocessed.parquet"
EVENTS = ROOT / "data/processed/ucdp_cambodia_thailand_state_conflict_candidates_preprocessed.parquet"
COMMUNES = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
THAILAND = ROOT / "data/raw/geography/thailand_rtsd_ocha_2019/extracted/tha_admbnda_adm0_rtsd_20190221.shp"
GRID = ROOT / "data/processed/cambodia_national_1km_grid_preprocessed.parquet"
FLOOD = ROOT / "data/processed/cambodia_national_2011_gfd_flood_exposure_preprocessed.parquet"

FIGURE_OUT = ROOT / "data/exp/legacy-results/figures/Figure_conflict_exposure_geography_and_evidence.png"
AUDIT_DIR = ROOT / "data/exp/experiment-design/cambodia-thailand-village-common-support"
WEIGHTS_OUT = AUDIT_DIR / "village_common_support_weights.csv"
BALANCE_OUT = AUDIT_DIR / "predetermined_balance_diagnostics.csv"
SUMMARY_OUT = AUDIT_DIR / "common_support_summary.json"

PRIMARY_RADIUS_KM = 5
PRE_START = 2001
PRE_END = 2007
BORDER_POOL_MAX_KM = 200.0
CONTROL_WEIGHT_CAP = 0.02
MAPPED_WEIGHT_MASS = 0.95

NPP = "Village Buffer Mean Annual Land NPP Anomaly Z 2001-2020"
RAIN = "Village Buffer Mean Annual Precipitation Total mm"
DRY = "Village Buffer Mean May October Dry Rainfall Intensity"
HEAT = "Village Buffer Mean May October Heat Intensity"
BORDER = "Village Buffer Mean Distance to Cambodia Thailand Border km"
ELEVATION = "Village Buffer Mean Mean Elevation m"
SLOPE = "Village Buffer Mean Mean Slope Degrees"
ROAD = "Village Buffer Mean Distance to Road Excluding Post 2007 AidData Corridors km"
CROPLAND = "Village Buffer Mean Baseline Cropland Share"
LOG_POP = "Village Buffer Mean Log Baseline Population 2000"

BALANCE_FEATURES = [
    "Pre-conflict NPP mean",
    "Pre-conflict NPP trend",
    "Border distance",
    "Elevation",
    "Slope",
    "Historical-road distance",
    "Baseline cropland share",
    "Log baseline population",
]

REPORT_FEATURES = [
    *BALANCE_FEATURES[:2],
    "Pre-conflict NPP SD",
    "Pre-conflict rainfall mean",
    "Pre-conflict rainfall SD",
    "Pre-conflict dry intensity",
    "Pre-conflict heat intensity",
    *BALANCE_FEATURES[2:],
]

SECTOR_COLORS = {
    "Preah Vihear": "#b2182b",
    "Ta Moan-Ta Krabey": "#ef8a62",
}
CONTROL_COLOR = "#2166ac"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[np.searchsorted(cumulative, probability, side="left")])


def weighted_ks(treated: np.ndarray, controls: np.ndarray, weights: np.ndarray) -> float:
    support = np.sort(np.unique(np.concatenate([treated, controls])))
    treated_cdf = np.searchsorted(np.sort(treated), support, side="right") / len(treated)
    order = np.argsort(controls)
    controls_sorted = controls[order]
    weights_sorted = weights[order] / weights.sum()
    cumulative = np.cumsum(weights_sorted)
    control_position = np.searchsorted(controls_sorted, support, side="right") - 1
    control_cdf = np.where(control_position >= 0, cumulative[np.maximum(control_position, 0)], 0.0)
    return float(np.max(np.abs(treated_cdf - control_cdf)))


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    weights = weights / weights.sum()
    mean = np.sum(weights * values)
    return float(np.sum(weights * (values - mean) ** 2))


def build_baseline() -> pd.DataFrame:
    columns = [
        "National Village Point ID",
        "Year",
        "Public Village Name",
        "Point Longitude",
        "Point Latitude",
        "Province Name",
        "District Name",
        "Candidate Affected District",
        "Candidate Conflict Sector",
        NPP,
        RAIN,
        DRY,
        HEAT,
        BORDER,
        ELEVATION,
        SLOPE,
        ROAD,
        CROPLAND,
        LOG_POP,
    ]
    frame = pd.read_parquet(
        PANEL,
        columns=columns,
        filters=[
            ("Buffer Radius km", "=", PRIMARY_RADIUS_KM),
            ("Year", ">=", PRE_START),
            ("Year", "<=", PRE_END),
        ],
    )
    if frame["Year"].max() > PRE_END:
        raise RuntimeError("Outcome-blind guard failed: post-conflict NPP entered the support step")

    frame["Pre-conflict NPP trend"] = frame.groupby("National Village Point ID", sort=False)[NPP].transform(
        lambda values: np.polyfit(frame.loc[values.index, "Year"], values, 1)[0]
        if values.notna().sum() >= 6
        else np.nan
    )
    identifiers = [
        "National Village Point ID",
        "Public Village Name",
        "Point Longitude",
        "Point Latitude",
        "Province Name",
        "District Name",
        "Candidate Affected District",
        "Candidate Conflict Sector",
    ]
    baseline = frame.groupby(identifiers, dropna=False).agg(
        **{
            "Valid pre-conflict NPP years": (NPP, "count"),
            "Pre-conflict NPP mean": (NPP, "mean"),
            "Pre-conflict NPP trend": ("Pre-conflict NPP trend", "first"),
            "Pre-conflict NPP SD": (NPP, "std"),
            "Pre-conflict rainfall mean": (RAIN, "mean"),
            "Pre-conflict rainfall SD": (RAIN, "std"),
            "Pre-conflict dry intensity": (DRY, "mean"),
            "Pre-conflict heat intensity": (HEAT, "mean"),
            "Border distance": (BORDER, "first"),
            "Elevation": (ELEVATION, "first"),
            "Slope": (SLOPE, "first"),
            "Historical-road distance": (ROAD, "first"),
            "Baseline cropland share": (CROPLAND, "first"),
            "Log baseline population": (LOG_POP, "first"),
        }
    ).reset_index()
    baseline["Eligible pre-conflict data"] = (
        baseline["Valid pre-conflict NPP years"].ge(6)
        & baseline[REPORT_FEATURES].notna().all(axis=1)
    )
    return baseline


def capped_calibration_weights(treated: pd.DataFrame, controls: pd.DataFrame) -> tuple[np.ndarray, dict]:
    scaler = StandardScaler().fit(pd.concat([treated[BALANCE_FEATURES], controls[BALANCE_FEATURES]]))
    treated_target = scaler.transform(treated[BALANCE_FEATURES]).mean(axis=0)
    control_matrix = scaler.transform(controls[BALANCE_FEATURES])

    # This bounded logistic calibration is the dual form of a capped balancing
    # problem.  The intercept normalizes weights; slopes target treated means.
    def residual(parameters: np.ndarray) -> np.ndarray:
        raw_weights = CONTROL_WEIGHT_CAP * expit(
            parameters[0] + control_matrix @ parameters[1:]
        )
        return np.r_[
            10.0 * (raw_weights.sum() - 1.0),
            control_matrix.T @ raw_weights - treated_target,
        ]

    fit = least_squares(
        residual,
        np.zeros(len(BALANCE_FEATURES) + 1),
        max_nfev=25_000,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-8,
    )
    weights = CONTROL_WEIGHT_CAP * expit(fit.x[0] + control_matrix @ fit.x[1:])
    weights = weights / weights.sum()
    standardized_residual = control_matrix.T @ weights - treated_target
    details = {
        "solver_terminated_normally": bool(fit.success),
        "optimizer_message": str(fit.message),
        "optimizer_function_evaluations": int(fit.nfev),
        "optimizer_first_order_optimality": float(fit.optimality),
        "maximum_standardized_calibration_residual": float(np.max(np.abs(standardized_residual))),
        "calibration_residual_below_0_05": bool(np.max(np.abs(standardized_residual)) < 0.05),
    }
    return weights, details


def select_support(baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    treated = baseline[
        baseline["Candidate Affected District"].eq(1)
        & baseline["Eligible pre-conflict data"]
    ].copy()
    controls = baseline[
        baseline["Candidate Affected District"].eq(0)
        & baseline["Eligible pre-conflict data"]
        & baseline["Border distance"].le(BORDER_POOL_MAX_KM)
    ].copy()
    if len(treated) != 88:
        raise RuntimeError(f"Expected 88 provisional treated villages; found {len(treated)}")

    weights, optimizer = capped_calibration_weights(treated, controls)
    controls["Control calibration weight"] = weights
    controls = controls.sort_values("Control calibration weight", ascending=False).reset_index(drop=True)
    controls["Cumulative control weight"] = controls["Control calibration weight"].cumsum()
    controls["Mapped 95 percent weight set"] = controls["Cumulative control weight"].le(MAPPED_WEIGHT_MASS)
    first_above = controls.index[controls["Cumulative control weight"].ge(MAPPED_WEIGHT_MASS)]
    if len(first_above):
        controls.loc[first_above[0], "Mapped 95 percent weight set"] = True

    pooled_control = baseline[
        baseline["Candidate Affected District"].eq(0)
        & baseline["Eligible pre-conflict data"]
        & baseline["Border distance"].le(BORDER_POOL_MAX_KM)
    ]
    diagnostics = []
    for variable in REPORT_FEATURES:
        treated_values = treated[variable].to_numpy(float)
        control_values = controls[variable].to_numpy(float)
        raw_values = pooled_control[variable].to_numpy(float)
        weight_values = controls["Control calibration weight"].to_numpy(float)
        pooled_sd = np.sqrt((np.var(treated_values, ddof=1) + np.var(raw_values, ddof=1)) / 2)
        raw_smd = (treated_values.mean() - raw_values.mean()) / pooled_sd
        weighted_mean = np.average(control_values, weights=weight_values)
        weighted_smd = (treated_values.mean() - weighted_mean) / pooled_sd
        variance_ratio = np.var(treated_values, ddof=1) / weighted_variance(control_values, weight_values)
        diagnostics.append(
            {
                "Variable": variable,
                "Used for calibration": variable in BALANCE_FEATURES,
                "Treated mean": treated_values.mean(),
                "Raw control mean": raw_values.mean(),
                "Weighted control mean": weighted_mean,
                "Raw standardized mean difference": raw_smd,
                "Weighted standardized mean difference": weighted_smd,
                "Weighted variance ratio treated/control": variance_ratio,
                "Weighted KS distance": weighted_ks(treated_values, control_values, weight_values),
                "Treated p05": np.quantile(treated_values, 0.05),
                "Treated p95": np.quantile(treated_values, 0.95),
                "Weighted control p05": weighted_quantile(control_values, weight_values, 0.05),
                "Weighted control p95": weighted_quantile(control_values, weight_values, 0.95),
            }
        )
    balance = pd.DataFrame(diagnostics)

    summary = {
        "design_status": "provisional; treatment-provenance gate unresolved",
        "outcome_blind_pre_period": [PRE_START, PRE_END],
        "primary_buffer_radius_km": PRIMARY_RADIUS_KM,
        "candidate_treated_villages": int(len(treated)),
        "candidate_treated_by_sector": {
            key: int(value)
            for key, value in treated.groupby("Candidate Conflict Sector", observed=True).size().items()
        },
        "eligible_border_context_controls": int(len(controls)),
        "mapped_controls_reaching_95_percent_weight": int(controls["Mapped 95 percent weight set"].sum()),
        "mapped_control_weight_mass": float(
            controls.loc[controls["Mapped 95 percent weight set"], "Control calibration weight"].sum()
        ),
        "maximum_control_weight": float(controls["Control calibration weight"].max()),
        "control_effective_sample_size": float(
            1.0 / np.square(controls["Control calibration weight"].to_numpy()).sum()
        ),
        "maximum_absolute_raw_smd": float(balance["Raw standardized mean difference"].abs().max()),
        "maximum_absolute_weighted_smd_calibration_variables": float(
            balance.loc[balance["Used for calibration"], "Weighted standardized mean difference"].abs().max()
        ),
        "maximum_absolute_weighted_smd_all_reported_variables": float(
            balance["Weighted standardized mean difference"].abs().max()
        ),
        "maximum_weighted_ks_all_reported_variables": float(balance["Weighted KS distance"].max()),
        "weight_cap": CONTROL_WEIGHT_CAP,
        "candidate_control_border_distance_max_km": BORDER_POOL_MAX_KM,
        "mapped_weight_mass_target": MAPPED_WEIGHT_MASS,
        "calibration_variables": BALANCE_FEATURES,
        **optimizer,
    }
    return treated, controls, balance, summary


def map_boundaries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    communes = gpd.read_file(COMMUNES).to_crs(4326)
    provinces = communes.dissolve(by="ADM1_EN", as_index=False)
    country = communes.dissolve()
    thailand = gpd.read_file(THAILAND).to_crs(4326)
    return country, provinces, thailand


def add_map_frame(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float], step: float) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(np.ceil(xlim[0] / step) * step, xlim[1] + 1e-9, step))
    ax.set_yticks(np.arange(np.ceil(ylim[0] / step) * step, ylim[1] + 1e-9, step))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.1f}°E")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1f}°N")
    ax.grid(True, color="#d9d9d9", linewidth=0.55, linestyle="--", zorder=0)
    ax.tick_params(labelsize=8, length=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#4d4d4d")
        spine.set_linewidth(0.8)
    ax.set_aspect("equal", adjustable="box")


def expand_extent_to_fill_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Expand, but never crop, a map extent to match its allocated panel box."""
    # GeoPandas sets a geographic aspect during plotting and can temporarily
    # shrink the active axes.  The GridSpec position is the intended frame that
    # all panels must fill, so calculate against it rather than ax.get_position().
    box = ax.get_subplotspec().get_position(fig)
    panel_ratio = (box.width * fig.get_figwidth()) / (box.height * fig.get_figheight())
    x_mid = (xlim[0] + xlim[1]) / 2
    y_mid = (ylim[0] + ylim[1]) / 2
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]
    if x_span / y_span < panel_ratio:
        x_span = y_span * panel_ratio
    else:
        y_span = x_span / panel_ratio
    return (
        (x_mid - x_span / 2, x_mid + x_span / 2),
        (y_mid - y_span / 2, y_mid + y_span / 2),
    )


def plot_base(ax: plt.Axes, country: gpd.GeoDataFrame, provinces: gpd.GeoDataFrame, thailand: gpd.GeoDataFrame) -> None:
    thailand.plot(ax=ax, facecolor="#f2f2f2", edgecolor="#969696", linewidth=0.6, zorder=1)
    country.plot(ax=ax, facecolor="#fafafa", edgecolor="#252525", linewidth=0.9, zorder=2)
    provinces.boundary.plot(ax=ax, color="#bdbdbd", linewidth=0.35, zorder=3)


def plot_sample(
    ax: plt.Axes,
    background: pd.DataFrame,
    treated: pd.DataFrame,
    mapped_controls: pd.DataFrame,
    cses_ids: set[str],
    events: pd.DataFrame,
    show_background: bool = True,
) -> None:
    if show_background:
        ax.scatter(
            background["Point Longitude"],
            background["Point Latitude"],
            s=2,
            color="#bdbdbd",
            alpha=0.28,
            linewidths=0,
            zorder=4,
        )
    control_sizes = 15 + 400 * mapped_controls["Control calibration weight"].to_numpy()
    ax.scatter(
        mapped_controls["Point Longitude"],
        mapped_controls["Point Latitude"],
        s=control_sizes,
        facecolor=CONTROL_COLOR,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.78,
        zorder=6,
    )
    for sector, color in SECTOR_COLORS.items():
        subset = treated[treated["Candidate Conflict Sector"].eq(sector)]
        ax.scatter(
            subset["Point Longitude"],
            subset["Point Latitude"],
            s=30,
            marker="^" if sector == "Preah Vihear" else "D",
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=8,
        )

    selected = pd.concat([treated, mapped_controls], ignore_index=True)
    linked = selected[selected["National Village Point ID"].isin(cses_ids)]
    if not linked.empty:
        ax.scatter(
            linked["Point Longitude"],
            linked["Point Latitude"],
            s=52,
            marker="s",
            facecolor="none",
            edgecolor="#111111",
            linewidth=0.8,
            zorder=9,
        )
    ax.scatter(
        events["longitude"],
        events["latitude"],
        s=75,
        marker="*",
        facecolor="#542788",
        edgecolor="white",
        linewidth=0.45,
        zorder=10,
    )


def build_figure(treated: pd.DataFrame, controls: pd.DataFrame) -> None:
    all_points = pd.read_parquet(
        POINTS,
        columns=["National Village Point ID", "Point Longitude", "Point Latitude"],
    )
    mapped_controls = controls[controls["Mapped 95 percent weight set"]].copy()
    cses = pd.read_parquet(CSES_CROSSWALK, columns=["National Village Point ID"])
    cses_ids = set(cses["National Village Point ID"].dropna().astype(str))
    events = pd.read_parquet(EVENTS)
    representative_events = events[events["id"].isin([98090, 99577])]
    country, provinces, thailand = map_boundaries()

    grid = pd.read_parquet(GRID, columns=["National Grid Cell ID", "Longitude", "Latitude"])
    flood = pd.read_parquet(
        FLOOD,
        columns=["National Grid Cell ID", "2011 Maximum Flooded Share Excluding Permanent Water"],
    )
    flood = grid.merge(flood, on="National Grid Cell ID", how="inner", validate="one_to_one")
    flood = flood[flood["2011 Maximum Flooded Share Excluding Permanent Water"].gt(0.05)]

    fig = plt.figure(figsize=(15.5, 9.0))
    axes = fig.subplot_mosaic(
        [
            ["a", "a", "b", "b"],
            ["a", "a", "c", "d"],
        ],
        gridspec_kw={"height_ratios": [1.04, 1.0]},
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.98, bottom=0.125, wspace=0.14, hspace=0.14)
    ax_a = axes["a"]
    ax_b = axes["b"]
    ax_c = axes["c"]
    ax_d = axes["d"]

    for ax in axes.values():
        plot_base(ax, country, provinces, thailand)

    requested_extents = {
        "a": ((102.2, 107.8), (10.2, 15.4), 1.0),
        "b": ((102.5, 105.6), (12.7, 15.0), 0.5),
        "c": ((104.15, 105.25), (13.65, 14.85), 0.25),
        "d": ((102.75, 103.75), (13.55, 14.8), 0.25),
    }
    aligned_extents = {
        label: (*expand_extent_to_fill_panel(fig, axes[label], xlim, ylim), step)
        for label, (xlim, ylim, step) in requested_extents.items()
    }

    plot_sample(ax_a, all_points, treated, mapped_controls, cses_ids, representative_events)
    add_map_frame(ax_a, *aligned_extents["a"])

    ax_b.scatter(
        flood["Longitude"],
        flood["Latitude"],
        s=2.0,
        c=flood["2011 Maximum Flooded Share Excluding Permanent Water"],
        cmap="Blues",
        vmin=0,
        vmax=1,
        alpha=0.55,
        linewidths=0,
        zorder=4,
    )
    plot_sample(ax_b, all_points, treated, mapped_controls, cses_ids, representative_events, show_background=False)
    add_map_frame(ax_b, *aligned_extents["b"])

    plot_sample(ax_c, all_points, treated, mapped_controls, cses_ids, representative_events)
    add_map_frame(ax_c, *aligned_extents["c"])

    plot_sample(ax_d, all_points, treated, mapped_controls, cses_ids, representative_events)
    add_map_frame(ax_d, *aligned_extents["d"])

    for label, ax in axes.items():
        ax.text(
            0.018,
            0.975,
            label,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            fontweight="bold",
            zorder=20,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6},
        )

    legend_handles = [
        Line2D([], [], marker="^", linestyle="", markersize=7, markerfacecolor=SECTOR_COLORS["Preah Vihear"], markeredgecolor="white", label="Candidate treated: Preah Vihear"),
        Line2D([], [], marker="D", linestyle="", markersize=6, markerfacecolor=SECTOR_COLORS["Ta Moan-Ta Krabey"], markeredgecolor="white", label="Candidate treated: Ta Moan–Ta Krabey"),
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=CONTROL_COLOR, markeredgecolor="white", label="Controls carrying 95% of weight"),
        Line2D([], [], marker="s", linestyle="", markersize=7, markerfacecolor="none", markeredgecolor="#111111", label="Also linked to CSES"),
        Line2D([], [], marker="*", linestyle="", markersize=9, markerfacecolor="#542788", markeredgecolor="white", label="Representative battle point"),
        Line2D([], [], marker="o", linestyle="", markersize=6, markerfacecolor="#6baed6", markeredgecolor="none", label="2011 observed flood (>5%)"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
        ncol=3,
        fontsize=8.1,
        columnspacing=1.55,
        handletextpad=0.6,
        borderpad=0.7,
        frameon=True,
        framealpha=0.94,
        edgecolor="#bdbdbd",
    )

    FIGURE_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = build_baseline()
    treated, controls, balance, summary = select_support(baseline)

    treated_out = treated.copy()
    treated_out["Analysis role"] = "Provisional treated"
    treated_out["Control calibration weight"] = np.nan
    treated_out["Cumulative control weight"] = np.nan
    treated_out["Mapped 95 percent weight set"] = True
    controls_out = controls.copy()
    controls_out["Analysis role"] = "Eligible weighted control"
    columns = [
        "National Village Point ID",
        "Public Village Name",
        "Province Name",
        "District Name",
        "Point Longitude",
        "Point Latitude",
        "Candidate Conflict Sector",
        "Analysis role",
        "Control calibration weight",
        "Cumulative control weight",
        "Mapped 95 percent weight set",
        "Valid pre-conflict NPP years",
        *REPORT_FEATURES,
    ]
    pd.concat([treated_out[columns], controls_out[columns]], ignore_index=True).to_csv(WEIGHTS_OUT, index=False)
    balance.to_csv(BALANCE_OUT, index=False)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    build_figure(treated, controls)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"figure: {FIGURE_OUT}")
    print(f"weights: {WEIGHTS_OUT}")
    print(f"balance: {BALANCE_OUT}")
    print(f"summary: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
