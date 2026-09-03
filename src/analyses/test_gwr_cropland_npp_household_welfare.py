#!/usr/bin/env python3
"""Adaptive GWR proof-of-concept for cropland NPP and household food welfare.

The experiment uses one long-run residual observation per surveyed village. It asks
whether a weak national average NPP-welfare slope conceals geographically varying
local slopes. The implementation uses an adaptive bi-square kernel, household-count
precision weights, leave-one-out bandwidth selection, 999 coordinate-randomisation
tests of slope variability, and Moran diagnostics for global and GWR residuals.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = (
    ROOT
    / "data/exp/analysis/climate-welfare/spatial-cropland-npp-household-welfare"
)
SOURCE = SOURCE_DIR / "village_long_run_residuals.csv"
BOUNDARIES = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
OUTPUT = ROOT / "data/exp/analysis/climate-welfare/gwr-cropland-npp-household-welfare"

ID = "National Village Point ID"
LONGITUDE = "Point Longitude"
LATITUDE = "Point Latitude"
X_NAME = "Mean cropland NPP residual"
Y_NAME = "Mean food welfare residual"
BASE_WEIGHT = "Households"
SEED = 20260824
PERMUTATIONS = 999
MORAN_NEIGHBORS = 8
MAP_EXTENT = (101.45, 108.45, 9.75, 15.35)
BANDWIDTH_CANDIDATES = [60, 90, 130, 180, 250, 350, 500, 700, 950, 1250, 1650, 2100, 2600]


def weighted_rmse(residual: np.ndarray, weights: np.ndarray) -> float:
    observed = np.isfinite(residual) & np.isfinite(weights) & (weights > 0)
    return float(np.sqrt(np.average(residual[observed] ** 2, weights=weights[observed])))


def project_coordinates(frame: pd.DataFrame) -> np.ndarray:
    points = gpd.GeoDataFrame(
        frame[[LONGITUDE, LATITUDE]].copy(),
        geometry=gpd.points_from_xy(frame[LONGITUDE], frame[LATITUDE]),
        crs="EPSG:4326",
    ).to_crs("EPSG:32648")
    return np.column_stack([points.geometry.x.to_numpy(), points.geometry.y.to_numpy()])


def global_model(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> tuple[object, np.ndarray, float]:
    design = sm.add_constant(x)
    result = sm.WLS(y, design, weights=weights).fit(cov_type="HC1")
    raw_result = sm.WLS(y, design, weights=weights).fit()
    inverse = np.linalg.inv(design.T @ (weights[:, None] * design))
    leverage = weights * np.einsum("ij,jk,ik->i", design, inverse, design)
    loo_residual = raw_result.resid / np.maximum(1.0 - leverage, 1e-8)
    return result, np.asarray(raw_result.resid), weighted_rmse(loo_residual, weights)


def kernel_arrays(
    distances: np.ndarray,
    indexes: np.ndarray,
    k: int,
    exclude_self: bool,
) -> tuple[np.ndarray, np.ndarray]:
    selected_distances = distances[:, :k]
    selected_indexes = indexes[:, :k]
    bandwidth = np.maximum(selected_distances[:, -1] * 1.000001, 1e-8)
    scaled = selected_distances / bandwidth[:, None]
    spatial_weight = np.square(np.clip(1.0 - np.square(scaled), 0.0, None))
    if exclude_self:
        spatial_weight[:, 0] = 0.0
    return selected_indexes, spatial_weight


def local_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    base_weight: np.ndarray,
    indexes: np.ndarray,
    spatial_weight: np.ndarray,
    predict_x: np.ndarray | None = None,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x)
    slopes = np.full(n, np.nan)
    intercepts = np.full(n, np.nan)
    predictions = np.full(n, np.nan)
    effective_n = np.full(n, np.nan)
    target_x = x if predict_x is None else predict_x
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        neighbor = indexes[start:stop]
        local_weight = spatial_weight[start:stop] * base_weight[neighbor]
        local_x = x[neighbor]
        local_y = y[neighbor]
        sw = local_weight.sum(axis=1)
        sx = (local_weight * local_x).sum(axis=1)
        sy = (local_weight * local_y).sum(axis=1)
        sxx = (local_weight * local_x * local_x).sum(axis=1)
        sxy = (local_weight * local_x * local_y).sum(axis=1)
        denominator = sxx - sx * sx / np.maximum(sw, 1e-12)
        numerator = sxy - sx * sy / np.maximum(sw, 1e-12)
        valid = (sw > 0) & (np.abs(denominator) > 1e-12)
        slope = np.full(stop - start, np.nan)
        intercept = np.full(stop - start, np.nan)
        slope[valid] = numerator[valid] / denominator[valid]
        intercept[valid] = (sy[valid] - slope[valid] * sx[valid]) / sw[valid]
        prediction = intercept + slope * target_x[start:stop]
        slopes[start:stop] = slope
        intercepts[start:stop] = intercept
        predictions[start:stop] = prediction
        effective_n[start:stop] = np.square(sw) / np.maximum(
            np.square(local_weight).sum(axis=1), 1e-12
        )
    return intercepts, slopes, predictions, effective_n


def select_bandwidth(
    x: np.ndarray,
    y: np.ndarray,
    base_weight: np.ndarray,
    distances: np.ndarray,
    indexes: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for k in BANDWIDTH_CANDIDATES:
        local_indexes, local_kernel = kernel_arrays(distances, indexes, k, exclude_self=True)
        _, slopes, prediction, effective_n = local_linear_fit(
            x,
            y,
            base_weight,
            local_indexes,
            local_kernel,
            predict_x=x,
        )
        rows.append(
            {
                "Adaptive Neighbor Bandwidth": k,
                "Leave-One-Out Weighted RMSE": weighted_rmse(y - prediction, base_weight),
                "Median Effective Neighbor Count": float(np.nanmedian(effective_n)),
                "Valid Local Fits": int(np.isfinite(slopes).sum()),
            }
        )
    return pd.DataFrame(rows)


def row_standard_knn(coordinates: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(coordinates)
    _, indexes = tree.query(coordinates, k=k + 1)
    weights = np.zeros((len(coordinates), len(coordinates)), dtype=np.float32)
    rows = np.repeat(np.arange(len(coordinates)), k)
    columns = indexes[:, 1 : k + 1].reshape(-1)
    weights[rows, columns] = 1.0 / k
    return weights


def moran_test(
    values: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
) -> tuple[float, float]:
    z = (values - values.mean()) / values.std(ddof=0)
    statistic = float(np.dot(z, weights @ z) / np.dot(z, z))
    null = np.empty(permutations)
    for iteration in range(permutations):
        permuted = rng.permutation(z)
        null[iteration] = float(
            np.dot(permuted, weights @ permuted) / np.dot(permuted, permuted)
        )
    pvalue = float((1 + np.sum(np.abs(null) >= abs(statistic))) / (permutations + 1))
    return statistic, pvalue


def coefficient_variability_test(
    x: np.ndarray,
    y: np.ndarray,
    base_weight: np.ndarray,
    indexes: np.ndarray,
    kernel: np.ndarray,
    observed_slopes: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, object]:
    observed_sd = float(np.nanstd(observed_slopes, ddof=0))
    observed_iqr = float(
        np.nanquantile(observed_slopes, 0.75) - np.nanquantile(observed_slopes, 0.25)
    )
    null_sd = np.empty(PERMUTATIONS)
    null_iqr = np.empty(PERMUTATIONS)
    for iteration in range(PERMUTATIONS):
        permutation = rng.permutation(len(x))
        _, slopes, _, _ = local_linear_fit(
            x[permutation],
            y[permutation],
            base_weight[permutation],
            indexes,
            kernel,
            predict_x=x[permutation],
            chunk_size=384,
        )
        null_sd[iteration] = np.nanstd(slopes, ddof=0)
        null_iqr[iteration] = np.nanquantile(slopes, 0.75) - np.nanquantile(slopes, 0.25)
    return {
        "permutations": PERMUTATIONS,
        "observed_local_slope_standard_deviation": observed_sd,
        "null_mean_local_slope_standard_deviation": float(null_sd.mean()),
        "slope_standard_deviation_probability_value": float(
            (1 + np.sum(null_sd >= observed_sd)) / (PERMUTATIONS + 1)
        ),
        "observed_local_slope_interquartile_range": observed_iqr,
        "null_mean_local_slope_interquartile_range": float(null_iqr.mean()),
        "slope_interquartile_range_probability_value": float(
            (1 + np.sum(null_iqr >= observed_iqr)) / (PERMUTATIONS + 1)
        ),
        "null_slope_standard_deviation_quantiles": {
            "2.5 percent": float(np.quantile(null_sd, 0.025)),
            "50 percent": float(np.quantile(null_sd, 0.5)),
            "97.5 percent": float(np.quantile(null_sd, 0.975)),
        },
    }


def map_context(
    ax: plt.Axes,
    countries: gpd.GeoDataFrame,
    provinces: gpd.GeoDataFrame,
) -> None:
    countries.plot(
        ax=ax,
        color="#f1eee6",
        edgecolor="#a7a39a",
        linewidth=0.55,
        zorder=0,
    )
    countries.loc[countries["ADMIN"].eq("Cambodia")].plot(
        ax=ax,
        color="#fbfaf6",
        edgecolor="#575757",
        linewidth=0.9,
        zorder=1,
    )
    provinces.boundary.plot(ax=ax, color="#77736a", linewidth=0.45, zorder=2)
    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_xticks(np.arange(102, 109, 1))
    ax.set_yticks(np.arange(10, 16, 1))
    ax.grid(color="#c8c8c8", linestyle=(0, (2, 3)), linewidth=0.45, alpha=0.65)
    ax.tick_params(axis="both", labelsize=7, length=2.5, color="#666666")
    ax.set_xlabel("Longitude (°E)", fontsize=8, labelpad=5)
    ax.set_ylabel("Latitude (°N)", fontsize=8, labelpad=5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#4c4c4c")
        spine.set_linewidth(0.7)
    for label, position in {
        "THAILAND": (101.72, 13.55),
        "LAO PDR": (107.18, 15.12),
        "VIETNAM": (108.10, 12.65),
    }.items():
        ax.text(
            *position,
            label,
            color="#77736a",
            fontsize=7,
            fontstyle="italic",
            ha="center",
            va="center",
            zorder=8,
        )


def make_figure(
    frame: pd.DataFrame,
    global_slope: float,
    summary: dict[str, object],
    path: Path,
) -> None:
    communes = gpd.read_file(BOUNDARIES).to_crs("EPSG:4326")
    provinces = communes.dissolve(by="ADM1_PCODE", as_index=False)
    countries = gpd.read_file(COUNTRIES).to_crs("EPSG:4326")
    countries = countries.loc[
        countries["ADMIN"].isin(["Cambodia", "Thailand", "Laos", "Vietnam"])
    ]
    points = gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame[LONGITUDE], frame[LATITUDE]),
        crs="EPSG:4326",
    )
    coefficients = frame["Local NPP-welfare coefficient"].to_numpy(float)
    bound = float(np.quantile(np.abs(coefficients), 0.98))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.4, 6.35),
        gridspec_kw={"width_ratios": [1.9, 1.0]},
    )
    fig.subplots_adjust(left=0.055, right=0.98, top=0.97, bottom=0.19, wspace=0.18)
    map_context(axes[0], countries, provinces)
    points.plot(
        ax=axes[0],
        column="Local NPP-welfare coefficient",
        cmap="RdBu_r",
        norm=norm,
        markersize=10,
        alpha=0.82,
        edgecolor="none",
        rasterized=True,
        zorder=5,
    )
    axes[0].text(
        0.018,
        0.982,
        "a",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.2},
        zorder=10,
    )
    axes[0].text(
        0.072,
        0.982,
        "Adaptive GWR local NPP–food-welfare coefficient",
        transform=axes[0].transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.2},
        zorder=10,
    )
    axes[0].text(
        0.025,
        0.035,
        (
            f"Adaptive bandwidth: {summary['selected_adaptive_neighbor_bandwidth']:,} villages\n"
            f"Slope-variability permutation p = "
            f"{summary['spatial_variability']['slope_standard_deviation_probability_value']:.3f}"
        ),
        transform=axes[0].transAxes,
        fontsize=7.6,
        ha="left",
        va="bottom",
        linespacing=1.35,
        zorder=10,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#a8a8a8",
            "linewidth": 0.55,
            "alpha": 0.92,
        },
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cax = inset_axes(
        axes[0],
        width="78%",
        height="3.2%",
        loc="lower center",
        bbox_to_anchor=(0.0, -0.155, 1.0, 1.0),
        bbox_transform=axes[0].transAxes,
        borderpad=0,
    )
    colorbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.outline.set_linewidth(0.55)
    colorbar.set_label(
        "Local log-food coefficient per 0.1 kg C m⁻² cropland NPP",
        fontsize=8,
        labelpad=4,
    )

    counts, bins, patches = axes[1].hist(coefficients, bins=34, edgecolor="white", linewidth=0.5)
    for left, right, patch in zip(bins[:-1], bins[1:], patches, strict=True):
        patch.set_facecolor(plt.cm.RdBu_r(norm((left + right) / 2)))
    axes[1].axvline(0, color="#4c4c4c", linewidth=1.0)
    axes[1].axvline(
        global_slope,
        color="#111111",
        linewidth=1.5,
        linestyle=(0, (4, 3)),
        label=f"Global slope = {global_slope:.3f}",
    )
    axes[1].text(
        0.02,
        0.985,
        "b",
        transform=axes[1].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    axes[1].text(
        0.10,
        0.985,
        "Distribution of local coefficients",
        transform=axes[1].transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )
    axes[1].set_xlabel(
        "Local log-food coefficient per 0.1 kg C m⁻² cropland NPP", fontsize=8
    )
    axes[1].set_ylabel("Villages", fontsize=8)
    axes[1].tick_params(labelsize=7)
    axes[1].grid(axis="y", color="#d3d3d3", linestyle=(0, (2, 3)), linewidth=0.5)
    axes[1].legend(loc="upper right", bbox_to_anchor=(0.98, 0.91), frameon=False, fontsize=8)
    for side in ["top", "right"]:
        axes[1].spines[side].set_visible(False)
    axes[1].text(
        0.04,
        0.04,
        (
            f"Negative local slopes: {summary['negative_local_slope_share'] * 100:.1f}%\n"
            f"Positive local slopes: {summary['positive_local_slope_share'] * 100:.1f}%\n"
            f"GWR CV-RMSE gain: {summary['gwr_cv_rmse_improvement_percent']:.2f}%"
        ),
        transform=axes[1].transAxes,
        fontsize=7.8,
        va="bottom",
        linespacing=1.4,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "#b0b0b0",
            "linewidth": 0.55,
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(SOURCE, dtype={"Province Code": str})
    frame = frame.dropna(subset=[X_NAME, Y_NAME, BASE_WEIGHT, LONGITUDE, LATITUDE]).copy()
    frame["Province Code"] = frame["Province Code"].str.zfill(2)
    x = frame[X_NAME].to_numpy(float)
    y = frame[Y_NAME].to_numpy(float)
    base_weight = frame[BASE_WEIGHT].to_numpy(float)
    coordinates = project_coordinates(frame)
    tree = cKDTree(coordinates)
    max_bandwidth = min(max(BANDWIDTH_CANDIDATES), len(frame))
    distances, indexes = tree.query(coordinates, k=max_bandwidth)

    global_result, global_residual, global_cv_rmse = global_model(x, y, base_weight)
    bandwidth = select_bandwidth(x, y, base_weight, distances, indexes)
    selected_k = int(
        bandwidth.loc[bandwidth["Leave-One-Out Weighted RMSE"].idxmin(), "Adaptive Neighbor Bandwidth"]
    )
    selected_indexes, selected_kernel = kernel_arrays(
        distances, indexes, selected_k, exclude_self=False
    )
    intercepts, slopes, prediction, effective_n = local_linear_fit(
        x,
        y,
        base_weight,
        selected_indexes,
        selected_kernel,
        predict_x=x,
    )
    gwr_residual = y - prediction
    selected_cv_rmse = float(
        bandwidth.loc[
            bandwidth["Adaptive Neighbor Bandwidth"].eq(selected_k),
            "Leave-One-Out Weighted RMSE",
        ].iloc[0]
    )

    selected_position = BANDWIDTH_CANDIDATES.index(selected_k)
    sensitivity_k = sorted(
        set(
            [
                BANDWIDTH_CANDIDATES[max(selected_position - 1, 0)],
                selected_k,
                BANDWIDTH_CANDIDATES[min(selected_position + 1, len(BANDWIDTH_CANDIDATES) - 1)],
            ]
        )
    )
    sensitivity_slopes: dict[int, np.ndarray] = {}
    for k in sensitivity_k:
        local_indexes, local_kernel = kernel_arrays(distances, indexes, k, exclude_self=False)
        _, local_slope, _, _ = local_linear_fit(
            x, y, base_weight, local_indexes, local_kernel, predict_x=x
        )
        sensitivity_slopes[k] = local_slope
    sign_matrix = np.column_stack([np.sign(sensitivity_slopes[k]) for k in sensitivity_k])
    stable_sign = np.all(sign_matrix == sign_matrix[:, [0]], axis=1)

    rng = np.random.default_rng(SEED)
    variability = coefficient_variability_test(
        x,
        y,
        base_weight,
        selected_indexes,
        selected_kernel,
        slopes,
        rng,
    )
    moran_weights = row_standard_knn(coordinates, MORAN_NEIGHBORS)
    global_moran_i, global_moran_p = moran_test(
        global_residual, moran_weights, rng, PERMUTATIONS
    )
    gwr_moran_i, gwr_moran_p = moran_test(gwr_residual, moran_weights, rng, PERMUTATIONS)

    frame["Local intercept"] = intercepts
    frame["Local NPP-welfare coefficient"] = slopes
    frame["GWR fitted food welfare residual"] = prediction
    frame["GWR residual"] = gwr_residual
    frame["Effective local neighbor count"] = effective_n
    frame["Local slope sign stable across adjacent bandwidths"] = stable_sign
    for k, local_slope in sensitivity_slopes.items():
        frame[f"Local coefficient bandwidth {k}"] = local_slope

    coefficient_quantiles = {
        f"{int(probability * 100)} percent": float(np.nanquantile(slopes, probability))
        for probability in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    }
    global_ci = global_result.conf_int(alpha=0.05)[1]
    summary: dict[str, object] = {
        "design": (
            "One long-run residual observation per village; adaptive bi-square GWR; "
            "household-count precision weights; leave-one-out bandwidth selection"
        ),
        "villages": int(len(frame)),
        "selected_adaptive_neighbor_bandwidth": selected_k,
        "selected_bandwidth_share_of_villages": float(selected_k / len(frame)),
        "adjacent_bandwidth_sensitivity": sensitivity_k,
        "global_weighted_slope": float(global_result.params[1]),
        "global_robust_standard_error": float(global_result.bse[1]),
        "global_95_percent_ci": [float(global_ci[0]), float(global_ci[1])],
        "global_probability_value": float(global_result.pvalues[1]),
        "global_r_squared": float(global_result.rsquared),
        "global_leave_one_out_weighted_rmse": global_cv_rmse,
        "gwr_leave_one_out_weighted_rmse": selected_cv_rmse,
        "gwr_cv_rmse_improvement_percent": float(
            100 * (global_cv_rmse - selected_cv_rmse) / global_cv_rmse
        ),
        "local_slope_quantiles": coefficient_quantiles,
        "negative_local_slope_share": float(np.mean(slopes < 0)),
        "positive_local_slope_share": float(np.mean(slopes > 0)),
        "stable_local_slope_sign_share_across_adjacent_bandwidths": float(
            np.mean(stable_sign)
        ),
        "spatial_variability": variability,
        "residual_spatial_diagnostics": {
            "neighbors": MORAN_NEIGHBORS,
            "permutations": PERMUTATIONS,
            "global_model_residual_moran_i": global_moran_i,
            "global_model_residual_moran_probability_value": global_moran_p,
            "gwr_residual_moran_i": gwr_moran_i,
            "gwr_residual_moran_probability_value": gwr_moran_p,
        },
        "interpretation_gate": {
            "spatial_variability_probability_below_0_05": bool(
                variability["slope_standard_deviation_probability_value"] < 0.05
            ),
            "gwr_cv_rmse_below_global": bool(selected_cv_rmse < global_cv_rmse),
            "gwr_reduces_absolute_residual_moran": bool(
                abs(gwr_moran_i) < abs(global_moran_i)
            ),
            "both_positive_and_negative_local_slopes_present": bool(
                np.any(slopes < 0) and np.any(slopes > 0)
            ),
        },
        "interpretation_limit": (
            "GWR local coefficients are exploratory spatial associations from residualised "
            "long-run village means, not causal household effects"
        ),
    }

    bandwidth.to_csv(OUTPUT / "gwr_bandwidth_cross_validation.csv", index=False)
    frame.to_csv(OUTPUT / "gwr_local_coefficients.csv", index=False)
    pd.DataFrame(
        [
            {
                "Model": "Global weighted regression",
                "Weighted CV RMSE": global_cv_rmse,
                "Residual Moran I": global_moran_i,
                "Residual Moran probability value": global_moran_p,
            },
            {
                "Model": f"Adaptive GWR, bandwidth {selected_k}",
                "Weighted CV RMSE": selected_cv_rmse,
                "Residual Moran I": gwr_moran_i,
                "Residual Moran probability value": gwr_moran_p,
            },
        ]
    ).to_csv(OUTPUT / "gwr_model_comparison.csv", index=False)
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Adaptive GWR proof-of-concept\n\n"
        "This exploratory experiment tests whether the long-run village-level cropland-NPP "
        "food-welfare association varies across Cambodia. It does not establish causality.\n",
        encoding="utf-8",
    )
    make_figure(
        frame,
        global_slope=float(global_result.params[1]),
        summary=summary,
        path=OUTPUT / "gwr_local_coefficient_map_and_distribution.png",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved outputs to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
