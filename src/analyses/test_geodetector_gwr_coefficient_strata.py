#!/usr/bin/env python3
"""GeoDetector experiment for spatially varying NPP-welfare coefficients.

The outcome is the adaptive-GWR local cropland-NPP food-welfare coefficient. All
continuous strata are outcome-blind distributional tertiles. The experiment reports
standard unweighted GeoDetector q statistics, 999-permutation probabilities, BH
adjustment, and spatial-block held-out q values to guard against interpreting a
smoothed coefficient surface solely through in-sample fit.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analyses"))
import test_annual_climate_cropland_npp_model as annual_model  # noqa: E402


GWR_DIR = ROOT / "data/exp/analysis/climate-welfare/gwr-cropland-npp-household-welfare"
GWR = GWR_DIR / "gwr_local_coefficients.csv"
CONTEXT_PANEL = ROOT / "data/processed/cambodia_public_village_monsoon_npp_panel_preprocessed.parquet"
BOUNDARIES = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"
OUTPUT = ROOT / "data/exp/analysis/climate-welfare/geodetector-gwr-coefficient-strata"

ID = "National Village Point ID"
OUTCOME = "Local NPP-welfare coefficient"
LONGITUDE = "Point Longitude"
LATITUDE = "Point Latitude"
PERMUTATIONS = 999
SEED = 20260824
SPATIAL_BLOCK_DEGREES = 0.75
MIN_COMPOUND_STRATUM_VILLAGES = 30
MAP_EXTENT = (101.45, 108.45, 9.75, 15.35)

CONTEXT_COLUMNS = {
    "Long-run annual mean temperature": "Annual mean temperature C",
    "Long-run annual precipitation": annual_model.RAIN_SOURCE,
    "Baseline cropland share": "Village Buffer Mean Baseline Cropland Share",
    "Baseline population": "Village Buffer Mean Log Baseline Population 2000",
    "Historical road distance": (
        "Village Buffer Mean Distance to Road Excluding Post 2007 AidData Corridors km"
    ),
    "Mean elevation": "Village Buffer Mean Mean Elevation m",
    "Mean slope": "Village Buffer Mean Mean Slope Degrees",
}

FACTOR_LABELS = {
    "Long-run annual mean temperature": "Temperature",
    "Long-run annual precipitation": "Precipitation",
    "Baseline cropland share": "Cropland share",
    "Baseline population": "Population",
    "Historical road distance": "Road distance",
    "Mean elevation": "Elevation",
    "Mean slope": "Slope",
    "Province strata": "Province",
}

FACTOR_UNITS = {
    "Long-run annual mean temperature": "degrees C",
    "Long-run annual precipitation": "mm per year",
    "Baseline cropland share": "share",
    "Baseline population": "mean log baseline population",
    "Historical road distance": "km",
    "Mean elevation": "m",
    "Mean slope": "degrees",
}


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


def load_frame() -> pd.DataFrame:
    gwr = pd.read_csv(GWR, dtype={"Province Code": str})
    annual = annual_model.load_panel()
    climate = annual.groupby(ID, observed=True).agg(
        **{
            "Long-run annual mean temperature": ("Annual mean temperature C", "mean"),
            "Long-run annual precipitation": (annual_model.RAIN_SOURCE, "mean"),
        }
    ).reset_index()
    panel_columns = [
        ID,
        "Buffer Radius km",
        "Year",
        "Province Name",
        "Village Buffer Mean Baseline Cropland Share",
        "Village Buffer Mean Log Baseline Population 2000",
        "Village Buffer Mean Distance to Road Excluding Post 2007 AidData Corridors km",
        "Village Buffer Mean Mean Elevation m",
        "Village Buffer Mean Mean Slope Degrees",
    ]
    context = pd.read_parquet(CONTEXT_PANEL, columns=panel_columns)
    context = context.loc[
        context["Buffer Radius km"].eq(5) & context["Year"].eq(2001)
    ].drop(columns=["Buffer Radius km", "Year"])
    context = context.rename(
        columns={
            "Village Buffer Mean Baseline Cropland Share": "Baseline cropland share",
            "Village Buffer Mean Log Baseline Population 2000": "Baseline population",
            "Village Buffer Mean Distance to Road Excluding Post 2007 AidData Corridors km": "Historical road distance",
            "Village Buffer Mean Mean Elevation m": "Mean elevation",
            "Village Buffer Mean Mean Slope Degrees": "Mean slope",
        }
    )
    frame = gwr.merge(climate, on=ID, how="left", validate="one_to_one")
    frame = frame.merge(context, on=ID, how="left", validate="one_to_one")
    frame["Spatial block"] = (
        np.floor((frame[LONGITUDE] - 102.0) / SPATIAL_BLOCK_DEGREES).astype("Int64").astype(str)
        + "_"
        + np.floor((frame[LATITUDE] - 10.0) / SPATIAL_BLOCK_DEGREES).astype("Int64").astype(str)
    )
    return frame


def make_tertile_factor(
    frame: pd.DataFrame,
    variable: str,
) -> tuple[pd.Series, list[dict[str, object]]]:
    values = pd.to_numeric(frame[variable], errors="coerce")
    categories = pd.qcut(values, q=3, duplicates="drop")
    ordered = list(categories.cat.categories)
    names = ["Low", "Middle", "High"][: len(ordered)]
    mapping = {interval: f"{name} {FACTOR_LABELS[variable].lower()}" for interval, name in zip(ordered, names, strict=True)}
    labels = categories.map(mapping).astype("object")
    definitions = []
    for order, (interval, name) in enumerate(zip(ordered, names, strict=True), start=1):
        selected = categories.eq(interval)
        definitions.append(
            {
                "Factor": variable,
                "Factor label": FACTOR_LABELS[variable],
                "Stratum order": order,
                "Stratum": mapping[interval],
                "Minimum": float(values.loc[selected].min()),
                "Maximum": float(values.loc[selected].max()),
                "Unit": FACTOR_UNITS[variable],
                "Villages": int(selected.sum()),
            }
        )
    return labels, definitions


def factor_codes(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    clean = values.fillna("Missing").astype(str)
    codes, labels = pd.factorize(clean, sort=True)
    return codes.astype(int), list(labels.astype(str))


def q_statistic(y: np.ndarray, codes: np.ndarray) -> float:
    valid = np.isfinite(y) & (codes >= 0)
    local_y = y[valid]
    local_codes = codes[valid]
    if len(local_y) < 2:
        return np.nan
    total_ss = float(np.sum(np.square(local_y - local_y.mean())))
    count = np.bincount(local_codes)
    sum_y = np.bincount(local_codes, weights=local_y)
    sum_y2 = np.bincount(local_codes, weights=np.square(local_y))
    within_ss = float(np.sum(sum_y2 - np.square(sum_y) / np.maximum(count, 1)))
    return float(1.0 - within_ss / total_ss) if total_ss > 0 else np.nan


def spatial_block_cv_q(y: np.ndarray, codes: np.ndarray, blocks: np.ndarray) -> float:
    prediction = np.full(len(y), np.nan)
    for block in np.unique(blocks):
        test = blocks == block
        train = ~test
        global_mean = float(np.mean(y[train]))
        for code in np.unique(codes[test]):
            target = test & (codes == code)
            reference = train & (codes == code)
            prediction[target] = float(np.mean(y[reference])) if reference.any() else global_mean
    valid = np.isfinite(prediction) & np.isfinite(y)
    sse = float(np.sum(np.square(y[valid] - prediction[valid])))
    sst = float(np.sum(np.square(y[valid] - y[valid].mean())))
    return float(1.0 - sse / sst) if sst > 0 else np.nan


def permutation_probability(
    y: np.ndarray,
    codes: np.ndarray,
    observed_q: float,
    rng: np.random.Generator,
) -> float:
    null = np.empty(PERMUTATIONS)
    for iteration in range(PERMUTATIONS):
        null[iteration] = q_statistic(rng.permutation(y), codes)
    return float((1 + np.sum(null >= observed_q)) / (PERMUTATIONS + 1))


def province_strata(frame: pd.DataFrame) -> tuple[pd.Series, list[dict[str, object]]]:
    counts = frame["Province Name"].value_counts()
    eligible = counts.loc[counts.ge(30)].index
    labels = frame["Province Name"].where(
        frame["Province Name"].isin(eligible), "Other low-support provinces"
    )
    definitions = []
    for order, (label, count) in enumerate(labels.value_counts().sort_index().items(), start=1):
        definitions.append(
            {
                "Factor": "Province strata",
                "Factor label": "Province",
                "Stratum order": order,
                "Stratum": label,
                "Minimum": np.nan,
                "Maximum": np.nan,
                "Unit": "administrative category",
                "Villages": int(count),
            }
        )
    return labels, definitions


def classify_interaction(q1: float, q2: float, interaction: float) -> str:
    tolerance = 1e-9
    if interaction > q1 + q2 + tolerance:
        return "Nonlinear enhancement"
    if interaction > max(q1, q2) + tolerance:
        return "Bivariate enhancement"
    if abs(interaction - (q1 + q2)) <= tolerance:
        return "Independent"
    if interaction < min(q1, q2) - tolerance:
        return "Nonlinear weakening"
    return "Univariate weakening"


def run_detectors(
    frame: pd.DataFrame,
    factors: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    y = frame[OUTCOME].to_numpy(float)
    blocks = frame["Spatial block"].astype(str).to_numpy()
    rng = np.random.default_rng(SEED)
    codes_by_factor: dict[str, np.ndarray] = {}
    rows = []
    for factor, values in factors.items():
        codes, labels = factor_codes(values)
        codes_by_factor[factor] = codes
        observed_q = q_statistic(y, codes)
        rows.append(
            {
                "Factor": factor,
                "Factor label": FACTOR_LABELS[factor],
                "Strata": len(labels),
                "Villages": len(frame),
                "GeoDetector q": observed_q,
                "Spatial-block held-out q": spatial_block_cv_q(y, codes, blocks),
                "Permutation probability value": permutation_probability(
                    y, codes, observed_q, rng
                ),
                "Permutations": PERMUTATIONS,
            }
        )
    factor_results = pd.DataFrame(rows)
    factor_results["BH-adjusted probability value"] = bh_adjust(
        factor_results["Permutation probability value"]
    )

    q_lookup = factor_results.set_index("Factor")["GeoDetector q"].to_dict()
    interaction_rows = []
    for first, second in combinations(factors, 2):
        first_codes = codes_by_factor[first]
        second_codes = codes_by_factor[second]
        interaction_codes, labels = pd.factorize(
            pd.Series(first_codes).astype(str) + "|" + pd.Series(second_codes).astype(str),
            sort=True,
        )
        observed_q = q_statistic(y, interaction_codes)
        interaction_rows.append(
            {
                "Factor 1": first,
                "Factor 1 label": FACTOR_LABELS[first],
                "Factor 2": second,
                "Factor 2 label": FACTOR_LABELS[second],
                "Joint strata": len(labels),
                "Interaction q": observed_q,
                "Maximum component q": max(q_lookup[first], q_lookup[second]),
                "q gain above strongest component": observed_q
                - max(q_lookup[first], q_lookup[second]),
                "Spatial-block held-out interaction q": spatial_block_cv_q(
                    y, interaction_codes, blocks
                ),
                "Interaction type": classify_interaction(
                    q_lookup[first], q_lookup[second], observed_q
                ),
                "Permutation probability value": permutation_probability(
                    y, interaction_codes, observed_q, rng
                ),
                "Permutations": PERMUTATIONS,
            }
        )
    interaction_results = pd.DataFrame(interaction_rows)
    interaction_results["BH-adjusted probability value"] = bh_adjust(
        interaction_results["Permutation probability value"]
    )
    return factor_results, interaction_results, codes_by_factor


def make_summary_figure(
    factor_results: pd.DataFrame,
    interaction_results: pd.DataFrame,
    path: Path,
) -> None:
    ordered = factor_results.sort_values("GeoDetector q", ascending=True)
    positions = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), gridspec_kw={"width_ratios": [1.0, 1.2]})
    fig.subplots_adjust(left=0.16, right=0.98, top=0.94, bottom=0.13, wspace=0.33)
    axes[0].barh(
        positions - 0.17,
        ordered["GeoDetector q"],
        height=0.32,
        color="#32658e",
        label="In-sample GeoDetector q",
    )
    axes[0].barh(
        positions + 0.17,
        ordered["Spatial-block held-out q"],
        height=0.32,
        color="#d88b45",
        label="Spatial-block held-out q",
    )
    axes[0].axvline(0, color="#555555", linewidth=0.8)
    axes[0].set_yticks(positions, ordered["Factor label"], fontsize=8)
    axes[0].set_xlabel("Explained local-coefficient variance (q)", fontsize=8.5)
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].grid(axis="x", color="#d3d3d3", linestyle=(0, (2, 3)), linewidth=0.5)
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    axes[0].text(-0.15, 1.035, "a", transform=axes[0].transAxes, fontsize=12, fontweight="bold", va="top")
    axes[0].text(
        -0.08,
        1.035,
        "Single-factor explanatory power",
        transform=axes[0].transAxes,
        fontsize=9.3,
        fontweight="bold",
        va="top",
    )
    for side in ["top", "right"]:
        axes[0].spines[side].set_visible(False)

    factor_order = [
        factor
        for factor in factor_results.sort_values("GeoDetector q", ascending=False)["Factor"]
    ]
    label_order = [FACTOR_LABELS[factor] for factor in factor_order]
    matrix = np.full((len(factor_order), len(factor_order)), np.nan)
    for _, row in interaction_results.iterrows():
        i = factor_order.index(row["Factor 1"])
        j = factor_order.index(row["Factor 2"])
        matrix[i, j] = row["Interaction q"]
        matrix[j, i] = row["Interaction q"]
    np.fill_diagonal(
        matrix,
        [
            float(factor_results.loc[factor_results["Factor"].eq(factor), "GeoDetector q"].iloc[0])
            for factor in factor_order
        ],
    )
    image = axes[1].imshow(matrix, cmap="YlGnBu", vmin=0, vmax=float(np.nanmax(matrix)))
    axes[1].set_xticks(np.arange(len(label_order)), label_order, rotation=42, ha="right", fontsize=7)
    axes[1].set_yticks(np.arange(len(label_order)), label_order, fontsize=7)
    for i in range(len(factor_order)):
        for j in range(len(factor_order)):
            value = matrix[i, j]
            if np.isfinite(value):
                color = "white" if value > 0.58 * np.nanmax(matrix) else "#222222"
                axes[1].text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    axes[1].text(-0.11, 1.035, "b", transform=axes[1].transAxes, fontsize=12, fontweight="bold", va="top")
    axes[1].text(
        -0.04,
        1.035,
        "Single-factor diagonal and interaction q",
        transform=axes[1].transAxes,
        fontsize=9.3,
        fontweight="bold",
        va="top",
    )
    colorbar = fig.colorbar(image, ax=axes[1], orientation="horizontal", fraction=0.05, pad=0.16)
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.set_label("GeoDetector q", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_leading_strata_figure(
    frame: pd.DataFrame,
    factor: str,
    labels: pd.Series,
    path: Path,
) -> None:
    communes = gpd.read_file(BOUNDARIES).to_crs("EPSG:4326")
    provinces = communes.dissolve(by="ADM1_PCODE", as_index=False)
    countries = gpd.read_file(COUNTRIES).to_crs("EPSG:4326")
    countries = countries.loc[countries["ADMIN"].isin(["Cambodia", "Thailand", "Laos", "Vietnam"])]
    points = gpd.GeoDataFrame(
        frame.assign(**{"Leading stratum": labels}),
        geometry=gpd.points_from_xy(frame[LONGITUDE], frame[LATITUDE]),
        crs="EPSG:4326",
    )
    ordered_labels = [label for label in labels.dropna().drop_duplicates()]
    colors = dict(zip(ordered_labels, ["#3f77a5", "#e7b65e", "#a53e3e"], strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.65, 1.0]})
    fig.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.12, wspace=0.22)
    countries.plot(ax=axes[0], color="#f1eee6", edgecolor="#aaa69e", linewidth=0.5)
    countries.loc[countries["ADMIN"].eq("Cambodia")].plot(
        ax=axes[0], color="#fbfaf6", edgecolor="#555555", linewidth=0.8
    )
    provinces.boundary.plot(ax=axes[0], color="#77736a", linewidth=0.45)
    for label in ordered_labels:
        points.loc[points["Leading stratum"].eq(label)].plot(
            ax=axes[0], color=colors[label], markersize=9, alpha=0.78, edgecolor="none", label=label
        )
    axes[0].set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    axes[0].set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    axes[0].set_xticks(np.arange(102, 109, 1))
    axes[0].set_yticks(np.arange(10, 16, 1))
    axes[0].grid(color="#c8c8c8", linestyle=(0, (2, 3)), linewidth=0.45, alpha=0.65)
    axes[0].set_xlabel("Longitude (°E)", fontsize=8)
    axes[0].set_ylabel("Latitude (°N)", fontsize=8)
    axes[0].tick_params(labelsize=7)
    axes[0].legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.92)
    axes[0].text(0.018, 0.982, "a", transform=axes[0].transAxes, fontsize=12, fontweight="bold", va="top")
    axes[0].text(
        0.065,
        0.982,
        f"Outcome-blind {FACTOR_LABELS[factor].lower()} strata",
        transform=axes[0].transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )

    plot_data = [
        frame.loc[labels.eq(label), OUTCOME].to_numpy(float) for label in ordered_labels
    ]
    box = axes[1].boxplot(
        plot_data,
        tick_labels=ordered_labels,
        patch_artist=True,
        showfliers=False,
        widths=0.62,
        medianprops={"color": "#222222", "linewidth": 1.2},
    )
    for patch, label in zip(box["boxes"], ordered_labels, strict=True):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.82)
    axes[1].axhline(0, color="#555555", linewidth=0.9)
    axes[1].set_ylabel("Local NPP–food-welfare coefficient", fontsize=8)
    axes[1].tick_params(axis="x", labelsize=7, rotation=18)
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].grid(axis="y", color="#d3d3d3", linestyle=(0, (2, 3)), linewidth=0.5)
    axes[1].text(-0.08, 0.982, "b", transform=axes[1].transAxes, fontsize=12, fontweight="bold", va="top")
    axes[1].text(
        0.00,
        0.982,
        "Local coefficients by stratum",
        transform=axes[1].transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )
    for side in ["top", "right"]:
        axes[1].spines[side].set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_validated_interaction_figure(
    frame: pd.DataFrame,
    province_labels: pd.Series,
    precipitation_labels: pd.Series,
    path: Path,
) -> pd.DataFrame:
    """Map and summarize the best spatial-block validated compound partition."""
    plot_frame = frame.copy()
    plot_frame["Province stratum"] = province_labels.to_numpy()
    plot_frame["Precipitation stratum"] = precipitation_labels.to_numpy()
    plot_frame["Compound stratum"] = (
        plot_frame["Province stratum"].astype(str)
        + " | "
        + plot_frame["Precipitation stratum"].astype(str)
    )
    summary = (
        plot_frame.groupby(
            ["Province stratum", "Precipitation stratum", "Compound stratum"],
            observed=True,
        )
        .agg(
            Villages=(ID, "size"),
            Mean_local_coefficient=(OUTCOME, "mean"),
            Median_local_coefficient=(OUTCOME, "median"),
            Negative_coefficient_share=(OUTCOME, lambda values: float(np.mean(values < 0))),
            Positive_coefficient_share=(OUTCOME, lambda values: float(np.mean(values > 0))),
        )
        .reset_index()
    )
    summary["Meets minimum village support"] = summary["Villages"].ge(
        MIN_COMPOUND_STRATUM_VILLAGES
    )
    plot_frame = plot_frame.merge(
        summary[
            [
                "Compound stratum",
                "Mean_local_coefficient",
                "Meets minimum village support",
            ]
        ],
        on="Compound stratum",
        how="left",
        validate="many_to_one",
    )
    plot_frame["Supported mean local coefficient"] = plot_frame[
        "Mean_local_coefficient"
    ].where(plot_frame["Meets minimum village support"])
    communes = gpd.read_file(BOUNDARIES).to_crs("EPSG:4326")
    provinces = communes.dissolve(by="ADM1_PCODE", as_index=False)
    countries = gpd.read_file(COUNTRIES).to_crs("EPSG:4326")
    countries = countries.loc[
        countries["ADMIN"].isin(["Cambodia", "Thailand", "Laos", "Vietnam"])
    ]
    points = gpd.GeoDataFrame(
        plot_frame,
        geometry=gpd.points_from_xy(plot_frame[LONGITUDE], plot_frame[LATITUDE]),
        crs="EPSG:4326",
    )
    values = summary["Mean_local_coefficient"].to_numpy(float)
    bound = float(np.quantile(np.abs(values), 0.98))
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.2, 7.15),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.96, bottom=0.16, wspace=0.22)
    countries.plot(ax=axes[0], color="#f1eee6", edgecolor="#aaa69e", linewidth=0.5)
    countries.loc[countries["ADMIN"].eq("Cambodia")].plot(
        ax=axes[0], color="#fbfaf6", edgecolor="#555555", linewidth=0.8
    )
    provinces.boundary.plot(ax=axes[0], color="#77736a", linewidth=0.45)
    points.plot(
        ax=axes[0],
        column="Supported mean local coefficient",
        cmap="RdBu_r",
        norm=norm,
        markersize=9,
        alpha=0.80,
        edgecolor="none",
        missing_kwds={"color": "#cfcfcf", "alpha": 0.52},
        rasterized=True,
        zorder=5,
    )
    axes[0].set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    axes[0].set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    axes[0].set_xticks(np.arange(102, 109, 1))
    axes[0].set_yticks(np.arange(10, 16, 1))
    axes[0].grid(color="#c8c8c8", linestyle=(0, (2, 3)), linewidth=0.45, alpha=0.65)
    axes[0].set_xlabel("Longitude (°E)", fontsize=8)
    axes[0].set_ylabel("Latitude (°N)", fontsize=8)
    axes[0].tick_params(labelsize=7)
    axes[0].text(
        0.018,
        0.982,
        "a",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0},
    )
    axes[0].text(
        0.068,
        0.982,
        "Province–precipitation compound strata",
        transform=axes[0].transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0},
    )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cax = axes[0].inset_axes([0.12, -0.135, 0.76, 0.032])
    colorbar = fig.colorbar(scalar, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.set_label("Mean local NPP–food-welfare coefficient within compound stratum", fontsize=8)
    unsupported = plot_frame.loc[~plot_frame["Meets minimum village support"]]
    if len(unsupported):
        axes[0].legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="none",
                    markerfacecolor="#cfcfcf",
                    markeredgecolor="none",
                    markersize=5,
                    label=f"Compound stratum < {MIN_COMPOUND_STRATUM_VILLAGES} villages",
                )
            ],
            loc="lower left",
            fontsize=7.2,
            frameon=True,
            framealpha=0.92,
        )

    supported_summary = summary.loc[summary["Meets minimum village support"]].copy()
    province_order = (
        supported_summary.groupby("Province stratum", observed=True)["Mean_local_coefficient"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    precipitation_order = [
        label
        for label in ["Low precipitation", "Middle precipitation", "High precipitation"]
        if label in supported_summary["Precipitation stratum"].unique()
    ]
    precipitation_colors = {
        "Low precipitation": "#c7953f",
        "Middle precipitation": "#5e9fbe",
        "High precipitation": "#294d78",
    }
    offsets = np.linspace(-0.23, 0.23, max(len(precipitation_order), 1))
    y_positions = {province: position for position, province in enumerate(province_order)}
    for offset, precipitation in zip(offsets, precipitation_order, strict=True):
        subset = supported_summary.loc[
            supported_summary["Precipitation stratum"].eq(precipitation)
        ]
        axes[1].scatter(
            subset["Mean_local_coefficient"],
            [y_positions[value] + offset for value in subset["Province stratum"]],
            s=np.clip(15 + subset["Villages"].to_numpy() * 0.24, 20, 70),
            color=precipitation_colors[precipitation],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.90,
            label=precipitation,
            zorder=4,
        )
    axes[1].axvline(0, color="#555555", linewidth=0.9)
    axes[1].set_yticks(np.arange(len(province_order)), province_order, fontsize=7)
    axes[1].set_xlabel("Mean local NPP–food-welfare coefficient", fontsize=8)
    axes[1].tick_params(axis="x", labelsize=7)
    axes[1].grid(axis="x", color="#d3d3d3", linestyle=(0, (2, 3)), linewidth=0.5)
    axes[1].legend(loc="lower right", fontsize=7.2, frameon=False)
    axes[1].text(-0.12, 1.015, "b", transform=axes[1].transAxes, fontsize=12, fontweight="bold", va="top")
    axes[1].text(
        -0.04,
        1.015,
        "Within-province precipitation regimes",
        transform=axes[1].transAxes,
        fontsize=9.2,
        fontweight="bold",
        va="top",
    )
    for side in ["top", "right"]:
        axes[1].spines[side].set_visible(False)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    required = [OUTCOME, *CONTEXT_COLUMNS]
    coverage = pd.DataFrame(
        [
            {
                "Variable": variable,
                "Observed villages": int(frame[variable].notna().sum()),
                "Missing villages": int(frame[variable].isna().sum()),
            }
            for variable in required
        ]
    )
    analysis = frame.dropna(subset=required).copy().reset_index(drop=True)
    factors: dict[str, pd.Series] = {}
    definitions: list[dict[str, object]] = []
    for variable in CONTEXT_COLUMNS:
        labels, rows = make_tertile_factor(analysis, variable)
        factors[variable] = labels
        analysis[f"Stratum: {variable}"] = labels
        definitions.extend(rows)
    province_labels, province_rows = province_strata(analysis)
    factors["Province strata"] = province_labels
    analysis["Stratum: Province strata"] = province_labels
    definitions.extend(province_rows)

    factor_results, interaction_results, _ = run_detectors(analysis, factors)
    nonadministrative = factor_results.loc[factor_results["Factor"].ne("Province strata")]
    leading_factor = str(nonadministrative.sort_values("GeoDetector q", ascending=False).iloc[0]["Factor"])
    strongest_factor = factor_results.sort_values("GeoDetector q", ascending=False).iloc[0]
    strongest_interaction = interaction_results.sort_values("Interaction q", ascending=False).iloc[0]
    validated_interaction = interaction_results.sort_values(
        "Spatial-block held-out interaction q", ascending=False
    ).iloc[0]
    compound_summary = make_validated_interaction_figure(
        analysis,
        factors["Province strata"],
        factors["Long-run annual precipitation"],
        OUTPUT / "validated_province_precipitation_strata.png",
    )
    compound_labels = (
        factors["Province strata"].astype(str)
        + " | "
        + factors["Long-run annual precipitation"].astype(str)
    )
    compound_counts = compound_labels.value_counts()
    support_rows = []
    for minimum in [1, 10, 20, 30, 50]:
        keep = compound_labels.map(compound_counts).ge(minimum).to_numpy()
        supported_y = analysis.loc[keep, OUTCOME].to_numpy(float)
        supported_codes, supported_names = factor_codes(compound_labels.loc[keep])
        supported_blocks = analysis.loc[keep, "Spatial block"].astype(str).to_numpy()
        supported_q = q_statistic(supported_y, supported_codes)
        support_rows.append(
            {
                "Minimum villages per compound stratum": minimum,
                "Retained villages": int(keep.sum()),
                "Retained village share": float(keep.mean()),
                "Retained compound strata": len(supported_names),
                "GeoDetector q": supported_q,
                "Spatial-block held-out q": spatial_block_cv_q(
                    supported_y, supported_codes, supported_blocks
                ),
                "Permutation probability value": permutation_probability(
                    supported_y,
                    supported_codes,
                    supported_q,
                    np.random.default_rng(SEED + minimum),
                ),
                "Permutations": PERMUTATIONS,
            }
        )
    support_sensitivity = pd.DataFrame(support_rows)
    summary = {
        "design": (
            "Outcome-blind low-middle-high geographic strata; unweighted GeoDetector q; "
            "999 permutations; BH adjustment; 0.75-degree spatial-block held-out q"
        ),
        "villages_in_gwr_source": int(len(frame)),
        "villages_in_complete_geodetector_sample": int(len(analysis)),
        "spatial_blocks": int(analysis["Spatial block"].nunique()),
        "factors": len(factor_results),
        "pairwise_interactions": len(interaction_results),
        "strongest_in_sample_factor": strongest_factor.to_dict(),
        "strongest_nonadministrative_in_sample_factor": leading_factor,
        "strongest_nonadministrative_in_sample_factor_label": FACTOR_LABELS[leading_factor],
        "nonadministrative_factors_with_positive_spatial_block_held_out_q": int(
            nonadministrative["Spatial-block held-out q"].gt(0).sum()
        ),
        "strongest_interaction": strongest_interaction.to_dict(),
        "strongest_spatial_block_validated_interaction": validated_interaction.to_dict(),
        "validated_compound_stratum_count": int(len(compound_summary)),
        "primary_compound_stratum_minimum_villages": MIN_COMPOUND_STRATUM_VILLAGES,
        "primary_compound_stratum_support": support_sensitivity.loc[
            support_sensitivity["Minimum villages per compound stratum"].eq(
                MIN_COMPOUND_STRATUM_VILLAGES
            )
        ].iloc[0].to_dict(),
        "interpretation_limit": (
            "GeoDetector explains variation in exploratory GWR local coefficients. The q values "
            "do not establish causal mechanisms; spatial-block held-out q is required because "
            "the coefficient surface is spatially smoothed."
        ),
    }
    coverage.to_csv(OUTPUT / "geodetector_variable_coverage.csv", index=False)
    pd.DataFrame(definitions).to_csv(OUTPUT / "geodetector_stratum_definitions.csv", index=False)
    factor_results.sort_values("GeoDetector q", ascending=False).to_csv(
        OUTPUT / "geodetector_factor_results.csv", index=False
    )
    interaction_results.sort_values("Interaction q", ascending=False).to_csv(
        OUTPUT / "geodetector_interaction_results.csv", index=False
    )
    compound_summary.to_csv(
        OUTPUT / "validated_province_precipitation_strata_summary.csv", index=False
    )
    support_sensitivity.to_csv(
        OUTPUT / "compound_stratum_support_sensitivity.csv", index=False
    )
    analysis.to_csv(OUTPUT / "village_geodetector_analysis_frame.csv", index=False)
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# GeoDetector experiment for local NPP-welfare coefficients\n\n"
        "The experiment uses outcome-blind geographic strata to explain the adaptive-GWR "
        "coefficient surface. Standard q is accompanied by spatial-block held-out q.\n",
        encoding="utf-8",
    )
    make_summary_figure(
        factor_results,
        interaction_results,
        OUTPUT / "geodetector_factor_and_interaction_results.png",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nFactor results")
    print(factor_results.sort_values("GeoDetector q", ascending=False).to_string(index=False))
    print("\nTop interactions")
    print(interaction_results.sort_values("Interaction q", ascending=False).head(12).to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
