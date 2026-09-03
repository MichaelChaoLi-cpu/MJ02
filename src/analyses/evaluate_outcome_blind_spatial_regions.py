#!/usr/bin/env python3
"""Evaluate outcome-blind SKATER and REDCAP regionalisations.

Plan: Compare four-through-seven-region candidate partitions before any
regional NPP or household-consumption coefficient is inspected.
Framework: AnaSOP Sections 5-7 pre-estimation gate. The candidate features are
predetermined geography, long-run climate, cropland, population, and road
accessibility. NPP, consumption, residuals, coefficient signs, and p-values
are deliberately excluded.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pygeoda
from scipy.spatial import Delaunay
from sklearn.metrics import adjusted_rand_score


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/cses_public_village_cropland_npp_absolute_climate_panel_preprocessed.parquet"
GRID = ROOT / "data/processed/cambodia_national_predetermined_covariates_preprocessed.parquet"
CROSSWALK = ROOT / "data/processed/cambodia_public_village_buffer_grid_crosswalk/radius_5_km.parquet"
HOUSEHOLDS = ROOT / "data/processed/cses_household_cropland_npp_analysis_preprocessed.parquet"
COMMUNES = ROOT / "data/raw/geography/odc_cambodia_communes_2014.gpkg"
COUNTRIES = ROOT / "data/raw/geography/natural_earth_admin0/ne_10m_admin_0_countries.shp"

OUTPUT_DIR = ROOT / "data/exp/analysis/climate-npp/outcome-blind-regionalisation-gate"
CANDIDATES = OUTPUT_DIR / "outcome_blind_candidate_features_and_support.parquet"
LABELS = OUTPUT_DIR / "outcome_blind_candidate_region_labels.csv"
DIAGNOSTICS = OUTPUT_DIR / "outcome_blind_region_selection_diagnostics.csv"
ADJACENCY = OUTPUT_DIR / "village_delaunay_adjacency.gal"
MAP_FIGURE = OUTPUT_DIR / "Outcome_blind_region_candidate_maps.png"
DIAGNOSTIC_FIGURE = OUTPUT_DIR / "Outcome_blind_region_selection_diagnostics.png"

ID = "National Village Point ID"
LONGITUDE = "Point Longitude"
LATITUDE = "Point Latitude"
YEAR = "Year"
STAGE1 = "Stage 1 Strict Cropland Complete Case"
STAGE2 = "Stage 2 Total Consumption Complete Case"

HEAT = "Village Buffer Mean Annual Heat Days at or Above 35 C"
HDD = "Village Buffer Mean Annual Heat Degree-Days Above 35 C"
RX5DAY = "Village Buffer Mean Annual Maximum Consecutive Five-Day Precipitation Rx5day mm"
DRY_DAYS = "Village Buffer Mean Annual Maximum Consecutive Dry Days Below 1 mm"
RAIN_TOTAL = "Village Buffer Mean Annual Precipitation Total mm"

ELEVATION = "Mean Elevation m"
SLOPE = "Mean Slope Degrees"
CROPLAND = "Baseline Cropland Share"
POPULATION = "Log Baseline Population 2000"
ROAD_DISTANCE = "Distance to Nearest Historical Road Proxy km"

FEATURES = [
    ELEVATION,
    SLOPE,
    RAIN_TOTAL,
    HEAT,
    HDD,
    RX5DAY,
    DRY_DAYS,
    CROPLAND,
    POPULATION,
    ROAD_DISTANCE,
]
K_VALUES = range(4, 8)
ALGORITHMS = ("SKATER", "REDCAP")
N_PERTURBATIONS = 20
PERTURBATION_SD = 0.05
RANDOM_SEED = 20260824

REGION_COLORS = ["#1B4965", "#5FA8D3", "#62B6CB", "#77B28C", "#F2C14E", "#F78154", "#A44A3F"]
DARK_GRAY = "#4D5960"
GRID_GRAY = "#D9DEE1"


def assemble_candidate_frame() -> pd.DataFrame:
    """Create one outcome-blind row per national village point."""
    panel_columns = [ID, LONGITUDE, LATITUDE, YEAR, STAGE1, HEAT, HDD, RX5DAY, DRY_DAYS, RAIN_TOTAL]
    panel = pd.read_parquet(PANEL, columns=panel_columns)
    panel = panel.loc[panel[YEAR].between(2001, 2021)].copy()
    for column in [STAGE1, HEAT, HDD, RX5DAY, DRY_DAYS, RAIN_TOTAL]:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    climate = (
        panel.groupby([ID, LONGITUDE, LATITUDE], as_index=False)
        .agg(
            **{
                HEAT: (HEAT, "mean"),
                HDD: (HDD, "mean"),
                RX5DAY: (RX5DAY, "mean"),
                DRY_DAYS: (DRY_DAYS, "mean"),
                RAIN_TOTAL: (RAIN_TOTAL, "mean"),
                "Stage 1 Complete Village-Years": (STAGE1, "sum"),
            }
        )
    )

    grid_columns = ["National Grid Cell ID", ELEVATION, SLOPE, CROPLAND, POPULATION, ROAD_DISTANCE]
    grid = pd.read_parquet(GRID, columns=grid_columns)
    crosswalk = pd.read_parquet(CROSSWALK, columns=[ID, "National Grid Cell ID"])
    static = (
        crosswalk.merge(grid, on="National Grid Cell ID", how="left", validate="many_to_one")
        .groupby(ID, as_index=False)[[ELEVATION, SLOPE, CROPLAND, POPULATION, ROAD_DISTANCE]]
        .mean()
    )

    household = pd.read_parquet(HOUSEHOLDS, columns=[ID, STAGE2])
    household[STAGE2] = pd.to_numeric(household[STAGE2], errors="coerce").fillna(0)
    support = (
        household.dropna(subset=[ID])
        .groupby(ID, as_index=False)[STAGE2]
        .sum()
        .rename(columns={STAGE2: "Stage 2 Complete Households"})
    )

    frame = climate.merge(static, on=ID, how="left", validate="one_to_one")
    frame = frame.merge(support, on=ID, how="left", validate="one_to_one")
    frame["Stage 2 Complete Households"] = frame["Stage 2 Complete Households"].fillna(0)
    frame = frame.sort_values(ID).reset_index(drop=True)
    if frame[[ID, LONGITUDE, LATITUDE, *FEATURES]].isna().any().any():
        missing = frame[[ID, LONGITUDE, LATITUDE, *FEATURES]].isna().sum()
        raise ValueError(f"Candidate clustering input contains missing values:\n{missing[missing > 0]}")
    if frame[[LONGITUDE, LATITUDE]].duplicated().any():
        raise ValueError("Village coordinates must be unique for Delaunay adjacency.")
    return frame


def delaunay_edges(frame: pd.DataFrame) -> set[tuple[int, int]]:
    """Return symmetric sparse edges from projected point Delaunay triangles."""
    points = gpd.GeoDataFrame(
        frame[[ID]].copy(),
        geometry=gpd.points_from_xy(frame[LONGITUDE], frame[LATITUDE]),
        crs=4326,
    ).to_crs(32648)
    coordinates = np.column_stack([points.geometry.x, points.geometry.y])
    triangles = Delaunay(coordinates).simplices
    edges: set[tuple[int, int]] = set()
    for triangle in triangles:
        a, b, c = (int(value) for value in triangle)
        for left, right in ((a, b), (a, c), (b, c)):
            edges.add((min(left, right), max(left, right)))
    return edges


def adjacency_lists(n: int, edges: set[tuple[int, int]]) -> dict[int, set[int]]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    if len(neighbors) != n or any(not neighbors[index] for index in range(n)):
        raise ValueError("Delaunay adjacency contains isolates.")
    visited = {0}
    queue: deque[int] = deque([0])
    while queue:
        current = queue.popleft()
        for neighbor in neighbors[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    if len(visited) != n:
        raise ValueError("Delaunay adjacency graph is not connected.")
    return neighbors


def write_gal(path: Path, neighbors: dict[int, set[int]]) -> None:
    """Write a standard unweighted GAL file with one-based observation IDs."""
    lines = [f"0 {len(neighbors)} villages observation_id"]
    for index in range(len(neighbors)):
        adjacent = sorted(neighbor + 1 for neighbor in neighbors[index])
        lines.append(f"{index + 1} {len(adjacent)}")
        lines.append(" ".join(str(value) for value in adjacent))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def standardize(frame: pd.DataFrame) -> np.ndarray:
    values = frame[FEATURES].to_numpy(dtype=float)
    standard_deviation = values.std(axis=0, ddof=0)
    if np.any(standard_deviation == 0):
        raise ValueError("A candidate clustering feature has zero variance.")
    return (values - values.mean(axis=0)) / standard_deviation


def fit_partition(
    algorithm: str,
    k: int,
    weights: pygeoda.Weight,
    values: np.ndarray,
    stage2_support: np.ndarray,
    minimum_stage2_support: int,
    seed: int,
) -> dict[str, object]:
    kwargs = {
        "bound_variable": stage2_support.tolist(),
        "min_bound": minimum_stage2_support,
        "scale_method": "raw",
        "random_seed": seed,
        "cpu_threads": 1,
    }
    data = pd.DataFrame(values, columns=FEATURES)
    if algorithm == "SKATER":
        return pygeoda.skater(k, weights, data, **kwargs)
    if algorithm == "REDCAP":
        return pygeoda.redcap(k, weights, data, "fullorder-wardlinkage", **kwargs)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def region_is_connected(labels: np.ndarray, neighbors: dict[int, set[int]]) -> bool:
    for region in np.unique(labels):
        members = set(np.flatnonzero(labels == region).tolist())
        start = next(iter(members))
        reached = {start}
        queue: deque[int] = deque([start])
        while queue:
            current = queue.popleft()
            for adjacent in neighbors[current] & members:
                if adjacent not in reached:
                    reached.add(adjacent)
                    queue.append(adjacent)
        if reached != members:
            return False
    return True


def geographic_compactness(frame: pd.DataFrame, labels: np.ndarray) -> float:
    points = gpd.GeoDataFrame(
        {"Region": labels},
        geometry=gpd.points_from_xy(frame[LONGITUDE], frame[LATITUDE]),
        crs=4326,
    ).to_crs(32648)
    distances: list[float] = []
    for _, group in points.groupby("Region"):
        centre_x = group.geometry.x.mean()
        centre_y = group.geometry.y.mean()
        distances.extend(np.hypot(group.geometry.x - centre_x, group.geometry.y - centre_y) / 1000.0)
    return float(np.mean(distances))


def support_summary(frame: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    audit = pd.DataFrame(
        {
            "Region": labels,
            "Villages": 1,
            "Stage 1 Village-Years": frame["Stage 1 Complete Village-Years"].to_numpy(dtype=float),
            "Stage 2 Households": frame["Stage 2 Complete Households"].to_numpy(dtype=float),
        }
    )
    return audit.groupby("Region", as_index=False).sum()


def evaluate_candidates(
    frame: pd.DataFrame,
    weights: pygeoda.Weight,
    neighbors: dict[int, set[int]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], np.ndarray]]:
    values = standardize(frame)
    stage2_support = frame["Stage 2 Complete Households"].to_numpy(dtype=float)
    minimums = {
        "Villages": int(np.ceil(len(frame) / (2 * max(K_VALUES)))),
        "Stage 1 Village-Years": int(np.ceil(frame["Stage 1 Complete Village-Years"].sum() / (2 * max(K_VALUES)))),
        "Stage 2 Households": int(np.ceil(stage2_support.sum() / (2 * max(K_VALUES)))),
    }
    rng = np.random.default_rng(RANDOM_SEED)
    perturbations = [rng.normal(0, PERTURBATION_SD, size=values.shape) for _ in range(N_PERTURBATIONS)]

    diagnostics: list[dict[str, object]] = []
    label_rows: list[pd.DataFrame] = []
    solutions: dict[tuple[str, int], np.ndarray] = {}
    for algorithm in ALGORITHMS:
        for k in K_VALUES:
            result = fit_partition(
                algorithm,
                k,
                weights,
                values,
                stage2_support,
                minimums["Stage 2 Households"],
                RANDOM_SEED,
            )
            labels = np.asarray(result["Clusters"], dtype=int)
            if len(np.unique(labels)) != k:
                raise ValueError(f"{algorithm} K={k} returned {len(np.unique(labels))} regions.")
            solutions[(algorithm, k)] = labels
            support = support_summary(frame, labels)
            stability_scores: list[float] = []
            for index, noise in enumerate(perturbations):
                perturbed = fit_partition(
                    algorithm,
                    k,
                    weights,
                    values + noise,
                    stage2_support,
                    minimums["Stage 2 Households"],
                    RANDOM_SEED + index + 1,
                )
                stability_scores.append(adjusted_rand_score(labels, np.asarray(perturbed["Clusters"], dtype=int)))

            min_villages = int(support["Villages"].min())
            min_stage1 = int(support["Stage 1 Village-Years"].min())
            min_stage2 = int(support["Stage 2 Households"].min())
            contiguous = region_is_connected(labels, neighbors)
            support_pass = (
                min_villages >= minimums["Villages"]
                and min_stage1 >= minimums["Stage 1 Village-Years"]
                and min_stage2 >= minimums["Stage 2 Households"]
            )
            diagnostics.append(
                {
                    "Algorithm": algorithm,
                    "Regions": k,
                    "Contiguous": contiguous,
                    "Minimum Villages": min_villages,
                    "Required Minimum Villages": minimums["Villages"],
                    "Minimum Stage 1 Village-Years": min_stage1,
                    "Required Minimum Stage 1 Village-Years": minimums["Stage 1 Village-Years"],
                    "Minimum Stage 2 Households": min_stage2,
                    "Required Minimum Stage 2 Households": minimums["Stage 2 Households"],
                    "All Support Gates Pass": support_pass,
                    "Between-to-Total Feature SS Ratio": float(
                        result["The ratio of between to total sum of squares"]
                    ),
                    "Mean Distance to Region Centroid km": geographic_compactness(frame, labels),
                    "Mean Perturbation Adjusted Rand Index": float(np.mean(stability_scores)),
                    "Minimum Perturbation Adjusted Rand Index": float(np.min(stability_scores)),
                    "Perturbations": N_PERTURBATIONS,
                    "Perturbation SD": PERTURBATION_SD,
                }
            )
            label_rows.append(
                pd.DataFrame(
                    {
                        ID: frame[ID],
                        "Algorithm": algorithm,
                        "Regions": k,
                        "Region ID": labels,
                    }
                )
            )

    diagnostic_frame = pd.DataFrame(diagnostics)
    label_frame = pd.concat(label_rows, ignore_index=True)
    agreement = []
    for k in K_VALUES:
        agreement.append(
            {
                "Regions": k,
                "SKATER-REDCAP Adjusted Rand Index": adjusted_rand_score(
                    solutions[("SKATER", k)], solutions[("REDCAP", k)]
                ),
            }
        )
    diagnostic_frame = diagnostic_frame.merge(pd.DataFrame(agreement), on="Regions", how="left")
    return diagnostic_frame, label_frame, solutions


def map_context() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    communes = gpd.read_file(COMMUNES).to_crs(4326)
    provinces = communes.dissolve(by="pro_code").reset_index()
    cambodia = communes.dissolve()
    countries = gpd.read_file(COUNTRIES).to_crs(4326)
    region = countries.cx[101.5:108.5, 9.7:15.5].copy()
    return region, cambodia, provinces


def style_map(ax: plt.Axes, region: gpd.GeoDataFrame, cambodia: gpd.GeoDataFrame, provinces: gpd.GeoDataFrame) -> None:
    region.plot(ax=ax, facecolor="#F0F2F1", edgecolor="#AAB2B6", linewidth=0.35, zorder=0)
    cambodia.plot(ax=ax, facecolor="#FFFEFA", edgecolor="#4D5960", linewidth=0.65, zorder=1)
    provinces.boundary.plot(ax=ax, color="#B2B9BC", linewidth=0.25, zorder=2)
    ax.set_xlim(102.2, 107.7)
    ax.set_ylim(10.25, 14.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#7A858A")
        spine.set_linewidth(0.55)


def draw_candidate_maps(frame: pd.DataFrame, solutions: dict[tuple[str, int], np.ndarray]) -> None:
    region, cambodia, provinces = map_context()
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 8.0), facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.925, bottom=0.09, wspace=0.055, hspace=0.13)
    for row, algorithm in enumerate(ALGORITHMS):
        for column, k in enumerate(K_VALUES):
            ax = axes[row, column]
            labels = solutions[(algorithm, k)]
            style_map(ax, region, cambodia, provinces)
            for region_id in sorted(np.unique(labels)):
                selected = labels == region_id
                ax.scatter(
                    frame.loc[selected, LONGITUDE],
                    frame.loc[selected, LATITUDE],
                    s=7.2,
                    color=REGION_COLORS[(int(region_id) - 1) % len(REGION_COLORS)],
                    alpha=0.93,
                    linewidths=0,
                    rasterized=True,
                    zorder=3,
                )
            ax.set_title(f"{algorithm}, {k} regions", fontsize=10, color=DARK_GRAY, pad=4)
            ax.text(
                0.015,
                0.985,
                chr(ord("a") + row * 4 + column),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11.5,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.1},
                zorder=10,
            )
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6, color=color, label=f"Region {index + 1}")
        for index, color in enumerate(REGION_COLORS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=7,
        frameon=False,
        fontsize=8.5,
        bbox_to_anchor=(0.5, 0.02),
    )
    fig.savefig(MAP_FIGURE, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_diagnostics(diagnostics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.96, bottom=0.10, wspace=0.26, hspace=0.30)
    colors = {"SKATER": "#1B4965", "REDCAP": "#D97757"}
    panels = [
        ("Mean Perturbation Adjusted Rand Index", "Perturbation stability (ARI)", "a"),
        ("Between-to-Total Feature SS Ratio", "Between/total feature variation", "b"),
        ("Mean Distance to Region Centroid km", "Mean distance to region centroid (km)", "c"),
        ("SKATER-REDCAP Adjusted Rand Index", "SKATER–REDCAP agreement (ARI)", "d"),
    ]
    for ax, (variable, ylabel, label) in zip(axes.flat, panels, strict=True):
        if variable == "SKATER-REDCAP Adjusted Rand Index":
            display = diagnostics.drop_duplicates("Regions").sort_values("Regions")
            ax.plot(display["Regions"], display[variable], marker="o", color="#5B6F7A", linewidth=1.8)
        else:
            for algorithm in ALGORITHMS:
                display = diagnostics.loc[diagnostics["Algorithm"] == algorithm].sort_values("Regions")
                ax.plot(
                    display["Regions"],
                    display[variable],
                    marker="o",
                    color=colors[algorithm],
                    linewidth=1.8,
                    label=algorithm,
                )
        ax.set_xticks(list(K_VALUES))
        ax.set_xlabel("Number of contiguous regions", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(color=GRID_GRAY, linewidth=0.55, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8.5, colors=DARK_GRAY)
        ax.text(0.01, 0.99, label, transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.savefig(DIAGNOSTIC_FIGURE, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = assemble_candidate_frame()
    edges = delaunay_edges(frame)
    neighbors = adjacency_lists(len(frame), edges)
    write_gal(ADJACENCY, neighbors)
    weights = pygeoda.read_gal(str(ADJACENCY), [str(index + 1) for index in range(len(frame))])
    if not weights.is_symmetric() or weights.has_isolates():
        raise ValueError("Clustering adjacency must be symmetric and contain no isolates.")

    diagnostics, labels, solutions = evaluate_candidates(frame, weights, neighbors)
    frame.to_parquet(CANDIDATES, index=False)
    labels.to_csv(LABELS, index=False)
    diagnostics.to_csv(DIAGNOSTICS, index=False)
    draw_candidate_maps(frame, solutions)
    draw_diagnostics(diagnostics)

    print(f"Saved: {CANDIDATES.relative_to(ROOT)}")
    print(f"Saved: {LABELS.relative_to(ROOT)}")
    print(f"Saved: {DIAGNOSTICS.relative_to(ROOT)}")
    print(f"Saved: {MAP_FIGURE.relative_to(ROOT)}")
    print(f"Saved: {DIAGNOSTIC_FIGURE.relative_to(ROOT)}")
    print(
        f"villages={len(frame):,}; edges={len(edges):,}; "
        f"neighbors_mean={weights.mean_neighbors():.2f}; symmetric={weights.is_symmetric()}"
    )
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
