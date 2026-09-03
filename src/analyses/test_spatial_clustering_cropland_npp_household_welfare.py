#!/usr/bin/env python3
"""Test whether spatial heterogeneity masks cropland-NPP household-welfare links."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import AbsorbingLS
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analyses"))
import test_annual_climate_cropland_npp_model as annual_model  # noqa: E402
import test_prior_year_heat_cropland_npp_household_welfare_chain as chain  # noqa: E402


OUTPUT = ROOT / "data/exp/analysis/climate-welfare/spatial-cropland-npp-household-welfare"
BOUNDARIES = ROOT / "data/raw/geography/cambodia_commune_boundaries_2018_2024.geojson"

ID = chain.ID
YEAR = chain.YEAR
TIME = chain.TIME
WEIGHT = chain.WEIGHT
BLOCK = chain.BLOCK
FOOD = "Log real food consumption per household member"
NPP_VALUE = "Annual Strict-Cropland Mean NPP kg C per m2"
NPP_VALID = "Strict-Cropland Valid NPP 500m Pixel Count"
NPP_TERM = "Prior-year strict-cropland NPP per 0.1 kg C per m2"
TEMP = "Temperature deviation C"
RAIN = "Precipitation deviation per 100 mm"
MIN_PIXELS = 10
MIN_CELL_HOUSEHOLDS = 5
K_NEIGHBORS = 8
PERMUTATIONS = 999
RNG_SEED = 20260824
MIN_PROVINCE_HOUSEHOLDS = 500
MIN_PROVINCE_VILLAGES = 30

COMPOSITION = [
    "Household Size",
    "Female Household Member Share",
    "Mean Household Member Age Years",
    "Child Age 0-14 Share",
    "Older Age 65 Plus Share",
    "Household Dependency Ratio",
    "Rural household",
]


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


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    observed = values.notna() & weights.notna() & weights.gt(0)
    if not observed.any():
        return np.nan
    return float(np.average(values.loc[observed].astype(float), weights=weights.loc[observed]))


def load_households() -> pd.DataFrame:
    frame = chain.load_frame()
    annual = annual_model.load_panel()[[ID, "Year", TEMP, RAIN]].rename(columns={"Year": YEAR})
    frame = frame.merge(annual, on=[ID, YEAR], how="inner", validate="many_to_one")
    frame = frame.loc[
        pd.to_numeric(frame[NPP_VALUE], errors="coerce").notna()
        & pd.to_numeric(frame[NPP_VALID], errors="coerce").ge(MIN_PIXELS)
        & pd.to_numeric(frame[FOOD], errors="coerce").notna()
    ].copy()
    frame[NPP_TERM] = pd.to_numeric(frame[NPP_VALUE], errors="coerce") / 0.1
    return frame


def aggregate_cells(frame: pd.DataFrame) -> pd.DataFrame:
    variables = [FOOD, NPP_TERM, TEMP, RAIN, *COMPOSITION]
    rows: list[dict[str, object]] = []
    keys = [ID, TIME, YEAR]
    for key, group in frame.groupby(keys, observed=True):
        weights = pd.to_numeric(group[WEIGHT], errors="coerce")
        row: dict[str, object] = {
            ID: key[0],
            TIME: key[1],
            YEAR: int(key[2]),
            "Household Count": int(len(group)),
            "Survey Weight Sum": float(weights.sum()),
            "Point Longitude": float(group["Point Longitude"].iloc[0]),
            "Point Latitude": float(group["Point Latitude"].iloc[0]),
            "Province Code": str(group["Province Code"].iloc[0]).zfill(2),
        }
        for variable in variables:
            row[variable] = weighted_mean(
                pd.to_numeric(group[variable], errors="coerce"), weights
            )
        rows.append(row)
    return pd.DataFrame(rows).loc[lambda data: data["Household Count"].ge(MIN_CELL_HOUSEHOLDS)]


def wls_residual(
    cells: pd.DataFrame,
    outcome: str,
    controls: list[str],
) -> pd.Series:
    required = [outcome, "Survey Weight Sum", TIME, *controls]
    sample = cells.dropna(subset=required).copy()
    time = pd.get_dummies(sample[TIME].astype(str), prefix="time", drop_first=True, dtype=float)
    x = pd.concat([sample[controls].astype(float), time], axis=1)
    x = sm.add_constant(x, has_constant="add")
    result = sm.WLS(
        sample[outcome].astype(float),
        x,
        weights=sample["Survey Weight Sum"].astype(float),
    ).fit()
    residual = pd.Series(np.nan, index=cells.index, dtype=float)
    residual.loc[sample.index] = result.resid
    return residual


def long_run_spatial_frame(cells: pd.DataFrame) -> pd.DataFrame:
    work = cells.copy()
    work["Food welfare residual"] = wls_residual(
        work, FOOD, [TEMP, RAIN, *COMPOSITION]
    )
    work["Cropland NPP residual"] = wls_residual(work, NPP_TERM, [TEMP, RAIN])
    rows: list[dict[str, object]] = []
    for village, group in work.groupby(ID, observed=True):
        weights = group["Survey Weight Sum"]
        rows.append(
            {
                ID: village,
                "Point Longitude": float(group["Point Longitude"].iloc[0]),
                "Point Latitude": float(group["Point Latitude"].iloc[0]),
                "Province Code": str(group["Province Code"].iloc[0]).zfill(2),
                "Survey Cells": int(len(group)),
                "Households": int(group["Household Count"].sum()),
                "Mean food welfare residual": weighted_mean(
                    group["Food welfare residual"], weights
                ),
                "Mean cropland NPP residual": weighted_mean(
                    group["Cropland NPP residual"], weights
                ),
            }
        )
    return pd.DataFrame(rows).dropna(
        subset=["Mean food welfare residual", "Mean cropland NPP residual"]
    ).reset_index(drop=True)


def weights_matrix(frame: pd.DataFrame, k: int) -> np.ndarray:
    coordinates = frame[["Point Longitude", "Point Latitude"]].to_numpy(float)
    tree = cKDTree(coordinates)
    _, indexes = tree.query(coordinates, k=min(k + 1, len(frame)))
    weights = np.zeros((len(frame), len(frame)), dtype=np.float32)
    for row, neighbors in enumerate(indexes):
        selected = [int(index) for index in np.atleast_1d(neighbors) if int(index) != row][:k]
        if selected:
            weights[row, selected] = 1.0 / len(selected)
    return weights


def standardized(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return (array - array.mean()) / array.std(ddof=0)


def global_moran(z: np.ndarray, weights: np.ndarray) -> float:
    return float(np.dot(z, weights @ z) / np.dot(z, z))


def bivariate_moran(zx: np.ndarray, zy: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean(zx * (weights @ zy)))


def spatial_tests(spatial: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = weights_matrix(spatial, K_NEIGHBORS)
    znpp = standardized(spatial["Mean cropland NPP residual"])
    zfood = standardized(spatial["Mean food welfare residual"])
    rng = np.random.default_rng(RNG_SEED)
    observed = {
        "Univariate cropland NPP": global_moran(znpp, weights),
        "Univariate food welfare": global_moran(zfood, weights),
        "Bivariate NPP-food welfare": bivariate_moran(znpp, zfood, weights),
    }
    permuted = {key: np.empty(PERMUTATIONS, dtype=float) for key in observed}
    local_observed = znpp * (weights @ zfood)
    local_permuted = np.empty((PERMUTATIONS, len(spatial)), dtype=np.float32)
    for iteration in range(PERMUTATIONS):
        perm_npp = rng.permutation(znpp)
        perm_food = rng.permutation(zfood)
        permuted["Univariate cropland NPP"][iteration] = global_moran(perm_npp, weights)
        permuted["Univariate food welfare"][iteration] = global_moran(perm_food, weights)
        permuted["Bivariate NPP-food welfare"][iteration] = bivariate_moran(
            znpp, perm_food, weights
        )
        local_permuted[iteration] = znpp * (weights @ perm_food)
    global_rows = []
    for test, statistic in observed.items():
        null = permuted[test]
        pvalue = (1 + np.sum(np.abs(null) >= abs(statistic))) / (PERMUTATIONS + 1)
        global_rows.append(
            {
                "Test": test,
                "Moran Statistic": statistic,
                "Permutation Probability Value": float(pvalue),
                "Permutations": PERMUTATIONS,
                "Neighbors": K_NEIGHBORS,
                "Villages": len(spatial),
            }
        )
    local_p = (
        1
        + np.sum(np.abs(local_permuted) >= np.abs(local_observed)[None, :], axis=0)
    ) / (PERMUTATIONS + 1)
    local = spatial.copy()
    local["Standardized NPP residual"] = znpp
    local["Spatial lag standardized food residual"] = weights @ zfood
    local["Local bivariate Moran statistic"] = local_observed
    local["Local permutation probability value"] = local_p
    local["Local BH-adjusted probability value"] = bh_adjust(
        local["Local permutation probability value"]
    )
    local["Cluster Type"] = np.select(
        [
            (znpp >= 0) & ((weights @ zfood) >= 0),
            (znpp < 0) & ((weights @ zfood) < 0),
            (znpp >= 0) & ((weights @ zfood) < 0),
            (znpp < 0) & ((weights @ zfood) >= 0),
        ],
        ["High NPP-High Welfare", "Low NPP-Low Welfare", "High NPP-Low Welfare", "Low NPP-High Welfare"],
        default="Unclassified",
    )
    local["FDR Significant Cluster"] = np.where(
        local["Local BH-adjusted probability value"].lt(0.05),
        local["Cluster Type"],
        "Not significant",
    )
    return pd.DataFrame(global_rows), local


def province_interactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    work = frame.copy()
    work["Province Code"] = work["Province Code"].astype(str).str.zfill(2)
    support = work.groupby("Province Code", observed=True).agg(
        Households=(ID, "size"),
        Villages=(ID, "nunique"),
    )
    eligible = support.loc[
        support["Households"].ge(MIN_PROVINCE_HOUSEHOLDS)
        & support["Villages"].ge(MIN_PROVINCE_VILLAGES)
    ].index
    excluded = support.loc[~support.index.isin(eligible)].reset_index()
    work = work.loc[work["Province Code"].isin(eligible)].copy()
    provinces = sorted(work["Province Code"].dropna().unique())
    terms = []
    for province in provinces:
        term = f"NPP province {province}"
        work[term] = work[NPP_TERM] * work["Province Code"].eq(province).astype(float)
        terms.append(term)
    controls = [TEMP, RAIN, *COMPOSITION]
    required = [FOOD, WEIGHT, BLOCK, "Commune Code", TIME, *terms, *controls]
    sample = work.dropna(subset=required).copy()
    absorb = pd.DataFrame(
        {
            "Commune": sample["Commune Code"].astype("category"),
            "Survey time": sample[TIME].astype("category"),
        },
        index=sample.index,
    )
    result = AbsorbingLS(
        sample[FOOD].astype(float),
        sample[[*terms, *controls]].astype(float),
        absorb=absorb,
        weights=sample[WEIGHT].astype(float),
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        clusters=pd.Categorical(sample[BLOCK]).codes,
        debiased=True,
    )
    intervals = result.conf_int(level=0.95)
    rows = []
    for province, term in zip(provinces, terms, strict=True):
        subset = sample.loc[sample["Province Code"].eq(province)]
        rows.append(
            {
                "Province Code": province,
                "Households": int(len(subset)),
                "Villages": int(subset[ID].nunique()),
                "Coefficient": float(result.params[term]),
                "Clustered Standard Error": float(result.std_errors[term]),
                "95 Percent CI Lower": float(intervals.loc[term, "lower"]),
                "95 Percent CI Upper": float(intervals.loc[term, "upper"]),
                "Probability Value": float(result.pvalues[term]),
            }
        )
    coefficients = pd.DataFrame(rows)
    coefficients["BH-Adjusted Probability Value"] = bh_adjust(coefficients["Probability Value"])
    coefficients["FDR Significant"] = coefficients[
        "BH-Adjusted Probability Value"
    ].lt(0.05)
    coefficients["Positive and FDR Significant"] = (
        coefficients["Coefficient"].gt(0)
        & coefficients["FDR Significant"]
    )
    restriction = np.zeros((len(terms) - 1, len(result.params)))
    parameter_positions = {name: index for index, name in enumerate(result.params.index)}
    reference = terms[0]
    for row, term in enumerate(terms[1:]):
        restriction[row, parameter_positions[term]] = 1.0
        restriction[row, parameter_positions[reference]] = -1.0
    test = result.wald_test(restriction, np.zeros(len(terms) - 1))
    summary = {
        "reference_province_for_equality_test": provinces[0],
        "joint_equal_province_coefficients_wald_statistic": float(test.stat),
        "joint_equal_province_coefficients_degrees_of_freedom": int(test.df),
        "joint_equal_province_coefficients_probability_value": float(test.pval),
        "households": int(len(sample)),
        "provinces": len(provinces),
        "spatial_blocks": int(sample[BLOCK].nunique()),
        "minimum_province_households": MIN_PROVINCE_HOUSEHOLDS,
        "minimum_province_villages": MIN_PROVINCE_VILLAGES,
        "excluded_provinces": excluded.to_dict(orient="records"),
    }
    return coefficients, summary


def make_map(local: pd.DataFrame, provinces: pd.DataFrame, path: Path) -> None:
    boundaries = gpd.read_file(BOUNDARIES).to_crs("EPSG:4326")
    province_map = boundaries.dissolve(by="ADM1_PCODE", as_index=False)
    province_map["Province Code"] = province_map["ADM1_PCODE"].str[-2:]
    province_map = province_map.merge(provinces, on="Province Code", how="left")
    points = gpd.GeoDataFrame(
        local,
        geometry=gpd.points_from_xy(local["Point Longitude"], local["Point Latitude"]),
        crs="EPSG:4326",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.4), constrained_layout=True)
    boundaries.boundary.plot(ax=axes[0], color="#b8b8b8", linewidth=0.25)
    colors = {
        "Not significant": "#d8d8d8",
        "High NPP-High Welfare": "#1b7837",
        "Low NPP-Low Welfare": "#762a83",
        "High NPP-Low Welfare": "#d73027",
        "Low NPP-High Welfare": "#4575b4",
    }
    for label, group in points.groupby("FDR Significant Cluster", observed=True):
        group.plot(
            ax=axes[0],
            color=colors.get(label, "#d8d8d8"),
            markersize=8 if label != "Not significant" else 3,
            alpha=0.9 if label != "Not significant" else 0.45,
            label=label,
        )
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=8)
    axes[0].set_title("a  Local bivariate NPP-welfare clusters", loc="left", fontsize=11)
    axes[0].set_axis_off()

    values = province_map["Coefficient"].dropna()
    bound = float(max(abs(values.min()), abs(values.max()))) if len(values) else 1.0
    province_map.plot(
        ax=axes[1],
        column="Coefficient",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound),
        edgecolor="white",
        linewidth=0.5,
        missing_kwds={"color": "#eeeeee"},
    )
    significant = province_map.loc[province_map["FDR Significant"].fillna(False)]
    if len(significant):
        significant.boundary.plot(ax=axes[1], color="black", linewidth=1.5)
    scalar = plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound), cmap="RdBu_r")
    colorbar = fig.colorbar(scalar, ax=axes[1], orientation="horizontal", fraction=0.045, pad=0.03)
    colorbar.set_label("Province-specific coefficient per 0.1 kg C m$^{-2}$ NPP", fontsize=8)
    axes[1].set_title("b  Province-specific NPP-food coefficients", loc="left", fontsize=11)
    axes[1].set_axis_off()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = load_households()
    cells = aggregate_cells(frame)
    spatial = long_run_spatial_frame(cells)
    global_tests, local = spatial_tests(spatial)
    province_coefficients, heterogeneity = province_interactions(frame)
    make_map(local, province_coefficients, OUTPUT / "spatial_clustering_and_province_coefficients.png")

    summary = {
        "design": (
            "Survey-weighted village-survey cells residualized for survey timing, annual "
            "climate, and household composition; eight-nearest-neighbor Moran diagnostics; "
            "province-interacted household fixed-effect regression"
        ),
        "minimum_households_per_village_survey_cell": MIN_CELL_HOUSEHOLDS,
        "villages_in_spatial_analysis": int(len(spatial)),
        "village_survey_cells": int(len(cells)),
        "permutations": PERMUTATIONS,
        "global_tests": global_tests.to_dict(orient="records"),
        "fdr_significant_local_cluster_count": int(
            local["FDR Significant Cluster"].ne("Not significant").sum()
        ),
        "positive_fdr_significant_province_count": int(
            province_coefficients["Positive and FDR Significant"].sum()
        ),
        "negative_fdr_significant_province_count": int(
            (
                province_coefficients["FDR Significant"]
                & province_coefficients["Coefficient"].lt(0)
            ).sum()
        ),
        "province_heterogeneity": heterogeneity,
        "interpretation_limit": (
            "Spatial clustering and regime heterogeneity are exploratory associations and do "
            "not establish a local causal NPP effect"
        ),
    }
    cells.to_parquet(OUTPUT / "village_survey_welfare_cells.parquet", index=False)
    spatial.to_csv(OUTPUT / "village_long_run_residuals.csv", index=False)
    global_tests.to_csv(OUTPUT / "global_moran_tests.csv", index=False)
    local.to_csv(OUTPUT / "local_bivariate_moran_clusters.csv", index=False)
    province_coefficients.to_csv(OUTPUT / "province_npp_food_coefficients.csv", index=False)
    (OUTPUT / "results_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "README.md").write_text(
        "# Spatial clustering of prior-year cropland NPP and household food welfare\n\n"
        "The diagnostic tests whether a national null masks spatially clustered residual "
        "relationships. Local cluster p-values and province coefficients are multiplicity "
        "adjusted; the outputs remain exploratory.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nGlobal Moran tests")
    print(global_tests.to_string(index=False))
    print("\nProvince coefficients")
    print(province_coefficients.sort_values("Coefficient", ascending=False).to_string(index=False))
    print("\nLocal cluster counts")
    print(local["FDR Significant Cluster"].value_counts().to_string())


if __name__ == "__main__":
    main()
