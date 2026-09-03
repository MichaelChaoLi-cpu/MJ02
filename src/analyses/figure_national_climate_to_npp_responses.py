#!/usr/bin/env python3
"""National Climate-to-NPP Responses.

Plan: Present the national Stage-1 fixed-effect coefficients under the primary
heat-day and alternative heat-degree-day specifications and the flexible
Rx5day response in natural units.
Framework: AnaSOP Sections 5-7 Stage-1 village and year fixed-effect model,
village-clustered inference, separate heat definitions, and the prespecified
three-knot restricted cubic spline for Rx5day.
"""

from __future__ import annotations

from pathlib import Path

from linearmodels.iv import AbsorbingLS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/cses_public_village_cropland_npp_absolute_climate_panel_preprocessed.parquet"
OUTPUT = ROOT / "data/results/figures/Figure_national_climate_to_npp_responses.png"
EVIDENCE = ROOT / "data/exp/analysis/climate-npp/national-climate-to-npp-responses"

ID = "National Village Point ID"
YEAR = "Year"
OUTCOME = "Annual Strict-Cropland Mean NPP kg C per m2"
HEAT = "Village Buffer Mean Annual Heat Days at or Above 35 C"
HDD = "Village Buffer Mean Annual Heat Degree-Days Above 35 C"
RX5DAY = "Village Buffer Mean Annual Maximum Consecutive Five-Day Precipitation Rx5day mm"
DRY_DAYS = "Village Buffer Mean Annual Maximum Consecutive Dry Days Below 1 mm"
RAIN_TOTAL = "Village Buffer Mean Annual Precipitation Total mm"

HEAT_10 = "Heat days at or above 35 C per 10 days"
HDD_10 = "Heat degree-days above 35 C per 10 degree-days"
RX5DAY_10 = "Rx5day per 10 mm"
DRY_DAYS_10 = "Maximum dry spell per 10 days"
RAIN_TOTAL_100 = "Annual precipitation per 100 mm"
SPLINE_LINEAR = "Rx5day restricted-spline linear basis"
SPLINE_NONLINEAR = "Rx5day restricted-spline nonlinear basis"

NAVY = "#173F5F"
BLUE = "#2F80A2"
TEAL = "#3A9D8F"
ORANGE = "#D97757"
GOLD = "#D4A72C"
DARK_GRAY = "#4D5960"
GRID_GRAY = "#D9DEE1"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="top",
        zorder=20,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.4},
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", color=GRID_GRAY, linewidth=0.55, linestyle="--", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#707A80")
    ax.tick_params(labelsize=8.5, colors=DARK_GRAY)


def rcs_basis(values: np.ndarray, knots: tuple[float, float, float]) -> np.ndarray:
    """Return a two-column restricted cubic spline basis for three knots."""
    x = np.asarray(values, dtype=float)
    k1, k2, k3 = knots
    if not k1 < k2 < k3:
        raise ValueError(f"Restricted cubic spline knots must be ordered: {knots}")
    positive = lambda z: np.maximum(z, 0.0) ** 3
    nonlinear = (
        positive(x - k1)
        - positive(x - k2) * (k3 - k1) / (k3 - k2)
        + positive(x - k3) * (k2 - k1) / (k3 - k2)
    ) / (k3 - k1) ** 2
    linear = x - k2
    return np.column_stack([linear / 10.0, nonlinear / 10.0])


def fit_absorbed(sample: pd.DataFrame, regressors: list[str]) -> object:
    absorb = pd.DataFrame(
        {
            "Village fixed effect": sample[ID].astype("category"),
            "Calendar-year fixed effect": sample[YEAR].astype("category"),
        },
        index=sample.index,
    )
    clusters = pd.DataFrame(
        {"Village cluster": pd.Categorical(sample[ID]).codes},
        index=sample.index,
    )
    return AbsorbingLS(
        sample[OUTCOME].astype(float),
        sample[regressors].astype(float),
        absorb=absorb,
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)


def coefficient_frame(
    result: object,
    model: str,
    terms: list[tuple[str, str, str]],
    observations: int,
    villages: int,
) -> pd.DataFrame:
    intervals = result.conf_int(level=0.95)
    rows: list[dict[str, object]] = []
    for term, label, color in terms:
        rows.append(
            {
                "Model": model,
                "Term": term,
                "Display Label": label,
                "Color": color,
                "Estimate": float(result.params[term]),
                "Clustered Standard Error": float(result.std_errors[term]),
                "95 Percent CI Lower": float(intervals.loc[term, "lower"]),
                "95 Percent CI Upper": float(intervals.loc[term, "upper"]),
                "Probability Value": float(result.pvalues[term]),
                "Observations": observations,
                "Villages": villages,
            }
        )
    return pd.DataFrame(rows)


def forest_panel(ax: plt.Axes, frame: pd.DataFrame, model_label: str, label: str) -> None:
    display = frame.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(display))
    for ypos, (_, row) in zip(y, display.iterrows(), strict=True):
        ax.hlines(
            ypos,
            row["95 Percent CI Lower"],
            row["95 Percent CI Upper"],
            color=row["Color"],
            linewidth=1.9,
            zorder=2,
        )
        ax.scatter(
            row["Estimate"],
            ypos,
            s=40,
            color=row["Color"],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
    ax.axvline(0, color="#5F686D", linewidth=0.9, zorder=1)
    ax.set_yticks(y, display["Display Label"])
    ax.set_xlabel("Change in annual cropland NPP (kg C m⁻² yr⁻¹)", fontsize=8.8)
    ax.text(
        0.985,
        0.965,
        model_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=NAVY,
        fontweight="bold",
    )
    style_axis(ax)
    panel_label(ax, label)


def main() -> None:
    columns = [ID, YEAR, OUTCOME, HEAT, HDD, RX5DAY, DRY_DAYS, RAIN_TOTAL]
    sample = pd.read_parquet(PANEL, columns=columns)
    sample = sample.loc[sample[YEAR].between(2001, 2021)].copy()
    for column in [OUTCOME, HEAT, HDD, RX5DAY, DRY_DAYS, RAIN_TOTAL]:
        sample[column] = pd.to_numeric(sample[column], errors="coerce")
    sample = sample.dropna(subset=columns).reset_index(drop=True)

    sample[HEAT_10] = sample[HEAT] / 10.0
    sample[HDD_10] = sample[HDD] / 10.0
    sample[RX5DAY_10] = sample[RX5DAY] / 10.0
    sample[DRY_DAYS_10] = sample[DRY_DAYS] / 10.0
    sample[RAIN_TOTAL_100] = sample[RAIN_TOTAL] / 100.0

    primary_regressors = [HEAT_10, RX5DAY_10, DRY_DAYS_10, RAIN_TOTAL_100]
    alternative_regressors = [HDD_10, RX5DAY_10, DRY_DAYS_10, RAIN_TOTAL_100]
    primary = fit_absorbed(sample, primary_regressors)
    alternative = fit_absorbed(sample, alternative_regressors)

    knots_array = sample[RX5DAY].quantile([0.10, 0.50, 0.90]).to_numpy(dtype=float)
    knots = tuple(float(value) for value in knots_array)
    basis = rcs_basis(sample[RX5DAY].to_numpy(), knots)
    sample[SPLINE_LINEAR] = basis[:, 0]
    sample[SPLINE_NONLINEAR] = basis[:, 1]
    spline_regressors = [HEAT_10, SPLINE_LINEAR, SPLINE_NONLINEAR, DRY_DAYS_10, RAIN_TOTAL_100]
    spline = fit_absorbed(sample, spline_regressors)

    observations = len(sample)
    villages = sample[ID].nunique()
    primary_terms = [
        (HEAT_10, "Heat days ≥35°C\n(per 10 days)", ORANGE),
        (RX5DAY_10, "Rx5day\n(per 10 mm)", BLUE),
        (DRY_DAYS_10, "Maximum dry spell\n(per 10 days)", GOLD),
        (RAIN_TOTAL_100, "Annual precipitation\n(per 100 mm)", TEAL),
    ]
    alternative_terms = [
        (HDD_10, "Heat degree-days >35°C\n(per 10 degree-days)", ORANGE),
        (RX5DAY_10, "Rx5day\n(per 10 mm)", BLUE),
        (DRY_DAYS_10, "Maximum dry spell\n(per 10 days)", GOLD),
        (RAIN_TOTAL_100, "Annual precipitation\n(per 100 mm)", TEAL),
    ]
    primary_frame = coefficient_frame(
        primary, "Primary heat-day specification", primary_terms, observations, villages
    )
    alternative_frame = coefficient_frame(
        alternative, "Alternative heat-intensity specification", alternative_terms, observations, villages
    )
    coefficients = pd.concat([primary_frame, alternative_frame], ignore_index=True)

    lower_support, upper_support = sample[RX5DAY].quantile([0.01, 0.99])
    grid = np.linspace(float(lower_support), float(upper_support), 220)
    reference = knots[1]
    grid_basis = rcs_basis(grid, knots)
    reference_basis = rcs_basis(np.array([reference]), knots)[0]
    differences = grid_basis - reference_basis
    spline_terms = [SPLINE_LINEAR, SPLINE_NONLINEAR]
    parameters = spline.params.loc[spline_terms].to_numpy(dtype=float)
    covariance = spline.cov.loc[spline_terms, spline_terms].to_numpy(dtype=float)
    estimates = differences @ parameters
    variances = np.einsum("ij,jk,ik->i", differences, covariance, differences)
    standard_errors = np.sqrt(np.maximum(variances, 0.0))
    curve = pd.DataFrame(
        {
            "Rx5day mm": grid,
            "Adjusted NPP Difference": estimates,
            "95 Percent CI Lower": estimates - 1.96 * standard_errors,
            "95 Percent CI Upper": estimates + 1.96 * standard_errors,
            "Reference Rx5day mm": reference,
        }
    )

    model_rows = []
    for name, result in [
        ("Primary heat-day specification", primary),
        ("Alternative heat-intensity specification", alternative),
        ("Restricted cubic spline Rx5day specification", spline),
    ]:
        model_rows.append(
            {
                "Model": name,
                "Observations": int(result.nobs),
                "Villages": villages,
                "Years": sample[YEAR].nunique(),
                "R Squared": float(result.rsquared),
                "Absorbed R Squared": float(result.absorbed_rsquared),
            }
        )
    models = pd.DataFrame(model_rows)

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    coefficients.drop(columns="Color").to_csv(EVIDENCE / "national_climate_npp_coefficients.csv", index=False)
    curve.to_csv(EVIDENCE / "rx5day_restricted_cubic_spline_curve.csv", index=False)
    models.to_csv(EVIDENCE / "national_climate_npp_model_support.csv", index=False)
    pd.DataFrame(
        {
            "Knot Percentile": [10, 50, 90],
            "Rx5day Knot mm": knots,
        }
    ).to_csv(EVIDENCE / "rx5day_restricted_cubic_spline_knots.csv", index=False)

    fig = plt.figure(figsize=(14.6, 8.8), facecolor="white")
    axes = fig.subplot_mosaic(
        [["a", "b"], ["c", "c"]],
        gridspec_kw={"height_ratios": [1.0, 1.15]},
    )
    fig.subplots_adjust(left=0.155, right=0.985, top=0.97, bottom=0.105, wspace=0.38, hspace=0.38)

    all_intervals = coefficients[["95 Percent CI Lower", "95 Percent CI Upper"]].to_numpy()
    forest_bound = float(np.nanmax(np.abs(all_intervals))) * 1.12
    forest_panel(axes["a"], primary_frame, "Heat-day specification", "a")
    forest_panel(axes["b"], alternative_frame, "Degree-day specification", "b")
    axes["a"].set_xlim(-forest_bound, forest_bound)
    axes["b"].set_xlim(-forest_bound, forest_bound)

    ax = axes["c"]
    ax.fill_between(
        curve["Rx5day mm"],
        curve["95 Percent CI Lower"],
        curve["95 Percent CI Upper"],
        color=BLUE,
        alpha=0.18,
        linewidth=0,
        zorder=1,
    )
    ax.plot(curve["Rx5day mm"], curve["Adjusted NPP Difference"], color=BLUE, linewidth=2.1, zorder=2)
    ax.axhline(0, color="#5F686D", linewidth=0.9, zorder=1)
    for knot in knots:
        ax.axvline(knot, color="#9CA5AA", linewidth=0.75, linestyle=":", zorder=0)
    rug_pool = sample.loc[sample[RX5DAY].between(lower_support, upper_support), RX5DAY]
    rug = rug_pool.sample(min(1200, len(rug_pool)), random_state=20260824)
    ymin, ymax = ax.get_ylim()
    ax.plot(rug, np.full(len(rug), ymin + 0.015 * (ymax - ymin)), "|", color="#7B858A", alpha=0.18, markersize=4)
    ax.set_xlabel("Annual Rx5day (mm)", fontsize=8.9)
    ax.set_ylabel("Adjusted NPP difference from median Rx5day\n(kg C m⁻² yr⁻¹)", fontsize=8.9)
    ax.set_xlim(float(lower_support), float(upper_support))
    nonlinear_p = float(spline.pvalues[SPLINE_NONLINEAR])
    nonlinear_p_text = "<0.001" if nonlinear_p < 0.001 else f"{nonlinear_p:.3f}"
    ax.text(
        0.985,
        0.95,
        f"Restricted cubic spline; nonlinearity p {nonlinear_p_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=NAVY,
    )
    style_axis(ax)
    panel_label(ax, "c")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print(f"observations={observations:,}; villages={villages:,}; years={sample[YEAR].nunique()}")
    print(f"Rx5day knots={knots}; nonlinear_p={float(spline.pvalues[SPLINE_NONLINEAR]):.6f}")
    print(coefficients.drop(columns="Color").to_string(index=False))


if __name__ == "__main__":
    main()
