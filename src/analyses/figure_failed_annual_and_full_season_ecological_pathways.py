#!/usr/bin/env python3
"""Failed Annual and Full-Season Ecological Pathways.

Plan: Transparently report why annual NPP and full May-February vegetation
were not promoted as the main ecological pathway. Coefficient evidence is
paired with strict held-out prediction comparisons against rainfall totals.
Framework: AnaSOP Appendix failure-mode evidence using the current absolute
35 C heat definition, 5 km village buffers, village and year fixed effects,
and 0.75-degree spatial-block-clustered uncertainty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS


ROOT = Path(__file__).resolve().parents[2]
NPP_PANEL = ROOT / "data/processed/cambodia_public_village_monsoon_npp_panel_preprocessed.parquet"
NPP_COEFFICIENTS = ROOT / "data/exp/analysis/climate-welfare/model-a-monsoon-npp/coefficients.csv"
NPP_COMPARISON = ROOT / "data/exp/analysis/climate-welfare/model-a-monsoon-npp/model_comparison.csv"
VEG_COEFFICIENTS = ROOT / "data/exp/analysis/climate-welfare/model-a-seasonal-vegetation/coefficients.csv"
VEG_COMPARISON = ROOT / "data/exp/analysis/climate-welfare/model-a-seasonal-vegetation/model_comparison.csv"
ANALYSIS_DIR = ROOT / "data/exp/analysis/climate-welfare/failed-annual-full-season-pathways"
OUTPUT = ROOT / "data/exp/legacy-results/figures/Figure_failed_annual_and_full_season_ecological_pathways.png"

ID = "National Village Point ID"
YEAR = "Year"
RADIUS = "Buffer Radius km"
LON = "Point Longitude"
LAT = "Point Latitude"
LAND_NPP = "Village Buffer Mean Annual Land NPP Anomaly kg C per m2"
CROP_NPP = "Baseline-Cropland-Weighted Annual Land NPP Anomaly kg C per m2"
RAIN = "Village Buffer Mean May October Precipitation Total mm Anomaly Z"
ONSET = "Village Buffer Mean Wet-Season Onset DOY Candidate B Anomaly Z"
DRY = "Village Buffer Mean Longest Intraseasonal Dry Spell Days Candidate B Anomaly Z"
HEAT = "Village Buffer Mean Post-Onset Absolute Heat Day Count 35 C Candidate B"
JOINT = "Joint rainfall and monsoon structure"
PRIMARY_EVI = "Village Buffer Mean May-February Production-Season Mean EVI Anomaly Z"
PRIMARY_NDVI = "Village Buffer Mean May-February Production-Season Mean NDVI Anomaly Z"

BLUE = "#2a6f97"
TEAL = "#2a9d8f"
ORANGE = "#e76f51"
GRAY = "#7a858c"

VARIABLE_LABELS = {
    RAIN: "Rainfall total (1 SD)",
    ONSET: "Onset timing (1 SD)",
    DRY: "Longest dry spell (1 SD)",
    HEAT: "Days ≥35°C (10 days)",
}
VARIABLE_ORDER = [RAIN, ONSET, DRY, HEAT]


def spatial_block(frame: pd.DataFrame, degrees: float = 0.75) -> pd.Series:
    lon = np.floor((frame[LON] - 102.0) / degrees).astype("Int64")
    lat = np.floor((frame[LAT] - 10.0) / degrees).astype("Int64")
    return lon.astype(str) + "_" + lat.astype(str)


def fit_cropland_weighted_npp() -> pd.DataFrame:
    columns = [RADIUS, ID, YEAR, LON, LAT, CROP_NPP] + VARIABLE_ORDER
    frame = pd.read_parquet(NPP_PANEL, columns=columns)
    frame = frame.loc[frame[RADIUS].eq(5)].dropna(subset=[CROP_NPP] + VARIABLE_ORDER).copy()
    frame[HEAT] = frame[HEAT] / 10.0
    frame["Spatial Block"] = spatial_block(frame)
    panel = frame.set_index([ID, YEAR]).sort_index()
    clusters = pd.DataFrame(
        {"Spatial Block": pd.Categorical(panel["Spatial Block"]).codes}, index=panel.index
    )
    result = PanelOLS(
        panel[CROP_NPP], panel[VARIABLE_ORDER], entity_effects=True,
        time_effects=True, drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)
    interval = result.conf_int(level=0.95)
    rows = []
    for variable in VARIABLE_ORDER:
        rows.append(
            {
                "Outcome": "Cropland-weighted NPP",
                "Variable": variable,
                "Coefficient": float(result.params[variable]),
                "95 Percent CI Lower": float(interval.loc[variable, "lower"]),
                "95 Percent CI Upper": float(interval.loc[variable, "upper"]),
                "Probability Value": float(result.pvalues[variable]),
                "Observations": int(result.nobs),
                "Villages": int(frame[ID].nunique()),
            }
        )
    return pd.DataFrame(rows)


def land_npp_rows(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[
        frame["Model"].eq(JOINT)
        & frame["Result Role"].eq("primary")
        & frame["Variable"].isin(VARIABLE_ORDER)
    ].copy()
    selected = selected.rename(
        columns={
            "Coefficient kg C per m2 per Exposure Unit": "Coefficient",
        }
    )
    selected["Outcome"] = "All-land NPP"
    return selected[
        ["Outcome", "Variable", "Coefficient", "95 Percent CI Lower", "95 Percent CI Upper", "Probability Value"]
    ]


def forest_two_series(
    ax: plt.Axes,
    frame: pd.DataFrame,
    series_order: list[str],
    colors: list[str],
    xlabel: str,
) -> None:
    y = np.arange(len(VARIABLE_ORDER))[::-1]
    offsets = np.linspace(-0.12, 0.12, len(series_order))
    ax.axvline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    for series, color, offset in zip(series_order, colors, offsets):
        rows = frame.loc[frame["Outcome"].eq(series)].set_index("Variable").loc[VARIABLE_ORDER]
        estimate = rows["Coefficient"].to_numpy(float)
        lower = rows["95 Percent CI Lower"].to_numpy(float)
        upper = rows["95 Percent CI Upper"].to_numpy(float)
        ax.errorbar(
            estimate, y + offset,
            xerr=[estimate - lower, upper - estimate],
            fmt="o", color=color, ecolor=color, markersize=5.0,
            elinewidth=1.35, capsize=3, label=series, zorder=2,
        )
    ax.set_yticks(y, [VARIABLE_LABELS[v] for v in VARIABLE_ORDER])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#d9dde0", linewidth=0.65, linestyle="--")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.45, len(VARIABLE_ORDER) - 0.15)
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.0, 1.005), ncol=2,
        frameon=False, fontsize=8.2, handletextpad=0.5, columnspacing=1.1,
    )


def rmse_change(frame: pd.DataFrame, rmse_column: str) -> pd.DataFrame:
    primary = frame.drop_duplicates("Model").set_index("Model")[rmse_column].dropna()
    benchmark = float(primary["Rainfall totals only"])
    rows = []
    for model in ["Monsoon structure only", JOINT]:
        rmse = float(primary[model])
        rows.append(
            {
                "Model": model,
                "RMSE": rmse,
                "Improvement versus rainfall only percent": (benchmark - rmse) / benchmark * 100,
            }
        )
    return pd.DataFrame(rows)


def improvement_bars(ax: plt.Axes, frame: pd.DataFrame, xlabel: str) -> None:
    labels = ["Monsoon structure only", "Joint model"]
    values = frame["Improvement versus rainfall only percent"].to_numpy(float)
    colors = [TEAL if value > 0 else ORANGE for value in values]
    y = np.arange(len(values))[::-1]
    ax.axvline(0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    ax.barh(y, values, color=colors, height=0.52)
    span = max(0.35, np.max(np.abs(values)))
    for yi, value in zip(y, values):
        ax.text(
            value + (0.04 * span if value >= 0 else -0.04 * span), yi,
            f"{value:+.2f}%", va="center",
            ha="left" if value >= 0 else "right", fontsize=9, color="#333333",
        )
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="#d9dde0", linewidth=0.65, linestyle="--")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.22)


def vegetation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[
        frame["Model"].eq(JOINT)
        & frame["Outcome"].isin([PRIMARY_EVI, PRIMARY_NDVI])
        & frame["Variable"].isin(VARIABLE_ORDER)
    ].copy()
    selected["Outcome"] = selected["Outcome"].map(
        {PRIMARY_EVI: "Full-season EVI", PRIMARY_NDVI: "Full-season NDVI"}
    )
    selected = selected.rename(columns={"Coefficient Outcome SD per Exposure Unit": "Coefficient"})
    return selected[
        ["Outcome", "Variable", "Coefficient", "95 Percent CI Lower", "95 Percent CI Upper", "Probability Value"]
    ]


def panel_header(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.14, 1.08, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left")
    ax.text(0.0, 1.08, title, transform=ax.transAxes, fontsize=10.5,
            va="top", ha="left")


def main() -> None:
    npp_coefficients = pd.read_csv(NPP_COEFFICIENTS)
    npp_comparison = pd.read_csv(NPP_COMPARISON)
    vegetation_coefficients = pd.read_csv(VEG_COEFFICIENTS)
    vegetation_comparison = pd.read_csv(VEG_COMPARISON)

    annual_rows = pd.concat(
        [land_npp_rows(npp_coefficients), fit_cropland_weighted_npp()],
        ignore_index=True,
    )
    vegetation = vegetation_rows(vegetation_coefficients)
    npp_prediction = rmse_change(npp_comparison, "Cross-Fitted RMSE kg C per m2")
    evi_prediction = rmse_change(vegetation_comparison, "Cross-Fitted RMSE SD")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    annual_rows.to_csv(ANALYSIS_DIR / "annual_npp_coefficients.csv", index=False)
    vegetation.to_csv(ANALYSIS_DIR / "full_season_vegetation_coefficients.csv", index=False)
    npp_prediction.to_csv(ANALYSIS_DIR / "annual_npp_prediction_gate.csv", index=False)
    evi_prediction.to_csv(ANALYSIS_DIR / "full_season_evi_prediction_gate.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2), constrained_layout=True)

    forest_two_series(
        axes[0, 0], annual_rows,
        ["All-land NPP", "Cropland-weighted NPP"], [BLUE, TEAL],
        "Annual NPP change (kg C m$^{-2}$ per exposure unit)",
    )
    panel_header(axes[0, 0], "a", "Annual NPP coefficients under two land masks")

    improvement_bars(
        axes[0, 1], npp_prediction,
        "Held-out RMSE improvement versus rainfall only",
    )
    panel_header(axes[0, 1], "b", "Annual NPP fails the prediction gate")

    forest_two_series(
        axes[1, 0], vegetation,
        ["Full-season EVI", "Full-season NDVI"], [BLUE, ORANGE],
        "Vegetation response (outcome SD per exposure unit)",
    )
    panel_header(axes[1, 0], "c", "Full-season vegetation is sensor-sensitive")

    improvement_bars(
        axes[1, 1], evi_prediction,
        "Held-out RMSE improvement versus rainfall only",
    )
    panel_header(axes[1, 1], "d", "Joint full-season EVI fails the prediction gate")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.relative_to(ROOT)}")
    print("\nAnnual NPP coefficient evidence")
    print(annual_rows.to_string(index=False))
    print("\nAnnual NPP prediction gate")
    print(npp_prediction.to_string(index=False))
    print("\nFull-season vegetation coefficient evidence")
    print(vegetation.to_string(index=False))
    print("\nFull-season EVI prediction gate")
    print(evi_prediction.to_string(index=False))


if __name__ == "__main__":
    main()
