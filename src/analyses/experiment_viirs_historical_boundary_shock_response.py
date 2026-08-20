#!/usr/bin/env python3
"""Run the pre-specified VIIRS historical-boundary rainfall-response experiment.

Outputs are exploratory validation artifacts under data/exp. The script reuses the
frozen 5 km design, fixed alternative bandwidths, and mandatory within-commune model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from linearmodels.iv import AbsorbingLS


ROOT = Path(__file__).resolve().parents[2]
PANEL = ROOT / "data/processed/viirs_historical_boundary_climate_preprocessed.parquet"
OUTPUT = ROOT / "data/exp/viirs-boundary-experiment"

CELL = "Grid Cell ID"
YEAR = "Year"
COMMUNE = "Linked Climate Commune Code"
SEGMENT = "Historical Boundary Segment"
TREATMENT = "Higher-Repression Southwest Zone"
DISTANCE = "Signed Distance to Historical Repression Boundary km"
ABS_DISTANCE = "Absolute Distance to Historical Repression Boundary km"
PRIMARY_OUTCOME = "Asinh Annual Mean Radiance"
MEDIAN_OUTCOME = "Asinh Annual Median Radiance"
EXTENSIVE_OUTCOME = "Any Nonzero Annual Mean Radiance"
PRIMARY_SHOCK = "May October Rainfall Anomaly Z (1991-2020)"
ALTERNATIVE_SHOCK = "Annual Rainfall Anomaly Z (1991-2020)"
QUALITY_FLAG = "At Least 40 Cloud-Free Observations"
PRIMARY_BANDWIDTH = 5
ALTERNATIVE_BANDWIDTHS = (2, 10, 15, 20, 30)
SESOI = 0.20


@dataclass(frozen=True)
class Estimate:
    specification: str
    outcome: str
    shock: str
    bandwidth_km: int
    confirmation_model: bool
    quality_restricted: bool
    triangular_weights: bool
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    p_value: float
    observations: int
    grid_cells: int
    climate_communes: int
    cross_side_communes: int
    boundary_segments: int
    district_year_clusters: int
    outcome_sd_reference: float
    overall_outcome_sd_reference: float
    standardized_estimate: float
    standardized_ci_low: float
    standardized_ci_high: float
    overall_standardized_estimate: float
    overall_standardized_ci_low: float
    overall_standardized_ci_high: float
    sesoi_classification: str


def prepare_panel() -> pd.DataFrame:
    panel = pd.read_parquet(PANEL).copy()
    panel = panel.loc[panel[TREATMENT].notna() & panel[DISTANCE].notna()].copy()
    panel[TREATMENT] = panel[TREATMENT].astype("int8")
    panel["_district"] = panel[COMMUNE].astype("string").str[:4]
    panel["_district_year"] = panel["_district"] + "_" + panel[YEAR].astype("string")
    panel["_segment_year"] = panel[SEGMENT].astype("string") + "_" + panel[YEAR].astype("string")
    primary_cells = panel.loc[
        panel[f"Historical-Boundary Common Support {PRIMARY_BANDWIDTH} km"].eq(1)
    ].drop_duplicates(CELL)
    side_count = primary_cells.groupby(COMMUNE, observed=True)[TREATMENT].nunique()
    cross_side = set(side_count.loc[side_count.eq(2)].index.astype(str))
    panel["_cross_side_commune"] = panel[COMMUNE].astype(str).isin(cross_side)
    return panel


def sesoi_classification(low: float, high: float) -> str:
    if low >= -SESOI and high <= SESOI:
        return "substantively precise null"
    if low <= -SESOI and high >= SESOI:
        return "inconclusive relative to SESOI"
    return "bounded but not equivalent"


def fit_model(
    panel: pd.DataFrame,
    *,
    specification: str,
    outcome: str = PRIMARY_OUTCOME,
    shock: str = PRIMARY_SHOCK,
    bandwidth_km: int = PRIMARY_BANDWIDTH,
    confirmation: bool = False,
    quality_restricted: bool = False,
    triangular: bool = False,
    outcome_sd_reference: float,
    overall_outcome_sd_reference: float,
) -> Estimate:
    support_column = f"Historical-Boundary Common Support {bandwidth_km} km"
    sample = panel.loc[panel[support_column].eq(1)].copy()
    if confirmation:
        sample = sample.loc[sample["_cross_side_commune"]].copy()
    if quality_restricted:
        sample = sample.loc[sample[QUALITY_FLAG].eq(1)].copy()
    required = [
        outcome,
        shock,
        CELL,
        YEAR,
        COMMUNE,
        SEGMENT,
        TREATMENT,
        DISTANCE,
        "_district_year",
        "_segment_year",
    ]
    sample = sample.dropna(subset=required).copy()
    if sample.empty:
        raise ValueError(f"No complete observations for {specification}")

    sample["_shock"] = sample[shock].astype(float)
    sample["_treat_shock"] = sample[TREATMENT] * sample["_shock"]
    sample["_distance_shock"] = sample[DISTANCE] * sample["_shock"]
    sample["_treat_distance_shock"] = (
        sample[TREATMENT] * sample[DISTANCE] * sample["_shock"]
    )
    exog = ["_treat_shock", "_distance_shock", "_treat_distance_shock"]
    if not confirmation:
        exog.insert(0, "_shock")

    absorb_names = [CELL, "_segment_year"]
    if confirmation:
        sample["_commune_year"] = sample[COMMUNE].astype(str) + "_" + sample[YEAR].astype(str)
        absorb_names.append("_commune_year")
    absorb = sample[absorb_names].astype("category")
    clusters = pd.DataFrame(
        {
            "grid_cell": pd.factorize(sample[CELL])[0],
            "district_year": pd.factorize(sample["_district_year"])[0],
        },
        index=sample.index,
    )
    weights = None
    if triangular:
        weights = (1 - sample[ABS_DISTANCE] / bandwidth_km).clip(lower=1e-8)

    fitted = AbsorbingLS(
        dependent=sample[outcome].astype(float),
        exog=sample[exog].astype(float),
        absorb=absorb,
        weights=weights,
        drop_absorbed=True,
    ).fit(cov_type="clustered", clusters=clusters, debiased=True)
    target = "_treat_shock"
    interval = fitted.conf_int().loc[target]
    estimate = float(fitted.params[target])
    ci_low = float(interval["lower"])
    ci_high = float(interval["upper"])
    standardized = estimate / outcome_sd_reference
    standardized_low = ci_low / outcome_sd_reference
    standardized_high = ci_high / outcome_sd_reference
    primary_cells = sample.drop_duplicates(CELL)
    side_count = primary_cells.groupby(COMMUNE, observed=True)[TREATMENT].nunique()
    return Estimate(
        specification=specification,
        outcome=outcome,
        shock=shock,
        bandwidth_km=bandwidth_km,
        confirmation_model=confirmation,
        quality_restricted=quality_restricted,
        triangular_weights=triangular,
        estimate=estimate,
        standard_error=float(fitted.std_errors[target]),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=float(fitted.pvalues[target]),
        observations=int(fitted.nobs),
        grid_cells=int(sample[CELL].nunique()),
        climate_communes=int(sample[COMMUNE].nunique()),
        cross_side_communes=int(side_count.eq(2).sum()),
        boundary_segments=int(sample[SEGMENT].nunique()),
        district_year_clusters=int(sample["_district_year"].nunique()),
        outcome_sd_reference=outcome_sd_reference,
        overall_outcome_sd_reference=overall_outcome_sd_reference,
        standardized_estimate=standardized,
        standardized_ci_low=standardized_low,
        standardized_ci_high=standardized_high,
        overall_standardized_estimate=estimate / overall_outcome_sd_reference,
        overall_standardized_ci_low=ci_low / overall_outcome_sd_reference,
        overall_standardized_ci_high=ci_high / overall_outcome_sd_reference,
        sesoi_classification=sesoi_classification(standardized_low, standardized_high),
    )


def run_experiment(panel: pd.DataFrame) -> pd.DataFrame:
    specifications = [
        dict(specification="Primary 5 km"),
        dict(specification="Within-commune confirmation 5 km", confirmation=True),
        dict(specification="Annual rainfall alternative 5 km", shock=ALTERNATIVE_SHOCK),
        dict(specification="Annual median radiance 5 km", outcome=MEDIAN_OUTCOME),
        dict(specification="Any nonzero radiance 5 km", outcome=EXTENSIVE_OUTCOME),
        dict(specification="At least 40 cloud-free observations 5 km", quality_restricted=True),
        dict(specification="Triangular distance weights 5 km", triangular=True),
    ]
    specifications.extend(
        dict(specification=f"Fixed bandwidth {bandwidth} km", bandwidth_km=bandwidth)
        for bandwidth in ALTERNATIVE_BANDWIDTHS
    )

    scale_references: dict[str, tuple[float, float]] = {}
    for outcome in {item.get("outcome", PRIMARY_OUTCOME) for item in specifications}:
        reference = panel.loc[
            panel[f"Historical-Boundary Common Support {PRIMARY_BANDWIDTH} km"].eq(1),
            [CELL, outcome],
        ].dropna()
        overall_sd = float(reference[outcome].std(ddof=1))
        demeaned = reference[outcome] - reference.groupby(CELL)[outcome].transform("mean")
        residual_degrees = len(reference) - reference[CELL].nunique()
        within_cell_sd = float(np.sqrt(np.square(demeaned).sum() / residual_degrees))
        scale_references[outcome] = (within_cell_sd, overall_sd)

    estimates = []
    for specification in specifications:
        outcome = specification.get("outcome", PRIMARY_OUTCOME)
        outcome_sd_reference, overall_outcome_sd_reference = scale_references[outcome]
        estimates.append(
            fit_model(
                panel,
                outcome_sd_reference=outcome_sd_reference,
                overall_outcome_sd_reference=overall_outcome_sd_reference,
                **specification,
            )
        )
    return pd.DataFrame([asdict(estimate) for estimate in estimates])


def plot_estimates(results: pd.DataFrame) -> None:
    display = results.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(display))
    fig, axis = plt.subplots(figsize=(9.4, 6.3), constrained_layout=True)
    axis.axvspan(-SESOI, SESOI, color="#E8EFE4", alpha=0.9, zorder=0)
    axis.axvline(0, color="#333333", linewidth=0.9, zorder=1)
    axis.axvline(-SESOI, color="#71896A", linewidth=0.8, linestyle="--")
    axis.axvline(SESOI, color="#71896A", linewidth=0.8, linestyle="--")
    for index, row in display.iterrows():
        color = "#111111" if row["confirmation_model"] else "#2F6690"
        marker = "D" if row["confirmation_model"] else "o"
        axis.errorbar(
            row["standardized_estimate"],
            index,
            xerr=[[row["standardized_estimate"] - row["standardized_ci_low"]],
                  [row["standardized_ci_high"] - row["standardized_estimate"]]],
            fmt=marker,
            color=color,
            ecolor=color,
            capsize=2.5,
            markersize=5,
            linewidth=1.2,
            zorder=3,
        )
    axis.set_yticks(y, display["specification"], fontsize=8)
    axis.set_xlabel("Southwest − West rainfall response (primary-outcome SD per 1-SD shock)")
    axis.set_title("VIIRS historical-boundary rainfall-response experiment")
    axis.grid(axis="x", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "viirs_boundary_shock_response_experiment.png", dpi=220)
    plt.close(fig)


def write_readme(results: pd.DataFrame) -> None:
    primary = results.loc[results["specification"].eq("Primary 5 km")].iloc[0]
    confirmation = results.loc[
        results["specification"].eq("Within-commune confirmation 5 km")
    ].iloc[0]
    text = f"""# VIIRS historical-boundary rainfall-response experiment

These are experimental independent-validation results, not final manuscript outputs.

- Primary sample: {int(primary['grid_cells']):,} grid cells and {int(primary['observations']):,} cell-years within 5 km.
- Primary estimate: {primary['standardized_estimate']:.4f} primary-outcome SD per 1-SD shock (95% CI {primary['standardized_ci_low']:.4f}, {primary['standardized_ci_high']:.4f}).
- Mandatory within-commune estimate: {confirmation['standardized_estimate']:.4f} (95% CI {confirmation['standardized_ci_low']:.4f}, {confirmation['standardized_ci_high']:.4f}).
- Primary SESOI classification: {primary['sesoi_classification']}.
- Confirmation SESOI classification: {confirmation['sesoi_classification']}.

All models absorb grid-cell and boundary-segment-by-year fixed effects. The mandatory
confirmation also absorbs climate-commune-by-year fixed effects and uses only cross-side
communes. Uncertainty is two-way clustered by grid cell and district-by-year. The outcome-SD
conversion uses the pooled within-grid-cell standard deviation in the frozen 5 km primary
sample because the model absorbs grid-cell fixed effects. The overall outcome standard deviation
is retained in the estimate table as a secondary scale diagnostic.
"""
    (OUTPUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    results = run_experiment(panel)
    results.to_csv(OUTPUT / "viirs_boundary_shock_response_estimates.csv", index=False)
    plot_estimates(results)
    write_readme(results)
    print(results[["specification", "standardized_estimate", "standardized_ci_low", "standardized_ci_high", "p_value", "observations", "grid_cells"]].to_string(index=False))
    print(f"Wrote experiment outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
